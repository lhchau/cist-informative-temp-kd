"""Off-policy distillation (or teacher SFT) on Dolly; keeps the checkpoint with the best dev ROUGE-L.

Launched by scripts/distill.sh and scripts/sft.sh.
"""
import json
import math
import os
import time

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig, get_constant_schedule_with_warmup

from arguments import get_args
from data_utils.lm_datasets import LMTrainDataset
from losses import distillation_loss
from rouge_metric import compute_metrics
from utils import (gather_rows, get_optimizer_params, get_tokenizer, initialize, load_model, mixed_precision_dtype,
                   print_rank, save_rank, to_device)


def lm_loss(logits, labels):
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


@torch.no_grad()
def evaluate(args, tokenizer, model, dataset, epoch, device):
    """Dev-set LM loss and, with --eval-gen, ROUGE-L of sampled responses (computed on rank 0, then broadcast)."""
    world_size = dist.get_world_size()
    sampler = DistributedSampler(dataset, shuffle=False, rank=dist.get_rank(), num_replicas=world_size)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                        num_workers=args.num_workers, collate_fn=dataset.collate)
    generation_config = GenerationConfig(
        do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature,
        repetition_penalty=args.repetition_penalty, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_length - args.max_prompt_length)

    model.eval()
    total_loss, response_ids = 0.0, []
    for batch in tqdm(loader, desc="Evaluating", disable=dist.get_rank() != 0):
        model_batch, no_model_batch, gen_batch = (to_device(b, device) for b in batch)
        loss = lm_loss(model(**model_batch).logits, no_model_batch["label"])
        dist.all_reduce(loss)
        total_loss += loss.item() / world_size
        if args.eval_gen:
            prompt_len = gen_batch["input_ids"].size(1)
            ids = model.generate(**gen_batch, generation_config=generation_config)[:, prompt_len:]
            response_ids.append(F.pad(ids, (0, args.max_length - prompt_len - ids.size(1)), value=tokenizer.pad_token_id))
    avg_loss = total_loss / len(loader)

    rouge_l = -1.0
    if args.eval_gen:
        response_ids = gather_rows(torch.cat(response_ids))[:len(dataset)]  # collective: all ranks take part
    if dist.get_rank() == 0:
        res = {}
        if args.eval_gen:
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            res = compute_metrics(responses, dataset.answers)
            rouge_l = res["rougeL"]
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                f.writelines(json.dumps({"text": r}) + "\n" for r in responses)
        log_str = f"dev | epoch {epoch} | avg_loss: {avg_loss:.4f} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))

    # every rank must take the same "new best checkpoint" branch
    metric = torch.tensor(rouge_l, device=device)
    dist.broadcast(metric, src=0)
    return avg_loss, metric.item()


def save_best(args, tokenizer, model, step, epoch, rouge_l, avg_loss):
    save_dir = os.path.join(args.save, "best_rougeL")
    if dist.get_rank() == 0:
        print_rank(f"New best ROUGE-L {rouge_l:.4f} at step {step}, saving to {save_dir}")
        tokenizer.save_pretrained(save_dir)
        model.module.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "best_metric.json"), "w") as f:
            json.dump({"best_rougeL": rouge_l, "best_step": step, "avg_loss": avg_loss, "epoch": epoch}, f, indent=2)
    dist.barrier()


def train(args, tokenizer, model, lr_scheduler, teacher, dataset, device):
    world_size = dist.get_world_size()
    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dist.get_rank(),
                                 num_replicas=world_size)
    loader = DataLoader(dataset["train"], sampler=sampler, batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    step, global_step, best_rouge_l = 1, 1, -1.0
    total_loss = total_distil_loss = total_time = 0.0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            model_batch, no_model_batch, _ = (to_device(b, device) for b in batch)
            start = time.time()

            logits = model(**model_batch, use_cache=False).logits
            loss = lm_loss(logits, no_model_batch["label"])
            distil_loss = torch.zeros((), device=device)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(**model_batch, use_cache=False).logits
                distil_loss = distillation_loss(args, logits, teacher_logits, no_model_batch)
                loss = (1 - args.kd_ratio) * loss + args.kd_ratio * distil_loss
            model.backward(loss)
            model.step()

            stats = torch.stack([loss.detach(), distil_loss.detach()])
            dist.all_reduce(stats)
            loss_value, distil_value = (stats / world_size).tolist()  # also waits for the step to finish
            total_loss += loss_value
            total_distil_loss += distil_value
            total_time += time.time() - start

            at_boundary = step % args.gradient_accumulation_steps == 0
            if at_boundary and global_step % args.log_interval == 0:
                n = args.log_interval * args.gradient_accumulation_steps
                log_str = (f"train | epoch {epoch:3d} | iter {global_step:6d}/{args.total_iters:6d} | "
                           f"loss {total_loss / n:.4f} | ds_loss {total_distil_loss / n:.4f} | "
                           f"lr {lr_scheduler.get_last_lr()[0]:.4e} | step time {total_time / args.log_interval:.3f}s")
                print_rank(log_str)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss = total_distil_loss = total_time = 0.0

            if at_boundary and global_step % args.eval_interval == 0:
                avg_loss, rouge_l = evaluate(args, tokenizer, model, dataset["dev"], epoch, device)
                if rouge_l > best_rouge_l:
                    best_rouge_l = rouge_l
                    save_best(args, tokenizer, model, global_step, epoch, rouge_l, avg_loss)
                model.train()

            step += 1
            if at_boundary:
                global_step += 1
            if global_step > args.total_iters:
                return


def main():
    args = get_args()
    initialize(args)
    device = torch.cuda.current_device()
    if dist.get_rank() == 0:
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)
    ds_config.update(gradient_accumulation_steps=args.gradient_accumulation_steps,
                     train_micro_batch_size_per_gpu=args.batch_size,
                     gradient_clipping=args.clip_grad, steps_per_print=10_000_000)
    args.deepspeed_config = None  # the dict above is passed to deepspeed.initialize instead
    dtype = mixed_precision_dtype(ds_config)

    tokenizer = get_tokenizer(args.model_path)
    dataset = {
        "train": LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num),
        "dev": LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num),
    }
    iters_per_epoch = len(dataset["train"]) // (args.batch_size * dist.get_world_size() * args.gradient_accumulation_steps)
    args.total_iters = args.total_iters or iters_per_epoch * args.epochs
    args.epochs = math.ceil(args.total_iters / iters_per_epoch)
    if args.eval_interval == -1:
        args.eval_interval = iters_per_epoch
    print_rank(f"train: {len(dataset['train'])}, dev: {len(dataset['dev'])}, "
               f"iters/epoch: {iters_per_epoch}, total iters: {args.total_iters}")

    model = load_model(args.model_path, device, dtype, args.gradient_checkpointing)
    optimizer = AdamW(get_optimizer_params(model), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_iters)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iters, eta_min=args.lr_min)
    model, _, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, args=args, lr_scheduler=lr_scheduler, config_params=ds_config)
    teacher = None if args.type == "sft" else load_model(args.teacher_model_path, device, dtype).eval()

    train(args, tokenizer, model, lr_scheduler, teacher, dataset, device)


if __name__ == "__main__":
    main()
