"""Sample responses on an instruction-following benchmark and report ROUGE-L (Table 4).

Writes ``<save>/<data-names>/seed<seed>/{answers.jsonl,log.txt}``. Launched by scripts/eval.sh.
"""
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig

from arguments import get_args
from data_utils.prompt_datasets import PromptDataset
from rouge_metric import compute_metrics
from utils import (gather_rows, get_tokenizer, initialize, load_model, mixed_precision_dtype, print_rank, save_rank,
                   to_device)


@torch.no_grad()
def generate(args, tokenizer, model, dataset, device):
    sampler = DistributedSampler(dataset, shuffle=False, rank=dist.get_rank(), num_replicas=dist.get_world_size())
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size,
                        num_workers=args.num_workers, collate_fn=dataset.collate)
    max_new_tokens = args.max_length - args.max_prompt_length
    generation_config = GenerationConfig(
        do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size, repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=max_new_tokens)

    response_ids = []
    for batch in tqdm(loader, desc=f"Evaluating {args.data_names}", disable=dist.get_rank() != 0):
        ids = model.generate(**to_device(batch, device), generation_config=generation_config)
        ids = ids[:, args.max_prompt_length:]  # drop the (left-padded) prompt
        response_ids.append(F.pad(ids, (0, max_new_tokens - ids.size(1)), value=tokenizer.pad_token_id))
    return gather_rows(torch.cat(response_ids))[:len(dataset)]


def main():
    args = get_args()
    args.save = os.path.join(args.save, args.data_names, f"seed{args.seed}")
    initialize(args)
    device = torch.cuda.current_device()
    with open(args.deepspeed_config) as f:
        dtype = mixed_precision_dtype(json.load(f))

    tokenizer = get_tokenizer(args.model_path)
    dataset = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)
    model = load_model(args.model_path, device, dtype).eval()
    response_ids = generate(args, tokenizer, model, dataset, device)

    if dist.get_rank() == 0:
        responses = [r.replace("<n>", "\n").strip()
                     for r in tokenizer.batch_decode(response_ids, skip_special_tokens=True)]
        with open(os.path.join(args.save, "answers.jsonl"), "w") as f:
            f.writelines(json.dumps({"text": r}) + "\n" for r in responses)
        res = compute_metrics(responses, dataset.answers)
        gen_len = np.mean([len(tokenizer.encode(r)) for r in responses])
        log_str = f"test | name: {args.data_names} | {res} | avg. gen length: {gen_len:.1f}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))


if __name__ == "__main__":
    main()
