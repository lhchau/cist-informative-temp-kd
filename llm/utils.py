import os
import random
import time
from datetime import timedelta

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_rank(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def save_rank(log_str, save_path):
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def initialize(args):
    """Init the process group (one process per GPU, launched by torchrun), seed, and create ``args.save``."""
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    deepspeed.init_distributed(timeout=timedelta(minutes=300))
    seed = args.seed + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(args.save, exist_ok=True)
    if dist.get_rank() == 0:
        print("arguments:\n" + "\n".join(f"  {k} {'.' * (29 - len(k))} {v}" for k, v in vars(args).items()), flush=True)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def gather_rows(t):
    """Gather equally sized [n, L] tensors from all ranks, back in DistributedSampler (interleaved) order."""
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.stack(gathered, dim=1).view(-1, t.size(-1))


def mixed_precision_dtype(ds_config):
    if ds_config.get("bf16", {}).get("enabled", False):
        return torch.bfloat16
    if ds_config.get("fp16", {}).get("enabled", False):
        return torch.float16
    return torch.float32


def load_model(path, device, dtype, gradient_checkpointing=False):
    """Load a causal LM, preferring flash-attention-2, then SDPA, then the library default."""
    start = time.time()
    for attn_impl in ("flash_attention_2", "sdpa", None):
        try:
            kwargs = {"attn_implementation": attn_impl} if attn_impl else {}
            model = AutoModelForCausalLM.from_pretrained(path, device_map={"": device}, torch_dtype=dtype, **kwargs)
            break
        except Exception:  # backend not installed / unsupported for this model
            if attn_impl is None:
                raise
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print_rank(f" > loaded {path}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters "
               f"in {time.time() - start:.1f}s")
    return model


def get_optimizer_params(model):
    """No weight decay on biases and GPT-2 LayerNorm weights (as in MiniLLM / DistiLLM)."""
    no_decay = ["bias", "ln_f.weight", "ln_1.weight", "ln_2.weight", "ln_cross_attn"]
    params = list(model.named_parameters())
    return [
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)]},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]


def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # GPT-2 and OPT
    return tokenizer
