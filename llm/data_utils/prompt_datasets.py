import json
import os

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Evaluation prompts and reference answers from ``<data_dir>/<split>[_<model_type>].jsonl``."""

    def __init__(self, args, tokenizer, split, data_dir, num=-1):
        self.pad_id = tokenizer.eos_token_id
        self.max_prompt_length = args.max_prompt_length
        path = os.path.join(data_dir, f"{split}_{args.model_type}.jsonl")
        if not os.path.exists(path):
            path = os.path.join(data_dir, f"{split}.jsonl")
        with open(path) as f:
            raw = [json.loads(line) for line in f]
        if num > 0:
            raw = raw[:num]
        self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in raw]
        self.prompts = [tokenizer.encode(x["prompt"].replace("<n>", "\n"))[:self.max_prompt_length] for x in raw]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def collate(self, prompts):
        batch = {
            "input_ids": torch.full((len(prompts), self.max_prompt_length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(len(prompts), self.max_prompt_length, dtype=torch.long),
        }
        for i, prompt in enumerate(prompts):
            # left padding, so generation continues right after the prompt
            batch["input_ids"][i, -len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            batch["attention_mask"][i, -len(prompt):] = 1
        return batch
