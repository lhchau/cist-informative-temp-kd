import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .indexed_dataset import MMapIndexedDataset


class LMTrainDataset(Dataset):
    """Tokenized ``prompt <sep> response`` pairs from ``<path>/<split>_0.{bin,idx}`` for teacher-forced training.

    Each batch is (model_batch, no_model_batch, gen_batch): the full sequence, its labels (-100 on prompt and
    padding), and the left-padded prompt alone for generating dev-set responses.
    """

    SEP_ID = 65535  # the -1 separator written by tools/process_data_dolly.py, as stored in uint16

    def __init__(self, args, tokenizer, path, split, num=-1):
        self.model_type = args.model_type
        self.pad_id = tokenizer.eos_token_id
        self.max_length, self.max_prompt_length = args.max_length, args.max_prompt_length
        self.data = MMapIndexedDataset(os.path.join(path, f"{split}_0"))
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        answers_file = os.path.join(path, f"{split}.jsonl")
        if os.path.exists(answers_file):
            with open(answers_file) as f:
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]]
                                for x in map(json.loads, f)]

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.data[index].astype(np.int64)

    def _split(self, ids):
        """Remove the separator; returns (ids truncated to max_length, prompt ids or None)."""
        sep = np.flatnonzero(ids == self.SEP_ID)
        if len(sep) == 0:
            return ids[:self.max_length], None
        s = sep[0]
        return np.concatenate([ids[:s], ids[s + 1:]])[:self.max_length], ids[:s]

    def collate(self, samples):
        samples = [self._split(ids) for ids in samples]
        bs = len(samples)
        width = max(2, max(len(ids) for ids, _ in samples))  # pad to the longest sequence in the batch
        model_batch = {
            "input_ids": torch.full((bs, width), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, width),
        }
        if self.model_type == "gpt2":
            model_batch["position_ids"] = torch.zeros(bs, width, dtype=torch.long)
        no_model_batch = {"label": torch.full((bs, width), -100, dtype=torch.long)}
        gen_batch = {
            "input_ids": torch.full((bs, self.max_prompt_length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }

        for i, (ids, prompt) in enumerate(samples):
            n = len(ids) - 1
            model_batch["input_ids"][i, :n] = torch.from_numpy(ids[:-1])
            model_batch["attention_mask"][i, :n] = 1.0
            if "position_ids" in model_batch:
                model_batch["position_ids"][i, :n] = torch.arange(n)
            no_model_batch["label"][i, :n] = torch.from_numpy(ids[1:])
            if prompt is not None:
                no_model_batch["label"][i, :len(prompt) - 1] = -100  # train on response tokens only
                gen_batch["input_ids"][i, -len(prompt):] = torch.from_numpy(prompt)
                gen_batch["attention_mask"][i, -len(prompt):] = 1
        return model_batch, no_model_batch, gen_batch
