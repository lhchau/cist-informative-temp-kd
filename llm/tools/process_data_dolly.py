"""Tokenize Dolly (``<data-dir>/raw.jsonl``) into the binary format read by ``LMTrainDataset``.

    python tools/process_data_dolly.py --data-dir data/dolly --processed-data-dir processed_data/dolly/full \
        --model-path gpt2 --model-type gpt2

Each instance is stored as ``prompt + [-1] + response + [eos]`` (the -1 separator becomes 65535 in uint16).
The first ``--dev-num`` lines form the validation split used for checkpoint selection.
"""
import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.indexed_dataset import MMapIndexedDatasetBuilder  # noqa: E402

TEMPLATE = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n")
TEMPLATE_WITH_INPUT = ("Below is an instruction that describes a task, paired with an input that provides further context. "
                       "Write a response that appropriately completes the request.\n\n"
                       "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")


class Encoder:
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        line = json.loads(line)
        if line.get("input"):
            prompt = TEMPLATE_WITH_INPUT.format(instruction=line["instruction"], input=line["input"])
        else:
            prompt = TEMPLATE.format(instruction=line["instruction"])
        tok = Encoder.tokenizer
        prompt_tokens = tok.encode(prompt, add_special_tokens=False)
        full_tokens = tok.encode(prompt + line["output"], add_special_tokens=False) + [tok.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        return line, prompt, prompt_tokens[:self.args.max_prompt_length], response_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--processed-data-dir", required=True)
    parser.add_argument("--model-path", required=True, help="tokenizer path / HF id")
    parser.add_argument("--model-type", required=True, choices=["gpt2", "opt"])
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--dev-num", type=int, default=1000)
    parser.add_argument("--data-process-workers", type=int, default=32)
    args = parser.parse_args()

    out_dir = os.path.join(args.processed_data_dir, args.model_type)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.data_dir, "raw.jsonl")) as f:
        raw = f.readlines()
    splits = {"valid": raw[:args.dev_num], "train": raw[args.dev_num:]}

    for split, lines in splits.items():
        encoder = Encoder(args)
        with multiprocessing.Pool(args.data_process_workers, initializer=encoder.initializer) as pool:
            builder = MMapIndexedDatasetBuilder(os.path.join(out_dir, f"{split}_0"), np.uint16)
            prompt_lens, response_lens = [], []
            with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as json_file:
                for line, prompt_str, prompt, response in pool.imap_unordered(encoder.encode, lines, chunksize=50):
                    builder.add_item(prompt + [-1] + response)
                    json_file.write(json.dumps({"instruction": line["instruction"], "prompt": prompt_str,
                                                "input": line.get("input", ""), "output": line["output"]}) + "\n")
                    prompt_lens.append(len(prompt))
                    response_lens.append(len(response))
            builder.finalize()
        print(f"{split}: {len(prompt_lens)} instances | prompt len mean {np.mean(prompt_lens):.1f} max {max(prompt_lens)}"
              f" | response len mean {np.mean(response_lens):.1f} max {max(response_lens)}")


if __name__ == "__main__":
    main()
