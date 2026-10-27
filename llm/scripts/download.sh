#!/bin/bash
# Evaluation data, pre-tokenized Dolly and the GPT-2 XL teacher released by MiniLLM (Gu et al., 2024).
# Requires `pip install -U huggingface_hub` (run `hf auth login` first if downloads are rate-limited).
set -e
cd "$(dirname "$0")/.."
hf download MiniLLM/dolly  --repo-type dataset --local-dir data/dolly
hf download MiniLLM/sinst  --repo-type dataset --local-dir data/sinst
hf download MiniLLM/uinst  --repo-type dataset --local-dir data/uinst
hf download MiniLLM/Vicuna --repo-type dataset --local-dir data/vicuna
hf download MiniLLM/dolly-processed --repo-type dataset --local-dir processed_data/dolly
hf download MiniLLM/teacher-gpt2-1.5B --local-dir checkpoints/teacher-gpt2-1.5B
