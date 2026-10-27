#!/bin/bash
# ROUGE-L on Dolly / Vicuna / Super-NI / UnNI over seeds 10..50 (Table 4).
#   bash scripts/eval.sh gpt2 results/gpt2/cist/seed10/best_rougeL
# Each run writes <ckpt>/eval/<benchmark>/seed<seed>/log.txt; summarize with tools/collect_results.py.
set -e
source "$(dirname "$0")/common.sh"
CKPT=$1; shift
SEEDS=${SEEDS:-"10 20 30 40 50"}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}

# name  data-dir  dev-num
BENCHMARKS=(
  "dolly  data/dolly       -1"
  "vicuna data/vicuna      -1"
  "sinst  data/sinst/11_   -1"
  "uinst  data/uinst/11_   10000"
)
for SEED in $SEEDS; do
  for B in "${BENCHMARKS[@]}"; do
    read -r NAME DIR NUM <<< "$B"
    $LAUNCH evaluate.py \
      --type eval --model-path "$CKPT" $MODEL_OPTS \
      --data-dir "$DIR" --data-names "$NAME" --dev-num "$NUM" --num-workers 0 \
      --eval-batch-size "$EVAL_BATCH_SIZE" --max-length 512 --max-prompt-length 256 \
      --save "$CKPT/eval" --seed "$SEED" \
      --deepspeed_config configs/ds_config.json $GEN_OPTS "$@"
  done
done
