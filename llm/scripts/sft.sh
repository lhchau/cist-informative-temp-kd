#!/bin/bash
# Supervised fine-tuning of a teacher on Dolly (only needed if you do not download a released teacher).
#   bash scripts/sft.sh gpt2 gpt2-xl
#   bash scripts/sft.sh opt  facebook/opt-6.7b      # needs a ZeRO-3 / offload DeepSpeed config on 40GB GPUs
set -eo pipefail
source "$(dirname "$0")/common.sh"
BASE=$1; shift
SEED=${SEED:-10}
SAVE=${SAVE:-results/$FAMILY/sft/$(basename "$BASE")}
mkdir -p "$SAVE"

$LAUNCH finetune.py \
  --type sft --model-path "$BASE" $MODEL_OPTS $TRAIN_OPTS \
  --data-dir processed_data/dolly/full/$FAMILY --dev-num 1000 \
  --lr 5e-5 --batch-size 2 --eval-batch-size 8 --epochs 10 --weight-decay 1e-2 --clip-grad 1.0 --lr-decay-style cosine \
  --max-length 512 --max-prompt-length 256 \
  --eval-gen --eval-interval -1 --log-interval 4 --save "$SAVE" --seed "$SEED" \
  $DS_OPTS $GEN_OPTS "$@" 2>&1 | tee -a "$SAVE/train.log"
