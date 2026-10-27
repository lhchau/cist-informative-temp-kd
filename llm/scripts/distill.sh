#!/bin/bash
# Off-policy distillation on Dolly (Table 4). Batch size = 8 per GPU x 4 GPUs = 32, 20 epochs.
#   bash scripts/distill.sh gpt2 cist --rho <rho>
#   bash scripts/distill.sh opt  rkl
#   SEED=20 GPUS=8 bash scripts/distill.sh gpt2 fkl
# Losses: fkl rkl symkl jsd srkl akl ab cist. The best dev-ROUGE-L checkpoint goes to <save>/best_rougeL.
set -eo pipefail
source "$(dirname "$0")/common.sh"
TYPE=$1; shift
SEED=${SEED:-10}
SAVE=${SAVE:-results/$FAMILY/$TYPE/seed$SEED}
mkdir -p "$SAVE"

$LAUNCH finetune.py \
  --type "$TYPE" --model-path "$STUDENT" --teacher-model-path "$TEACHER" $MODEL_OPTS $TRAIN_OPTS \
  --data-dir processed_data/dolly/full/$FAMILY --dev-num 1000 --num-workers 4 \
  --lr "$LR" --batch-size 8 --eval-batch-size 32 --gradient-accumulation-steps 1 \
  --epochs 20 --weight-decay 1e-2 --clip-grad 1.0 --lr-decay-style cosine --warmup-iters 0 --kd-ratio 1.0 \
  --max-length 512 --max-prompt-length 256 \
  --eval-gen --eval-interval -1 --log-interval 4 --save "$SAVE" --seed "$SEED" \
  $DS_OPTS $GEN_OPTS "$@" 2>&1 | tee -a "$SAVE/train.log"
