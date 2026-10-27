#!/bin/bash
# Reproduce Table 2 (KD and KD+Ours rows) on CIFAR-100.
#   bash scripts/cifar100.sh cist        # or: kd
#   SEEDS="42 43 44" bash scripts/cifar100.sh cist
# Teachers follow the mdistiller layout: checkpoints/cifar_teachers/<arch>_vanilla/ckpt_epoch_240.pth
set -e
cd "$(dirname "$0")/.."
METHOD=${1:-cist}
SEEDS=${SEEDS:-42}
TEACHER_DIR=${TEACHER_DIR:-checkpoints/cifar_teachers}

# teacher student lr
PAIRS=(
  "wrn_40_2   wrn_16_2    0.05"
  "resnet56   resnet20    0.05"
  "resnet110  resnet32    0.05"
  "resnet32x4 resnet8x4   0.05"
  "vgg13      vgg8        0.05"
  "resnet32x4 wrn_16_2    0.05"
  "resnet32x4 wrn_40_2    0.05"
  "wrn_40_2   resnet8x4   0.05"
  "ResNet50   MobileNetV2 0.01"
  "resnet32x4 ShuffleV2   0.01"
)

for SEED in $SEEDS; do
  for PAIR in "${PAIRS[@]}"; do
    read -r TEACHER STUDENT LR <<< "$PAIR"
    python train.py --config configs/cifar100/${METHOD}.yaml \
      model.teacher=$TEACHER model.arch=$STUDENT optim.lr=$LR seed=$SEED \
      model.teacher_ckpt=$TEACHER_DIR/${TEACHER}_vanilla/ckpt_epoch_240.pth
  done
done
