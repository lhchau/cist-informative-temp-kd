#!/bin/bash
# Reproduce Table 3 on ImageNet (teachers use torchvision weights).
#   bash scripts/imagenet.sh cist        # or: kd
set -e
cd "$(dirname "$0")/.."
METHOD=${1:-cist}

python train.py --config configs/imagenet/${METHOD}.yaml model.teacher=ResNet34 model.arch=ResNet18
python train.py --config configs/imagenet/${METHOD}.yaml model.teacher=ResNet50 model.arch=MobileNetV1
