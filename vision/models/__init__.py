from .imagenet import imagenet_model_dict
from .mobilenetv2 import mobile_half
from .resnet import resnet20, resnet32, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .ShuffleNetv2 import ShuffleV2
from .vgg import vgg8_bn, vgg13_bn
from .wrn import wrn_16_2, wrn_16_4, wrn_28_2, wrn_28_4, wrn_40_2

# CIFAR-100 architectures used in the paper (Tables 2 and 6).
model_dict = {
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet8x4": resnet8x4,
    "resnet32x4": resnet32x4,
    "ResNet50": ResNet50,
    "wrn_16_2": wrn_16_2,
    "wrn_16_4": wrn_16_4,
    "wrn_28_2": wrn_28_2,
    "wrn_28_4": wrn_28_4,
    "wrn_40_2": wrn_40_2,
    "vgg8": vgg8_bn,
    "vgg13": vgg13_bn,
    "MobileNetV2": mobile_half,
    "ShuffleV2": ShuffleV2,
}


def build_model(name, dataset, num_classes, pretrained=False):
    """``pretrained`` loads torchvision ImageNet weights and only applies to ImageNet models."""
    if dataset == "imagenet":
        return imagenet_model_dict[name](pretrained=pretrained)
    return model_dict[name](num_classes=num_classes)
