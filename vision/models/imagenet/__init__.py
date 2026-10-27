from .mobilenetv1 import MobileNetV1
from .resnet import resnet18, resnet34, resnet50

imagenet_model_dict = {
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "MobileNetV1": MobileNetV1,
}
