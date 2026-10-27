import os

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_dataloaders(name, root="./data", batch_size=64, test_batch_size=256, num_workers=4):
    """Returns (train_loader, test_loader, num_classes)."""
    if name == "cifar100":
        normalize = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        test_tf = T.Compose([T.ToTensor(), normalize])
        train_set = torchvision.datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
        num_classes = 100
    elif name == "imagenet":
        # Expects the standard ImageFolder layout: <root>/imagenet/{train,val}/<wnid>/*.JPEG
        normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        test_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
        train_set = torchvision.datasets.ImageFolder(os.path.join(root, "imagenet", "train"), train_tf)
        test_set = torchvision.datasets.ImageFolder(os.path.join(root, "imagenet", "val"), test_tf)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # persistent workers avoid re-spawning the worker pool every epoch
    kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, num_classes
