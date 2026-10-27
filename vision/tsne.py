"""t-SNE of penultimate CIFAR-100 test features (Fig. 4a/b).

    python tsne.py --arch resnet8x4 --ckpt outputs/cifar100/cist/resnet32x4_resnet8x4_seed42/best.pth --out tsne_cist.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from data import get_dataloaders
from models import build_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out", default="tsne.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, loader, num_classes = get_dataloaders("cifar100", args.data_root)
    model = build_model(args.arch, "cifar100", num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()

    feats, labels = [], []
    for x, y in loader:
        f, _ = model(x.to(device), is_feat=True)
        feats.append(f[-1].cpu().numpy())
        labels.append(y.numpy())
    emb = TSNE().fit_transform(np.concatenate(feats))
    labels = np.concatenate(labels)

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="nipy_spectral", s=1, alpha=0.4)
    plt.xticks([]), plt.yticks([])
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
