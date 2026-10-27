"""Teacher soft-label entropy under a fixed temperature vs. CIST (Fig. 2, Fig. 3a) and rho calibration (Appendix A).

    python analyze_entropy.py --teacher resnet32x4 \
        --ckpt checkpoints/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth --rhos 2 3 4 5 --plot entropy.png

Use ``--num-samples 512`` for the quick calibration sweep described in Appendix A.
"""
import argparse
import math

import torch
import torch.nn.functional as F

from data import get_dataloaders
from losses import cist_temperature, entropy
from models import build_model


@torch.no_grad()
def soft_label_entropies(model, loader, device, T, rhos, num_samples):
    """Entropy of every sample's teacher soft label, for fixed T and for CIST with each rho."""
    model.eval()
    results = {f"Fixed T={T:g}": [], **{f"CIST rho={rho:g}": [] for rho in rhos}}
    seen = 0
    for x, _ in loader:
        logits = model(x.to(device)).float()
        results[f"Fixed T={T:g}"].append(entropy(F.softmax(logits / T, -1)).cpu())
        for rho in rhos:
            centered, tau = cist_temperature(logits, rho)
            results[f"CIST rho={rho:g}"].append(entropy(F.softmax(centered / tau[:, None], -1)).cpu())
        seen += x.size(0)
        if num_samples and seen >= num_samples:
            break
    limit = num_samples or seen
    return {name: torch.cat(h)[:limit] for name, h in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100", "imagenet"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--ckpt", default=None, help="CIFAR teacher checkpoint (ImageNet uses torchvision weights)")
    parser.add_argument("--T", type=float, default=4.0, help="fixed KD temperature to compare against")
    parser.add_argument("--rhos", type=float, nargs="+", default=[3.0])
    parser.add_argument("--num-samples", type=int, default=0, help="0 = whole training set")
    parser.add_argument("--plot", default=None, help="save a box plot to this path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Soft labels are measured on (augmented) training images, i.e. exactly what the student distills from.
    loader, _, num_classes = get_dataloaders(args.dataset, args.data_root, batch_size=256)
    model = build_model(args.teacher, args.dataset, num_classes, pretrained=args.dataset == "imagenet")
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
    results = soft_label_entropies(model.to(device), loader, device, args.T, args.rhos, args.num_samples)

    print(f"{len(next(iter(results.values())))} samples, max entropy log(C) = {math.log(num_classes):.3f}")
    print(f"{'setting':<16}{'mean':>8}{'std':>8}{'min':>8}{'p5':>8}{'p95':>8}{'max':>8}")
    for name, h in results.items():
        p5, p95 = torch.quantile(h, torch.tensor([0.05, 0.95])).tolist()
        print(f"{name:<16}{h.mean():8.3f}{h.std():8.3f}{h.min():8.3f}{p5:8.3f}{p95:8.3f}{h.max():8.3f}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(1.6 * len(results) + 2, 4))
        plt.boxplot([h.numpy() for h in results.values()])
        plt.xticks(range(1, len(results) + 1), list(results))
        plt.ylabel("Entropy of teacher soft labels")
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
