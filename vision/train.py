"""Train a teacher (``loss.method: ce``) or distill a student (``kd``, ``cist``, ...).

    python train.py --config configs/cifar100/cist.yaml \
        model.teacher=wrn_40_2 model.arch=wrn_16_2 \
        model.teacher_ckpt=checkpoints/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth

Any ``key.subkey=value`` after the config overrides that entry (values are parsed as Python literals).
"""
import argparse
import ast
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data import get_dataloaders
from losses import build_loss
from models import build_model


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key.subkey=value")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.overrides:
        key, value = item.lstrip("-").split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = torch.zeros(2, device=device)  # top-1, top-5
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        hit = model(x).topk(5, dim=1).indices.eq(y[:, None])
        correct += torch.stack([hit[:, 0].sum(), hit.any(dim=1).sum()])
    top1, top5 = (100.0 * correct / len(loader.dataset)).tolist()
    return top1, top5


def train_one_epoch(model, teacher, criterion, loader, optimizer, device):
    model.train()
    totals = torch.zeros(3, device=device)  # loss, ce, kl; summed on-device to avoid a GPU sync every step
    for x, y in tqdm(loader, leave=False, dynamic_ncols=True):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if teacher is None:
            loss = F.cross_entropy(logits, y)
            parts = {"ce": loss.detach(), "kl": torch.zeros((), device=device)}
        else:
            with torch.no_grad():
                teacher_logits = teacher(x)
            loss, parts = criterion(logits, teacher_logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        totals += torch.stack([loss.detach(), parts["ce"], parts["kl"]])
    return dict(zip(["loss", "ce", "kl"], (totals / len(loader)).tolist()))


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_cfg, model_cfg, optim_cfg, loss_cfg = cfg["data"], cfg["model"], cfg["optim"], dict(cfg["loss"])
    method = loss_cfg.pop("method")

    pair = model_cfg["arch"] if method == "ce" else f"{model_cfg['teacher']}_{model_cfg['arch']}"
    run_name = f"{pair}_seed{cfg['seed']}"
    out_dir = os.path.join(cfg.get("output_dir", "outputs"), data_cfg["name"], method, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(yaml.safe_dump(cfg, sort_keys=False))

    train_loader, test_loader, num_classes = get_dataloaders(**data_cfg)
    dataset = data_cfg["name"]

    teacher, criterion = None, None
    if method != "ce":
        teacher = build_model(model_cfg["teacher"], dataset, num_classes, pretrained=dataset == "imagenet")
        assert dataset == "imagenet" or model_cfg.get("teacher_ckpt"), "CIFAR distillation needs model.teacher_ckpt"
        if model_cfg.get("teacher_ckpt"):
            state = torch.load(model_cfg["teacher_ckpt"], map_location="cpu", weights_only=False)
            teacher.load_state_dict(state.get("model", state))
        teacher = teacher.to(device).eval()
        print(f"Teacher {model_cfg['teacher']} top-1/top-5: {evaluate(teacher, test_loader, device)}")
        criterion = build_loss(method, **loss_cfg)

    model = build_model(model_cfg["arch"], dataset, num_classes).to(device)
    print(f"Training {model_cfg['arch']} ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")

    optimizer = torch.optim.SGD(model.parameters(), lr=optim_cfg["lr"], momentum=optim_cfg["momentum"],
                                weight_decay=optim_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, optim_cfg["milestones"], optim_cfg["gamma"])

    best_acc = -1.0
    with open(os.path.join(out_dir, "log.jsonl"), "w") as log_file:
        for epoch in range(1, optim_cfg["epochs"] + 1):
            start, lr = time.time(), optimizer.param_groups[0]["lr"]
            train_stats = train_one_epoch(model, teacher, criterion, train_loader, optimizer, device)
            scheduler.step()
            top1, top5 = evaluate(model, test_loader, device)

            state = {"model": model.state_dict(), "epoch": epoch, "acc": top1}
            torch.save(state, os.path.join(out_dir, "last.pth"))
            if top1 > best_acc:
                best_acc = top1
                torch.save(state, os.path.join(out_dir, "best.pth"))

            record = {"epoch": epoch, **train_stats, "top1": top1, "top5": top5, "best_top1": best_acc,
                      "lr": lr, "time_s": round(time.time() - start, 1)}
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    print(f"Best top-1: {best_acc:.2f}  ->  {out_dir}")


if __name__ == "__main__":
    main()
