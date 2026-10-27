"""Distillation objectives used in the paper.

Every loss takes ``(student_logits, teacher_logits, target)`` and returns
``(total_loss, {"ce": ce, "kl": kl})``; the parts are detached tensors, logged by the training loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy(probs):
    return -(probs * (probs + 1e-12).log()).sum(-1)


def cist_temperature(logits, rho):
    """Eq. (5) + Appendix A: tau_i = max(max_c(v_i - mean(v_i)) / rho, 1)."""
    centered = logits - logits.mean(dim=-1, keepdim=True)
    return centered, (centered.max(dim=-1).values / rho).clamp(min=1.0)


class KD(nn.Module):
    """Vanilla KD (Hinton et al., 2015) with a fixed temperature shared by teacher and student."""

    def __init__(self, T=4.0, ce_weight=0.1, kl_weight=0.9):
        super().__init__()
        self.T, self.ce_weight, self.kl_weight = T, ce_weight, kl_weight

    def forward(self, student_logits, teacher_logits, target):
        ce = F.cross_entropy(student_logits, target)
        kl = F.kl_div(
            F.log_softmax(student_logits / self.T, -1),
            F.softmax(teacher_logits / self.T, -1),
            reduction="batchmean",
        ) * self.T ** 2
        return self.ce_weight * ce + self.kl_weight * kl, {"ce": ce.detach(), "kl": kl.detach()}


class CIST(nn.Module):
    """Consistently Informative Soft-label Temperature (Algorithm 1).

    Teacher and student each get a sample-wise temperature tau = max(max_logit / rho, 1)
    computed on mean-centered logits; the per-sample KL is rescaled by tau_t * tau_s (Eq. 7).
    """

    def __init__(self, rho=3.0, ce_weight=0.1, kl_weight=8.0):
        super().__init__()
        self.rho, self.ce_weight, self.kl_weight = rho, ce_weight, kl_weight

    def forward(self, student_logits, teacher_logits, target):
        ce = F.cross_entropy(student_logits, target)
        teacher_logits, tau_t = cist_temperature(teacher_logits, self.rho)
        # The student temperature is a scale estimate only; no gradient flows through it.
        student_logits = student_logits - student_logits.mean(dim=-1, keepdim=True).detach()
        tau_s = (student_logits.detach().max(dim=-1).values / self.rho).clamp(min=1.0)

        kl = F.kl_div(
            F.log_softmax(student_logits / tau_s[:, None], -1),
            F.softmax(teacher_logits / tau_t[:, None], -1),
            reduction="none",
        ).sum(-1)
        kl = (kl * tau_t * tau_s).mean()
        return self.ce_weight * ce + self.kl_weight * kl, {"ce": ce.detach(), "kl": kl.detach()}


class KDEntOut(nn.Module):
    """KD variants of Table 1 that treat the lowest-entropy ``ratio`` of each batch specially.

    mode="ce": outliers drop the KL term and use plain cross-entropy (KD-EntOut-CE).
    mode="ht": outliers use temperature T + 1 instead of T (KD-EntOut-HT).
    """

    def __init__(self, mode, T=4.0, ce_weight=0.1, kl_weight=0.9, ratio=0.05):
        super().__init__()
        assert mode in ("ce", "ht"), mode
        self.mode, self.T, self.ce_weight, self.kl_weight, self.ratio = mode, T, ce_weight, kl_weight, ratio

    def forward(self, student_logits, teacher_logits, target):
        with torch.no_grad():
            ent = entropy(F.softmax(teacher_logits / self.T, -1))
            k = max(1, int(ent.numel() * self.ratio))
            outlier = torch.zeros_like(ent, dtype=torch.bool)
            outlier[ent.topk(k, largest=False).indices] = True

        T = torch.where(outlier, self.T + 1.0, self.T) if self.mode == "ht" else torch.full_like(ent, self.T)
        kl = F.kl_div(
            F.log_softmax(student_logits / T[:, None], -1),
            F.softmax(teacher_logits / T[:, None], -1),
            reduction="none",
        ).sum(-1) * T ** 2
        ce = F.cross_entropy(student_logits, target, reduction="none")

        if self.mode == "ce":
            loss = torch.where(outlier, ce, self.ce_weight * ce + self.kl_weight * kl).mean()
        else:
            loss = self.ce_weight * ce.mean() + self.kl_weight * kl.mean()
        return loss, {"ce": ce.mean().detach(), "kl": kl.mean().detach()}


def build_loss(method, **kwargs):
    if method == "kd":
        return KD(**kwargs)
    if method == "cist":
        return CIST(**kwargs)
    if method == "kd_entout_ce":
        return KDEntOut("ce", **kwargs)
    if method == "kd_entout_ht":
        return KDEntOut("ht", **kwargs)
    raise ValueError(f"Unknown distillation method: {method}")
