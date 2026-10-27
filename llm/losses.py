"""Token-level distillation objectives compared in Table 4.

All losses take student ``logits`` and ``teacher_logits`` of shape [B, L, V] plus ``no_model_batch``
(whose ``label`` marks response tokens; -100 = ignored) and average over valid tokens.
Baseline implementations follow DistiLLM (Ko et al., 2024).
"""
import torch
import torch.nn.functional as F


def _valid_tokens(logits, teacher_logits, no_model_batch):
    valid = (no_model_batch["label"].reshape(-1) != -100).nonzero(as_tuple=True)[0]
    return (logits.reshape(-1, logits.size(-1))[valid],
            teacher_logits.reshape(-1, teacher_logits.size(-1))[valid])


def forward_kl(logits, teacher_logits, no_model_batch):
    logits, teacher_logits = _valid_tokens(logits, teacher_logits, no_model_batch)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    x = torch.masked_fill(teacher_probs * student_logprobs, torch.isinf(logits), 0).sum(-1)
    return -x.mean()


def reverse_kl(logits, teacher_logits, no_model_batch):
    logits, teacher_logits = _valid_tokens(logits, teacher_logits, no_model_batch)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    x = torch.masked_fill(student_probs * (teacher_logprobs - student_logprobs), inf_mask, 0).sum(-1)
    return -x.mean()


def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.5):
    """Sym-KL = (1 - lam) * FKL + lam * RKL; the paper uses lam = 0.5."""
    return ((1 - lam) * forward_kl(logits, teacher_logits, no_model_batch)
            + lam * reverse_kl(logits, teacher_logits, no_model_batch))


def js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    logits, teacher_logits = _valid_tokens(logits, teacher_logits, no_model_batch)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    x = torch.masked_fill(student_probs * (mixed_logprobs - student_logprobs), inf_mask, 0).sum(-1)
    loss = lam * -x.mean()
    x = torch.masked_fill(teacher_probs * (mixed_logprobs - teacher_logprobs), inf_mask, 0).sum(-1)
    return loss + (1 - lam) * -x.mean()


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    """SRKL (DistiLLM): KL(q || (1 - lam) p + lam q)."""
    logits, teacher_logits = _valid_tokens(logits, teacher_logits, no_model_batch)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    x = torch.masked_fill(student_probs * (mixed_logprobs - student_logprobs), inf_mask, 0).sum(-1)
    return -x.mean()


def _akl_ratio(teacher_logits, logits, mu=0.5):
    teacher_probs = F.softmax(teacher_logits.masked_fill(torch.isinf(teacher_logits), 0).float(), dim=-1)
    student_probs = F.softmax(logits.masked_fill(torch.isinf(logits), 0).float(), dim=-1).detach()
    sorted_teacher, idx = teacher_probs.sort(dim=-1, descending=True)
    errors = (sorted_teacher - student_probs.gather(-1, idx)).abs()
    tail = sorted_teacher.cumsum(-1) > mu
    tail[:, 0] = False  # the top-1 token always belongs to the head
    head_err = errors.masked_fill(tail, 0.0).sum(-1)
    tail_err = errors.masked_fill(~tail, 0.0).sum(-1)
    return head_err / (head_err + tail_err), tail_err / (head_err + tail_err)


def _weighted_kl(p_logits, q_logits, inf_mask, weight):
    """mean_i weight_i * KL(softmax(p_logits_i) || softmax(q_logits_i))."""
    p_probs = F.softmax(p_logits, dim=-1, dtype=torch.float32)
    p_logprobs = F.log_softmax(p_logits, dim=-1, dtype=torch.float32)
    q_logprobs = F.log_softmax(q_logits, dim=-1, dtype=torch.float32)
    kl = torch.masked_fill(p_probs * (p_logprobs - q_logprobs), inf_mask, 0).sum(-1)
    return (kl * weight).mean()


def adaptive_kl(logits, teacher_logits, no_model_batch):
    """AKL (Wu et al., 2025): head/tail-adaptive mix of FKL and RKL."""
    logits, teacher_logits = _valid_tokens(logits, teacher_logits, no_model_batch)
    inf_mask = torch.isinf(logits)
    head, tail = _akl_ratio(teacher_logits, logits)
    return (_weighted_kl(teacher_logits, logits, inf_mask, head)
            + _weighted_kl(logits, teacher_logits, inf_mask, tail))


def ab_div(logits, teacher_logits, no_model_batch, alpha=0.2, beta=0.7):
    """Alpha-beta divergence (ABKD, Wang et al., 2025); general case alpha, beta, alpha + beta != 0."""
    assert alpha != 0 and beta != 0 and alpha + beta != 0, "only the general alpha-beta case is implemented"
    logits, teacher_logits = _valid_tokens(logits, teacher_logits, no_model_batch)
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    apb = alpha + beta
    term1 = torch.logsumexp(alpha * log_p + beta * log_q, dim=-1).exp()
    term2 = alpha / apb * torch.logsumexp(apb * log_p, dim=-1).exp()
    term3 = beta / apb * torch.logsumexp(apb * log_q, dim=-1).exp()
    div = -(term1 - term2 - term3) / (alpha * beta)
    return torch.where(torch.isfinite(div), div, 0.0).mean()


def _center(logits):
    """Subtract the per-token mean over finite vocabulary entries; masked (-inf) entries stay masked."""
    logits = logits.float()
    finite = torch.isfinite(logits)
    filled = logits.masked_fill(~finite, 0.0)
    mean = filled.sum(-1, keepdim=True) / finite.sum(-1, keepdim=True).clamp_min(1)
    return (filled - mean).masked_fill(~finite, torch.finfo(torch.float32).min), finite


def cist_reverse_kl(logits, teacher_logits, no_model_batch, rho, kl_weight=10.0, min_student_temp=1.5):
    """CIST applied to reverse KL (Sec. 5.3 / Appendix A.2).

    Per-token temperatures tau = max(max_logit / rho, floor) on mean-centered logits, and the per-token
    RKL is rescaled by tau_t * tau_s (Eq. 7). ``min_student_temp`` and ``kl_weight`` are the values used
    for the language experiments.
    """
    logits, teacher_logits = _valid_tokens(logits, teacher_logits, no_model_batch)
    logits, student_finite = _center(logits)
    teacher_logits, teacher_finite = _center(teacher_logits)
    tau_t = (teacher_logits.max(-1).values / rho).clamp(min=1.0)
    tau_s = (logits.detach().max(-1).values / rho).clamp(min=min_student_temp)

    student_logprobs = F.log_softmax(logits / tau_s[:, None], dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits / tau_t[:, None], dim=-1)
    rkl = torch.masked_fill(student_logprobs.exp() * (student_logprobs - teacher_logprobs),
                            ~(student_finite & teacher_finite), 0).sum(-1)
    return kl_weight * (rkl * tau_s * tau_t).mean()


def distillation_loss(args, logits, teacher_logits, no_model_batch):
    if args.type == "fkl":
        return forward_kl(logits, teacher_logits, no_model_batch)
    if args.type == "rkl":
        return reverse_kl(logits, teacher_logits, no_model_batch)
    if args.type == "symkl":
        return symmetric_kl(logits, teacher_logits, no_model_batch)
    if args.type == "jsd":
        return js_distance(logits, teacher_logits, no_model_batch)
    if args.type == "srkl":
        return skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
    if args.type == "akl":
        return adaptive_kl(logits, teacher_logits, no_model_batch)
    if args.type == "ab":
        return ab_div(logits, teacher_logits, no_model_batch, alpha=args.ab_alpha, beta=args.ab_beta)
    if args.type == "cist":
        return cist_reverse_kl(logits, teacher_logits, no_model_batch, rho=args.rho)
    raise ValueError(f"Unknown distillation type: {args.type}")
