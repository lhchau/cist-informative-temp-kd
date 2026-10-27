"""Sanity checks for the distillation losses: ``python test_losses.py`` (or ``pytest test_losses.py``)."""
import torch

from losses import CIST, KD, KDEntOut, cist_temperature


def test_cist_temperature_rule():
    logits = torch.tensor([[9.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    _, tau = cist_temperature(logits, rho=3.0)
    assert torch.allclose(tau, torch.tensor([2.0, 1.0]))  # (9 - 3) / 3 = 2; small logits clamp to 1


def test_zero_kl_when_student_matches_teacher():
    logits, target = torch.randn(8, 100) * 5, torch.randint(0, 100, (8,))
    for loss in (KD(), CIST(), KDEntOut("ce"), KDEntOut("ht")):
        _, parts = loss(logits, logits, target)
        assert abs(parts["kl"]) < 1e-5, type(loss).__name__


if __name__ == "__main__":
    test_cist_temperature_rule()
    test_zero_kl_when_student_matches_teacher()
    print("ok")
