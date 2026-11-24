"""Motor-target weighting and loss regression checks; no simulator required."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from Agents.ftg_motor_loss import (make_motor_class_weights, motor_aware_loss,
                                    motor_logits)


def test_motor_logits_are_exact_marginals():
    scores = torch.arange(15, dtype=torch.float32).reshape(1, 15)
    actual = motor_logits(scores)
    expected = torch.logsumexp(scores.reshape(1, 3, 5), dim=2)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        motor_logits(scores[:, :14])


def test_default_loss_matches_old_bc_implementation():
    scores = torch.randn(8, 15, generator=torch.Generator().manual_seed(8))
    labels = torch.tensor([0, 5, 9, 12, 14, 3, 8, 13])
    total, action, motor = motor_aware_loss(scores, labels)
    torch.testing.assert_close(total, F.cross_entropy(scores, labels))
    torch.testing.assert_close(action, total)
    assert motor.item() == 0


def test_motor_loss_pushes_away_from_coast_when_teacher_drives():
    logits = torch.zeros(1, 15, requires_grad=True)
    total, action, motor = motor_aware_loss(logits, torch.tensor([12]), 2.0)
    total.backward()
    assert motor.item() > 0
    # Negative gradient for the target motor's five actions means optimizer
    # increases those logits; it does not invent continuous motor values.
    assert logits.grad[0, 10:15].mean().item() < 0
    assert logits.grad[0, 5:10].mean().item() > 0


def test_balancing_does_not_zero_out_minority_drive_samples():
    actions = np.array([5] * 61 + [10] * 23 + [0] * 16)
    weights = make_motor_class_weights(actions, power=0.65)
    assert weights.shape == (3,)
    assert float(weights[1]) < float(weights[2]) < float(weights[0])
    assert weights.isfinite().all()
    with pytest.raises(ValueError):
        make_motor_class_weights(actions, power=2)
