"""Unit tests: steering grouping, objective, weighting and start sampling."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from Agents.ftg_steering_loss import (
    combined_bc_loss, make_steering_class_weights, steering_logits)
from Agents.ftg_behavior_cloning import boost_start_indices


def test_steering_logits_group_correctly():
    scores = torch.arange(15, dtype=torch.float32).reshape(1, 15)
    torch.testing.assert_close(
        steering_logits(scores), torch.logsumexp(scores.reshape(1, 3, 5), dim=1))
    with pytest.raises(ValueError):
        steering_logits(scores[:, :14])


def test_zero_new_weights_gives_original_cross_entropy():
    logits = torch.randn(7, 15, generator=torch.Generator().manual_seed(16))
    targets = torch.tensor([14, 10, 9, 3, 0, 12, 7])
    torch.testing.assert_close(combined_bc_loss(logits, targets),
                               F.cross_entropy(logits, targets))


def test_steering_loss_pushes_target_steering_across_motor_groups():
    logits = torch.zeros(1, 15, requires_grad=True)
    loss = combined_bc_loss(logits, torch.tensor([14]), steering_loss_weight=2.0)
    loss.backward()
    grad = logits.grad.reshape(3, 5)
    assert grad[:, 4].mean().item() < grad[:, 2].mean().item()


def test_balancing_steering_minority():
    actions = np.array([2] * 70 + [4] * 10 + [0] * 20)
    w = make_steering_class_weights(actions, power=0.65)
    assert w[4] > w[0] > w[2]
    assert w[1] == 0 and w[3] == 0
    assert bool(torch.isfinite(w).all())
    with pytest.raises(ValueError):
        make_steering_class_weights(actions, power=1.1)


def test_start_sampling_raises_start_frequency_without_leaking_validation():
    train_idx = np.array([0, 1, 2, 3, 10, 11, 12, 13])
    rng = np.random.default_rng(77)
    choose = boost_start_indices(rng, train_idx, batch_size=1000,
                                 start_prefix=4, fraction=0.25)
    assert len(choose) == 1000
    assert set(choose).issubset(set(train_idx))
    assert (choose < 4).mean() > 0.55  # 25% forced + 75% * 50% baseline
    with pytest.raises(ValueError):
        boost_start_indices(rng, np.array([8, 9]), 10, 4, 0.2)
