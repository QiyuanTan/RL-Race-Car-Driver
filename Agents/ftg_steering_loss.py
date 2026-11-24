"""Optional steering-focused imitation loss for 3 motor x 5 steering actions
Adds a supervised steering objective without changing the DQN architecture, discrete actions, or the environment. Defaults preserve prior motor-aware BC.
"""

import numpy as np
import torch
import torch.nn.functional as F

from Agents.ftg_motor_loss import motor_aware_loss


NUM_MOTOR = 3
NUM_STEER = 5


def steering_logits(action_logits):
    """Marginalize 15-way action logits over the three motor groups."""
    if action_logits.ndim != 2 or action_logits.shape[1] != NUM_MOTOR * NUM_STEER:
        raise ValueError('expected action logits with shape [batch, 15]')
    return torch.logsumexp(action_logits.reshape(-1, NUM_MOTOR, NUM_STEER), dim=1)


def make_steering_class_weights(action_ids, power=0.5, device='cpu'):
    """Inverse-frequency steering weights; absent labels have zero weight."""
    if not 0 <= power <= 1:
        raise ValueError('steering balance power must be in [0, 1]')
    ids = np.asarray(action_ids).reshape(-1)
    if not ids.size or not np.isfinite(ids).all() or (ids != np.round(ids)).any():
        raise ValueError('need finite integer action IDs')
    if (ids < 0).any() or (ids >= NUM_MOTOR * NUM_STEER).any():
        raise ValueError('action IDs must be in [0, 14]')
    counts = np.bincount(ids.astype(np.int64) % NUM_STEER,
                         minlength=NUM_STEER).astype(np.float64)
    weights = np.zeros(NUM_STEER, dtype=np.float64)
    seen = counts > 0
    weights[seen] = (counts[seen].mean() / counts[seen]) ** power
    weights[seen] /= weights[seen].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def combined_bc_loss(logits, actions, motor_loss_weight=0.0, motor_weights=None,
                     steering_loss_weight=0.0, steering_weights=None):
    """Action CE + motor CE + steering CE; all weights are opt-in."""
    if steering_loss_weight < 0:
        raise ValueError('steering loss weight must be nonnegative')
    total, _, _ = motor_aware_loss(logits, actions, motor_loss_weight, motor_weights)
    if steering_loss_weight == 0:
        return total
    return total + steering_loss_weight * F.cross_entropy(
        steering_logits(logits), actions % NUM_STEER, weight=steering_weights)
