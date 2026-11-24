"""Motor-aware supervised learning for a 3x5 discrete throttle/steering DQN
"""

import numpy as np
import torch
import torch.nn.functional as F


NUM_MOTOR_LEVELS = 3
NUM_STEERING_LEVELS = 5
NUM_ACTIONS = NUM_MOTOR_LEVELS * NUM_STEERING_LEVELS


def motor_logits(action_logits):
    """Marginalize a 15-action categorical distribution into 3 motor groups."""
    if action_logits.ndim != 2 or action_logits.shape[1] != NUM_ACTIONS:
        raise ValueError(f'expected [batch, {NUM_ACTIONS}] action logits')
    return torch.logsumexp(action_logits.reshape(-1, NUM_MOTOR_LEVELS,
                                                 NUM_STEERING_LEVELS), dim=2)


def make_motor_class_weights(action_ids, power=0.5, device='cpu'):
    """Mild inverse-frequency weights computed on training labels only."""
    if not 0 <= power <= 1:
        raise ValueError('motor balance power must be in [0, 1]')
    ids = np.asarray(action_ids).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= NUM_ACTIONS):
        raise ValueError('action IDs must be nonempty and in [0, 14]')
    counts = np.bincount(ids.astype(np.int64) // NUM_STEERING_LEVELS,
                         minlength=NUM_MOTOR_LEVELS).astype(np.float64)
    # A missing class has no valid target; weight 0 instead of dividing by 0.
    weights = np.zeros(NUM_MOTOR_LEVELS, dtype=np.float64)
    seen = counts > 0
    weights[seen] = (counts[seen].mean() / counts[seen]) ** power
    if weights[seen].mean() > 0:
        weights[seen] /= weights[seen].mean()
    return torch.tensor(weights, device=device, dtype=torch.float32)


def motor_aware_loss(logits, target_action, motor_loss_weight=0.0,
                     motor_class_weights=None):
    """Keep action imitation while explicitly rewarding correct motor groups.

    Weight=0 reproduces the previous cross entropy exactly. A positive weight
    adds cross-entropy on logsumexp of the five steering actions per motor.
    """
    if motor_loss_weight < 0:
        raise ValueError('motor loss weight must be nonnegative')
    if target_action.ndim != 1 or logits.shape[0] != target_action.shape[0]:
        raise ValueError('expected [batch] target IDs')
    action_ce = F.cross_entropy(logits, target_action)
    if motor_loss_weight == 0:
        return action_ce, action_ce, action_ce.new_zeros(())
    motor_ce = F.cross_entropy(motor_logits(logits), target_action // NUM_STEERING_LEVELS,
                               weight=motor_class_weights)
    return action_ce + motor_loss_weight * motor_ce, action_ce, motor_ce
