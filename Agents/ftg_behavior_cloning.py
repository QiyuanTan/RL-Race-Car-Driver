"""start an SB3 DQN using successful, discrete-action FTG demonstrations

DQN Q-network outputs are used as action logits for cross entropy. No FTG or LiDAR is needed at deployment or evaluation.
The caller must provide a replay buffer whose first `keep` rows are the FTG transitions. Validation uses a held-out fraction of these stored decisions.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from Agents.ftg_motor_loss import (make_motor_class_weights, motor_aware_loss)
from Agents.ftg_steering_loss import (make_steering_class_weights, combined_bc_loss)


def check_demo_buffer(model):
    """Verify preloaded teacher transitions are compatible with the student."""
    buffer = model.replay_buffer
    keep = int(getattr(buffer, 'keep', 0))
    if keep < 100:
        raise ValueError('Replay buffer has no usable FTG demo prefix (keep < 100).')
    if keep > buffer.buffer_size:
        raise ValueError('Demo prefix exceeds replay buffer capacity.')
    if buffer.size() < keep:
        raise ValueError('Demo prefix exceeds replay buffer content.')

    expected = model.observation_space.spaces
    actual = buffer.observation_space.spaces
    if set(actual) != set(expected):
        raise ValueError(f'Demo observation keys {sorted(actual)} do not match '
                         f'student keys {sorted(expected)}')
    for key, space in expected.items():
        if actual[key].shape != space.shape or actual[key].dtype != space.dtype:
            raise ValueError(f'Demo observation space mismatch for {key}: '
                             f'{actual[key]} != {space}')
    if 'lidar' in expected:
        raise ValueError('DQN student must not receive LiDAR')
    if 'track' not in expected or 'lookahead' not in expected:
        raise ValueError('DQN student must use track-lookahead observations')

    labels = np.asarray(buffer.actions[:keep, 0]).reshape(-1)
    if not np.all(np.isfinite(labels)) or np.any(labels != np.round(labels)):
        raise ValueError('Demonstration action labels must be finite integer IDs')
    if np.any(labels < 0) or np.any(labels >= model.action_space.n):
        raise ValueError('Demonstration contains invalid discrete action IDs')
    return keep


def project_demo_buffer(buffer, target_space):
    """Discard saved image/other data; retain only the target observation keys.

    The original successful FTG data is stored in the same coordinate frame.
    Only the student's input fields are changed; actions/rewards are untouched.
    """
    expected = target_space.spaces
    actual = buffer.observation_space.spaces
    if not set(expected).issubset(actual):
        raise ValueError('Saved demos do not contain all student feature keys')
    for key, space in expected.items():
        if space.shape != actual[key].shape or space.dtype != actual[key].dtype:
            raise ValueError(f'Incompatible demo observation field: {key}')
    if 'lidar' in expected:
        raise ValueError('LiDAR cannot be a student input')
    buffer.observations = {key: buffer.observations[key] for key in expected}
    buffer.next_observations = {key: buffer.next_observations[key] for key in expected}
    buffer.observation_space = target_space


def demo_split(keep, seed=0, validation_fraction=0.10):
    """Deterministically partition demonstration indices, independent of training RNG."""
    if keep < 2 or not 0 < validation_fraction < 1:
        raise ValueError('Need at least 2 samples and a validation fraction in (0,1)')
    rng = np.random.default_rng(seed)
    indices = rng.permutation(keep)
    n_val = min(keep - 1, max(1, int(round(keep * validation_fraction))))
    return indices[n_val:], indices[:n_val]


def boost_start_indices(rng, train_idx, batch_size, start_prefix, fraction):
    """Sample training-only FTG indices, optionally emphasizing the first curve."""
    if batch_size <= 0 or not 0 <= fraction <= 1 or start_prefix < 1:
        raise ValueError('Invalid start sampling configuration')
    choose = rng.choice(train_idx, size=batch_size, replace=True)
    if fraction:
        pool = train_idx[train_idx < start_prefix]
        if not len(pool):
            raise ValueError('No training samples in start-prefix window')
        n = int(round(batch_size * fraction))
        if n:
            choose[:n] = rng.choice(pool, size=n, replace=True)
            rng.shuffle(choose)
    return choose


def demo_batch(model, indices):
    """Turn selected SB3 DictReplayBuffer entries into Q-network-ready tensors."""
    buffer = model.replay_buffer
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= int(buffer.keep)):
        raise ValueError('Batch indices must refer only to the retained demo prefix')
    observations = {
        key: torch.as_tensor(data[indices, 0], device=model.device)
        for key, data in buffer.observations.items()
    }
    actions = torch.as_tensor(
        buffer.actions[indices, 0].reshape(-1), device=model.device, dtype=torch.long
    )
    return observations, actions


def batch_metrics(model, indices, batch_size=64, motor_loss_weight=0.0,
                  motor_weights=None, steering_loss_weight=0.0,
                  steering_weights=None):
    """Held-out action loss, motor loss and per-motor recall."""
    model.policy.q_net.eval()
    total_loss = 0.0
    correct = 0
    correct_motor = 0
    true_motor_counts = np.zeros(3, dtype=np.int64)
    correct_motor_counts = np.zeros(3, dtype=np.int64)
    predicted_motor_counts = np.zeros(3, dtype=np.int64)
    steer_total = np.zeros(5, dtype=np.int64)
    steer_correct = np.zeros(5, dtype=np.int64)
    count = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            obs, actions = demo_batch(model, indices[start:start + batch_size])
            logits = model.policy.q_net(obs)
            loss = combined_bc_loss(logits, actions, motor_loss_weight,
                                    motor_weights, steering_loss_weight,
                                    steering_weights)
            total_loss += float(loss.item()) * actions.numel()
            predicted = logits.argmax(dim=1)
            correct += int((predicted == actions).sum().item())
            target_motor = actions // 5
            pred_motor = predicted // 5
            correct_motor += int((pred_motor == target_motor).sum().item())
            for m in range(3):
                mask = (target_motor == m)
                true_motor_counts[m] += int(mask.sum().item())
                correct_motor_counts[m] += int(((pred_motor == m) & mask).sum().item())
                predicted_motor_counts[m] += int((pred_motor == m).sum().item())
            for s in range(5):
                mask = (actions % 5 == s)
                steer_total[s] += int(mask.sum().item())
                steer_correct[s] += int(((predicted % 5 == s) & mask).sum().item())
            count += int(actions.numel())
    return {
        'loss': total_loss / count, 'accuracy': correct / count,
        'motor_accuracy': correct_motor / count,
        'motor_recall': [round(float(correct_motor_counts[m] / true_motor_counts[m]), 4)
                         if true_motor_counts[m] else None for m in range(3)],
        'predicted_motor_fraction': (predicted_motor_counts / count).round(4).tolist(),
        'true_motor_fraction': (true_motor_counts / count).round(4).tolist(),
        'steering_accuracy': float(steer_correct.sum() / count),
        'steering_recall': [round(float(steer_correct[s] / steer_total[s]), 4)
                            if steer_total[s] else None for s in range(5)],
    }


def behavior_clone_dqn(model, steps=800, batch_size=64, seed=0,
                       validation_fraction=0.1, validate_every=50, patience=250,
                       motor_loss_weight=0.0, motor_balance_power=0.5,
                       steering_loss_weight=0.0, steering_balance_power=0.5,
                       start_boost_fraction=0.0, start_prefix=80):
    """Imitate demonstration actions, restore best validation weights, sync DQN target.

    `steps` is the maximum number of supervised optimizer updates. This does
    not change model.num_timesteps or SB3's RL exploration schedule.
    """
    if steps <= 0 or batch_size <= 0 or validate_every <= 0 or patience < validate_every:
        raise ValueError('Invalid BC training/validation parameters')
    if motor_loss_weight < 0 or not 0 <= motor_balance_power <= 1:
        raise ValueError('Invalid motor-loss or balancing parameters')
    if steering_loss_weight < 0 or not 0 <= steering_balance_power <= 1:
        raise ValueError('Invalid steering-loss or balancing parameters')
    if not 0 <= start_boost_fraction <= 1 or start_prefix < 1:
        raise ValueError('Invalid start sampling parameters')
    keep = check_demo_buffer(model)
    train_idx, val_idx = demo_split(keep, seed=seed, validation_fraction=validation_fraction)
    rng = np.random.default_rng(seed + 729)
    motor_weights = None
    if motor_loss_weight:
        train_labels = model.replay_buffer.actions[train_idx, 0].reshape(-1)
        motor_weights = make_motor_class_weights(
            train_labels, power=motor_balance_power, device=model.device)
        print(f'[bc] motor weights (brake/coast/drive): '
              f'{motor_weights.detach().cpu().numpy().round(3).tolist()}', flush=True)
    steering_weights = None
    if steering_loss_weight:
        train_labels = model.replay_buffer.actions[train_idx, 0].reshape(-1)
        steering_weights = make_steering_class_weights(
            train_labels, power=steering_balance_power, device=model.device)
        print(f'[bc] steering weights (left to right): '
              f'{steering_weights.detach().cpu().numpy().round(3).tolist()}', flush=True)
    if start_boost_fraction and not np.any(train_idx < start_prefix):
        raise ValueError('No retained teacher start samples in train split')
    optimizer = model.policy.optimizer  # same optimizer state used by SB3 DQN later
    qnet = model.policy.q_net
    best_loss = float('inf')
    best_weights = None
    best_step = 0
    performed = 0
    initial = batch_metrics(model, val_idx, batch_size, motor_loss_weight, motor_weights,
                            steering_loss_weight, steering_weights)
    print(f'[bc] FTG demo samples={keep} train={len(train_idx)} validation={len(val_idx)} '
          f'initial_val_accuracy={initial["accuracy"]:.3f}', flush=True)

    for step in range(1, steps + 1):
        qnet.train()
        choose = boost_start_indices(rng, train_idx, batch_size, start_prefix,
                                     start_boost_fraction)
        obs, actions = demo_batch(model, choose)
        logits = qnet(obs)
        loss = combined_bc_loss(logits, actions, motor_loss_weight, motor_weights,
                                steering_loss_weight, steering_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
        optimizer.step()
        performed = step

        if step % validate_every == 0 or step == steps:
            val = batch_metrics(model, val_idx, batch_size, motor_loss_weight,
                                motor_weights, steering_loss_weight, steering_weights)
            print(f'[bc] step={step}/{steps} train_loss={loss.item():.4f} '
                  f'val_loss={val["loss"]:.4f} val_action_accuracy={val["accuracy"]:.3f} '
                  f'motor_accuracy={val["motor_accuracy"]:.3f} '
                  f'motor_recall={val["motor_recall"]} '
                  f'steering_accuracy={val["steering_accuracy"]:.3f} '
                  f'steering_recall={val["steering_recall"]}', flush=True)
            if val['loss'] < best_loss - 1e-5:
                best_loss = val['loss']
                best_weights = deepcopy(qnet.state_dict())
                best_step = step
            if step - best_step >= patience:
                print('[bc] early stopping: held-out loss no longer improved', flush=True)
                break

    if best_weights is None:
        raise RuntimeError('BC did not produce a validation checkpoint')
    qnet.load_state_dict(best_weights)
    model.policy.q_net_target.load_state_dict(qnet.state_dict())
    model.policy.q_net_target.eval()
    final = batch_metrics(model, val_idx, batch_size, motor_loss_weight,
                          motor_weights, steering_loss_weight, steering_weights)
    qnet.eval()
    print(f'[bc] best_step={best_step} final_val_action_accuracy={final["accuracy"]:.3f} '
          f'(initial={initial["accuracy"]:.3f})', flush=True)
    return {'demo_decisions': keep, 'requested_steps': steps, 'performed_steps': performed,
            'best_step': best_step, 'initial_validation': initial, 'final_validation': final,
            'validation_fraction': validation_fraction, 'seed': seed,
            'method': 'action cross-entropy with optional motor-aware supervision',
            'motor_loss_weight': motor_loss_weight,
            'motor_balance_power': motor_balance_power,
            'steering_loss_weight': steering_loss_weight,
            'steering_balance_power': steering_balance_power,
            'start_boost_fraction': start_boost_fraction,
            'start_prefix': start_prefix}
