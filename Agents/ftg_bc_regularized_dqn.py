"""SB3 DQN with a small, simultaneous FTG imitation loss

Based on Stable-Baselines3 2.3.2 DQN.train. The Bellman
update is unchanged; an optional supervised BC term is added to the SAME
optimizer step so its coefficient has a meaningful relative scale
"""

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN

from Agents.ftg_behavior_cloning import (
    boost_start_indices, check_demo_buffer, demo_batch, demo_split,
)
from Agents.ftg_motor_loss import make_motor_class_weights
from Agents.ftg_steering_loss import combined_bc_loss, make_steering_class_weights


def joint_dqn_bc_loss(q_values, actions, rewards, dones, next_q_values,
                      gamma, expert_logits, expert_actions, bc_coefficient,
                      motor_loss_weight=1.5, motor_weights=None,
                      steering_loss_weight=2.0, steering_weights=None):
    """Differentiable Bellman Huber + weighted expert action/motor/steering CE.

    Returns (total, td, bc); next_q_values are stop-gradient target values.
    """
    if bc_coefficient < 0:
        raise ValueError('BC coefficient must be nonnegative')
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0, 1]')
    chosen = torch.gather(q_values, dim=1, index=actions.long())
    with torch.no_grad():
        best_next = next_q_values.max(dim=1).values.reshape(-1, 1)
        target = rewards + (1 - dones) * gamma * best_next
    td_loss = F.smooth_l1_loss(chosen, target)
    if bc_coefficient == 0:
        return td_loss, td_loss, td_loss.detach().new_zeros(())
    bc_loss = combined_bc_loss(
        expert_logits, expert_actions, motor_loss_weight, motor_weights,
        steering_loss_weight, steering_weights,
    )
    return td_loss + bc_coefficient * bc_loss, td_loss, bc_loss


class BCRetainedDQN(DQN):
    """Regular DQN except supervised FTG updates stay active during online RL."""

    def configure_bc(self, seed=0, coefficient_start=0.15, coefficient_end=0.10,
                     demo_batch_size=32, motor_loss_weight=1.5,
                     steering_loss_weight=2.0, motor_balance_power=0.65,
                     steering_balance_power=0.6, start_boost_fraction=0.20,
                     start_prefix=80):
        if coefficient_start < 0 or coefficient_end < 0 or demo_batch_size < 1:
            raise ValueError('Invalid BC coefficient or batch size')
        if not 0 <= start_boost_fraction <= 1 or start_prefix < 1:
            raise ValueError('Invalid demonstration start sampling')
        if motor_loss_weight < 0 or steering_loss_weight < 0:
            raise ValueError('Imitation loss weights must be nonnegative')
        check_demo_buffer(self)
        train, validation = demo_split(int(self.replay_buffer.keep), seed=seed)
        self._bc_train_indices = train
        self._bc_validation_indices = validation
        self._bc_rng = np.random.default_rng(seed + 2026)
        self._bc_coefficient_start = float(coefficient_start)
        self._bc_coefficient_end = float(coefficient_end)
        self._bc_batch_size = int(demo_batch_size)
        self._bc_motor_loss_weight = float(motor_loss_weight)
        self._bc_steering_loss_weight = float(steering_loss_weight)
        self._bc_start_boost_fraction = float(start_boost_fraction)
        self._bc_start_prefix = int(start_prefix)
        labels = self.replay_buffer.actions[train, 0].reshape(-1)
        self._bc_motor_weights = make_motor_class_weights(
            labels, power=motor_balance_power, device=self.device)
        self._bc_steering_weights = make_steering_class_weights(
            labels, power=steering_balance_power, device=self.device)
        if self._bc_start_boost_fraction and not np.any(train < self._bc_start_prefix):
            raise ValueError('No start-curve expert samples in the BC training subset')
        print(f'[dqn+bc] {len(train)} expert training decisions; '
              f'{len(validation)} held out for BC diagnostics; '
              f'BC weight {coefficient_start:.3f} -> {coefficient_end:.3f}', flush=True)

    def train(self, gradient_steps, batch_size=100):
        if not hasattr(self, '_bc_train_indices'):
            raise RuntimeError('Call configure_bc() after loading demonstration buffer')
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        td_losses, bc_losses, total_losses = [], [], []
        # SB3 2.3.2 DQN Bellman update, with expert regularization on each step.
        for _ in range(gradient_steps):
            replay = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with torch.no_grad():
                next_q_values = self.q_net_target(replay.next_observations)
            q_values = self.q_net(replay.observations)
            progress = float(np.clip(self._current_progress_remaining, 0.0, 1.0))
            coefficient = (self._bc_coefficient_end + progress *
                           (self._bc_coefficient_start - self._bc_coefficient_end))
            expert_obs, expert_actions = demo_batch(
                self,
                boost_start_indices(
                    self._bc_rng, self._bc_train_indices, self._bc_batch_size,
                    self._bc_start_prefix, self._bc_start_boost_fraction,
                ),
            )
            expert_logits = self.q_net(expert_obs)
            total_loss, td_loss, bc_loss = joint_dqn_bc_loss(
                q_values, replay.actions, replay.rewards, replay.dones,
                next_q_values, self.gamma, expert_logits, expert_actions,
                coefficient, self._bc_motor_loss_weight, self._bc_motor_weights,
                self._bc_steering_loss_weight, self._bc_steering_weights,
            )
            self.policy.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            td_losses.append(float(td_loss.item()))
            bc_losses.append(float(bc_loss.item()))
            total_losses.append(float(total_loss.item()))
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', float(np.mean(td_losses)))
        self.logger.record('train/bc_loss', float(np.mean(bc_losses)))
        self.logger.record('train/joint_loss', float(np.mean(total_losses)))
        self.logger.record('train/bc_coefficient', coefficient)