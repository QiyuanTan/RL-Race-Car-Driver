
"""add a large-margin loss to DQN for demos .

the TD loss trains the Q-value of the action taken. in a demo state,
another action can still score higher. the margin loss pushes the demo
action at least margin above every other action.

margin_loss = mean(max_a(Q(s, a) + l(a)) - Q(s, a_demo))
l(a) is 0 for the demo action and margin for every other action.

use TD loss on the whole batch and add margin_weight * margin_loss on
the demo samples. pretrain on demos before DQN starts driving.

evaluate_sb3.py can still load the saved model with DQN.load.
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN


class MarginDQN(DQN):
    margin_weight = 1.0
    margin = 0.8

    def _losses(self, replay_data, n_demo):
        """compute TD loss on the whole batch and margin loss on the last n_demo samples."""
        with th.no_grad():
            next_q = self.q_net_target(replay_data.next_observations).max(dim=1)[0].reshape(-1, 1)
            target = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q
        q_all = self.q_net(replay_data.observations)
        actions = replay_data.actions.long()
        td_loss = F.smooth_l1_loss(th.gather(q_all, dim=1, index=actions), target)
        if n_demo == 0 or self.margin_weight == 0:
            return td_loss, td_loss, th.zeros(())
        q_demo, a_demo = q_all[-n_demo:], actions[-n_demo:]
        margins = th.full_like(q_demo, self.margin).scatter(1, a_demo, 0.0)
        margin_loss = ((q_demo + margins).max(dim=1)[0] - q_demo.gather(1, a_demo).squeeze(1)).mean()
        return td_loss + self.margin_weight * margin_loss, td_loss, margin_loss

    def _step(self, loss):
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

    def train(self, gradient_steps, batch_size=100):
        """run DQN updates and log the TD and margin losses separately."""        
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        td_losses, margin_losses = [], []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            loss, td_loss, margin_loss = self._losses(replay_data, getattr(self.replay_buffer, 'last_n_demo', 0))
            td_losses.append(td_loss.item())
            margin_losses.append(margin_loss.item())
            self._step(loss)
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', np.mean(td_losses))
        self.logger.record('train/margin_loss', np.mean(margin_losses))

    def pretrain_on_demos(self, steps, batch_size=None):
        """train on demos for steps updates, then sync the target network.

return the first and last margin losses to check whether it dropped.
"""
        batch_size = batch_size or self.batch_size
        demo = self.replay_buffer.demo
        self.policy.set_training_mode(True)
        first = last = None
        for i in range(steps):
            loss, _, margin_loss = self._losses(demo.sample(batch_size), batch_size)
            self._step(loss)
            if i == 0:
                first = margin_loss.item()
            last = margin_loss.item()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.policy.set_training_mode(False)
        return first, last

    def demo_agreement(self, n=1000):
        """return how often the greedy action matches the demo action on up to n samples."""
        demo = self.replay_buffer.demo
        data = demo.sample(min(n, demo.size()))
        with th.no_grad():
            chosen = self.q_net(data.observations).argmax(dim=1)
        return float((chosen == data.actions.long().squeeze(1)).float().mean())
