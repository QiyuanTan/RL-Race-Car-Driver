"""DQN decision-level steering smoothness reward for CSE 150A.

Place immediately OUTSIDE DiscreteActionWrapper and ActionRepeatWrapper.
Receives action IDs (one call per DQN decision) and subtracts
  coefficient * abs(steering_now - steering_previous).
Does not change dynamics, observation, or action spaces.
"""
import gymnasium as gym
import numpy as np


class SteeringSmoothnessRewardWrapper(gym.Wrapper):
    def __init__(self, env, coefficient=0.02):
        super().__init__(env)
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError('SteeringSmoothnessRewardWrapper must wrap DiscreteActionWrapper')
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError('coefficient must be finite and nonnegative')
        if not hasattr(env, '_actions'):
            raise TypeError('Pass a DiscreteActionWrapper with _actions directly as env')
        self._steering_by_action = np.array(
            [float(np.asarray(action['steering']).reshape(-1)[0]) for action in env._actions],
            dtype=np.float64,
        )
        self.coefficient = float(coefficient)
        self._previous_steering = None
        self.decisions = 0
        self.total_penalty = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_steering = None
        return observation, info

    def step(self, action):
        action_id = int(action)
        if action_id < 0 or action_id >= len(self._steering_by_action):
            raise ValueError(f'invalid discrete action ID {action_id}')
        current_steering = float(self._steering_by_action[action_id])
        change = (0.0 if self._previous_steering is None
                  else abs(current_steering - self._previous_steering))
        obs, reward, terminated, truncated, info = self.env.step(action)
        penalty = self.coefficient * change
        self._previous_steering = current_steering
        self.decisions += 1
        self.total_penalty += penalty
        info = dict(info)
        info['reward_v3_steering_delta'] = float(change)
        info['reward_v3_steering_penalty'] = float(penalty)
        return obs, float(reward) - penalty, terminated, truncated, info
