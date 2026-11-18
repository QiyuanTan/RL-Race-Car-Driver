from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
"""DQN+ lidar (compare to dqn baseline)add 3 new classes: DiscreteActionWrapper, StartModeWrapper, ActionRepeatWrapper, LidarObservationWrapper"""

class DiscreteActionWrapper(gym.ActionWrapper):
    """Map discrete IDs to absolute motor and steering commands.

    Defaults provide 15 actions: brake/coast/drive crossed with five steering
    positions. Steering varies fastest: ID = motor_index * 5 + steering_index.
    For custom levels, replace 5 with the number of steering levels.
    """

    def __init__(
        self,
        env,
        motor_levels=(-0.5, 0.0, 0.5),
        steering_levels=(-1.0, -0.5, 0.0, 0.5, 1.0),
    ):
        super().__init__(env)
        action_space = env.action_space

        levels = {}
        for key, values in (
            ("motor", motor_levels), ("steering", steering_levels)
        ):
            space = action_space[key]

            values = np.asarray(values, dtype=space.dtype)

            levels[key] = [np.full(space.shape, value, dtype=space.dtype) for value in values]


        self._actions = tuple(
            {"motor": motor, "steering": steering}
            for motor in levels["motor"]
            for steering in levels["steering"]
        )
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        """Decode an ID, preserving the original actuator shapes and dtypes."""
        return {key: value.copy() for key, value in self._actions[int(action)].items()}


class StartModeWrapper(gym.Wrapper):
    """Training only: start each episode with racecar_gym's reset mode, e.g. 'random'.

    'random' puts the car at a random point on the track centre line, facing forward, so the
    agent also practises later parts of the track (such as the hairpin) instead of always
    starting from the grid. An explicit options={'mode': ...} passed to reset() still wins.
    """

    def __init__(self, env, mode='grid'):
        super().__init__(env)
        self.mode = mode

    def reset(self, *, seed=None, options=None):
        if options is None or 'mode' not in options:
            options = dict(options or {}, mode=self.mode)
        return self.env.reset(seed=seed, options=options)


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action for `repeat` simulator steps (0.01 s each); rewards are added up.

    The policy then decides every `repeat` steps, e.g. repeat=4 -> 25 Hz, which matches the
    LiDAR update rate. Stops early when the episode ends, and returns the last observation/info.
    """

    def __init__(self, env, repeat=4):
        super().__init__(env)
        if repeat < 1:
            raise ValueError('repeat must be at least 1')
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class LidarObservationWrapper(gym.ObservationWrapper):
    """Clean and scale the raw LiDAR scan so it can be fed to the policy as a vector.

    racecar_gym's `lidar` observation is Box(low=0.25, high=15.25, shape=(1080,), float64),
    distances in metres. The sensor itself clips readings to [0.25, 15.0], so the space's
    upper bound is never reached; we still scale with the space bounds so the output always
    lies inside the new observation space.

    Output: float32 in [0, 1], computed as (distance - low) / (high - low).
    NaN is treated as the closest distance (a wall, the safe assumption), +inf as the farthest.
    All other observation keys pass through unchanged.
    """

    def __init__(self, env, key="lidar"):
        super().__init__(env)
        observation_space = env.observation_space
        if key not in observation_space.spaces:
            raise KeyError(
                f"observation has no '{key}' key; use a scenario whose sensors include "
                f"'{key}', e.g. scenarios/austria_lidar.yml"
            )
        space = observation_space[key]
        self._key = key
        self._low = float(np.min(space.low))
        self._high = float(np.max(space.high))
        new_spaces = dict(observation_space.spaces)
        new_spaces[key] = gym.spaces.Box(low=0.0, high=1.0, shape=space.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        scan = np.asarray(obs[self._key], dtype=np.float64)
        scan = np.nan_to_num(scan, nan=self._low, posinf=self._high, neginf=self._low)
        scan = np.clip(scan, self._low, self._high)
        observation = dict(obs)
        observation[self._key] = ((scan - self._low) / (self._high - self._low)).astype(np.float32)
        return observation


class FlattenActionWrapper(gym.ActionWrapper):
    """Expose a single agent's Dict of Box actions as one flat Box.

    Coordinates follow the original Dict space's key order. For the racecar,
    this is [motor, steering]. Bounds and each actuator's shape are preserved.
    """

    def __init__(self, env):
        super().__init__(env)
        action_space = env.action_space


        self._original_action_space = action_space
        self.action_space = flatten_space(action_space)

    def action(self, action):
        action = np.asarray(action, dtype=self.action_space.dtype)
        return unflatten(self._original_action_space, action)

    def reverse_action(self, action):
        return flatten(self._original_action_space, action)
