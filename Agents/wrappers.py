from typing import Dict

import gymnasium as gym
import numpy as np
from collections import deque
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
    """for training only: start each episode with racecar_gym's reset mode, e.g. 'random'.

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
    """repeat each action for `repeat` simulator steps (0.01 s each); rewards are added up.

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
    """clean and scale the raw LiDAR scan so it can be fed to the policy as a vector.

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


class SimStepCounter(gym.Wrapper):
    """Counts real simulator steps (training only), so config.json can report the actual numbers."""

    def __init__(self, env):
        super().__init__(env)
        self.sim_steps = 0
        self.episodes = 0

    def reset(self, **kwargs):
        self.episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        self.sim_steps += 1
        return self.env.step(action)

class RemoveCameraWrapper(gym.ObservationWrapper):
    """Remove RGB camera observations while keeping numeric features."""

    def __init__(self, env):
        super().__init__(env)

        spaces = dict(env.observation_space.spaces)
        spaces.pop("rgb_camera", None)

        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs):
        return {
            k: v for k, v in obs.items()
            if k != "rgb_camera"
        }

class ForwardProgressRewardWrapper(gym.Wrapper):
    """Training-only reward: positive signed forward progress minus collision cost.

    Uses `lap + progress` to handle the finish-line wrap. Must sit INSIDE
    ActionRepeatWrapper so each simulator step has its own progress delta.
    The environment still decides when an episode terminates.
    """

    def __init__(self, env, progress_scale=100.0, collision_penalty=1.0):
        super().__init__(env)
        if progress_scale < 0 or collision_penalty < 0:
            raise ValueError("progress_scale and collision_penalty must be nonnegative")
        self.progress_scale = float(progress_scale)
        self.collision_penalty = float(collision_penalty)
        self._previous_progress = None

    @staticmethod
    def _absolute_progress(info):
        return float(info["lap"]) + float(info["progress"])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Read from reset info, not a fixed 0: random starts can begin anywhere.
        self._previous_progress = self._absolute_progress(info)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        current = self._absolute_progress(info)
        if self._previous_progress is None:
            raise RuntimeError("Call reset() before step()")
        delta = current - self._previous_progress
        self._previous_progress = current

        # Ignore implausible jumps rather than rewarding a state discontinuity.
        forward_delta = max(0.0, delta) if np.isfinite(delta) and abs(delta) <= 0.5 else 0.0
        collided = bool(info.get("wall_collision", False)) or bool(info.get("opponent_collisions", []))
        reward = self.progress_scale * forward_delta - self.collision_penalty * float(collided)
        return obs, float(reward), terminated, truncated, info


class ForwardProgressStallRewardWrapper(ForwardProgressRewardWrapper):
    """Train-only forward/collision reward plus a sustained-stall penalty.

    Installed INSIDE ActionRepeatWrapper: one call here == one simulator step.
    Each active simulator step applies -stall_penalty_per_decision/action_repeat,
    yielding at most -stall_penalty_per_decision per complete DQN decision.

    A stall requires BOTH a low current speed and almost no accumulated
    forward progress over the last stall_window simulator steps. This avoids
    penalizing individual zero-progress samples from discretized map projection.
    """

    def __init__(self, env, progress_scale=100.0, collision_penalty=1.0,
                 stall_window=40, stall_progress_threshold=1e-4,
                 stall_speed_threshold=0.2, stall_penalty_per_decision=0.03,
                 action_repeat=4):
        super().__init__(env, progress_scale=progress_scale,
                         collision_penalty=collision_penalty)
        if not isinstance(stall_window, int) or stall_window < 1:
            raise ValueError("stall_window must be a positive integer")
        if not isinstance(action_repeat, int) or action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        values = (stall_progress_threshold, stall_speed_threshold,
                  stall_penalty_per_decision)
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError("stall thresholds and penalty must be finite and nonnegative")
        self.stall_window = stall_window
        self.stall_progress_threshold = float(stall_progress_threshold)
        self.stall_speed_threshold = float(stall_speed_threshold)
        self.stall_penalty_per_sim_step = float(stall_penalty_per_decision) / action_repeat
        self._progress_window = deque(maxlen=stall_window)
        self.stall_sim_steps = 0
        self.total_stall_penalty = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._progress_window.clear()
        self.stall_sim_steps = 0
        self.total_stall_penalty = 0.0
        return obs, info

    def step(self, action):
        # super().step updates _previous_progress and preserves collision cost.
        before = self._previous_progress
        obs, reward, terminated, truncated, info = super().step(action)
        after = self._previous_progress
        delta = after - before
        # Include only plausible forward motion; backward does not count as progress.
        forward_delta = max(delta, 0.0) if np.isfinite(delta) and abs(delta) <= 0.5 else 0.0
        self._progress_window.append(forward_delta)
        velocity = np.asarray(info['velocity'], dtype=np.float64).reshape(-1)
        if velocity.size < 2:
            raise ValueError("info['velocity'] must provide at least x and y components")
        speed = float(np.hypot(velocity[0], velocity[1]))
        stalled = (len(self._progress_window) == self.stall_window
                   and sum(self._progress_window) < self.stall_progress_threshold
                   and np.isfinite(speed) and speed < self.stall_speed_threshold)
        penalty = self.stall_penalty_per_sim_step if stalled else 0.0
        if stalled:
            self.stall_sim_steps += 1
            self.total_stall_penalty += penalty
        # Diagnostics are training-only; do not alter the policy observation.
        info = dict(info)
        info['reward_v2_stalled'] = bool(stalled)
        info['reward_v2_stall_penalty'] = float(penalty)
        return obs, float(reward - penalty), terminated, truncated, info
