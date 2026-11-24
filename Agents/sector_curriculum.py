"""Stage 1B safe bounded rejection sampling and a terminal sector goal.

This DOES NOT modify racecar_gym internals. Reset repeatedly with mode='random',
accepting only actual reset info['progress'] in the configured interval.
Place SectorStartFilter directly around the base environment, BEFORE any reward
wrapper and BEFORE simulated-step counters. The reset loop is expensive but safe.
"""
from __future__ import annotations

import math
import gymnasium as gym


class SectorStartFilter(gym.Wrapper):
    def __init__(self, env, start_min=0.26, start_max=0.31, max_attempts=120):
        super().__init__(env)
        if not (0 <= start_min < start_max < 1):
            raise ValueError('Require 0 <= start_min < start_max < 1')
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError('max_attempts must be a positive integer')
        self.start_min = float(start_min)
        self.start_max = float(start_max)
        self.max_attempts = max_attempts
        self.episode_start_progress = None
        self.last_reset_attempts = 0
        self.total_resets_attempted = 0
        self.total_episodes_accepted = 0

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if 'mode' in options and options['mode'] != 'random':
            raise ValueError('SectorStartFilter must use mode=random')
        options['mode'] = 'random'
        sample = []
        for attempt in range(self.max_attempts):
            # For an explicit seed, use a different reproducible seed on each
            # rejection; for seed=None, allow the environment RNG to advance.
            attempt_seed = None if seed is None else int(seed) + attempt
            obs, info = self.env.reset(seed=attempt_seed, options=options)
            self.total_resets_attempted += 1
            if 'progress' not in info:
                raise KeyError('racecar_gym reset info missing progress; cannot filter sector')
            progress = float(info['progress'])
            sample.append(progress)
            if math.isfinite(progress) and self.start_min <= progress <= self.start_max:
                self.last_reset_attempts = attempt + 1
                self.episode_start_progress = progress
                self.total_episodes_accepted += 1
                result = dict(info)
                result.update({'sector/start_progress': progress,
                               'sector/reset_attempts': attempt + 1})
                return obs, result
        raise RuntimeError(
            f'No random start in [{self.start_min:.3f}, {self.start_max:.3f}] '
            f'after {self.max_attempts} resets. Recent progress samples: '
            f'{[round(x, 4) for x in sample[-10:]]}. '
            'Verify racecar_gym random-start support and progress coverage.'
        )


class SectorGoalReward(gym.Wrapper):
    """Terminate upon actually reaching a sector exit progress threshold.

    Install OUTSIDE the V2 reward and INSIDE ActionRepeatWrapper. It therefore
    receives the already shaped V2 single-simulator-step reward. A success bonus
    can be applied exactly once, and simulated action repeating ends immediately.
    """
    def __init__(self, env, target_progress=0.42, success_bonus=2.0):
        super().__init__(env)
        if not (0 < target_progress < 1):
            raise ValueError('target_progress must be in (0,1)')
        if not math.isfinite(success_bonus) or success_bonus < 0:
            raise ValueError('success_bonus must be nonnegative and finite')
        self.target_progress = float(target_progress)
        self.success_bonus = float(success_bonus)
        self._reached = False

    def reset(self, **kwargs):
        self._reached = False
        obs, info = self.env.reset(**kwargs)
        progress = float(info['progress'])
        if progress >= self.target_progress:
            raise ValueError('Sector start must precede the goal; check sector interval')
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        progress = float(info['progress'])
        collision = bool(info.get('wall_collision', False)) or bool(info.get('opponent_collisions', []))
        success = (not self._reached and not terminated and not truncated
                   and not collision and math.isfinite(progress)
                   and self.target_progress <= progress < 0.95)
        if success:
            self._reached = True
            reward = float(reward) + self.success_bonus
            terminated = True
        info['sector/success'] = success
        info['sector/goal_progress'] = self.target_progress
        return obs, float(reward), terminated, truncated, info
