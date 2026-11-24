"""Decision-level, progress-gated centerline guidance for DQN.

Installed OUTSIDE dqn_wrappers(...), which means it observes the `track` key
produced by TrackFeatureWrapper and is called once per DQN action (AR4),
not once per physics step.

Uses the verified Agents/track_features.py feature layout:
  track[0] = lateral offset / 1.5 metres
  track[1] = sin(yaw - track direction)
  track[2] = cos(yaw - track direction).
"""
from dataclasses import dataclass
import math
import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class Guidance:
    lateral_index: int = 0
    heading_index: int = 1         # sin heading error
    heading_cos_index: int = 2     # cos heading error
    lateral_units: float = 1.5     # physical meters per normalized feature unit
    half_width_m: float = 1.0       # explicit normalization; not necessarily track true width
    lateral_weight: float = 0.15
    heading_weight: float = 0.15
    progress_scale: float = 100.0

    def validate(self):
        indices = [self.lateral_index, self.heading_index, self.heading_cos_index]
        if any(not isinstance(i, int) or i < 0 for i in indices) or len(set(indices)) != 3:
            raise ValueError('lateral, heading-sin and heading-cos indices must be distinct nonnegative integers')
        for name in ('lateral_units', 'half_width_m', 'progress_scale'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and > 0')
        if (min(self.lateral_weight, self.heading_weight) < 0 or
                self.lateral_weight + self.heading_weight >= 1.0):
            raise ValueError('weights must be >=0 and sum to < 1, preserving forward incentive')


def guidance_terms(track, config):
    """Return (normalized offset, angular error rad, bounded discount in [0,1))."""
    config.validate()
    features = np.asarray(track, dtype=np.float64).reshape(-1)
    if max(config.lateral_index, config.heading_index, config.heading_cos_index) >= features.size:
        raise ValueError(f'track has {features.size} features but configured index is out of bounds')
    lateral_raw = float(features[config.lateral_index])
    heading_sin = float(features[config.heading_index])
    heading_cos = float(features[config.heading_cos_index])
    if not all(map(math.isfinite, (lateral_raw, heading_sin, heading_cos))):
        raise ValueError('track contains nonfinite lateral or heading feature')
    if abs(heading_sin) > 1.01 or abs(heading_cos) > 1.01 or math.hypot(heading_sin, heading_cos) < 0.8:
        raise ValueError('heading sin/cos values are inconsistent with track_features.py')
    heading_rad = math.atan2(heading_sin, heading_cos)
    offset = abs(lateral_raw * config.lateral_units) / config.half_width_m
    lateral_cost = min(offset, 1.0)
    heading_cost = (1.0 - math.cos(heading_rad)) / 2.0
    discount = config.lateral_weight * lateral_cost + config.heading_weight * heading_cost
    return float(offset), float(heading_rad), float(discount)


class CenterlineGuidanceReward(gym.Wrapper):
    """Add NON-POSITIVE guided cost proportional to actual forward motion.

    reward = V2_reward - forward_progress_reward * geometry_discount

    The forward progress component is derived from lap+progress in info.
    A stationary, correctly aligned vehicle never collects heading rewards;
    a progressing off-center vehicle still has positive progress incentive because
    geometry_discount < 1. Collision and stall terms from V2 are untouched.
    """
    def __init__(self, env, config: Guidance):
        super().__init__(env)
        config.validate()
        if not isinstance(env.observation_space, gym.spaces.Dict) or 'track' not in env.observation_space.spaces:
            raise ValueError('Guidance requires obs_mode=track-lookahead and observation key "track"')
        shape = env.observation_space['track'].shape
        if math.prod(shape) <= max(config.lateral_index, config.heading_index, config.heading_cos_index):
            raise ValueError(f'track shape {shape} is incompatible with configured indices')
        self.config = config
        self.last_progress = None
        self.total_guidance_cost = 0.0
        self.steps = 0

    @staticmethod
    def _absolute_progress(info):
        return float(info['lap']) + float(info['progress'])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_progress = self._absolute_progress(info)
        self.total_guidance_cost = 0.0
        self.steps = 0
        return obs, info

    def step(self, action):
        if self.last_progress is None:
            raise RuntimeError('reset must precede step')
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        now = self._absolute_progress(info)
        dp = now - self.last_progress
        self.last_progress = now
        forward_dp = max(0.0, dp) if math.isfinite(dp) and abs(dp) <= 0.5 else 0.0
        offset, heading, discount = guidance_terms(obs['track'], self.config)
        penalty = self.config.progress_scale * forward_dp * discount
        self.total_guidance_cost += penalty
        self.steps += 1
        info = dict(info)
        info.update({
            'curriculum/forward_progress_reward': self.config.progress_scale * forward_dp,
            'curriculum/lateral_abs_halfwidth': offset,
            'curriculum/heading_error_rad': heading,
            'curriculum/guidance_penalty': -penalty,
            'curriculum/base_reward': float(base_reward),
            'curriculum/total_reward': float(base_reward - penalty),
        })
        return obs, float(base_reward - penalty), terminated, truncated, info
