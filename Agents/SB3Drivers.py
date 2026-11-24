import gymnasium as gym
import numpy as np
from functools import partial

import gymnasium as gym
import numpy as np
from Agents.track_features import OBS_MODES, TrackFeatureWrapper
from Agents.wrappers2 import SignedProgressRewardWrapper
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import FilterObservation
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from Agents.Driver import Driver
from Agents.wrappers import (DiscreteActionWrapper, LidarObservationWrapper,
                             ActionRepeatWrapper, RemoveCameraWrapper)
from Agents.wrappers import FlattenActionWrapper, TrackObservationWrapper
from Agents.wrappers import (
    DiscreteActionWrapper,
    DrivableTrackObservationWrapper,
    FlattenActionWrapper,
    LocalDrivableTrackObservationWrapper,
    LocalTrackObservationWrapper,
    TrackObservationWrapper,
)


class SB3Driver(Driver):
    def __init__(self, env, model_class, wrappers=None, policy="MultiInputPolicy", **kwargs):
        if wrappers is None:
            wrappers = []
        self.env_wrappers = wrappers
        if isinstance(env, (list, tuple)):
            self.env = SubprocVecEnv(
                [partial(self._adapt_env, factory) for factory in env],
                start_method="spawn",
            )
        else:
            self.env = self.apply_wrappers(env)
        super().__init__(self.env)
        self.model_class = model_class
        self.model = model_class(policy, self.env, **kwargs)


    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self.model_class.load(path)

    def train(self, total_timesteps, progress_bar=True, callback=None, reset_num_timesteps=True):
        self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar,
                         callback=callback, reset_num_timesteps=reset_num_timesteps)

    def act(self, obs):
        action, _states = self.model.predict(obs)
        return action

    def reset(self):
        pass

    def act_raw(self, obs, deterministic=True):
        """Raw environment observation -> raw environment action.
        Use this when stepping the unwrapped environment, e.g. from
        evaluate_baseline.run_episode.
        Only works for a single/non-vectorized) environment.
        """
        if isinstance(self.env, VecEnv):
            raise TypeError('act_raw needs a single environment, not a vectorized one')
        chain = [] # outermost wrapper first
        env = self.env
        while isinstance(env, gym.Wrapper):
            chain.append(env)
            env = env.env
        for wrapper in reversed(chain):# obs: innermost wrapper first
            if isinstance(wrapper, ObservationWrapper):
                obs = wrapper.observation(obs)
        action, _states = self.model.predict(obs, deterministic=deterministic)

        for wrapper in chain:# actions: outermost wrapper first
            if isinstance(wrapper, gym.ActionWrapper):
                action = wrapper.action(action)
        return action

    def apply_wrappers(self, env):
        for wrapper in self.env_wrappers:
            env = wrapper(env)
        return env

    def _adapt_env(self, factory):
        env = factory()
        env = self.apply_wrappers(env)
        return env


class SB3ObservationWrapper(ObservationWrapper):
    """Convert scalar Box entries in Dict observations to one-element vectors."""

    def __init__(self, env):
        super().__init__(env)
        self._scalar_spaces = {}
        observation_space = env.observation_space
        if isinstance(observation_space, spaces.Dict):
            self._scalar_spaces = {
                key: space
                for key, space in observation_space.spaces.items()
                if isinstance(space, spaces.Box) and space.shape == ()
            }
            if self._scalar_spaces:
                self.observation_space = spaces.Dict([
                    (key, spaces.Box(
                        low=space.low.reshape(1),
                        high=space.high.reshape(1),
                        dtype=space.dtype,
                    ) if key in self._scalar_spaces else space)
                    for key, space in observation_space.spaces.items()
                ])

    def observation(self, obs):
        if not self._scalar_spaces:
            return obs
        observation = obs.copy()
        for key, space in self._scalar_spaces.items():
            observation[key] = np.asarray(obs[key], dtype=space.dtype).reshape(1)
        return observation


class TrackFeaturesExtractor(CombinedExtractor):
    """Encode the normalized track channels with a CNN alongside the sensors."""

    def __init__(self, observation_space, **kwargs):
        track_key = TrackObservationWrapper.KEY
        sensor_space = spaces.Dict([
            (key, space) for key, space in observation_space.spaces.items() if key != track_key
        ])
        super().__init__(sensor_space, **kwargs)
        track_features = 128
        self.extractors[track_key] = nn.Sequential(
            nn.Conv2d(observation_space[track_key].shape[0], 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, track_features),
            nn.ReLU(),
        )
        self._features_dim += track_features


class DrivableTrackFeaturesExtractor(TrackFeaturesExtractor):
    TRACK_KEY = DrivableTrackObservationWrapper.KEY


class _TrackViewDriver(SB3Driver):
    _features_extractor_class = TrackFeaturesExtractor

    def __init__(self, env, model_class, action_wrapper, track_wrapper, *, policy_kwargs=None, **kwargs):
        wrappers = [
            action_wrapper,
            SB3ObservationWrapper,
            track_wrapper,
            self._exclude_rgb_camera,
            Monitor,  # SB3 skips automatic monitoring for already-vectorized environments.
        ]
        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs.setdefault("features_extractor_class", self._features_extractor_class)
        super().__init__(env, model_class, wrappers=wrappers, policy_kwargs=policy_kwargs, **kwargs)

    @staticmethod
    def _exclude_rgb_camera(env):
        keys = [key for key in env.observation_space.spaces if key != "rgb_camera"]
        return FilterObservation(env, filter_keys=keys)


def resolve_obs_mode(lidar=False, obs_mode=None):
    """--lidar is the old spelling of --obs-mode lidar; both together must agree."""
    if obs_mode is None:
        return 'lidar' if lidar else 'base'
    if obs_mode not in OBS_MODES:
        raise ValueError(f'obs_mode must be one of {OBS_MODES}, got {obs_mode!r}')
    if lidar and obs_mode != 'lidar':
        raise ValueError(f'--lidar conflicts with --obs-mode {obs_mode}')
    return obs_mode


def dqn_wrappers(lidar=False, action_repeat=1, obs_mode=None,
                 reward_mode='original', no_camera=False):
    obs_mode = resolve_obs_mode(lidar, obs_mode)

    if reward_mode not in ('original', 'forward-collision', 'signed-progress'):
        raise ValueError(f'Unknown reward mode: {reward_mode}')

    wrappers = []

    # Reward must be inside ActionRepeatWrapper.
    if reward_mode == 'signed-progress':
        wrappers.append(SignedProgressRewardWrapper)

    if action_repeat > 1:
        wrappers.append(
            partial(ActionRepeatWrapper, repeat=action_repeat)
        )

    wrappers.append(DiscreteActionWrapper)

    if obs_mode == 'lidar':
        wrappers.append(LidarObservationWrapper)
    elif obs_mode.startswith('track'):
        wrappers.append(
            partial(
                TrackFeatureWrapper,
                lookahead=obs_mode == 'track-lookahead'
            )
        )
    if no_camera:
        wrappers.append(RemoveCameraWrapper)
    wrappers.append(SB3ObservationWrapper)
    return wrappers


class SB3EvalDriver(Driver):

    """
     evaluate an SB3 model with the same loop as FTG.
    act takes the raw observation and returns motor and steering.

    Eval uses deterministic=True, so there's no random exploration.
    With action_repeat=N, we pick an action every N steps
    and keep using it until the next decision, just like in training.
    """

    def __init__(self, sb3_driver, action_repeat=1):
        self.sb3_driver = sb3_driver
        self.action_repeat = action_repeat
        self._steps = 0
        self._action = None

    def reset(self):
        self.sb3_driver.reset()
        self._steps = 0
        self._action = None

    def act(self, obs):
        if self._steps % self.action_repeat == 0:
            self._action = self.sb3_driver.act_raw(obs, deterministic=True)
        self._steps += 1
        return {key: value.copy() for key, value in self._action.items()}


class TrackPPODriver(_TrackViewDriver):
    """Train or run PPO with a full-track view and non-camera observations."""

    def __init__(self, env, *, track=None, map_size=256, crop_to_track=True, n_steps=256, **kwargs):
        track_wrapper = partial(TrackObservationWrapper, track=track, map_size=map_size, crop_to_track=crop_to_track)
        super().__init__(env, PPO, FlattenActionWrapper, track_wrapper, n_steps=n_steps, **kwargs)


class LocalTrackPPODriver(_TrackViewDriver):
    """Train or run PPO with a car-relative track view and non-camera observations."""

    def __init__(
        self, env, *, track=None, map_size=128, view_size_m=12.8,
        include_distance=False, n_steps=256, **kwargs,
    ):
        track_wrapper = partial(
            LocalTrackObservationWrapper, track=track, map_size=map_size, view_size_m=view_size_m,
            include_distance=include_distance,
        )
        super().__init__(env, PPO, FlattenActionWrapper, track_wrapper, n_steps=n_steps, **kwargs)


class TrackDQNDriver(_TrackViewDriver):
    """Train or run DQN with a full-track view and discrete controls."""

    # Current and next 2 x 256 x 256 float32 views use 1 GiB at this capacity.
    DEFAULT_BUFFER_SIZE = 1024

    def __init__(self, env, *, track=None, map_size=256, crop_to_track=True, buffer_size=DEFAULT_BUFFER_SIZE, **kwargs):
        track_wrapper = partial(TrackObservationWrapper, track=track, map_size=map_size, crop_to_track=crop_to_track)
        super().__init__(env, DQN, DiscreteActionWrapper, track_wrapper, buffer_size=buffer_size, **kwargs)


class LocalTrackDQNDriver(_TrackViewDriver):
    """Train or run DQN with a car-relative track view and discrete controls."""

    # Current and next 1 x 128 x 128 float32 views use 512 MiB at this capacity.
    DEFAULT_BUFFER_SIZE = 4096

    def __init__(
        self, env, *, track=None, map_size=128, view_size_m=12.8,
        include_distance=False, buffer_size=DEFAULT_BUFFER_SIZE, **kwargs,
    ):
        track_wrapper = partial(
            LocalTrackObservationWrapper, track=track, map_size=map_size, view_size_m=view_size_m,
            include_distance=include_distance,
        )
        super().__init__(env, DQN, DiscreteActionWrapper, track_wrapper, buffer_size=buffer_size, **kwargs)


class _DrivableTrackDriver(_TrackViewDriver):
    _features_extractor_class = DrivableTrackFeaturesExtractor

    def __init__(self, env, model_class, track_wrapper, default_buffer_size, **kwargs):
        if model_class is DQN:
            action_wrapper = DiscreteActionWrapper
            kwargs.setdefault("buffer_size", default_buffer_size)
        elif model_class is PPO:
            action_wrapper = FlattenActionWrapper
            kwargs.setdefault("n_steps", 256)
        else:
            raise ValueError("Drivable track drivers support DQN and PPO")
        super().__init__(env, model_class, action_wrapper, track_wrapper, **kwargs)


class DrivableTrackDriver(_DrivableTrackDriver):
    """Train or run DQN or PPO with a full-track drivable-area view."""

    def __init__(self, env, *, model_class=DQN, track=None, map_size=256, crop_to_track=True, **kwargs):
        wrapper = partial(DrivableTrackObservationWrapper, track=track, map_size=map_size, crop_to_track=crop_to_track)
        super().__init__(env, model_class, wrapper, TrackDQNDriver.DEFAULT_BUFFER_SIZE, **kwargs)


class LocalDrivableTrackDriver(_DrivableTrackDriver):
    """Train or run DQN or PPO with a car-relative drivable-area view."""

    def __init__(
        self, env, *, model_class=DQN, track=None, map_size=128, view_size_m=12.8,
        include_distance=False, **kwargs,
    ):
        wrapper = partial(
            LocalDrivableTrackObservationWrapper, track=track, map_size=map_size, view_size_m=view_size_m,
            include_distance=include_distance,
        )
        super().__init__(env, model_class, wrapper, LocalTrackDQNDriver.DEFAULT_BUFFER_SIZE, **kwargs)
