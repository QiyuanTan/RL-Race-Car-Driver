from functools import partial

import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import FilterObservation
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from Agents.Driver import Driver
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

    def train(self, total_timesteps, progress_bar=True):
        self.model.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    def act(self, obs):
        action, _states = self.model.predict(obs)
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

    TRACK_KEY = TrackObservationWrapper.KEY

    def __init__(self, observation_space, **kwargs):
        track_key = self.TRACK_KEY
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
