from functools import partial

import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import FilterObservation
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from Agents.Driver import Driver
from Agents.wrappers import FlattenActionWrapper, TrackObservationWrapper


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


class TrackPPODriver(SB3Driver):
    """Train or run PPO with a full-track view and non-camera observations."""

    def __init__(self, env, *, track=None, map_size=256, n_steps=256, policy_kwargs=None, **kwargs):
        wrappers = [
            FlattenActionWrapper,
            SB3ObservationWrapper,
            partial(TrackObservationWrapper, track=track, map_size=map_size),
            self._exclude_rgb_camera,
        ]
        policy_kwargs = dict(policy_kwargs or {})
        policy_kwargs.setdefault("features_extractor_class", TrackFeaturesExtractor)
        super().__init__(env, PPO, wrappers=wrappers, n_steps=n_steps, policy_kwargs=policy_kwargs, **kwargs)

    @staticmethod
    def _exclude_rgb_camera(env):
        keys = [key for key in env.observation_space.spaces if key != "rgb_camera"]
        return FilterObservation(env, filter_keys=keys)
