from Agents.Driver import Driver
from Agents.wrappers import FlattenActionWrapper
from gymnasium import ObservationWrapper, spaces
import numpy as np
from functools import partial
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor

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
