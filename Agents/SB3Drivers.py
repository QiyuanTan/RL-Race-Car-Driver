from Agents.Driver import Driver
from Agents.wrappers import (
    FlattenActionWrapper,
    DiscreteActionWrapper,
    LidarObservationWrapper,
    ActionRepeatWrapper,
    RemoveCameraWrapper
)
from Agents.track_features import OBS_MODES, TrackFeatureWrapper
import gymnasium as gym
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
"""Convert scalar Box entries in Dict observations to one-element vectors."""
class SB3ObservationWrapper(ObservationWrapper):
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

    
def resolve_obs_mode(lidar=False, obs_mode=None):
    """--lidar is the old spelling of --obs-mode lidar; both together must agree."""
    if obs_mode is None:
        return 'lidar' if lidar else 'base'
    if obs_mode not in OBS_MODES:
        raise ValueError(f'obs_mode must be one of {OBS_MODES}, got {obs_mode!r}')
    if lidar and obs_mode != 'lidar':
        raise ValueError(f'--lidar conflicts with --obs-mode {obs_mode}')
    return obs_mode


def dqn_wrappers(lidar=False, action_repeat=1, obs_mode=None, no_camera=False):
    """
    Wrappers used by both training and eval.

    base: discrete actions and the SB3 observation fix
    lidar: also cleans and scales the lidar readings, needs a lidar sensor
    track: adds 5 track-relative values
    track-lookahead: also adds 5 points ahead, 10 values total for lookahead

    All modes keep the camera, pose, velocity, acceleration and time.
    The old lidar flag still works.

    Action repeat goes on the inside so it repeats the decoded
    motor and steering for N simulator steps.
    """
    obs_mode = resolve_obs_mode(lidar, obs_mode)
    wrappers = [partial(ActionRepeatWrapper, repeat=action_repeat)] if action_repeat > 1 else []
    wrappers.append(DiscreteActionWrapper)
    if obs_mode == 'lidar':
        wrappers.append(LidarObservationWrapper)
    elif obs_mode.startswith('track'):
        wrappers.append(partial(TrackFeatureWrapper, lookahead=obs_mode == 'track-lookahead'))
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
