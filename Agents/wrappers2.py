import gymnasium as gym
import numpy as np

class SignedProgressRewardWrapper(gym.Wrapper):
    """reward only forward progress"""

    def __init__(self, env, progress_scale=100.0):
        super().__init__(env)
        self.progress_scale = progress_scale
        self.last_progress = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_progress = float(info["progress"])
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        progress = float(info["progress"])
        delta = progress - self.last_progress

        # handle crossing the start/finish line
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0

        reward = delta * self.progress_scale
        self.last_progress = progress
        return obs, reward, terminated, truncated, info



class FTGTeacherLidarWrapper(gym.ObservationWrapper):
    """keep LiDAR for the FTG teacher but hide it from the DQN student"""

    def __init__(self, env):
        super().__init__(env)

        spaces = dict(env.observation_space.spaces)
        if "lidar" not in spaces:
            raise ValueError("FTG teacher requires a LiDAR sensor")

        spaces.pop("lidar")
        self.observation_space = gym.spaces.Dict(spaces)
        self.latest_lidar = None

    def observation(self, obs):
        # save the raw LiDAR scan for FTG
        self.latest_lidar = np.asarray(
            obs["lidar"], dtype=np.float64
        ).copy()

        # DQN will not receive LiDAR
        result = dict(obs)
        result.pop("lidar")
        return result