"""Checks FTG teacher's raw LiDAR, hidden student observations and replay transitions."""
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

import Agents.ftg_track_demos as demo_module
from Agents.demos import find_wrapper
from Agents.wrappers import DiscreteActionWrapper
from Agents.wrappers2 import FTGTeacherLidarWrapper, SignedProgressRewardWrapper


class TinyRaceEnv(gym.Env):
    """Two-step race with a completion; the terminal obs differs from reset obs."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            'lidar': gym.spaces.Box(0.25, 15.25, (1080,), dtype=np.float64),
            'velocity': gym.spaces.Box(-10, 10, (6,), dtype=np.float64),
            'position': gym.spaces.Box(0, 10, (1,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Dict({
            'motor': gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
            'steering': gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
        })
        self.actions = []
        self.steps = 0

    def obs(self):
        return {
            'lidar': np.full(1080, 5.0 + self.steps, dtype=np.float64),
            'velocity': np.zeros(6, dtype=np.float64),
            'position': np.array([self.steps], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.obs(), {'progress': 0.98, 'lap': 1}

    def step(self, action):
        self.actions.append((float(action['motor'][0]), float(action['steering'][0])))
        self.steps += 1
        done = self.steps == 2
        info = {'lap': 2 if done else 1, 'progress': .02 if done else .99,
                'time': .01 * self.steps, 'wall_collision': False}
        return self.obs(), 999.0, done, False, info


class FakeFTG:
    def reset(self):
        self.scans = []

    def act(self, observation):
        self.scans.append(observation['lidar'].copy())
        return {'motor': np.array([.48]), 'steering': np.array([.9])}


class FakeBuffer:
    buffer_size = 10

    def __init__(self):
        self.items = []

    def add(self, *transition):
        self.items.append(transition)


def make_demo_env():
    base = TinyRaceEnv()
    env = FTGTeacherLidarWrapper(base)
    teacher = env
    env = SignedProgressRewardWrapper(env)
    env = DiscreteActionWrapper(env)
    discrete = find_wrapper(env, DiscreteActionWrapper)
    vec = DummyVecEnv([lambda: env])
    return base, env, vec, teacher, discrete


def test_hidden_lidar_and_terminal_observation():
    base, env, vec, teacher, discrete = make_demo_env()
    try:
        ftg = FakeFTG()
        transitions, info = demo_module.drive_track_demo_episode(vec, ftg, discrete, teacher, 1000)
        assert info['lap'] == 2
        assert len(transitions) == 2
        assert all('lidar' not in s for t in transitions for s in (t[0], t[1]))
        assert [scan[0] for scan in ftg.scans] == [5., 6.]
        assert base.actions == [(0.5, 1.0), (0.5, 1.0)]
        assert transitions[0][3][0] == pytest.approx(1.0)
        # Crossing the finish from .99 to .02 produces +.03 * 100.
        # The collector must store the signed reward unchanged.
        assert transitions[1][3][0] == pytest.approx(3.0)
        # DummyVecEnv has already reset: 0, but the stored terminal obs must be 2.
        assert transitions[-1][1]['position'][0, 0] == 2.
        assert teacher.latest_lidar.shape == (1080,)
    finally:
        vec.close()


def test_collector_prefills_without_lidar(monkeypatch):
    base, env, vec, teacher, discrete = make_demo_env()
    monkeypatch.setattr(demo_module, 'FollowTheGapDriver', FakeFTG)
    buffer = FakeBuffer()
    model = type('Model', (), {'env': vec, 'replay_buffer': buffer})()
    try:
        summary = demo_module.collect_ftg_track_demos(model, env, laps=1)
        assert summary['laps_completed'] == 1
        assert summary['decisions'] == 2
        assert summary['seeds'] == [1000]
        assert len(buffer.items) == 2
        assert 'lidar' not in buffer.items[0][0]
    finally:
        vec.close()
