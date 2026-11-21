
import gymnasium as gym
import numpy as np
import pytest

from Agents.wrappers2 import SignedProgressRewardWrapper


class FakeProgressEnv(gym.Env):
    """environment for testing progress rewards"""

    def __init__(self, progress_values):
        super().__init__()
        self.progress_values = progress_values
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = np.array([0.0], dtype=np.float32)
        return obs, {"progress": self.progress_values[0]}

    def step(self, action):
        self.step_count += 1
        obs = np.array([0.0], dtype=np.float32)
        info = {"progress": self.progress_values[self.step_count]}
        return obs, 0.0, False, False, info


def test_forward_progress():
    env = SignedProgressRewardWrapper(
        FakeProgressEnv([0.10, 0.11])
    )
    env.reset()

    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(1.0)


def test_backward_progress():
    env = SignedProgressRewardWrapper(
        FakeProgressEnv([0.11, 0.10])
    )
    env.reset()

    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(-1.0)


def test_cross_finish_line():
    env = SignedProgressRewardWrapper(
        FakeProgressEnv([0.99, 0.01])
    )
    env.reset()

    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(2.0)


def test_backward_across_finish_line():
    env = SignedProgressRewardWrapper(
        FakeProgressEnv([0.01, 0.99])
    )
    env.reset()

    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(-2.0)


def test_reset_clears_progress_history():
    env = SignedProgressRewardWrapper(
        FakeProgressEnv([0.20, 0.25])
    )

    env.reset()
    _, reward1, _, _, _ = env.step(0)

    env.reset()
    _, reward2, _, _, _ = env.step(0)

    assert reward1 == pytest.approx(5.0)
    assert reward2 == pytest.approx(5.0)