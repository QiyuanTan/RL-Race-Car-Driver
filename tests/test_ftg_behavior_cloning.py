"""Fast unit tests for demonstration-only sampling and BC data validation."""
import numpy as np
import pytest
import gymnasium as gym

from Agents.ftg_behavior_cloning import check_demo_buffer, demo_split, project_demo_buffer
from Agents.ftg_geometry import GeometryOnlyObservationWrapper


class FakeBuffer:
    def __init__(self, keep=2103):
        self.keep = keep
        self.buffer_size = 10000
        self.observation_space = gym.spaces.Dict({
            'lookahead': gym.spaces.Box(-3, 3, (10,), dtype=np.float32),
            'track': gym.spaces.Box(-3, 3, (5,), dtype=np.float32),
            'rgb_camera': gym.spaces.Box(0, 255, (3, 4, 4), dtype=np.uint8),
        })
        self.actions = np.zeros((keep, 1, 1), dtype=np.int64)
        self.observations = {k: np.zeros((keep, 1) + v.shape, dtype=v.dtype)
                             for k, v in self.observation_space.spaces.items()}
        self.next_observations = {k: a.copy() for k, a in self.observations.items()}

    def size(self):
        return self.keep


class FakeModel:
    def __init__(self, keep=2103):
        self.replay_buffer = FakeBuffer(keep)
        self.observation_space = self.replay_buffer.observation_space
        self.action_space = gym.spaces.Discrete(15)


def test_split_has_no_overlap_and_is_reproducible():
    train, val = demo_split(2103, seed=0)
    assert len(train) + len(val) == 2103
    assert not set(train).intersection(val)
    assert sorted(np.concatenate([train, val]).tolist()) == list(range(2103))
    train2, val2 = demo_split(2103, seed=0)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(val, val2)


def test_rejects_missing_demos():
    with pytest.raises(ValueError, match='demo prefix'):
        check_demo_buffer(FakeModel(0))


def test_valid_demo_prefix():
    assert check_demo_buffer(FakeModel()) == 2103


def test_rejects_invalid_action_id():
    model = FakeModel()
    model.replay_buffer.actions[4, 0, 0] = 16
    with pytest.raises(ValueError, match='invalid discrete'):
        check_demo_buffer(model)


def test_rejects_lidar_leakage():
    model = FakeModel()
    spaces = dict(model.observation_space.spaces)
    spaces['lidar'] = gym.spaces.Box(0, 1, (1080,), dtype=np.float32)
    model.observation_space = gym.spaces.Dict(spaces)
    model.replay_buffer.observation_space = model.observation_space
    with pytest.raises(ValueError, match='must not receive LiDAR'):
        check_demo_buffer(model)


def test_project_buffer_discards_images_but_preserves_actions():
    model = FakeModel()
    target = gym.spaces.Dict({
        k: model.observation_space.spaces[k] for k in ('track', 'lookahead')
    })
    model.observation_space = target
    model.replay_buffer.actions[5, 0, 0] = 7
    project_demo_buffer(model.replay_buffer, target)
    assert set(model.replay_buffer.observations) == {'track', 'lookahead'}
    assert set(model.replay_buffer.next_observations) == {'track', 'lookahead'}
    assert model.replay_buffer.actions[5, 0, 0] == 7
    assert check_demo_buffer(model) == 2103


def test_geometry_wrapper_hides_camera_and_other_fields():
    class TinyEnv(gym.Env):
        def __init__(self):
            self.observation_space = FakeBuffer(1).observation_space
            self.action_space = gym.spaces.Discrete(15)
        def reset(self, *, seed=None, options=None):
            return {k: v.sample() for k, v in self.observation_space.spaces.items()}, {}

    env = GeometryOnlyObservationWrapper(TinyEnv())
    obs, _ = env.reset()
    assert set(obs) == {'track', 'lookahead'}
    assert set(env.observation_space.spaces) == {'track', 'lookahead'}
