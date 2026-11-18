"""Tests for the DQN + LiDAR pieces, using a small fake environment (no simulator needed).

Run with:  python -m pytest tests/
"""
import os
import sys

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agents.SB3Drivers import SB3Driver, SB3EvalDriver, dqn_wrappers  # noqa: E402
from Agents.wrappers import DiscreteActionWrapper, LidarObservationWrapper  # noqa: E402

N_RAYS = 1080


class FakeRaceEnv(gym.Env):
    """Same observation/action layout as racecar_gym (minus the camera, to stay small)."""

    def __init__(self, with_lidar=True):
        obs_spaces = {
            'pose': gym.spaces.Box(-100, 100, (6,), np.float64),
            'velocity': gym.spaces.Box(-14, 14, (6,), np.float64),
            'time': gym.spaces.Box(0, 1, (), np.float32),
        }
        if with_lidar:
            # identical to racecar_gym's lidar space
            obs_spaces['lidar'] = gym.spaces.Box(0.25, 15.25, (N_RAYS,), np.float64)
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Dict({
            'motor': gym.spaces.Box(-1, 1, (1,), np.float32),
            'steering': gym.spaces.Box(-1, 1, (1,), np.float32),
        })

    def make_obs(self, lidar=None):
        obs = {'pose': np.zeros(6), 'velocity': np.zeros(6), 'time': np.array(0.0, dtype=np.float32)}
        if 'lidar' in self.observation_space.spaces:
            obs['lidar'] = np.full(N_RAYS, 5.0) if lidar is None else np.asarray(lidar, dtype=np.float64)
        return obs

    def reset(self, seed=None, options=None):
        return self.make_obs(), {}

    def step(self, action):
        return self.make_obs(), 0.0, False, False, {}


def test_lidar_is_scaled_to_0_1_and_matches_the_space():
    wrapper = LidarObservationWrapper(FakeRaceEnv())
    scan = np.linspace(0.25, 15.25, N_RAYS)
    out = wrapper.observation(FakeRaceEnv().make_obs(scan))
    assert out['lidar'].dtype == np.float32 and out['lidar'].shape == (N_RAYS,)
    assert out['lidar'][0] == pytest.approx(0.0) and out['lidar'][-1] == pytest.approx(1.0)
    assert out['lidar'][N_RAYS // 2] == pytest.approx((scan[N_RAYS // 2] - 0.25) / 15.0, abs=1e-6)
    assert wrapper.observation_space.contains(out)


def test_lidar_bad_values_are_cleaned():
    wrapper = LidarObservationWrapper(FakeRaceEnv())
    scan = np.full(N_RAYS, 5.0)
    scan[:4] = [np.nan, np.inf, -np.inf, 100.0]
    out = wrapper.observation(FakeRaceEnv().make_obs(scan))['lidar']
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0          # NaN -> treated as a wall (closest)
    assert out[1] == 1.0          # +inf -> farthest
    assert out[2] == 0.0          # -inf -> closest
    assert out[3] == 1.0          # too far -> clipped


def test_other_observation_keys_are_unchanged():
    env = FakeRaceEnv()
    obs = env.make_obs()
    obs['pose'] = np.arange(6, dtype=np.float64)
    out = LidarObservationWrapper(env).observation(obs)
    assert np.array_equal(out['pose'], obs['pose'])
    assert set(out) == set(obs)


def test_missing_lidar_gives_a_clear_error():
    with pytest.raises(KeyError, match='austria_lidar.yml'):
        LidarObservationWrapper(FakeRaceEnv(with_lidar=False))


def test_discrete_action_table():
    wrapper = DiscreteActionWrapper(FakeRaceEnv())
    assert wrapper.action_space.n == 15
    # ID = motor_index * 5 + steering_index; motor (-0.5, 0, 0.5), steering (-1, -0.5, 0, 0.5, 1)
    for action_id, (motor, steering) in [(0, (-0.5, -1.0)), (7, (0.0, 0.0)), (14, (0.5, 1.0))]:
        action = wrapper.action(action_id)
        assert action['motor'][0] == motor and action['steering'][0] == steering


@pytest.mark.parametrize('lidar', [False, True])
def test_eval_driver_matches_training_wrappers(lidar):
    """SB3EvalDriver on the raw env must give the same action as model + wrappers in training."""
    from stable_baselines3 import DQN
    env = FakeRaceEnv(with_lidar=lidar)
    driver = SB3Driver(env, model_class=DQN, wrappers=dqn_wrappers(lidar), buffer_size=100, seed=0)
    eval_driver = SB3EvalDriver(driver)
    rng = np.random.default_rng(0)
    for _ in range(5):
        raw = env.make_obs(rng.uniform(0.25, 15.0, N_RAYS) if lidar else None)
        raw['pose'] = rng.normal(size=6)
        wrapped = raw
        for wrapper_class in dqn_wrappers(lidar):        # rebuild the training preprocessing by hand
            if issubclass(wrapper_class, gym.ObservationWrapper):
                wrapped = driver_wrapper(driver, wrapper_class).observation(wrapped)
        action_id, _ = driver.model.predict(wrapped, deterministic=True)
        expected = DiscreteActionWrapper(env).action(action_id)
        got = eval_driver.act(raw)
        assert set(got) == {'motor', 'steering'}
        assert got['motor'][0] == expected['motor'][0] and got['steering'][0] == expected['steering'][0]
    if lidar:
        assert driver.model.observation_space['lidar'].shape == (N_RAYS,)
    else:
        assert 'lidar' not in driver.model.observation_space.spaces


def driver_wrapper(driver, wrapper_class):
    """Find the wrapper instance of a given class inside the driver's wrapped env."""
    env = driver.env
    while isinstance(env, gym.Wrapper):
        if type(env) is wrapper_class:
            return env
        env = env.env
    raise LookupError(wrapper_class)


class CountingEnv(FakeRaceEnv):
    """Fake env that records the actions it receives, gives reward 1 per step and ends after `length` steps."""

    def __init__(self, length=10):
        super().__init__(with_lidar=False)
        self.length = length

    def reset(self, seed=None, options=None):
        self.received = []
        return self.make_obs(), {}

    def step(self, action):
        self.received.append(action)
        done = len(self.received) >= self.length
        return self.make_obs(), 1.0, done, False, {'step': len(self.received)}


def test_action_repeat_holds_action_and_adds_rewards():
    from Agents.wrappers import ActionRepeatWrapper
    env = ActionRepeatWrapper(CountingEnv(length=10), repeat=4)
    env.reset()
    obs, reward, done, truncated, info = env.step('A')
    assert env.env.received == ['A'] * 4 and reward == 4.0 and not done
    env.step('B')
    obs, reward, done, truncated, info = env.step('C')      # only 2 steps left in the episode
    assert env.env.received[-2:] == ['C', 'C'] and reward == 2.0 and done and info['step'] == 10


def test_eval_driver_decides_every_n_steps():
    class FakeSB3Driver:
        calls = 0

        def reset(self):
            pass

        def act_raw(self, obs, deterministic=True):
            FakeSB3Driver.calls += 1
            return {'motor': np.array([FakeSB3Driver.calls], np.float32),
                    'steering': np.array([0.0], np.float32)}

    driver = SB3EvalDriver(FakeSB3Driver(), action_repeat=4)
    driver.reset()
    motors = [driver.act({})['motor'][0] for _ in range(9)]
    assert motors == [1, 1, 1, 1, 2, 2, 2, 2, 3]            # new decision on steps 0, 4, 8
    driver.reset()
    assert driver.act({})['motor'][0] == 4                  # reset starts a fresh decision


def test_dqn_wrappers_put_action_repeat_innermost():
    from Agents.wrappers import ActionRepeatWrapper
    driver = SB3Driver(CountingEnv(), model_class=__import__('stable_baselines3').DQN,
                       wrappers=dqn_wrappers(False, action_repeat=4), buffer_size=100)
    chain = []
    env = driver.env
    while isinstance(env, gym.Wrapper):
        chain.append(type(env))
        env = env.env
    assert chain[-1] is ActionRepeatWrapper                  # innermost: repeats motor/steering
    assert DiscreteActionWrapper in chain


def test_start_mode_wrapper_injects_reset_mode():
    from Agents.wrappers import StartModeWrapper

    class RecordingEnv(FakeRaceEnv):
        def reset(self, seed=None, options=None):
            self.last_options = options
            return super().reset(seed=seed, options=options)

    env = StartModeWrapper(RecordingEnv(), mode='random')
    env.reset()
    assert env.env.last_options == {'mode': 'random'}
    env.reset(options={'mode': 'grid'})                       # an explicit mode still wins
    assert env.env.last_options == {'mode': 'grid'}


def test_demo_slots_are_never_overwritten():
    from Agents.demos import DemoKeepingDictReplayBuffer
    obs_space = gym.spaces.Dict({'x': gym.spaces.Box(-1e6, 1e6, (1,), np.float32)})
    buffer = DemoKeepingDictReplayBuffer(10, obs_space, gym.spaces.Discrete(15), device='cpu')

    def add(value):
        obs = {'x': np.array([[value]], np.float32)}
        buffer.add(obs, obs, np.array([0]), np.array([0.0]), np.array([False]), [{}])

    for demo in (-1, -2, -3):          # three demonstration transitions
        add(demo)
    buffer.keep = 3
    for own in range(1, 26):           # 25 of DQN's own transitions into the 7 remaining slots
        add(own)
    stored = buffer.observations['x'][:, 0, 0].tolist()
    assert stored[:3] == [-1, -2, -3]                        # demos still there
    assert sorted(stored[3:]) == list(range(19, 26))         # the 7 newest own transitions
    assert buffer.full and buffer.size() == 10


def test_nearest_action_id_matches_dqn_table():
    from Agents.demos import nearest_action_id
    actions = DiscreteActionWrapper(FakeRaceEnv())._actions
    command = {'motor': np.array([0.31]), 'steering': np.array([-0.8])}
    action_id = nearest_action_id(actions, command)
    assert actions[action_id]['motor'][0] == 0.5 and actions[action_id]['steering'][0] == -1.0
    assert action_id == 2 * 5 + 0                            # ID = motor_index * 5 + steering_index
