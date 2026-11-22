"""Tests for the DQN + LiDAR pieces
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

"""Same observation/action layout as racecar_gym (minus the camera, to stay small)."""
class FakeRaceEnv(gym.Env):

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
    """assert out[0]means NaN, treated as a wall(clossest)
    out[1]means +inf, treated as farthest
    out[2]means -inf, treated as closest
    out[3]means too far, treated as clipped
    """
    assert out[0] == 0.0          
    assert out[1] == 1.0          
    assert out[2] == 0.0          
    assert out[3] == 1.0         


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
    #ID = motor_index*5 + steering_index
    #motor (-0.5, 0, 0.5), steering (-1, -0.5, 0, 0.5, 1)
    for action_id, (motor, steering) in [(0, (-0.5, -1.0)), (7, (0.0, 0.0)), (14, (0.5, 1.0))]:
        action = wrapper.action(action_id)
        assert action['motor'][0] == motor and action['steering'][0] == steering

"""SB3EvalDriver on the raw env must give the same action as model + wrappers in training."""
@pytest.mark.parametrize('lidar', [False, True])
def test_eval_driver_matches_training_wrappers(lidar):
    from stable_baselines3 import DQN
    env = FakeRaceEnv(with_lidar=lidar)
    driver = SB3Driver(env, model_class=DQN, wrappers=dqn_wrappers(lidar), buffer_size=100, seed=0)
    eval_driver = SB3EvalDriver(driver)
    rng = np.random.default_rng(0)
    for _ in range(5):
        raw = env.make_obs(rng.uniform(0.25, 15.0, N_RAYS) if lidar else None)
        raw['pose'] = rng.normal(size=6)
        wrapped = raw
        for wrapper_class in dqn_wrappers(lidar): 
             # rebuild the training preprocessing by hand
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

"""Find the wrapper instance of a given class inside the driver's wrapped env."""
def driver_wrapper(driver, wrapper_class):
    
    env = driver.env
    while isinstance(env, gym.Wrapper):
        if type(env) is wrapper_class:
            return env
        env = env.env
    raise LookupError(wrapper_class)

"""Fake env that records the actions it receives, gives reward 1 per step and ends after `length` steps."""
class CountingEnv(FakeRaceEnv):
    

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
    obs, reward, done, truncated, info = env.step('C')      
    # only 2 steps left in the episode
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
    assert motors == [1, 1, 1, 1, 2, 2, 2, 2, 3]            
    # new decision on steps 0, 4, 8
    driver.reset()
    assert driver.act({})['motor'][0] == 4            


def test_dqn_wrappers_put_action_repeat_innermost():
    from Agents.wrappers import ActionRepeatWrapper
    driver = SB3Driver(CountingEnv(), model_class=__import__('stable_baselines3').DQN,
                       wrappers=dqn_wrappers(False, action_repeat=4), buffer_size=100)
    chain = []
    env = driver.env
    while isinstance(env, gym.Wrapper):
        chain.append(type(env))
        env = env.env
    assert chain[-1] is ActionRepeatWrapper                  
    # innermost: repeats motor/steering
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
    env.reset(options={'mode': 'grid'})                      
     # an explicit mode still wins
    assert env.env.last_options == {'mode': 'grid'}


def test_demo_slots_are_never_overwritten():
    from Agents.demos import DemoKeepingDictReplayBuffer
    obs_space = gym.spaces.Dict({'x': gym.spaces.Box(-1e6, 1e6, (1,), np.float32)})
    buffer = DemoKeepingDictReplayBuffer(10, obs_space, gym.spaces.Discrete(15), device='cpu')

    def add(value):
        obs = {'x': np.array([[value]], np.float32)}
        buffer.add(obs, obs, np.array([0]), np.array([0.0]), np.array([False]), [{}])
    #3d transitions
    for demo in (-1, -2, -3):         
        add(demo)
    buffer.keep = 3
    #25 of DQN's own transitions into the 7 remaining slots
    for own in range(1, 26):           
        add(own)
    stored = buffer.observations['x'][:, 0, 0].tolist()
    assert stored[:3] == [-1, -2, -3]  
    assert sorted(stored[3:]) == list(range(19, 26)) 
    assert buffer.full and buffer.size() == 10


def test_nearest_action_id_matches_dqn_table():
    from Agents.demos import nearest_action_id
    actions = DiscreteActionWrapper(FakeRaceEnv())._actions
    command = {'motor': np.array([0.31]), 'steering': np.array([-0.8])}
    action_id = nearest_action_id(actions, command)
    assert actions[action_id]['motor'][0] == 0.5 and actions[action_id]['steering'][0] == -1.0
    assert action_id == 2 * 5 + 0  # ID = motor_index * 5 + steering_index


def test_demo_laps_keeps_only_full_laps(monkeypatch):
    """Tries that crash are discarded; collection stops once enough full laps are kept."""
    import Agents.demos as demos

    outcome = {1000: 1.0, 1001: 0.36, 1002: 0.54, 1003: 1.0, 1004: 1.0}   
    # seed -> progress reached

    def fake_drive(vec_env, ftg, discrete, lidar, seed):

        lap = outcome[seed] >= 1.0
        info = {'lap': 2 if lap else 1, 'progress': 0.0 if lap else outcome[seed], 'time': 1.0}
        return [('transition', seed)] * 3, info

    class FakeBuffer:
        def __init__(self):
            self.added = []

        def add(self, *transition):
            self.added.append(transition)

    class FakeModel:
        env = None
        replay_buffer = FakeBuffer()

    monkeypatch.setattr(demos, 'drive_demo_episode', fake_drive)
    monkeypatch.setattr(demos, 'find_wrapper', lambda env, cls: None)
    model = FakeModel()

    summary = demos.collect_ftg_demos(model, None, laps=2)
    assert summary['seeds'] == [1000, 1003] and summary['laps_completed'] == 2
    assert [d['seed'] for d in summary['discarded']] == [1001, 1002]
    assert {t[1] for t in model.replay_buffer.added} == {1000, 1003}          #dont storre failed tries
    assert summary['decisions'] == 6
    
    with pytest.raises(RuntimeError, match='only 1 of 2'):
        demos.collect_ftg_demos(FakeModel(), None, laps=2, max_attempts=2)    


class ProgressEnv(FakeRaceEnv):
    """Each 'random' reset lands at the next progress value from the list; 'grid' is progress 0.02."""

    def __init__(self, progresses):
        super().__init__(with_lidar=False)
        self.progresses, self.modes = list(progresses), []

    def reset(self, seed=None, options=None):
        mode = (options or {}).get('mode')
        self.modes.append(mode)
        progress = self.progresses.pop(0) if mode == 'random' else 0.02
        return self.make_obs(), {'progress': progress}


def test_hairpin_start_retries_until_inside_the_window():
    from Agents.wrappers import HairpinStartWrapper
    env = HairpinStartWrapper(ProgressEnv([0.6, 0.1, 0.31, 0.25]), share=1.0)
    _, info = env.reset(seed=0)
    assert info['progress'] == 0.25 and env.env.modes == ['random'] * 4 and env.hairpin_starts == 1


def test_hairpin_start_share_and_seed():
    from Agents.wrappers import HairpinStartWrapper
    always_grid = HairpinStartWrapper(ProgressEnv([]), share=0.0)
    assert all(always_grid.reset(seed=s)[1]['progress'] == 0.02 for s in range(5))
    picks = []
    for _ in range(2):            
        env = HairpinStartWrapper(ProgressEnv([0.25] * 100), share=0.5)
        env.reset(seed=3)
        picks.append([env.reset()[1]['progress'] for _ in range(40)])
    assert picks[0] == picks[1] and 0.02 in picks[0] and 0.25 in picks[0]


def test_hairpin_start_gives_up_with_a_clear_error():
    from Agents.wrappers import HairpinStartWrapper
    with pytest.raises(RuntimeError):
        HairpinStartWrapper(ProgressEnv([0.9] * 10), share=1.0, max_tries=10).reset(seed=0)


def test_init_from_must_match_lidar_and_action_repeat(tmp_path):
    import json
    from train import check_init_model
    folder = tmp_path / 'checkpoints' / 'run'
    (folder / 'eval').mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'lidar': True, 'action_repeat': 4}))
    model = folder / 'eval' / 'run_140000_steps.zip'
    model.write_bytes(b'')
    check_init_model(model, True, 4)
    with pytest.raises(ValueError):
        check_init_model(model, True, 2)
    with pytest.raises(FileNotFoundError):
        check_init_model(tmp_path / 'missing.zip', True, 4)


class EndingEnv(FakeRaceEnv):
    """Gives reward 1 per step; the 3rd step ends the episode, with or without a wall contact."""

    def __init__(self, wall):
        super().__init__(with_lidar=False)
        self.wall, self.t = wall, 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return self.make_obs(), {}

    def step(self, action):
        self.t += 1
        done = self.t == 3
        return self.make_obs(), 1.0, done, False, {'wall_collision': done and self.wall}


def test_crash_penalty_only_on_a_wall_ending():
    from Agents.wrappers import ActionRepeatWrapper, CrashPenaltyWrapper
    for wall, last_reward in ((True, 1.0 - 10.0), (False, 1.0)):
        env = CrashPenaltyWrapper(EndingEnv(wall), penalty=10.0)
        env.reset()
        rewards = [env.step(0)[1] for _ in range(3)]
        assert rewards == [1.0, 1.0, last_reward] and env.crashes == int(wall)
    # under action repeat 4 the penalty is still counted once: 3 steps x 1 - 10
    env = ActionRepeatWrapper(CrashPenaltyWrapper(EndingEnv(True), penalty=10.0), repeat=4)
    env.reset()
    assert env.step(0)[1] == pytest.approx(3.0 - 10.0)


def test_hairpin_start_respects_an_explicit_mode():
    from Agents.wrappers import HairpinStartWrapper
    env = HairpinStartWrapper(ProgressEnv([0.25] * 5), share=1.0)
    _, info = env.reset(seed=0, options={'mode': 'grid'})      
    assert info['progress'] == 0.02 and env.env.modes == ['grid'] and env.hairpin_starts == 0


def test_demo_mix_buffer_takes_a_fixed_share_from_the_demos():
    import torch as th
    from Agents.demos import DemoMixDictReplayBuffer
    from stable_baselines3.common.buffers import DictReplayBuffer
    obs_space = gym.spaces.Dict({'x': gym.spaces.Box(-10, 10, (1,), np.float32)})
    act_space = gym.spaces.Discrete(3)
    own = DemoMixDictReplayBuffer(100, obs_space, act_space, device='cpu')
    own.demo = DictReplayBuffer(20, obs_space, act_space, device='cpu')
    for buffer, value in ((own, 0.0), (own.demo, 1.0)):   
        for _ in range(20):
            o = {'x': np.full((1, 1), value, np.float32)}
            buffer.add(o, o, np.array([0]), np.array([0.0]), np.array([False]), [{}])
    own.demo_fraction = 0.25
    batch = own.sample(32)
    assert batch.observations['x'].shape[0] == 32
    assert int(th.sum(batch.observations['x'] == 1.0)) == 8      
    own.demo_fraction = 0.0
    assert int(th.sum(own.sample(32).observations['x'] == 1.0)) == 0


def margin_dqn_with_demos(action=3):
    """A small MarginDQN on the fake env, with 64 demo transitions that all use `action`."""
    from stable_baselines3.common.buffers import DictReplayBuffer
    from Agents.demos import DemoMixDictReplayBuffer
    from Agents.dqfd import MarginDQN
    driver = SB3Driver(FakeRaceEnv(with_lidar=False), model_class=MarginDQN, wrappers=dqn_wrappers(False),
                       buffer_size=200, seed=0, replay_buffer_class=DemoMixDictReplayBuffer)
    model = driver.model
    buffer = model.replay_buffer
    buffer.demo = DictReplayBuffer(64, buffer.observation_space, buffer.action_space, device='cpu')
    rng = np.random.default_rng(0)
    for _ in range(64):
        obs = {'pose': rng.normal(size=(1, 6)), 'velocity': rng.normal(size=(1, 6)), 'time': np.zeros((1, 1), np.float32)}
        buffer.demo.add(obs, obs, np.array([action]), np.array([0.1]), np.array([False]), [{}])
    return model


def test_margin_loss_is_zero_only_when_the_demo_action_wins_by_the_margin():
    import torch as th
    from stable_baselines3.common.type_aliases import DictReplayBufferSamples
    model = margin_dqn_with_demos()
    data = model.replay_buffer.demo.sample(8)
    with th.no_grad():                        
        last = model.q_net.q_net[-1]
        last.weight.zero_()
        last.bias.zero_()
        last.bias[3] = 1.0
    zero_target = DictReplayBufferSamples(data.observations, data.actions, data.next_observations,
                                          th.ones_like(data.dones), data.rewards)
    _, _, margin_loss = model._losses(zero_target, n_demo=8)
    assert float(margin_loss) == pytest.approx(0.0)
    with th.no_grad():
        last.bias[3] = 0.5                     # wins by only 0.5 < margin 0.8 -> loss 0.3
    _, _, margin_loss = model._losses(zero_target, n_demo=8)
    assert float(margin_loss) == pytest.approx(0.3, abs=1e-5)


def test_pretraining_on_demos_makes_the_greedy_action_the_demo_action():
    model = margin_dqn_with_demos(action=3)
    first, last = model.pretrain_on_demos(300, batch_size=32)
    assert last < first
    assert model.demo_agreement() >= 0.95
