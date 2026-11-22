"""Tests for the shared evaluation contract and SB3 action adapters."""
import numpy as np
import pytest

from shared_evaluation import EvaluationSpec, build_driver
from test_dqn_lidar import FakeRaceEnv


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match='model path'):
        EvaluationSpec('dqn')
    with pytest.raises(ValueError, match='at least 1'):
        EvaluationSpec('dqn', 'x.zip', action_repeat=0)
    with pytest.raises(ValueError, match='PPO/SAC'):
        EvaluationSpec('ppo', 'x.zip', obs_mode='lidar')
    with pytest.raises(ValueError, match='geometry-only'):
        EvaluationSpec('dqn', 'x.zip', geometry_only=True)


@pytest.mark.parametrize('algorithm', ['dqn', 'ppo', 'sac'])
def test_saved_model_produces_raw_actions(algorithm, tmp_path):
    from stable_baselines3 import DQN, PPO, SAC
    from Agents.SB3Drivers import SB3Driver, SB3ObservationWrapper, dqn_wrappers
    from Agents.wrappers import FlattenActionWrapper
    env = FakeRaceEnv(with_lidar=False)
    cls = {'dqn': DQN, 'ppo': PPO, 'sac': SAC}[algorithm]
    wrappers = dqn_wrappers() if algorithm == 'dqn' else [FlattenActionWrapper, SB3ObservationWrapper]
    kwargs = {'buffer_size': 1} if algorithm in ('dqn', 'sac') else {}
    trained = SB3Driver(env, cls, wrappers=wrappers, seed=0, **kwargs)
    path = str(tmp_path / 'model.zip')
    trained.save(path)
    expected = trained.act_raw(env.make_obs(), deterministic=True)
    adapter = build_driver(env, EvaluationSpec(algorithm, path))
    adapter.reset()
    actual = adapter.act(env.make_obs())
    assert set(actual) == {'motor', 'steering'}
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key])
        assert env.action_space[key].contains(actual[key])


def test_model_space_mismatch_is_rejected(tmp_path):
    from stable_baselines3 import DQN
    from Agents.SB3Drivers import SB3Driver, dqn_wrappers
    env = FakeRaceEnv(with_lidar=False)
    driver = SB3Driver(env, DQN, wrappers=dqn_wrappers(), buffer_size=1)
    path = str(tmp_path / 'model.zip')
    driver.save(path)
    with pytest.raises(ValueError, match='observation space'):
        build_driver(FakeRaceEnv(with_lidar=True), EvaluationSpec('dqn', path, obs_mode='lidar'))

@pytest.mark.parametrize('mode,geometry', [('lidar', False), ('track', False), ('track-lookahead', False), ('track-lookahead', True)])
def test_dqn_feature_modes_match_training(mode, geometry, tmp_path):
    from stable_baselines3 import DQN
    from Agents.SB3Drivers import SB3Driver, dqn_wrappers
    from Agents.ftg_geometry import GeometryOnlyObservationWrapper
    env = FakeRaceEnv(with_lidar=(mode == 'lidar'))
    wrappers = dqn_wrappers(obs_mode=mode, action_repeat=2)
    if geometry:
        wrappers.append(GeometryOnlyObservationWrapper)
    original = SB3Driver(env, DQN, wrappers=wrappers, buffer_size=1, seed=0)
    path = str(tmp_path / 'model.zip')
    original.save(path)
    expected = original.act_raw(env.make_obs(), deterministic=True)
    adapter = build_driver(env, EvaluationSpec('dqn', path, mode, 2, geometry))
    adapter.reset()
    first = adapter.act(env.make_obs())
    second = adapter.act(env.make_obs())
    for key in expected:
        np.testing.assert_allclose(first[key], expected[key])
        np.testing.assert_allclose(second[key], first[key])


def test_recorded_dqn_settings_must_match(tmp_path):
    import json
    from shared_evaluation import check_dqn_config
    folder = tmp_path / 'checkpoints' / 'model'
    folder.mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'lidar': False, 'action_repeat': 4}))
    check_dqn_config(EvaluationSpec('dqn', str(tmp_path / 'model.zip'), action_repeat=4))
    with pytest.raises(ValueError, match='training configuration'):
        check_dqn_config(EvaluationSpec('dqn', str(tmp_path / 'model.zip'), action_repeat=1))


def test_ftg_uses_same_raw_action_contract():
    env = FakeRaceEnv(with_lidar=True)
    driver = build_driver(env, EvaluationSpec('ftg'))
    driver.reset()
    action = driver.act(env.make_obs())
    assert set(action) == {'motor', 'steering'}
    for key in action:
        assert env.action_space[key].contains(action[key])
