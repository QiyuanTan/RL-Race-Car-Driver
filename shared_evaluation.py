"""Shared episode evaluation for FTG and the project's SB3 drivers.

The caller supplies a compatible environment; this module does not change its
termination rules or sensor layout. Model wrappers must match training.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSpec:
    algorithm: str
    model: str = None
    obs_mode: str = 'base'
    action_repeat: int = 1
    geometry_only: bool = False

    def __post_init__(self):
        if self.algorithm not in ('ftg', 'dqn', 'ppo', 'sac'):
            raise ValueError('algorithm must be ftg, dqn, ppo, or sac')
        if self.action_repeat < 1:
            raise ValueError('action_repeat must be at least 1')
        if self.algorithm != 'ftg' and not self.model:
            raise ValueError('a model path is required for SB3 algorithms')
        if self.algorithm == 'ftg' and (self.model or self.obs_mode != 'base' or self.action_repeat != 1 or self.geometry_only):
            raise ValueError('FTG does not use model, observation-mode, repeat or geometry options')
        if self.algorithm in ('ppo', 'sac') and (self.obs_mode != 'base' or self.action_repeat != 1 or self.geometry_only):
            raise ValueError('PPO/SAC training uses base observations and action repeat 1')
        if self.geometry_only and (self.algorithm != 'dqn' or self.obs_mode != 'track-lookahead'):
            raise ValueError('geometry-only requires DQN track-lookahead observations')


def check_dqn_config(spec):
    """Check the saved DQN configuration when the training run recorded one."""
    import json
    from pathlib import Path
    from Agents.track_features import feature_config
    path = Path(spec.model)
    config_path = (path.parent.parent / 'config.json' if path.parent.name == 'eval'
                   else path.parent / 'checkpoints' / path.stem / 'config.json')
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    old_mode = 'lidar' if config.get('lidar', False) else 'base'
    trained = (config.get('obs_mode', old_mode), config.get('action_repeat', 1),
               config.get('features', feature_config(old_mode)))
    current = (spec.obs_mode, spec.action_repeat, feature_config(spec.obs_mode))
    if trained != current:
        raise ValueError('model training configuration does not match requested evaluation settings')


def build_driver(env, spec):
    """Adapt the selected implementation to reset()/act(raw observation)."""
    if spec.algorithm == 'ftg':
        from Agents.FollowTheGapDriver import FollowTheGapDriver
        return FollowTheGapDriver(env)

    from stable_baselines3 import DQN, PPO, SAC
    from Agents.SB3Drivers import SB3Driver, SB3EvalDriver, SB3ObservationWrapper, dqn_wrappers
    from Agents.wrappers import FlattenActionWrapper
    from gymnasium import spaces
    from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
    from stable_baselines3.common.vec_env import VecTransposeImage

    if spec.algorithm == 'dqn':
        if not spec.geometry_only:
            check_dqn_config(spec)
        wrappers = dqn_wrappers(obs_mode=spec.obs_mode, action_repeat=spec.action_repeat)
        if spec.geometry_only:
            from Agents.ftg_geometry import GeometryOnlyObservationWrapper
            wrappers.append(GeometryOnlyObservationWrapper)
    else:
        wrappers = [FlattenActionWrapper, SB3ObservationWrapper]
    model_class = {'dqn': DQN, 'ppo': PPO, 'sac': SAC}[spec.algorithm]
    kwargs = {'buffer_size': 1} if spec.algorithm in ('dqn', 'sac') else {}
    driver = SB3Driver(env, model_class=model_class, wrappers=wrappers, **kwargs)
    driver.load(spec.model)
    expected = dict(driver.model.observation_space.spaces)
    actual = dict(spaces.Dict({
        key: VecTransposeImage.transpose_space(space, key)
        if is_image_space(space) and not is_image_space_channels_first(space) else space
        for key, space in driver.env.observation_space.spaces.items()
    }).spaces)
    if expected != actual:
        raise ValueError(f'model observation space does not match selected wrappers: {spec.model}')
    if driver.model.action_space != driver.env.action_space:
        raise ValueError(f'model action space does not match selected wrappers: {spec.model}')
    return SB3EvalDriver(driver, action_repeat=spec.action_repeat)


def evaluate_episode(env, driver, seed):
    """Run any compatible driver through the existing shared metrics loop."""
    from evaluate_baseline import run_episode
    return run_episode(env, driver, seed)
