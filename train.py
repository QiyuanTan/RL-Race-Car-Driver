from functools import partial
from pathlib import Path

from Agents.SB3Drivers import (
    DrivableTrackDriver,
    LocalDrivableTrackDriver,
    LocalTrackDQNDriver,
    LocalTrackPPODriver,
    SB3Driver,
    SB3ObservationWrapper,
    TrackDQNDriver,
    TrackPPODriver,
)
from Agents.wrappers import FlattenActionWrapper, DiscreteActionWrapper
from stable_baselines3 import PPO, DQN, SAC
from racecar_gym.envs import gym_api

def train_ppo(env, total_timesteps=10000, save_path='ppo_model'):
    log_dir = Path(__file__).resolve().parent / 'logs' / 'ppo'
    wrappers = [FlattenActionWrapper, SB3ObservationWrapper]
    driver = SB3Driver(env, model_class=PPO, wrappers=wrappers, verbose=1, tensorboard_log=str(log_dir))
    try:
        driver.train(total_timesteps=total_timesteps)
        driver.save(save_path)
    finally:
        driver.env.close()


def train_track_ppo(env, total_timesteps=10000, save_path='track_ppo_model', *, log_dir=None):
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / 'logs' / 'track_ppo'
    _train_track_driver(TrackPPODriver, env, total_timesteps, save_path, log_dir)


def train_local_track_ppo(env, total_timesteps=10000, save_path='local_track_ppo_model', *, log_dir=None):
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / 'logs' / 'local_track_ppo'
    _train_track_driver(LocalTrackPPODriver, env, total_timesteps, save_path, log_dir)


def train_track_dqn(
    env, total_timesteps=10000, save_path='track_dqn_model', *,
    buffer_size=TrackDQNDriver.DEFAULT_BUFFER_SIZE, log_dir=None,
):
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / 'logs' / 'track_dqn'
    _train_track_driver(TrackDQNDriver, env, total_timesteps, save_path, log_dir, buffer_size=buffer_size)


def train_local_track_dqn(
    env, total_timesteps=10000, save_path='local_track_dqn_model', *,
    buffer_size=LocalTrackDQNDriver.DEFAULT_BUFFER_SIZE, log_dir=None,
):
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / 'logs' / 'local_track_dqn'
    _train_track_driver(LocalTrackDQNDriver, env, total_timesteps, save_path, log_dir, buffer_size=buffer_size)


def train_drivable_track(
    env, total_timesteps=10_000, save_path=None, *, model_class=DQN, log_dir=None, **driver_kwargs,
):
    """Train DQN or PPO on the full-track drivable-area view."""
    _train_drivable_track_driver(
        DrivableTrackDriver, 'drivable_track', env, total_timesteps, save_path,
        model_class=model_class, log_dir=log_dir, **driver_kwargs,
    )


def train_local_drivable_track(
    env, total_timesteps=10_000, save_path=None, *, model_class=DQN, log_dir=None, **driver_kwargs,
):
    """Train DQN or PPO on the car-relative drivable-area view."""
    _train_drivable_track_driver(
        LocalDrivableTrackDriver, 'local_drivable_track', env, total_timesteps, save_path,
        model_class=model_class, log_dir=log_dir, **driver_kwargs,
    )


def _train_drivable_track_driver(
    driver_class, view_name, env, total_timesteps, save_path, *, model_class, log_dir, **driver_kwargs,
):
    run_name = f'{view_name}_{model_class.__name__.lower()}'
    if save_path is None:
        save_path = f'{run_name}_model'
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / 'logs' / run_name
    _train_track_driver(
        driver_class, env, total_timesteps, save_path, log_dir, model_class=model_class, **driver_kwargs,
    )


def _train_track_driver(driver_class, env, total_timesteps, save_path, log_dir, **driver_kwargs):
    driver = driver_class(env, verbose=1, tensorboard_log=str(log_dir), **driver_kwargs)
    try:
        driver.train(total_timesteps=total_timesteps)
        driver.save(save_path)
    finally:
        driver.env.close()


def train_dqn(env, total_timesteps=10000, save_path='dqn_model', buffer_size=10_000):
    """Train DQN with a bounded replay buffer, independent of training duration.

    At 128x128 RGB uint8, 10,000 transitions use about 0.92 GiB for current
    and next images alone; other observations and training need extra memory.
    """
    log_dir = Path(__file__).resolve().parent / 'logs' / 'dqn'
    wrappers = [DiscreteActionWrapper, SB3ObservationWrapper]
    driver = SB3Driver(
        env, model_class=DQN, wrappers=wrappers, verbose=1,
        tensorboard_log=str(log_dir), buffer_size=buffer_size,
    )
    try:
        driver.train(total_timesteps=total_timesteps)
        driver.save(save_path)
    finally:
        driver.env.close()


def train_sac(env, total_timesteps=10000, save_path='sac_model'):
    log_dir = Path(__file__).resolve().parent / 'logs' / 'sac'
    wrappers = [FlattenActionWrapper, SB3ObservationWrapper]
    driver = SB3Driver(env, model_class=SAC, wrappers=wrappers, verbose=1, tensorboard_log=str(log_dir))
    try:
        driver.train(total_timesteps=total_timesteps)
        driver.save(save_path)
    finally:
        driver.env.close()


# if __name__ == '__main__':
#     project_dir = Path(__file__).resolve().parent
#     n_envs = 4
#     runs = (
#         ('allow_collisions', 'austria_allow_collisions.yml'),
#         ('terminate_on_collision', 'austria.yml'),
#     )
#     for run_name, scenario_file in runs:
#         scenario = str(project_dir / 'scenarios' / scenario_file)
#         env_factories = [
#             partial(gym_api.SingleAgentRaceEnv, scenario=scenario, render_mode='rgb_array_birds_eye')
#             for _ in range(n_envs)
#         ]
#         print(f'Training Track PPO ({run_name})')
#         train_track_ppo(
#             env_factories,
#             total_timesteps=1_000_000,
#             save_path=str(project_dir / 'models' / f'track_ppo_{run_name}'),
#             log_dir=project_dir / 'logs' / 'track_ppo' / run_name,
#         )


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    scenario = str(project_dir / 'scenarios' / 'austria_no_camera.yml')
    make_env = partial(gym_api.SingleAgentRaceEnv, scenario=scenario, render_mode='rgb_array_birds_eye')
    ppo_n_envs = 6
    total_timesteps = 2_000_000
    runs = (
        ('drivable_track_ppo', train_drivable_track, PPO),
        ('local_drivable_track_ppo', train_local_drivable_track, PPO),
        ('drivable_track_dqn', train_drivable_track, DQN),
        ('local_drivable_track_dqn', train_local_drivable_track, DQN),
    )
    for run_name, train, model_class in runs:
        print(f'Training {run_name} (no_collisions)', flush=True)
        env = [make_env] * ppo_n_envs if model_class is PPO else make_env()
        train(
            env,
            total_timesteps=total_timesteps,
            save_path=str(project_dir / 'models' / f'{run_name}_no_collisions'),
            log_dir=project_dir / 'logs' / run_name / 'no_collisions',
            model_class=model_class,
        )
