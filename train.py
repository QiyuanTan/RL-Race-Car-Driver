import argparse
from functools import partial
from pathlib import Path

from Agents.SB3Drivers import SB3Driver, SB3ObservationWrapper, dqn_wrappers
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


REPO_ROOT = Path(__file__).resolve().parent


def dqn_scenario(lidar=False):
    """Scenario file for DQN. The LiDAR variant only adds the `lidar` sensor."""
    name = 'austria_lidar.yml' if lidar else 'austria.yml'
    return str(REPO_ROOT / 'scenarios' / name)


def train_dqn(env, total_timesteps=10000, save_path='dqn_model', buffer_size=10_000,
              lidar=False, seed=None):
    """Train DQN with a bounded replay buffer, independent of training duration.

    At 128x128 RGB uint8, 10,000 transitions use about 0.92 GiB for current
    and next images alone; other observations and training need extra memory.

    lidar=True adds the LiDAR scan to the observation (env must come from
    dqn_scenario(lidar=True)); everything else (actions, reward, budget) is unchanged.
    """
    log_dir = REPO_ROOT / 'logs' / ('dqn_lidar' if lidar else 'dqn')
    driver = SB3Driver(
        env, model_class=DQN, wrappers=dqn_wrappers(lidar), verbose=1,
        tensorboard_log=str(log_dir), buffer_size=buffer_size, seed=seed,
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


if __name__ == '__main__':
    # With no arguments this trains base DQN for 1M steps -> models/dqn_model.zip, as before.
    parser = argparse.ArgumentParser(description='Train DQN (optionally with LiDAR observations).')
    parser.add_argument('--lidar', action='store_true', help='add the LiDAR scan to the observation')
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save-path', default=None,
                        help='default: models/dqn_model, or models/dqn_lidar_model with --lidar')
    parser.add_argument('--buffer-size', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=None, help='SB3 seed (use the same one for both variants)')
    args = parser.parse_args()

    save_path = args.save_path or ('models/dqn_lidar_model' if args.lidar else 'models/dqn_model')

    # n_parallel_instances = 4
    # scenario = str(Path(__file__).resolve().parent / 'scenarios' / 'austria.yml')
    # env_factories = [
    #     partial(gym_api.SingleAgentRaceEnv, scenario=scenario, render_mode='rgb_array_birds_eye')
    #     for _ in range(n_parallel_instances)
    # ]
    # train_ppo(env_factories, total_timesteps=1_000_000, save_path='models/ppo_model')

    env = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(args.lidar), render_mode='rgb_array_birds_eye')
    train_dqn(env, total_timesteps=args.timesteps, save_path=save_path,
              buffer_size=args.buffer_size, lidar=args.lidar, seed=args.seed)
