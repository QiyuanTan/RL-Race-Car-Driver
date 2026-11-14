from functools import partial
from pathlib import Path

from Agents.SB3Drivers import SB3Driver, SB3ObservationWrapper
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


if __name__ == '__main__':
    n_parallel_instances = 4
    scenario = str(Path(__file__).resolve().parent / 'scenarios' / 'austria.yml')
    # env_factories = [
    #     partial(gym_api.SingleAgentRaceEnv, scenario=scenario, render_mode='rgb_array_birds_eye')
    #     for _ in range(n_parallel_instances)
    # ]
    # train_ppo(env_factories, total_timesteps=1_000_000, save_path='models/ppo_model')

    env = gym_api.SingleAgentRaceEnv(scenario=scenario, render_mode='rgb_array_birds_eye')
    train_dqn(env, total_timesteps=1_000_000, save_path='models/dqn_model')
