from functools import partial
from pathlib import Path

from Agents.SB3Drivers import SB3Driver
from stable_baselines3 import PPO
from racecar_gym.envs import gym_api

def train_ppo(env, total_timesteps=10000, save_path='ppo_model'):
    driver = SB3Driver(env, model_class=PPO)
    try:
        driver.train(total_timesteps=total_timesteps)
        driver.save(save_path)
    finally:
        driver.env.close()


if __name__ == '__main__':
    n_parallel_instances = 8
    scenario = str(Path(__file__).resolve().parent / 'scenarios' / 'austria.yml')
    env_factories = [
        partial(gym_api.SingleAgentRaceEnv, scenario=scenario, render_mode='rgb_array_birds_eye')
        for _ in range(n_parallel_instances)
    ]
    train_ppo(env_factories, total_timesteps=30000, save_path='models/ppo_model')
