import gymnasium

from Agents.SB3Drivers import SB3Driver
from stable_baselines3 import PPO
from racecar_gym.envs import gym_api

def train_ppo(env, total_timesteps=10000, save_path='ppo_model'):
    driver = SB3Driver(env, model_class=PPO, policy="MlpPolicy")
    driver.train(total_timesteps=total_timesteps)
    driver.save(save_path)


if __name__ == '__main__':
    # n_parallel_instances = 2
    # scenarios = ['scenarios/austria.yml' for _ in range(n_parallel_instances)]
    # env = gym_api.VectorizedSingleAgentRaceEnv(scenarios=scenarios)
    env = gymnasium.make('SingleAgentAustria-v0')
    train_ppo(env, total_timesteps=10000, save_path='ppo_model')
    env.close()
