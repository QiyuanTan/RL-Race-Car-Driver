from stable_baselines3 import PPO
from pathlib import Path

from Agents.Driver import *
from racecar_gym.envs import gym_api
import gymnasium as gym
from time import sleep

from Agents.SB3Drivers import SB3Driver, SB3ObservationWrapper
from Agents.wrappers import FlattenActionWrapper

scenario = str(Path(__file__).resolve().parent / 'scenarios' / 'austria.yml')
env = gym.make('SingleAgentRaceEnv-v0', scenario=scenario, render_mode='human')

wrappers = [FlattenActionWrapper, SB3ObservationWrapper]
driver = SB3Driver(env, PPO, wrappers)
env = driver.env
driver.load("models/ppo_model_w_collision.zip")

obs, info = env.reset(options=dict(mode='grid'))
done = False
truncated = False
t = 0
while not (done or truncated):
    action = driver.act(obs)
    obs, rewards, done, truncated, states = env.step(action)
    sleep(0.01)
    if t % 30 == 0:
        image = env.render()
    t+=1


env.close()
