from Agents.Driver import *
from racecar_gym.envs import gym_api
import gymnasium as gym
from time import sleep

env = gym.make('SingleAgentAustria-v0', render_mode='human')

driver = HitTheGasDriver(env)

obs = env.reset(options=dict(mode='grid'))
done = False
t = 0
while not done:
    action = driver.act(obs)
    obs, rewards, done, truncated, states = env.step(action)
    sleep(0.01)
    if t % 30 == 0:
        image = env.render()
    t+=1


env.close()