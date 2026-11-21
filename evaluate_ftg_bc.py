#!/usr/bin/env python3
"""Evaluate FTG-BC DQN with the same Racecar Gym metrics as evaluate_sb3.py.

Supports track-lookahead geometry-only models, whose inputs intentionally omit
camera and other raw sensors. `--max-sim-time` can cut off a stuck-car preview;
120 is the project's standard evaluation limit.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from racecar_gym.envs import gym_api

from Agents.SB3Drivers import SB3Driver, SB3EvalDriver, dqn_wrappers
from Agents.ftg_geometry import GeometryOnlyObservationWrapper
from evaluate_baseline import run_episode, summarize
from evaluate_sb3 import EpisodeRecorder
from train import dqn_scenario


class PreviewTimeLimit(gym.Wrapper):
    def __init__(self, env, max_seconds):
        super().__init__(env)
        self.max_seconds = max_seconds

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('time', 0) >= self.max_seconds and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info


def evaluate_one(model_path, seed, action_repeat, geometry, max_seconds):
    base = gym_api.SingleAgentRaceEnv(
        scenario=dqn_scenario(False), render_mode='rgb_array_birds_eye')
    if max_seconds < 120:
        base = PreviewTimeLimit(base, max_seconds)
    env = EpisodeRecorder(base)
    wrappers = dqn_wrappers(obs_mode='track-lookahead', action_repeat=action_repeat)
    if geometry:
        wrappers.append(GeometryOnlyObservationWrapper)
    driver = SB3Driver(env, model_class=DQN, wrappers=wrappers, buffer_size=1)
    try:
        driver.load(str(model_path))
        expected = driver.model.observation_space.spaces
        actual = driver.env.observation_space.spaces
        if set(expected) != set(actual) or any(
            expected[k].shape != actual[k].shape or expected[k].dtype != actual[k].dtype
            for k in expected
        ):
            raise ValueError('Student observation spaces do not match the saved model')
        policy = SB3EvalDriver(driver, action_repeat=action_repeat)
        result = run_episode(env, policy, seed)
        result.update(env.summary())
        return result
    finally:
        driver.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--features', choices=['geometry', 'full'], default='geometry')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--action-repeat', type=int, default=4)
    parser.add_argument('--max-sim-time', type=float, default=120.0)
    parser.add_argument('--out', default='/tmp/jx_ftg_bc_eval')
    args = parser.parse_args()
    if args.episodes < 1 or not 0 < args.max_sim_time <= 120:
        parser.error('Need at least one episode and 0 < max-sim-time <= 120')
    if not Path(args.model).exists():
        parser.error(f'Model not found: {args.model}')

    results = []
    for seed in range(args.seed, args.seed + args.episodes):
        result = evaluate_one(args.model, seed, args.action_repeat,
                              args.features == 'geometry', args.max_sim_time)
        print({k: v for k, v in result.items() if k != 'trajectory_every_10_steps'}, flush=True)
        results.append(result)
    summary = summarize(results)
    progress = [x['progress'] for x in results]
    summary['mean_progress'] = round(sum(progress) / len(progress), 3)
    summary['progress_min'] = min(progress)
    summary['progress_max'] = max(progress)
    summary['lap_rate'] = sum(x['lap_completed'] for x in results) / len(results)
    summary['eval_max_sim_time'] = args.max_sim_time
    summary['preview_only'] = args.max_sim_time < 120
    summary['features'] = args.features
    summary['model'] = args.model
    print(json.dumps(summary, indent=2), flush=True)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f'ftg_bc_{datetime.now():%Y%m%d_%H%M%S}.json'
    dest.write_text(json.dumps({'summary': summary, 'episodes': results}, indent=2))
    print('Saved:', dest, flush=True)


if __name__ == '__main__':
    main()
