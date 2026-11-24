"""Evaluates a DQN model in base, lidar, track or track-lookahead mode.

Uses run_episode() and summarize() from evaluate_baseline.py, so we get the same
metrics as FTG: lap completion, lap time, collisions, progress and smoothness.

SB3EvalDriver processes observations the same way as training, picks the
highest-scoring action, then turns the action ID into motor and steering.

A few things differ from FTG to match how DQN was trained:
- Base and track modes use scenarios/austria.yml. Lidar uses scenarios/austria_lidar.yml.
  FTG uses SingleAgentAustria-v0.
- DQN episodes stop at the first wall contact, so collisions can only be 0 or 1.
  FTG keeps driving after contact.
- All DQN modes use the scenario's 120 s limit, or 12,000 simulator steps.
"""
import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import VecTransposeImage
from racecar_gym.envs import gym_api

from Agents.SB3Drivers import SB3Driver, SB3EvalDriver, dqn_wrappers, resolve_obs_mode
from Agents.track_features import feature_config
from evaluate_baseline import run_episode, summarize
from train import dqn_scenario

"""SB3 stores image observations channels-first, so transpose image keys the same way before comparing spaces."""
def as_sb3_sees_it(observation_space):
    return spaces.Dict({
        key: VecTransposeImage.transpose_space(space, key)
        if is_image_space(space) and not is_image_space_channels_first(space) else space
        for key, space in observation_space.spaces.items()
    })

"""config.json written by train.py for this model, or None for models trained without it. """

def training_config(model_path):
    path = Path(model_path)

    if path.parent.name == 'eval':
        config = path.parent.parent / 'config.json'
    else:
        config = path.parent / 'checkpoints' / path.stem / 'config.json'
    return json.loads(config.read_text()) if config.exists() else None

"""refuse to evaluate with settings that differ from how the model was trained."""
def check_against_training(model_path, lidar, action_repeat, obs_mode=None, no_camera=False):
    """Compares the obs mode, every feature setting (names, scales, centre line hash) and the action
    repeat, so e.g. a model trained on an older centre line is refused even if the sizes match."""
    config = training_config(model_path)
    if config is None:
        return
    obs_mode = resolve_obs_mode(lidar, obs_mode)
    old_mode = 'lidar' if config.get('lidar', False) else 'base'      # configs from before --obs-mode
    trained = {'obs_mode': config.get('obs_mode', old_mode), 'action_repeat': config.get('action_repeat', 1),
               'features': config.get('features', feature_config(old_mode)),
               'no_camera': config.get('no_camera', False)}
    now = {'obs_mode': obs_mode, 'action_repeat': action_repeat,
           'features': feature_config(obs_mode), 'no_camera': no_camera}
    if trained != now:
        differences = {key: (trained[key], now[key]) for key in trained if trained[key] != now[key]}
        if 'features' in differences:     # show only the feature settings that differ
            old, new = differences.pop('features')
            differences.update({f'features.{k}': (old.get(k), new.get(k)) for k in old.keys() | new.keys()
                                if old.get(k) != new.get(k)})
        raise ValueError(f'{model_path} was trained differently, (training, now): {differences}. '
                         'Pass the same --obs-mode (or --lidar), --action-repeat, and --no-camera as in training, '
                         'and use the centre line file it was trained with.')

"""rebuild the training wrappers around env and load the trained model into them."""

def load_dqn_driver(env, model_path, lidar, action_repeat=1, obs_mode=None, no_camera=False):
    obs_mode = resolve_obs_mode(lidar, obs_mode)
    #buffer_size=1: the throwaway model built here is replaced by load(); no need for a big buffer
    driver = SB3Driver(env, model_class=DQN, wrappers=dqn_wrappers(action_repeat=action_repeat, obs_mode=obs_mode,
                                                           no_camera=no_camera),
                       buffer_size=1)
    driver.load(model_path)
    #compare key by key: gymnasium's Dict equality also depends on key order, which differs harmlessly

    if dict(driver.model.observation_space.spaces) != dict(as_sb3_sees_it(driver.env.observation_space).spaces):
        raise ValueError(
            f'{model_path} was trained on a different observation than '
            f'--obs-mode {obs_mode} gives.\n'
            f'model: {driver.model.observation_space}\nnow:   {driver.env.observation_space}\n'
            'Use the --obs-mode and --no-camera settings the model was trained with.'
        )

    
    return driver


TIME_LIMIT = 120.0   #seconds, from scenarios/austria*.yml


class EpisodeRecorder(gym.Wrapper):
    """Watches the raw environment during evaluation only (never seen by the policy).

    Records how the episode ended and a coarse trajectory (every 10 simulator steps), so we can
    tell a wall crash from a time-out, spot cars that stopped, and check where on the track a run
    ended (progress > 35% alone does not prove the hairpin was passed).
    """

    def __init__(self, env, every=10):
        super().__init__(env)
        self.every = every

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps, self.trajectory, self.progress_log = 0, [], []
        self.ended = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if self.steps % self.every == 0 or terminated or truncated:
            self.trajectory.append([round(float(info['pose'][0]), 2), round(float(info['pose'][1]), 2)])
            self.progress_log.append((float(info['time']), float(info['progress'])))
        if terminated or truncated:
            self.ended = info
            #racecar_gym ends the 120 s limit with terminated=True, so check the time too
            self.how = 'lap' if info['lap'] >= 2 else 'wall_crash' if info['wall_collision'] else \
                'time_limit' if truncated or info['time'] >= TIME_LIMIT - 0.02 else 'other_termination'
        return obs, reward, terminated, truncated, info

    def summary(self):
        # "stuck": progress changed less than 0.5% of a lap in the last 10 s of the episode
        end_time = self.progress_log[-1][0]
        recent = [p for time, p in self.progress_log if time >= end_time - 10.0]
        moved = np.abs(np.diff(recent))
        moved = moved[moved < 0.5].sum()# ignore the 1 -> 0 jump at the start line
        speed = np.hypot(*self.ended['velocity'][:2])
        return {'end': self.how, 'stuck_last_10s': bool(end_time >= 10.0 and moved < 0.005),
                'final_speed': round(float(speed), 2), 'final_xy': self.trajectory[-1],
                'trajectory_every_10_steps': self.trajectory}


def run_episode_in_new_process(job):
    """Fresh process per episode, because racecar_gym's reset does not fully clear the last one."""
    model_path, obs_mode, action_repeat, seed, render, no_camera = job

    env = EpisodeRecorder(gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(obs_mode == 'lidar'),
                                                     render_mode='human' if render else 'rgb_array_birds_eye'))
    driver = SB3EvalDriver(load_dqn_driver(env, model_path, False, action_repeat, obs_mode,
                                         no_camera=no_camera), action_repeat)
    result = run_episode(env, driver, seed)
    result.update(env.summary())
    env.close()

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN model.')
    parser.add_argument('--model', required=True, help='path to the saved model .zip')
    parser.add_argument('--lidar', action='store_true', help='model was trained with LiDAR (= --obs-mode lidar)')
    parser.add_argument('--obs-mode', choices=['base', 'lidar', 'track', 'track-lookahead'], default=None,
                        help='same as in training (checked against the training config.json)')
    parser.add_argument('--no-camera', action='store_true',
                        help='model was trained without RGB camera observations')
    parser.add_argument('--action-repeat', type=int, default=1,
                        help='same value as in training (checked against the training config.json)')
    
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100, help='episode i uses seed + i')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--render', action='store_true', help='show the PyBullet window (1 worker)')
    parser.add_argument('--out', default='results/dqn')
    args = parser.parse_args()

    obs_mode = resolve_obs_mode(args.lidar, args.obs_mode)
    check_against_training(args.model, False, args.action_repeat, obs_mode, no_camera=args.no_camera)

    jobs = [(args.model, obs_mode, args.action_repeat, args.seed + i, args.render, args.no_camera)
            for i in range(args.episodes)]
    workers = 1 if args.render else args.workers
    episodes = []

    with mp.get_context('spawn').Pool(workers, maxtasksperchild=1) as pool:
        for result in pool.imap(run_episode_in_new_process, jobs):
            print({k: v for k, v in result.items() if k != 'trajectory_every_10_steps'}, flush=True)
            episodes.append(result)

    summary = summarize(episodes)
    summary['env'] = os.path.basename(dqn_scenario(obs_mode == 'lidar'))
    progress = np.array([e['progress'] for e in episodes])
    summary['mean_progress'] = round(float(progress.mean()), 3)
    summary.update(progress_std=round(float(progress.std()), 3), progress_min=float(progress.min()),
                   progress_max=float(progress.max()),
                   lap_rate=round(sum(e['lap_completed'] for e in episodes) / len(episodes), 2),
                   ends={end: sum(e['end'] == end for e in episodes) for end in sorted({e['end'] for e in episodes})},
                   stuck=sum(e['stuck_last_10s'] for e in episodes), action_repeat=args.action_repeat,
                   obs_mode=obs_mode, no_camera=args.no_camera)
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)

    name = {'base': 'dqn', 'lidar': 'dqn_lidar'}.get(obs_mode, f"dqn_{obs_mode.replace('-', '_')}") + (f'_ar{args.action_repeat}' if args.action_repeat > 1 else '')
    if args.no_camera:
        name += "_nocamera"
    path = os.path.join(args.out, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json")

    with open(path, 'w') as f:
        json.dump(dict(model=args.model, obs_mode=obs_mode, lidar=obs_mode == 'lidar', action_repeat=args.action_repeat,
                       no_camera=args.no_camera,
                       scenario=os.path.join('scenarios', os.path.basename(dqn_scenario(obs_mode == 'lidar'))),
                       summary=summary, episodes=episodes), f, indent=2)
        
    print('saved', path)


if __name__ == '__main__':
    main()
