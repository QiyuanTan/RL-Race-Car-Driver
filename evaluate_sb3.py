"""Evaluate a trained DQN model (base or LiDAR) with the same episode loop as the FTG baseline.

I reuse run_episode() and summarize() from evaluate_baseline.py, so the results have the same
fields as the FTG results (eg.lap_completed, lap_time, collisions, progress, smoothness).

The model is wrapped in SB3EvalDriver:
 raw observation -> same observation wrappers as training-> greedy DQN action ID -> same discrete action table -> {'motor', 'steering'}.

differ part from the FTG evaluation (by design, so both DQN variants match their training setup):
    1. scenario: scenarios/austria.yml (base) or scenarios/austria_lidar.yml (--lidar),
      not the registered SingleAgentAustria-v0 used for FTG
    2. terminate_on_collision is True in these scenarios, so an episode ends at the first wall
      contact (collisions is then 0 or 1); FTG's scenario keeps driving after a contact
    3, step limit: the scenario's 120 s time limit (12,000 steps), same for both variant
"""
import argparse
import json
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import VecTransposeImage
from racecar_gym.envs import gym_api

from Agents.SB3Drivers import SB3Driver, SB3EvalDriver, dqn_wrappers
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
def check_against_training(model_path, lidar, action_repeat):
    config = training_config(model_path)
    if config is None:
        return


    
    trained = {'lidar': config.get('lidar', False), 'action_repeat': config.get('action_repeat', 1)}
    now = {'lidar': lidar, 'action_repeat': action_repeat}
    if trained != now:
        raise ValueError(f'{model_path} was trained with {trained}, but this evaluation uses {now}. '
                         'Pass the same --lidar / --action-repeat as in training.')

"""rebuild the training wrappers around env and load the trained model into them."""

def load_dqn_driver(env, model_path, lidar, action_repeat=1):
    # buffer_size=1: the throwaway model built here is replaced by load(); no need for a big buffer
    driver = SB3Driver(env, model_class=DQN, wrappers=dqn_wrappers(lidar, action_repeat), buffer_size=1)
    driver.load(model_path)
    # compare key by key: gymnasium's Dict equality also depends on key order, which differs harmlessly

    if dict(driver.model.observation_space.spaces) != dict(as_sb3_sees_it(driver.env.observation_space).spaces):
        raise ValueError(
            f'{model_path} was trained on a different observation than '
            f'{"--lidar" if lidar else "the base setting"} gives.\n'
            f'model: {driver.model.observation_space}\nnow:   {driver.env.observation_space}\n'
            'Add or remove --lidar to match how the model was trained.'
        )

    
    return driver


def run_episode_in_new_process(job):
    """Fresh process per episode, because racecar_gym's reset does not fully clear the last one."""
    model_path, lidar, action_repeat, seed, render = job

    env = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(lidar),
                                     render_mode='human' if render else 'rgb_array_birds_eye')
    driver = SB3EvalDriver(load_dqn_driver(env, model_path, lidar, action_repeat), action_repeat)
    result = run_episode(env, driver, seed)
    env.close()

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained DQN model.')
    parser.add_argument('--model', required=True, help='path to the saved model .zip')
    parser.add_argument('--lidar', action='store_true', help='model was trained with LiDAR observations')
    parser.add_argument('--action-repeat', type=int, default=1,
                        help='same value as in training (checked against the training config.json)')
    
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100, help='episode i uses seed + i')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--render', action='store_true', help='show the PyBullet window (1 worker)')
    parser.add_argument('--out', default='results/dqn')
    args = parser.parse_args()

    check_against_training(args.model, args.lidar, args.action_repeat)

    jobs = [(args.model, args.lidar, args.action_repeat, args.seed + i, args.render)
            for i in range(args.episodes)]
    workers = 1 if args.render else args.workers
    episodes = []

    with mp.get_context('spawn').Pool(workers, maxtasksperchild=1) as pool:
        for result in pool.imap(run_episode_in_new_process, jobs):
            print(result, flush=True)
            episodes.append(result)

    summary = summarize(episodes)
    summary['env'] = os.path.basename(dqn_scenario(args.lidar))
    summary['mean_progress'] = round(sum(e['progress'] for e in episodes) / len(episodes), 3)
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)

    name = ('dqn_lidar' if args.lidar else 'dqn') + (f'_ar{args.action_repeat}' if args.action_repeat > 1 else '')
    path = os.path.join(args.out, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json")

    with open(path, 'w') as f:
        json.dump(dict(model=args.model, lidar=args.lidar, action_repeat=args.action_repeat,
                       scenario=os.path.join('scenarios', os.path.basename(dqn_scenario(args.lidar))),
                       summary=summary, episodes=episodes), f, indent=2)
        
    print('saved', path)


if __name__ == '__main__':
    main()
