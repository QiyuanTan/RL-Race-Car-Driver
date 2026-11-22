"""Run all supported drivers through the same episode metrics interface."""
import argparse
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

from shared_evaluation import EvaluationSpec, build_driver, evaluate_episode


def run_job(job):
    spec, seed = job
    import gymnasium as gym
    import racecar_gym.envs.gym_api  # register FTG environment
    from racecar_gym.envs import gym_api
    from train import dqn_scenario
    if spec.algorithm == 'ftg':
        env = gym.make('SingleAgentAustria-v0', render_mode='rgb_array_birds_eye')
        scenario = 'SingleAgentAustria-v0'
    else:
        scenario = dqn_scenario(spec.obs_mode == 'lidar')
        env = gym_api.SingleAgentRaceEnv(scenario=scenario, render_mode='rgb_array_birds_eye')
    try:
        return evaluate_episode(env, build_driver(env, spec), seed)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['ftg', 'dqn', 'ppo', 'sac'], required=True)
    parser.add_argument('--model')
    parser.add_argument('--obs-mode', choices=['base', 'lidar', 'track', 'track-lookahead'], default='base')
    parser.add_argument('--action-repeat', type=int, default=1)
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--out', default='results/shared')
    args = parser.parse_args()
    try:
        spec = EvaluationSpec(args.algorithm, args.model, args.obs_mode, args.action_repeat, args.geometry_only)
        if args.episodes < 1 or args.workers < 1:
            raise ValueError('episodes and workers must be positive')
    except ValueError as error:
        parser.error(str(error))
    from evaluate_baseline import summarize
    jobs = [(spec, args.seed + i) for i in range(args.episodes)]
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        episodes = list(pool.imap(run_job, jobs))
    summary = summarize(episodes)
    summary['mean_progress'] = round(sum(e['progress'] for e in episodes) / len(episodes), 3)
    settings = vars(spec).copy()
    settings.update(seeds=[args.seed + i for i in range(args.episodes)],
                    scenario=('SingleAgentAustria-v0' if spec.algorithm == 'ftg' else
                              'austria_lidar.yml' if spec.obs_mode == 'lidar' else 'austria.yml'),
                    collision_policy=('continue' if spec.algorithm == 'ftg' else 'terminate on first contact'))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f'{spec.algorithm}_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    path.write_text(json.dumps({'settings': settings, 'summary': summary, 'episodes': episodes}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
