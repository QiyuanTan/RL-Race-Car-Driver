"""Can FTG clear the Austria hairpin with DQN's controls

Start just before the hairpinand
stop at the first wall hit, like DQN + LiDAR.

ftg uses continuous controls every step. ar4, ar2, and ar1 use DQN's
15 actions and pick a new action every 4, 2, or 1 steps, like in
tools/ftg_action_check.py. DQN uses ar4.

Count it as a pass if progress goes above 0.37 (about 3 m past the apex at 0.351).
If FTG rarely passes with `ar4`, try changing the actions or how often we pick
them before spending more time training DQN on the hairpin.

if anyone want to run command:python tools/ftg_hairpin_check.py --seeds 1000-1019 --workers 4
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import multiprocessing as mp

import numpy as np

PASSED = 0.37 # ~3 m past the apex
STOP_AT = 0.40  # well past the hairpin, no need to drive further
TIME_LIMIT_S = 30.0 # starting 4.5-11 m before the apex, this is plenty


def run_one(job):
    name, seed = job
    from racecar_gym.envs import gym_api
    from Agents.FollowTheGapDriver import FollowTheGapDriver
    from Agents.wrappers import HairpinStartWrapper
    from tools.ftg_action_check import SETS, SnappedFTG
    from train import dqn_scenario

    env = HairpinStartWrapper(gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(True),
                                                         render_mode='rgb_array_birds_eye'), share=1.0)
    ftg = FollowTheGapDriver(env)
    driver = ftg if name == 'ftg' else SnappedFTG(ftg, *SETS[name])
    np.random.seed(seed)                    
    obs, info = env.reset(seed=seed)
    driver.reset()
    start, best, done, truncated = info['progress'], info['progress'], False, False
    while not (done or truncated) and info['time'] < TIME_LIMIT_S and best < STOP_AT:
        obs, _, done, truncated, info = env.step(driver.act(obs))
        if info['progress'] > 0.5:          
            continue
        best = max(best, info['progress'])
    env.close()
    end = 'passed' if best > PASSED else 'wall' if info['wall_collision'] else 'stopped/time'
    return name, seed, start, best, end


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sets', nargs='+', default=['ftg', 'ar1', 'ar2', 'ar4'],
                        choices=['ftg', 'ar1', 'ar2', 'ar4', 'ar2_steer9'])
    parser.add_argument('--seeds', default='1000-1019', help='range like 1000-1019')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    first, last = (int(x) for x in args.seeds.split('-'))
    jobs = [(name, seed) for name in args.sets for seed in range(first, last + 1)]

    results = {}
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        for name, seed, start, best, end in pool.imap_unordered(run_one, jobs):
            results.setdefault(name, []).append(end)
            print(f'{name:10s} seed {seed}: start progress {start:.3f} -> best {best:.3f}  {end}', flush=True)

    print(f'\npassed the hairpin (progress > {PASSED}), starting 4.5-11 m before it:')
    for name in args.sets:
        ends = results[name]
        print(f'  {name:10s} {ends.count("passed"):2d}/{len(ends)}   '
              f'(wall {ends.count("wall")}, stopped/time {ends.count("stopped/time")})')


if __name__ == '__main__':
    main()
