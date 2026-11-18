"""This file only to solve one Question: Can FTG finish a lap when it may only use a discrete action set at a given decision rate?

No training. FTG decides every `repeat` simulator steps and each command is snapped to the nearest
allowed motor and steering level. Same scenario and termination as DQN + LiDAR.
Each run is a fresh process,like evaluate_sb3.py. 
you guys can run e.g.:
    python ftg_action_check.py --seeds 1000-1009 --workers 4
    python ftg_action_check.py --sets ar2 ar1 --seeds 1000-1009 --workers 4
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import multiprocessing as mp
import numpy as np

MOTOR_3 = [-0.5, 0.0, 0.5]
STEER_5 = [-1.0, -0.5, 0.0, 0.5, 1.0]
STEER_9 = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
SETS = {   # name: (decide every N simulator steps, motor levels, steering levels)
    'ar4':         (4, MOTOR_3, STEER_5),   # current DQN setting
    'ar2':         (2, MOTOR_3, STEER_5),
    'ar1':         (1, MOTOR_3, STEER_5),
    'ar2_steer9':  (2, MOTOR_3, STEER_9),
}


class SnappedFTG:
    def __init__(self, ftg, repeat, motors, steers):
        self.ftg, self.repeat = ftg, repeat
        self.motors, self.steers = np.array(motors, float), np.array(steers, float)

    def reset(self):
        self.k, self.action = 0, None

    def act(self, obs):
        if self.k % self.repeat == 0:
            a = self.ftg.act(obs)
            m = self.motors[np.abs(self.motors - a['motor'][0]).argmin()]
            s = self.steers[np.abs(self.steers - a['steering'][0]).argmin()]
            self.action = {'motor': np.array([m], np.float32), 'steering': np.array([s], np.float32)}
        self.k += 1
        return {k: v.copy() for k, v in self.action.items()}


def run_one(job):
    name, seed = job
    from racecar_gym.envs import gym_api
    from Agents.FollowTheGapDriver import FollowTheGapDriver
    from evaluate_baseline import run_episode
    from train import dqn_scenario
    repeat, motors, steers = SETS[name]
    env = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(True), render_mode='rgb_array_birds_eye')
    result = run_episode(env, SnappedFTG(FollowTheGapDriver(env), repeat, motors, steers), seed)
    env.close()
    return name, seed, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sets', nargs='+', default=['ar4', 'ar2', 'ar1', 'ar2_steer9'], choices=list(SETS))
    parser.add_argument('--seeds', default='1000-1009', help='range like 1000-1009')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    first, last = (int(x) for x in args.seeds.split('-'))
    jobs = [(name, seed) for name in args.sets for seed in range(first, last + 1)]
    results = {}
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        for name, seed, r in pool.imap_unordered(run_one, jobs):
            results.setdefault(name, []).append((seed, r))
            print(f"{name:11s} seed {seed}: lap={r['lap_completed']} progress={r['progress']:.3f} "
                  f"time={r['sim_time']:.1f}s", flush=True)
    print('\nset          laps    past hairpin (>0.37)  past 0.55   mean lap time   progress by seed')
    for name in args.sets:
        rs = sorted(results[name], key=lambda x: x[0])
        laps = sum(r['lap_completed'] for _, r in rs)
        hairpin = sum(r['progress'] > 0.37 for _, r in rs)
        second = sum(r['progress'] > 0.55 for _, r in rs)
        times = [r['lap_time'] for _, r in rs if r['lap_completed']]
        mean_time = f'{sum(times) / len(times):.1f}s' if times else '-'
        print(f"{name:11s} {laps:2d}/{len(rs)}        {hairpin:2d}/{len(rs)}              {second:2d}/{len(rs)}"
              f"      {mean_time:>8s}        " + ' '.join(f"{r['progress']:.2f}" for _, r in rs))


if __name__ == '__main__':
    main()
