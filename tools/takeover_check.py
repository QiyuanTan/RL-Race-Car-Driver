"""Can FTG save the car from the state DQN brings it into before the hairpin? (no training)

tools/hairpin_entry_compare.py showed that DQN reaches the hairpin at ~3.3 m/s, close to the inside
(right) wall, and never turns in, while FTG enters at <=2 m/s, centred, and turns in at progress
~0.32-0.33. Demonstrations recorded from FTG's own runs would therefore only show what to do in
states DQN never visits. Idea (like DAgger): let DQN drive from the grid and hand over to FTG at
some progress value; if FTG gets through from there, those takeover runs are demonstrations that
start in DQN's own states.

For every handover point and every FTG variant we drive the given seeds and report how many runs
passed the hairpin (progress > 0.37 without touching a wall) and the state at the handover.
FTG variants (same as tools/ftg_action_check.py):
  ar4  FTG with the turn-in rule, snapped to DQN's 15 actions, deciding every 4 steps
       (the only variant whose runs DQN could use as demonstrations)
  ftg  plain FTG, continuous commands every step (tells us if the state can be saved at all)

Usage (repo root):
    python tools/takeover_check.py --model models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --seeds 0-9 --workers 4
    python tools/takeover_check.py --model ftg ...   # sanity check: FTG (ar4) drives before the handover too
Saves results/takeover_check/<time>.json.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import multiprocessing as mp
from datetime import datetime

import numpy as np

from tools.hairpin_entry_compare import PASSED, space

STOP_AT = 0.40           # well past the hairpin
TIME_LIMIT_S = 60.0


def make_ftg(env, variant, repeat):
    from Agents.FollowTheGapDriver import FollowTheGapDriver
    from tools.ftg_action_check import MOTOR_3, STEER_5, SnappedFTG
    ftg = FollowTheGapDriver(env)
    return ftg if variant == 'ftg' else SnappedFTG(ftg, repeat, MOTOR_3, STEER_5)


def run_one(job):
    model_path, repeat, handover, variant, seed = job
    from racecar_gym.envs import gym_api
    from train import dqn_scenario

    env = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(True), render_mode='rgb_array_birds_eye')
    if model_path == 'ftg':
        first = make_ftg(env, 'ar4', repeat)
    else:
        from Agents.SB3Drivers import SB3EvalDriver
        from evaluate_sb3 import load_dqn_driver
        first = SB3EvalDriver(load_dqn_driver(env, model_path, True, repeat), repeat)
    second = make_ftg(env, variant, repeat)

    np.random.seed(seed)                     # sensor noise, as in evaluate_baseline.run_episode
    obs, info = env.reset(seed=seed, options={'mode': 'grid'})
    first.reset()
    driver, state, step, done, truncated = first, None, 0, False, False
    best = float(info['progress'])
    while not (done or truncated) and best < STOP_AT and info['time'] < TIME_LIMIT_S:
        # hand over at the first decision at or after the handover point
        if state is None and step % repeat == 0 and info['progress'] >= handover:
            state = {'progress': round(float(info['progress']), 4), 'time': round(float(info['time']), 2),
                     'speed': round(float(np.hypot(*info['velocity'][:2])), 2),
                     'ahead': round(space(obs['lidar'], 0), 2), 'left': round(space(obs['lidar'], 90), 2),
                     'right': round(space(obs['lidar'], -90), 2)}
            driver = second
            driver.reset()
        obs, _, done, truncated, info = env.step(driver.act(obs))
        step += 1
        if info['progress'] < 0.5:           
            best = max(best, float(info['progress']))
    env.close()
    crashed = bool(info['wall_collision'])
    if state is None:
        end = 'crashed before handover' if crashed else 'never reached handover'
    else:
        end = 'passed' if best > PASSED and not crashed else 'crashed' if crashed else 'stopped/time'
    return {'handover': handover, 'variant': variant, 'seed': seed, 'end': end,
            'best_progress': round(best, 4), 'at_handover': state}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help="DQN + LiDAR model .zip, or 'ftg' for the sanity check")
    parser.add_argument('--action-repeat', type=int, default=4)
    parser.add_argument('--handover', nargs='+', type=float, default=[0.24, 0.26, 0.28, 0.30])
    parser.add_argument('--variants', nargs='+', default=['ar4', 'ftg'], choices=['ar4', 'ftg'])
    parser.add_argument('--seeds', default='0-9', help='range like 0-9')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default='results/takeover_check')
    args = parser.parse_args()

    if args.model != 'ftg':
        from evaluate_sb3 import check_against_training
        check_against_training(args.model, True, args.action_repeat)
    first, last = (int(x) for x in args.seeds.split('-'))
    jobs = [(args.model, args.action_repeat, h, v, s)
            for h in args.handover for v in args.variants for s in range(first, last + 1)]
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        results = list(pool.imap(run_one, jobs))

    print(f'\nfirst driver: {args.model}')
    print('handover  FTG   passed   crashed  other   at handover (mean): speed  ahead  left  right')
    for h in args.handover:
        for v in args.variants:
            rs = [r for r in results if r['handover'] == h and r['variant'] == v]
            ends = [r['end'] for r in rs]
            states = [r['at_handover'] for r in rs if r['at_handover']]
            mean = {k: np.mean([s[k] for s in states]) for k in ('speed', 'ahead', 'left', 'right')} if states else None
            where = (f"{mean['speed']:5.2f}  {mean['ahead']:5.2f}  {mean['left']:4.2f}  {mean['right']:4.2f}"
                     if mean else '-')
            other = len(rs) - ends.count('passed') - ends.count('crashed')
            print(f'  {h:.2f}    {v:4s}  {ends.count("passed"):2d}/{len(rs)}    {ends.count("crashed"):2d}       '
                  f'{other:2d}      {where}')
    others = sorted({r['end'] for r in results} - {'passed', 'crashed'})
    if others:
        print('other =', ', '.join(f"{e}: {sum(r['end'] == e for r in results)}" for e in others))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f'{datetime.now():%Y%m%d_%H%M%S}.json')
    with open(path, 'w') as f:
        json.dump({'model': args.model, 'action_repeat': args.action_repeat, 'runs': results}, f, indent=1)
    print('saved', path)


if __name__ == '__main__':
    main()
