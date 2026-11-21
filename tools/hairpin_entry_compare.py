"""compare how DQN and FTG enter the first hairpin.

oth start on the grid in the DQN + LiDAR scenario and stop at the first
wall hit. they use the same 15 actions and decision interval. FTG uses
its turn-in rule, with commands snapped to DQN's actions as in
tools/ftg_action_check.py.

log each decision from progress 0.20 to 0.40. at the same points on the
track, compare speed (m/s), yaw (deg), LiDAR clearance ahead and on both
sides (m), and steering and motor commands. at the same progress, the
yaw difference shows how differently the cars are pointing, and left
vs right clearance gives a rough idea of where each car is across the track.

count turn-in from the first of 8+ straight decisions (0.32 s) at full
right (+1) after progress 0.30. the hairpin turns right, and FTG's small
corrections on the straight can already snap to -1 or +1. record speed
and space ahead when the turn starts.

a pass means reaching progress above 0.37 without hitting a wall, about
3 m past the apex at 0.351.

    python tools/hairpin_entry_compare.py --model models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --seeds 0-9 --workers 4
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import multiprocessing as mp
from datetime import datetime

import numpy as np

WINDOW = (0.20, 0.40)   # progress range that is logged
PASSED = 0.37           # ~3 m past the apex
BINS = np.round(np.arange(0.24, 0.371, 0.02), 2)   # places on the track to compare
RAY_ANGLES = -(-2.36 + np.arange(1080) * (4.71 / 1080))   # radians, 0 = ahead, + = left (as in FollowTheGapDriver)


def space(scan, centre_deg, half_width_deg=10):
    """Closest LiDAR distance within +-half_width around a direction."""
    near = np.abs(RAY_ANGLES - np.deg2rad(centre_deg)) <= np.deg2rad(half_width_deg)
    return float(np.min(np.asarray(scan)[near]))


def run_one(job):
    """Drive one episode from the grid and log the decisions near the hairpin."""
    driver_name, model_path, repeat, seed = job
    from racecar_gym.envs import gym_api
    from train import dqn_scenario

    env = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(True), render_mode='rgb_array_birds_eye')
    if driver_name == 'dqn':
        from Agents.SB3Drivers import SB3EvalDriver
        from evaluate_sb3 import load_dqn_driver
        driver = SB3EvalDriver(load_dqn_driver(env, model_path, True, repeat), repeat)
    else:
        from Agents.FollowTheGapDriver import FollowTheGapDriver
        from tools.ftg_action_check import MOTOR_3, STEER_5, SnappedFTG
        driver = SnappedFTG(FollowTheGapDriver(env), repeat, MOTOR_3, STEER_5)

    np.random.seed(seed)                     # sensor noise, as in evaluate_baseline.run_episode
    obs, info = env.reset(seed=seed, options={'mode': 'grid'})
    driver.reset()
    rows, yaws, step, done, truncated = [], [], 0, False, False
    best = float(info['progress'])
    while not (done or truncated) and best < 0.40 and info['time'] < 60.0:
        action = driver.act(obs)
        if step % repeat == 0:               # a new decision
            yaws.append(float(info['pose'][5]))
            if WINDOW[0] <= info['progress'] <= WINDOW[1]:
                rows.append({
                    'progress': round(float(info['progress']), 4),
                    'x': round(float(info['pose'][0]), 2), 'y': round(float(info['pose'][1]), 2),
                    'yaw_deg': round(float(np.degrees(np.unwrap(yaws)[-1])), 1),
                    'speed': round(float(np.hypot(*info['velocity'][:2])), 2),
                    'ahead': round(space(obs['lidar'], 0), 2),
                    'left': round(space(obs['lidar'], 90), 2), 'right': round(space(obs['lidar'], -90), 2),
                    'motor': float(action['motor'][0]), 'steering': float(action['steering'][0]),
                })
        obs, _, done, truncated, info = env.step(action)
        step += 1
        if info['progress'] < 0.5:          
            best = max(best, float(info['progress']))
    env.close()
    crashed = bool(info['wall_collision'])
    return {'driver': driver_name, 'seed': seed, 'passed': bool(best > PASSED and not crashed),
            'crashed': crashed, 'best_progress': round(best, 4), 'end_xy': [round(float(v), 2) for v in info['pose'][:2]],
            'time': round(float(info['time']), 2), 'decisions': rows}


def first_turn(rows, start=0.30, run=8):
    """Turn-in: first decision of the first run of `run` full-right decisions after `start`."""
    for i in range(len(rows) - run + 1):
        if rows[i]['progress'] >= start and all(r['steering'] >= 0.99 for r in rows[i:i + run]):
            return rows[i]
    return None


def at(rows, progress):
    """The logged decision closest to a progress value (None if the car never got near it)."""
    if not rows:
        return None
    r = min(rows, key=lambda r: abs(r['progress'] - progress))
    return r if abs(r['progress'] - progress) <= 0.01 else None


def summary(episodes, label):
    print(f'\n== {label}: passed {sum(e["passed"] for e in episodes)}/{len(episodes)}, '
          f'crashed {sum(e["crashed"] for e in episodes)}/{len(episodes)}')
    print('  at progress   ' + '  '.join(f'{b:>5.2f}' for b in BINS))
    for key, fmt in (('speed', '{:5.2f}'), ('yaw_deg', '{:5.0f}'), ('ahead', '{:5.2f}'),
                     ('left', '{:5.2f}'), ('right', '{:5.2f}'), ('steering', '{:+5.2f}'), ('motor', '{:+5.2f}')):
        cells = []
        for b in BINS:
            values = [r[key] for e in episodes for r in [at(e['decisions'], b)] if r is not None]
            cells.append(fmt.format(np.mean(values)) if values else '    -')
        print(f'  {key:13s} ' + '  '.join(cells))
    print('  turn-in (first 8+ full-right decisions in a row after progress 0.30), per episode:')
    for e in episodes:
        t = first_turn(e['decisions'])
        where = (f'progress {t["progress"]:.3f}, speed {t["speed"]:.2f} m/s, {t["ahead"]:.2f} m ahead, '
                 f'left {t["left"]:.2f} / right {t["right"]:.2f} m') if t else 'never'
        print(f'    seed {e["seed"]:4d}  {"passed " if e["passed"] else "CRASHED" if e["crashed"] else "stopped"}  '
              f'best {e["best_progress"]:.3f}  turn-in: {where}')


def plot(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    styles = {'dqn': ('tab:red', 'DQN'), 'ftg': ('tab:blue', 'FTG (DQN actions)')}
    for e in results:
        rows = e['decisions']
        if not rows:
            continue
        colour, name = styles[e['driver']]
        ls = '-' if e['passed'] else ':'
        p = [r['progress'] for r in rows]
        for ax, key in zip(axes.flat[:5], ('speed', 'yaw_deg', 'steering', 'ahead', 'right')):
            ax.plot(p, [r[key] for r in rows], ls, color=colour, lw=1, alpha=0.7)
        axes.flat[5].plot([r['x'] for r in rows], [r['y'] for r in rows], ls, color=colour, lw=1, alpha=0.7)
    titles = ('speed [m/s]', 'heading yaw [deg]', 'steering command (+ = right)', 'space ahead [m]',
              'space on the right [m]', 'path (x, y)')
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        if title != 'path (x, y)':
            ax.axvline(0.351, color='k', lw=0.8)
            ax.set_xlabel('progress (apex at 0.351)')
    axes.flat[5].set_aspect('equal')
    for colour, name in styles.values():
        axes.flat[0].plot([], [], color=colour, label=name)
    axes.flat[0].plot([], [], 'k-', label='passed'), axes.flat[0].plot([], [], 'k:', label='crashed / stopped')
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', help='DQN + LiDAR model .zip (trained with --lidar)')
    parser.add_argument('--from-json', help='only re-print the summary of a saved run, no driving')
    parser.add_argument('--action-repeat', type=int, default=4, help='same as the model was trained with')
    parser.add_argument('--seeds', default='0-9', help='range like 0-9')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default='results/hairpin_entry')
    args = parser.parse_args()
    if args.from_json:
        with open(args.from_json) as f:
            saved = json.load(f)
        for name in ('dqn', 'ftg'):
            summary([e for e in saved['episodes'] if e['driver'] == name], name.upper())
        return
    if not args.model:
        parser.error('--model is required (or use --from-json)')

    from evaluate_sb3 import check_against_training
    check_against_training(args.model, True, args.action_repeat)
    first, last = (int(x) for x in args.seeds.split('-'))
    jobs = [(name, args.model, args.action_repeat, seed) for name in ('dqn', 'ftg') for seed in range(first, last + 1)]
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        results = list(pool.imap(run_one, jobs))

    for name, label in (('dqn', f'DQN {os.path.basename(args.model)}'),
                        ('ftg', 'FTG with turn-in rule, snapped to DQN actions')):
        summary([e for e in results if e['driver'] == name], label)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f'{datetime.now():%Y%m%d_%H%M%S}')
    with open(stem + '.json', 'w') as f:
        json.dump({'model': args.model, 'action_repeat': args.action_repeat, 'episodes': results}, f)
    plot(results, stem + '.png')
    print(f'\nsaved {stem}.json and {stem}.png')


if __name__ == '__main__':
    main()
