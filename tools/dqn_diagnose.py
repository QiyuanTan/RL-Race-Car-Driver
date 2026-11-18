"""Summarize DQN runs: training curves (TensorBoard) and checkpoint evaluations (results JSON).

Usage (from the repo root):
    python dqn_diagnose.py curves            # training curve of the latest base and LiDAR runs
    python dqn_diagnose.py evals results/dqn_ckpt   # mean progress of every evaluated checkpoint
"""
import glob
import json
import os
import sys


def curves():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for variant in ('dqn', 'dqn_lidar'):
        runs = sorted(glob.glob(f'logs/{variant}/DQN_*'), key=os.path.getmtime)
        if not runs:
            print(f'{variant}: no runs in logs/{variant}')
            continue
        run = runs[-1]                                   # latest run = the 200k run
        points = {}
        for f in sorted(glob.glob(run + '/events*')):    # a resumed run writes a second file
            ea = EventAccumulator(f, size_guidance={'scalars': 0})
            ea.Reload()
            for tag in ('rollout/ep_rew_mean', 'rollout/ep_len_mean'):
                if tag in ea.Tags()['scalars']:
                    for e in ea.Scalars(tag):
                        points.setdefault(e.step, {})[tag] = e.value
        steps = sorted(points)
        print(f'== {variant}  ({run}, {len(steps)} points, last step {steps[-1] if steps else None})')
        print('    step   ep_rew_mean  ep_len_mean   (reward ~= % of a lap driven per episode)')
        shown = []
        for i in range(1, 11):                           # ~10 evenly spaced points over the run
            near = [s for s in steps if s <= steps[-1] * i / 10]
            if near and near[-1] not in shown:
                shown.append(near[-1])
                p = points[near[-1]]
                print(f'  {near[-1]:7d}   {p.get("rollout/ep_rew_mean", float("nan")):9.2f}   '
                      f'{p.get("rollout/ep_len_mean", float("nan")):9.0f}')


def evals(folder):
    rows = []
    for f in glob.glob(os.path.join(folder, '*.json')):
        d = json.load(open(f))
        model = os.path.basename(d['model']).replace('.zip', '')
        if model.endswith('_steps'):                     # <run>_<step>_steps
            run, step = model.rsplit('_', 2)[0], int(model.split('_')[-2])
        else:                                            # final model: <run>
            run, step = model, None
        eps = d['episodes']
        rows.append((run, step if step is not None else -1, d['summary']['mean_progress'],
                     min(e['progress'] for e in eps), max(e['progress'] for e in eps),
                     sum(e['lap_completed'] for e in eps), len(eps),
                     sum(e['collisions'] for e in eps) / len(eps),
                     sum(e['sim_time'] for e in eps) / len(eps)))
    if not rows:
        print(f'no results in {folder} (did the evaluations fail? run one without "> /dev/null")')
        return
    print(f'{"run":26s} {"step":>7s}  mean_progress  (min-max)      laps   collisions  mean episode s')
    for r in sorted(rows):
        step = 'final' if r[1] < 0 else str(r[1])
        print(f'{r[0]:26s} {step:>7s}      {r[2]:.3f}     ({r[3]:.3f}-{r[4]:.3f})  {r[5]}/{r[6]}     {r[7]:.1f}        {r[8]:.1f}')


if __name__ == '__main__':
    if sys.argv[1] == 'curves':
        curves()
    else:
        evals(sys.argv[2])
