"""Instrument DQN evaluation without changing evaluate_sb3.py.

Run from the repository root, with --workers 1:
    python diagnose_actions.py --model models/...zip --obs-mode track-lookahead \
      --action-repeat 4 --episodes 10 --seed 100 --workers 1 --out results/diag_run

Outputs to results/action_diagnostics/:
    decisions.csv    One row per DQN decision, including the decoded motor/steering
    state_samples.jsonl   Selected raw environment info fields for schema inspection
"""
import csv
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np
from Agents.wrappers import DiscreteActionWrapper

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / 'results' / 'action_diagnostics'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / 'decisions.csv'
SAMPLES_PATH = OUT_DIR / 'state_samples.jsonl'

FIELDS = [
    'episode', 'decision', 'discrete_action', 'motor', 'steering',
    'sim_time', 'lap', 'progress', 'absolute_progress',
    'reward', 'terminated', 'truncated', 'wall_collision',
    'opponent_collisions', 'speed', 'pose', 'velocity',
]

original_reset = DiscreteActionWrapper.reset
original_step = DiscreteActionWrapper.step
tracker = {'episode': -1, 'decision': 0, 'sample_count': 0}


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def short_json(value):
    return json.dumps(to_jsonable(value), ensure_ascii=False, separators=(',', ':'))


def scalar(value):
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
        return float(a[0]) if a.size == 1 and np.isfinite(a[0]) else None
    except (ValueError, TypeError):
        return None


def speed_from_info(info):
    for key in ('speed', 'linear_speed'):
        if key in info:
            result = scalar(info[key])
            if result is not None:
                return result
    vel = info.get('velocity')
    if isinstance(vel, dict):
        for key in ('linear', 'linear_velocity', 'velocity', 'xyz'):
            if key in vel:
                try:
                    a = np.asarray(vel[key], dtype=float).reshape(-1)
                    if a.size >= 2:
                        return float(np.linalg.norm(a[:3]))
                except (ValueError, TypeError):
                    pass
    elif vel is not None:
        try:
            a = np.asarray(vel, dtype=float).reshape(-1)
            if a.size >= 2:
                return float(np.linalg.norm(a[:3]))
        except (ValueError, TypeError):
            pass
    return None


def logged_reset(self, *args, **kwargs):
    result = original_reset(self, *args, **kwargs)
    tracker['episode'] += 1
    tracker['decision'] = 0
    return result


def logged_step(self, action):
    decoded = self.action(action)
    obs, reward, terminated, truncated, info = original_step(self, action)
    info = info or {}
    p = scalar(info.get('progress'))
    lap = scalar(info.get('lap'))
    row = {
        'episode': tracker['episode'],
        'decision': tracker['decision'],
        'discrete_action': int(action),
        'motor': scalar(decoded.get('motor')),
        'steering': scalar(decoded.get('steering')),
        'sim_time': scalar(info.get('time')),
        'lap': lap,
        'progress': p,
        'absolute_progress': (lap + p) if lap is not None and p is not None else None,
        'reward': float(reward),
        'terminated': bool(terminated),
        'truncated': bool(truncated),
        'wall_collision': bool(info.get('wall_collision', False)),
        'opponent_collisions': short_json(info.get('opponent_collisions')),
        'speed': speed_from_info(info),
        'pose': short_json(info.get('pose')),
        'velocity': short_json(info.get('velocity')),
    }
    with CSV_PATH.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writerow(row)

    # State samples preserve the actual shape of important simulator fields.
    if tracker['sample_count'] < 15 or terminated or truncated:
        sample = {
            'episode': tracker['episode'], 'decision': tracker['decision'],
            'info_keys': list(info),
            'info': {k: to_jsonable(info.get(k)) for k in
                     ('time', 'lap', 'progress', 'pose', 'velocity', 'speed',
                      'wall_collision', 'opponent_collisions') if k in info},
        }
        with SAMPLES_PATH.open('a', encoding='utf-8') as file:
            file.write(json.dumps(sample, ensure_ascii=False) + '\n')
        tracker['sample_count'] += 1

    tracker['decision'] += 1
    return obs, reward, terminated, truncated, info


def main():
    if not (REPO_ROOT / 'evaluate_sb3.py').exists():
        raise FileNotFoundError('Please place diagnose_actions.py in the same directory as evaluate_sb3.py')
    with CSV_PATH.open('w', newline='', encoding='utf-8') as file:
        csv.DictWriter(file, fieldnames=FIELDS).writeheader()
    SAMPLES_PATH.write_text('', encoding='utf-8')
    DiscreteActionWrapper.reset = logged_reset
    DiscreteActionWrapper.step = logged_step
    sys.argv = [str(REPO_ROOT / 'evaluate_sb3.py')] + sys.argv[1:]
    try:
        runpy.run_path(str(REPO_ROOT / 'evaluate_sb3.py'), run_name='__main__')
    finally:
        DiscreteActionWrapper.reset = original_reset
        DiscreteActionWrapper.step = original_step
        print(f'\n[diagnostics] Decisions: {CSV_PATH}')
        print(f'[diagnostics] State samples: {SAMPLES_PATH}')
        print(f'[diagnostics] Rows logged: {max(0, sum(1 for _ in CSV_PATH.open(encoding="utf-8")) - 1)}')


if __name__ == '__main__':
    main()
