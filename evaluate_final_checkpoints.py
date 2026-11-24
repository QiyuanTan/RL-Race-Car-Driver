"""Evaluate the Stage-1 baseline and all Final DQN 25k checkpoints on identical grid seeds.

Run from the RL-Race-Car-Driver repository root. Uses the project's existing
`evaluate_sb3.py`, which reproduces the track-lookahead/AR4 evaluation pipeline.
No training, no checkpoint mutation. Individual evaluations are cached.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', default='models/dqn_centerline_stage1_300k.zip')
    p.add_argument('--checkpoint-dir', default='models/checkpoints/dqn_final_stall_500k')
    p.add_argument('--final-model', default='models/dqn_final_stall_500k.zip')
    p.add_argument('--out', default='results/final_checkpoint_grid_eval')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--include-final', action='store_true', help='Also evaluate final zip (may duplicate 500k checkpoint)')
    p.add_argument('--force', action='store_true', help='Rerun even existing evaluations')
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path.cwd()
    evaluator = repo / 'evaluate_sb3.py'
    if not evaluator.is_file():
        raise SystemExit('Run from repository root containing evaluate_sb3.py')
    folder = repo / args.checkpoint_dir
    if not folder.is_dir():
        alternate = repo / 'models/checkpoints/dqn_final_stall_500k'
        print(f'Checkpoint folder not found: {folder}')
        print(f'If the zip files are directly in {alternate}, use --checkpoint-dir "{alternate}"')
        raise SystemExit(2)
    matches = []
    for path in folder.rglob('*.zip'):
        found = re.search(r'_(\d+)_steps\.zip$', path.name)
        if found:
            matches.append((int(found.group(1)), path))
    matches.sort(key=lambda x: x[0])
    if not matches:
        raise SystemExit(f'No *_<steps>_steps.zip found under {folder}')
    models = [('stage1_baseline', repo / args.baseline)] + [(f'final_{step:06d}', path) for step, path in matches]
    if args.include_final:
        models.append(('final_saved', repo / args.final_model))
    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary_file = out / 'checkpoint_comparison.csv'
    rows = []
    failures = []
    print(f'Found {len(matches)} checkpoint files. Evaluating {len(models)} models on seeds {args.seed}–{args.seed+args.episodes-1}.', flush=True)
    for ix, (label, model) in enumerate(models, 1):
        print(f'[{ix}/{len(models)}] {label}: {model}', flush=True)
        if not model.is_file():
            failures.append((label, f'Missing model: {model}'))
            continue
        label_dir = out / label
        label_dir.mkdir(exist_ok=True)
        cached = label_dir / 'evaluation.json'
        if cached.exists() and not args.force:
            data = json.loads(cached.read_text(encoding='utf-8'))
            print('  Cached evaluation found; skipping simulation.', flush=True)
        else:
            cmd = [sys.executable, str(evaluator), '--model', str(model),
                   '--obs-mode', 'track-lookahead', '--action-repeat', '4',
                   '--episodes', str(args.episodes), '--seed', str(args.seed),
                   '--workers', str(args.workers), '--out', str(label_dir)]
            with (label_dir / 'console.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                msg = f'Evaluator failed ({result.returncode}); check {label_dir / "console.log"}'
                print('  ' + msg, flush=True)
                failures.append((label, msg))
                continue
            available = list(label_dir.glob('dqn_track_lookahead_ar4_*.json'))
            if not available:
                msg = f'Missing evaluation JSON; check {label_dir / "console.log"}'
                failures.append((label, msg))
                continue
            latest = max(available, key=lambda p: p.stat().st_mtime)
            data = json.loads(latest.read_text(encoding='utf-8'))
            cached.write_text(json.dumps(data, indent=2), encoding='utf-8')
        eps = data.get('episodes', [])
        progress = [float(e['progress']) for e in eps]
        if not progress:
            failures.append((label, 'Evaluation JSON contains no episodes'))
            continue
        ends = [e.get('end', 'unknown') for e in eps]
        row = dict(model=label, path=str(model), episodes=len(eps),
                   mean_progress=sum(progress) / len(progress), max_progress=max(progress),
                   min_progress=min(progress), lap_completions=sum(bool(e.get('lap_completed', False)) for e in eps),
                   wall_crashes=ends.count('wall_crash'), stuck=sum(bool(e.get('stuck_last_10s', False)) for e in eps))
        rows.append(row)
        rows_sorted = sorted(rows, key=lambda r: (-r['mean_progress'], -r['max_progress']))
        with summary_file.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows_sorted)
        print(f"  Mean {row['mean_progress']*100:.2f}% | max {row['max_progress']*100:.2f}% | wall crashes {row['wall_crashes']}/{len(eps)}", flush=True)
    if failures:
        (out / 'failures.txt').write_text('\n'.join(f'{name}: {reason}' for name, reason in failures), encoding='utf-8')
        print(f'WARNING: {len(failures)} evaluations failed. See {out / "failures.txt"}', flush=True)
    if rows:
        best = max(rows, key=lambda r: (r['mean_progress'], r['max_progress']))
        print('\nBest by grid mean:', best['model'], f"{best['mean_progress']*100:.2f}%", flush=True)
    print('Saved comparison:', summary_file, flush=True)


if __name__ == '__main__':
    main()
