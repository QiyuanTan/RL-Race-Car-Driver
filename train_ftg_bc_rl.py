#!/usr/bin/env python3
"""Fine-tune the *existing* steering-aware BC checkpoint with DQN+BC jointly.

Does NOT retrain BC, overwrite prior models, or edit teammates' FTG files.
Saves short RL checkpoints and selects a winner against the SAME BC baseline.
If no RL checkpoint improves on the BC baseline, the selection stays BC-only.
"""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from Agents.SB3Drivers import dqn_wrappers
from Agents.ftg_bc_regularized_dqn import BCRetainedDQN
from Agents.ftg_behavior_cloning import check_demo_buffer, project_demo_buffer
from Agents.ftg_geometry import GeometryOnlyObservationWrapper
from Agents.wrappers import SimStepCounter
from racecar_gym.envs import gym_api
from train import dqn_scenario


class ModelCheckpointCallback(BaseCallback):
    def __init__(self, every, prefix):
        super().__init__(verbose=0)
        self.every = every
        self.prefix = Path(prefix)
        self.next_step = every
        self.saved = []

    def _on_step(self):
        while self.num_timesteps >= self.next_step:
            path = Path(f'{self.prefix}_step{self.next_step:05d}.zip')
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
            self.model.save(str(path))
            self.saved.append(path)
            print(f'[checkpoint] saved {path}', flush=True)
            self.next_step += self.every
        return True


def run_evaluation(model_path, eval_dir, episodes, seed, max_seconds):
    eval_dir.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, 'evaluate_ftg_bc.py',
           '--model', str(model_path), '--features', 'geometry',
           '--episodes', str(episodes), '--seed', str(seed),
           '--action-repeat', '4', '--max-sim-time', str(max_seconds),
           '--out', str(eval_dir)]
    print(f'[eval] {model_path.name}: {episodes} seeds, {max_seconds:g}s limit', flush=True)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        print(result.stdout, flush=True)
        raise RuntimeError(f'Evaluation failed for {model_path}')
    results = list(eval_dir.glob('ftg_bc_*.json'))
    if len(results) != 1:
        raise RuntimeError(f'Expected one evaluation JSON in {eval_dir}')
    data = json.loads(results[0].read_text())
    summary = data['summary']
    progress = [round(x['progress'] * 100, 1) for x in data['episodes']]
    print(f'[eval] mean={summary["mean_progress"] * 100:.1f}% '
          f'per-seed={progress}, laps={summary["lap_rate"]:.0%}', flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description='Safeguarded DQN fine-tuning with retained FTG BC')
    p.add_argument('--base-model', default='models/jx_dqn_ftg_steeringaware_2500_bc_only.zip')
    p.add_argument('--demo-buffer', default='models/checkpoints/jx_dqn_ftg_track_10k/resume/replay_buffer.pkl')
    p.add_argument('--output-prefix', default='models/jx_dqn_ftg_bc_regularized_2k')
    p.add_argument('--rl-timesteps', type=int, default=2000)
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--bc-weight-start', type=float, default=0.15)
    p.add_argument('--bc-weight-end', type=float, default=0.10)
    p.add_argument('--bc-batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-seeds', type=int, default=3)
    p.add_argument('--eval-start-seed', type=int, default=100)
    p.add_argument('--eval-max-sim-time', type=float, default=15.0)
    p.add_argument('--min-improvement', type=float, default=0.01,
                   help='Absolute track-progress improvement needed to select RL over BC')
    args = p.parse_args()

    if not Path('Agents/SB3Drivers.py').is_file() or not Path('evaluate_ftg_bc.py').is_file():
        p.error('Run from the RL-Race-Car-Driver repository root')
    for name in ('base_model', 'demo_buffer'):
        if not Path(getattr(args, name)).is_file():
            p.error(f'File not found: {getattr(args, name)}')
    if not (0 < args.rl_timesteps and 0 < args.checkpoint_every and
            args.rl_timesteps % args.checkpoint_every == 0):
        p.error('Steps must be positive and rl-timesteps divisible by checkpoint-every')
    if (not 0 < args.learning_rate <= 1e-3 or args.eval_seeds < 1 or
            not 0 < args.eval_max_sim_time <= 120 or args.min_improvement < 0):
        p.error('Invalid learning rate, evaluation settings, or min-improvement')
    if args.bc_weight_start < 0 or args.bc_weight_end < 0 or args.bc_batch_size < 1:
        p.error('Invalid BC settings')

    prefix = Path(args.output_prefix)
    checkpoints = [Path(f'{prefix}_step{step:05d}.zip') for step in
                   range(args.checkpoint_every, args.rl_timesteps + 1, args.checkpoint_every)]
    report_file = Path(f'{prefix}_selection.json')
    eval_root = Path('/tmp') / f'{prefix.name}_evaluation'
    for f in checkpoints + [report_file, eval_root]:
        if f.exists():
            p.error(f'Output already exists (no overwrite): {f}')
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Match the exact geometry-only student environment and signed reward.
    base = gym_api.SingleAgentRaceEnv(
        scenario=dqn_scenario(False), render_mode='rgb_array_birds_eye')
    env = SimStepCounter(base)
    for wrapper in dqn_wrappers(obs_mode='track-lookahead', action_repeat=4,
                                reward_mode='signed-progress'):
        env = wrapper(env)
    env = GeometryOnlyObservationWrapper(env)
    model = None
    try:
        model = BCRetainedDQN.load(
            args.base_model, env=env, device='cpu',
            learning_rate=args.learning_rate, target_update_interval=1000,
            learning_starts=0, exploration_initial_eps=0.05,
            exploration_final_eps=0.02, exploration_fraction=0.25,
            tensorboard_log='logs/dqn_track-lookahead_bc_regularized', verbose=1)
        # Avoid simultaneously allocating the saved camera replay and an empty
        # replacement replay buffer; the saved teacher prefix is pre-existing.
        model.replay_buffer = None
        gc.collect()
        model.load_replay_buffer(args.demo_buffer)
        project_demo_buffer(model.replay_buffer, model.observation_space)
        keep = check_demo_buffer(model)
        buffer = model.replay_buffer
        if keep >= buffer.buffer_size:
            raise ValueError('Demo buffer is full, cannot collect new RL transitions')
        buffer.pos, buffer.full, buffer.keep = keep, False, keep
        model.configure_bc(seed=args.seed,
                           coefficient_start=args.bc_weight_start,
                           coefficient_end=args.bc_weight_end,
                           demo_batch_size=args.bc_batch_size)
        print(f'[model] loaded BC-only checkpoint: {args.base_model}', flush=True)
        print(f'[model] retained {keep} demonstrations; optimizing TD + BC together', flush=True)
        cb = ModelCheckpointCallback(args.checkpoint_every, prefix)
        model.learn(total_timesteps=args.rl_timesteps,
                    callback=cb, progress_bar=False)
        if cb.saved != checkpoints:
            raise RuntimeError(f'Missing checkpoints: {cb.saved} != {checkpoints}')
    finally:
        if model is not None:
            model.env.close()
        else:
            env.close()
        del model
        gc.collect()

    eval_root.mkdir(parents=True, exist_ok=False)
    candidates = [Path(args.base_model)] + checkpoints
    evaluations = []
    for index, path in enumerate(candidates):
        summary = run_evaluation(path, eval_root / f'candidate_{index:02d}',
                                 args.eval_seeds, args.eval_start_seed,
                                 args.eval_max_sim_time)
        evaluations.append({'model': str(path), 'mean_progress': summary['mean_progress'],
                            'lap_rate': summary['lap_rate'],
                            'progress_min': summary['progress_min'],
                            'progress_max': summary['progress_max']})
    bc_progress = evaluations[0]['mean_progress']
    rl_best = max(evaluations[1:], key=lambda entry: entry['mean_progress'])
    selected = (rl_best if rl_best['mean_progress'] >= bc_progress + args.min_improvement
                else evaluations[0])
    report = {
        'method': 'SB3 DQN Bellman TD loss + retained FTG BC loss, same optimizer step',
        'base_checkpoint': args.base_model,
        'source_replay': args.demo_buffer,
        'settings': vars(args),
        'candidates': evaluations,
        'base_progress': bc_progress,
        'best_rl_progress': rl_best['mean_progress'],
        'selected_model': selected['model'],
        'selected_is_rl': selected is not evaluations[0],
        'selection_rule': 'RL must improve BC on SAME eval seeds by min-improvement',
        'warning': 'Selection on only a few seeds is exploratory; verify on 10+ new seeds.',
    }
    report_file.write_text(json.dumps(report, indent=2))
    print('\n[selection] BC baseline: '
          f'{bc_progress * 100:.1f}%; best RL: {rl_best["mean_progress"] * 100:.1f}%',
          flush=True)
    print(f'[selection] SELECTED: {selected["model"]}', flush=True)
    if not report['selected_is_rl']:
        print('[selection] No RL checkpoint cleared the improvement threshold; '
              'keeping original BC-only model. No false improvement claim.', flush=True)
    print(f'[selection] report: {report_file}', flush=True)


if __name__ == '__main__':
    main()
