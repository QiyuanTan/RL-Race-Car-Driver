#!/usr/bin/env python3
"""Train track-lookahead DQN with FTG behavior cloning + optional online RL.

Uses only the successful FTG transition prefix from the prior demonstrator run.
No data recollection, no teacher LiDAR, no edits to train.py or teammates' code.

Run from repository root. Usage: python train_ftg_bc.py --help
"""

import argparse
import gc
import json
from pathlib import Path

from stable_baselines3 import DQN

from Agents.SB3Drivers import SB3Driver, dqn_wrappers
from Agents.demos import DemoKeepingDictReplayBuffer
from Agents.ftg_behavior_cloning import behavior_clone_dqn, check_demo_buffer, project_demo_buffer
from Agents.ftg_geometry import GeometryOnlyObservationWrapper
from Agents.wrappers import SimStepCounter
from racecar_gym.envs import gym_api
from train import dqn_scenario


def checked_output(path):
    path = Path(path)
    if path.suffix != '.zip':
        path = path.with_suffix('.zip')
    if path.exists():
        raise FileExistsError(f'{path} already exists: refusing to overwrite a previous experiment')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description='FTG behavior cloning -> DQN fine tuning (track lookahead)')
    parser.add_argument('--demo-buffer', default='models/checkpoints/jx_dqn_ftg_track_10k/resume/replay_buffer.pkl')
    parser.add_argument('--output-prefix', default='models/jx_dqn_ftg_bc_10k')
    parser.add_argument('--features', choices=['geometry', 'full'], default='geometry',
                        help='geometry = only track/lookahead inputs (recommended)')
    parser.add_argument('--bc-steps', type=int, default=1200)
    parser.add_argument('--bc-batch-size', type=int, default=64)
    parser.add_argument('--motor-loss-weight', type=float, default=0.0,
                        help='strength of added 3-way motor imitation objective')
    parser.add_argument('--motor-balance-power', type=float, default=0.5,
                        help='inverse-frequency motor weight exponent (0..1)')
    parser.add_argument('--steering-loss-weight', type=float, default=0.0,
                        help='extra loss for the five steering levels')
    parser.add_argument('--steering-balance-power', type=float, default=0.5,
                        help='inverse-frequency steering weighting power')
    parser.add_argument('--start-boost-fraction', type=float, default=0.0,
                        help='fraction of BC minibatch deliberately sampled near grid start')
    parser.add_argument('--start-prefix', type=int, default=80,
                        help='number of earliest FTG decisions considered the start')
    parser.add_argument('--rl-timesteps', type=int, default=10000)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--action-repeat', type=int, default=4)
    parser.add_argument('--target-update-interval', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.rl_timesteps < 0:
        parser.error('--rl-timesteps must be >= 0')
    source = Path(args.demo_buffer)
    if not source.is_file():
        parser.error(f'Missing saved FTG replay buffer: {source}')

    prefix = Path(args.output_prefix)
    bc_model_path = checked_output(str(prefix) + '_bc_only.zip')
    final_model_path = (checked_output(str(prefix) + '.zip')
                        if args.rl_timesteps else None)
    report_path = Path(str(prefix) + '_training.json')
    if report_path.exists():
        raise FileExistsError(f'{report_path} already exists: refusing to overwrite results')

    # Re-create the *student-only* environment used during normal evaluation.
    base = gym_api.SingleAgentRaceEnv(
        scenario=dqn_scenario(False), render_mode='rgb_array_birds_eye')
    counter = SimStepCounter(base)
    wrappers = dqn_wrappers(obs_mode='track-lookahead', action_repeat=args.action_repeat,
                            reward_mode='signed-progress')
    if args.features == 'geometry':
        wrappers.append(GeometryOnlyObservationWrapper)
    driver = SB3Driver(
        counter, model_class=DQN,
        wrappers=wrappers,
        verbose=1, buffer_size=args.buffer_size, seed=args.seed,
        target_update_interval=args.target_update_interval,
        # Start near the cloned expert instead of random actions (default epsilon=1).
        exploration_initial_eps=0.15, exploration_final_eps=0.02,
        exploration_fraction=0.5, learning_starts=0,
        replay_buffer_class=DemoKeepingDictReplayBuffer,
        tensorboard_log=str(Path('logs') / 'dqn_track-lookahead_bc'),
    )
    try:
        model = driver.model
        # Restore teacher transitions but NOT the collapsed DQN weights.
        # This model has fresh, seed-controlled network parameters.
        # Drop the empty ~1 GiB replay allocation before unpickling the saved one.
        model.replay_buffer = None
        gc.collect()
        model.load_replay_buffer(str(source))
        project_demo_buffer(model.replay_buffer, model.observation_space)
        gc.collect()
        demos = check_demo_buffer(model)
        rb = model.replay_buffer
        if rb.buffer_size != args.buffer_size:
            raise ValueError(f'Saved replay capacity {rb.buffer_size} != requested '
                             f'--buffer-size {args.buffer_size}')
        # Drop the prior failed model's online replay logically, without copying
        # the ~1 GiB array. Online transitions can now overwrite those slots.
        rb.pos = demos
        rb.full = False
        rb.keep = demos
        print(f'[data] Loaded {demos} successful FTG decisions from {source}; '
              'discarded previous DQN online transitions', flush=True)
        gc.collect()

        bc_report = behavior_clone_dqn(
            model, steps=args.bc_steps, batch_size=args.bc_batch_size,
            seed=args.seed, motor_loss_weight=args.motor_loss_weight,
            motor_balance_power=args.motor_balance_power,
            steering_loss_weight=args.steering_loss_weight,
            steering_balance_power=args.steering_balance_power,
            start_boost_fraction=args.start_boost_fraction,
            start_prefix=args.start_prefix)
        model.save(str(bc_model_path))
        print(f'[bc] saved standalone imitation model: {bc_model_path}', flush=True)

        report = {'base_model': str(source), 'teacher_decisions': demos,
                  'bc': bc_report, 'bc_only_model': str(bc_model_path),
                  'reward_mode': 'signed-progress', 'obs_mode': 'track-lookahead',
                  'student_lidar': False, 'features': args.features,
                  'action_repeat': args.action_repeat,
                  'seed': args.seed, 'rl_timesteps': args.rl_timesteps,
                  'exploration_initial_eps': 0.15, 'exploration_final_eps': 0.02,
                  'exploration_fraction': 0.5, 'learning_starts': 0,
                  'rl_model': str(final_model_path) if final_model_path else None}
        report_path.write_text(json.dumps(report, indent=2))
        if args.rl_timesteps:
            # Online DQN updates begin AFTER behavior cloning.
            # DQN replay buffer continues to preserve demo prefix.
            driver.train(total_timesteps=args.rl_timesteps)
            model.save(str(final_model_path))
            report['online_steps_completed'] = model.num_timesteps
            report_path.write_text(json.dumps(report, indent=2))
            print(f'[rl] saved fine-tuned model: {final_model_path}', flush=True)
    finally:
        driver.env.close()


if __name__ == '__main__':
    main()
