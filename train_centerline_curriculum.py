"""Train a stage of centerline-guided DQN without changing observation space.

Stage 1: random starts, centerline-guided V2 reward.
Stage 2: grid starts, reduced guidance; LOAD STAGE 1 WEIGHTS, NEW BUFFER.
Stage 3: grid starts, no guidance; LOAD STAGE 2 WEIGHTS, NEW BUFFER.

This intentionally does not auto-advance stages: inspect identical grid-start
checkpoint evaluations, then explicitly select the next stage.
"""
import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from racecar_gym.envs import gym_api
from Agents.SB3Drivers import SB3Driver, dqn_wrappers
from Agents.wrappers import SimStepCounter, StartModeWrapper, ForwardProgressStallRewardWrapper
from Agents.centerline_curriculum_reward import Guidance, CenterlineGuidanceReward
from train import dqn_scenario, checkpoint_dir, EvalCheckpoint, ResumableCheckpoint, dqn_settings
from Agents.track_features import feature_config


class GuidanceLogCallback(BaseCallback):
    """Sample reward diagnostics in actual training decisions, not raw eval rewards."""
    def __init__(self, output, every=1000):
        super().__init__()
        self.path = Path(output)
        self.every = every
        self.rows = []
        self.aggregate = {}

    def _on_training_start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w', newline='', encoding='utf8')
        self.writer = csv.DictWriter(self.file, fieldnames=[
            'total_decisions','samples','mean_base_reward','mean_forward_reward',
            'mean_guidance_penalty','mean_total_reward','mean_abs_lateral_halfwidth',
            'mean_abs_heading_rad','v2_stall_last_sim_step_share'])
        self.writer.writeheader()
        self.count = 0
        self.totals = dict.fromkeys(self.writer.fieldnames[2:],0.0)

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'curriculum/total_reward' not in info:
                continue
            mapping = {
                'mean_base_reward': info['curriculum/base_reward'],
                'mean_forward_reward': info['curriculum/forward_progress_reward'],
                'mean_guidance_penalty': info['curriculum/guidance_penalty'],
                'mean_total_reward': info['curriculum/total_reward'],
                'mean_abs_lateral_halfwidth': info['curriculum/lateral_abs_halfwidth'],
                'mean_abs_heading_rad': abs(info['curriculum/heading_error_rad']),
                # V2's last simulator step flag is not a full AR4 decision total.
                'v2_stall_last_sim_step_share': float(info.get('reward_v2_stalled',False))
            }
            for key, val in mapping.items():
                self.totals[key] += float(val)
            self.count += 1
        if self.num_timesteps % self.every == 0 and self.count:
            row = {'total_decisions':self.num_timesteps, 'samples':self.count}
            row.update({k: v/self.count for k,v in self.totals.items()})
            self.writer.writerow(row)
            self.file.flush()
            self.count = 0
            self.totals = dict.fromkeys(self.totals, 0.0)
        return True

    def _on_training_end(self):
        self.file.close()


def main():
    p = argparse.ArgumentParser(description='Stage-wise centerline-guided DQN, AR4')
    p.add_argument('--stage', type=int, choices=(1,2,3), required=True)
    p.add_argument('--timesteps', type=int, default=200000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-path', required=True)
    p.add_argument('--source-model', default=None, help='required for stages 2 and 3')
    p.add_argument('--action-repeat', type=int, default=4)
    p.add_argument('--buffer-size', type=int, default=10000)
    p.add_argument('--target-update-interval', type=int, default=2500)
    p.add_argument('--eval-every', type=int, default=10000)
    p.add_argument('--resume-every', type=int, default=50000)
    p.add_argument('--stall-window', type=int, default=40)
    p.add_argument('--stall-penalty', type=float, default=0.03)
    p.add_argument('--collision-penalty', type=float, default=1.0)
    # Confirmed by user's Agents/track_features.py: [offset/1.5, sin(error), cos(error), ...].
    p.add_argument('--lateral-index', type=int, default=0)
    p.add_argument('--heading-index', type=int, default=1, help='sin heading error feature index')
    p.add_argument('--heading-cos-index', type=int, default=2)
    p.add_argument('--lateral-units', type=float, default=1.5)
    p.add_argument('--half-width-m', type=float, default=1.0)
    p.add_argument('--lateral-weight', type=float, default=None)
    p.add_argument('--heading-weight', type=float, default=None)
    p.add_argument('--initial-eps', type=float, default=None,
                   help='exploration initial epsilon, default 1.0 stage 1 and 0.25 transfer')
    p.add_argument('--final-eps', type=float, default=0.05)
    p.add_argument('--exploration-fraction', type=float, default=0.3)
    args = p.parse_args()
    if args.timesteps < 1 or args.action_repeat < 1:
        p.error('timesteps and action-repeat must be positive')
    if args.stage > 1 and not args.source_model:
        p.error('stage 2/3 requires --source-model (previous stage model ZIP)')
    if args.stage == 1 and args.source_model:
        p.error('stage 1 is fresh training: omit --source-model')
    lateral_weight = {1:0.15,2:0.05,3:0.0}[args.stage] if args.lateral_weight is None else args.lateral_weight
    heading_weight = {1:0.15,2:0.05,3:0.0}[args.stage] if args.heading_weight is None else args.heading_weight
    guided = bool(lateral_weight or heading_weight)
    # Fail early if the local checkout changed the established feature order.
    from Agents.track_features import TRACK_FEATURES, LATERAL_SCALE
    if guided and (TRACK_FEATURES[args.lateral_index] != 'lateral_offset'
                   or TRACK_FEATURES[args.heading_index] != 'sin_heading_error'
                   or TRACK_FEATURES[args.heading_cos_index] != 'cos_heading_error'
                   or abs(args.lateral_units - LATERAL_SCALE) > 1e-8):
        p.error('track feature mapping/scaling differs from uploaded track_features.py; stop and review')

    model_path = Path(args.save_path)
    if model_path.suffix == '.zip':
        p.error('use --save-path without .zip (same style as train_v2.py)')
    output = checkpoint_dir(str(model_path))
    if output.joinpath('config.json').exists():
        p.error(f'{output} already contains a run; choose a new save-path, do not overwrite a prior stage')
    mode = 'random' if args.stage == 1 else 'grid'
    raw = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(), render_mode='rgb_array_birds_eye')
    counter = raw = SimStepCounter(raw)
    raw = ForwardProgressStallRewardWrapper(raw,
        progress_scale=100.0, collision_penalty=args.collision_penalty,
        stall_window=args.stall_window, stall_progress_threshold=1e-4,
        stall_speed_threshold=0.2, stall_penalty_per_decision=args.stall_penalty,
        action_repeat=args.action_repeat)
    if mode == 'random':
        raw = StartModeWrapper(raw, mode='random')
    wrappers = dqn_wrappers(action_repeat=args.action_repeat, obs_mode='track-lookahead')
    if guided:
        g = Guidance(lateral_index=args.lateral_index, heading_index=args.heading_index,
            heading_cos_index=args.heading_cos_index, lateral_units=args.lateral_units,
            half_width_m=args.half_width_m, lateral_weight=lateral_weight,
            heading_weight=heading_weight)
        wrappers.append(lambda env: CenterlineGuidanceReward(env, g))
    initial_eps = (1.0 if args.stage == 1 else 0.25) if args.initial_eps is None else args.initial_eps
    if not (0 <= args.final_eps <= initial_eps <= 1):
        p.error('require 0 <= final_eps <= initial_eps <= 1')
    driver = SB3Driver(raw, model_class=DQN, wrappers=wrappers, verbose=1,
        tensorboard_log=str(Path('logs')/'dqn_centerline_curriculum'),
        buffer_size=args.buffer_size, seed=args.seed,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=initial_eps,
        exploration_final_eps=args.final_eps,
        exploration_fraction=args.exploration_fraction)
    try:
        if args.stage > 1:
            # DQN.load makes a FRESH replay buffer with the new environment; do NOT
            # load_replay_buffer() from the previous stage (different reward semantics).
            # Avoid holding two image replay buffers in RAM simultaneously.
            import gc
            del driver.model
            gc.collect()
            driver.model = DQN.load(args.source_model, env=driver.env, device='auto',
                                    tensorboard_log=str(Path('logs')/'dqn_centerline_curriculum'),
                                    force_reset=True)
            if driver.model.buffer_size != args.buffer_size:
                raise ValueError('source model buffer size differs; use consistent --buffer-size')
            driver.model.exploration_initial_eps = initial_eps
            driver.model.exploration_final_eps = args.final_eps
            driver.model.exploration_fraction = args.exploration_fraction
            driver.model.exploration_schedule = get_linear_fn(initial_eps, args.final_eps, args.exploration_fraction)
            driver.model.exploration_rate = initial_eps
            if driver.model.replay_buffer is None or driver.model.replay_buffer.size() != 0:
                raise RuntimeError('expected fresh empty replay buffer for stage transfer')
        else:
            # SB3Driver already configured exploration for stage 1.
            pass
        output.mkdir(parents=True, exist_ok=True)
        config = {
            'stage':args.stage, 'source_model':args.source_model, 'timesteps':args.timesteps,
            'save_path':str(model_path), 'train_start':mode, 'action_repeat':args.action_repeat,
            'obs_mode':'track-lookahead', 'features': feature_config('track-lookahead'),
            'buffer_size':args.buffer_size, 'target_update_interval':args.target_update_interval,
            'stall_window':args.stall_window,'stall_penalty':args.stall_penalty,
            'collision_penalty':args.collision_penalty,
            'geometry_indices_verified_against_track_features':guided,
            'lateral_index':args.lateral_index,'heading_index':args.heading_index,
            'heading_cos_index':args.heading_cos_index,'heading_encoding':'sin-cos','lateral_units':args.lateral_units,
            'half_width_m':args.half_width_m,
            'lateral_weight':lateral_weight,'heading_weight':heading_weight,
            'exploration_initial_eps':initial_eps,'exploration_final_eps':args.final_eps,
            'exploration_fraction':args.exploration_fraction,
            'dqn':dqn_settings(driver.model),
        }
        (output/'config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf8')
        callbacks = [GuidanceLogCallback(output/'reward_diagnostics.csv')]
        if args.eval_every:
            callbacks.append(EvalCheckpoint(output/'eval',args.eval_every,model_path.name))
        if args.resume_every:
            callbacks.append(ResumableCheckpoint(output/'resume',args.resume_every,args.timesteps,counter))
        print('[curriculum]',json.dumps({k:config[k] for k in (
            'stage','train_start','action_repeat','lateral_weight','heading_weight',
            'source_model','timesteps')},indent=2),flush=True)
        driver.train(total_timesteps=args.timesteps,callback=CallbackList(callbacks),reset_num_timesteps=True)
        driver.save(str(model_path))
        config['finished'] = {'decision_steps':driver.model.num_timesteps,
                              'sim_steps':counter.sim_steps,'episodes':counter.episodes}
        (output/'config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf8')
        print('[saved]',str(model_path)+'.zip')
    finally:
        driver.env.close()


if __name__ == '__main__':
    main()
