"""Final DQN mixed-start curriculum. Put beside train_v2.py; no source changes required.

Train with Stage 1 network weights, a fresh optimizer and replay buffer. Only
Hairpin episodes end at the local sector goal; grid/random episodes are full runs.
"""
import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from racecar_gym.envs import gym_api
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn

from Agents.SB3Drivers import SB3Driver, dqn_wrappers
from Agents.track_features import Centerline, feature_config
from Agents.wrappers import SimStepCounter, ForwardProgressRewardWrapper
from train_v2 import dqn_scenario, checkpoint_dir, EvalCheckpoint, ResumableCheckpoint, dqn_settings


class SustainedStallReward(ForwardProgressRewardWrapper):
    """Per-simulator-step progress reward, conservative stall detection and timeout.

    This wrapper MUST be inside ActionRepeatWrapper. A 4-step action gives up to
    stall_penalty_per_decision total penalty, not 4x that penalty.

    A low-speed stall requires a full net-progress window AND low speed, then
    an optional extra grace period near the hairpin. Slow but moving is safe.
    A long-term no-progress fallback catches back-and-forth oscillation.
    """
    def __init__(self, env, progress_scale=100., collision_penalty=1.,
                 action_repeat=4, stall_window_steps=60,
                 stall_progress_threshold=1e-4, stall_speed_threshold=0.2,
                 stall_penalty_per_decision=0.03, stall_timeout_steps=200,
                 oscillation_timeout_steps=250, curve_from=0.28,
                 curve_until=0.42, curve_grace_steps=40):
        super().__init__(env, progress_scale=progress_scale,
                         collision_penalty=collision_penalty)
        if action_repeat < 1 or stall_window_steps < 1 or stall_timeout_steps < 1:
            raise ValueError('Invalid action repeat or stall window/timeout')
        self.window = deque(maxlen=stall_window_steps)
        self.window_size = stall_window_steps
        self.threshold = float(stall_progress_threshold)
        self.speed_threshold = float(stall_speed_threshold)
        self.penalty_step = float(stall_penalty_per_decision) / action_repeat
        self.stall_timeout = stall_timeout_steps
        self.oscillation_timeout = oscillation_timeout_steps
        self.curve_from, self.curve_until = curve_from, curve_until
        self.curve_grace = curve_grace_steps
        self.reset_counters()

    def reset_counters(self):
        self.window.clear()
        self.low_stall_streak = 0
        self.no_progress_streak = 0
        self.active_stall_streak = 0
        self.stall_sim_steps = 0
        self.total_stall_penalty = 0.
        self.stuck_truncations = 0
        self.total_signed_progress = 0.
        self.total_forward_progress = 0.

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.reset_counters()
        return obs, info

    def step(self, action):
        before = self._previous_progress
        obs, reward, terminated, truncated, raw_info = super().step(action)
        info = dict(raw_info)
        delta = self._previous_progress - before
        if not (math.isfinite(delta) and abs(delta) <= 0.5):
            delta = 0.
        self.total_signed_progress += delta
        self.total_forward_progress += max(0., delta)
        self.window.append(delta)
        progress_window = sum(self.window)
        progress = float(info.get('progress', float('nan')))
        velocity = np.asarray(info['velocity'],dtype=float).reshape(-1)
        speed = float(np.hypot(velocity[0],velocity[1]))
        # Both conditions are necessary for a low-speed stall. Use signed net
        # movement, so reversing and regaining the same segment is not progress.
        stagnant = (len(self.window) == self.window_size and
                    progress_window <= self.threshold)
        if stagnant and math.isfinite(speed) and speed < self.speed_threshold:
            self.low_stall_streak += 1
        else:
            self.low_stall_streak = 0
        if stagnant:
            self.no_progress_streak += 1
        else:
            self.no_progress_streak = 0
        in_curve = self.curve_from <= progress <= self.curve_until
        # Give extra grace near the sharp hairpin; never penalize one or two
        # isolated braking decisions.
        low_speed_active = self.low_stall_streak > (self.curve_grace if in_curve else 0)
        oscillating_active = self.no_progress_streak >= self.oscillation_timeout
        active = low_speed_active or oscillating_active
        penalty = self.penalty_step if active and not (terminated or truncated) else 0.
        if active:
            self.active_stall_streak += 1
        else:
            self.active_stall_streak = 0
        if penalty:
            self.total_stall_penalty += penalty
            self.stall_sim_steps += 1
        stuck = self.active_stall_streak >= self.stall_timeout and not (terminated or truncated)
        if stuck:
            truncated = True
            self.stuck_truncations += 1
        # Gymnasium convention: timeout is a truncation, distinct from collision.
        info.update({
            'final/stalled': bool(active),
            'final/stuck_timeout': bool(stuck),
            'final/stall_penalty_sim_step': -float(penalty),
            'final/stall_sim_steps': self.stall_sim_steps,
            'final/stall_penalty_total': -self.total_stall_penalty,
            'final/stall_streak_steps': self.active_stall_streak,
            'final/no_progress_streak_steps': self.no_progress_streak,
            'final/rolling_net_progress': progress_window,
        })
        return obs, float(reward-penalty), terminated, truncated, info


class MixedStart(gym.Wrapper):
    def __init__(self, env, seed=0, grid_prob=0.40, sector_prob=0.35,
                 start_min=0.26, start_max=0.31, max_attempts=600,
                 target_progress=0.42):
        super().__init__(env)
        self.rng = np.random.default_rng(seed + 91721)
        self.grid_prob = grid_prob
        self.sector_prob = sector_prob
        self.start_min, self.start_max = start_min, start_max
        self.max_attempts = max_attempts
        self.target_progress = target_progress
        self.kind = None
        self.start_progress = None
        self.reset_attempts = 0
        self.total_attempts = 0
        self.episode_count = 0

    def reset(self, *, seed=None, options=None):
        # We explicitly do NOT reseed the RNG on each rejection. This is the
        # same-env sampling path that produced distinct starts in diagnostics.
        self.episode_count += 1
        r = self.rng.random()
        self.kind = 'grid' if r < self.grid_prob else ('hairpin' if r < self.grid_prob + self.sector_prob else 'random')
        mode = 'grid' if self.kind == 'grid' else 'random'
        self.reset_attempts = 0
        for attempt in range(1 if self.kind != 'hairpin' else self.max_attempts):
            obs, info = self.env.reset(seed=seed if attempt == 0 else None, options={'mode':mode})
            self.total_attempts += 1
            self.reset_attempts = attempt + 1
            p = float(info['progress'])
            if self.kind != 'hairpin' or (math.isfinite(p) and self.start_min <= p <= self.start_max):
                self.start_progress = p
                info = dict(info)
                info.update({'final/start_mode': self.kind, 'final/start_progress': p,
                             'final/reset_attempts': attempt + 1})
                return obs, info
        raise RuntimeError('Hairpin start rejection sampling exhausted; save run and investigate RNG')


class EpisodeAndSector(gym.Wrapper):
    """Tracks start and max progress at simulator-step granularity; local goal only for hairpin."""
    def __init__(self, env, mixed, stall, target_progress=0.42, success_bonus=1.0):
        super().__init__(env)
        self.mixed = mixed
        self.stall = stall
        self.target_progress = target_progress
        self.success_bonus = success_bonus
        self.finished = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.start_info = dict(info)
        self.max_progress = float(info['progress'])
        self.episode_reward = 0.0
        self.sim_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.sim_steps += 1
        info = dict(info)
        p = float(info['progress'])
        self.max_progress = max(self.max_progress, p)
        success = (self.mixed.kind == 'hairpin' and not done and not truncated and
                   not bool(info.get('wall_collision')) and
                   self.target_progress <= p < 0.95)
        if success:
            done = True
            reward = float(reward) + self.success_bonus
        self.episode_reward += float(reward)
        info['final/sector_success'] = bool(success)
        if done or truncated:
            pose = np.asarray(info.get('pose', np.zeros(6))).reshape(-1)
            vel = np.asarray(info.get('velocity', np.zeros(6))).reshape(-1)
            end = ('sector_success' if success else 'wall_crash' if bool(info.get('wall_collision'))
                   else 'stuck' if bool(info.get('final/stuck_timeout'))
                   else ('truncated' if truncated else 'terminated'))
            ep = {
                'episode_id':len(self.finished), 'start_mode':self.mixed.kind,
                'start_progress':self.mixed.start_progress, 'reset_attempts':self.mixed.reset_attempts,
                'max_progress':self.max_progress,'final_progress':p,'end_reason':end,
                'sim_steps':self.sim_steps,'sim_time':float(info.get('time',float('nan'))),
                'x':float(pose[0]), 'y':float(pose[1]),
                'end_speed':float(np.hypot(vel[0],vel[1])),
                'episode_reward_raw':self.episode_reward,
                'stall_sim_steps':self.stall.stall_sim_steps,
                'stall_penalty_total':-self.stall.total_stall_penalty,
                'stuck_timeout':bool(info.get('final/stuck_timeout')),
                'stall_streak_at_end':self.stall.active_stall_streak,
            }
            self.finished.append(ep)
            info['final/episode'] = ep
        return obs, float(reward), done, truncated, info


class CoordinatedReward(gym.Wrapper):
    """Decision-level geometry/speed/smoothness penalties, progress remains dominant."""
    def __init__(self, env, lateral_weight=0.12, heading_weight=0.12,
                 speed_weight=0.018, steering_smooth=0.004, motor_smooth=0.002,
                 curve_from=0.29, curve_until=0.42, curve_speed=1.0):
        super().__init__(env)
        self.weights = dict(lateral=lateral_weight, heading=heading_weight,
                            speed=speed_weight, steer=steering_smooth, motor=motor_smooth)
        self.curve_from, self.curve_until, self.curve_speed = curve_from, curve_until, curve_speed
        self.last = None
        self.prev_progress = None
        self.start_mode = None
        self.start_progress = None
        self.episode_total = 0.
        self.episode_components = dict.fromkeys(['forward', 'geometry', 'overspeed', 'smooth','stall','base','total'], 0.)
        self.decision_index = 0
        self.decision_logs = []
        self.episode_logs = []
        self.completed_episodes = 0
        self.last_stall_total = 0.

    @staticmethod
    def action_values(action):
        idx = int(np.asarray(action).reshape(-1)[0])
        if not 0 <= idx < 15:
            raise ValueError(f'Unexpected action id {idx}; expected original 15 actions')
        return (-0.5, 0., 0.5)[idx // 5], (-1., -0.5, 0., 0.5, 1.)[idx % 5]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_progress = float(info['lap']) + float(info['progress'])
        self.start_mode = info.get('final/start_mode')
        self.start_progress = info.get('final/start_progress')
        self.last = None
        self.last_stall_total = 0.
        self.episode_components = dict.fromkeys(self.episode_components, 0.)
        self.decision_index = 0
        return obs, info

    def step(self, action):
        motor, steer = self.action_values(action)
        obs, base, done, truncated, info = self.env.step(action)
        info = dict(info)
        progress = float(info['progress'])
        absolute = float(info['lap']) + progress
        dp = absolute - self.prev_progress
        self.prev_progress = absolute
        forward = 100 * max(0.,dp) if math.isfinite(dp) and abs(dp) <= 0.5 else 0.
        track = np.asarray(obs['track']).reshape(-1)
        lateral = abs(float(track[0])*1.5)
        heading = math.atan2(float(track[1]),float(track[2]))
        speed = float(np.hypot(*np.asarray(info['velocity']).reshape(-1)[:2]))
        geometry = -forward * (self.weights['lateral'] * min(lateral,1.) +
                              self.weights['heading'] * (1-math.cos(heading))/2.)
        # Gentle penalty for overspeed in approach/hairpin. This is intentionally
        # capped to avoid rewarding a stationary policy more than forward driving.
        in_curve = self.curve_from <= progress <= self.curve_until
        overspeed = -min(0.035, self.weights['speed'] * max(0.,speed-self.curve_speed)**2) if in_curve else 0.
        smooth = 0.
        if self.last is not None:
            smooth = -self.weights['steer'] * abs(steer-self.last[1]) - self.weights['motor'] * abs(motor-self.last[0])
        self.last = (motor,steer)
        stall_total = (float(info.get('final/stall_penalty_total', 0.)))
        stall = stall_total - self.last_stall_total
        self.last_stall_total = stall_total
        total = float(base) + geometry + overspeed + smooth
        # 'base' already includes stall; stall is only a separated diagnostic.
        values = dict(base=float(base),forward=forward,geometry=geometry,
                      overspeed=overspeed,smooth=smooth,stall=stall,total=total)
        for k,v in values.items():
            self.episode_components[k] += v
            info['final/reward_'+k] = v
        self.decision_index += 1
        if 0.28 <= progress <= 0.42:
            self.decision_logs.append({'episode_id':self.completed_episodes, 'start_mode':self.start_mode,
                'decision_index':self.decision_index,'progress':progress,'speed':speed,'lateral_m':float(track[0])*1.5,
                'heading_rad':heading,'motor':motor,'steering':steer,
                'stall_active':bool(info.get('final/stalled')),
                'stall_streak_steps':int(info.get('final/stall_streak_steps',0)),
                'rolling_net_progress':float(info.get('final/rolling_net_progress',0.)), **values})
        if done or truncated:
            record = dict(info.get('final/episode', {}))
            record.update({f'reward_{k}':v for k,v in self.episode_components.items()})
            self.episode_logs.append(record)
            self.completed_episodes += 1
        return obs,total,done,truncated,info


class LogsAndCheckpoint(BaseCallback):
    def __init__(self, folder, rewards, every=1000):
        super().__init__()
        self.folder = Path(folder)
        self.rewards = rewards
        self.every = every
        self.last_episode = 0
        self.last_decision = 0

    @staticmethod
    def append_csv(path, rows):
        if not rows: return
        path.parent.mkdir(parents=True,exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        with path.open('a',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()), extrasaction='ignore')
            if is_new: w.writeheader()
            w.writerows(rows)

    def flush(self):
        self.append_csv(self.folder/'episode_diagnostics.csv',self.rewards.episode_logs[self.last_episode:])
        self.last_episode = len(self.rewards.episode_logs)
        self.append_csv(self.folder/'hairpin_decisions.csv',self.rewards.decision_logs[self.last_decision:])
        self.last_decision = len(self.rewards.decision_logs)
        # Cap resident decision logs for a long training run.
        if self.last_decision > 10000:
            self.rewards.decision_logs.clear()
            self.last_decision=0
        if self.last_episode > 500:
            self.rewards.episode_logs.clear()
            self.last_episode=0

    def _on_step(self):
        if self.num_timesteps % self.every == 0: self.flush()
        return True
    def _on_training_end(self): self.flush()


def main():
    p=argparse.ArgumentParser(description='Final DQN coordinated reward and mixed-start training')
    p.add_argument('--source-model',required=True)
    p.add_argument('--save-path',required=True)
    p.add_argument('--timesteps',type=int,default=300000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--action-repeat',type=int,default=4)
    p.add_argument('--buffer-size',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--target-update-interval',type=int,default=5000)
    p.add_argument('--initial-eps',type=float,default=0.15)
    p.add_argument('--final-eps',type=float,default=0.03)
    p.add_argument('--exploration-fraction',type=float,default=0.4)
    p.add_argument('--grid-prob',type=float,default=0.40)
    p.add_argument('--hairpin-prob',type=float,default=0.35)
    p.add_argument('--start-min',type=float,default=0.26)
    p.add_argument('--start-max',type=float,default=0.31)
    p.add_argument('--target-progress',type=float,default=0.42)
    p.add_argument('--max-reset-attempts',type=int,default=600)
    p.add_argument('--eval-every',type=int,default=25000)
    p.add_argument('--resume-every',type=int,default=10000)
    p.add_argument('--lateral-weight',type=float,default=0.12)
    p.add_argument('--heading-weight',type=float,default=0.12)
    p.add_argument('--speed-weight',type=float,default=0.018)
    p.add_argument('--steer-smooth-weight',type=float,default=0.004)
    p.add_argument('--motor-smooth-weight',type=float,default=0.002)
    p.add_argument('--curve-speed',type=float,default=1.0)
    p.add_argument('--collision-penalty',type=float,default=1.0)
    p.add_argument('--stall-penalty',type=float,default=0.03)
    p.add_argument('--stall-window-steps',type=int,default=60)
    p.add_argument('--stall-speed-threshold',type=float,default=0.2)
    p.add_argument('--stall-progress-threshold',type=float,default=1e-4)
    p.add_argument('--stall-timeout-steps',type=int,default=200)
    p.add_argument('--oscillation-timeout-steps',type=int,default=250)
    p.add_argument('--curve-grace-steps',type=int,default=40)
    args=p.parse_args()
    if not (0 <= args.grid_prob and 0 <= args.hairpin_prob and args.grid_prob + args.hairpin_prob <= 1):p.error('start probabilities must sum to <=1')
    if not (0 <= args.start_min < args.start_max < args.target_progress < 1):p.error('invalid sector range')
    if (args.stall_window_steps < 1 or args.stall_timeout_steps < 1 or
        args.oscillation_timeout_steps < 1 or args.curve_grace_steps < 0 or
        args.stall_speed_threshold < 0 or args.stall_progress_threshold < 0 or
        args.stall_penalty < 0):
        p.error('invalid stall configuration: thresholds/penalty must be nonnegative, windows/timeouts positive')
    if args.action_repeat != 4:p.error('final model preserves original action-repeat=4')
    if not Path(args.source_model).is_file():p.error('source model is missing')
    path=Path(args.save_path)
    if path.suffix=='.zip': p.error('save-path must omit .zip')
    output=checkpoint_dir(str(path))
    if (output/'config.json').exists():p.error(f'run already exists: {output}. Choose new save-path')
    source_config=checkpoint_dir(str(Path(args.source_model).with_suffix('')))/'config.json'
    if not source_config.exists():p.error(f'source config missing: {source_config}')
    old=json.loads(source_config.read_text(encoding='utf8'))
    # Accept both historical metadata names; normalize only OS path separators.
    # Never bypass feature ordering, scaling, centerline hash, or AR compatibility.
    source_features = old.get('features', old.get('feature_config'))
    current_features = feature_config('track-lookahead')
    def normalized_feature_metadata(obj):
        if not isinstance(obj, dict):
            return obj
        obj = dict(obj)
        for key in ('centerline_file', 'centerline_source'):
            if isinstance(obj.get(key), str):
                obj[key] = obj[key].replace('\\', '/')
        return obj
    old_ar = old.get('action_repeat')
    if old_ar != args.action_repeat:
        p.error(f'source action_repeat={old_ar!r} != requested {args.action_repeat!r}; cannot transfer')
    if normalized_feature_metadata(source_features) != normalized_feature_metadata(current_features):
        import difflib
        previous = json.dumps(normalized_feature_metadata(source_features), indent=2, sort_keys=True, ensure_ascii=False)
        current = json.dumps(normalized_feature_metadata(current_features), indent=2, sort_keys=True, ensure_ascii=False)
        print('SOURCE FEATURE CONFIG:')
        print(previous)
        print('CURRENT FEATURE CONFIG:')
        print(current)
        print('DIFFERENCE:')
        print('\n'.join(difflib.unified_diff(previous.splitlines(), current.splitlines(),
                                             fromfile='source', tofile='current', lineterm='')))
        p.error('source feature configuration truly differs; verify centerline/features before transfer')
    raw=gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(),render_mode='rgb_array_birds_eye')
    mixed=MixedStart(raw,args.seed,args.grid_prob,args.hairpin_prob,args.start_min,args.start_max,args.max_reset_attempts,args.target_progress)
    counter=SimStepCounter(mixed)
    shaped=SustainedStallReward(counter, progress_scale=100.,collision_penalty=args.collision_penalty,
        action_repeat=args.action_repeat,stall_window_steps=args.stall_window_steps,
        stall_progress_threshold=args.stall_progress_threshold,
        stall_speed_threshold=args.stall_speed_threshold,
        stall_penalty_per_decision=args.stall_penalty,
        stall_timeout_steps=args.stall_timeout_steps,
        oscillation_timeout_steps=args.oscillation_timeout_steps,
        curve_grace_steps=args.curve_grace_steps)
    episode=EpisodeAndSector(shaped,mixed,shaped,target_progress=args.target_progress,success_bonus=1.0)
    driver=SB3Driver(episode,model_class=DQN,wrappers=dqn_wrappers(action_repeat=4,obs_mode='track-lookahead'),
        verbose=1,tensorboard_log=str(Path('logs')/'dqn_final_run'),seed=args.seed,
        buffer_size=args.buffer_size,batch_size=args.batch_size,learning_rate=args.learning_rate,
        target_update_interval=args.target_update_interval,exploration_initial_eps=args.initial_eps,
        exploration_final_eps=args.final_eps,exploration_fraction=args.exploration_fraction)
    reward_wrapper=CoordinatedReward(driver.env, lateral_weight=args.lateral_weight,heading_weight=args.heading_weight,
        speed_weight=args.speed_weight,steering_smooth=args.steer_smooth_weight,motor_smooth=args.motor_smooth_weight,
        curve_speed=args.curve_speed)
    driver.env=reward_wrapper
    driver.model.set_env(reward_wrapper, force_reset=True)
    try:
        # Network transfer only. Fresh optimizer, replay, schedule, timestep count.
        source=DQN.load(args.source_model,device='cpu')
        if dict(source.observation_space.spaces) != dict(driver.model.observation_space.spaces):
            p.error('source and final observation spaces differ')
        if source.action_space.n != driver.model.action_space.n:
            p.error('source and final action action spaces differ')
        driver.model.policy.q_net.load_state_dict(source.policy.q_net.state_dict())
        driver.model.policy.q_net_target.load_state_dict(source.policy.q_net_target.state_dict())
        del source
        output.mkdir(parents=True,exist_ok=True)
        cfg={'stage':'final','source_model':args.source_model,'timesteps':args.timesteps,'seed':args.seed,
            'save_path':str(path),'obs_mode':'track-lookahead','action_repeat':4,
            'features':feature_config('track-lookahead'), 'dqn':dqn_settings(driver.model),
            'reward':'progress + sustained stall/stuck timeout + decision-level geometry/curve overspeed/action smoothness',
            'start':'mixed grid/hairpin/random','parameters':vars(args),'transfer_replay_buffer':False,
            'transfer_optimizer':False}
        (output/'config.json').write_text(json.dumps(cfg,indent=2,default=str),encoding='utf8')
        callbacks=[LogsAndCheckpoint(output,reward_wrapper)]
        if args.eval_every:callbacks.append(EvalCheckpoint(output/'eval',args.eval_every,path.name))
        if args.resume_every:callbacks.append(ResumableCheckpoint(output/'resume',args.resume_every,args.timesteps,counter))
        driver.train(total_timesteps=args.timesteps,callback=CallbackList(callbacks),reset_num_timesteps=True)
        driver.save(str(path))
        cfg['finished']={'decisions':driver.model.num_timesteps,'sim_steps':counter.sim_steps,
                         'episodes':mixed.episode_count}
        (output/'config.json').write_text(json.dumps(cfg,indent=2,default=str),encoding='utf8')
        print('[saved]',str(path)+'.zip')
    finally:
        driver.env.close()

if __name__=='__main__':
    main()
