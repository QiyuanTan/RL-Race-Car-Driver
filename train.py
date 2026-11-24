import argparse
import json
import os
import subprocess
from functools import partial
from pathlib import Path

from Agents.SB3Drivers import SB3Driver, SB3ObservationWrapper, dqn_wrappers, resolve_obs_mode
from Agents.track_features import feature_config
from Agents.demos import DemoKeepingDictReplayBuffer, collect_ftg_demos
from Agents.dqfd import MarginDQN
from Agents.demos import DemoMixDictReplayBuffer, collect_takeover_demos
from Agents.ftg_track_demos import collect_ftg_track_demos
from Agents.wrappers2 import FTGTeacherLidarWrapper
from Agents.wrappers import (FlattenActionWrapper, DiscreteActionWrapper, CrashPenaltyWrapper, HairpinStartWrapper,
                             SimStepCounter, StartModeWrapper, ForwardProgressRewardWrapper)
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from racecar_gym.envs import gym_api

def train_ppo(env, total_timesteps=10000, save_path='ppo_model'):
    log_dir = Path(__file__).resolve().parent / 'logs' / 'ppo'
    wrappers = [FlattenActionWrapper, SB3ObservationWrapper]
    driver = SB3Driver(env, model_class=PPO, wrappers=wrappers, verbose=1, tensorboard_log=str(log_dir))
    try:
        driver.train(total_timesteps=total_timesteps)
        driver.save(save_path)
    finally:
        driver.env.close()


REPO_ROOT = Path(__file__).resolve().parent


def dqn_scenario(lidar=False):
    """Scenario file for DQN. The LiDAR variant only adds the `lidar` sensor.
    The track modes use the base scenario (they only need the pose and velocity sensors)."""
    name = 'austria_lidar.yml' if lidar else 'austria.yml'
    return str(REPO_ROOT / 'scenarios' / name)


def git_commit():
    """Current commit and whether there were uncommitted changes (None outside a git checkout)."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                             cwd=REPO_ROOT, text=True).strip())
        return {'commit': commit, 'uncommitted_changes': dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def dqn_settings(model):
    """The DQN hyperparameters as the model actually has them (identical for every obs mode)."""
    settings = {key: getattr(model, key) for key in (
        'learning_rate', 'buffer_size', 'learning_starts', 'batch_size', 'gamma', 'gradient_steps',
        'target_update_interval', 'exploration_fraction', 'exploration_initial_eps', 'exploration_final_eps')}
    settings.update(train_freq=str(model.train_freq), net_arch=str(model.policy.net_arch),
                    features_extractor=type(model.policy.q_net.features_extractor).__name__,
                    features_dim=int(model.policy.q_net.features_extractor.features_dim))
    return settings


class EvalCheckpoint(BaseCallback):
    """Every `every` steps, save the model (no replay buffer) as <prefix>_<step>_steps.zip.

    Counts with the model's total step counter, so the files stay on multiples of `every`
    after a resume too (SB3's CheckpointCallback restarts its own counter).
    """

    def __init__(self, folder, every, prefix):
        super().__init__()
        self.folder = Path(folder)
        self.every = every
        self.prefix = prefix

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self.folder.mkdir(parents=True, exist_ok=True)
            self.model.save(self.folder / f'{self.prefix}_{self.num_timesteps}_steps')
        return True


class ResumableCheckpoint(BaseCallback):
    """Every steps, save model + replay buffer so training can be resumed.

    Only the latest copy is kept: it is written to a temporary folder first and then moved into
    place to avoid any interruptions
    """

    def __init__(self, folder, every, total_timesteps, counter=None, sim_steps_before=0):
        super().__init__()
        self.folder = Path(folder)
        self.every = every
        self.total_timesteps = total_timesteps
        self.counter = counter                    # SimStepCounter, to store the real simulator steps
        self.sim_steps_before = sim_steps_before  # steps done before a resume

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self.save()
        return True

    def save(self):
        tmp = self.folder / 'tmp'
        tmp.mkdir(parents=True, exist_ok=True)
        self.model.save(tmp / 'model')  # weights, optimizer, step counters
        self.model.save_replay_buffer(tmp / 'replay_buffer') # not part of model.zip
        (tmp / 'state.json').write_text(json.dumps(
            {'num_timesteps': self.num_timesteps, 'total_timesteps': self.total_timesteps,
             'simulated_steps': self.sim_steps_before + self.counter.sim_steps if self.counter else None}))
        for name in ('model.zip', 'replay_buffer.pkl', 'state.json'): # state.json last
            os.replace(tmp / name, self.folder / name)
        print(f'[checkpoint] resumable copy saved at step {self.num_timesteps} -> {self.folder}', flush=True)


def checkpoint_dir(save_path):
    """models/dqn_model -> models/checkpoints/dqn_model (one folder per run, so variants never mix)."""
    save_path = Path(save_path)
    return save_path.parent / 'checkpoints' / save_path.name


def train_dqn(env, total_timesteps=10000, save_path='dqn_model', buffer_size=10_000,
              lidar=False, seed=None, eval_every=0, resume_every=0, resume=False,
              action_repeat=1, target_update_interval=10_000, train_start='grid', demo_episodes=0,
              demo_laps=0, obs_mode=None, reward_mode='original', hairpin_share=0.5, init_from=None,
              crash_penalty=0.0, takeover_demos=0, takeover_points=(0.24, 0.26, 0.28), demo_fraction=0.2,
              margin_loss=0.0, demo_pretrain_steps=0, no_camera=False, collision_penalty=1.0):
    """Train DQN with a bounded replay buffer, independent of training duration.

    At 128x128 RGB uint8, 10,000 transitions use about 0.92 GiB for current
    and next images alone; other observations and training need extra memory.

    obs_mode can be base, lidar, track or track-lookahead.
    lidar=True still works as the old way to select lidar.
    Lidar needs dqn_scenario(lidar=True), the others use dqn_scenario().
    observation mode changes the inputs, reward mode optionally changes the training reward. actions, episode endings, and DQN settings stay the same.

    Checkpoints go in checkpoint_dir(save_path). Set the interval to 0 to turn them off.
    eval_every saves just the model in eval/<name>_<step>_steps.zip.
    These can be tested with evaluate_sb3.py.
    resume_every saves the model and replay buffer in resume/, keeping only the latest copy.
    resume=True picks up from there with the same total_timesteps plan.
    The step count and exploration schedule continue from where they stopped.

    action_repeat=N keeps each action for N simulator steps.
    SB3 counts decisions, so total_timesteps, buffer_size and target_update_interval
    are measured in decisions. Each decision covers up to N * 0.01 seconds of sim time.

    train_start='random' starts training episodes at random places on the track.
    Eval still starts from the grid.

    demo_episodes=K adds K FTG episodes to the replay buffer before training.
    For track-lookahead, FTG sees hidden LiDAR; the DQN inputs remain unchanged.
    FTG uses DQN's action choices, and the demos stay
    in the buffer for the whole run.
    demo_laps=K collects demos until it has K completed laps, keeping only those.
    Demo steps are extra. total_timesteps only counts DQN's own decisions.

    train_start='hairpin' starts a fraction (hairpin_share) of episodes just before
    the Austria hairpin. the rest start on the grid.

    init_from=<model.zip> fine-tunes a saved model with exploration fixed at 5%.
    match the saved model's LiDAR inputs and action repeat.

    crash_penalty=P subtracts P from the training reward on a wall hit that ends
    the episode. 0 turns it off.

    takeover_demos=K collects K successful hairpin demos before training.
    use init_from and lidar. the loaded model drives from the grid, then
    FTG takes over at the progress values in takeover_points, cycling
    through them across runs. FTG uses DQN's actions. demo_fraction sets
    how much of each training batch comes from the separate demo buffer
    (see Agents/demos.py).

    margin_loss=W adds the DQfD loss on takeover demos with weight W.
    it pushes the demo action 0.8 above the other Q-values (Agents/dqfd.py).
    with margin_loss on, demo_pretrain_steps=N runs N updates on demo-only
    batches before DQN starts driving.
    """
    obs_mode = resolve_obs_mode(lidar, obs_mode)
    lidar = obs_mode == 'lidar'
    use_track_demos = bool(demo_episodes or demo_laps) and obs_mode == 'track-lookahead'
    log_dir = REPO_ROOT / 'logs' / {'base': 'dqn', 'lidar': 'dqn_lidar'}.get(obs_mode, f'dqn_{obs_mode}')
    folder = checkpoint_dir(save_path)
    features = feature_config(obs_mode)
    env = counter = SimStepCounter(env)     # innermost: counts every real simulator step
    if reward_mode not in ('original', 'forward-collision', 'signed-progress'):
        raise ValueError(f'unknown reward_mode: {reward_mode}')
    if reward_mode == 'forward-collision':
        env = ForwardProgressRewardWrapper(env, progress_scale=100.0,
                                           collision_penalty=collision_penalty)
    penalty = None
    if crash_penalty:
        env = penalty = CrashPenaltyWrapper(env, penalty=crash_penalty)   # one penalty per crash
    starts = None
    if train_start == 'hairpin':
        env = starts = HairpinStartWrapper(env, share=hairpin_share)   # training only
    elif train_start != 'grid':
        env = StartModeWrapper(env, mode=train_start)   # training only, not part of dqn_wrappers
    if init_from:
        check_init_model(init_from, obs_mode, action_repeat)
        extra_init = {'exploration_initial_eps': 0.05}   # = the usual final value
    else:
        extra_init = {}
    if demo_episodes and demo_laps:
        raise ValueError('use either --demo-episodes or --demo-laps, not both')
    if takeover_demos and (demo_episodes or demo_laps):
        raise ValueError('use either --takeover-demos or --demo-episodes/--demo-laps, not both')
    if takeover_demos and not init_from:
        raise ValueError('--takeover-demos needs --init-from (the trained model drives before the handover)')
    if (margin_loss or demo_pretrain_steps) and not takeover_demos:
        raise ValueError('--margin-loss / --demo-pretrain-steps need --takeover-demos')
    if use_track_demos:
        # The LiDAR sensor is available to FTG only; DQN sees the original observations.
        env = FTGTeacherLidarWrapper(env)
    model_class = MarginDQN if margin_loss else DQN
    if takeover_demos:
        extra = {'replay_buffer_class': DemoMixDictReplayBuffer}
    elif demo_episodes or demo_laps:
        extra = {'replay_buffer_class': DemoKeepingDictReplayBuffer}
    else:
        extra = {}
    driver = SB3Driver(
        env, model_class=model_class,
        wrappers=dqn_wrappers(action_repeat=action_repeat, obs_mode=obs_mode,
                              reward_mode=reward_mode, no_camera=no_camera), verbose=1,
        tensorboard_log=str(log_dir), buffer_size=buffer_size, seed=seed,
        target_update_interval=target_update_interval, **extra, **extra_init,
    )
    if init_from:
        driver.model.set_parameters(str(init_from), exact_match=True)   # weights only, fresh replay buffer
    try:
        if resume:
            state_file = folder / 'resume' / 'state.json'
            if not state_file.exists():
                raise FileNotFoundError(f'no resumable checkpoint in {state_file.parent}; '
                                        'one is written every --resume-every steps')
            state = json.loads(state_file.read_text())
            config = json.loads((folder / 'config.json').read_text())
            defaults = {'lidar': False, 'action_repeat': 1, 'train_start': 'grid', 'demo_episodes': 0,
                        'demo_laps': 0, 'reward_mode': 'original', 'hairpin_share': None, 'init_from': None,
                        'no_camera': False, 'collision_penalty': 1.0,
                        'crash_penalty': 0.0, 'takeover_demos': 0, 'demo_fraction': None,
                        'margin_loss': 0.0, 'demo_pretrain_steps': 0}
            settings = (('lidar', lidar), ('action_repeat', action_repeat), ('train_start', train_start),
                        ('demo_episodes', demo_episodes), ('demo_laps', demo_laps), ('reward_mode', reward_mode),
                        ('no_camera', no_camera), ('collision_penalty', collision_penalty),
                        ('hairpin_share', hairpin_share if train_start == 'hairpin' else None),
                        ('init_from', str(init_from) if init_from else None),
                        ('crash_penalty', float(crash_penalty)), ('takeover_demos', takeover_demos),
                        ('demo_fraction', demo_fraction if takeover_demos else None),
                        ('margin_loss', float(margin_loss)), ('demo_pretrain_steps', demo_pretrain_steps))
            for key, value in settings:
                if config.get(key, defaults[key]) != value:
                    raise ValueError(f'this run was started with {key}={config.get(key)}; '
                                     f'resume it with the same setting')
            trained_features = config.get('features', feature_config('lidar' if config.get('lidar') else 'base'))
            if trained_features != features:
                raise ValueError(f'this run was started with observation {trained_features}, '
                                 f'but now it would be {features}; resume it with the same --obs-mode '
                                 '(and the same centre line file)')
            if state['total_timesteps'] != total_timesteps:
                raise ValueError(f"the checkpoint belongs to a {state['total_timesteps']}-step plan; "
                                 f"pass --timesteps {state['total_timesteps']} to resume it")
            driver.model = model_class.load(folder / 'resume' / 'model.zip', env=driver.env,
                                    tensorboard_log=str(log_dir))
            driver.model.load_replay_buffer(folder / 'resume' / 'replay_buffer.pkl')
            if driver.model.buffer_size != buffer_size:
                raise ValueError(f'checkpoint buffer_size is {driver.model.buffer_size}, '
                                 f'but --buffer-size is {buffer_size}')
            done = driver.model.num_timesteps
            sim_steps_before = state.get('simulated_steps') or 0
            print(f'[resume] step {done} of {total_timesteps}, replay buffer holds '
                  f'{driver.model.replay_buffer.size()} transitions', flush=True)
        else:
            done = 0
            sim_steps_before = 0
            demos = None
            if demo_episodes or demo_laps:
                demo_collector = collect_ftg_track_demos if use_track_demos else collect_ftg_demos
                demos = demo_collector(driver.model, driver.env, episodes=demo_episodes, laps=demo_laps)
                if demos['decisions'] >= buffer_size:
                    raise ValueError(f"{demos['decisions']} demo decisions do not fit in buffer_size "
                                     f'{buffer_size}; use fewer --demo-episodes')
                driver.model.replay_buffer.keep = demos['decisions']
                print(f"[demos] kept {demos['episodes']} FTG episodes (seeds {demos['seeds']}), "
                      f"{demos['decisions']} decisions ({demos['simulated_steps']} simulated steps), laps completed "
                      f"{demos['laps_completed']}/{demos['episodes']}, progress {demos['progress']}; "
                      f"discarded {demos['discarded']}", flush=True)
            if takeover_demos:
                demos = collect_takeover_demos(driver.model, driver.env, takeover_demos, tuple(takeover_points),
                                               demo_fraction)
                print(f"[takeover demos] kept {demos['episodes']} of {demos['attempts']} runs "
                      f"(seeds {demos['seeds'][0]}-{demos['seeds'][-1]}), {demos['decisions']} decisions, "
                      f"{demo_fraction:.0%} of every batch; failed {demos['failed']}", flush=True)
                if margin_loss:
                    driver.model.margin_weight = float(margin_loss)
                    demos['agreement_before'] = driver.model.demo_agreement()
                    if demo_pretrain_steps:
                        first, last = driver.model.pretrain_on_demos(demo_pretrain_steps)
                        demos['pretrain_margin_loss'] = [round(first, 4), round(last, 4)]
                    demos['agreement_after_pretrain'] = driver.model.demo_agreement()
                    print(f"[margin loss] weight {margin_loss}, pretrain {demo_pretrain_steps} steps "
                          f"(margin loss {demos.get('pretrain_margin_loss')}); greedy action = demo action in "
                          f"{demos['agreement_before']:.0%} -> {demos['agreement_after_pretrain']:.0%} of demo states",
                          flush=True)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'config.json').write_text(json.dumps({
                'algorithm': 'DQN', 'obs_mode': obs_mode, 'lidar': lidar,
                'no_camera': no_camera, 'reward_mode': reward_mode,
                'collision_penalty': collision_penalty,
                'scenario': os.path.basename(dqn_scenario(lidar or use_track_demos)), 'features': features,
                'observation_space': {k: list(s.shape) for k, s in driver.env.observation_space.spaces.items()},
                'dqn': dqn_settings(driver.model), 'code': git_commit(),
                'total_timesteps': total_timesteps, 'seed': seed, 'buffer_size': buffer_size,
                'action_repeat': action_repeat, 'target_update_interval': target_update_interval,
                'simulated_steps': total_timesteps * action_repeat, 'train_start': train_start,
                'reward_mode': reward_mode,
                'online_decisions': total_timesteps, 'demo_episodes': demo_episodes, 'demo_laps': demo_laps,
                'demos': demos,
                'hairpin_share': hairpin_share if train_start == 'hairpin' else None,
                'init_from': str(init_from) if init_from else None,
                'crash_penalty': float(crash_penalty),
                'takeover_demos': takeover_demos, 'demo_fraction': demo_fraction if takeover_demos else None,
                'takeover_points': list(takeover_points) if takeover_demos else None,
                'margin_loss': float(margin_loss), 'demo_pretrain_steps': demo_pretrain_steps,
                'exploration_initial_eps': driver.model.exploration_initial_eps,
                'eval_every': eval_every, 'resume_every': resume_every, 'save_path': str(save_path),
            }, indent=2))

        callbacks = []
        if eval_every:
            callbacks.append(EvalCheckpoint(folder / 'eval', eval_every, Path(save_path).name))
        if resume_every:
            callbacks.append(ResumableCheckpoint(folder / 'resume', resume_every, total_timesteps, counter,
                                                 sim_steps_before))

        # with reset_num_timesteps=False SB3 continues counting, so its plan is again total_timesteps
        driver.train(total_timesteps=total_timesteps - done, callback=callbacks or None,
                     reset_num_timesteps=not resume)
        driver.save(save_path)
        config = json.loads((folder / 'config.json').read_text())
        config['finished'] = {'decisions': driver.model.num_timesteps,
                              'simulated_steps': sim_steps_before + counter.sim_steps,
                              'episodes_this_process': counter.episodes, 'resumed': resume,
                              'track_values_clipped': track_clip_count(driver.env)}
        if starts is not None:   # what really happened during training
            config['start_counts_this_process'] = {'hairpin': starts.hairpin_starts, 'grid': starts.grid_starts}
        if penalty is not None:
            config['penalized_crashes_this_process'] = penalty.crashes
        (folder / 'config.json').write_text(json.dumps(config, indent=2))
    finally:
        driver.env.close()


def track_clip_count(env):
    """How many track/lookahead values were clipped to the observation bounds during training."""
    from Agents.demos import find_wrapper
    from Agents.track_features import TrackFeatureWrapper
    try:
        wrapper = find_wrapper(env, TrackFeatureWrapper)
    except LookupError:
        return None
    return {'clipped': wrapper.clipped, 'values': wrapper.values_seen}


def check_init_model(init_from, obs_mode, action_repeat):
    """The model we fine-tune from must have been trained with the same observation and action repeat."""
    from evaluate_sb3 import training_config
    if not Path(init_from).exists():
        raise FileNotFoundError(f'--init-from model not found: {init_from}')
    config = training_config(init_from)
    if config is None:
        raise ValueError(f'no config.json found for {init_from}; cannot check how it was trained')
    trained = {'obs_mode': config.get('obs_mode') or ('lidar' if config.get('lidar') else 'base'),
               'action_repeat': config.get('action_repeat', 1)}
    if trained != {'obs_mode': obs_mode, 'action_repeat': action_repeat}:
        raise ValueError(f'{init_from} was trained with {trained}; use the same --obs-mode / --action-repeat')


def train_sac(env, total_timesteps=10000, save_path='sac_model'):
    log_dir = Path(__file__).resolve().parent / 'logs' / 'sac'
    wrappers = [FlattenActionWrapper, SB3ObservationWrapper]
    driver = SB3Driver(env, model_class=SAC, wrappers=wrappers, verbose=1, tensorboard_log=str(log_dir))
    try:
        driver.train(total_timesteps=total_timesteps)
        driver.save(save_path)
    finally:
        driver.env.close()


if __name__ == '__main__':
    #with no arguments this trains base DQN for 1M steps -> models/dqn_model.zip
    parser = argparse.ArgumentParser(description='Train DQN (optionally with LiDAR or track observations).')
    parser.add_argument('--lidar', action='store_true', help='add the LiDAR scan (same as --obs-mode lidar)')
    parser.add_argument('--obs-mode', choices=['base', 'lidar', 'track', 'track-lookahead'], default=None,
                        help="extra observations: track = track-relative features, track-lookahead = "
                             "+ centre line points ahead (both use the map's centre line). Default: base")
    parser.add_argument('--no-camera', action='store_true', help='Remove RGB camera from DQN observations')
    parser.add_argument('--reward-mode',
                        choices=['original', 'forward-collision', 'signed-progress'],
                        default='original',
                        help='training reward (original preserves baseline behavior)')
    parser.add_argument('--collision-penalty', type=float, default=1.0,
                        help='positive penalty magnitude for forward-collision reward')
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--save-path', default=None,
                        help='default: models/dqn_model, or models/dqn_lidar_model with --lidar')
    parser.add_argument('--buffer-size', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=None, help='SB3 seed (use the same one for both variants)')
    parser.add_argument('--action-repeat', type=int, default=1,
                        help='hold each decision for N simulator steps (1 = off, 4 = 25 Hz like the LiDAR)')
    parser.add_argument('--target-update-interval', type=int, default=10_000,
                        help='DQN target network update interval, in decisions (SB3 default 10000)')
    parser.add_argument('--train-start', choices=['grid', 'random', 'hairpin'], default='grid',
                        help="where training episodes start; 'random' = random point on the track, "
                             "'hairpin' = part of the episodes just before the hairpin "
                             '(evaluation always starts from the grid)')
    parser.add_argument('--hairpin-share', type=float, default=0.5,
                        help='with --train-start hairpin: share of episodes starting before the hairpin')
    parser.add_argument('--crash-penalty', type=float, default=0.0,
                        help='subtract this from the reward when an episode ends at a wall (0 = off)')
    parser.add_argument('--takeover-demos', type=int, default=0,
                        help='collect N hairpin demos where FTG takes over from the --init-from model (0 = off)')
    parser.add_argument('--takeover-points', nargs='+', type=float, default=[0.24, 0.26, 0.28],
                        help='progress values where FTG takes over, used in turn')
    parser.add_argument('--demo-fraction', type=float, default=0.2,
                        help='with --takeover-demos: share of every training batch taken from the demos')
    parser.add_argument('--margin-loss', type=float, default=0.0,
                        help='with --takeover-demos: weight of the DQfD large-margin loss on demo samples (0 = off)')
    parser.add_argument('--demo-pretrain-steps', type=int, default=0,
                        help='with --margin-loss: updates on demo-only batches before training starts')
    parser.add_argument('--init-from', default=None,
                        help='fine-tune from this trained model .zip (same --obs-mode / --action-repeat)')
    parser.add_argument('--demo-episodes', type=int, default=0,
                        help='prefill the replay buffer with N FTG demo episodes (LiDAR only, 0 = off)')
    parser.add_argument('--demo-laps', type=int, default=0,
                        help='like --demo-episodes, but keep only demo episodes that finish a lap, '
                             'until N laps are kept (failed tries are discarded)')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='save an evaluable model every N steps (0 = off)')
    parser.add_argument('--resume-every', type=int, default=0,
                        help='save model + replay buffer every N steps, latest copy only (0 = off)')
    parser.add_argument('--resume', action='store_true',
                        help='continue from the latest resumable checkpoint of this --save-path')
    args = parser.parse_args()

    obs_mode = resolve_obs_mode(args.lidar, args.obs_mode)
    save_path = args.save_path or {'base': 'models/dqn_model', 'lidar': 'models/dqn_lidar_model'}.get(
        obs_mode, f"models/dqn_{obs_mode.replace('-', '_')}_model")

    # n_parallel_instances = 4
    # scenario = str(Path(__file__).resolve().parent / 'scenarios' / 'austria.yml')
    # env_factories = [
    #     partial(gym_api.SingleAgentRaceEnv, scenario=scenario, render_mode='rgb_array_birds_eye')
    #     for _ in range(n_parallel_instances)
    # ]
    # train_ppo(env_factories, total_timesteps=1_000_000, save_path='models/ppo_model')

    training_lidar = (obs_mode == 'lidar' or
                      (obs_mode == 'track-lookahead' and bool(args.demo_episodes or args.demo_laps)))
    env = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(training_lidar), render_mode='rgb_array_birds_eye')
    train_dqn(env, total_timesteps=args.timesteps, save_path=save_path, obs_mode=obs_mode,
              no_camera=args.no_camera, reward_mode=args.reward_mode,
              collision_penalty=args.collision_penalty,
              buffer_size=args.buffer_size, seed=args.seed,
              eval_every=args.eval_every, resume_every=args.resume_every, resume=args.resume,
              action_repeat=args.action_repeat, target_update_interval=args.target_update_interval,
              train_start=args.train_start, demo_episodes=args.demo_episodes, demo_laps=args.demo_laps,
              hairpin_share=args.hairpin_share, init_from=args.init_from,
              crash_penalty=args.crash_penalty, takeover_demos=args.takeover_demos,
              takeover_points=args.takeover_points, demo_fraction=args.demo_fraction,
              margin_loss=args.margin_loss, demo_pretrain_steps=args.demo_pretrain_steps)
