import argparse
import json
import os
from functools import partial
from pathlib import Path

from Agents.SB3Drivers import SB3Driver, SB3ObservationWrapper, dqn_wrappers
from Agents.demos import DemoKeepingDictReplayBuffer, collect_ftg_demos
from Agents.wrappers import (FlattenActionWrapper, DiscreteActionWrapper, CrashPenaltyWrapper, HairpinStartWrapper,
                             StartModeWrapper)
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
    """Scenario file for DQN. The LiDAR variant only adds the `lidar` sensor."""
    name = 'austria_lidar.yml' if lidar else 'austria.yml'
    return str(REPO_ROOT / 'scenarios' / name)


class EvalCheckpoint(BaseCallback):
    """every steps, save the model (no replay buffer) as <prefix>_<step>_steps.zip.

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

    def __init__(self, folder, every, total_timesteps):
        super().__init__()
        self.folder = Path(folder)
        self.every = every
        self.total_timesteps = total_timesteps

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
            {'num_timesteps': self.num_timesteps, 'total_timesteps': self.total_timesteps}))
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
              demo_laps=0, hairpin_share=0.5, init_from=None, crash_penalty=0.0):
    """Train DQN with a bounded replay buffer, independent of training duration.

    At 128x128 RGB uint8, 10,000 transitions use about 0.92 GiB for current
    and next images alone; other observations and training need extra memory.

    lidar=True adds the LiDAR scan to the observation (env must come from
    dqn_scenario(lidar=True)); everything else (actions, reward, budget) is unchanged.

    checkpoints go in checkpoint_dir(save_path). set either interval to 0 to turn it off.
    eval_every saves model-only snapshots to eval/<name>_<step>_steps.zip for evaluate_sb3.py.
    resume_every keeps the latest model and replay buffer in resume/.
    resume=True picks up the step count and exploration schedule. keep total_timesteps the same.

    action_repeat=N holds each action for N sim steps. total_timesteps, buffer_size,
    and target_update_interval count decisions. each decision covers N * 0.01 s.

    train_start='random' uses random track starts for training. eval starts on the grid.

    demos (LiDAR only) use FTG with DQN's actions (see Agents/demos.py).
    demo_episodes=K keeps K episodes. demo_laps=K keeps only K completed laps.
    demos stay in the replay buffer and don't count toward total_timesteps.

    train_start='hairpin' starts a fraction (hairpin_share) of episodes just before
    the Austria hairpin. the rest start on the grid.

    init_from=<model.zip> fine-tunes a saved model with exploration fixed at 5%.
    match the saved model's LiDAR inputs and action repeat.

    crash_penalty=P subtracts P from the training reward on a wall hit that ends
    the episode. 0 turns it off.
    """
    log_dir = REPO_ROOT / 'logs' / ('dqn_lidar' if lidar else 'dqn')
    folder = checkpoint_dir(save_path)
    penalty = None
    if crash_penalty:
        env = penalty = CrashPenaltyWrapper(env, penalty=crash_penalty)   
    starts = None
    if train_start == 'hairpin':
        env = starts = HairpinStartWrapper(env, share=hairpin_share) 
    elif train_start != 'grid':
        env = StartModeWrapper(env, mode=train_start)   #training only, not part of dqn_wrappers
    if init_from:
        check_init_model(init_from, lidar, action_repeat)
        extra_init = {'exploration_initial_eps': 0.05}  
    else:
        extra_init = {}
    if demo_episodes and demo_laps:
        raise ValueError('use either --demo-episodes or --demo-laps, not both')
    extra = {'replay_buffer_class': DemoKeepingDictReplayBuffer} if (demo_episodes or demo_laps) else {}
    driver = SB3Driver(
        env, model_class=DQN, wrappers=dqn_wrappers(lidar, action_repeat), verbose=1,
        tensorboard_log=str(log_dir), buffer_size=buffer_size, seed=seed,
        target_update_interval=target_update_interval, **extra, **extra_init,
    )
    if init_from:
        driver.model.set_parameters(str(init_from), exact_match=True) 
    try:
        if resume:
            state_file = folder / 'resume' / 'state.json'
            if not state_file.exists():
                raise FileNotFoundError(f'no resumable checkpoint in {state_file.parent}; '
                                        'one is written every --resume-every steps')
            state = json.loads(state_file.read_text())
            config = json.loads((folder / 'config.json').read_text())
            defaults = {'lidar': False, 'action_repeat': 1, 'train_start': 'grid', 'demo_episodes': 0,
                        'demo_laps': 0, 'hairpin_share': None, 'init_from': None, 'crash_penalty': 0.0}
            settings = (('lidar', lidar), ('action_repeat', action_repeat), ('train_start', train_start),
                        ('demo_episodes', demo_episodes), ('demo_laps', demo_laps),
                        ('hairpin_share', hairpin_share if train_start == 'hairpin' else None),
                        ('init_from', str(init_from) if init_from else None),
                        ('crash_penalty', float(crash_penalty)))
            for key, value in settings:
                if config.get(key, defaults[key]) != value:
                    raise ValueError(f'this run was started with {key}={config.get(key)}; '
                                     f'resume it with the same setting')
            if state['total_timesteps'] != total_timesteps:
                raise ValueError(f"the checkpoint belongs to a {state['total_timesteps']}-step plan; "
                                 f"pass --timesteps {state['total_timesteps']} to resume it")
            driver.model = DQN.load(folder / 'resume' / 'model.zip', env=driver.env,
                                    tensorboard_log=str(log_dir))
            driver.model.load_replay_buffer(folder / 'resume' / 'replay_buffer.pkl')
            if driver.model.buffer_size != buffer_size:
                raise ValueError(f'checkpoint buffer_size is {driver.model.buffer_size}, '
                                 f'but --buffer-size is {buffer_size}')
            done = driver.model.num_timesteps
            print(f'[resume] step {done} of {total_timesteps}, replay buffer holds '
                  f'{driver.model.replay_buffer.size()} transitions', flush=True)
        else:
            done = 0
            demos = None
            if demo_episodes or demo_laps:
                demos = collect_ftg_demos(driver.model, driver.env, episodes=demo_episodes, laps=demo_laps)
                if demos['decisions'] >= buffer_size:
                    raise ValueError(f"{demos['decisions']} demo decisions do not fit in buffer_size "
                                     f'{buffer_size}; use fewer --demo-episodes')
                driver.model.replay_buffer.keep = demos['decisions']
                print(f"[demos] kept {demos['episodes']} FTG episodes (seeds {demos['seeds']}), "
                      f"{demos['decisions']} decisions ({demos['simulated_steps']} simulated steps), laps completed "
                      f"{demos['laps_completed']}/{demos['episodes']}, progress {demos['progress']}; "
                      f"discarded {demos['discarded']}", flush=True)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'config.json').write_text(json.dumps({
                'algorithm': 'DQN', 'lidar': lidar, 'scenario': os.path.basename(dqn_scenario(lidar)),
                'total_timesteps': total_timesteps, 'seed': seed, 'buffer_size': buffer_size,
                'action_repeat': action_repeat, 'target_update_interval': target_update_interval,
                'simulated_steps': total_timesteps * action_repeat, 'train_start': train_start,
                'online_decisions': total_timesteps, 'demo_episodes': demo_episodes, 'demo_laps': demo_laps,
                'demos': demos,
                'hairpin_share': hairpin_share if train_start == 'hairpin' else None,
                'init_from': str(init_from) if init_from else None,
                'crash_penalty': float(crash_penalty),
                'exploration_initial_eps': driver.model.exploration_initial_eps,
                'eval_every': eval_every, 'resume_every': resume_every, 'save_path': str(save_path),
            }, indent=2))

        callbacks = []
        if eval_every:
            callbacks.append(EvalCheckpoint(folder / 'eval', eval_every, Path(save_path).name))
        if resume_every:
            callbacks.append(ResumableCheckpoint(folder / 'resume', resume_every, total_timesteps))

        # with reset_num_timesteps=False SB3 continues counting, so its plan is again total_timesteps
        driver.train(total_timesteps=total_timesteps - done, callback=callbacks or None,
                     reset_num_timesteps=not resume)
        driver.save(save_path)
        if starts is not None or penalty is not None:  
            config = json.loads((folder / 'config.json').read_text())
            if starts is not None:
                config['start_counts_this_process'] = {'hairpin': starts.hairpin_starts, 'grid': starts.grid_starts}
            if penalty is not None:
                config['penalized_crashes_this_process'] = penalty.crashes
            (folder / 'config.json').write_text(json.dumps(config, indent=2))
    finally:
        driver.env.close()

def check_init_model(init_from, lidar, action_repeat):
    """The model we fine-tune from must have been trained with the same observation and action repeat."""
    from evaluate_sb3 import training_config
    if not Path(init_from).exists():
        raise FileNotFoundError(f'--init-from model not found: {init_from}')
    config = training_config(init_from)
    if config is None:
        raise ValueError(f'no config.json found for {init_from}; cannot check how it was trained')
    trained = {'lidar': config.get('lidar', False), 'action_repeat': config.get('action_repeat', 1)}
    if trained != {'lidar': lidar, 'action_repeat': action_repeat}:
        raise ValueError(f'{init_from} was trained with {trained}; use the same --lidar / --action-repeat')


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
    parser = argparse.ArgumentParser(description='Train DQN (optionally with LiDAR observations).')
    parser.add_argument('--lidar', action='store_true', help='add the LiDAR scan to the observation')
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
    parser.add_argument('--init-from', default=None,
                        help='fine-tune from this trained model .zip (same --lidar / --action-repeat)')
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

    save_path = args.save_path or ('models/dqn_lidar_model' if args.lidar else 'models/dqn_model')

    # n_parallel_instances = 4
    # scenario = str(Path(__file__).resolve().parent / 'scenarios' / 'austria.yml')
    # env_factories = [
    #     partial(gym_api.SingleAgentRaceEnv, scenario=scenario, render_mode='rgb_array_birds_eye')
    #     for _ in range(n_parallel_instances)
    # ]
    # train_ppo(env_factories, total_timesteps=1_000_000, save_path='models/ppo_model')

    env = gym_api.SingleAgentRaceEnv(scenario=dqn_scenario(args.lidar), render_mode='rgb_array_birds_eye')
    train_dqn(env, total_timesteps=args.timesteps, save_path=save_path,
              buffer_size=args.buffer_size, lidar=args.lidar, seed=args.seed,
              eval_every=args.eval_every, resume_every=args.resume_every, resume=args.resume,
              action_repeat=args.action_repeat, target_update_interval=args.target_update_interval,
              train_start=args.train_start, demo_episodes=args.demo_episodes, demo_laps=args.demo_laps,
              hairpin_share=args.hairpin_share, init_from=args.init_from, crash_penalty=args.crash_penalty)
