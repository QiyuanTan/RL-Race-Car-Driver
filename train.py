import argparse
import json
import os
from functools import partial
from pathlib import Path

from Agents.SB3Drivers import SB3Driver, SB3ObservationWrapper, dqn_wrappers
from Agents.demos import DemoKeepingDictReplayBuffer, collect_ftg_demos
from Agents.wrappers import FlattenActionWrapper, DiscreteActionWrapper, StartModeWrapper
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
    """Every `every` steps, save model + replay buffer so training can be resumed.

    Only the latest copy is kept: it is written to a temporary folder first and then moved into
    place, so an interruption while saving never destroys the previous good copy.
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
        self.model.save(tmp / 'model')                         # weights, optimizer, step counters
        self.model.save_replay_buffer(tmp / 'replay_buffer')   # not part of model.zip
        (tmp / 'state.json').write_text(json.dumps(
            {'num_timesteps': self.num_timesteps, 'total_timesteps': self.total_timesteps}))
        for name in ('model.zip', 'replay_buffer.pkl', 'state.json'):   # state.json last
            os.replace(tmp / name, self.folder / name)
        print(f'[checkpoint] resumable copy saved at step {self.num_timesteps} -> {self.folder}', flush=True)


def checkpoint_dir(save_path):
    """models/dqn_model -> models/checkpoints/dqn_model (one folder per run, so variants never mix)."""
    save_path = Path(save_path)
    return save_path.parent / 'checkpoints' / save_path.name


def train_dqn(env, total_timesteps=10000, save_path='dqn_model', buffer_size=10_000,
              lidar=False, seed=None, eval_every=0, resume_every=0, resume=False,
              action_repeat=1, target_update_interval=10_000, train_start='grid', demo_episodes=0):
    """Train DQN with a bounded replay buffer, independent of training duration.

    At 128x128 RGB uint8, 10,000 transitions use about 0.92 GiB for current
    and next images alone; other observations and training need extra memory.

    lidar=True adds the LiDAR scan to the observation (env must come from
    dqn_scenario(lidar=True)); everything else (actions, reward, budget) is unchanged.

    Optional checkpoints (0 = off), stored in checkpoint_dir(save_path):
        eval_every:   eval/<name>_<step>_steps.zip, model only, can be evaluated with evaluate_sb3.py
        resume_every: resume/ model + replay buffer, latest copy only
    resume=True continues from resume/ with the same total_timesteps plan, so the step count and
    the exploration schedule carry on where they stopped.

    action_repeat=N holds each decision for N simulator steps. SB3 then counts decisions, so
    total_timesteps, buffer_size and target_update_interval are all in decisions
    (simulated time = decisions * N * 0.01 s).

    train_start='random' starts training episodes at random points of the track (racecar_gym's
    'random' reset mode). Evaluation (evaluate_sb3.py) always starts from the grid.

    demo_episodes=K (LiDAR only) first drives K episodes with FTG restricted to DQN's actions and
    puts them in the replay buffer, where they stay for the whole run (see Agents/demos.py).
    Demo decisions are extra; total_timesteps still counts only DQN's own (online) decisions.
    """
    log_dir = REPO_ROOT / 'logs' / ('dqn_lidar' if lidar else 'dqn')
    folder = checkpoint_dir(save_path)
    if train_start != 'grid':
        env = StartModeWrapper(env, mode=train_start)   # training only, not part of dqn_wrappers
    extra = {'replay_buffer_class': DemoKeepingDictReplayBuffer} if demo_episodes else {}
    driver = SB3Driver(
        env, model_class=DQN, wrappers=dqn_wrappers(lidar, action_repeat), verbose=1,
        tensorboard_log=str(log_dir), buffer_size=buffer_size, seed=seed,
        target_update_interval=target_update_interval, **extra,
    )
    try:
        if resume:
            state_file = folder / 'resume' / 'state.json'
            if not state_file.exists():
                raise FileNotFoundError(f'no resumable checkpoint in {state_file.parent}; '
                                        'one is written every --resume-every steps')
            state = json.loads(state_file.read_text())
            config = json.loads((folder / 'config.json').read_text())
            defaults = {'lidar': False, 'action_repeat': 1, 'train_start': 'grid', 'demo_episodes': 0}
            for key, value in (('lidar', lidar), ('action_repeat', action_repeat), ('train_start', train_start),
                               ('demo_episodes', demo_episodes)):
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
            if demo_episodes:
                demos = collect_ftg_demos(driver.model, driver.env, demo_episodes)
                if demos['decisions'] >= buffer_size:
                    raise ValueError(f"{demos['decisions']} demo decisions do not fit in buffer_size "
                                     f'{buffer_size}; use fewer --demo-episodes')
                driver.model.replay_buffer.keep = demos['decisions']
                print(f"[demos] {demos['episodes']} FTG episodes, {demos['decisions']} decisions "
                      f"({demos['simulated_steps']} simulated steps), laps completed "
                      f"{demos['laps_completed']}/{demos['episodes']}, progress {demos['progress']}", flush=True)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'config.json').write_text(json.dumps({
                'algorithm': 'DQN', 'lidar': lidar, 'scenario': os.path.basename(dqn_scenario(lidar)),
                'total_timesteps': total_timesteps, 'seed': seed, 'buffer_size': buffer_size,
                'action_repeat': action_repeat, 'target_update_interval': target_update_interval,
                'simulated_steps': total_timesteps * action_repeat, 'train_start': train_start,
                'online_decisions': total_timesteps, 'demo_episodes': demo_episodes, 'demos': demos,
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
    finally:
        driver.env.close()


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
    # With no arguments this trains base DQN for 1M steps -> models/dqn_model.zip, as before.
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
    parser.add_argument('--train-start', choices=['grid', 'random'], default='grid',
                        help="where training episodes start; 'random' = random point on the track "
                             '(evaluation always starts from the grid)')
    parser.add_argument('--demo-episodes', type=int, default=0,
                        help='prefill the replay buffer with N FTG demo episodes (LiDAR only, 0 = off)')
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
              train_start=args.train_start, demo_episodes=args.demo_episodes)
