"""FTG demonstrations for DQN: prefill the replay buffer with laps driven by Follow-the-Gap.

The demonstrator is FollowTheGapDriver restricted to what DQN can do: it decides through the same
training wrappers (same action repeat, same LiDAR scaling) and each command is snapped to the
nearest of DQN's discrete actions. So every demo transition is one DQN could have produced itself.

Demo transitions are kept for the whole training run: DemoKeepingDictReplayBuffer never
overwrites its first `keep` slots, and DQN's own transitions cycle through the rest.
"""
import gymnasium as gym
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer

from Agents.FollowTheGapDriver import FollowTheGapDriver
from Agents.wrappers import DiscreteActionWrapper, LidarObservationWrapper

DEMO_SEED_START = 1000   # demo episodes use seeds 1000, 1001, ... (not the validation/test seeds)


class DemoKeepingDictReplayBuffer(DictReplayBuffer):
    """DictReplayBuffer whose first `keep` slots (the demonstrations) are never overwritten."""

    keep = 0

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        if self.full and self.pos == 0:    # the parent just wrapped around to slot 0
            self.pos = self.keep           # skip the demo slots


def find_wrapper(env, wrapper_class):
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_class):
            return env
        env = env.env
    raise LookupError(f'{wrapper_class.__name__} is not in the wrapper chain')


def nearest_action_id(actions, command):
    """Index of the discrete action closest to FTG's command (motor first, then steering)."""
    motor, steering = float(command['motor'][0]), float(command['steering'][0])
    return min(range(len(actions)),
               key=lambda i: (abs(actions[i]['motor'][0] - motor), abs(actions[i]['steering'][0] - steering)))


def collect_ftg_demos(model, wrapped_env, episodes, seed_start=DEMO_SEED_START):
    """Drive `episodes` demo episodes from the grid start and add every transition to the buffer.

    model:       the DQN model (its .env is the vectorized training env, its replay buffer is filled)
    wrapped_env: the training wrapper chain (SB3Driver.env), used to read the action table and
                 the LiDAR scaling
    Returns a summary dict (episodes, decisions, simulated steps, laps completed, progress).
    """
    discrete = find_wrapper(wrapped_env, DiscreteActionWrapper)
    try:
        lidar = find_wrapper(wrapped_env, LidarObservationWrapper)
    except LookupError:
        raise ValueError('FTG demonstrations need the LiDAR scan: use --lidar') from None

    vec_env = model.env           # DummyVecEnv (+ VecTransposeImage), exactly what DQN trains on
    buffer = model.replay_buffer
    ftg = FollowTheGapDriver()
    summary = {'episodes': episodes, 'seeds': [], 'decisions': 0, 'simulated_steps': 0,
               'laps_completed': 0, 'progress': []}
    for episode in range(episodes):
        seed = seed_start + episode
        np.random.seed(seed)                     # LiDAR noise
        vec_env.seed(seed)
        vec_env.set_options({'mode': 'grid'})
        obs = vec_env.reset()
        ftg.reset()
        done = False
        while not done:
            # FTG reads distances in metres: undo the 0-1 scaling of LidarObservationWrapper
            scan = lidar._low + obs['lidar'][0].astype(np.float64) * (lidar._high - lidar._low)
            command = ftg.act({'lidar': scan, 'velocity': obs['velocity'][0]})
            action_id = nearest_action_id(discrete._actions, command)
            new_obs, rewards, dones, infos = vec_env.step(np.array([action_id]))
            done = bool(dones[0])
            next_obs = {key: value.copy() for key, value in new_obs.items()}
            if done:   # the vec env already reset; the real last observation is in the info
                for key, value in infos[0]['terminal_observation'].items():
                    next_obs[key] = value[np.newaxis]
            buffer.add(obs, next_obs, np.array([action_id]), rewards, dones, infos)
            summary['decisions'] += 1
            obs = new_obs
        info = infos[0]
        summary['seeds'].append(seed)
        summary['simulated_steps'] += int(round(info['time'] / 0.01))
        summary['laps_completed'] += int(info['lap'] >= 2)
        summary['progress'].append(round(float(info['lap'] - 1 + info['progress']), 3))
    return summary
