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


def drive_demo_episode(vec_env, ftg, discrete, lidar, seed):
    """Drive one episode with FTG snapped to DQN's actions. Returns (transitions, last info)."""
    np.random.seed(seed)                     # LiDAR noise
    vec_env.seed(seed)
    vec_env.set_options({'mode': 'grid'})
    obs = vec_env.reset()
    ftg.reset()
    transitions, done = [], False
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
        transitions.append((obs, next_obs, np.array([action_id]), rewards, dones, infos))
        obs = new_obs
    return transitions, infos[0]


def collect_ftg_demos(model, wrapped_env, episodes=0, laps=0, max_attempts=10,
                      seed_start=DEMO_SEED_START):
    """Fill the replay buffer with FTG demonstrations, driven from the grid start.

    episodes=K: drive K episodes (seeds 1000, 1001, ...) and keep all of them.
    laps=K:     try seeds 1000, 1001, ... and keep only episodes that complete a full lap,
                until K laps are kept (at most max_attempts tries); failed tries are discarded.

    model:       the DQN model (its .env is the vectorized training env, its replay buffer is filled)
    wrapped_env: the training wrapper chain (SB3Driver.env), used to read the action table and
                 the LiDAR scaling
    Returns a summary dict for config.json.
    """
    discrete = find_wrapper(wrapped_env, DiscreteActionWrapper)
    try:
        lidar = find_wrapper(wrapped_env, LidarObservationWrapper)
    except LookupError:
        raise ValueError('FTG demonstrations need the LiDAR scan: use --lidar') from None

    vec_env = model.env           # DummyVecEnv (+ VecTransposeImage), exactly what DQN trains on
    buffer = model.replay_buffer
    ftg = FollowTheGapDriver()
    summary = {'mode': 'laps' if laps else 'episodes', 'episodes': 0, 'seeds': [], 'decisions': 0,
               'simulated_steps': 0, 'laps_completed': 0, 'progress': [],
               'discarded': [], 'simulated_steps_all_attempts': 0}
    attempts = laps and max_attempts or episodes
    for attempt in range(attempts):
        seed = seed_start + attempt
        transitions, info = drive_demo_episode(vec_env, ftg, discrete, lidar, seed)
        lap_done = info['lap'] >= 2
        progress = round(float(info['lap'] - 1 + info['progress']), 3)
        sim_steps = int(round(info['time'] / 0.01))
        summary['simulated_steps_all_attempts'] += sim_steps
        if laps and not lap_done:
            summary['discarded'].append({'seed': seed, 'progress': progress})
            continue
        for transition in transitions:
            buffer.add(*transition)
        summary['episodes'] += 1
        summary['seeds'].append(seed)
        summary['decisions'] += len(transitions)
        summary['simulated_steps'] += sim_steps
        summary['laps_completed'] += int(lap_done)
        summary['progress'].append(progress)
        if laps and summary['laps_completed'] == laps:
            break
    if laps and summary['laps_completed'] < laps:
        raise RuntimeError(f"only {summary['laps_completed']} of {laps} demo laps completed in "
                           f'{max_attempts} tries: {summary}')
    return summary
