"""FTG demonstrations for Track Lookahead DQN, without LiDAR in student observations

Teacher uses raw LiDAR held by FTGTeacherLidarWrapper. Transitions are collected through the same SB3 vectorized environment and DQN discrete actions as online training, including the selected reward wrapper and action repeat 
"""

import numpy as np

from Agents.FollowTheGapDriver import FollowTheGapDriver
from Agents.demos import DEMO_SEED_START, find_wrapper, nearest_action_id
from Agents.wrappers import DiscreteActionWrapper
from Agents.wrappers2 import FTGTeacherLidarWrapper


def drive_track_demo_episode(vec_env, ftg, discrete, teacher, seed):
    """Return (one episode of DQN transitions, final info), using hidden LiDAR."""
    np.random.seed(seed)
    vec_env.seed(seed)
    vec_env.set_options({'mode': 'grid'})
    obs = vec_env.reset()
    ftg.reset()
    transitions = []
    while True:
        if teacher.latest_lidar is None:
            raise RuntimeError('FTG teacher did not receive a LiDAR scan')
        if 'lidar' in obs:
            raise RuntimeError('LiDAR leaked into DQN observations')

        command = ftg.act({
            'lidar': teacher.latest_lidar.copy(),
            'velocity': obs['velocity'][0].copy(),
        })
        action_id = nearest_action_id(discrete._actions, command)
        next_obs, rewards, dones, infos = vec_env.step(np.array([action_id]))
        done = bool(dones[0])
        buffer_next_obs = {key: value.copy() for key, value in next_obs.items()}
        if done:
            # DummyVecEnv auto-resets. Keep the actual final state in the replay buffer.
            terminal = infos[0].get('terminal_observation')
            if terminal is None:
                raise RuntimeError('missing terminal_observation on completed FTG episode')
            for key, value in terminal.items():
                buffer_next_obs[key] = np.asarray(value)[np.newaxis].copy()
        if 'lidar' in buffer_next_obs:
            raise RuntimeError('LiDAR leaked into DQN next observations')

        transitions.append((obs, buffer_next_obs, np.array([action_id]),
                            rewards.copy(), dones.copy(), infos))
        if done:
            return transitions, infos[0]
        obs = next_obs


def collect_ftg_track_demos(model, wrapped_env, episodes=0, laps=0,
                            max_attempts=10, seed_start=DEMO_SEED_START):
    """Prefill the Track Lookahead DQN replay buffer using LiDAR-only FTG teacher.

    episodes=K retains K attempts; laps=K retains only successful completed laps.
    The teacher is used for collecting samples, never in DQN evaluation.
    """
    if episodes and laps:
        raise ValueError('use either episodes or laps, not both')
    if episodes < 0 or laps < 0:
        raise ValueError('episodes and laps must be non-negative')

    discrete = find_wrapper(wrapped_env, DiscreteActionWrapper)
    teacher = find_wrapper(wrapped_env, FTGTeacherLidarWrapper)
    vec_env = model.env
    buffer = model.replay_buffer
    ftg = FollowTheGapDriver()
    summary = {'mode': 'laps' if laps else 'episodes', 'episodes': 0, 'seeds': [],
               'decisions': 0, 'simulated_steps': 0, 'laps_completed': 0,
               'progress': [], 'discarded': [], 'simulated_steps_all_attempts': 0,
               'teacher': 'FTG hidden LiDAR', 'student_obs_mode': 'track-lookahead'}

    attempts = max_attempts if laps else episodes
    for attempt in range(attempts):
        seed = seed_start + attempt
        transitions, info = drive_track_demo_episode(vec_env, ftg, discrete, teacher, seed)
        lap_done = info['lap'] >= 2
        progress = round(float(info['lap'] - 1 + info['progress']), 3)
        sim_steps = int(round(info['time'] / 0.01))
        summary['simulated_steps_all_attempts'] += sim_steps

        if laps and not lap_done:
            summary['discarded'].append({'seed': seed, 'progress': progress})
            continue
        if summary['decisions'] + len(transitions) >= buffer.buffer_size:
            raise ValueError('FTG demonstrations do not fit in the replay buffer: '
                             'increase --buffer-size or use fewer demonstrations')

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
        raise RuntimeError(f"only {summary['laps_completed']} of {laps} demo laps "
                           f"completed in {max_attempts} tries: {summary}")
    return summary
