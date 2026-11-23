"""FTG demonstrations for DQN: prefill the replay buffer with laps driven by Follow-the-Gap.

The demonstrator is FollowTheGapDriver restricted to what DQN can do: it decides through the same
training wrapper(same action repeat,same lidar scaling)
 and each command is snapped to the nearest of DQN's discrete actions. 
 So every demo transition is one DQN could have produced itself.
"""
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.buffers import DictReplayBuffer

from Agents.FollowTheGapDriver import FollowTheGapDriver
from Agents.wrappers import DiscreteActionWrapper, LidarObservationWrapper

DEMO_SEED_START = 1000   # demo episodes use seeds 1000, 1001, ... (not the validation/test seeds)


class DemoKeepingDictReplayBuffer(DictReplayBuffer):
    """DictReplayBuffer whose first `keep` slots (the demonstrations) are never overwritten."""

    keep = 0

    def add(self, *args, **kwargs):
        super().add(*args, **kwargs)
        if self.full and self.pos == 0: # the parent just wrapped around to slot 0
            self.pos = self.keep # skip the demo slots


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
    #lidar noise
    np.random.seed(seed)                    
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
        if done:#the vec env already reset. the real last observation is in the info
            for key, value in infos[0]['terminal_observation'].items():
                next_obs[key] = value[np.newaxis]
        transitions.append((obs, next_obs, np.array([action_id]), rewards, dones, infos))
        obs = new_obs
    return transitions, infos[0]


def collect_ftg_demos(model, wrapped_env, episodes=0, laps=0, max_attempts=10,
                      seed_start=DEMO_SEED_START):
    """Fill the replay buffer with FTG demonstrations, driven from the grid start.

    episodes=K: drive K episodes (seeds 1000, 1001  ...) and keep all of them.
    laps=K:     try seeds 1000, 1001 ... and keep only episodes that complete a full lap,
                    until K laps are kept (at most max_attempts tries)
                  failed tries are discarded.
    model:       the DQN model
    wrapped_env: the training wrapper chain (SB3Driver.env), used to read the action table and
                 the LiDAR scaling
    returns a summary dict for config.json.
    """
    discrete = find_wrapper(wrapped_env, DiscreteActionWrapper)
    try:
        lidar = find_wrapper(wrapped_env, LidarObservationWrapper)
    except LookupError:
        raise ValueError('FTG demonstrations need the LiDAR scan: use --lidar') from None

    vec_env = model.env   
     # DummyVecEnv (+ VecTransposeImage), exactly what DQN trains on
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


class DemoMixDictReplayBuffer(DictReplayBuffer):
    """Normal replay buffer + a separate buffer of demonstrations.
    sample demo_fraction of each batch from the demos so successful
    hairpin runs keep showing up throughout training.
    """

    demo = None            #set by collect_takeover_demos
    demo_fraction = 0.0

    last_n_demo = 0        #number of demo samples at the end of the last batch

    def sample(self, batch_size, env=None):
        n_demo = int(round(batch_size * self.demo_fraction)) if self.demo is not None and self.demo.size() else 0
        self.last_n_demo = n_demo
        own = super().sample(batch_size - n_demo, env=env)
        if n_demo == 0:
            return own
        demo = self.demo.sample(n_demo, env=env)

        def join(a, b):
            if isinstance(a, dict):
                return {key: th.cat([a[key], b[key]]) for key in a}
            return th.cat([a, b])
        return type(own)(*[join(a, b) for a, b in zip(own, demo)])


def drive_takeover_episode(vec_env, model, ftg, discrete, lidar, seed, handover, stop_at=0.40):
    """let DQN drive from the grid until progress reaches handover.

    FTG takes over at the next decision using DQN's actions.
    return the FTG transitions, whether it passed, and the last info.
    """
    np.random.seed(seed)
    vec_env.seed(seed)
    vec_env.set_options({'mode': 'grid'})
    obs = vec_env.reset()
    ftg.reset()
    transitions, handed_over, info = [], False, {'progress': 0.0}   #grid starts at ~0.016
    while True:
        if not handed_over and info['progress'] >= handover:
            handed_over = True
        if handed_over:
            scan = lidar._low + obs['lidar'][0].astype(np.float64) * (lidar._high - lidar._low)
            action = np.array([nearest_action_id(discrete._actions, ftg.act({'lidar': scan, 'velocity': obs['velocity'][0]}))])
        else:
            action, _ = model.predict(obs, deterministic=True)
        new_obs, rewards, dones, infos = vec_env.step(action)
        info, done = infos[0], bool(dones[0])
        if handed_over:
            next_obs = {key: value.copy() for key, value in new_obs.items()}
            if done:   #auto-reset already ran, so get the last obs from info
                for key, value in info['terminal_observation'].items():
                    next_obs[key] = value[np.newaxis]
            transitions.append((obs, next_obs, action, rewards, dones, infos))
        obs = new_obs
        if done:
            return transitions, False, info               #crash or timeout
        if info['progress'] < 0.5 and info['progress'] >= stop_at:
            return transitions, True, info                #past the hairpin, stop here


def collect_takeover_demos(model, wrapped_env, episodes, handovers=(0.24, 0.26, 0.28), demo_fraction=0.2,
                           max_attempts=None, seed_start=DEMO_SEED_START):
    """collect FTG demos starting from states the trained DQN reaches.

    cycle through handovers and keep successful runs from takeover to
    progress 0.40. save them in a separate buffer for demo_fraction
    of each batch. load trained weights with --init-from first.
    return the counts and seeds for config.json.
    """
    discrete = find_wrapper(wrapped_env, DiscreteActionWrapper)
    try:
        lidar = find_wrapper(wrapped_env, LidarObservationWrapper)
    except LookupError:
        raise ValueError('takeover demonstrations need the LiDAR scan: use --lidar') from None
    buffer = model.replay_buffer
    if not isinstance(buffer, DemoMixDictReplayBuffer):
        raise TypeError('takeover demonstrations need replay_buffer_class=DemoMixDictReplayBuffer')
    ftg = FollowTheGapDriver()
    max_attempts = max_attempts or 2 * episodes
    kept, summary = [], {'episodes': 0, 'seeds': [], 'handovers': [], 'decisions': 0, 'failed': [],
                         'attempts': 0, 'demo_fraction': demo_fraction}
    for attempt in range(max_attempts):
        seed, handover = seed_start + attempt, handovers[attempt % len(handovers)]
        transitions, passed, info = drive_takeover_episode(model.env, model, ftg, discrete, lidar, seed, handover)
        summary['attempts'] += 1
        if not passed:
            summary['failed'].append({'seed': seed, 'handover': handover, 'progress': round(float(info['progress']), 3)})
            continue
        kept.append(transitions)
        summary['episodes'] += 1
        summary['seeds'].append(seed)
        summary['handovers'].append(handover)
        summary['decisions'] += len(transitions)
        if summary['episodes'] == episodes:
            break
    if summary['episodes'] < episodes:
        raise RuntimeError(f"only {summary['episodes']} of {episodes} takeover demos got through the hairpin "
                           f'in {max_attempts} tries: {summary}')
    buffer.demo = DictReplayBuffer(summary['decisions'], buffer.observation_space, buffer.action_space,
                                   device=buffer.device, n_envs=1)
    for transitions in kept:
        for transition in transitions:
            buffer.demo.add(*transition)
    buffer.demo_fraction = demo_fraction
    return summary