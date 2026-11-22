import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
"""DQN+ lidar (compare to dqn baseline)add 3 new classes: DiscreteActionWrapper, StartModeWrapper, ActionRepeatWrapper, LidarObservationWrapper"""

class DiscreteActionWrapper(gym.ActionWrapper):
    """Map discrete IDs to absolute motor and steering commands.

    Defaults provide 15 actions: brake/coast/drive crossed with five steering
    positions. Steering varies fastest: ID = motor_index * 5 + steering_index.
    For custom levels, replace 5 with the number of steering levels.
    """

    def __init__(
        self,
        env,
        motor_levels=(-0.5, 0.0, 0.5),
        steering_levels=(-1.0, -0.5, 0.0, 0.5, 1.0),
    ):
        super().__init__(env)
        action_space = env.action_space

        levels = {}
        for key, values in (
                ("motor", motor_levels), ("steering", steering_levels)
        ):
            space = action_space[key]

            values = np.asarray(values, dtype=space.dtype)

            levels[key] = [np.full(space.shape, value, dtype=space.dtype) for value in values]

        self._actions = tuple(
            {"motor": motor, "steering": steering}
            for motor in levels["motor"]
            for steering in levels["steering"]
        )
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        """Decode an ID, preserving the original actuator shapes and dtypes."""
        return {key: value.copy() for key, value in self._actions[int(action)].items()}


class StartModeWrapper(gym.Wrapper):
    """for training only: start each episode with racecar_gym's reset mode, e.g. 'random'.

    'random' starts the car at a random point on the track centerline,
    facing forward, so it can practice later sections like the hairpin.

    if reset() gets options={'mode': ...}, use that mode.
    """

    def __init__(self, env, mode='grid'):
        super().__init__(env)
        self.mode = mode

    def reset(self, *, seed=None, options=None):
        if options is None or 'mode' not in options:
            options = dict(options or {}, mode=self.mode)
        return self.env.reset(seed=seed, options=options)


class ActionRepeatWrapper(gym.Wrapper):
    """repeat each action for `repeat` simulator steps (0.01 s each); rewards are added up.

    The policy then decides every `repeat` steps, e.g. repeat=4 -> 25 Hz, which matches the
    LiDAR update rate. Stops early when the episode ends, and returns the last observation/info.
    """

    def __init__(self, env, repeat=4):
        super().__init__(env)
        if repeat < 1:
            raise ValueError('repeat must be at least 1')
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class LidarObservationWrapper(gym.ObservationWrapper):
    """clean and scale the raw LiDAR scan so it can be fed to the policy as a vector.

    racecar_gym's `lidar` observation is Box(low=0.25, high=15.25, shape=(1080,), float64),
    distances in metres. The sensor itself clips readings to [0.25, 15.0], so the space's
    upper bound is never reached; we still scale with the space bounds so the output always
    lies inside the new observation space.

    Output: float32 in [0, 1], computed as (distance - low) / (high - low).
    NaN is treated as the closest distance (a wall, the safe assumption), +inf as the farthest.
    All other observation keys pass through unchanged.
    """

    def __init__(self, env, key="lidar"):
        super().__init__(env)
        observation_space = env.observation_space
        if key not in observation_space.spaces:
            raise KeyError(
                f"observation has no '{key}' key; use a scenario whose sensors include "
                f"'{key}', e.g. scenarios/austria_lidar.yml"
            )
        space = observation_space[key]
        self._key = key
        self._low = float(np.min(space.low))
        self._high = float(np.max(space.high))
        new_spaces = dict(observation_space.spaces)
        new_spaces[key] = gym.spaces.Box(low=0.0, high=1.0, shape=space.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(new_spaces)

    def observation(self, obs):
        scan = np.asarray(obs[self._key], dtype=np.float64)
        scan = np.nan_to_num(scan, nan=self._low, posinf=self._high, neginf=self._low)
        scan = np.clip(scan, self._low, self._high)
        observation = dict(obs)
        observation[self._key] = ((scan - self._low) / (self._high - self._low)).astype(np.float32)
        return observation


class FlattenActionWrapper(gym.ActionWrapper):
    """Expose a single agent's Dict of Box actions as one flat Box.

    Coordinates follow the original Dict space's key order. For the racecar,
    this is [motor, steering]. Bounds and each actuator's shape are preserved.
    """

    def __init__(self, env):
        super().__init__(env)
        action_space = env.action_space

        self._original_action_space = action_space
        self.action_space = flatten_space(action_space)

    def action(self, action):
        action = np.asarray(action, dtype=self.action_space.dtype)
        return unflatten(self._original_action_space, action)

    def reverse_action(self, action):
        return flatten(self._original_action_space, action)


class SimStepCounter(gym.Wrapper):
    """Counts real simulator steps (training only), so config.json can report the actual numbers."""

    def __init__(self, env):
        super().__init__(env)
        self.sim_steps = 0
        self.episodes = 0

    def reset(self, **kwargs):
        self.episodes += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        self.sim_steps += 1
        return self.env.step(action)


class HairpinStartWrapper(gym.Wrapper):
    """start some training episodes just before the Austria hairpin.
    share is the chance of starting at a random point in window. The default
    is 0.22–0.30, about 4.5–11 m before the apex (0.351), so the car has room
    to get up to speed. Other training runs and all eval runs start on the grid.

    racecar_gym picks the random start, so keep resetting until it lands in
    the window. About 1 in 15 resets works, and each takes ~0.03 s.

    stop at 0.30 because closer starts can face 40–110 degrees off the track
    direction. The car aims at a point ahead that's already around the corner.
    """

    def __init__(self, env, share=0.5, window=(0.22, 0.30), max_tries=1000):
        super().__init__(env)
        if not 0.0 <= share <= 1.0:
            raise ValueError('share must be between 0 and 1')
        self.share, self.window, self.max_tries = share, window, max_tries
        self.rng = np.random.default_rng()
        self.hairpin_starts = 0
        self.grid_starts = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and 'mode' in options:   # an explicit start (e.g. demonstrations from the grid) wins
            return self.env.reset(seed=seed, options=options)
        if self.rng.random() >= self.share:
            self.grid_starts += 1
            return self.env.reset(seed=seed, options=dict(options or {}, mode='grid'))
        for _ in range(self.max_tries):
            obs, info = self.env.reset(seed=seed, options=dict(options or {}, mode='random'))
            seed = None                     # seed only the first try
            if self.window[0] <= info['progress'] <= self.window[1]:
                self.hairpin_starts += 1
                return obs, info
        raise RuntimeError(f'no random start in progress window {self.window} after {self.max_tries} tries')


class CrashPenaltyWrapper(gym.Wrapper):
    """subtract penalty from the reward if a training episode ends with a wall hit.
the normal reward is 100 times progress gained, so a full lap pays 100 and
crashes cost nothing. At the hairpin, driving straight into the wall near
the apex earns progress quickly. Going wide and turning early earns very
little at first, so the crash penalty can make that line pay off.

eval ignores rewards.
    """

    def __init__(self, env, penalty=10.0):
        super().__init__(env)
        if penalty < 0:
            raise ValueError('penalty must be >= 0 (it is subtracted)')
        self.penalty = penalty
        self.crashes = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and info.get('wall_collision', False):
            reward -= self.penalty
            self.crashes += 1
        return obs, reward, terminated, truncated, info


class TrackObservationWrapper(gym.ObservationWrapper):
    """Add a full-track view with boundary and car-distance channels.

    The track defaults to the environment's occupancy GridMap. An explicit
    GridMap can be supplied for environments that expose the track differently.
    The distance channel measures from each cell center to the observed car
    position in units of the map diagonal, capped at 1.
    """

    KEY = "track_view"

    def __init__(self, env, track=None, map_size=256):
        super().__init__(env)
        original_space = env.observation_space

        if track is None:
            scenario = getattr(env.unwrapped, "scenario", None)
            world = getattr(scenario, "world", None)
            # RacecarGym stores its GridMaps on the world before the first reset.
            track = getattr(world, "_maps", {}).get("occupancy")
            if track is None:
                raise ValueError("Pass an occupancy GridMap as track for this environment")

        drivable = np.asarray(track.map, dtype=bool)
        height, width = drivable.shape
        padded = np.pad(drivable, 1, constant_values=False)
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        boundary = drivable & ~interior
        boundary = self._resize_boundary(boundary, map_size, axis=0)
        self._boundary = self._resize_boundary(boundary, map_size, axis=1)

        # GridMap's integer coordinates mark cell edges. Rows run toward -y.
        x_left, y_top = track.to_meter(0, 0)
        width_m = width * track.resolution
        height_m = height * track.resolution
        centers = (np.arange(map_size, dtype=np.float32) + 0.5) / map_size
        self._x = x_left + centers[None, :] * width_m
        self._y = y_top - centers[:, None] * height_m
        self._diagonal = np.hypot(width_m, height_m)

        self.observation_space = gym.spaces.Dict([
            *original_space.spaces.items(),
            (self.KEY, gym.spaces.Box(0.0, 1.0, (2, map_size, map_size), np.float32)),
        ])

    @staticmethod
    def _resize_boundary(boundary, size, axis):
        length = boundary.shape[axis]
        if size > length:
            indices = ((np.arange(size) + 0.5) * length / size).astype(np.intp)
            return np.take(boundary, indices, axis=axis)

        # Pool all source cell centers in each output cell to retain thin edges.
        edges = np.linspace(0, length, size + 1)
        starts = np.searchsorted(np.arange(length) + 0.5, edges[:-1])
        return np.logical_or.reduceat(boundary, starts, axis=axis)

    def observation(self, obs):
        car_x, car_y = obs["pose"][:2]
        view = np.empty(self.observation_space[self.KEY].shape, dtype=np.float32)
        view[0] = self._boundary
        np.hypot(self._x - car_x, self._y - car_y, out=view[1])
        view[1] /= self._diagonal
        np.minimum(view[1], 1.0, out=view[1])
        return {**obs, self.KEY: view}
