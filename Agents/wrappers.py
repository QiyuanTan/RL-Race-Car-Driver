import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium.spaces.utils import flatten, flatten_space, unflatten
"""DQN+ lidar (compare to dqn baseline)add 3 new classes: DiscreteActionWrapper, StartModeWrapper, ActionRepeatWrapper, LidarObservationWrapper"""

class SpeedLimitActionWrapper(gym.ActionWrapper):
    """Cut positive motor commands above max_speed in m/s.

    Wrap the base race environment before encoding its Dict actions. The
    environment reports its current velocity in the reset and step info.
    """

    def __init__(self, env, max_speed):
        super().__init__(env)
        if not np.isfinite(max_speed) or max_speed <= 0:
            raise ValueError("max_speed must be finite and positive")
        self.max_speed = max_speed
        self._speed = 0.0

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._speed = np.hypot(*info["velocity"][:2])
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self._speed = np.hypot(*info["velocity"][:2])
        return observation, reward, terminated, truncated, info

    def action(self, action):
        if self._speed > self.max_speed and np.asarray(action["motor"]).item() > 0:
            return {**action, "motor": np.zeros_like(action["motor"])}
        return action


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

class RemoveCameraWrapper(gym.ObservationWrapper):
    """Remove RGB camera observations while keeping numeric features."""

    def __init__(self, env):
        super().__init__(env)

        spaces = dict(env.observation_space.spaces)
        spaces.pop("rgb_camera", None)

        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, obs):
        return {
            k: v for k, v in obs.items()
            if k != "rgb_camera"
        }

class ForwardProgressRewardWrapper(gym.Wrapper):
    """Training-only reward: positive signed forward progress minus collision cost.

    Uses `lap + progress` to handle the finish-line wrap. Must sit INSIDE
    ActionRepeatWrapper so each simulator step has its own progress delta.
    The environment still decides when an episode terminates.
    """

    def __init__(self, env, progress_scale=100.0, collision_penalty=1.0):
        super().__init__(env)
        if progress_scale < 0 or collision_penalty < 0:
            raise ValueError("progress_scale and collision_penalty must be nonnegative")
        self.progress_scale = float(progress_scale)
        self.collision_penalty = float(collision_penalty)
        self._previous_progress = None

    @staticmethod
    def _absolute_progress(info):
        return float(info["lap"]) + float(info["progress"])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Read from reset info, not a fixed 0: random starts can begin anywhere.
        self._previous_progress = self._absolute_progress(info)
        return obs, info

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        current = self._absolute_progress(info)
        if self._previous_progress is None:
            raise RuntimeError("Call reset() before step()")
        delta = current - self._previous_progress
        self._previous_progress = current

        # Ignore implausible jumps rather than rewarding a state discontinuity.
        forward_delta = max(0.0, delta) if np.isfinite(delta) and abs(delta) <= 0.5 else 0.0
        collided = bool(info.get("wall_collision", False)) or bool(info.get("opponent_collisions", []))
        reward = self.progress_scale * forward_delta - self.collision_penalty * float(collided)
        return obs, float(reward), terminated, truncated, info


class ForwardProgressStallRewardWrapper(ForwardProgressRewardWrapper):
    """Train-only forward/collision reward plus a sustained-stall penalty.

    Installed INSIDE ActionRepeatWrapper: one call here == one simulator step.
    Each active simulator step applies -stall_penalty_per_decision/action_repeat,
    yielding at most -stall_penalty_per_decision per complete DQN decision.

    A stall requires BOTH a low current speed and almost no accumulated
    forward progress over the last stall_window simulator steps. This avoids
    penalizing individual zero-progress samples from discretized map projection.
    """

    def __init__(self, env, progress_scale=100.0, collision_penalty=1.0,
                 stall_window=40, stall_progress_threshold=1e-4,
                 stall_speed_threshold=0.2, stall_penalty_per_decision=0.03,
                 action_repeat=4):
        super().__init__(env, progress_scale=progress_scale,
                         collision_penalty=collision_penalty)
        if not isinstance(stall_window, int) or stall_window < 1:
            raise ValueError("stall_window must be a positive integer")
        if not isinstance(action_repeat, int) or action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        values = (stall_progress_threshold, stall_speed_threshold,
                  stall_penalty_per_decision)
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError("stall thresholds and penalty must be finite and nonnegative")
        self.stall_window = stall_window
        self.stall_progress_threshold = float(stall_progress_threshold)
        self.stall_speed_threshold = float(stall_speed_threshold)
        self.stall_penalty_per_sim_step = float(stall_penalty_per_decision) / action_repeat
        self._progress_window = deque(maxlen=stall_window)
        self.stall_sim_steps = 0
        self.total_stall_penalty = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._progress_window.clear()
        self.stall_sim_steps = 0
        self.total_stall_penalty = 0.0
        return obs, info

    def step(self, action):
        # super().step updates _previous_progress and preserves collision cost.
        before = self._previous_progress
        obs, reward, terminated, truncated, info = super().step(action)
        after = self._previous_progress
        delta = after - before
        # Include only plausible forward motion; backward does not count as progress.
        forward_delta = max(delta, 0.0) if np.isfinite(delta) and abs(delta) <= 0.5 else 0.0
        self._progress_window.append(forward_delta)
        velocity = np.asarray(info['velocity'], dtype=np.float64).reshape(-1)
        if velocity.size < 2:
            raise ValueError("info['velocity'] must provide at least x and y components")
        speed = float(np.hypot(velocity[0], velocity[1]))
        stalled = (len(self._progress_window) == self.stall_window
                   and sum(self._progress_window) < self.stall_progress_threshold
                   and np.isfinite(speed) and speed < self.stall_speed_threshold)
        penalty = self.stall_penalty_per_sim_step if stalled else 0.0
        if stalled:
            self.stall_sim_steps += 1
            self.total_stall_penalty += penalty
        # Diagnostics are training-only; do not alter the policy observation.
        info = dict(info)
        info['reward_v2_stalled'] = bool(stalled)
        info['reward_v2_stall_penalty'] = float(penalty)
        return obs, float(reward - penalty), terminated, truncated, info
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
    The default frame is the smallest square that contains the drivable area.
    Set crop_to_track=False for checkpoints trained on the original grid extent.
    The distance channel measures from each cell center to the observed car
    position in units of the frame's diagonal, capped at 1.
    """

    KEY = "track_view"
    _channel_count = 2

    def __init__(self, env, track=None, map_size=256, *, crop_to_track=True):
        super().__init__(env)
        original_space = env.observation_space
        self._crop_to_track = crop_to_track

        if track is None:
            scenario = getattr(env.unwrapped, "scenario", None)
            world = getattr(scenario, "world", None)
            # RacecarGym stores its GridMaps on the world before the first reset.
            track = getattr(world, "_maps", {}).get("occupancy")
            if track is None:
                raise ValueError("Pass an occupancy GridMap as track for this environment")

        self._prepare_view(track, map_size)
        self.observation_space = gym.spaces.Dict([
            *original_space.spaces.items(),
            (self.KEY, gym.spaces.Box(0.0, 1.0, (self._channel_count, map_size, map_size), np.float32)),
        ])

    def _prepare_view(self, track, map_size):
        drivable = np.asarray(track.map, dtype=bool)
        row_offset = col_offset = 0
        if self._crop_to_track:
            drivable, row_offset, col_offset = self._crop_drivable(drivable)
        height, width = drivable.shape
        self._track_mask = self._build_track_mask(drivable, map_size)

        # GridMap's integer coordinates mark cell edges. Rows run toward -y.
        x_left, y_top = track.to_meter(0, 0)
        resolution = track.resolution
        x_left += col_offset * resolution
        y_top -= row_offset * resolution
        width_m = width * resolution
        height_m = height * resolution
        centers = (np.arange(map_size, dtype=np.float32) + 0.5) / map_size
        self._x = x_left + centers[None, :] * width_m
        self._y = y_top - centers[:, None] * height_m
        self._diagonal = np.hypot(width_m, height_m)

    @staticmethod
    def _crop_drivable(drivable):
        rows = np.flatnonzero(drivable.any(axis=1))
        if rows.size == 0:
            return drivable, 0, 0
        cols = np.flatnonzero(drivable.any(axis=0))
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(cols[0]), int(cols[-1]) + 1
        cropped = drivable[top:bottom, left:right]
        height, width = cropped.shape

        # Pad the shorter dimension so both axes use the same meters per pixel.
        side = max(height, width)
        pad_top = (side - height) // 2
        pad_left = (side - width) // 2
        square = np.pad(cropped, (
            (pad_top, side - height - pad_top),
            (pad_left, side - width - pad_left),
        ), constant_values=False)
        return square, top - pad_top, left - pad_left

    def _build_track_mask(self, drivable, map_size):
        boundary = self._mask_from_padded_drivable(np.pad(drivable, 1, constant_values=False))
        boundary = self._resize_boundary(boundary, map_size, axis=0)
        return self._resize_boundary(boundary, map_size, axis=1)

    @staticmethod
    def _mask_from_padded_drivable(padded):
        interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        return padded[1:-1, 1:-1] & ~interior

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
        view = np.empty(self.observation_space[self.KEY].shape, dtype=np.float32)
        self._fill_view(view, obs["pose"])
        return {**obs, self.KEY: view}

    def _fill_view(self, view, pose):
        car_x, car_y = pose[:2]
        view[0] = self._track_mask
        np.hypot(self._x - car_x, self._y - car_y, out=view[1])
        view[1] /= self._diagonal
        np.minimum(view[1], 1.0, out=view[1])


class LocalTrackObservationWrapper(TrackObservationWrapper):
    """Add a car-relative mask with the car 3/4 down the image.

    Set include_distance=True to restore the fixed channel for older checkpoints.
    """

    def __init__(self, env, track=None, map_size=128, view_size_m=12.8, *, include_distance=False):
        if isinstance(map_size, bool) or not isinstance(map_size, (int, np.integer)) or map_size <= 0:
            raise ValueError("map_size must be a positive integer")
        if not np.isfinite(view_size_m) or view_size_m <= 0:
            raise ValueError("view_size_m must be finite and positive")
        space = env.observation_space
        if not isinstance(space, gym.spaces.Dict) or "pose" not in space.spaces or space["pose"].shape != (6,):
            raise ValueError("LocalTrackObservationWrapper requires a Dict observation with a six-component pose")
        self._view_size_m = view_size_m
        self._channel_count = 2 if include_distance else 1
        super().__init__(env, track=track, map_size=map_size)

    def _prepare_view(self, track, map_size):
        self._drivable = np.asarray(track.map, dtype=bool)
        self._resolution = track.resolution
        self._x_left, self._y_top = track.to_meter(0, 0)

        # Sample a one-cell border as well so the crop itself creates no edges.
        cell_size = self._view_size_m / map_size
        centers = np.arange(-1, map_size + 1, dtype=np.float64) + 0.5
        self._forward = (0.75 * map_size - centers[:, None]) * cell_size
        self._right = (centers[None, :] - 0.5 * map_size) * cell_size
        self._distance = None
        if self._channel_count == 2:
            distance = np.hypot(self._forward[1:-1], self._right[:, 1:-1])
            self._distance = (distance / (np.sqrt(2) * self._view_size_m)).astype(np.float32)

    def _fill_view(self, view, pose):
        car_x, car_y = pose[:2]
        yaw = pose[5]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        world_x = car_x + self._forward * cos_yaw + self._right * sin_yaw
        world_y = car_y + self._forward * sin_yaw - self._right * cos_yaw
        cols = (world_x - self._x_left) / self._resolution
        rows = (self._y_top - world_y) / self._resolution

        height, width = self._drivable.shape
        valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        drivable = np.zeros(valid.shape, dtype=bool)
        drivable[valid] = self._drivable[rows[valid].astype(np.intp), cols[valid].astype(np.intp)]
        view[0] = self._mask_from_padded_drivable(drivable)
        if self._distance is not None:
            view[1] = self._distance


class DrivableTrackObservationWrapper(TrackObservationWrapper):
    """Add a full-track occupancy mask and distance to the car."""

    KEY = "drivable_track_view"

    def _build_track_mask(self, drivable, map_size):
        height, width = drivable.shape
        centers = (np.arange(map_size) + 0.5) / map_size
        # Read the occupancy of the source cell containing each output cell center.
        rows = (centers * height).astype(np.intp)
        cols = (centers * width).astype(np.intp)
        return drivable[np.ix_(rows, cols)]


class LocalDrivableTrackObservationWrapper(LocalTrackObservationWrapper):
    """Add a car-relative occupancy mask."""

    KEY = DrivableTrackObservationWrapper.KEY

    @staticmethod
    def _mask_from_padded_drivable(padded):
        return padded[1:-1, 1:-1]
