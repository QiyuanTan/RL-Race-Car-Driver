import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space, unflatten


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
