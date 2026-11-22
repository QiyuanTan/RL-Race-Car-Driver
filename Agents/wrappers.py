import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten, flatten_space, unflatten


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
