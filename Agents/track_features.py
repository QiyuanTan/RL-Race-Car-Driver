"""Adds track-relative inputs to DQN.

The centre line is built once from the racecar_gym map by tools/build_centerline.py
and saved in scenarios/austria_centerline.npz.
We use it with the car's current pose and velocity to get the features.
Nothing comes from info or future car states.

'track' adds 5 numbers:
- distance from the centre line, positive on the left, divided by 1.5 m
- sin and cos of the heading error, so angles don't jump at +/-180 degrees
- speed along the track, divided by 5 m/s
- speed across the track, positive to the left, divided by 5 m/s

'track-lookahead' also adds 5 points at 1, 2, 3, 5 and 8 m along the line.
Each point has x forward and y left relative to the car, both divided by 8 m.
That's 10 extra numbers. In a hairpin, some points can be beside or behind the car.

Each step we find the closest segment on the whole centre line.
We don't keep the previous position, so there's nothing to reset.
Training and eval use the same code.
The map tests check for wrong matches between the two sides of the hairpin.
"""
import hashlib
import os

import gymnasium as gym
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CENTERLINE_FILE = os.path.join(REPO_ROOT, 'scenarios', 'austria_centerline.npz')

TRACK_FEATURES = ['lateral_offset', 'sin_heading_error', 'cos_heading_error', 'v_along_track', 'v_across_track']
LOOKAHEAD_M = (1.0, 2.0, 3.0, 5.0, 8.0)
LOOKAHEAD_FEATURES = [f'ahead_{d:g}m_{axis}' for d in LOOKAHEAD_M for axis in ('x', 'y')]
LATERAL_SCALE = 1.5     # m, the track is at most ~2.6 m wide (1.3 m from the middle to a wall)
SPEED_SCALE = 5.0       # m/s
LOOKAHEAD_SCALE = 8.0   # m, the furthest look-ahead distance
LIMIT = 3.0             # observation_space bound after scaling; values outside are clipped and counted

OBS_MODES = ('base', 'lidar', 'track', 'track-lookahead')


def racecar_gym_map_file(track='austria'):
    """The maps.npz that racecar_gym itself loads: bullet/providers.py looks in
    <folder of providers.py>/../../models/scenes. We use providers.py's location too, because
    racecar_gym.__file__ is None when the racecar_gym clone folder sits inside our repo."""
    from racecar_gym.bullet import providers
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(providers.__file__))))
    return os.path.join(root, 'models', 'scenes', track, 'maps', 'maps.npz')

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Centerline:
    """Closed polyline; segment i goes from points[i] to points[i+1] (the last one back to points[0])."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        self.seg = np.roll(self.points, -1, axis=0) - self.points
        self.seg_len = np.hypot(self.seg[:, 0], self.seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])   # arc length at each point
        self.length = float(self.seg_len.sum())
        self.heading = np.arctan2(self.seg[:, 1], self.seg[:, 0])

    @classmethod
    def load(cls, path=CENTERLINE_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} is missing; build it with: python tools/build_centerline.py')
        return cls(np.load(path)['points'])

    def project(self, x, y):
        """Closest point on the line -> (s, lateral offset, track heading there)."""
        rel = np.array([x, y]) - self.points
        t = np.clip((rel * self.seg).sum(axis=1) / self.seg_len ** 2, 0.0, 1.0)
        diff = rel - t[:, None] * self.seg  # from the closest point to the car
        dist2 = (diff ** 2).sum(axis=1)
        i = int(np.argmin(dist2))
        cross = self.seg[i, 0] * diff[i, 1] - self.seg[i, 1] * diff[i, 0]   # > 0: car is on the left
        lateral = np.sqrt(dist2[i]) * (1.0 if cross >= 0 else -1.0)
        s = (self.s[i] + t[i] * self.seg_len[i]) % self.length
        return s, lateral, self.heading[i]

    def point_at(self, s):
        """Point(s) at arc length s, wrapping around the start line."""
        s = np.asarray(s, dtype=np.float64) % self.length
        closed = np.vstack([self.points, self.points[:1]])
        arc = np.append(self.s, self.length)
        return np.stack([np.interp(s, arc, closed[:, 0]), np.interp(s, arc, closed[:, 1])], axis=-1)


def to_car_frame(points, x, y, yaw):
    """World points -> car frame (x forward, y left)."""
    dx, dy = points[..., 0] - x, points[..., 1] - y
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def track_features(centerline, pose, velocity):
    """Unscaled features in metres / radians / m/s. Returns (s, [5 track values]).

    pose = [x, y, z, roll, pitch, yaw] in the world frame.
    velocity[:2] = [forward, left] in the CAR frame: racecar_gym's util.get_velocity rotates
    PyBullet's world velocity by the inverse body rotation (checked in the tests while driving).
    """
    x, y, yaw = pose[0], pose[1], pose[5]
    s, lateral, track_dir = centerline.project(x, y)
    error = wrap_angle(yaw - track_dir)
    # car frame -> world frame -> along / across the track
    vx = np.cos(yaw) * velocity[0] - np.sin(yaw) * velocity[1]
    vy = np.sin(yaw) * velocity[0] + np.cos(yaw) * velocity[1]
    along = np.cos(track_dir) * vx + np.sin(track_dir) * vy
    across = -np.sin(track_dir) * vx + np.cos(track_dir) * vy
    return s, np.array([lateral, np.sin(error), np.cos(error), along, across])


def lookahead_features(centerline, pose, s):
    """Centre line points LOOKAHEAD_M metres ahead of s, in the car frame, flattened [x1, y1, x2, y2, ...]."""
    ahead = centerline.point_at(s + np.array(LOOKAHEAD_M))
    return to_car_frame(ahead, pose[0], pose[1], pose[5]).reshape(-1)


def file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def centerline_sha256(points):
    """Hash of the centre line coordinates (not the .npz file, whose zip timestamps change on rebuild)."""
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float64).tobytes()).hexdigest()


def feature_config(obs_mode, path=CENTERLINE_FILE):
    """Everything that defines the observation; saved in config.json and compared at eval/resume."""
    if obs_mode not in OBS_MODES:
        raise ValueError(f'obs_mode must be one of {OBS_MODES}, got {obs_mode!r}')
    config = {'obs_mode': obs_mode}
    if obs_mode.startswith('track'):
        data = np.load(path)
        config.update({
            'track_features': TRACK_FEATURES,
            'lookahead_features': LOOKAHEAD_FEATURES if obs_mode == 'track-lookahead' else [],
            'lookahead_m': list(LOOKAHEAD_M) if obs_mode == 'track-lookahead' else [],
            'scales': {'lateral_m': LATERAL_SCALE, 'speed_mps': SPEED_SCALE, 'lookahead_m': LOOKAHEAD_SCALE,
                       'clip': LIMIT},
            'centerline_file': os.path.relpath(path, REPO_ROOT),
            'centerline_sha256': centerline_sha256(data['points']),
            'centerline_source': str(data['source']),
            'centerline_source_sha256': str(data['source_sha256']),
            'uses_map': True,
        })
    return config


class TrackFeatureWrapper(gym.ObservationWrapper):
    """Adds obs['track'] (5) and, with lookahead=True, obs['lookahead'] (10), float32 in [-LIMIT, LIMIT].

    All other keys (camera, pose, velocity, ...) are passed through unchanged.
    Raises instead of filling in zeros if the pose / velocity is missing or not finite.
    `clipped` counts how many values had to be clipped (should stay 0 on the track).
    """

    def __init__(self, env, lookahead=False, centerline=None):
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        for key in ('pose', 'velocity'):
            if key not in spaces:
                raise KeyError(f"track features need the '{key}' sensor in the scenario")
        self.centerline = centerline if centerline is not None else Centerline.load()
        self.lookahead = lookahead
        spaces['track'] = gym.spaces.Box(-LIMIT, LIMIT, (len(TRACK_FEATURES),), np.float32)
        if lookahead:
            spaces['lookahead'] = gym.spaces.Box(-LIMIT, LIMIT, (len(LOOKAHEAD_FEATURES),), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.clipped = 0
        self.values_seen = 0

    def _scaled(self, values):
        clipped = np.clip(values, -LIMIT, LIMIT)
        self.clipped += int(np.sum(clipped != values))
        self.values_seen += values.size
        return clipped.astype(np.float32)

    def observation(self, obs):
        pose = np.asarray(obs['pose'], dtype=np.float64)
        velocity = np.asarray(obs['velocity'], dtype=np.float64)
        if not (np.all(np.isfinite(pose)) and np.all(np.isfinite(velocity))):
            raise ValueError(f'non-finite pose/velocity: pose={pose}, velocity={velocity}')
        s, raw = track_features(self.centerline, pose, velocity)
        out = dict(obs)
        out['track'] = self._scaled(raw / np.array([LATERAL_SCALE, 1, 1, SPEED_SCALE, SPEED_SCALE]))
        if self.lookahead:
            out['lookahead'] = self._scaled(lookahead_features(self.centerline, pose, s) / LOOKAHEAD_SCALE)
        return out
