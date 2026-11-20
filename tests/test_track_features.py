"""Tests for the track-relative

The first tests use small hand-made tracks where we know the right answer, the later ones use
the real Austria centre line and racecar_gym's map.
"""
import json
import os
import sys

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agents.SB3Drivers import SB3Driver, SB3EvalDriver, dqn_wrappers, resolve_obs_mode  # noqa: E402
from Agents.track_features import (CENTERLINE_FILE, Centerline, TrackFeatureWrapper, file_sha256,  # noqa: E402
                                   lookahead_features, racecar_gym_map_file, track_features)
from Agents.wrappers import DiscreteActionWrapper  # noqa: E402


def square(side=10.0, spacing=0.1):
    """Counter-clockwise square track starting at (0, 0), driving towards +x."""
    t = np.arange(int(side / spacing)) * spacing
    edges = [np.c_[t, 0 * t], np.c_[side + 0 * t, t], np.c_[side - t, side + 0 * t], np.c_[0 * t, side - t]]
    return Centerline(np.vstack(edges))


def circle(radius=10.0, n=2000):
    """Counter-clockwise circle, so the inside is on the left."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Centerline(np.c_[radius * np.cos(a), radius * np.sin(a)])


def pose(x, y, yaw):
    return np.array([x, y, 0.05, 0, 0, yaw])


class FakeEnv(gym.Env):
    """Only pose / velocity / time, like racecar_gym without the camera."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            'pose': gym.spaces.Box(-100, 100, (6,), np.float64),
            'velocity': gym.spaces.Box(-100, 100, (6,), np.float64),
            'time': gym.spaces.Box(0, 1, (), np.float32)})
        self.action_space = gym.spaces.Dict({'motor': gym.spaces.Box(-1, 1, (1,), np.float32),
                                             'steering': gym.spaces.Box(-1, 1, (1,), np.float32)})
        self.next_obs = {'pose': np.zeros(6), 'velocity': np.zeros(6), 'time': np.array(0.0, np.float32)}

    def reset(self, seed=None, options=None):
        return dict(self.next_obs), {}

    def step(self, action):
        return dict(self.next_obs), 0.0, False, False, {}


# ---geometry on hand-made tracks -------------

def test_offset_is_zero_on_the_centre_line():
    line = circle()
    for a in np.linspace(0, 2 * np.pi, 37):
        assert abs(line.project(10 * np.cos(a), 10 * np.sin(a))[1]) < 1e-3


def test_offset_sign_left_is_positive():
    assert square().project(5.0, 0.4)[1] == pytest.approx(0.4)      # bottom edge drives +x, +y is left
    assert square().project(5.0, -0.4)[1] == pytest.approx(-0.4)
    assert circle().project(9.0, 0.0)[1] == pytest.approx(1.0, abs=1e-3)   # inside of the circle = left


def test_heading_error_has_no_jump_at_180_degrees():
    line = square() # top edge drives towards -x (track direction = +-pi)
    a = track_features(line, pose(5.0, 10.0, np.pi - 0.01), np.zeros(6))[1]
    b = track_features(line, pose(5.0, 10.0, -np.pi + 0.01), np.zeros(6))[1]
    assert np.hypot(a[1] - b[1], a[2] - b[2]) < 0.03  # 0.02 rad apart -> almost the same sin/cos


def test_velocity_projection():
    line = square()
    #bottom edge (track goes +x), car facing +y and driving forward 2 m/s -> moving left across the track
    f = track_features(line, pose(5.0, 0.0, np.pi / 2), np.array([2.0, 0, 0, 0, 0, 0]))[1]
    assert f[3] == pytest.approx(0.0, abs=1e-9) and f[4] == pytest.approx(2.0)
    #facing along the track, sliding left at 0.5 m/s in its own frame
    f = track_features(line, pose(5.0, 0.0, 0.0), np.array([3.0, 0.5, 0, 0, 0, 0]))[1]
    assert f[3] == pytest.approx(3.0) and f[4] == pytest.approx(0.5)


def test_arc_length_sampling_wraps_around():
    line = square()
    assert line.length == pytest.approx(40.0)
    assert line.point_at(12.5) == pytest.approx([10.0, 2.5])
    assert line.point_at(45.0) == pytest.approx([5.0, 0.0])        # one lap later = same point


def test_lookahead_across_the_start_line_in_car_frame():
    line = square()
    car = pose(0.0, 0.5, -np.pi / 2)            # 0.5 m before the start line, facing -y
    s = track_features(line, car, np.zeros(6))[0]
    assert s == pytest.approx(39.5)
    ahead = lookahead_features(line, car, s).reshape(-1, 2)
    assert ahead[0] == pytest.approx([0.5, 0.5])   # 1 m ahead = world (0.5, 0): 0.5 m forward, 0.5 m left
    assert ahead[-1] == pytest.approx([0.5, 7.5])    # 8 m ahead is already round the corner


# ---- the wrapper and the obs modes ----------

def test_wrapper_output_matches_its_space():
    for lookahead, keys in ((False, {'track': 5}), (True, {'track': 5, 'lookahead': 10})):
        env = TrackFeatureWrapper(FakeEnv(), lookahead=lookahead, centerline=square())
        env.env.next_obs = {'pose': pose(5.0, 0.3, 0.2), 'velocity': np.array([2.0, 0.1, 0, 0, 0, 0]),
                            'time': np.array(0.0, np.float32)}
        obs, _ = env.reset()
        for key, n in keys.items():
            assert obs[key].shape == (n,) and obs[key].dtype == np.float32
        assert env.observation_space.contains(obs)
        assert obs['track'][0] == pytest.approx(0.3 / 1.5)        # lateral offset / LATERAL_SCALE


def test_wrapper_raises_instead_of_using_zeros(tmp_path):
    env = TrackFeatureWrapper(FakeEnv(), centerline=square())
    with pytest.raises(ValueError):
        env.observation({'pose': np.full(6, np.nan), 'velocity': np.zeros(6)})
    with pytest.raises(FileNotFoundError):
        Centerline.load(str(tmp_path / 'missing.npz'))


def test_obs_mode_flags():
    assert resolve_obs_mode(False, None) == 'base'
    assert resolve_obs_mode(True, None) == 'lidar' # old --lidar also works
    with pytest.raises(ValueError):
        resolve_obs_mode(True, 'track')  # --lidar --obs-mode track makes no sense


# ------real Austria centre line -------------------

real = pytest.mark.skipif(not os.path.exists(CENTERLINE_FILE), reason='centre line not built')


@real
def test_real_track_projection_never_jumps_to_another_part_of_the_track():
    """Project every 3rd drivable pixel and compare with racecar_gym's own progress map.
    Snapping to the other leg of the hairpin would be off by >10% of a lap; we allow 2% because
    the progress map is blocky."""
    map_file = racecar_gym_map_file()
    assert str(np.load(CENTERLINE_FILE)['source_sha256']) == file_sha256(map_file), (
        f'{map_file} is not the map the centre line was built from; run python tools/build_centerline.py')
    maps = np.load(map_file)
    drivable, progress_map = maps['drivable_area'], maps['norm_distance_from_start']
    px, py = np.nonzero(drivable)
    px, py = px[::3], py[::3]
    xs, ys = -50 + (py + 0.5) * 0.05, -50 + (2000 - px - 0.5) * 0.05  # racecar_gym GridMap pixel centres

    line = Centerline.load()
    rows = (2000 - (line.points[:, 1] + 50) / 0.05).astype(int)
    cols = ((line.points[:, 0] + 50) / 0.05).astype(int)
    line_progress = np.unwrap(progress_map[rows, cols] * 2 * np.pi) / (2 * np.pi)
    worst, worst_lateral = 0.0, 0.0
    for x, y, p in zip(xs, ys, progress_map[px, py]):
        s, lateral, _ = line.project(x, y)
        expected = np.interp(s, line.s, line_progress) % 1.0
        worst = max(worst, abs((p - expected + 0.5) % 1.0 - 0.5))
        worst_lateral = max(worst_lateral, abs(lateral))
    assert worst < 0.02
    assert worst_lateral < 1.5          
    # = LATERAL_SCALE, so the scaled offset stays within [-1, 1]


@real
def test_evaluation_sees_the_same_observation_as_training():
    """Same raw observation -> same action, whether it goes through the training env wrappers
    (reset) or through SB3EvalDriver (what evaluate_sb3.py uses)."""
    from stable_baselines3 import DQN
    rng = np.random.default_rng(0)
    line = Centerline.load()
    for obs_mode in ('track', 'track-lookahead'):
        env = FakeEnv()
        driver = SB3Driver(env, model_class=DQN, wrappers=dqn_wrappers(obs_mode=obs_mode), buffer_size=100, seed=0)
        eval_driver = SB3EvalDriver(driver)
        actions = DiscreteActionWrapper(env)._actions
        for i in rng.integers(0, len(line.points), 5):
            x, y = line.points[i] + rng.normal(scale=0.3, size=2)
            raw = {'pose': pose(x, y, rng.uniform(-np.pi, np.pi)), 'velocity': rng.normal(size=6),
                   'time': np.array(1.0, dtype=np.float32)}
            env.next_obs = raw
            expected = actions[int(driver.model.predict(driver.env.reset()[0], deterministic=True)[0])]
            got = eval_driver.act(raw)
            assert got['motor'][0] == expected['motor'][0] and got['steering'][0] == expected['steering'][0]
            eval_driver.reset()


def test_old_lidar_configs_still_evaluate(tmp_path):
    """config.json files from before --obs-mode only have 'lidar' / 'action_repeat'."""
    from evaluate_sb3 import check_against_training
    folder = tmp_path / 'checkpoints' / 'old_run'
    (folder / 'eval').mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'lidar': True, 'action_repeat': 4}))
    model = folder / 'eval' / 'old_run_140000_steps.zip'
    check_against_training(model, True, 4)               
    with pytest.raises(ValueError):
        check_against_training(model, False, 4, obs_mode='track')
