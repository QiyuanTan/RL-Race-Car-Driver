"""Tests with fake LiDAR scans. Run: python -m pytest tests/"""
import os
import sys

import numpy as np

#add the repo root to find Agents
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agents.FollowTheGapDriver import FollowTheGapDriver  # noqa: E402

#shared scan size for these tests, matching the simulator.
N = 1080
def make_obs(scan, speed=1.0):
    """Build an observation with forward speed and the given scan."""
    # Only vx is nonzero; the driver uses vx and vy to get speed.
    return {'lidar': np.array(scan, dtype=float), 'velocity': np.array([speed, 0, 0, 0, 0, 0])}


def corridor_with_gap(gap_angle_deg, wall=0.8, gap_width_deg=10):
    """Put walls around the car, with one opening centred at gap_angle_deg."""
    #angles are in degrees, positive left. gap_width_deg is the HALF-width
    #most rays hit a wall at `wall` metres; rays in the opening reach 10 m
    angles = FollowTheGapDriver.ray_angles(N)
    scan = np.full(N, wall)
    scan[np.abs(angles - np.deg2rad(gap_angle_deg)) <= np.deg2rad(gap_width_deg)] = 10.0
    return scan


def plain_ftg():
    """Disable early turns so these tests isolate the basic gap logic."""
    return FollowTheGapDriver(turn_in_distance=0)


def test_ray_directions_match_racecar_gym():
    # check the left-to-right scan order assumed by the driver
    angles = FollowTheGapDriver.ray_angles(N)
    # left
    assert np.isclose(np.rad2deg(angles[0]), 135, atol=1)  
    # ahead
    assert abs(np.rad2deg(angles[541])) < 0.5  
    # right
    assert np.isclose(np.rad2deg(angles[-1]), -135, atol=1) 


def test_steers_toward_the_gap():
    driver = plain_ftg()
    left = driver.act(make_obs(corridor_with_gap(+40)))
    right = driver.act(make_obs(corridor_with_gap(-40)))
    # Steering signs = opposite to our LiDAR angles.
    assert left['steering'][0] < 0
    assert right['steering'][0] > 0


def test_gap_straight_ahead_gives_almost_no_steering():
    driver = plain_ftg()
    action = driver.act(make_obs(corridor_with_gap(0, wall=1.0)))
    #allow a small offset because the scan uses discrete rays
    assert abs(action['steering'][0]) < 0.1


def test_safety_bubble_blanks_rays_around_closest_point():
    driver = plain_ftg()
    angles = FollowTheGapDriver.ray_angles(N)
    ranges = np.full(N, 5.0)
    ranges[541] = 1.0  
    # measn one obstacle 1 m ahead

    bubbled = driver.apply_bubble(ranges, angles, np.ones(N, dtype=bool))
    half_width = np.arcsin(driver.bubble_radius / 1.0)
    inside = np.abs(angles - angles[541]) <= half_width

    # clear bubble, but leave other dist alone
    assert np.all(bubbled[inside] == 0) and np.all(bubbled[~inside] == 5.0)


def test_largest_gap():
    # True = open. Indices 4-7 form the longest opening
    free = [False, True, True, False, True, True, True, True, False, True]
    assert FollowTheGapDriver.largest_gap(free) == (4, 7)
    assert FollowTheGapDriver.largest_gap([False, False]) is None


def test_tied_farthest_rays_pick_the_middle():
    ranges = np.zeros(20)
    ranges[5:15] = 8.0  # all rays in this gap are equally far
    target = FollowTheGapDriver.pick_target(ranges, 5, 14)

    # Either centre ray is fine; choosing an edge would bias the steering
    assert target in (9, 10)


def test_no_gap_brakes():
    driver = plain_ftg()
    # Walls are only 0.5 m away, and the car is still moving at 2 m/s.
    action = driver.act(make_obs(np.full(N, 0.5), speed=2.0))
    assert action['motor'][0] < 0


def test_bad_lidar_values_still_give_valid_actions():
    driver = plain_ftg()
    for bad in (np.nan, np.inf, -5.0):
        scan = corridor_with_gap(0)
        scan[::7] = bad  # replace every seventh reading
        action = driver.act(make_obs(scan))

        # Bad readings should not produce NaN or out-of-range commands.
        for value in (action['motor'][0], action['steering'][0]):
            assert np.isfinite(value) and -1 <= value <= 1


def test_sharper_turn_means_lower_speed():
    driver = plain_ftg()
    driver.act(make_obs(corridor_with_gap(0)))
    straight_speed = driver.last_target_speed

    driver.act(make_obs(corridor_with_gap(45)))
    turning_speed = driver.last_target_speed
    assert turning_speed < straight_speed


def test_turn_in_rule_turns_toward_open_side_near_a_wall():
    #wall ahead at 2.5 m, with more room on the right.
    angles = FollowTheGapDriver.ray_angles(N)
    scan = np.full(N, 1.5)
    scan[np.abs(angles) <= np.deg2rad(25)] = 2.5

    right_front = (angles <= -np.deg2rad(30)) & (angles >= -np.deg2rad(75))
    scan[right_front] = 6.0

    with_rule = FollowTheGapDriver(turn_in_distance=3.5).act(make_obs(scan))
    assert with_rule['steering'][0] > 0  #positive steering turn right