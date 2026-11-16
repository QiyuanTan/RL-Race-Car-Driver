"""Our non-RL baseline. No training needed.

I use the LiDAR scan to find an open gap ahead, then steer toward a point inside it.
 Sharp turns and nearby walls lower the target speed.

Also,I added an early-turn rule for the Austria hairpin: when the wall ahead
gets close, aim toward the more open side. See BASELINE.md for the comparison
with plain FTG.

Reference for the general approach:
https://github.com/gonultasbu/f1tenth_bullet/blob/main/gap_follower.py

A few environment details matter here: LiDAR runs from left to right,but positive 
steering turns right. Motor is throttle/brake, not speed,and negative motor does not reverse the car.
"""

import numpy as np

from Agents.Driver import Driver

# Module-level constants: shared by all FTG drivers in this file.these first four come from the car/sensor settings in racecar.yml.
LIDAR_ANGLE_START = -2.36  #rad, before converting to our left-positive angles
LIDAR_FOV = 4.71  #rad, about 270 degrees across the scan
LIDAR_MAX_RANGE = 15.0  #m
MAX_STEER = 0.21  #rad, wheel angle at full steering

# algo settings kept fixed during tuning.
# The six settings I varied are in __init__ below.
VIEW_ANGLE_DEG = 90  #use the front half of the scan: +/-90 degrees
SMOOTH_WINDOW = 5  # avg 5 nearby rays to reduce noise
RANGE_CAP = 8.0  # m, no need to distinguish distances beyond this
FRONT_CONE_DEG = 10  # +/-10 degrees counts as directly ahead
FULL_SPEED_CLEARANCE = 6.0  # m ahead needed to allow max speed
TURN_IN_ANGLE_DEG = 45  # target direction for early turns, not wheel angle
TURN_IN_SECTOR_DEG = (30, 75)  # compare these angles on each side


class FollowTheGapDriver(Driver):

    def __init__(self, env=None,
                 bubble_radius=0.4,  #m,space to avoid around the nearest obstacle
                 gap_threshold=1.2,  #m,a ray must reach farther than this to count as open
                 max_speed=3.0,  #m/s, target speed on a clear straight
                 min_speed=0.8,  # m/s, target speed for sharp turns
                 turn_slowdown_deg=10,  #target angle at which speed drops to min_speed
                 turn_in_distance=3.5):  #m, start turning early below this; 0 disables it
        
        #defaults were tuned on Austria seeds 0-7.
        if env is not None:
            super().__init__(env)

        #each driver keeps its own settings, so experiments can override them.
        self.bubble_radius = bubble_radius
        self.gap_threshold = gap_threshold
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.turn_slowdown_deg = turn_slowdown_deg
        self.turn_in_distance = turn_in_distance

        #keep the latest targets here for our future debugging and tests.
        self.last_target_angle = 0.0
        self.last_target_speed = 0.0

    def reset(self):
        """no driving history to clear; each action uses the current scan."""
        pass

    def act(self, obs):
        """read LiDAR and velocity, then return motor and steering commands."""
        angles = self.ray_angles(len(obs['lidar']))
        ranges = self.preprocess(obs['lidar'])
        in_view = np.abs(angles) <= np.deg2rad(VIEW_ANGLE_DEG)

        no_gap = False
        if not np.any(ranges[in_view] > 0):
            #no usable distances ahead,fall back to straight and slow
            target_angle = 0.0
            no_gap = True
        else:
            bubbled = self.apply_bubble(ranges, angles, in_view)
            free = in_view & (bubbled > self.gap_threshold)
            gap = self.largest_gap(free)
            if gap is None:#=no gap pass the threshold. Aim at the farthest ray, slowly.
                view_idx = np.flatnonzero(in_view)
                target_angle = angles[view_idx[np.argmax(ranges[view_idx])]]
                no_gap = True
            else:
                start, end = gap
                target_angle = angles[self.pick_target(bubbled, start, end)]
                target_angle = self.turn_in(target_angle, ranges, angles)

        # Our angles are positive left; the simulator's steering is positive right.
        # Anything beyond the wheel limit becomes full steering.
        steering = np.clip(-target_angle / MAX_STEER, -1.0, 1.0)

        target_speed = self.target_speed(target_angle, ranges, angles, no_gap)
        vx, vy = obs['velocity'][0], obs['velocity'][1]
        speed = np.hypot(vx, vy)

        #below target speed -> throttle; above it -> brake.
        motor = np.clip(target_speed - speed, -1.0, 1.0)

        self.last_target_angle = float(target_angle)
        self.last_target_speed = float(target_speed)

        # environment expects one-element arrays.
        return {'motor': np.array([motor], dtype=np.float32),
                'steering': np.array([steering], dtype=np.float32)}

    @staticmethod
    def ray_angles(n_rays):
        """Convert ray indices to radians: 0 ahead, posittive left."""
        return -(LIDAR_ANGLE_START + np.arange(n_rays) * (LIDAR_FOV / n_rays))

    @staticmethod
    def preprocess(scan):
        """clean bad readings, smooth the scan, and cap far distances."""
        scan = np.asarray(scan, dtype=np.float64)

        #treat NaN/negative readings as blocked, and +inf as no nearby hit.
        scan = np.nan_to_num(scan, nan=0.0, posinf=LIDAR_MAX_RANGE, neginf=0.0)
        scan = np.clip(scan, 0.0, LIDAR_MAX_RANGE)

        #repeat the edge values so the 1st/last rays also get a full window.
        half = SMOOTH_WINDOW // 2
        padded = np.pad(scan, half, mode='edge')

        # Moving average: kept convolve to preserve the original rounding.
        scan = np.convolve(padded, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode='valid')
        return np.minimum(scan, RANGE_CAP)

    def apply_bubble(self, ranges, angles, in_view):
        """block directions around the nearest obstacle before choosing a gap."""
        ranges = ranges.copy()
        view_idx = np.flatnonzero(in_view)
        closest = view_idx[np.argmin(ranges[view_idx])]
        d = max(ranges[closest], 1e-3)  # avoid dividing by zero

        # bubble radius in metres --->(convertto) an angle around the obstacle.
        if d <= self.bubble_radius:
            half_width = np.pi / 2
        else:
            half_width = np.arcsin(self.bubble_radius / d)

        ranges[np.abs(angles - angles[closest]) <= half_width] = 0.0
        return ranges

    @staticmethod
    def largest_gap(free):
        """return the longest True run as inclusive (start, end), or none."""
        best, best_len = None, 0
        start = None

        # Extra False lets us finish a gap that reaches the end of the scan.
        for i, is_free in enumerate(list(free) + [False]):
            if is_free and start is None:
                start = i
            elif not is_free and start is not None:
                # Equal-length gaps keep the first one found.
                if i - start > best_len:
                    best, best_len = (start, i - 1), i - start
                start = None

        return best

    @staticmethod
    def pick_target(ranges, start, end):
        """Aim halfway between the gap centre and its farthest ray."""
        segment = ranges[start:end + 1]

        # Capping distances creates ties. Take the middle tied ray;
        # argmax alone would keep choosing the leftmost one.
        tied = np.flatnonzero(segment >= segment.max() - 1e-6)
        farthest = start + tied[len(tied) // 2]
        centre = (start + end) // 2
        return (farthest + centre) // 2

    def turn_in(self, target_angle, ranges, angles):
        """If a wall is close ahead, turn early toward the more open side."""
        if self.turn_in_distance <= 0:
            return target_angle

        front = ranges[np.abs(angles) <= np.deg2rad(FRONT_CONE_DEG)]
        if front.min() >= self.turn_in_distance:
            return target_angle

        low, high = np.deg2rad(TURN_IN_SECTOR_DEG[0]), np.deg2rad(TURN_IN_SECTOR_DEG[1])
        sector = (np.abs(angles) >= low) & (np.abs(angles) <= high)
        left_space = ranges[sector & (angles > 0)].mean()
        right_space = ranges[sector & (angles < 0)].mean()
        side = 1.0 if left_space > right_space else -1.0  # left positive; ties go right

        turn_in_angle = np.deg2rad(TURN_IN_ANGLE_DEG)
        if np.sign(target_angle) == side and abs(target_angle) >= turn_in_angle:
            return target_angle  # already aiming far enough toward that side

        return side * turn_in_angle

    def target_speed(self, target_angle, ranges, angles, no_gap):
        """< target speed for turns and nearby walls."""
        if no_gap:
            return self.min_speed * 0.5

        #both factors are capped at 1
        turn = min(1.0, abs(target_angle) / np.deg2rad(self.turn_slowdown_deg))
        front = ranges[np.abs(angles) <= np.deg2rad(FRONT_CONE_DEG)]
        clearance = min(1.0, front.min() / FULL_SPEED_CLEARANCE)

        #full speed needs both a straight target and enough room ahead
        return self.min_speed + (self.max_speed - self.min_speed) * (1.0 - turn) * clearance