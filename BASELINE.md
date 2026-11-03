# Non-RL baseline: LiDAR Follow-the-Gap (FTG)

`Agents/FollowTheGapDriver.py` is a rule-based driver for racecar_gym. It follows the team `Driver`
interface: `act(obs)` returns `{'motor': ..., 'steering': ...}`. No training is needed.

## How to run

```bash
python -m pytest tests/                                          # 14 unit tests, no simulator needed
python evaluate_baseline.py --episodes 10 --seed 100 --workers 4  # evaluate the default settings
python evaluate_baseline.py --episodes 10 --seed 100 --workers 4 --param turn_in_distance=0  # plain FTG
python evaluate_baseline.py --episodes 1 --render                 # watch it drive (needs a display)
```

Results are saved to `results/baseline_ftg/ftg_<date>_<time>.json` (parameters, summary, every episode).

## How the driver works

Every step:

1. **Clean the scan**: NaN -> 0, +inf -> 15 m, average each ray with its 4 neighbours, cap at 8 m,
   and keep only the front +/-90 deg.
2. **Safety bubble**: find the closest point and blank out the rays within 0.4 m of it, so the car
   does not aim right next to an obstacle.
3. **Find the gap**: the longest run of rays farther than 1.2 m.
4. **Pick a target**: halfway between the centre of the gap and its farthest ray.
5. **Turn-in rule**: if the wall straight ahead is closer than 3.5 m, aim at least 45 deg toward the
   side (30-75 deg) with more free space.
6. **Steer and set speed**: steering = -target_angle / 0.21 (clipped to [-1, 1]). The target speed goes
   from 3.0 m/s (straight and open) down to 0.8 m/s (sharp turn or wall close ahead). The motor command is
   target speed minus current speed, so the car pushes when too slow and brakes when too fast.

The structure follows the GapFollower in
<https://github.com/gonultasbu/f1tenth_bullet/blob/main/gap_follower.py>; the code and parameters
were written for racecar_gym (nothing copied).

### Why the turn-in rule is needed

Plain FTG scraped the outer wall at the exit of the Austria hairpin (around x = 15, y = -20.5).
- The car's wheelbase is 0.325 m and full steering lock is only 0.21 rad, so the smallest turning
  radius is about 0.325 / tan(0.21) ≈ 1.5 m. The logged path matched this (~1.56 m).
- Plain FTG only turns once it can *see* the hairpin exit. By then, even at full lock, a 1.5 m circle
  reaches the outer wall.
- Driving slower does not help, because the radius is set by the steering limit, not by speed.
  The only fix is to start turning earlier, which is what the turn-in rule does.

## racecar_gym facts (checked in the source code and with small experiments)

| item | value |
|---|---|
| `lidar` | 1080 rays over 270 deg, metres, [0.25, 15]. Ray 0 ≈ 135 deg **left**, ray 541 straight ahead, ray 1079 ≈ 135 deg **right**. Updated every 4 steps. |
| `steering` | [-1, 1], **+1 = right**. Full lock = 0.21 rad wheel angle. |
| `motor` | [-1, 1], how hard to push (a throttle, not a speed); negative brakes. **No reverse.** |
| time | 0.01 s per step, 120 s limit per episode; hitting a wall does not end the episode. |

## Evaluation

- Same start for every episode (`mode='grid'`). Seed i fixes the LiDAR noise.
- Lap time = simulated time when `info['lap']` reaches 2. Collisions = number of separate wall contacts.
- Smoothness = average absolute change of the motor (or steering) command between two consecutive steps.
  **Lower = smoother.** It is measured per simulation step, so every method must output an action every
  step for a fair comparison (frame skip / action repeat would make a method look smoother than it is).
- Each episode runs in a fresh process, so the same seed always gives the same result.
- Parameters were tuned on seeds 0-7; the results below use seeds 100-109, which were not used for tuning.
- These metrics are a minimal version. They should be aligned with the team metrics (issue #4).

## Results (Austria, seeds 100-109)

All three settings were run on the same machine (M1 Mac, Python 3.9, conda-forge pybullet).

| setting | laps completed | collision-free laps | collisions (mean) | lap time (mean ± std) | motor smoothness | steering smoothness | file |
|---|---|---|---|---|---|---|---|
| **default (with turn-in rule)** | 10/10 | **10/10** | 0.0 | **81.77 ± 0.77 s** | 0.0451 | 0.0410 | `ftg_20261102_190856.json` |
| plain FTG (`turn_in_distance=0`) | 10/10 | 2/10 | 1.3 | 68.56 ± 0.35 s | 0.0494 | 0.0177 | `ftg_20261102_191031.json` |
| default + `max_speed=4.0` | 10/10 | 10/10 | 0.0 | 77.64 ± 0.71 s | 0.0456 | 0.0415 | `ftg_20261102_191159.json` |

- The turn-in rule removes the hairpin contacts but costs about 13 s per lap.
- The turn-in rule makes steering about twice as jumpy (0.041 vs 0.018), because it switches the target to
  45 deg at once when a wall gets close. It trades steering smoothness for safety.
- `max_speed=4.0` was also collision-free on these seeds and about 4 s faster. It is not the default
  because it was not chosen on the tuning seeds. Picking it now would mean tuning on the test seeds.
- Plain FTG is right at the limit at the hairpin, so its collision count depends on the machine: the same
  seeds gave 8/10 collision-free laps on Linux (pybullet 3.2.5) but 2/10 on the M1 Mac. The default setting
  was collision-free on both.

## Limitations

- Only tested on Austria with the grid start.
- The turn-in distance (3.5 m) was tuned for this track.
- The car cannot reverse. If it ever gets stuck against a wall, it stays there until the time limit.
- Results differ between machines and pybullet versions (e.g. seed 100: 81.34 s on Linux / pybullet 3.2.5,
  81.77 s on the M1 Mac; plain FTG collisions differ even more, see above). Compare all methods on one machine.
- The driver only reads `lidar` and `velocity` from the observation, never `info`.
