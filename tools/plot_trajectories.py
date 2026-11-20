"""Plot where evaluation episodes went, and check whether they really got past the hairpin.

progress > 35% alone does not prove the hairpin was passed: the apex itself is at progress 0.351
(racecar_gym's progress map is not proportional to distance; the apex is at s = 29.9 of 94.2 m).
Here we project each recorded trajectory point onto the centre line and follow the arc length s:
an episode passed the hairpin if it got at least 3 m (along the track) beyond the apex. Evaluation-only analysis; the policy never sees this.

Usage (repo root):  python tools/plot_trajectories.py results/test_ar4/dqn_track_ar4_<time>.json [more.json]
Saves <result>.png next to each result file.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from Agents.track_features import Centerline  # noqa: E402

HAIRPIN_APEX_S = 29.9       # m, sharpest point of the centre line; racecar_gym progress there = 0.351
PASSED_MARGIN = 3.0         # m beyond the apex


def furthest_s(line, trajectory):
    """Furthest arc length reached, following s continuously (no jump at the start line)."""
    s = np.array([line.project(x, y)[0] for x, y in trajectory])
    steps = (np.diff(s) + line.length / 2) % line.length - line.length / 2
    return float(s[0] + np.max(np.concatenate([[0.0], np.cumsum(steps)])))


def main(paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    line = Centerline.load()
    for path in paths:
        data = json.load(open(path))
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(*np.vstack([line.points, line.points[:1]]).T, color='0.75', lw=6, label='centre line')
        ax.plot(*line.point_at(HAIRPIN_APEX_S), 'r*', ms=14, label='hairpin apex')
        print(path)
        for e in data['episodes']:
            traj = np.array(e.get('trajectory_every_10_steps') or [])
            if len(traj) == 0:
                print(f"  seed {e['seed']}: no trajectory recorded (older evaluation file)")
                continue
            reach = furthest_s(line, traj)
            passed = reach >= HAIRPIN_APEX_S + PASSED_MARGIN
            ax.plot(traj[:, 0], traj[:, 1], lw=1, label=f"seed {e['seed']} ({e['end']})")
            ax.plot(*traj[-1], 'kx')
            print(f"  seed {e['seed']:4d}  end={e['end']:11s} progress={e['progress']:.3f}  furthest s={reach:5.1f} m"
                  f"  passed hairpin={passed}  stuck={e['stuck_last_10s']}  final speed={e['final_speed']}")
        ax.set_aspect('equal'), ax.legend(fontsize=6, loc='lower left')
        ax.set_title(f"{data.get('obs_mode', '?')}  {os.path.basename(data['model'])}", fontsize=9)
        out = os.path.splitext(path)[0] + '.png'
        fig.savefig(out, dpi=80, bbox_inches='tight')
        print('  saved', out)


if __name__ == '__main__':
    main(sys.argv[1:])
