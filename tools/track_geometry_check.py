"""Check the track features on the real track: drive one lap with Follow-the-Gap and plot them.

Saves (default folder results/track_geometry/):
  geometry_views.png   5 close-ups (start, straight, normal turn, hairpin, start/finish seam) with the
                       centre line, the car, its projection, the track tangent and the 5 look-ahead points
  s_along_lap.png      arc length s and the track features over the lap (any wrong match = a jump)
  check.json           ranges of every feature, clip count, biggest jumps in s

FTG only drives here (it needs the LiDAR); the features are computed with the same
TrackFeatureWrapper code as in DQN training, from the pose/velocity observations.
Usage (repo root):  python tools/track_geometry_check.py [--seed 1000]
"""
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from Agents.FollowTheGapDriver import FollowTheGapDriver  # noqa: E402
from Agents.track_features import (LOOKAHEAD_FEATURES, LOOKAHEAD_SCALE, TRACK_FEATURES,  # noqa: E402
                                   TrackFeatureWrapper, track_features)
from evaluate_baseline import ENV_ID  # noqa: E402


def drive(seed):
    env = gym.make(ENV_ID, render_mode='rgb_array_birds_eye')
    features = TrackFeatureWrapper(env, lookahead=True)       # only used for its observation() here
    driver = FollowTheGapDriver(env)
    obs, info = env.reset(seed=seed, options={'mode': 'grid'})
    rows, done, truncated = [], False, False
    while not (done or truncated):
        out = features.observation(obs)
        s, _ = track_features(features.centerline, obs['pose'], obs['velocity'])
        rows.append(dict(pose=obs['pose'].copy(), track=out['track'], lookahead=out['lookahead'], s=s,
                         progress=info['progress'], lap=info['lap'], time=info['time']))
        obs, _, done, truncated, info = env.step(driver.act(obs))
        if info['lap'] >= 2:
            break
    env.close()
    return rows, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--out', default=os.path.join(REPO, 'results', 'track_geometry'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, wrapper = drive(args.seed)
    line = wrapper.centerline
    s = np.array([r['s'] for r in rows])
    track = np.array([r['track'] for r in rows])
    ahead = np.array([r['lookahead'] for r in rows])
    step = (np.diff(s) + line.length / 2) % line.length - line.length / 2
    lap_done = rows[-1]['lap'] >= 2 or rows[-1]['progress'] > 0.99
    check = {'seed': args.seed, 'sim_steps': len(rows), 'lap_completed': bool(lap_done),
             'sim_time': float(rows[-1]['time']),
             'biggest_forward_jump_in_s_m': float(step.max()), 'at_s_m': float(s[np.argmax(step)]),
             'at_xy': [round(float(v), 2) for v in rows[int(np.argmax(step))]['pose'][:2]],
             'biggest_backward_step_in_s_m': float(step.min()),
             'clipped_values': wrapper.clipped, 'values': wrapper.values_seen,
             'track_ranges': {n: [float(track[:, i].min()), float(track[:, i].max())] for i, n in enumerate(TRACK_FEATURES)},
             'lookahead_ranges': {n: [float(ahead[:, i].min()), float(ahead[:, i].max())]
                                  for i, n in enumerate(LOOKAHEAD_FEATURES)}}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'check.json'), 'w') as f:
        json.dump(check, f, indent=2)
    print(json.dumps({k: v for k, v in check.items() if 'ranges' not in k}, indent=2))

    # --- close-ups: first moment the car is near each place (by arc length) ---
    places = [('start (grid)', None), ('straight', 5.0), ('normal turn', 10.5), ('hairpin', 30.0),
              ('start/finish seam', line.length - 0.5)]
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.6))
    for ax, (name, target) in zip(axes, places):
        i = 0 if target is None else int(np.argmin(np.abs((s - target + line.length / 2) % line.length - line.length / 2)
                                                   + 1e3 * (np.arange(len(s)) < 50)))
        r = rows[i]
        x, y, yaw = r['pose'][0], r['pose'][1], r['pose'][5]
        lateral = r['track'][0] * 1.5
        proj = line.point_at(r['s'])
        tangent = line.point_at(r['s'] + 0.6) - proj
        pts = r['lookahead'].reshape(-1, 2) * LOOKAHEAD_SCALE                 # car frame
        world = np.c_[x + np.cos(yaw) * pts[:, 0] - np.sin(yaw) * pts[:, 1],
                      y + np.sin(yaw) * pts[:, 0] + np.cos(yaw) * pts[:, 1]]  # back to world, to plot
        ax.plot(line.points[:, 0], line.points[:, 1], '-', color='0.6', lw=1, label='centre line')
        ax.plot(*np.array([[x, y], proj]).T, 'b--', lw=1, label=f'projection (lateral {lateral:+.2f} m)')
        ax.arrow(*proj, *tangent, width=0.04, color='g', label='track tangent')
        ax.arrow(x, y, 0.6 * np.cos(yaw), 0.6 * np.sin(yaw), width=0.04, color='k', label='car heading')
        ax.plot(world[:, 0], world[:, 1], 'o', color='orange', ms=6, label='look-ahead 1,2,3,5,8 m')
        for d, w in zip((1, 2, 3, 5, 8), world):
            ax.annotate(f'{d}', w, fontsize=8)
        ax.plot(x, y, 'ks', ms=7)
        ax.set_xlim(x - 9, x + 9), ax.set_ylim(y - 9, y + 9), ax.set_aspect('equal')
        err = np.degrees(np.arctan2(r['track'][1], r['track'][2]))
        ax.set_title(f'{name}\ns={r["s"]:.1f} m, t={r["time"]:.1f} s, heading err {err:+.0f} deg', fontsize=10)
        ax.legend(fontsize=7, loc='lower left')
    fig.savefig(os.path.join(args.out, 'geometry_views.png'), dpi=80, bbox_inches='tight')

    t = np.array([r['time'] for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, s), axes[0].set_ylabel('s along centre line [m]')
    axes[0].plot(t, np.array([r['progress'] for r in rows]) * line.length, '--', lw=0.8,
                 label='racecar_gym progress x length (blocky map)')
    axes[0].legend()
    for i, n in enumerate(TRACK_FEATURES):
        axes[1].plot(t, track[:, i], label=n)
    axes[1].legend(fontsize=8), axes[1].set_ylabel('scaled track features')
    for i in range(0, 10, 2):
        axes[2].plot(t, ahead[:, i], label=LOOKAHEAD_FEATURES[i])
    axes[2].legend(fontsize=8), axes[2].set_ylabel('look-ahead x (forward) / 8 m'), axes[2].set_xlabel('time [s]')
    fig.savefig(os.path.join(args.out, 's_along_lap.png'), dpi=80, bbox_inches='tight')
    print('saved to', args.out)


if __name__ == '__main__':
    main()
