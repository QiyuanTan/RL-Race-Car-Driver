"""Builds the Austria centre line from the map and saves it for later use.

There isn't a ready-made centre line file, so we use maps.npz from
models/scenes/austria/maps/.
drivable_area tells us where the car can drive.
norm_distance_from_start gives the order around the track, from 0 to 1.
Pixel coordinates follow core/gridmaps.py: 2000x2000, 0.05 m per pixel,
origin at -50, -50.

First find each drivable pixel's distance from the nearest wall.
Split the track into 600 progress bins. In each bin, take pixels with
at least 95% of the largest wall distance and average their positions.
This gives us a point near the middle of that part of the track.

Skip bins with too few pixels, then smooth the closed line 3 times
with a roughly 0.9 m window. Resample it every 0.1 m along the line.
Check the result before saving. If a check fails, stop.

s measures distance along the line in the driving direction.
It starts near the start/finish line and wraps around after one lap.
Save the points and source details in scenarios/austria_centerline.npz.

Run from the repo root:
python tools/build_centerline.py
Add --plot results/track_geometry/centerline.png to save a plot too.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from Agents.track_features import CENTERLINE_FILE, centerline_sha256, racecar_gym_map_file  # noqa: E402

RES, ORIGIN, SIZE = 0.05, -50.0, 2000      # from models/scenes/austria/austria.yml
N_BINS, SPACING, SMOOTH, PASSES = 600, 0.1, 9, 3


def wall_distance(drivable):
    """Distance in metres from every drivable pixel to the nearest non-drivable pixel (0 elsewhere).

    Same result as scipy.ndimage.distance_transform_edt(drivable) * RES, but numpy only (scipy crashed
    on one of our Macs). The nearest non-drivable pixel is always one right next to the track, so we
    only compare against those "wall" pixels.
    """
    track = np.pad(drivable, 1)                         # border counts as wall
    touching = np.zeros_like(track)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            touching |= np.roll(np.roll(track, dr, axis=0), dc, axis=1)
    wr, wc = np.nonzero(touching & ~track)
    tr, tc = np.nonzero(track)
    dist = np.zeros(track.shape)
    for i in range(0, len(tr), 500):                     # chunks, to keep memory small
        d2 = (tr[i:i + 500, None] - wr) ** 2 + (tc[i:i + 500, None] - wc) ** 2
        dist[tr[i:i + 500], tc[i:i + 500]] = np.sqrt(d2.min(axis=1))
    return dist[1:-1, 1:-1] * RES


def pixel_centres(px, py):
    """Inverse of racecar_gym GridMap.to_pixel, at the pixel centre."""
    return ORIGIN + (py + 0.5) * RES, ORIGIN + (SIZE - px - 0.5) * RES


def world_to_pixel(x, y):
    """Same as racecar_gym GridMap.to_pixel."""
    return (SIZE - (np.asarray(y) - ORIGIN) / RES).astype(int), ((np.asarray(x) - ORIGIN) / RES).astype(int)


def resample_closed(points, spacing):
    closed = np.vstack([points, points[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(round(s[-1] / spacing))
    new_s = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(new_s, s, closed[:, 0]), np.interp(new_s, s, closed[:, 1])], axis=1)


def smooth_closed(points, window):
    kernel = np.ones(window) / window
    pad = window // 2
    wrapped = np.vstack([points[-pad:], points, points[:pad]])
    return np.stack([np.convolve(wrapped[:, k], kernel, mode='valid') for k in range(2)], axis=1)


def build(map_file):
    maps = np.load(map_file)
    drivable, progress = maps['drivable_area'], maps['norm_distance_from_start']
    wall_dist = wall_distance(drivable)                 # metres to the nearest non-drivable pixel

    px, py = np.nonzero(drivable)
    x, y = pixel_centres(px, py)
    p, d = progress[px, py], wall_dist[px, py]
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    counts = np.bincount(bins, minlength=N_BINS)
    raw = []
    for b in range(N_BINS):
        inside = bins == b
        if counts[b] < 0.5 * np.median(counts):
            # the progress map grows from a point near (0, 0), so the first bins only cover a small
            # blob instead of a full cross-section of the track; their middle is not the track middle
            continue
        best = inside & (d >= 0.95 * d[inside].max())
        raw.append((x[best].mean(), y[best].mean()))
    raw = np.array(raw)                       # ordered by progress, starts at the start line

    points = resample_closed(raw, SPACING)
    for _ in range(PASSES):
        points = smooth_closed(points, SMOOTH)
    points = resample_closed(points, SPACING)
    points = np.roll(points, -int(np.argmin(np.hypot(*(points - raw[0]).T))), axis=0)   # s=0 at the start line
    method = (f'progress bins={N_BINS} ({int(np.sum(counts < 0.5 * np.median(counts)))} partial bins skipped), '
              f'ridge>=95%, moving average {SMOOTH} pts x{PASSES}, spacing {SPACING} m')
    return points, raw, method, (drivable, progress, wall_dist)


def check(points, drivable, progress, wall_dist):
    """Raise if the centre line looks wrong. Returns numbers for the report."""
    rx, ry = world_to_pixel(points[:, 0], points[:, 1])
    closed = np.vstack([points, points[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    heading = np.arctan2(*np.diff(closed, axis=0)[:, ::-1].T)
    turn = np.abs(np.angle(np.exp(1j * np.diff(np.append(heading, heading[0])))))
    prog = progress[rx, ry]
    dprog = np.diff(np.append(prog, prog[0]))
    wraps = int(np.sum(dprog < -0.5))
    n = len(points)
    # closest approach between parts of the line that are > 3 m apart along the track (hairpin legs)
    gap = np.inf
    for i in range(n):
        far = np.abs((np.arange(n) - i + n // 2) % n - n // 2) * SPACING > 3.0
        gap = min(gap, np.hypot(*(points[far] - points[i]).T).min())
    report = dict(points=n, length_m=float(seg.sum()), spacing_min=float(seg.min()), spacing_max=float(seg.max()),
                  min_wall_distance_m=float(wall_dist[rx, ry].min()), all_drivable=bool(drivable[rx, ry].all()),
                  max_turn_per_step_deg=float(np.degrees(turn.max())), progress_wraps=wraps,
                  biggest_progress_step_back=float(-dprog[dprog > -0.5].min()),
                  start_heading_deg=float(np.degrees(heading[0])), closest_other_leg_m=float(gap))
    problems = []
    if not report['all_drivable'] or report['min_wall_distance_m'] < 0.5:
        problems.append('centre line too close to a wall / off the track')
    if report['spacing_max'] > 1.5 * SPACING:
        problems.append('gap in the centre line')
    if wraps != 1 or report['biggest_progress_step_back'] > 0.01:
        problems.append('progress along the centre line is not increasing (wrong order or direction)')
    if abs(report['start_heading_deg']) > 20:
        problems.append('centre line does not start in the grid direction (+x)')
    if gap < 1.0:
        problems.append('two parts of the centre line are closer than 1 m (possible branch/shortcut)')
    if problems:
        raise RuntimeError('; '.join(problems) + f'\n{report}')
    return report


def file_hash(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', default=None, help='also save a picture of the centre line')
    args = parser.parse_args()

    map_file = racecar_gym_map_file()
    points, raw, method, (drivable, progress, wall_dist) = build(map_file)
    report = check(points, drivable, progress, wall_dist)
    for key, value in report.items():
        print(f'{key:28s} {value}')
    np.savez(CENTERLINE_FILE, points=points.astype(np.float64), source='racecar_gym/models/scenes/austria/maps/maps.npz',
             source_sha256=file_hash(map_file), method=method)
    print('saved', CENTERLINE_FILE, 'points sha256', centerline_sha256(points)[:12])

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ext = [ORIGIN, ORIGIN + SIZE * RES, ORIGIN, ORIGIN + SIZE * RES]
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.imshow(np.where(drivable, wall_dist, np.nan), extent=ext, cmap='Greys', origin='upper')
        ax.plot(raw[:, 0], raw[:, 1], 'b.', ms=2, label='bin centres (raw)')
        ax.plot(points[:, 0], points[:, 1], 'r-', lw=1, label='centre line')
        for s in range(0, int(report['length_m']), 10):
            i = int(s / SPACING)
            ax.annotate(f's={s}', points[i], color='darkred', fontsize=8)
        ax.arrow(*points[0], *(points[5] - points[0]) * 4, width=0.15, color='g', label='driving direction')
        ax.set_xlim(-9, 19), ax.set_ylim(-22, 2), ax.set_aspect('equal'), ax.legend(loc='lower left')
        ax.set_title(f"Austria centre line from maps.npz ({report['length_m']:.1f} m, s=0 at start line)")
        os.makedirs(os.path.dirname(args.plot) or '.', exist_ok=True)
        fig.savefig(args.plot, dpi=90, bbox_inches='tight')
        print('saved', args.plot)


if __name__ == '__main__':
    main()
