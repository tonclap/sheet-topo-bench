"""The two remaining PROTOCOL §7 baselines: self-intersection and the ray
heuristic (TOPO-009).

§7 declares four baselines; random ranking and the normal-kink detector ship
with the metric (`sheet_erl.py`), and these two ride with the detector stack
because they need its machinery:

- **Self-intersection (windcheck class).** Two non-adjacent segments of one
  row's polyline crossing in the plane. The classic mesh sanity check: a switch
  that doubles back or a merger that folds the trace can produce one; a clean
  spiral cannot.
- **Ray winding-order inversion (the wave-2 heuristic).** Along a ray from the
  centre, crossings sorted by radius must advance monotonically in accumulated
  turn. An inversion is a winding-count disagreement — the global signature
  wave 2 built its ladder on.

Each baseline is reported **twice**: raw, and differenced against the substrate
atlas built from the *same* channel on the pristine grids — the same subtraction
detector v1 gets. That isolates what the comparison is about: if a baseline
plus differencing matched v1, the feature would be irrelevant and only the
subtraction would matter; the honest table has to be able to show that.

Usage:

    python baselines.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --report ../../output/topo/baselines_paris4.json --bands dev
"""
import argparse
import json
import os
import sys

import numpy as np

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v1                                                      # noqa: E402

N_ANGLES = 72            # v0's bundle: 5 degrees apart
MIN_INDEX_GAP = 3        # closer segments share a node's neighbourhood, not a fold
TURN_NOISE = 0.02        # pos inversions smaller than this are interpolation noise


def row_polyline(grid, row, centre):
    """(points, turns, cols) of one row, or None if too short."""
    mask = (grid[row, :, 0] != -1) & (grid[row, :, 1] != -1)
    if mask.sum() < 30:
        return None
    try:
        points, turns, _ = scrolls.turn_profile(grid, row, mask, centre)
    except ValueError:
        # No counting centre at this row's height (PHerc0139 rows outside the
        # innermost labelled turn's z range): the row contributes nothing,
        # same as a row too short to measure.
        return None
    return points.astype(np.float64), turns, np.where(mask)[0]


def cross2(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def self_intersections(grid, row, centre):
    """(col, 1.0) per polyline self-crossing of this row.

    Candidate pairs come from a vectorised bounding-box overlap over all
    segment pairs (the polyline is short enough that the n^2 boolean mask is
    cheap), then the surviving pairs get the exact orientation test.
    """
    parts = row_polyline(grid, row, centre)
    if parts is None:
        return []
    points, _, cols = parts
    a = points[:-1, :2]
    b = points[1:, :2]
    n = len(a)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    overlap = ((lo[:, None, 0] <= hi[None, :, 0])
               & (lo[None, :, 0] <= hi[:, None, 0])
               & (lo[:, None, 1] <= hi[None, :, 1])
               & (lo[None, :, 1] <= hi[:, None, 1]))
    index = np.arange(n)
    overlap &= (index[None, :] - index[:, None]) >= MIN_INDEX_GAP
    i, j = np.where(overlap)
    if len(i) == 0:
        return []
    p, q, r, s = a[i], b[i], a[j], b[j]
    d1 = cross2(q - p, r - p)
    d2 = cross2(q - p, s - p)
    d3 = cross2(s - r, p - r)
    d4 = cross2(s - r, q - r)
    crossing = (d1 * d2 < 0) & (d3 * d4 < 0)
    out = []
    for ii, jj in zip(i[crossing], j[crossing]):
        out.append((int(cols[ii]), 1.0))
        out.append((int(cols[jj]), 1.0))
    return out


def ray_inversions(grid, row, centre):
    """(col, 1.0) per winding-order inversion along the ray bundle."""
    parts = row_polyline(grid, row, centre)
    if parts is None:
        return []
    points, turns, cols = parts
    z = float(np.median(points[:, 2]))
    origin = np.array(centre.at(z), dtype=float)
    reach = float(np.max(np.linalg.norm(points[:, :2] - origin, axis=1))) + 10.0
    position = np.arange(len(points), dtype=float)
    out = []
    for k in range(N_ANGLES):
        angle = 2.0 * np.pi * k / N_ANGLES
        unit = np.array([np.cos(angle), np.sin(angle)])
        crossings = scrolls.ray_crossings(points, position, origin, unit, reach)
        if len(crossings) < 3:
            continue
        crossings.sort()
        pos = np.array([c[1] for c in crossings])
        # `position` is the polyline index; convert to turn to judge direction.
        turn_at = np.interp(pos, position, turns)
        steps = np.diff(turn_at)
        direction = np.sign(np.median(steps)) or 1.0
        for i, step in enumerate(steps):
            if step * direction < -TURN_NOISE:
                for p in (pos[i], pos[i + 1]):
                    index = int(np.clip(round(p), 0, len(cols) - 1))
                    out.append((int(cols[index]), 1.0))
    return out


CHANNELS = {'selfx': self_intersections, 'rayinv': ray_inversions}


def channel_cells(grid, centre, channel):
    """Evidence cells (row, block) -> (mass, best col, best score) of one grid."""
    evidence, best = {}, {}
    for row in range(grid.shape[0]):
        for col, s in CHANNELS[channel](grid, row, centre):
            key = (row, col // detect_v1.BLOCK)
            evidence[key] = evidence.get(key, 0.0) + s
            if s > best.get(key, (None, -1.0))[1]:
                best[key] = (col, s)
    return evidence, best


def clusters_of(evidence, best):
    """Connected components of evidence cells: same rules as detect_v1."""
    parent = {key: key for key in evidence}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for row, blk in evidence:
        for dr in range(-detect_v1.CELL_ROW_GAP, detect_v1.CELL_ROW_GAP + 1):
            for db in (-1, 0, 1):
                other = (row + dr, blk + db)
                if other != (row, blk) and other in evidence:
                    parent[find(other)] = find((row, blk))
    clusters = {}
    for key in evidence:
        clusters.setdefault(find(key), []).append(key)
    return clusters


def candidates_of(name, evidence, best, atlas=None):
    """Ranked candidates; with `atlas`, differenced like detect_v1's clusters."""
    dilated = (None if atlas is None else
               {(r + dr, b) for r, b in atlas for dr in (-1, 0, 1)})
    out = []
    for cells in clusters_of(evidence, best).values():
        rows = [r for r, _ in cells]
        if max(rows) - min(rows) + 1 > detect_v1.MAX_CLUSTER_ROWS:
            continue
        if dilated is not None and sum(c in dilated for c in cells) * 2 > len(cells):
            continue
        mass = sum(evidence[c] for c in cells)
        top = max(cells, key=lambda c: best[c][1])
        out.append((name, top[0], best[top][0], mass))
    return out


def evaluate(ranking, scoped, pristine, voxel_mm, row_step, erl_broken):
    injections = scoped['injections']
    n = len(injections)
    result = sheet_erl.evaluate_ranking(ranking, scoped, pristine, voxel_mm,
                                        row_step, erl_broken=erl_broken)
    outcomes = sheet_erl.hits(ranking[:n], injections, pristine)
    found = {injections[h]['id'] for h, *_ in outcomes if h is not None}
    by_type = {}
    for t in 'SMH':
        of_type = [r for r in injections if r['type'] == t]
        by_type[t] = (sum(1 for r in of_type if r['id'] in found) / len(of_type)
                      if of_type else None)
    return {'metrics': result, 'recall_by_type': by_type}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--row-step', type=int, default=1)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)

    def in_scope(record):
        if args.bands == 'dev':
            return record['winding_low'] < 100
        if args.bands == 'heldout':
            return record['winding_low'] >= 100
        return True

    injections = [r for r in manifest['injections'] if in_scope(r)]
    scoped = dict(manifest, injections=injections)
    names = sorted({r['segment'] for r in injections})
    n = len(injections)

    per_channel = {ch: {'raw': [], 'atlas': []} for ch in CHANNELS}
    stats = {ch: {'corrupted_cells': 0, 'atlas_cells': 0} for ch in CHANNELS}
    pristine = {}
    for name in names:
        corrupted = np.load(os.path.join(args.corpus, 'grids', f'{name}.npy'))
        pristine[name] = scrolls.segment_grid(name, scroll, args.grid_cache)
        for ch in CHANNELS:
            evidence, best = channel_cells(corrupted, centre, ch)
            atlas, _ = channel_cells(pristine[name], centre, ch)
            stats[ch]['corrupted_cells'] += len(evidence)
            stats[ch]['atlas_cells'] += len(atlas)
            per_channel[ch]['raw'] += candidates_of(name, evidence, best)
            per_channel[ch]['atlas'] += candidates_of(name, evidence, best,
                                                      atlas=set(atlas))
        print(f'{name}: done', flush=True)

    zones = {name: sheet_erl.segment_zones(scoped, name) for name in names}
    erl_broken, *_ = sheet_erl.sheet_erl(pristine, zones, voxel_mm, args.row_step)

    report = {'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n, 'cells': stats}
    for ch in CHANNELS:
        for variant in ('raw', 'atlas'):
            ranked = sorted(per_channel[ch][variant], key=lambda t: -t[3])
            ranking = [(nm, row, col, mass) for nm, row, col, mass in ranked][:4 * n]
            report[f'{ch}_{variant}'] = dict(
                evaluate(ranking, scoped, pristine, voxel_mm, args.row_step,
                         erl_broken),
                candidates=len(per_channel[ch][variant]))
            m = report[f'{ch}_{variant}']['metrics']
            print(f"{ch}/{variant}: AP {m['ap']:.4f}, "
                  f"recall@N {m['recall_at_n']:.3f}, "
                  f"candidates {report[f'{ch}_{variant}']['candidates']}",
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
