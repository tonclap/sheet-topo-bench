"""Detector v0: topology errors from the ray ladder (TOPO-009).

The target's central hypothesis (pipeline/TARGET.md): a sheet switch can be locally
perfect — smooth surface, continuous normal — while being globally impossible. The
detector therefore works in the global frame the wave-2 code already knows how to
compute, not in the local patch.

**The dead first variant, kept on the record (PROTOCOL §9).** v0 first scored the
residual of radius against a running median along the grid row. Measured on the real
substrate, the row parameterisation itself defeats it: columns sit ~48 voxels apart
with a radial jitter of 14-40 voxels per step, while the anomaly is ~25 voxels —
the feature drowns in the mesh's own wiggle (open problem № 4 in the flesh). AP on
the dev corpus: 0.0000. The lesson matches DEV-069's: integrate the polyline, do not
trust its columns.

**The live variant: the ray ladder.** For each grid row, rays from the centre cross
the row's polyline; crossings sorted by radius form a ladder whose rungs are one
sheet pitch apart (wave 2 measured: 120 crossings for 120 windings, returns 0-1).
An injected switch parks the trace on the neighbour's rung — two crossings collide;
a merger parks it mid-gap — two half-gaps. Score = (median gap - gap) / median gap
at the collision, attributed to the exact polyline position `ray_crossings` already
interpolates. Holes stay a separate channel (interior invalid runs). Channels are
merged by percentile rank, so a hole cannot bury a collision or vice versa.

Development runs restrict to the dev bands of PROTOCOL §6 (windings < 100 on
Paris 4); the held-out shot is spent once, later, not here.

Usage:

    python detect_v0.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --report ../../output/topo/detector_v0_paris4.json --bands dev
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
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402

N_ANGLES = 72            # 5 degrees apart; the narrowest injected arc spans ~13
COLLISION_MIN = 0.3      # collisions below this score are ladder noise, not errors


def ladder_scores(grid, row, mask, centre):
    """(col, score) collision anomalies of one row, from the ray ladder."""
    points = grid[row][mask].astype(np.float64)
    cols = np.where(mask)[0]
    if len(cols) < 30:
        return []
    z = float(np.median(points[:, 2]))
    origin = np.array(centre.at(z), dtype=float)
    reach = float(np.max(np.linalg.norm(points[:, :2] - origin, axis=1))) + 10.0
    position = np.arange(len(points), dtype=float)
    anomalies = []
    for k in range(N_ANGLES):
        angle = 2.0 * np.pi * k / N_ANGLES
        unit = np.array([np.cos(angle), np.sin(angle)])
        crossings = scrolls.ray_crossings(points, position, origin, unit, reach)
        if len(crossings) < 3:
            continue
        crossings.sort()
        radii = np.array([c[0] for c in crossings])
        pos = np.array([c[1] for c in crossings])
        gaps = np.diff(radii)
        # Pitch varies with radius (compression zones), so a ladder-wide median
        # mislabels every tight zone as a collision. Each gap is judged against the
        # median of its neighbours instead — the sheet's own local pitch. Measured on
        # the pristine substrate, single gaps do not separate signal from noise (16 %
        # of clean gaps sit in the merger band, 11 % in the missing band): the number
        # each gap contributes here is only evidence, and the detector's signal is
        # its *coherence* across rays and rows, aggregated in `segment_channels`.
        for i, gap in enumerate(gaps):
            around = [gaps[j] for j in range(max(0, i - 3), min(len(gaps), i + 4))
                      if j != i]
            local = float(np.median(around)) if around else 0.0
            if local <= 1.0:
                continue
            ratio = gap / local
            if ratio < 0.2:               # polyline return through the ray: noise
                continue
            merger = max(0.0, 1.0 - abs(ratio - 0.5) / 0.3)     # peak at half-gap
            missing = min(1.0, max(0.0, ratio - 1.0))           # peak at doubled
            score = max(merger, missing)
            if score <= 0.0:
                continue
            for p in (pos[i], pos[i + 1]):
                index = int(np.clip(round(p), 0, len(cols) - 1))
                anomalies.append((int(cols[index]), float(score)))
    return anomalies


def hole_scores(grid, row):
    """(col centre, size) of interior invalid runs in this row."""
    valid = (grid[row, :, 0] != -1) & (grid[row, :, 1] != -1)
    cols = np.where(valid)[0]
    if len(cols) < 2:
        return []
    holes = []
    gaps = np.where(np.diff(cols) > 1)[0]
    for g in gaps:
        lo, hi = int(cols[g]), int(cols[g + 1])
        holes.append(((lo + hi) // 2, float(hi - lo - 1)))
    return holes


def segment_channels(name, grid, centre, row_reach=4, block=8):
    """Aggregated per-channel candidates of one corrupted grid.

    Measured on the pristine substrate, a single anomalous gap is worthless: the
    clean ladder is that irregular. What separates an injection is coherence — the
    same anomaly at the same place across neighbouring rays and rows. Per-gap
    evidence is therefore summed over a (±row_reach rows) x (±1 col-block) window,
    and the window's score is that mass. The reported column is the exact position
    of the strongest single contribution, so narrow errors survive the pooling.
    """
    evidence = {}       # (row, block) -> summed score
    best = {}           # (row, block) -> (col, top single score)
    holes = {}
    for row in range(grid.shape[0]):
        mask = (grid[row, :, 0] != -1) & (grid[row, :, 1] != -1)
        if mask.sum() >= 30:
            for col, score in ladder_scores(grid, row, mask, centre):
                key = (row, col // block)
                evidence[key] = evidence.get(key, 0.0) + score
                if score > best.get(key, (None, -1.0))[1]:
                    best[key] = (col, score)
        for col, size in hole_scores(grid, row):
            key = (row, col // block)
            if size > holes.get(key, (None, -1.0))[1]:
                holes[key] = (col, size)
    pooled = {}
    for (row, blk), _mass in evidence.items():
        total = sum(evidence.get((r, b), 0.0)
                    for r in range(row - row_reach, row_reach + row + 1)
                    for b in (blk - 1, blk, blk + 1))
        pooled[(row, blk)] = total
    out = []
    for (row, blk), mass in pooled.items():
        col = best[(row, blk)][0]
        out.append((name, row, col, 'ladder', mass))
    for (row, _blk), (col, size) in holes.items():
        out.append((name, row, col, 'hole', size))
    return out


def merge_channels(candidates, top):
    """One ranking from several channels, by within-channel percentile rank."""
    ranked = []
    for channel in {c[3] for c in candidates}:
        of_channel = sorted((c for c in candidates if c[3] == channel),
                            key=lambda t: t[4])
        n = len(of_channel)
        for i, (name, row, col, _channel, _score) in enumerate(of_channel):
            ranked.append((name, row, col, (i + 1) / n))
    ranked.sort(key=lambda t: -t[3])
    return ranked[:top]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--bands', choices=('dev', 'all'), default='dev',
                        help="dev = windings < 100 only (PROTOCOL §6); 'all' is "
                             "reserved for the single held-out shot")
    parser.add_argument('--row-step', type=int, default=1)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)

    def in_scope(record):
        return args.bands == 'all' or record['winding_low'] < 100

    injections = [r for r in manifest['injections'] if in_scope(r)]
    scoped = dict(manifest, injections=injections)
    names = sorted({r['segment'] for r in injections})
    n = len(injections)

    corrupted = {name: np.load(os.path.join(args.corpus, 'grids', f'{name}.npy'))
                 for name in names}
    pristine = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
                for name in names}

    candidates = []
    for name in names:
        candidates += segment_channels(name, corrupted[name], centre)
        print(f'{name}: {len(candidates)} candidates so far', flush=True)
    ranking = merge_channels(candidates, top=4 * n)

    zones = {name: sheet_erl.segment_zones(scoped, name) for name in names}
    erl_broken, *_ = sheet_erl.sheet_erl(pristine, zones, voxel_mm, args.row_step)
    result = sheet_erl.evaluate_ranking(ranking, scoped, pristine, voxel_mm,
                                        args.row_step, erl_broken=erl_broken)
    by_type = {}
    outcomes = sheet_erl.hits(ranking[:n], injections, pristine)
    found = {injections[h]['id'] for h, *_ in outcomes if h is not None}
    for t in 'SMH':
        of_type = [r for r in injections if r['type'] == t]
        by_type[t] = (sum(1 for r in of_type if r['id'] in found) / len(of_type)
                      if of_type else None)
    plausible = [r for r in injections if r.get('plausible')]
    recall_plausible = (sum(1 for r in plausible if r['id'] in found)
                        / len(plausible)) if plausible else None

    report = {'detector': 'v0 ray-ladder collisions + hole-edge',
              'dead_variant': 'radius-vs-running-median residual: AP 0.0000 on '
                              'this corpus; row parameterisation jitter (14-40 vx '
                              'per ~48 vx column step) drowns a ~25 vx anomaly',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'metrics': result,
              'recall_by_type': by_type,
              'recall_on_locally_plausible': recall_plausible}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f"v0 on {args.bands} bands: AP {result['ap']:.4f}, "
          f"recall@N {result['recall_at_n']:.3f}, "
          f"dERL {result['delta_erl_mm']:+.2f} mm")
    print("recall by type: " + ', '.join(
        f'{t}={v:.3f}' if v is not None else f'{t}=n/a'
        for t, v in by_type.items()))
    if recall_plausible is not None:
        print(f"recall on locally-plausible injections: {recall_plausible:.3f}")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
