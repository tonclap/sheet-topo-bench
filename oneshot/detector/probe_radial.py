"""TOPO-045 mechanism probe: does the radial self-residual see the 22 uncovered?

Session 23 measured (from the frozen TOPO-026 probe artefacts, no new CT
reads) that 16 of the 17 uncovered M sit at papyrus-level CT (median peak1
113-136 vs 121 on pristine traces): the injector parks the merged trace in
zones of *natural sheet contact*, where contact evidence is indistinguishable
from the substrate atlas and no dark gap exists. What still distinguishes an
injected arc there is not contact but *displacement*: the arc leaves the
smooth radial trend of its own row by half a pitch (M) or a full pitch (S)
and returns, while naturally pressed sheets are smooth on both sides. This
axis is orthogonal to vjump (displacement against the rows above/below) and
to all seven v5 features. Branch U-017; donor frame is the seismic
mis-tie/loop-tie family (ANGLES_2026-08-18.md paragraph 3, move 1).

Per the TOPO-023/026 lesson, no channel is built here: the probe measures the
benefit side alone. Entirely offline: corpus grids + pristine grid cache +
counting centre; no CT volume, no network.

**Declared before the run, not tuned on the result:**

- Signal: per row of a grid, r(col) = distance of the node to the counting
  centre in its own z-plane (``detect_v1.radial_map``). The trend is a
  running median of r over the row's *valid* nodes within a column index
  half-window W, at two declared scales ``W in (24, 96)`` (an excursion of
  column-length L is suppressed by the median for L < W; injected windows
  observed in the corpus span ~5-35 columns, lambda up to 10 mm). The node
  evidence value is ``max_W |r - trend_W|``.
- Threshold: one global number per scale, ``T_W = q99`` of |r - trend_W|
  over all valid nodes of all scoped *pristine* grids, computed before any
  window is looked at and published with both distributions. A node is
  evidence iff |r - trend_W| >= T_W at any scale.
- Cells and atlas differencing, v1's own shapes: evidence nodes accumulate
  into (row, col // BLOCK) cells; the same computation on the pristine grid
  gives the substrate residual atlas, and a corrupted cell that coincides
  with an atlas cell is dropped (the mesh's natural waviness is not our
  injection). Surviving cells merge into clusters by 8-adjacency on the
  (row, block) lattice; a cluster counts iff its node mass >= 4
  (sensitivity at masses 2 and 8 is published alongside, not used by the
  rule).
- Windows: each injection window from the corpus manifest grown by
  (+-2 rows, +-8 cols) — the probe_ct margins. A window is covered iff at
  least one surviving cluster's cells intersect it.
- **Decision rule: the channel is worth building iff at least 8 of the 22
  uncovered-union windows (>= 8/22, the probe_ct one-third convention)
  contain a surviving cluster.** Below that the probe's answer is negative,
  TOPO-045 closes with outcome (b), and the remainder falls to U-018.
- Published alongside (no part of the rule): coverage of all 232 scoped
  windows by type (S should fire — full-pitch excursions; H reads zero by
  construction — holes carry no nodes), per-segment pristine cluster counts
  (the false-alarm floor a future channel would pay), and the mass
  sensitivity table.

Determinism: pure numpy, no RNG, no network; a rerun is bit-identical.

**Dated amendment, 18.08.2026, declared BEFORE the v2 rerun (v1 run recorded
first, report kept as probe_radial_paris4_v1centre.json).** The v1 estimator
(|r - trend| against the counting centre) has no power: its pristine null is
dominated by model error, not mesh waviness — q50 at scale 24 is already
11.9 vx (half a pitch), q50 at scale 96 is 145 vx, thresholds 325/803 vx.
Mechanism: ``radial_map`` uses one median z per row for ``centre.at``, while
a row's chain wanders up to 165 slices in z (SESSION_HANDOFF, session 19),
so axis tilt turns into tens of voxels of apparent radial wobble; median
windows spanning invalid gaps mix geometrically distant nodes on top. That
is outcome (v) of TOPO-045 for the v1 estimator — a probe defect, not an
answer about the mechanism. The v2 estimator drops the centre entirely:

- Signal v2: per row, each coordinate (x, y, z) of the node polyline is
  smoothed by the same running median (scales unchanged, 24/96), but the
  window is confined to the node's *contiguous valid run* (a hole splits the
  row; windows never span holes). The node evidence value is the euclidean
  norm ``|p - median_W(p)|`` — displacement of the node from its own row's
  smoothed 3D path. An injected arc is displaced by half a pitch (M) or a
  full pitch (S); natural waviness is measured by the same pristine null.
- Everything else — quantile Q, cells, atlas differencing, cluster floors,
  margins, and the decision rule (>= 8 of 22) — unchanged.

Usage (from oneshot/detector/):

    python probe_radial.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --coverage ../../output/topo/coverage_breakdown_paris4.json \
        --report ../../output/topo/probe_radial_paris4.json
"""
import argparse
import collections
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
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import detect_v1 as v1                                                # noqa: E402

SCALES = (24, 96)         # trend half-windows, columns (declared)
Q = 99.0                  # pristine quantile for the per-scale threshold
MASS_MIN = 4              # cluster node-mass floor (sensitivity: 2, 8)
ROW_MARGIN = 2            # window growth, rows  (probe_ct's margins)
COL_MARGIN = 8            # window growth, cols


def run_residual(points, w):
    """3D displacement from the running componentwise median, one valid run."""
    n = len(points)
    med = np.empty_like(points)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        med[i] = np.median(points[lo:hi], axis=0)
    return np.linalg.norm(points - med, axis=1)


def grid_residuals(grid, centre):
    """dict scale -> per-node |residual| array (NaN where invalid).

    v2 estimator (see the dated amendment above): centre-free 3D
    self-residual per contiguous valid run; ``centre`` is unused but kept in
    the signature so the two protocol versions stay call-compatible.
    """
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    out = {w: np.full(grid.shape[:2], np.nan) for w in SCALES}
    for row in range(grid.shape[0]):
        cols = np.where(valid[row])[0]
        if len(cols) < 5:
            continue
        # split the row into contiguous valid runs; windows never span holes
        breaks = np.where(np.diff(cols) > 1)[0]
        for seg in np.split(cols, breaks + 1):
            if len(seg) < 5:
                continue
            pts = grid[row, seg, :3].astype(np.float64)
            for w in SCALES:
                out[w][row, seg] = run_residual(pts, w)
    return out


def evidence_cells(residuals, thresholds):
    """(row, block) -> node mass of threshold exceedances (any scale)."""
    hit = None
    for w in SCALES:
        h = residuals[w] >= thresholds[w]   # NaN compares False: invalid mute
        hit = h if hit is None else (hit | h)
    cells = collections.Counter()
    rows, cols = np.where(hit)
    for row, col in zip(rows, cols):
        cells[(int(row), int(col) // v1.BLOCK)] += 1
    return cells


def clusters(cells, mass_min):
    """8-adjacent components of cells; keep total mass >= mass_min."""
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            r0, b0 = cur
            for dr in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nxt = (r0 + dr, b0 + db)
                    if nxt in cells and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
        mass = sum(cells[c] for c in comp)
        if mass >= mass_min:
            out.append((comp, mass))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--coverage', required=True,
                        help='coverage_breakdown_paris4.json (uncovered ids)')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.coverage, encoding='utf-8') as f:
        uncovered = {r['id'] for r in
                     json.load(f)['uncovered_union']['rows']}
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)

    injections = [r for r in manifest['injections'] if r['winding_low'] < 100]
    names = sorted({r['segment'] for r in injections})

    # pass 1 — pristine: thresholds first, then atlas cells per segment
    residual_samples = {w: [] for w in SCALES}
    pristine_res = {}
    for name in names:
        grid = scrolls.segment_grid(name, scroll, args.grid_cache)
        res = grid_residuals(grid, centre)
        pristine_res[name] = res
        for w in SCALES:
            vals = res[w][np.isfinite(res[w])]
            residual_samples[w].append(vals)
    thresholds = {w: float(np.percentile(np.concatenate(residual_samples[w]), Q))
                  for w in SCALES}
    pristine_dist = {w: {'q50': float(np.median(np.concatenate(residual_samples[w]))),
                         'q90': float(np.percentile(np.concatenate(residual_samples[w]), 90)),
                         'q99': thresholds[w]}
                     for w in SCALES}

    atlas_cells = {name: evidence_cells(pristine_res[name], thresholds)
                   for name in names}
    atlas = {name: set(atlas_cells[name]) for name in names}
    pristine_clusters = {name: {str(m): len(clusters(atlas_cells[name], m))
                                for m in (2, MASS_MIN, 8)}
                         for name in names}

    # pass 2 — corrupted: surviving clusters per segment
    surviving = {}
    for name in names:
        path = os.path.join(args.corpus, 'grids', f'{name}.npy')
        grid = np.load(path)
        res = grid_residuals(grid, centre)
        cells = evidence_cells(res, thresholds)
        diff = {c: n for c, n in cells.items() if c not in atlas[name]}
        surviving[name] = {m: clusters(diff, m) for m in (2, MASS_MIN, 8)}

    def window_covered(rec, mass):
        r0, r1 = rec['row_lo'] - ROW_MARGIN, rec['row_hi'] + ROW_MARGIN
        c0, c1 = rec['col_lo'] - COL_MARGIN, rec['col_hi'] + COL_MARGIN
        b0, b1 = c0 // v1.BLOCK, c1 // v1.BLOCK
        for comp, _ in surviving[rec['segment']][mass]:
            for row, blk in comp:
                if r0 <= row <= r1 and b0 <= blk <= b1:
                    return True
        return False

    coverage = {m: {'uncovered': [], 'by_type': collections.Counter(),
                    'all_by_type': collections.Counter(),
                    'all_total': collections.Counter()}
                for m in (2, MASS_MIN, 8)}
    for rec in injections:
        for m in coverage:
            cov = window_covered(rec, m)
            coverage[m]['all_total'][rec['type']] += 1
            if cov:
                coverage[m]['all_by_type'][rec['type']] += 1
            if rec['id'] in uncovered and cov:
                coverage[m]['uncovered'].append(rec['id'])
                coverage[m]['by_type'][rec['type']] += 1

    n_unc = coverage[MASS_MIN]['uncovered']
    report = {
        'question': 'TOPO-045: does the radial self-residual cover the '
                    'uncovered-union injections?',
        'protocol': {'scales': SCALES, 'quantile': Q, 'mass_min': MASS_MIN,
                     'margins': [ROW_MARGIN, COL_MARGIN],
                     'rule': 'build iff >= 8 of 22 uncovered windows covered '
                             'at mass_min'},
        'thresholds_vx': thresholds,
        'pristine_residual_dist': pristine_dist,
        'pristine_cluster_counts': pristine_clusters,
        'uncovered_windows': 22,
        'covered_of_uncovered': sorted(n_unc),
        'n_covered_of_uncovered': len(n_unc),
        'by_type_of_uncovered': dict(coverage[MASS_MIN]['by_type']),
        'all_windows_coverage': {
            str(m): {t: [int(coverage[m]['all_by_type'][t]),
                         int(coverage[m]['all_total'][t])]
                     for t in ('S', 'M', 'H')}
            for m in (2, MASS_MIN, 8)},
        'mass_sensitivity': {str(m): len(coverage[m]['uncovered'])
                             for m in (2, MASS_MIN, 8)},
        'verdict': ('BUILD' if len(n_unc) >= 8 else 'NEGATIVE'),
    }
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in
                      ('thresholds_vx', 'n_covered_of_uncovered',
                       'covered_of_uncovered', 'mass_sensitivity', 'verdict')},
                     indent=2))


if __name__ == '__main__':
    main()
