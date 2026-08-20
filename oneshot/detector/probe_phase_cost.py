"""TOPO-049 cost probe: the phase channel's pristine false-alarm floor (U-024).

TOPO-048 measured the phase signature's benefit (node farther than half the
local half-period from every bright-run centre): 13/22 uncovered windows,
11/17 invisible M. This probe measures the price — the "trace between runs"
background on the pristine substrate of ragged production meshes, which the
window probe never touched. Stage order is declared in U-024: price first;
the full channel (corrupted grids, atlas differencing, 232-window coverage,
the shipping rule against the exam form) only through the gate below, under
its own dated declaration.

**Declared before the run, not tuned on the result:**

- Evidence: probe_phase.py's construction verbatim — radial CT profiles
  -36..+36 vx step 1 along the in-plane radial unit vector at the node's own
  z; ``T = 80.0`` (frozen TOPO-026 calibration); bright runs of length >= 2;
  period = median spacing of run centres, mute if < 2 runs; node evidence
  iff its distance to the *nearest* run centre >= 0.5 of the half-period.
- Scope: the **pristine** grids of all 16 dev segments; rows of the MIDDLE
  z-quantile only (index 2 of the manifest's 5) within SUPPORT_BAND_VX;
  every valid column. One quantile of five prices the run at ~1/5 of the
  eventual channel scope; the extrapolation below is declared, not fitted.
- Cells (row, col // BLOCK); clusters via ``differenced_clusters`` with an
  EMPTY atlas (this run *is* the atlas side); only the MAX_CLUSTER_ROWS = 14
  cut applies (the v1 default, as in probe_phase).
- **Gate (the decision): the full channel is worth building iff the median
  per-segment pristine cluster count, extrapolated x5 (five quantiles),
  is <= 40** — about twice the v1 pool's per-segment median (304 candidates
  / 16 segments = 19). Published alongside (no part of the rule): the
  per-node evidence rate, mute share, cluster-mass distribution, and the
  gate at bar multipliers x1.5 / x3.
- Below the gate the probe's answer is negative: U-024 closes as an honest
  boundary ("the floor outprices the benefit"), the W1 remainder falls to
  U-022.

Checkpoints: per-segment pickle of raw node stats — the run resumes, and
evidence replays offline at any floor.

Usage (from oneshot/detector/):

    python probe_phase_cost.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --report ../../output/topo/probe_phase_cost_paris4.json \
        --ckpt ../../output/topo/ckpt_phase_cost
"""
import argparse
import json
import os
import pickle
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
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)
from probe_ct import CTVolume                                         # noqa: E402
from probe_phase import OFFSETS, node_stats, PHASE_DIAG_FLOOR         # noqa: E402

QUANTILE_INDEX = 2        # the middle of the manifest's five z-quantiles
EXTRAPOLATION = 5         # declared: one quantile prices 1/5 of the scope
GATE_MAX_MEDIAN = 40      # ~2x the v1 pool's per-segment median (304/16)


def segment_phase_nodes(ct, grid, centre, z_quantile, ckpt):
    """Raw per-node phase stats for the mid-quantile band rows, checkpointed."""
    key = {'offsets': [float(t) for t in OFFSETS], 'quantile': float(z_quantile)}
    if os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') == key:
            return payload['nodes']
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    heights, _ = scrolls.row_heights(grid)
    band_rows = [row for row in range(grid.shape[0])
                 if np.isfinite(heights[row])
                 and abs(heights[row] - z_quantile) <= v1.SUPPORT_BAND_VX]
    out = {'row': [], 'col': [], 'phase': [], 'runs': []}
    for k, row in enumerate(band_rows):
        mask = valid[row]
        if not mask.any():
            continue
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)
        z_med = np.median(points[:, 2])
        try:
            cx, cy = centre.at(float(z_med))
        except ValueError:
            continue
        radial = points[:, :2] - np.array([cx, cy])
        norm = np.linalg.norm(radial, axis=1)
        ok = norm > 1e-6
        if not ok.any():
            continue
        points, cols = points[ok], cols[ok]
        u = radial[ok] / norm[ok][:, None]
        stacked = np.concatenate([
            np.concatenate([points[:, :2] + t * u, points[:, 2:3]], 1)
            for t in OFFSETS])
        sampled = ct.values(stacked).reshape(len(OFFSETS), len(points))
        for j, col in enumerate(cols):
            _thick, _centred, phase, n_runs = node_stats(
                sampled[:, j].astype(float))
            out['row'].append(int(row))
            out['col'].append(int(col))
            out['phase'].append(phase)
            out['runs'].append(int(n_runs))
        if (k + 1) % 10 == 0 or k + 1 == len(band_rows):
            print(f'  row {k + 1}/{len(band_rows)} '
                  f'({len(out["col"])} nodes probed)', flush=True)
    payload = {'key': key, 'nodes': out}
    tmp = f'{ckpt}.{os.getpid()}.part'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp, ckpt)
    return out


def phase_clusters(nodes):
    """Clusters of the declared phase evidence; returns (n, masses, rates)."""
    evidence, best = {}, {}
    n_nodes = len(nodes['col'])
    n_evidence = n_mute = 0
    for row, col, phase in zip(nodes['row'], nodes['col'], nodes['phase']):
        if phase is None:
            n_mute += 1
            continue
        if phase < PHASE_DIAG_FLOOR:
            continue
        n_evidence += 1
        key = (row, col // v1.BLOCK)
        evidence[key] = evidence.get(key, 0.0) + 1.0
        if best.get(key, (None, -1.0))[1] < 1.0:
            best[key] = (col, 1.0)
    masses = []
    for cells, mass, top in v1.differenced_clusters(evidence, best, {}):
        if cells is None:
            continue
        masses.append(mass)
    return {'nodes': n_nodes, 'evidence_nodes': n_evidence,
            'mute_nodes': n_mute, 'clusters': len(masses),
            'cluster_masses': sorted(masses, reverse=True)[:20],
            'evidence_rate': round(n_evidence / n_nodes, 4) if n_nodes else None,
            'mute_rate': round(n_mute / n_nodes, 4) if n_nodes else None}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--ckpt', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    ct = CTVolume(args.cache, base=scroll.ct, level=scroll.level)
    z_quantile = manifest['z_quantiles'][QUANTILE_INDEX]

    names = sorted({r['segment'] for r in manifest['injections']
                    if r['winding_low'] < 100})
    os.makedirs(args.ckpt, exist_ok=True)

    per_segment = {}
    for name in names:
        print(f'{name}: probing phase floor (mid-quantile '
              f'{z_quantile:.0f})', flush=True)
        grid = scrolls.segment_grid(name, scroll, args.grid_cache)
        nodes = segment_phase_nodes(
            ct, grid, centre, z_quantile,
            os.path.join(args.ckpt, f'{name}.pkl'))
        per_segment[name] = phase_clusters(nodes)
        print(f'{name}: {per_segment[name]["clusters"]} clusters, '
              f'evidence rate {per_segment[name]["evidence_rate"]}', flush=True)

    counts = [e['clusters'] for e in per_segment.values()]
    median_extrapolated = float(np.median(counts)) * EXTRAPOLATION
    decision = median_extrapolated <= GATE_MAX_MEDIAN
    report = {
        'probe': 'phase-channel pristine false-alarm floor (U-024 stage 1): '
                 'probe_phase evidence verbatim, mid-quantile band rows, '
                 'all 16 dev segments, empty atlas',
        'declaration': 'probe_phase_cost.py header, committed before the run '
                       '(TOPO-049, twenty-seventh session)',
        'quantile_index': QUANTILE_INDEX, 'z_quantile': z_quantile,
        'extrapolation': EXTRAPOLATION,
        'gate_rule': f'build the full channel iff median per-segment '
                     f'clusters x{EXTRAPOLATION} <= {GATE_MAX_MEDIAN} '
                     f'(declared in advance; ~2x v1 per-segment pool 304/16)',
        'median_clusters_mid_quantile': float(np.median(counts)),
        'median_extrapolated': median_extrapolated,
        'decision_build_full_channel': bool(decision),
        'gate_sensitivity': {
            'x1.5': bool(median_extrapolated <= GATE_MAX_MEDIAN * 1.5),
            'x3': bool(median_extrapolated <= GATE_MAX_MEDIAN * 3)},
        'per_segment': per_segment,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f'median clusters (mid-quantile): {np.median(counts):.0f}; '
          f'extrapolated x{EXTRAPOLATION}: {median_extrapolated:.0f} '
          f'(gate <= {GATE_MAX_MEDIAN})')
    print(f'decision (pre-declared): '
          f'{"BUILD the full channel" if decision else "NEGATIVE - floor too high"}')
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
