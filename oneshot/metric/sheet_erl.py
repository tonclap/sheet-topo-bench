"""Sheet-ERL: expected run length along the winding axis of tifxyz meshes (TOPO-008).

The metric of `oneshot/PROTOCOL.md` §2, with the funkelab reference implementation
(`funlib.evaluate/run_length.py`) declared the arbiter of interpretation:

    ERL = sum over runs of len(run)^2 / L_total

A *run* is a maximal stretch of a grid row's polyline containing no error. Errors
come from the corpus manifest (injected S/M/H windows); runs also break at invalid
nodes and mesh ends. Lengths are euclidean voxel distances scaled to mm.

One decision the protocol's text leaves to the implementation, fixed here and
everywhere: **the skeleton and all lengths come from the pristine grid**, the corpus
manifest only says where errors are. Measuring lengths on the corrupted grid would
let a hole *raise* ERL by deleting skeleton it should be penalised for losing —
the metric would reward vandalism. L_total therefore includes error-zone length.

The same module ships the detector-evaluation harness of PROTOCOL §3 (AP, Recall@N,
delta-ERL@k) and the two baselines that need no external features: random ranking
(the floor) and the local normal-kink heuristic (the naive local detector whose
failure on locally-plausible errors is the target's central hypothesis). The two
remaining protocol baselines (self-intersection, ray winding-count mismatch) ride
with the TOPO-009 feature stack, where the machinery they need is built once.

Usage:

    python sheet_erl.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --report ../../output/topo/erl_paris4.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'injector'))
import scrolls                                                        # noqa: E402
import inject_errors                                                  # noqa: E402


def segment_zones(manifest, name, subset=None):
    """Error windows of one segment as (row_lo, row_hi, col_lo, col_hi)."""
    return [(r['row_lo'], r['row_hi'], r['col_lo'], r['col_hi'])
            for r in manifest['injections']
            if r['segment'] == name and (subset is None or r['id'] in subset)]


def row_runs(grid, row, zones, voxel_mm):
    """(run lengths, counted length) of one grid row, in mm.

    Length is accrued between consecutive valid columns; a pair is *wrong* when
    either endpoint sits in an error zone of this row. Wrong pairs count toward
    L_total but never toward a run — that is the protocol's "merger zones are wrong
    in their entirety" rule applied uniformly to all zone types.
    """
    points = grid[row]
    valid = (points[:, 0] != -1) & (points[:, 1] != -1)
    cols = np.where(valid)[0]
    if len(cols) < 2:
        return np.empty(0), 0.0
    in_zone = np.zeros(len(points), bool)
    for row_lo, row_hi, col_lo, col_hi in zones:
        if row_lo <= row < row_hi:
            in_zone[col_lo:col_hi] = True
    steps = np.linalg.norm(np.diff(grid[row][cols].astype(np.float64), axis=0),
                           axis=1) * voxel_mm
    adjacent = np.diff(cols) == 1
    wrong = in_zone[cols[:-1]] | in_zone[cols[1:]]
    # A step counts toward L_total when the grid is contiguous there; it belongs to
    # a run when it is additionally outside every error zone. Runs are the maximal
    # stretches of member steps; a gap or a zone ends the run either way.
    member = adjacent & ~wrong
    padded = np.concatenate(([False], member, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    cum = np.concatenate(([0.0], np.cumsum(steps)))
    runs = cum[edges[1::2]] - cum[edges[::2]]
    return runs, float(steps[adjacent].sum())


def sheet_erl(grids, manifest_zones, voxel_mm, row_step=1):
    """(ERL mm, L_total mm, run count) over a set of segments."""
    sq_sum, total, count = 0.0, 0.0, 0
    for name, grid in grids.items():
        zones = manifest_zones.get(name, [])
        for row in range(0, grid.shape[0], row_step):
            runs, length = row_runs(grid, row, zones, voxel_mm)
            sq_sum += float(np.sum(runs * runs))
            total += length
            count += len(runs)
    return (sq_sum / total if total > 0 else 0.0), total, count


# ---------------------------------------------------------------- detector harness

def hits(ranking, injections, grids):
    """Match ranked windows to injections: PROTOCOL §3 rules.

    A hit is a window centre inside the error rectangle grown by 50 % per axis.
    Each injection is credited once, to its best-ranked window; later windows on the
    same injection are false positives.

    One degree of freedom the rule leaves open, fixed here and recorded in
    PROTOCOL §3 (insert of 2026-08-20): a window centre inside two grown
    rectangles is credited to the first still-uncredited injection in *manifest
    order*, not to the nearest one. Deterministic, and it is what produced every
    published number — stated so a third party scoring against this harness lands
    on the same figures.
    """
    outcomes, taken = [], set()
    for segment, row, col, _score in ranking:
        hit = None
        for k, r in enumerate(injections):
            if k in taken or r['segment'] != segment:
                continue
            growth_r = (r['row_hi'] - r['row_lo']) * 0.25
            growth_c = (r['col_hi'] - r['col_lo']) * 0.25
            if (r['row_lo'] - growth_r <= row < r['row_hi'] + growth_r
                    and r['col_lo'] - growth_c <= col < r['col_hi'] + growth_c):
                hit = k
                break
        if hit is None:
            outcomes.append((None, segment, row, col))
        else:
            taken.add(hit)
            outcomes.append((hit, segment, row, col))
    return outcomes


def average_precision(outcomes, n_truth):
    tp, ap = 0, 0.0
    for i, (hit, *_rest) in enumerate(outcomes, 1):
        if hit is not None:
            tp += 1
            ap += tp / i
    return ap / n_truth if n_truth else 0.0


def evaluate_ranking(ranking, manifest, grids, voxel_mm, row_step=1,
                     erl_broken=None):
    """AP, Recall@N and delta-ERL@N for one ranking (PROTOCOL §3).

    `erl_broken` may be passed in when the caller already computed it — the zones
    are a function of the manifest alone, so the value is identical across rankings.
    """
    injections = manifest['injections']
    n = len(injections)
    outcomes = hits(ranking, injections, grids)
    recall_at_n = sum(1 for h, *_ in outcomes[:n] if h is not None) / n if n else 0.0

    zones = {r['segment']: segment_zones(manifest, r['segment'])
             for r in injections}
    if erl_broken is None:
        erl_broken, *_ = sheet_erl(grids, zones, voxel_mm, row_step)
    # Cutting at a detected window turns the error into a split: the window becomes
    # a declared run boundary, so its zone stops being *wrong* and starts being a
    # mere gap. Model: remove the zones of hit injections, add the detector's own
    # windows as break-only zones of zero tolerance around the cut column.
    cut_zones = {name: list(z) for name, z in zones.items()}
    changed = False
    for hit, segment, row, col in outcomes[:n]:
        if hit is None:
            continue
        r = injections[hit]
        # Identity filter: two injections of one segment sharing a rectangle
        # byte for byte would both drop here. The injector draws positions
        # continuously, so no shipped corpus has such a pair — recorded because
        # the filter, not the model, is what would be wrong if one ever did.
        cut_zones[segment] = [z for z in cut_zones[segment]
                              if z != (r['row_lo'], r['row_hi'],
                                       r['col_lo'], r['col_hi'])]
        cut_zones[segment].append((r['row_lo'], r['row_hi'], col, col + 1))
        changed = True
    erl_cut = (sheet_erl(grids, cut_zones, voxel_mm, row_step)[0] if changed
               else erl_broken)
    return {'ap': average_precision(outcomes, n),
            'recall_at_n': recall_at_n,
            'erl_broken_mm': erl_broken,
            'erl_after_cuts_mm': erl_cut,
            'delta_erl_mm': erl_cut - erl_broken}


# ---------------------------------------------------------------------- baselines

def windows_of(grid, size=12, stride=12):
    """Window centres of the baselines' scan.

    Validity here is the x coordinate alone, while `row_runs` requires x and y.
    The two tests are equivalent on this substrate: across all 65 cached grids
    of both scrolls, x-validity and y-validity never disagree on a single node
    (checked 2026-08-20). The narrower test is kept because it is what ran.
    """
    for row in range(size, grid.shape[0] - size, stride):
        for col in range(size, grid.shape[1] - size, stride):
            if grid[row, col, 0] != -1:
                yield row, col


def random_ranking(rng, grids, count):
    pool = [(name, row, col)
            for name, grid in sorted(grids.items())
            for row, col in windows_of(grid)]
    picks = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
    return [(*pool[i], 0.0) for i in picks]


def kink_ranking(grids, count, size=8):
    """The naive local detector: rank windows by normal kink."""
    scored = []
    for name, grid in sorted(grids.items()):
        for row, col in windows_of(grid, size=size, stride=size * 2):
            kink = inject_errors.max_kink_deg(
                grid, slice(row - size, row + size), slice(col - size, col + size))
            scored.append((name, row, col, kink))
    scored.sort(key=lambda t: -t[3])
    return scored[:count]


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--random-seeds', type=int, default=100)
    parser.add_argument('--skip-kink', action='store_true',
                        help='omit the kink baseline (it is the slow one)')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']
    names = sorted({r['segment'] for r in manifest['injections']})
    # Pristine geometry for lengths (see module docstring); corrupted grids exist on
    # disk for the detectors of TOPO-009, not for the metric.
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}
    zones = {name: segment_zones(manifest, name) for name in names}

    report = {'corpus': os.path.abspath(args.corpus),
              'protocol': manifest['protocol'],
              'scroll': manifest['scroll'], 'row_step': args.row_step}

    erl_ceiling, total_mm, _ = sheet_erl(grids, {}, voxel_mm, args.row_step)
    erl_broken, _, n_runs = sheet_erl(grids, zones, voxel_mm, args.row_step)
    report['erl_pristine_mm'] = erl_ceiling
    report['erl_broken_mm'] = erl_broken
    report['skeleton_total_mm'] = total_mm
    report['runs'] = n_runs

    # Sanity of PROTOCOL §2 / TOPO-008: ERL must fall monotonically as injections
    # accumulate, in manifest order interleaved across types.
    order = sorted(manifest['injections'], key=lambda r: (r['id'][1:], r['id'][0]))
    curve = []
    for share in (0.25, 0.5, 0.75, 1.0):
        subset = {r['id'] for r in order[:int(len(order) * share)]}
        part = {name: segment_zones(manifest, name, subset) for name in names}
        erl, *_ = sheet_erl(grids, part, voxel_mm, args.row_step)
        curve.append({'share': share, 'erl_mm': erl})
    report['density_curve'] = curve
    report['density_monotone'] = all(
        curve[i]['erl_mm'] >= curve[i + 1]['erl_mm'] - 1e-9
        for i in range(len(curve) - 1))

    n = len(manifest['injections'])
    randoms = []
    for seed in range(args.random_seeds):
        rng = np.random.default_rng(seed)
        randoms.append(evaluate_ranking(random_ranking(rng, grids, n),
                                        manifest, grids, voxel_mm, args.row_step,
                                        erl_broken=erl_broken))
    report['baseline_random'] = {
        key: {'median': float(np.median([r[key] for r in randoms])),
              'iqr': [float(np.percentile([r[key] for r in randoms], 25)),
                      float(np.percentile([r[key] for r in randoms], 75))]}
        for key in ('ap', 'recall_at_n', 'delta_erl_mm')}

    if not args.skip_kink:
        report['baseline_kink'] = evaluate_ranking(
            kink_ranking(grids, n), manifest, grids, voxel_mm, args.row_step,
            erl_broken=erl_broken)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f"pristine ERL {erl_ceiling:.2f} mm over {total_mm:.0f} mm of skeleton; "
          f"broken {erl_broken:.2f} mm; monotone={report['density_monotone']}")
    print(f"random floor: AP {report['baseline_random']['ap']['median']:.4f}, "
          f"recall@N {report['baseline_random']['recall_at_n']['median']:.3f}")
    if 'baseline_kink' in report:
        b = report['baseline_kink']
        print(f"kink baseline: AP {b['ap']:.4f}, recall@N {b['recall_at_n']:.3f}, "
              f"dERL {b['delta_erl_mm']:+.2f} mm")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
