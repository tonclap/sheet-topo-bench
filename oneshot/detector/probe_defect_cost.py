"""TOPO-052 cost probe: pristine floor of the volume disclination feature (U-022).

The W1 remainder (22 uncovered windows: 17 M, 5 S) fell to U-022 after
TOPO-045 (mesh smoothness ~ the signature itself), TOPO-048 (phase benefit
5/22) and TOPO-049 (phase floor catches the median honest node). U-022's
taking condition carries the TOPO-049 lesson verbatim: **price before
benefit — the background sets the floor, benefit is measured at that floor.**

Feature (ANGLES_2026-08-18 §3, move 4 — the plaquette RP^2 defect detector
of the CT normal field): a Y-junction of sheets is a disclination line of
the layering's normal field. Per node, CT is sampled on a step-1 cube; the
structure tensor's principal eigenvector gives the (sign-ambiguous, hence
RP^2) normal; an elementary plaquette pierced by a disclination transports
the director to its negation (negative sign product around the loop). The
node's feature D = defective / coherent plaquettes of its inner cube.

Honest transfer, recorded before the run: injections never touch the CT
volume, so disclinations are substrate property, not injection property.
What transfers is the trace's *position* relative to the volume topology —
the injector parks merged traces mid-gap / in natural-contact zones, honest
traces ride their own sheet; if disclination lines concentrate where sheets
meet, displaced (corrupted) node positions carry more defect density than
honest (pristine) positions at the same cells. Class risk named by the
literature itself: tangential merging of near-parallel sheets yields weak
signal — exactly our blind class.

**Declared before the run, not tuned on the result:**

- Node lattice: cube half-size 11 vx, step 1 (23^3 CT samples via
  ``CTVolume``; off-mask zeros read as air, as in probe_ct). Intensity
  smoothing ``SIGMA1 = 1.0``; structure tensor J = G_{SIGMA2} * (grad I
  grad I^T), ``SIGMA2 = 2.0`` (half sheet thickness: TOPO-046 medians
  8-11 vx); 2-vx smoothing halo cut -> inner cube 19^3.
- Normal n = eigenvector of the largest eigenvalue; coherence = Westin
  linearity (l3 - l2) / (l1 + l2 + l3); coherent voxel iff >= ``CL_MIN =
  0.5`` (sensitivity 0.25 / 0.75 published alongside, no part of any rule).
- Defect test: **ring transport at radius ``RING_R = 3``** (the lattice
  standard adapted to a field with a coherence mask: the elementary-
  plaquette variant is structurally blind — synthetic positive controls
  showed the sign flip's own corners carry coherence 0.03-0.05, i.e. the
  mask kills the signal exactly at the core; fixed on synthetic controls
  BEFORE any corpus read). A centre voxel is *eligible* in a plane
  orientation iff all 8 ring waypoints (Chebyshev ring, radius 3) are
  coherent; it is *defective* iff additionally the product of
  sign(n_i . n_{i+1}) around the 8-point loop < 0 (half-integer
  disclination piercing the ring; the centre's own coherence is NOT
  required — the core is incoherent by physics). Ring-radius sensitivity
  2 / 4 published alongside, no part of any rule.
- ``D`` = defective / eligible centres summed over the three plane
  orientations of the inner cube; the node is mute if eligible centres
  < ``MIN_ELIGIBLE = 100``.
- Scope: the **pristine** grids of all 16 dev segments; rows of the MIDDLE
  z-quantile (index 2 of the manifest's 5) within SUPPORT_BAND_VX; every
  ``COL_STRIDE = 8``-th valid column (declared up front: the volume lattice
  costs 12 167 CT samples per node against the radial probe's 73).
- **The background sets the floor: FLOOR := pooled pristine q99 of D**
  (stage-2 node evidence will be strictly D > FLOOR, so the honest-node
  background is <= 1 % by construction). q95 / q99.5 published alongside.
- **Degeneracy gates (the decision of this stage): (v) if the median
  per-segment mute share > 0.5, the feature does not exist on this
  substrate — technical boundary; (b) if the pooled median D > 0, defect
  lines are the normal state of the honest node (the TOPO-049 degenerate
  mode) — U-022's construction closes without the benefit probe.** Passing
  both gates only earns stage 2 (the 22-window benefit probe at FLOOR,
  bar >= 8/22 — TOPO-045's bar verbatim — under its own declaration).

Synthetic controls (recorded before any corpus read; scratchpad, seed 1,
pitch 12): laminar stack D = 0 at every setting; pure noise fully mute;
**Y-junction of three walls (the papyrus merger geometry proper) D = 0.031
at the main setting (CL 0.5, ring 3), 186 defective centres**; a smooth
smectic half core (b = p/2) is caught only at CL 0.25 (19/58 defects at
ring 3/4) — the core garbles coherence out to ~p/2; an integer core (b = p)
is blind at every setting, as sign transport must be (pi-winding only).
The declared main setting stands on the Y-junction control; the published
sensitivity grid covers the smooth-core regime.

Checkpoints: per-segment pickle of raw per-node stats (eligible/defective
at every (CL, ring) setting) — the run resumes, the floor replays offline.

Usage (from oneshot/detector/):

    python probe_defect_cost.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --report ../../output/topo/probe_defect_cost_paris4.json \
        --ckpt ../../output/topo/ckpt_defect_cost
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy import ndimage

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

HALF = 11                 # cube half-size, vx (lattice 23^3)
EDGE = 2                  # smoothing halo cut -> inner cube 19^3
SIGMA1 = 1.0              # intensity smoothing
SIGMA2 = 2.0              # tensor integration scale ~ half sheet thickness
CL_MIN = 0.5              # Westin-linearity coherence bar (sens. 0.25/0.75)
CL_BARS = (0.25, 0.5, 0.75)
RING_R = 3                # ring-transport radius, vx (sens. 2/4 alongside)
RING_RS = (2, 3, 4)
RING_OFFS = ((1, 0), (1, 1), (0, 1), (-1, 1),
             (-1, 0), (-1, -1), (0, -1), (1, -1))   # Chebyshev ring x R
MIN_ELIGIBLE = 100        # below this the node is mute
COL_STRIDE = 8            # declared subsample: 12167 samples/node vs 73 radial
QUANTILE_INDEX = 2        # the middle of the manifest's five z-quantiles
FLOOR_QUANTILE = 99.0     # the background sets the floor (q95/q99.5 alongside)
GATE_MUTE_MAX = 0.5       # (v) median per-segment mute share above this
BATCH_NODES = 48          # CT read batching only, no effect on numbers

_rng = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1, -HALF:HALF + 1]
CUBE_OFFSETS = np.stack([a.ravel() for a in _rng], 1).astype(np.float64)
SIDE = 2 * HALF + 1


def _ring_way(arr, a, b, r, da, db):
    """Waypoint slab at centre + (da, db) * r in the (a, b) plane."""
    sl = [slice(None)] * 3
    sl[a] = slice(r + da * r, arr.shape[a] - r + da * r)
    sl[b] = slice(r + db * r, arr.shape[b] - r + db * r)
    return arr[tuple(sl)]


def ring_transport(n, cl, a, b, r, cl_min):
    """(eligible, defective) centre counts for one plane orientation.

    A centre is eligible iff all 8 Chebyshev-ring waypoints at radius
    ``r`` are coherent; defective iff the RP^2 sign transport around the
    ring is negative. The centre's own coherence is not consulted.
    """
    ways_n = [_ring_way(n, a, b, r, da, db) for da, db in RING_OFFS]
    ways_ok = [_ring_way(cl, a, b, r, da, db) >= cl_min
               for da, db in RING_OFFS]
    ok = np.logical_and.reduce(ways_ok)
    if not ok.any():
        return 0, 0
    s = np.ones(ok.shape)
    for i in range(len(RING_OFFS)):
        d = (ways_n[i] * ways_n[(i + 1) % len(RING_OFFS)]).sum(-1)
        si = np.sign(d)
        si[si == 0] = 1.0
        s = s * si
    return int(ok.sum()), int((ok & (s < 0)).sum())


def node_defects(cube):
    """{(cl_bar, ring_r): (eligible, defective)} for one node.

    ``cube`` is the (23, 23, 23) float CT read; counts are summed over the
    three plane orientations of the inner cube.
    """
    I = ndimage.gaussian_filter(cube, SIGMA1)
    g = np.gradient(I)
    J = np.empty(I.shape + (3, 3))
    for a in range(3):
        for b in range(a, 3):
            Jab = ndimage.gaussian_filter(g[a] * g[b], SIGMA2)
            J[..., a, b] = Jab
            J[..., b, a] = Jab
    inner = (slice(EDGE, -EDGE),) * 3
    w, vec = np.linalg.eigh(J[inner])            # ascending eigenvalues
    n = vec[..., :, 2]                           # principal direction
    tr = np.maximum(w.sum(-1), 1e-12)
    cl = (w[..., 2] - w[..., 1]) / tr
    out = {}
    for bar in CL_BARS:
        for r in RING_RS:
            n_elig = 0
            n_def = 0
            for a, b in ((0, 1), (0, 2), (1, 2)):
                e, d = ring_transport(n, cl, a, b, r, bar)
                n_elig += e
                n_def += d
            out[(bar, r)] = (n_elig, n_def)
    return out


def segment_defect_nodes(ct, grid, z_quantile, ckpt):
    """Raw per-node defect stats for the mid-quantile band rows, checkpointed."""
    key = {'half': HALF, 'edge': EDGE, 'sigma1': SIGMA1, 'sigma2': SIGMA2,
           'cl_bars': list(CL_BARS), 'ring_rs': list(RING_RS),
           'stride': COL_STRIDE, 'quantile': float(z_quantile)}
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
    out = {'row': [], 'col': [], 'stats': []}
    for k, row in enumerate(band_rows):
        mask = valid[row]
        if not mask.any():
            continue
        cols = np.where(mask)[0][::COL_STRIDE]
        points = grid[row][cols].astype(np.float64)
        for start in range(0, len(cols), BATCH_NODES):
            batch_pts = points[start:start + BATCH_NODES]
            batch_cols = cols[start:start + BATCH_NODES]
            stacked = (batch_pts[:, None, :] + CUBE_OFFSETS[None, :, :]
                       ).reshape(-1, 3)
            sampled = ct.values(stacked).astype(np.float64).reshape(
                len(batch_pts), SIDE, SIDE, SIDE)
            for j, col in enumerate(batch_cols):
                out['row'].append(int(row))
                out['col'].append(int(col))
                out['stats'].append(node_defects(sampled[j]))
        if (k + 1) % 5 == 0 or k + 1 == len(band_rows):
            print(f'  row {k + 1}/{len(band_rows)} '
                  f'({len(out["col"])} nodes probed)', flush=True)
    payload = {'key': key, 'nodes': out}
    tmp = f'{ckpt}.{os.getpid()}.part'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp, ckpt)
    return out


def density_summary(nodes, bar, ring_r):
    """Distribution of D at one (CL, ring) setting; mute nodes excluded."""
    dens, mute = [], 0
    for st in nodes['stats']:
        n_elig, n_def = st[(bar, ring_r)]
        if n_elig < MIN_ELIGIBLE:
            mute += 1
        else:
            dens.append(n_def / n_elig)
    n_total = len(nodes['stats'])
    if not dens:
        return {'nodes': n_total, 'mute_rate': 1.0 if n_total else None,
                'stats': None}
    d = np.array(dens)
    return {'nodes': n_total,
            'mute_rate': round(mute / n_total, 4),
            'stats': {
                'median': float(np.median(d)),
                'mean': float(d.mean()),
                'zero_share': round(float((d == 0).mean()), 4),
                'q90': float(np.percentile(d, 90)),
                'q95': float(np.percentile(d, 95)),
                'q99': float(np.percentile(d, 99)),
                'q99_5': float(np.percentile(d, 99.5)),
                'max': float(d.max())}}


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
    ct = CTVolume(args.cache, base=scroll.ct, level=scroll.level)
    z_quantile = manifest['z_quantiles'][QUANTILE_INDEX]

    names = sorted({r['segment'] for r in manifest['injections']
                    if r['winding_low'] < 100})
    os.makedirs(args.ckpt, exist_ok=True)

    combos = [(bar, r) for bar in CL_BARS for r in RING_RS]
    per_segment = {}
    pooled = {combo: [] for combo in combos}
    for name in names:
        print(f'{name}: probing disclination floor (mid-quantile '
              f'{z_quantile:.0f})', flush=True)
        grid = scrolls.segment_grid(name, scroll, args.grid_cache)
        nodes = segment_defect_nodes(
            ct, grid, z_quantile, os.path.join(args.ckpt, f'{name}.pkl'))
        per_segment[name] = {f'{bar}/{r}': density_summary(nodes, bar, r)
                             for bar, r in combos}
        for combo in combos:
            for st in nodes['stats']:
                n_elig, n_def = st[combo]
                if n_elig >= MIN_ELIGIBLE:
                    pooled[combo].append(n_def / n_elig)
        main_bar = per_segment[name][f'{CL_MIN}/{RING_R}']
        print(f'{name}: {main_bar["nodes"]} nodes, mute '
              f'{main_bar["mute_rate"]}, '
              f'median D {main_bar["stats"]["median"] if main_bar["stats"] else None}',
              flush=True)

    d_main = np.array(pooled[(CL_MIN, RING_R)])
    mute_rates = [per_segment[n][f'{CL_MIN}/{RING_R}']['mute_rate']
                  for n in per_segment]
    median_mute = float(np.median(mute_rates))
    gate_mute_dead = median_mute > GATE_MUTE_MAX
    pooled_median = float(np.median(d_main)) if len(d_main) else None
    gate_median_dead = (pooled_median is not None) and (pooled_median > 0)
    floor = float(np.percentile(d_main, FLOOR_QUANTILE)) if len(d_main) else None

    if gate_mute_dead:
        verdict = ('(v) DEGENERATE: median mute share > 0.5 - the feature '
                   'does not exist on this substrate')
    elif gate_median_dead:
        verdict = ('(b) NEGATIVE: pooled median D > 0 - defect lines are '
                   'the normal state of the honest node (TOPO-049 mode); '
                   'no benefit probe')
    else:
        verdict = ('gates passed - stage 2 (22-window benefit probe at '
                   'FLOOR, bar >= 8/22) earns its own declaration')

    report = {
        'probe': 'volume disclination pristine floor (U-022 stage 1): '
                 'plaquette RP2 defect density of the CT normal field, '
                 'mid-quantile band rows, every 8th column, all 16 dev '
                 'segments, pristine grids',
        'declaration': 'probe_defect_cost.py header, committed before the '
                       'run (TOPO-052, twenty-eighth session)',
        'constants': {'half': HALF, 'edge': EDGE, 'sigma1': SIGMA1,
                      'sigma2': SIGMA2, 'cl_min': CL_MIN, 'ring_r': RING_R,
                      'min_eligible': MIN_ELIGIBLE, 'col_stride': COL_STRIDE,
                      'quantile_index': QUANTILE_INDEX},
        'z_quantile': z_quantile,
        'gate_rules': {
            'v_degenerate': f'median per-segment mute share > {GATE_MUTE_MAX}',
            'b_negative': 'pooled median D > 0 (floor would catch the '
                          'median honest node)',
            'floor': f'FLOOR := pooled pristine q{FLOOR_QUANTILE:.0f} of D '
                     f'(stage-2 evidence strictly D > FLOOR)'},
        'pooled': {f'{bar}/{r}': {
            'n': len(pooled[(bar, r)]),
            'median': float(np.median(pooled[(bar, r)]))
            if pooled[(bar, r)] else None,
            'zero_share':
            round(float((np.array(pooled[(bar, r)]) == 0).mean()), 4)
            if pooled[(bar, r)] else None,
            'q95': float(np.percentile(pooled[(bar, r)], 95))
            if pooled[(bar, r)] else None,
            'q99': float(np.percentile(pooled[(bar, r)], 99))
            if pooled[(bar, r)] else None,
            'q99_5': float(np.percentile(pooled[(bar, r)], 99.5))
            if pooled[(bar, r)] else None} for bar, r in combos},
        'median_mute_share': median_mute,
        'pooled_median_D': pooled_median,
        'FLOOR': floor,
        'verdict': verdict,
        'per_segment': per_segment,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f'pooled: n={len(d_main)}, median D={pooled_median}, '
          f'zero share={(d_main == 0).mean() if len(d_main) else None}, '
          f'FLOOR (q99)={floor}')
    print(f'median mute share: {median_mute}')
    print(f'verdict (pre-declared): {verdict}')
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
