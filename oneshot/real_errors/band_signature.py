"""TOPO-034: the band-run signature as a number — does Z0136 separate?

TOPO-033 confirmed Z0136 by eye in the chain-normal band: ~250 vx of arc
where the chain runs through dark CT with no recto support, while the three
false candidates keep their band on n = 0. This file turns that reading into
a number computed from the same band data `band_zones_b.py` renders, blind
to any card or label.

**Protocol, declared 17.08.2026 BEFORE any run of this file (the commit
adding it precedes the first run; validation zones and the separation rule
are fixed here, thresholds are not tuned on output):**

1. **Band construction is TOPO-033's, verbatim:** the central valid row of
   the zone, columns [col_lo-24, col_hi+24), 1-vx arc resampling, boxcar-9
   tangents, CT and the recto prediction sampled along the in-plane normal
   (band_zones_b's own functions are imported, not copied).
2. **Support at s:** the recto prediction (>= the scroll's frozen threshold,
   the same one every detector uses) anywhere in the |n| <= 2 vx slab of
   that arc sample. 2 vx absorbs chain wobble; the sheet itself is ~10 vx.
3. **Darkness at s:** mean CT over the same |n| <= 2 vx slab below T_ct,
   where T_ct = (P20 + P80) / 2 of the zone's full band CT — a per-zone
   relative threshold: the band always contains both sheet (bright) and
   inter-sheet gap (dark), and the midpoint splits them without an absolute
   constant that would drift with scan brightness.
4. **The signature** is the length (vx) of the longest run of consecutive
   arc samples that are simultaneously unsupported (§2) and dark (§3),
   among runs that intersect the zone's own column interval [s_lo, s_hi];
   the longest run anywhere on the chain and the qualifying fraction of the
   zone interval are reported alongside as context, not as the signature.
5. **Validation rule on the 4 TOPO-033 zones (Z0058, Z0106, Z0136, Z0163):**
   the signature separates iff Z0136's signature is at least 2x each of the
   other three ('кратно' from the task, made a number). Only if it
   separates does the credited-57 run start; its reading (enrichment of
   real_error among high signatures) is TOPO-034's outcome (a).
6. **Checkpointing:** one JSONL line per zone, appended as computed; a
   rerun skips zones already in the output (the third bite of the
   'long run without a checkpoint' lesson stays third).

Usage (from oneshot/real_errors/):
    python band_signature.py \
        --zones ../../output/topo/real_paris4/supportB_zones.json \
        --ids Z0058,Z0106,Z0136,Z0163 \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --out ../../output/topo/real_paris4/band_signature.jsonl
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
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'standalone'))
import scrolls                                                        # noqa: E402
import absolute_winding_calibration as awc                            # noqa: E402
import band_zones_b as bz                                             # noqa: E402

N_SLAB = 2          # §2/§3: |n| <= 2 vx slab around the chain


def longest_run(mask, s, s_lo, s_hi):
    """(longest run intersecting [s_lo, s_hi], longest run anywhere), vx."""
    best_zone = best_any = 0
    start = None
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8),
                                                   [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        length = float(s[b - 1] - s[a]) + 1.0
        best_any = max(best_any, length)
        if s_lo is not None and s[b - 1] >= s_lo and s[a] <= s_hi:
            best_zone = max(best_zone, length)
    return best_zone, best_any


def zone_signature(zone, grid, heights, valid, sm, recto, threshold):
    zrows = [r for r in range(zone['row_lo'], zone['row_hi'])
             if np.isfinite(heights[r]) and valid[r].any()]
    if not zrows:
        return None
    rc = zrows[len(zrows) // 2]
    cols, pts = bz.chain_of_row(zone, grid, valid, rc)
    if cols is None:
        return None
    s, x, y, z, c = bz.resample_chain(cols, pts)

    tx = bz.smooth(np.gradient(x), bz.TANGENT_WIN)
    ty = bz.smooth(np.gradient(y), bz.TANGENT_WIN)
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    tx, ty = tx / norm, ty / norm
    nx, ny = -ty, tx

    ct = bz.band_ct(sm, x, y, z, nx, ny)
    ns = np.arange(-bz.N_HALF, bz.N_HALF + 1.0)
    px = x[None, :] + ns[:, None] * nx[None, :]
    py = y[None, :] + ns[:, None] * ny[None, :]
    pz = np.broadcast_to(z[None, :], px.shape)
    flat = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
    pred = (recto.sample(flat) >= threshold).reshape(px.shape)

    slab = np.abs(ns) <= N_SLAB
    unsupported = ~pred[slab].any(axis=0)
    t_ct = float((np.percentile(ct, 20) + np.percentile(ct, 80)) / 2.0)
    dark = ct[slab].mean(axis=0) < t_ct
    qualifies = unsupported & dark

    in_zone = (c >= zone['col_lo']) & (c < zone['col_hi'])
    s_lo = float(s[in_zone][0]) if in_zone.any() else None
    s_hi = float(s[in_zone][-1]) if in_zone.any() else None
    run_zone, run_any = longest_run(qualifies, s, s_lo, s_hi)
    zone_frac = (float(qualifies[in_zone].mean()) if in_zone.any() else None)

    return {'id': zone['id'], 'segment': zone['segment'], 'rc': int(rc),
            'arc_len': float(s[-1]), 'zone_s': [s_lo, s_hi],
            't_ct': round(t_ct, 2),
            'signature_run_vx': round(run_zone, 1),
            'run_anywhere_vx': round(run_any, 1),
            'zone_qualifying_frac': (None if zone_frac is None
                                     else round(zone_frac, 4))}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zones', required=True)
    parser.add_argument('--ids', default=None,
                        help='comma-separated subset; default = all zones')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.zones, encoding='utf-8') as f:
        zones = json.load(f)['zones']
    if args.ids:
        want = set(args.ids.split(','))
        zones = [z for z in zones if z['id'] in want]
        if {z['id'] for z in zones} != want:
            raise SystemExit(f'ids not all found: {want}')

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            for line in f:
                done.add(json.loads(line)['id'])
        print(f'checkpoint: {len(done)} zones already computed', flush=True)
    todo = [z for z in sorted(zones, key=lambda z: z['id'])
            if z['id'] not in done]
    if not todo:
        print('nothing to do', flush=True)
        return

    with open(os.path.join(args.corpus, 'manifest.json'),
              encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    print('opening scan mask...', flush=True)
    sm = awc.ScanMask(args.cache, base=scroll.ct, level=scroll.level)
    print('opening prediction...', flush=True)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=48)
    print('readers ready', flush=True)

    grids = {}
    for zone in todo:
        name = zone['segment']
        if name not in grids:
            g = scrolls.segment_grid(name, scroll, args.grid_cache)
            grids[name] = (g, *scrolls.row_heights(g))
        grid, heights, valid = grids[name]
        res = zone_signature(zone, grid, heights, valid, sm, recto,
                             scroll.threshold)
        if res is None:
            res = {'id': zone['id'], 'segment': zone['segment'],
                   'skipped': True}
        with open(args.out, 'a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(res, ensure_ascii=False, sort_keys=True)
                    + '\n')
        print(f"{zone['id']}: "
              + (f"run {res['signature_run_vx']:.0f} vx (zone), "
                 f"{res['run_anywhere_vx']:.0f} anywhere, "
                 f"frac {res['zone_qualifying_frac']}"
                 if not res.get('skipped') else 'SKIPPED'), flush=True)


if __name__ == '__main__':
    main()
