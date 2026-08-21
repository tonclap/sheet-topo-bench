"""TOPO-062: numbers behind the band reading of Z0136 and its neighbours.

Descriptive only — declared as such in the TOPO-062 insert of
`ZONE_CRITERIA_B.md`: the verdicts come from the eye on the band
(TOPO-033 classes), and TOPO-034 already showed that a pointwise
signature does not separate zones. This prints what a pointwise reader
*would* measure, so that the published prose can be checked against it:

  - the fraction of the zone interval with prediction under the chain
    (|n| <= 2) and the longest run without it;
  - the longest run with no prediction exactly on n = 0, and the longest
    run of dark CT (< 60) there;
  - an ASCII map of the prediction band around the chain line, which is
    what the eye actually reads (does the band that leaves come back?).

Geometry, thresholds and readers are `band_zones_b`'s, imported unchanged.

Usage (from pipeline/real_errors/):
    python band_row_readout.py \
        --zones ../../output/topo/real_paris4/supportB_zones.json \
        --id Z0136 --rows 678,680,682 \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --out ../../output/topo/real_paris4/zones_band_b/neighbour_readout.json
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'standalone'))
import scrolls                                                        # noqa: E402
import absolute_winding_calibration as awc                            # noqa: E402
import band_zones_b as B                                              # noqa: E402

DARK_CT = 60        # "dark" for the CT panel, the eye's rough cut
BIN_VX = 25         # ASCII map bin along the arc


def longest_run(mask):
    best = cur = 0
    start = best_start = -1
    for i, v in enumerate(mask):
        if v:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best, best_start


def chain_band(zone, grid, valid, row, sm, recto, threshold):
    cols, pts = B.chain_of_row(zone, grid, valid, row)
    if cols is None:
        return None
    s, x, y, z, c = B.resample_chain(cols, pts)
    tx = B.smooth(np.gradient(x), B.TANGENT_WIN)
    ty = B.smooth(np.gradient(y), B.TANGENT_WIN)
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    tx, ty = tx / norm, ty / norm
    nx, ny = -ty, tx
    ns = np.arange(-B.N_HALF, B.N_HALF + 1.0)
    px = x[None, :] + ns[:, None] * nx[None, :]
    py = y[None, :] + ns[:, None] * ny[None, :]
    pz = np.broadcast_to(z[None, :], px.shape)
    pred = (recto.sample(np.column_stack([px.ravel(), py.ravel(), pz.ravel()]))
            >= threshold).reshape(px.shape)
    ct = B.band_ct(sm, x, y, z, nx, ny)
    return s, c, ns, pred, ct


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zones', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--rows', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.zones, encoding='utf-8') as f:
        zones = [z for z in json.load(f)['zones'] if z['id'] == args.id]
    if len(zones) != 1:
        raise SystemExit(f'{args.id}: expected exactly one zone')
    zone = zones[0]
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    sm = awc.ScanMask(args.cache, base=scroll.ct, level=scroll.level)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=48)
    grid = scrolls.segment_grid(zone['segment'], scroll, args.grid_cache)
    heights, valid = scrolls.row_heights(grid)

    out = {}
    for row in (int(r) for r in args.rows.split(',')):
        band = chain_band(zone, grid, valid, row, sm, recto, scroll.threshold)
        if band is None:
            print(f'row {row}: no chain — SKIPPED', flush=True)
            continue
        s, c, ns, pred, ct = band
        idx = np.flatnonzero((c >= zone['col_lo']) & (c < zone['col_hi']))
        if not idx.size:
            # The chain covers the zone columns +-24, so a row can have a
            # chain and still no sample inside the zone itself.
            print(f'row {row}: chain has no sample inside cols '
                  f"{zone['col_lo']}-{zone['col_hi'] - 1} — SKIPPED",
                  flush=True)
            continue
        seg = slice(idx[0], idx[-1] + 1)
        core = np.abs(ns) <= 2
        on_line = np.argmin(np.abs(ns))
        sup = pred[core].any(axis=0)[seg]
        no_pred_0 = ~pred[on_line][seg]
        dark_0 = ct[on_line][seg] < DARK_CT
        rec = {
            'row': row,
            'zone_s': [float(s[idx[0]]), float(s[idx[-1]])],
            'len_vx': int(sup.size),
            'supported_frac_core': round(float(sup.mean()), 3),
            'longest_unsupported_core_vx': int(longest_run(~sup)[0]),
            'longest_no_pred_on_line_vx': int(longest_run(no_pred_0)[0]),
            'longest_dark_ct_on_line_vx': int(longest_run(dark_0)[0]),
        }
        out[f"{zone['id']}r{row}"] = rec
        print(json.dumps(rec), flush=True)

        # ASCII map: what the eye reads — where the band is, bin by bin.
        lo, hi = max(0.0, s[idx[0]] - 150), min(s[-1], s[idx[-1]] + 150)
        bins = [(b, b + BIN_VX) for b in np.arange(lo, hi, BIN_VX)]
        print(f'  band map, {BIN_VX}-vx bins, "|" = zone edge:')
        # 9 spaces: the map rows below print "  n=" + 4 digits + " " before
        # the line, and a marker one bin off is worse than no marker.
        print('         ' + ''.join(
            '|' if (b0 <= s[idx[0]] < b1 or b0 <= s[idx[-1]] < b1) else ' '
            for b0, b1 in bins))
        for nv in range(14, -15, -2):
            r = int(np.argmin(np.abs(ns - nv)))
            line = ''
            for b0, b1 in bins:
                sel = (s >= b0) & (s < b1)
                f = pred[r, sel].mean() if sel.any() else 0.0
                line += '#' if f > 0.7 else ('+' if f > 0.3 else
                                             ('.' if f > 0.05 else ' '))
            print(f'  n={nv:+4d} {line}')

    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f'readout at {args.out}', flush=True)


if __name__ == '__main__':
    main()
