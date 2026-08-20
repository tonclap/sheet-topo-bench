"""TOPO-033: chain-normal band projection of the 4 candidate zones.

Protocol — the dated TOPO-033 insert in `ZONE_CRITERIA_B.md`, declared
BEFORE any band was rendered. For each zone the valid nodes of the
TOPO-029 card's central row `rc` on columns [col_lo-24, col_hi+24) form a
chain, resampled to 1-vx arc steps; the band samples CT (and the recto
prediction) at the chain height z(s) along the in-plane normal
n in [-30, +30] vx. In this projection a sheet is a horizontal bright
band; a chain on its own sheet keeps the band on n = 0; a real switch
walks the band off n = 0 inside the zone columns. Banner points within
|dz| <= 3 vx of z(s) are projected to (s, n) as the reference witness.

Usage (from pipeline/real_errors/):
    python band_zones_b.py \
        --zones ../../output/topo/real_paris4/supportB_zones.json \
        --ids Z0058,Z0106,Z0136,Z0163 \
        --corpus ../../output/topo/corpus_paris4 \
        --banner ../../output/corpusB/20231231235900_GP.obj \
        --frame-report ../../output/topo/corpusB_frame.json \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --out ../../output/topo/real_paris4/zones_band_b
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import matplotlib

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                       # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'standalone'))
import scrolls                                                        # noqa: E402
import absolute_winding_calibration as awc                            # noqa: E402
from build_corpusB import banner_l2                                   # noqa: E402
from render_zones import ct_slice                                     # noqa: E402

COL_MARGIN = 24     # columns of context on both sides of the zone
N_HALF = 30         # band half-height along the normal, vx
DZ_BANNER = 3.0     # banner witness slab around the chain height, vx
TANGENT_WIN = 9     # boxcar window for the tangent estimate, samples


def chain_of_row(zone, grid, valid, row):
    cols = np.where(valid[row])[0]
    cols = cols[(cols >= zone['col_lo'] - COL_MARGIN)
                & (cols < zone['col_hi'] + COL_MARGIN)]
    if len(cols) < 4:
        return None, None
    pts = grid[row, cols].astype(np.float64)
    return cols, pts


def resample_chain(cols, pts):
    """1-vx arc resampling of the xy polyline; z and col follow linearly."""
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    arc = np.concatenate([[0.0], np.cumsum(d)])
    s = np.arange(0.0, arc[-1])
    x = np.interp(s, arc, pts[:, 0])
    y = np.interp(s, arc, pts[:, 1])
    z = np.interp(s, arc, pts[:, 2])
    c = np.interp(s, arc, cols.astype(float))
    return s, x, y, z, c


def smooth(a, win):
    k = np.ones(win) / win
    pad = np.pad(a, win // 2, mode='edge')
    return np.convolve(pad, k, mode='valid')[:len(a)]


def band_ct(sm, x, y, z, nx, ny):
    """CT sampled at (x + n*nx, y + n*ny, round(z)) — rows n, cols s."""
    ns = np.arange(-N_HALF, N_HALF + 1.0)
    out = np.zeros((len(ns), len(x)), np.float32)
    zi_all = np.round(z).astype(int)
    uz = np.unique(zi_all)
    print(f'  band_ct: {len(uz)} slices z {uz.min()}..{uz.max()}', flush=True)
    for zi in np.unique(zi_all):
        idx = np.flatnonzero(zi_all == zi)
        px = x[idx][None, :] + ns[:, None] * nx[idx][None, :]
        py = y[idx][None, :] + ns[:, None] * ny[idx][None, :]
        x0, x1 = int(px.min()) - 2, int(px.max()) + 3
        y0, y1 = int(py.min()) - 2, int(py.max()) + 3
        window = np.asarray(ct_slice(sm, int(zi), (x0, x1), (y0, y1)),
                            dtype=np.float32)
        out[:, idx] = map_coordinates(window, [py - y0, px - x0],
                                      order=1, mode='nearest')
    return out


def render_zone(zone, grid, heights, valid, banner, sm, recto, threshold,
                out_dir):
    zrows = [r for r in range(zone['row_lo'], zone['row_hi'])
             if np.isfinite(heights[r]) and valid[r].any()]
    if not zrows:
        return None
    rc = zrows[len(zrows) // 2]
    cols, pts = chain_of_row(zone, grid, valid, rc)
    if cols is None:
        return None
    s, x, y, z, c = resample_chain(cols, pts)

    tx = smooth(np.gradient(x), TANGENT_WIN)
    ty = smooth(np.gradient(y), TANGENT_WIN)
    norm = np.hypot(tx, ty)
    norm[norm == 0] = 1.0
    tx, ty = tx / norm, ty / norm
    nx, ny = -ty, tx

    ct = band_ct(sm, x, y, z, nx, ny)
    ns = np.arange(-N_HALF, N_HALF + 1.0)
    px = x[None, :] + ns[:, None] * nx[None, :]
    py = y[None, :] + ns[:, None] * ny[None, :]
    pz = np.broadcast_to(z[None, :], px.shape)
    flat = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
    pred = (recto.sample(flat) >= threshold).reshape(px.shape)

    # Banner witnesses near the chain: nearest arc sample in xy, then the
    # declared |dz| and |n| gates.
    lo = np.array([x.min() - N_HALF - 2, y.min() - N_HALF - 2,
                   z.min() - DZ_BANNER - 1])
    hi = np.array([x.max() + N_HALF + 2, y.max() + N_HALF + 2,
                   z.max() + DZ_BANNER + 1])
    inside = np.all((banner >= lo) & (banner <= hi), axis=1)
    bs = bn = None
    if inside.any():
        bp = banner[inside]
        tree = cKDTree(np.column_stack([x, y]))
        dist, idx = tree.query(bp[:, :2])
        keep = ((dist <= N_HALF)
                & (np.abs(bp[:, 2] - z[idx]) <= DZ_BANNER))
        if keep.any():
            bp, idx = bp[keep], idx[keep]
            n_val = ((bp[:, 0] - x[idx]) * nx[idx]
                     + (bp[:, 1] - y[idx]) * ny[idx])
            keep2 = np.abs(n_val) <= N_HALF
            bs, bn = s[idx][keep2], n_val[keep2]

    in_zone = (c >= zone['col_lo']) & (c < zone['col_hi'])
    s_lo = float(s[in_zone][0]) if in_zone.any() else None
    s_hi = float(s[in_zone][-1]) if in_zone.any() else None

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.6), sharex=True,
                             constrained_layout=True)
    extent = (0, s[-1], -N_HALF, N_HALF)
    axes[0].imshow(ct, cmap='gray', aspect='auto', origin='lower',
                   extent=extent, vmin=0, vmax=255)
    axes[0].set_ylabel('CT, n (vx)', fontsize=8)
    axes[1].imshow(pred.astype(float), cmap='gray', aspect='auto',
                   origin='lower', extent=extent, vmin=0, vmax=1)
    axes[1].set_ylabel('recto >= thr, n (vx)', fontsize=8)
    for ax in axes:
        ax.axhline(0, color='magenta', lw=1.0, alpha=0.9)
        if bs is not None:
            ax.scatter(bs, bn, s=3.5, c='#00e5ff', linewidths=0, alpha=0.85)
        if s_lo is not None:
            ax.axvline(s_lo, color='yellow', lw=0.9, ls='--', alpha=0.9)
            ax.axvline(s_hi, color='yellow', lw=0.9, ls='--', alpha=0.9)
        ax.tick_params(labelsize=7)
    axes[1].set_xlabel('s along the row chain (vx)', fontsize=8)
    fig.suptitle(
        f"{zone['id']}  {zone['segment']}  row {rc}  cols "
        f"{zone['col_lo']}-{zone['col_hi']} (context ±{COL_MARGIN})  "
        f"band |n| <= {N_HALF} vx at chain height",
        fontsize=10)
    path = os.path.join(out_dir, f"{zone['id']}_band.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {'file': os.path.basename(path), 'rc': int(rc),
            'arc_len': float(s[-1]),
            'zone_s': [s_lo, s_hi],
            'banner_points': 0 if bs is None else int(len(bs)),
            'z_span': [float(z.min()), float(z.max())]}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zones', required=True)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--banner', required=True)
    parser.add_argument('--frame-report', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    want = set(args.ids.split(','))
    with open(args.zones, encoding='utf-8') as f:
        zones = [z for z in json.load(f)['zones'] if z['id'] in want]
    if {z['id'] for z in zones} != want:
        raise SystemExit(f'ids not all found: {want}')
    with open(os.path.join(args.corpus, 'manifest.json'),
              encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]

    banner, info = banner_l2(args.banner, args.frame_report)
    print(f"banner: {info['sample_points']} sample points in L2", flush=True)

    print('opening scan mask...', flush=True)
    sm = awc.ScanMask(args.cache, base=scroll.ct, level=scroll.level)
    print('opening prediction...', flush=True)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=48)
    print('readers ready', flush=True)

    os.makedirs(args.out, exist_ok=True)
    meta = {}
    grids = {}
    for zone in sorted(zones, key=lambda z: z['id']):
        name = zone['segment']
        if name not in grids:
            g = scrolls.segment_grid(name, scroll, args.grid_cache)
            grids[name] = (g, *scrolls.row_heights(g))
        grid, heights, valid = grids[name]
        res = render_zone(zone, grid, heights, valid, banner, sm, recto,
                          scroll.threshold, args.out)
        meta[zone['id']] = res
        print(f"{zone['id']}: "
              + (f"done (arc {res['arc_len']:.0f} vx, banner "
                 f"{res['banner_points']} pts)" if res else 'SKIPPED'),
              flush=True)

    with open(os.path.join(args.out, 'band_meta.json'), 'w',
              encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"meta at {os.path.join(args.out, 'band_meta.json')}", flush=True)


if __name__ == '__main__':
    main()
