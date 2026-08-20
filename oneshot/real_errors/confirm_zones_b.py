"""TOPO-031: confirmation cards for the 4 real_error zones on independent slices.

Protocol — the dated insert in `ZONE_CRITERIA_B.md` (17.08.2026, session 19),
declared BEFORE any confirmation card was rendered. Windows are rendered on
zone rows other than the central row `rc` of the TOPO-029 card (first and
last valid rows of the zone rectangle); a window is independent when every
anchor satisfies |z_confirm - z_card| >= 12 vx (disjoint +-6 slabs). If no
zone row yields independence, a fallback window at `rc` with the slice
centre shifted by +-14 vx is rendered instead. All dz values go to
`confirm_meta.json` before any labelling happens.

Overlays, anchors, slab and colours are exactly `render_zones_b.py`'s.

Usage (from oneshot/real_errors/):
    python confirm_zones_b.py \
        --zones ../../output/topo/real_paris4/supportB_zones.json \
        --ids Z0058,Z0106,Z0136,Z0163 \
        --corpus ../../output/topo/corpus_paris4 \
        --banner ../../output/corpusB/20231231235900_GP.obj \
        --frame-report ../../output/topo/corpusB_frame.json \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --out ../../output/topo/real_paris4/zones_png_b_confirm
"""

import argparse
import json
import os
import sys

import numpy as np
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
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'standalone'))
import scrolls                                                        # noqa: E402
import absolute_winding_calibration as awc                            # noqa: E402
from build_corpusB import banner_l2                                   # noqa: E402
from render_zones import ct_slice                                     # noqa: E402
from render_zones_b import HALF, SLAB, distance_classes               # noqa: E402

INDEP_DZ = 12.0   # declared: disjoint +-6 slabs
FALLBACK_DZ = 14.0


def zone_anchor_points(zone, grid, valid, row):
    """The card's anchor columns/points for a given row — render_zones_b logic."""
    col_lo, col_hi = zone['col_lo'], zone['col_hi']
    width = col_hi - col_lo
    anchors = []
    for frac in (1 / 6, 1 / 2, 5 / 6):
        want = col_lo + int(frac * width)
        cols = np.where(valid[row])[0]
        cols = cols[(cols >= col_lo - 8) & (cols < col_hi + 8)]
        if not len(cols):
            continue
        col = int(cols[np.argmin(np.abs(cols - want))])
        pt = grid[row, col].astype(float)
        if not anchors or abs(anchors[-1][0] - col) > 12:
            anchors.append((col, pt))
    return anchors


def zone_rows_valid(zone, heights, valid):
    return [r for r in range(zone['row_lo'], zone['row_hi'])
            if np.isfinite(heights[r]) and valid[r].any()]


def render_window(zone, grid, row, z_shift, node_d, banner, sm, recto,
                  threshold, valid, path, tag):
    """One confirmation card: the render_zones_b card at an explicit row,
    with an optional z shift of every slice centre."""
    anchors = zone_anchor_points(zone, grid, valid, row)
    if not anchors:
        return None

    row_lo, row_hi = zone['row_lo'], zone['row_hi']
    col_lo, col_hi = zone['col_lo'], zone['col_hi']
    all_valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    rows_idx, cols_idx = np.where(all_valid)
    all_pts = grid[rows_idx, cols_idx].astype(np.float64)
    d_all = np.full(len(all_pts), np.nan, np.float32)
    for k, (r, c) in enumerate(zip(rows_idx, cols_idx)):
        d = node_d.get((int(r), int(c)))
        if d is not None:
            d_all[k] = d

    zone_nodes_in_any_slab = 0
    fig, axes = plt.subplots(1, len(anchors),
                             figsize=(5.4 * len(anchors), 5.8))
    if len(anchors) == 1:
        axes = [axes]
    zis = []
    for ax, (col, pt) in zip(axes, anchors):
        zi = int(round(pt[2] + z_shift))
        zis.append(zi)
        xs = (int(pt[0]) - HALF, int(pt[0]) + HALF)
        ys = (int(pt[1]) - HALF, int(pt[1]) + HALF)
        ax.imshow(ct_slice(sm, zi, xs, ys), cmap='gray', origin='lower',
                  extent=(xs[0], xs[1], ys[0], ys[1]), vmin=0, vmax=255)
        bmask = ((np.abs(banner[:, 2] - zi) <= SLAB)
                 & (banner[:, 0] >= xs[0]) & (banner[:, 0] < xs[1])
                 & (banner[:, 1] >= ys[0]) & (banner[:, 1] < ys[1]))
        if bmask.any():
            bp = banner[bmask]
            ax.scatter(bp[:, 0], bp[:, 1], s=2.5, c='#00e5ff',
                       linewidths=0, alpha=0.8)
        gx, gy = np.meshgrid(np.arange(xs[0], xs[1], 2),
                             np.arange(ys[0], ys[1], 2))
        flat = np.column_stack([gx.ravel(), gy.ravel(),
                                np.full(gx.size, float(zi))])
        mask = (recto.sample(flat) >= threshold).reshape(gx.shape)
        if mask.any():
            ax.contour(gx, gy, mask.astype(float), levels=[0.5],
                       colors='#ff9500', linewidths=0.8, alpha=0.9)
        slab = ((np.abs(all_pts[:, 2] - zi) <= SLAB)
                & (all_pts[:, 0] >= xs[0]) & (all_pts[:, 0] < xs[1])
                & (all_pts[:, 1] >= ys[0]) & (all_pts[:, 1] < ys[1]))
        pts_in = all_pts[slab]
        if len(pts_in):
            colours = distance_classes(d_all[slab])
            in_zone = ((rows_idx[slab] >= row_lo) & (rows_idx[slab] < row_hi)
                       & (cols_idx[slab] >= col_lo)
                       & (cols_idx[slab] < col_hi))
            zone_nodes_in_any_slab += int(in_zone.sum())
            ax.scatter(pts_in[~in_zone, 0], pts_in[~in_zone, 1], s=7,
                       c=colours[~in_zone], linewidths=0)
            ax.scatter(pts_in[in_zone, 0], pts_in[in_zone, 1], s=26,
                       c=colours[in_zone], edgecolors='yellow',
                       linewidths=0.9)
        ax.plot(pt[0], pt[1], marker='x', color='magenta', markersize=11,
                markeredgewidth=2.2)
        ax.set_title(f'col {col}  z={zi}', fontsize=9)
        ax.set_xlim(xs)
        ax.set_ylim(ys)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
    fig.suptitle(
        f"{zone['id']} [{tag}]  {zone['segment']}  rows {row_lo}-{row_hi} "
        f"cols {col_lo}-{col_hi}  mass {zone['mass']} "
        f"support rank {zone['support_rank']}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return {'anchors': [int(c) for c, _ in anchors], 'z': zis,
            'zone_nodes_in_slabs': zone_nodes_in_any_slab}


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
        raise SystemExit(f"ids not all found: {want}")
    with open(os.path.join(args.corpus, 'manifest.json'),
              encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]

    banner, info = banner_l2(args.banner, args.frame_report)
    print(f"banner: {info['sample_points']} sample points in L2", flush=True)

    cells_dir = os.path.join(os.path.dirname(os.path.abspath(args.zones)),
                             'cells_corpusB')
    sm = awc.ScanMask(args.cache, base=scroll.ct, level=scroll.level)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=48)

    os.makedirs(args.out, exist_ok=True)
    grids = {}
    dmaps = {}
    meta = {}
    for zone in sorted(zones, key=lambda z: z['id']):
        name = zone['segment']
        if name not in grids:
            g = scrolls.segment_grid(name, scroll, args.grid_cache)
            grids[name] = (g, *scrolls.row_heights(g))
            ck = np.load(os.path.join(cells_dir, f'{name}.npz'))
            dmaps[name] = {(int(r), int(c)): float(d) for r, c, d
                           in zip(ck['rows'], ck['cols'], ck['d'])}
        grid, heights, valid = grids[name]

        zrows = zone_rows_valid(zone, heights, valid)
        rc = zrows[len(zrows) // 2]
        card = zone_anchor_points(zone, grid, valid, rc)
        card_z = {c: float(p[2]) for c, p in card}

        # Candidate confirmation rows: first and last valid rows, minus rc.
        cand = [r for r in (zrows[0], zrows[-1]) if r != rc]
        windows = []
        for row in dict.fromkeys(cand):
            anchors = zone_anchor_points(zone, grid, valid, row)
            if not anchors:
                continue
            dz = [abs(float(p[2]) - card_z.get(c, float(p[2])))
                  if c in card_z else min(abs(float(p[2]) - v)
                                          for v in card_z.values())
                  for c, p in anchors]
            windows.append({'row': int(row), 'z_shift': 0.0, 'dz': dz,
                            'independent': bool(min(dz) >= INDEP_DZ)})
        if not any(w['independent'] for w in windows):
            for shift in (+FALLBACK_DZ, -FALLBACK_DZ):
                windows.append({'row': int(rc), 'z_shift': shift,
                                'dz': [abs(shift)] * len(card),
                                'independent': abs(shift) >= INDEP_DZ})

        meta[zone['id']] = {'rc': int(rc),
                            'card_z': {str(c): z for c, z in card_z.items()},
                            'windows': []}
        for w in windows:
            tag = (f"row {w['row']}" if w['z_shift'] == 0
                   else f"row {w['row']} dz{w['z_shift']:+.0f}")
            fname = (f"{zone['id']}_r{w['row']}.png" if w['z_shift'] == 0
                     else f"{zone['id']}_r{w['row']}_dz"
                          f"{int(w['z_shift']):+d}.png")
            path = os.path.join(args.out, fname)
            res = render_window(zone, grid, w['row'], w['z_shift'],
                                dmaps[name], banner, sm, recto,
                                scroll.threshold, valid, path, tag)
            entry = dict(w, file=fname,
                         rendered=res is not None, **(res or {}))
            meta[zone['id']]['windows'].append(entry)
            print(f"{zone['id']} {tag}: "
                  + (f"done (dz {['%.1f' % d for d in w['dz']]}, "
                     f"zone nodes in slabs "
                     f"{res['zone_nodes_in_slabs']})" if res else 'SKIPPED'),
                  flush=True)

    with open(os.path.join(args.out, 'confirm_meta.json'), 'w',
              encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"meta at {os.path.join(args.out, 'confirm_meta.json')}",
          flush=True)


if __name__ == '__main__':
    main()
