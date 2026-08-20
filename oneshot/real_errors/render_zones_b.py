"""Zone cards for the manual classification of support-credited corpus-B zones.

TOPO-029: the support signal on corpus B (AP 0.0330 against a random base
0.0003) becomes an addressed claim only if the 57 credited zones are read by
eye. The class criteria are declared in `ZONE_CRITERIA_B.md` BEFORE any card
was rendered or looked at; labels go to `zone_labels_b.csv`, summarised by
`summarize_zones_b.py`.

Card layout follows `render_zones.py` (three LOCAL axial windows along the
strip — a mesh row is not iso-z), with corpus-B overlays instead of the
recto/m7 support test:

- the banner reference sample (human-verified 2023 surface, 2026 L2 frame)
  inside a ±6 vx slab around the slice — cyan dots,
- all 2026 mesh nodes in the slab coloured by their distance to the banner
  from the corpus checkpoints: green d <= 5 (agree), red 5 < d <= 10
  (discrepant), dark grey d > 10 (uncovered), light grey — outside the
  measured domain,
- zone-rectangle nodes marked with yellow edges,
- the recto prediction contour (orange) — support credited the zone, so
  where recto leads a sheet and where it is silent is the thing to see.

Usage (from oneshot/real_errors/):
    python render_zones_b.py \
        --zones ../../output/topo/real_paris4/supportB_zones.json \
        --corpus ../../output/topo/corpus_paris4 \
        --banner ../../output/corpusB/20231231235900_GP.obj \
        --frame-report ../../output/topo/corpusB_frame.json \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --out ../../output/topo/real_paris4/zones_png_b [--limit N]
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
from build_corpusB import banner_l2, T_DISC                           # noqa: E402
from render_zones import ct_slice                                     # noqa: E402

HALF = 180        # local window half-side, vx (render_zones.py's)
SLAB = 6.0        # slab half-thickness around the slice, vx


def distance_classes(d):
    """Distance -> plot colour, the ZONE_CRITERIA_B.md legend."""
    colours = np.empty(len(d), object)
    colours[:] = '#555555'                    # covered, d > T_cover
    colours[d <= 10.0] = '#d62728'            # discrepant 5 < d <= 10
    colours[d <= T_DISC] = '#2ca02c'          # agree d <= 5
    colours[~np.isfinite(d)] = '#c8c8c8'      # outside banner bbox
    return colours


def render(zone, grid, heights, valid, node_d, banner, sm, recto, threshold,
           path):
    row_lo, row_hi = zone['row_lo'], zone['row_hi']
    col_lo, col_hi = zone['col_lo'], zone['col_hi']
    zone_rows = [r for r in range(row_lo, row_hi)
                 if np.isfinite(heights[r]) and valid[r].any()]
    if not zone_rows:
        return False
    rc = zone_rows[len(zone_rows) // 2]

    width = col_hi - col_lo
    anchors = []
    for frac in (1 / 6, 1 / 2, 5 / 6):
        want = col_lo + int(frac * width)
        cols = np.where(valid[rc])[0]
        cols = cols[(cols >= col_lo - 8) & (cols < col_hi + 8)]
        if not len(cols):
            continue
        col = int(cols[np.argmin(np.abs(cols - want))])
        pt = grid[rc, col].astype(float)
        if not anchors or abs(anchors[-1][0] - col) > 12:
            anchors.append((col, pt))
    if not anchors:
        return False

    all_valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    rows_idx, cols_idx = np.where(all_valid)
    all_pts = grid[rows_idx, cols_idx].astype(np.float64)
    # Distance per node from the corpus checkpoints; NaN = outside the
    # measured domain (non-band rows) — light grey, not a claim.
    d_all = np.full(len(all_pts), np.nan, np.float32)
    for k, (r, c) in enumerate(zip(rows_idx, cols_idx)):
        d = node_d.get((int(r), int(c)))
        if d is not None:
            d_all[k] = d

    fig, axes = plt.subplots(1, len(anchors),
                             figsize=(5.4 * len(anchors), 5.8))
    if len(anchors) == 1:
        axes = [axes]
    for ax, (col, pt) in zip(axes, anchors):
        zi = int(round(pt[2]))
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
        f"{zone['id']}  {zone['segment']}  rows {row_lo}-{row_hi} "
        f"cols {col_lo}-{col_hi}  mass {zone['mass']} "
        f"support rank {zone['support_rank']}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)   # dpi rationale: render_zones.py
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zones', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--banner', required=True)
    parser.add_argument('--frame-report', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=None,
                        help='stop after this many NEWLY rendered zones')
    args = parser.parse_args()

    with open(args.zones, encoding='utf-8') as f:
        zones = json.load(f)['zones']
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
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
    attempted = 0
    for i, zone in enumerate(sorted(zones, key=lambda z: z['id'])):
        meta[zone['id']] = dict(zone)
        path = os.path.join(args.out, f"{zone['id']}.png")
        if os.path.exists(path):
            print(f"[{i + 1}/{len(zones)}] {zone['id']}: exists", flush=True)
            continue
        if args.limit is not None and attempted >= args.limit:
            print(f"[{i + 1}/{len(zones)}] {zone['id']}: "
                  f"skipped (--limit {args.limit} reached)", flush=True)
            continue
        name = zone['segment']
        if name not in grids:
            g = scrolls.segment_grid(name, scroll, args.grid_cache)
            grids[name] = (g, *scrolls.row_heights(g))
            ck = np.load(os.path.join(cells_dir, f'{name}.npz'))
            dmaps[name] = {(int(r), int(c)): float(d) for r, c, d
                           in zip(ck['rows'], ck['cols'], ck['d'])}
        grid, heights, valid = grids[name]
        attempted += 1
        done = render(zone, grid, heights, valid, dmaps[name], banner, sm,
                      recto, scroll.threshold, path)
        print(f"[{i + 1}/{len(zones)}] {zone['id']}: "
              + ('done' if done else 'SKIPPED (no valid points)'), flush=True)
    # meta.json covers the FULL zone list every run, even under --limit.
    with open(os.path.join(args.out, 'zones_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
