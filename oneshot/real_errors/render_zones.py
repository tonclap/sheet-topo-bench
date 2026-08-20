"""Zone cards for the manual classification of the 124 double-evidence zones.

TOPO-020: the statistical signal of the real corpus (prox AP 0.0081 against a
random base 0.0015) becomes an addressed claim only if the 124 zones where the
detector and the recto/m7 disagreement map point at the same place are read by
eye. This renders one PNG card per zone; the class criteria are declared in
`ZONE_CRITERIA.md` BEFORE any card is looked at, and the labels go to
`zone_labels.csv`, summarised by `summarize_zones.py` — the summary sentence
comes from the same code as the table.

A zone is a long strip along the sheet (~10 vx per grid column), and a mesh
row is NOT iso-z (checked on Z0001: ±100 vx along one row), so one global
axial slice cannot carry the trace. Each card therefore shows up to three
LOCAL axial windows along the strip (at 1/6, 1/2, 5/6 of the zone's columns,
central zone row), each sliced at the local trace height, overlaid with
- the recto prediction (red contour) and the m7 prediction (cyan contour)
  sampled on that slice,
- a ±2.5 vx slab of ALL mesh rows' nodes around the slice, coloured by the
  same U-011 support test the disagreement map used: green = both models
  support, orange = recto only, blue = m7 only, red = both silent,
- nodes belonging to the zone's row/column range marked with yellow edges.

The neighbouring windings of the same multi-winding mesh appear in the
window by construction — a switch's contact with them is the thing to see.

Usage:
    python oneshot/real_errors/render_zones.py \
        --eval output/topo/real_paris4/eval.json \
        --corpus output/topo/corpus_paris4 \
        --grid-cache output/figgrids --cache output/figcache \
        --m7-cache output/m7cache \
        --out output/topo/real_paris4/zones_png
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
from build_disagreement import M7, M7_LEVEL, OFFSETS                  # noqa: E402

HALF = 180        # local window half-side, vx
SLAB = 6.0        # trace slab half-thickness around the slice, vx


def ct_slice(sm, zi, xs, ys):
    """Grey axial CT patch [ys[0]:ys[1], xs[0]:xs[1]] at voxel plane zi,
    assembled from ScanMask's cached 16 KB chunk planes."""
    c = sm.chunk
    img = np.zeros((ys[1] - ys[0], xs[1] - xs[0]), np.uint8)
    if not (0 <= zi < sm.shape[0]):
        return img
    for cy in range(max(ys[0], 0) // c, (min(ys[1], sm.shape[1]) - 1) // c + 1):
        for cx in range(max(xs[0], 0) // c,
                        (min(xs[1], sm.shape[2]) - 1) // c + 1):
            plane = sm._plane(zi // c, cy, cx, zi % c)
            y0, y1 = max(cy * c, ys[0]), min((cy + 1) * c, ys[1])
            x0, x1 = max(cx * c, xs[0]), min((cx + 1) * c, xs[1])
            if y0 >= y1 or x0 >= x1:
                continue
            img[y0 - ys[0]:y1 - ys[0], x0 - xs[0]:x1 - xs[0]] = \
                plane[y0 - cy * c:y1 - cy * c, x0 - cx * c:x1 - cx * c]
    return img


def support_colours(points, recto, m7, threshold):
    """U-011 support pattern per node -> plot colour."""
    def supported(prediction):
        stacked = points[None, :, :] + OFFSETS[:, None, :]
        values = prediction.sample(stacked.reshape(-1, 3))
        return values.reshape(len(OFFSETS), -1).max(axis=0) >= threshold
    sup_r, sup_m = supported(recto), supported(m7)
    colours = np.empty(len(points), object)
    colours[sup_r & sup_m] = '#2ca02c'      # both
    colours[sup_r & ~sup_m] = '#ff7f0e'     # recto only
    colours[~sup_r & sup_m] = '#1f77b4'     # m7 only
    colours[~sup_r & ~sup_m] = '#d62728'    # both silent
    return colours


def render(zone, grid, heights, valid, sm, recto, m7, threshold, path):
    row_lo, row_hi = zone['row_lo'], zone['row_hi']
    col_lo, col_hi = zone['col_lo'], zone['col_hi']
    zone_rows = [r for r in range(row_lo, row_hi)
                 if np.isfinite(heights[r]) and valid[r].any()]
    if not zone_rows:
        return False
    rc = zone_rows[len(zone_rows) // 2]

    # Anchor points along the strip: 1/6, 1/2, 5/6 of the zone's columns on
    # the central zone row (nearest valid column to each).
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

    # All valid nodes of the mesh once, for the slab overlay.
    all_valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    rows_idx, cols_idx = np.where(all_valid)
    all_pts = grid[rows_idx, cols_idx].astype(np.float64)

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
        gx, gy = np.meshgrid(np.arange(xs[0], xs[1], 2),
                             np.arange(ys[0], ys[1], 2))
        flat = np.column_stack([gx.ravel(), gy.ravel(),
                                np.full(gx.size, float(zi))])
        for pred, colour in ((recto, '#ff3333'), (m7, '#00cccc')):
            mask = (pred.sample(flat) >= threshold).reshape(gx.shape)
            if mask.any():
                ax.contour(gx, gy, mask.astype(float), levels=[0.5],
                           colors=colour, linewidths=0.7, alpha=0.85)
        slab = ((np.abs(all_pts[:, 2] - zi) <= SLAB)
                & (all_pts[:, 0] >= xs[0]) & (all_pts[:, 0] < xs[1])
                & (all_pts[:, 1] >= ys[0]) & (all_pts[:, 1] < ys[1]))
        pts_in = all_pts[slab]
        if len(pts_in):
            colours = support_colours(pts_in, recto, m7, threshold)
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
    share = zone['recto_only'] / max(zone['mass'], 1)
    fig.suptitle(
        f"{zone['id']}  {zone['segment']}  rows {row_lo}-{row_hi} "
        f"cols {col_lo}-{col_hi}  mass {zone['mass']} "
        f"(recto-only {share:.0%})  rank {zone['detector_rank']}",
        fontsize=10)
    fig.tight_layout()
    # dpi=70 keeps each panel near the raw CT resolution (HALF*2=360 vx across
    # a ~5.4in axis, i.e. ~67 px/in native) — 105 was oversampling and made
    # cards expensive to review: reading ~60 of them in one session was
    # burning the session's token budget in minutes.
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--m7-cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=None,
                        help='stop after this many NEWLY rendered zones '
                             '(already-existing cards do not count against '
                             'it) -- keeps a single run small after the '
                             'full-batch run got killed with no traceback')
    args = parser.parse_args()

    with open(args.eval, encoding='utf-8') as f:
        report = json.load(f)
    zones = [z for z in report['real']['primary']['candidates']
             if z['detector_rank'] is not None]
    zones.sort(key=lambda z: z['id'])
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]

    # max_chunks kept small: a decoded chunk is 16.8 MB and the first run of
    # this script was OOM-killed at 57/124 with the detector-sized caches
    # (192 + 256 chunks ~ 7.5 GB); the zone windows are local, so a small LRU
    # only costs re-decodes, not re-downloads.
    sm = awc.ScanMask(args.cache, base=scroll.ct, level=scroll.level)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=48)
    m7 = awc.Prediction(args.m7_cache, base=M7, level=M7_LEVEL, max_chunks=48)

    os.makedirs(args.out, exist_ok=True)
    grids = {}
    meta = {}
    attempted = 0
    for i, zone in enumerate(zones):
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
        grid, heights, valid = grids[name]
        attempted += 1
        done = render(zone, grid, heights, valid, sm, recto, m7,
                      scroll.threshold, path)
        print(f"[{i + 1}/{len(zones)}] {zone['id']}: "
              + ('done' if done else 'SKIPPED (no valid points)'), flush=True)
    # meta.json is written every run over the FULL zone list -- even under
    # --limit -- so summarize_zones.py (which requires exactly 124 entries)
    # never sees a partial file.
    with open(os.path.join(args.out, 'zones_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
