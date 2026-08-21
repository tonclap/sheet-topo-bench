"""TOPO-062: the TOPO-033 band on the rows neighbouring Z0136's row 680.

Protocol — the dated TOPO-062 insert in `ZONE_CRITERIA_B.md`, declared
BEFORE the first band was rendered. `band_zones_b.render_zone` is
imported unchanged (same columns, same +-30 vx normal, same banner
gate): `render_zone` takes the middle valid row of [row_lo, row_hi), so
a pseudo-zone with row_hi = row_lo + 1 pins the band to that one row.
Nothing else differs from TOPO-033 — that is the point of the check.

Usage (from pipeline/real_errors/):
    python band_neighbour_rows.py \
        --zones ../../output/topo/real_paris4/supportB_zones.json \
        --id Z0136 --rows 678,682 \
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'standalone'))
import scrolls                                                        # noqa: E402
import absolute_winding_calibration as awc                            # noqa: E402
from build_corpusB import banner_l2                                   # noqa: E402
import band_zones_b                                                   # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zones', required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--rows', required=True,
                        help='comma-separated grid rows, e.g. 678,682')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--banner', required=True)
    parser.add_argument('--frame-report', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.zones, encoding='utf-8') as f:
        zones = [z for z in json.load(f)['zones'] if z['id'] == args.id]
    if len(zones) != 1:
        raise SystemExit(f'{args.id}: expected exactly one zone, got {len(zones)}')
    zone = zones[0]
    rows = [int(r) for r in args.rows.split(',')]

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]

    banner, info = banner_l2(args.banner, args.frame_report)
    print(f"banner: {info['sample_points']} sample points in L2", flush=True)
    print('opening scan mask...', flush=True)
    sm = awc.ScanMask(args.cache, base=scroll.ct, level=scroll.level)
    print('opening prediction...', flush=True)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=48)
    print('readers ready', flush=True)

    grid = scrolls.segment_grid(zone['segment'], scroll, args.grid_cache)
    heights, valid = scrolls.row_heights(grid)

    os.makedirs(args.out, exist_ok=True)
    meta = {}
    bad = [r for r in rows if not 0 <= r < len(valid)]
    if bad:
        raise SystemExit(f'rows outside the grid (0..{len(valid) - 1}): {bad}')

    for row in rows:
        # The row must carry nodes inside the zone's own columns, or the band
        # would be a chain through a different part of the sheet and the
        # comparison with row 680 would not be like for like.
        in_zone = valid[row, zone['col_lo']:zone['col_hi']].sum()
        if not in_zone:
            print(f'row {row}: no valid node inside cols '
                  f"{zone['col_lo']}-{zone['col_hi']} — SKIPPED", flush=True)
            meta[f"{zone['id']}r{row}"] = None
            continue
        one_row = dict(zone, id=f"{zone['id']}r{row}",
                       row_lo=row, row_hi=row + 1)
        res = band_zones_b.render_zone(one_row, grid, heights, valid, banner,
                                       sm, recto, scroll.threshold, args.out)
        if res is not None:
            res['nodes_in_zone_cols'] = int(in_zone)
            res['row'] = row
        meta[one_row['id']] = res
        print(f"row {row}: " + (f"done (arc {res['arc_len']:.0f} vx, banner "
                                f"{res['banner_points']} pts, "
                                f"{in_zone} nodes in zone cols)"
                                if res else 'SKIPPED'), flush=True)

    path = os.path.join(args.out, 'band_neighbour_meta.json')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f'meta at {path}', flush=True)


if __name__ == '__main__':
    main()
