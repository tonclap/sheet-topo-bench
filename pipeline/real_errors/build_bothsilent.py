"""Real-corpus A2 builder: the both-silent map of Paris 4 (U-013, TOPO-015).

Declared in `CORPUS.md`, insert of 16.08.2026, before this ran. Corpus A
labelled the *argument* between two independent surface models; TOPO-020's
manual pass showed that argument to be mostly one model's silence (91/124
model_artifact). U-013 names the stronger form this script builds: a node
supported by *neither* model — double testimony against the trace instead of
single. One definition changes against A, one filter is added:

- a point of A2 is a valid node unsupported by recto AND unsupported by m7
  (the node-support definition itself is byte-for-byte corpus A's);
- nodes outside the scan mask are excluded and counted separately: there both
  models are silent trivially ("the scan sees nothing here"), not adversely
  (the wave-2 correction of 13.08 supplies the test — a masked-CT voxel
  reading exactly 0 was never scanned).

Domain, cell geometry, cluster connectivity, checkpointing — corpus A's,
imported from build_disagreement rather than copied. The cell floor and zone
mass threshold are NOT chosen here: the script prints the percolation table
(max zone width per candidate floor, the same evidence CORPUS.md's 14.08
insert used) and the mass distribution; the choice lands in a dated CORPUS.md
insert before eval_real.py runs on this map.

Usage:

    python build_bothsilent.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --m7-cache ../../output/m7cache --out ../../output/topo/real_paris4 \
        --cell-floor 0
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
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'standalone'))
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import absolute_winding_calibration as awc                            # noqa: E402
import build_disagreement as A                                        # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)

PART_EVERY = 40          # processed rows between mid-segment checkpoints


def segment_cells(name, scroll, grid_cache, recto, m7, scan_mask, z_quantiles,
                  part_path=None, mask_filter=True):
    """One segment's per-cell both-silent counts, plus row-level statistics.

    `part_path` is the mid-segment checkpoint: the first segment of the 16.08
    run computed for 40+ minutes and died on a DNS blip with nothing saved.
    Every PART_EVERY processed rows the accumulated state lands on disk; a
    rerun resumes at the first unprocessed row. Rows are independent, so the
    resumed result is identical to a straight run.
    """
    grid = scrolls.segment_grid(name, scroll, grid_cache)
    heights, valid = scrolls.row_heights(grid)
    cells = {}
    stats = {'rows': 0, 'nodes': 0, 'off_mask': 0, 'both_silent': 0,
             'values_recto': {}, 'values_m7': {}}
    start_row = 0
    if part_path and os.path.exists(part_path):
        with open(part_path, encoding='utf-8') as f:
            saved = json.load(f)
        cells, stats = saved['cells'], saved['stats']
        for key in ('values_recto', 'values_m7'):
            # JSON round-trips int histogram keys as strings; A.supported
            # will keep adding int keys, so normalise or the merge forks.
            stats[key] = {int(k): v for k, v in stats[key].items()}
        start_row = saved['next_row']
        print(f'  {name}: resuming at row {start_row}', flush=True)
    since_save = 0
    for row in range(start_row, grid.shape[0]):
        z = heights[row]
        if not np.isfinite(z) or not any(
                abs(z - q) <= A.BAND_VX for q in z_quantiles):
            continue
        mask = valid[row]
        if mask.sum() < A.MIN_ROW_POINTS:
            continue
        since_save += 1
        if part_path and since_save >= PART_EVERY:
            tmp = f'{part_path}.{os.getpid()}.part'
            with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
                json.dump({'cells': cells, 'stats': stats, 'next_row': row}, f,
                          ensure_ascii=False)
            os.replace(tmp, part_path)
            since_save = 0
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)
        sup_r = A.supported(recto, points, scroll.threshold,
                            stats['values_recto'])
        sup_m = A.supported(m7, points, scroll.threshold, stats['values_m7'])
        in_scan = (scan_mask.inside(points) if mask_filter
                   else np.ones(len(points), bool))
        silent = ~sup_r & ~sup_m & in_scan
        stats['rows'] += 1
        stats['nodes'] += int(len(cols))
        stats['off_mask'] += int((~in_scan).sum())
        stats['both_silent'] += int(silent.sum())
        for col in cols[silent]:
            key = f'{row}_{int(col) // A.BLOCK}'
            cell = cells.setdefault(key, {'n': 0})
            cell['n'] += 1
    return cells, stats


def percolation_table(per_segment_cells, floors=range(3, 9)):
    """Max zone width (columns) and zone count per candidate cell floor —
    the evidence the floor choice cites, printed before any choice is made."""
    table = []
    for floor in floors:
        widths, count = [], 0
        for cells in per_segment_cells.values():
            zones = A.cluster_zones(cells, floor)
            count += len(zones)
            widths += [z['col_hi'] - z['col_lo'] for z in zones]
        table.append({'floor': floor, 'zones': count,
                      'max_width_cols': max(widths) if widths else 0})
    return table


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True, help='recto/CT chunk cache')
    parser.add_argument('--m7-cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-mask-filter', action='store_true',
                        help='skip the scan-mask filter on dev bands, where '
                             'it was measured to exclude nothing (0 of '
                             '271929 nodes over segments 1-2, 17.08.2026) '
                             'while its cold CT reads dominate the runtime; '
                             'the filter stays mandatory for held-out and '
                             'outer bands — see the CORRECTION insert in '
                             'CORPUS.md. Recorded in the report as '
                             'mask_filter: false.')
    parser.add_argument('--cell-floor', type=int, default=0,
                        help='0 = build the map and print the percolation '
                             'table only (the floor is then declared in '
                             'CORPUS.md and the script rerun with it); '
                             'a positive value writes bothsilent.json with '
                             'zones at that declared floor')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    z_quantiles = manifest['z_quantiles']
    names = sorted({r['segment'] for r in manifest['injections']
                    if r['winding_low'] < 100})

    cells_dir = os.path.join(args.out, 'cells_bothsilent')
    os.makedirs(cells_dir, exist_ok=True)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=192)
    m7 = awc.Prediction(args.m7_cache, base=A.M7, level=A.M7_LEVEL,
                        max_chunks=256)
    scan_mask = scrolls.open_scan_mask(scroll, args.cache)

    for i, name in enumerate(names):
        path = os.path.join(cells_dir, f'{name}.json')
        if os.path.exists(path):
            print(f'[{i + 1}/{len(names)}] {name}: checkpoint exists, skipping',
                  flush=True)
            continue
        part_path = os.path.join(cells_dir, f'{name}.rows.json')
        cells, stats = segment_cells(name, scroll, args.grid_cache, recto, m7,
                                     scan_mask, z_quantiles,
                                     part_path=part_path,
                                     mask_filter=not args.no_mask_filter)
        tmp = f'{path}.{os.getpid()}.part'
        with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
            json.dump({'segment': name, 'cells': cells, 'stats': stats}, f,
                      ensure_ascii=False)
        os.replace(tmp, path)
        if os.path.exists(part_path):
            os.remove(part_path)
        print(f'[{i + 1}/{len(names)}] {name}: {stats["nodes"]} nodes, '
              f'{stats["both_silent"]} both-silent, '
              f'{stats["off_mask"]} off-mask', flush=True)

    per_segment_cells, totals = {}, {}
    value_histograms = {'recto': {}, 'm7': {}}
    for name in names:
        with open(os.path.join(cells_dir, f'{name}.json'), encoding='utf-8') as f:
            data = json.load(f)
        for cell in data['cells'].values():
            # A.cluster_zones sums a direction count that has no meaning for
            # both-silent cells; a zero keeps the reused clustering intact.
            cell.setdefault('recto_only', 0)
        per_segment_cells[name] = data['cells']
        for key, value in data['stats'].items():
            if key.startswith('values_'):
                hist = value_histograms[key.split('_', 1)[1]]
                for v, c in value.items():
                    hist[v] = hist.get(v, 0) + c
            else:
                totals[key] = totals.get(key, 0) + value

    print('\npercolation table (cell floor -> zones, max zone width, columns):')
    table = percolation_table(per_segment_cells)
    for entry in table:
        print(f"  floor {entry['floor']}: {entry['zones']} zones, "
              f"max width {entry['max_width_cols']}")
    rate = totals['both_silent'] / max(totals['nodes'], 1)
    print(f"\n{totals['nodes']} nodes over {totals['rows']} rows; "
          f"both-silent (in-mask) {rate:.1%}; "
          f"off-mask {totals['off_mask'] / max(totals['nodes'], 1):.2%}")

    if not args.cell_floor:
        print('\nno --cell-floor given: map built, no zones written; declare '
              'the floor in CORPUS.md and rerun with it')
        return

    zones = []
    for name in names:
        for zone in A.cluster_zones(per_segment_cells[name], args.cell_floor):
            zones.append(dict(zone, segment=name))
    zones.sort(key=lambda z: -z['mass'])
    masses = np.array([z['mass'] for z in zones])
    report = {
        'sources': {'recto': scroll.prediction, 'recto_level': scroll.level,
                    'm7': A.M7, 'm7_level': A.M7_LEVEL,
                    'threshold': scroll.threshold,
                    'neighbourhood_vx': [2, 2, 1],
                    'scan_mask': 'masked CT voxel == 0 is outside '
                                 '(wave-2 correction 13.08)'},
        'definition': 'both-silent: valid node inside the scan mask, '
                      'supported by neither recto nor m7 (U-013 / TOPO-015)',
        'corpus': os.path.abspath(args.corpus),
        'mask_filter': not args.no_mask_filter,
        'cell_floor': args.cell_floor,
        'z_quantiles': z_quantiles, 'band_vx': A.BAND_VX,
        'segments': names,
        'totals': totals,
        'rate_both_silent': round(rate, 4),
        'rate_off_mask': round(totals['off_mask'] / max(totals['nodes'], 1), 4),
        'value_histograms': value_histograms,
        'percolation_table': table,
        'mass_quantiles': {f'q{q}': float(np.quantile(masses, q))
                           for q in (0.5, 0.75, 0.9, 0.95, 0.99)} if len(masses)
                          else None,
        'n_clusters': len(zones),
        'zones': zones}
    out = os.path.join(args.out, 'bothsilent.json')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\n{len(zones)} zones at floor {args.cell_floor}; "
          f"mass quantiles {report['mass_quantiles']}")
    print(f"map at {out}")


if __name__ == '__main__':
    main()
