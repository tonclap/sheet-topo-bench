"""Real-corpus A builder: the recto/m7 disagreement map of Paris 4 (TOPO-010).

Everything this script decides is declared in `CORPUS.md` *before* it ran; the
script is the executable form of that declaration, not a second one. Two
published surface predictions of the same scan — recto (level 2) and m7
(level 0) — share the annotation frame voxel for voxel, so a mesh node can ask
both models the same question the detector's U-011 channel asks one of them:
is there a predicted sheet within ±2/±2/±1 vx of my trace? A node exactly one
model supports is a *disagreement point*; where the two independent models
argue about the trace is a cheap, noisy label for a real topological problem
in mesh or prediction — the corpus PROTOCOL §5.A names as source A.

The domain is the U-011 support band (rows within ±160 vx of the five corpus
z-quantiles, dev bands only), deliberately: the synthetic corpus and the real
map are then measured over the same rows, and the chunk traffic stays at the
level today's runs already cached. Sampling is batched per row across all 75
offsets — one `sample` call — because per-offset calls on an outer mesh touch
more chunks than the reader's LRU holds and re-decompress the ring per call
(the 40-minute stall of the 14.08 full run).

Checkpointing is per segment (`cells/<name>.json`): kill and rerun resumes
where it stopped — the incremental-checkpoint lesson is load-bearing here
because the m7 cache is cold and the first pass is network-bound.

Usage:

    python build_disagreement.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --m7-cache ../../output/m7cache --out ../../output/topo/real_paris4
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
import scrolls                                                        # noqa: E402
import absolute_winding_calibration as awc                            # noqa: E402

M7 = ('PHercParis4/representations/predictions/surfaces/'
      '20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/')
M7_LEVEL = '0'          # the m7 zarr is computed on the L2 volume, so its own
                        # level 0 *is* the annotation frame (checked 14.08.2026:
                        # shape 18946x8174x8174 against recto level 2)
BAND_VX = 160.0         # the U-011 support band — same domain as the synthetic run
BLOCK = 8               # detector cell geometry, frozen
CELL_ROW_GAP = 2        # detector cluster connectivity, frozen
MIN_ROW_POINTS = 30
OFFSETS = np.array([[dx, dy, dz]
                    for dx in (-2, -1, 0, 1, 2)
                    for dy in (-2, -1, 0, 1, 2)
                    for dz in (-1, 0, 1)], float)


def supported(prediction, points, threshold, seen_values):
    """Which nodes have a predicted sheet within the U-011 neighbourhood."""
    stacked = points[None, :, :] + OFFSETS[:, None, :]
    values = prediction.sample(stacked.reshape(-1, 3))
    for v, c in zip(*np.unique(values, return_counts=True)):
        seen_values[int(v)] = seen_values.get(int(v), 0) + int(c)
    return values.reshape(len(OFFSETS), -1).max(axis=0) >= threshold


def segment_cells(name, scroll, grid_cache, recto, m7, z_quantiles):
    """One segment's per-cell disagreement counts, plus row-level statistics."""
    grid = scrolls.segment_grid(name, scroll, grid_cache)
    heights, valid = scrolls.row_heights(grid)
    cells = {}
    stats = {'rows': 0, 'nodes': 0, 'recto_supported': 0, 'm7_supported': 0,
             'disagree': 0, 'recto_only': 0, 'm7_only': 0, 'both_silent': 0,
             'values_recto': {}, 'values_m7': {}}
    for row in range(grid.shape[0]):
        z = heights[row]
        if not np.isfinite(z) or not any(
                abs(z - q) <= BAND_VX for q in z_quantiles):
            continue
        mask = valid[row]
        if mask.sum() < MIN_ROW_POINTS:
            continue
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)
        sup_r = supported(recto, points, scroll.threshold,
                          stats['values_recto'])
        sup_m = supported(m7, points, scroll.threshold, stats['values_m7'])
        disagree = sup_r != sup_m
        stats['rows'] += 1
        stats['nodes'] += int(len(cols))
        stats['recto_supported'] += int(sup_r.sum())
        stats['m7_supported'] += int(sup_m.sum())
        stats['disagree'] += int(disagree.sum())
        stats['recto_only'] += int((sup_r & ~sup_m).sum())
        stats['m7_only'] += int((~sup_r & sup_m).sum())
        stats['both_silent'] += int((~sup_r & ~sup_m).sum())
        for col, r_only in zip(cols[disagree], (sup_r & ~sup_m)[disagree]):
            key = f'{row}_{int(col) // BLOCK}'
            cell = cells.setdefault(key, {'n': 0, 'recto_only': 0})
            cell['n'] += 1
            cell['recto_only'] += int(r_only)
    return cells, stats


def cluster_zones(cells, cell_floor):
    """Evidence cells clustered by the detector's connectivity.

    `cell_floor` is the CORPUS.md insert of 14.08: at the measured 21.8 %
    background disagreement rate, clustering every touched cell percolates
    into band-wide blobs (strongest clusters = whole ±160 vx bands, up to
    1712 columns), which the §3 hit rule cannot score. A cell testifies only
    if >= `cell_floor` of its <= 8 nodes disagree (floor 5 ~ 3x background,
    binomial tail ~0.006) — the minimal floor at which percolation breaks
    (max zone width 1712 -> 176 columns). No height cut-off: zones are
    labels, not detector candidates."""
    keys = [tuple(map(int, key.split('_'))) for key in cells
            if cells[key]['n'] >= cell_floor]
    parent = {key: key for key in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    lookup = set(keys)
    for row, blk in keys:
        for dr in range(-CELL_ROW_GAP, CELL_ROW_GAP + 1):
            for db in (-1, 0, 1):
                other = (row + dr, blk + db)
                if other != (row, blk) and other in lookup:
                    parent[find(other)] = find((row, blk))

    clusters = {}
    for key in keys:
        clusters.setdefault(find(key), []).append(key)

    zones = []
    for members in clusters.values():
        rows = [r for r, _ in members]
        blocks = [b for _, b in members]
        data = [cells[f'{r}_{b}'] for r, b in members]
        zones.append({
            'row_lo': min(rows), 'row_hi': max(rows) + 1,
            'col_lo': min(blocks) * BLOCK, 'col_hi': (max(blocks) + 1) * BLOCK,
            'mass': sum(d['n'] for d in data),
            'recto_only': sum(d['recto_only'] for d in data),
            'cells': len(members)})
    return zones


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True, help='recto/CT chunk cache')
    parser.add_argument('--m7-cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cell-floor', type=int, default=5,
                        help='disagreement points a cell needs to testify '
                             '(CORPUS.md insert of 14.08)')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    z_quantiles = manifest['z_quantiles']
    names = sorted({r['segment'] for r in manifest['injections']
                    if r['winding_low'] < 100})

    cells_dir = os.path.join(args.out, 'cells')
    os.makedirs(cells_dir, exist_ok=True)
    recto = scrolls.open_prediction(scroll, args.cache, max_chunks=192)
    m7 = awc.Prediction(args.m7_cache, base=M7, level=M7_LEVEL, max_chunks=256)

    for i, name in enumerate(names):
        path = os.path.join(cells_dir, f'{name}.json')
        if os.path.exists(path):
            print(f'[{i + 1}/{len(names)}] {name}: checkpoint exists, skipping',
                  flush=True)
            continue
        cells, stats = segment_cells(name, scroll, args.grid_cache, recto, m7,
                                     z_quantiles)
        tmp = f'{path}.{os.getpid()}.part'
        with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
            json.dump({'segment': name, 'cells': cells, 'stats': stats}, f,
                      ensure_ascii=False)
        os.replace(tmp, path)
        print(f'[{i + 1}/{len(names)}] {name}: {stats["nodes"]} nodes, '
              f'{stats["disagree"]} disagreements '
              f'({stats["recto_only"]} recto-only / {stats["m7_only"]} m7-only), '
              f'{stats["both_silent"]} both-silent', flush=True)

    zones, totals = [], {}
    value_histograms = {'recto': {}, 'm7': {}}
    for name in names:
        with open(os.path.join(cells_dir, f'{name}.json'), encoding='utf-8') as f:
            data = json.load(f)
        for zone in cluster_zones(data['cells'], args.cell_floor):
            zones.append(dict(zone, segment=name))
        for key, value in data['stats'].items():
            if key.startswith('values_'):
                hist = value_histograms[key.split('_', 1)[1]]
                for v, c in value.items():
                    hist[v] = hist.get(v, 0) + c
            else:
                totals[key] = totals.get(key, 0) + value

    zones.sort(key=lambda z: -z['mass'])
    masses = np.array([z['mass'] for z in zones])
    report = {
        'sources': {'recto': scroll.prediction, 'recto_level': scroll.level,
                    'm7': M7, 'm7_level': M7_LEVEL,
                    'threshold': scroll.threshold,
                    'neighbourhood_vx': [2, 2, 1]},
        'corpus': os.path.abspath(args.corpus),
        'cell_floor': args.cell_floor,
        'z_quantiles': z_quantiles, 'band_vx': BAND_VX,
        'segments': names,
        'totals': totals,
        'support_rate': {
            'recto': round(totals['recto_supported'] / totals['nodes'], 4),
            'm7': round(totals['m7_supported'] / totals['nodes'], 4),
            'disagree': round(totals['disagree'] / totals['nodes'], 4),
            'both_silent': round(totals['both_silent'] / totals['nodes'], 4)},
        'value_histograms': value_histograms,
        'mass_quantiles': {f'q{q}': float(np.quantile(masses, q))
                           for q in (0.5, 0.75, 0.9, 0.95, 0.99)} if len(masses)
                          else None,
        'n_clusters': len(zones),
        'zones': zones}
    out = os.path.join(args.out, 'disagreement.json')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f"{totals['nodes']} nodes over {totals['rows']} rows; "
          f"support recto {report['support_rate']['recto']:.1%} / "
          f"m7 {report['support_rate']['m7']:.1%}; "
          f"disagree {report['support_rate']['disagree']:.1%} "
          f"({totals['recto_only']} recto-only / {totals['m7_only']} m7-only); "
          f"both-silent {report['support_rate']['both_silent']:.1%}")
    print(f"{len(zones)} clusters; mass quantiles {report['mass_quantiles']}")
    print(f"map at {out}")


if __name__ == '__main__':
    main()
