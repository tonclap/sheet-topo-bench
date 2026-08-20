"""Real-corpus B builder: mesh-epoch discrepancies against the GP-banner (TOPO-014).

Everything this script decides is declared in `CORPUS.md` (insert of
17.08.2026) *before* it ran; the script is the executable form of that
declaration. The reference is the human-verified GP-banner
(`20231231235900_GP.obj`, 2023 frame), carried into the 2026 L2 frame by the
global affine measured in `frame_2023.py` (normal residual 0.39 vx L2 against
a pre-declared 6.7 vx bar). A valid node of a mesh is a *discrepancy point*
when it lies farther than the detector's frozen contact scale (T_disc = 5 vx
L2) from the banner surface sample, yet within the banner's zone of
responsibility (d <= T_cover, declared from the collected histogram before
clustering). No prediction is read anywhere: the label is purely geometric,
mesh against human-verified reference — corpus B has no circularity to any
model by construction.

Three phases, mirroring corpus A's declare-then-run flow:

    collect   distances for all domain nodes (per-segment npz checkpoints,
              pooled histogram, banner bbox and edge-length report)
    diagnose  percolation and cell-weight diagnostics for candidate floors
              at a given T_cover (feeds the dated CORPUS.md insert)
    map       zones (2026 production segments only) + the epoch table, at the
              declared T_cover / cell floor; output in the corpus-A schema so
              `eval_real.py --map .../corpusB.json` runs unchanged

Usage (from oneshot/real_errors/):

    python build_corpusB.py collect --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --frame2023-cache ../../output/frame2023 \
        --banner ../../output/corpusB/20231231235900_GP.obj \
        --frame-report ../../output/topo/corpusB_frame.json \
        --out ../../output/topo/real_paris4
    python build_corpusB.py diagnose --out ../../output/topo/real_paris4 --t-cover <T>
    python build_corpusB.py map --out ../../output/topo/real_paris4 \
        --t-cover <T> --cell-floor <K> --corpus ../../output/topo/corpus_paris4
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
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'figures'))
import scrolls                                                        # noqa: E402

T_DISC = 5.0            # frozen contact scale, vx L2 (CONTACT_REF=10 * PROX_FLOOR)
BAND_VX = 160.0         # the A/A2 domain, byte-for-byte
MIN_ROW_POINTS = 30
BLOCK = 8
CELL_ROW_GAP = 2
EDGE_MEDIAN_MAX = 4.0   # densification rule: midpoints added above this, vx L2
BBOX_MARGIN = 64.0      # nodes outside banner bbox + margin skip the KD query
FRAME_2026 = '20260411134726-2.4um'
HIST_EDGES = np.concatenate([np.arange(0, 20.5, 0.5),
                             np.arange(21, 101, 1.0),
                             [150, 200, 300, 500, 1e9]])


def parse_obj(path):
    """Vertices and faces of a (possibly slash-annotated) OBJ, cached as npz."""
    cache = path + '.npz'
    if os.path.exists(cache):
        data = np.load(cache)
        return data['vertices'], data['faces']
    vertices, faces = [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('v '):
                _, a, b, c = line.split()[:4]
                vertices.append((float(a), float(b), float(c)))
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) for tok in line.split()[1:4]]
                faces.append(idx)
    vertices = np.array(vertices, np.float64)
    faces = np.array(faces, np.int64) - 1
    np.savez_compressed(cache, vertices=vertices, faces=faces)
    return vertices, faces


def banner_l2(banner_path, frame_report):
    """The banner's reference sample in L2: affine to the 2026 L0, then /4."""
    with open(frame_report, encoding='utf-8') as f:
        frame = json.load(f)
    x = np.array(frame['global_affine']['matrix_rows_then_translation'],
                 np.float64)
    vertices, faces = parse_obj(banner_path)
    l2 = (vertices @ x[:3] + x[3]) / 4.0
    edges = np.linalg.norm(l2[faces[:, 0]] - l2[faces[:, 1]], axis=1)
    edge_median = float(np.median(edges))
    sample = [l2, l2[faces].mean(axis=1)]
    densified = edge_median > EDGE_MEDIAN_MAX
    if densified:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            sample.append((l2[faces[:, i]] + l2[faces[:, j]]) / 2.0)
    sample = np.concatenate(sample).astype(np.float32)
    info = {'vertices': int(len(vertices)), 'faces': int(len(faces)),
            'edge_median_l2': round(edge_median, 3),
            'densified_with_edge_midpoints': bool(densified),
            'sample_points': int(len(sample)),
            'bbox_l2': {axis: [round(float(l2[:, k].min()), 1),
                               round(float(l2[:, k].max()), 1)]
                        for k, axis in enumerate('xyz')}}
    return sample, info


def domain_nodes(grid, z_quantiles):
    """(rows, cols, points) of valid domain nodes — the A/A2 row rules."""
    heights, valid = scrolls.row_heights(grid)
    rows_out, cols_out, pts_out = [], [], []
    for row in range(grid.shape[0]):
        z = heights[row]
        if not np.isfinite(z) or not any(
                abs(z - q) <= BAND_VX for q in z_quantiles):
            continue
        mask = valid[row]
        if mask.sum() < MIN_ROW_POINTS:
            continue
        cols = np.where(mask)[0]
        rows_out.append(np.full(len(cols), row, np.int32))
        cols_out.append(cols.astype(np.int32))
        pts_out.append(grid[row][mask].astype(np.float64))
    if not rows_out:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty((0, 3), np.float64))
    return (np.concatenate(rows_out), np.concatenate(cols_out),
            np.concatenate(pts_out))


def load_2023_grid(cache, seg):
    import tifffile
    channels = []
    for c in 'xyz':
        path = os.path.join(cache, f'{seg}-on-{FRAME_2026}-{c}.tif')
        channels.append(tifffile.imread(path))
    return np.stack(channels, -1).astype(np.float32)


def collect(args):
    from scipy.spatial import cKDTree

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    z_quantiles = manifest['z_quantiles']
    names = sorted({r['segment'] for r in manifest['injections']
                    if r['winding_low'] < 100})

    sample, info = banner_l2(args.banner, args.frame_report)
    print(f"banner: {info['vertices']} vertices / {info['faces']} faces, "
          f"edge median {info['edge_median_l2']} vx L2 "
          f"(midpoints {'added' if info['densified_with_edge_midpoints'] else 'not needed'}), "
          f"{info['sample_points']} sample points, bbox {info['bbox_l2']}",
          flush=True)
    lo = sample.min(axis=0) - BBOX_MARGIN
    hi = sample.max(axis=0) + BBOX_MARGIN
    tree = cKDTree(sample)
    print('KD-tree built', flush=True)

    cells_dir = os.path.join(args.out, 'cells_corpusB')
    os.makedirs(cells_dir, exist_ok=True)

    def distances(points):
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        d = np.full(len(points), np.inf, np.float32)
        if inside.any():
            d[inside], _ = tree.query(points[inside], workers=-1)
        return d

    hist_total = {}

    def one_segment(seg, grid, epoch):
        path = os.path.join(cells_dir, f'{seg}.npz')
        if os.path.exists(path):
            data = np.load(path)
            print(f'{seg}: checkpoint exists ({len(data["d"])} nodes), skipping',
                  flush=True)
            return data['d']
        rows, cols, pts = domain_nodes(grid, z_quantiles)
        d = distances(pts)
        tmp = f'{path}.{os.getpid()}.part.npz'
        np.savez_compressed(tmp, rows=rows, cols=cols, d=d,
                            epoch=np.array([epoch]))
        os.replace(tmp, path)
        finite = np.isfinite(d)
        print(f'{seg} ({epoch}): {len(d)} domain nodes, '
              f'{int(finite.sum())} in banner bbox, '
              f'd median (finite) '
              f'{float(np.median(d[finite])) if finite.any() else float("nan"):.2f}',
              flush=True)
        return d

    for seg in names:
        grid = scrolls.segment_grid(seg, scroll, args.grid_cache)
        d = one_segment(seg, grid, '2026')
        hist_total.setdefault('2026', np.zeros(len(HIST_EDGES) - 1, np.int64))
        hist_total['2026'] += np.histogram(d[np.isfinite(d)], HIST_EDGES)[0]

    with open(args.frame_report, encoding='utf-8') as f:
        frame = json.load(f)
    for seg in frame['segments']:
        grid = load_2023_grid(args.frame2023_cache, seg) / 4.0
        d = one_segment(seg, grid, '2023')
        hist_total.setdefault('2023', np.zeros(len(HIST_EDGES) - 1, np.int64))
        hist_total['2023'] += np.histogram(d[np.isfinite(d)], HIST_EDGES)[0]

    stats = {'banner': info, 't_disc': T_DISC,
             'segments_2026': names, 'segments_2023': frame['segments'],
             'hist_edges': [float(e) for e in HIST_EDGES],
             'hist': {k: [int(c) for c in v] for k, v in hist_total.items()}}
    out = os.path.join(args.out, 'corpusB_collect.json')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, sort_keys=True)
    for epoch, hist in sorted(hist_total.items()):
        total = hist.sum()
        print(f'--- {epoch}: {total} nodes in bbox; histogram (vx L2: count)')
        for k in range(len(hist)):
            if hist[k]:
                print(f'  {HIST_EDGES[k]:.1f}-{HIST_EDGES[k + 1]:.1f}: {hist[k]}')
    print(f'collect report at {out}')


def load_checkpoints(out, epoch_filter=None):
    cells_dir = os.path.join(out, 'cells_corpusB')
    segments = {}
    for fn in sorted(os.listdir(cells_dir)):
        if not fn.endswith('.npz'):
            continue
        data = np.load(os.path.join(cells_dir, fn))
        epoch = str(data['epoch'][0])
        if epoch_filter and epoch != epoch_filter:
            continue
        segments[fn[:-4]] = (data['rows'], data['cols'], data['d'], epoch)
    return segments


def segment_cells(rows, cols, d, t_cover):
    covered = d <= t_cover
    disc = covered & (d > T_DISC)
    cells = {}
    for row, col in zip(rows[disc], cols[disc]):
        key = f'{row}_{int(col) // BLOCK}'
        cells[key] = cells.get(key, 0) + 1
    return cells, int(covered.sum()), int(disc.sum())


def cluster_zones(cells, cell_floor):
    keys = [tuple(map(int, key.split('_'))) for key in cells
            if cells[key] >= cell_floor]
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
        zones.append({
            'row_lo': min(rows), 'row_hi': max(rows) + 1,
            'col_lo': min(blocks) * BLOCK, 'col_hi': (max(blocks) + 1) * BLOCK,
            'mass': sum(cells[f'{r}_{b}'] for r, b in members),
            'cells': len(members)})
    return zones


def diagnose(args):
    segments = load_checkpoints(args.out, epoch_filter='2026')
    covered_total = disc_total = 0
    per_floor = {k: {'clusters': 0, 'max_width': 0, 'masses': []}
                 for k in range(1, 9)}
    for seg, (rows, cols, d, _) in segments.items():
        cells, covered, disc = segment_cells(rows, cols, d, args.t_cover)
        covered_total += covered
        disc_total += disc
        for floor in per_floor:
            for z in cluster_zones(cells, floor):
                per_floor[floor]['clusters'] += 1
                width = z['col_hi'] - z['col_lo']
                per_floor[floor]['max_width'] = max(
                    per_floor[floor]['max_width'], width)
                per_floor[floor]['masses'].append(z['mass'])
    background = disc_total / covered_total if covered_total else 0.0
    print(f'T_cover={args.t_cover}: covered {covered_total}, discrepant '
          f'{disc_total} (background {background:.1%})')
    from scipy.stats import binom
    for floor, entry in per_floor.items():
        masses = np.array(entry['masses']) if entry['masses'] else np.array([0])
        tail = float(binom.sf(floor - 1, 8, background))
        print(f"  floor {floor}: {entry['clusters']} clusters, max width "
              f"{entry['max_width']} cols, mass q50/q75/q90/q95 "
              f"{np.quantile(masses, [.5, .75, .9, .95]).round(1).tolist()}, "
              f"binomial tail P(X>={floor}|n=8,p={background:.3f}) = {tail:.4f}")


def build_map(args):
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(os.path.join(args.out, 'corpusB_collect.json'),
              encoding='utf-8') as f:
        collect_report = json.load(f)

    zones, totals = [], {}
    epoch_stats = {}
    for seg, (rows, cols, d, epoch) in load_checkpoints(args.out).items():
        cells, covered, disc = segment_cells(rows, cols, d, args.t_cover)
        entry = epoch_stats.setdefault(epoch, {'segments': 0, 'nodes': 0,
                                               'covered': 0, 'discrepant': 0})
        entry['segments'] += 1
        entry['nodes'] += len(d)
        entry['covered'] += covered
        entry['discrepant'] += disc
        if epoch == '2026':
            for zone in cluster_zones(cells, args.cell_floor):
                zones.append(dict(zone, segment=seg))
            totals[seg] = {'covered': covered, 'discrepant': disc}
    for entry in epoch_stats.values():
        entry['rate'] = (round(entry['discrepant'] / entry['covered'], 4)
                         if entry['covered'] else None)

    # Sensitivity of the coverage bound, as declared: point counts only.
    sensitivity = {}
    for factor in (0.75, 1.25):
        c = disc = 0
        for seg, (rows, cols, d, epoch) in load_checkpoints(
                args.out, epoch_filter='2026').items():
            _, cov, dis = segment_cells(rows, cols, d, args.t_cover * factor)
            c += cov
            disc += dis
        sensitivity[f'x{factor}'] = {'covered': c, 'discrepant': disc}

    zones.sort(key=lambda z: -z['mass'])
    masses = np.array([z['mass'] for z in zones])
    seg_with_zones = sorted({z['segment'] for z in zones})
    report = {
        'sources': {'reference': 'GP-banner 20231231235900_GP via global '
                                 'affine corpusB_frame.json, /4 to L2',
                    'banner': collect_report['banner'],
                    't_disc': T_DISC, 't_cover': args.t_cover,
                    'cell_floor': args.cell_floor},
        'corpus': os.path.abspath(args.corpus),
        'cell_floor': args.cell_floor,
        'z_quantiles': manifest['z_quantiles'], 'band_vx': BAND_VX,
        'segments': seg_with_zones,
        'segments_all_2026': collect_report['segments_2026'],
        'per_segment_2026': totals,
        'epoch_table': epoch_stats,
        'coverage_sensitivity': sensitivity,
        'mass_quantiles': {f'q{q}': float(np.quantile(masses, q))
                           for q in (0.5, 0.75, 0.9, 0.95, 0.99)} if len(masses)
                          else None,
        'n_clusters': len(zones),
        'zones': zones}
    out = os.path.join(args.out, 'corpusB.json')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print('epoch table:', json.dumps(epoch_stats, sort_keys=True))
    print(f"{len(zones)} clusters over {len(seg_with_zones)} segments; "
          f"mass quantiles {report['mass_quantiles']}")
    print(f'map at {out}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('collect')
    p.add_argument('--corpus', required=True)
    p.add_argument('--grid-cache', required=True)
    p.add_argument('--frame2023-cache', required=True)
    p.add_argument('--banner', required=True)
    p.add_argument('--frame-report', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=collect)
    p = sub.add_parser('diagnose')
    p.add_argument('--out', required=True)
    p.add_argument('--t-cover', type=float, required=True)
    p.set_defaults(func=diagnose)
    p = sub.add_parser('map')
    p.add_argument('--out', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--t-cover', type=float, required=True)
    p.add_argument('--cell-floor', type=int, required=True)
    p.set_defaults(func=build_map)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
