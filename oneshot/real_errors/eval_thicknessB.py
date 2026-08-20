"""TOPO-046: radial bright-structure thickness vs the corpus-B zones (U-019).

A real merger — two sheets riding one front — doubles the thickness of the
bright structure; this estimator reads only the papyrus/air boundary, the one
contrast a fusion zone still has (the kissing-bond limit). On synthetic
corpora the channel is undefined by the injection mechanism (the injector
does not thicken material), so the target is the human-verified corpus B
alone: a signal here would be the second prediction-free channel on real
labelling, in the slot where vjump read zero (TOPO-039).

The declaration (CORPUS.md insert, 18.08.2026, twenty-fourth session,
committed before this ran) fixes one row — **thick**:

- row scope = the frozen support-channel scope (rows within SUPPORT_BAND_VX
  of the corpus manifest's z-quantiles), all valid columns;
- probe_ct geometry verbatim (CTVolume, in-plane radial unit vector from the
  counting centre, z at the row height), offsets -24..+24 vx step 1;
- binarisation T = 80.0 — the frozen probe_ct calibration (TOPO-026),
  no new calibration; off-mask zeros read as air (truncation understates
  thickness, never overstates);
- node thickness = length of the contiguous >= T run containing offset 0;
  a node off papyrus carries no evidence; runs hitting the +-24 edge are
  taken as observed (saturation share published);
- reference = per-segment median thickness over nodes with thickness > 0
  (each grid against its own median — vjump's normalisation);
- evidence iff thickness >= THICK_REL_FLOOR = 2.0 x segment median, value =
  the ratio; (row, col // BLOCK) cells, `differenced_clusters` with an EMPTY
  atlas, MAX_CLUSTER_ROWS = 14 (the frozen v1 default — this channel has no
  v6 protocol commit), `merge_channels` ranking.
- Zones, thresholds (20/10/40), the 100-seed random baseline and the by-zone
  bootstrap (2000 resamples, seed 20260815) are TOPO-028's, byte for byte.

Reading rule (TOPO-028's, verbatim): the row signals iff the lower edge of
its 95 % bootstrap AP interval sits strictly above the upper edge of the
random baseline's IQR at the primary threshold; a primary signal that dies
at both sensitivity thresholds reads as unstable, not as a signal.

Per-segment checkpoints store per-node thickness, so evidence replays
offline without re-reading CT and a killed run resumes.

Usage:

    python eval_thicknessB.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --zone-min 20 --report ../../output/topo/real_paris4/eval_thicknessB.json
"""
import argparse
import json
import os
import pickle
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402
import sheet_erl                                                      # noqa: E402
from probe_ct import CTVolume                                         # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_supportB import bootstrap_ci                                # noqa: E402

THICK_T = 80.0            # frozen probe_ct calibration (TOPO-026), no retuning
THICK_OFFSETS = np.arange(-24.0, 25.0)   # radial probe, 1 vx step
THICK_REL_FLOOR = 2.0     # the doubling signature, declared before the run


def segment_thickness(ct, grid, centre, z_quantiles, ckpt):
    """Per-node radial run-length thickness for the support-scope rows.

    Returns arrays (rows, cols, thickness, saturated); checkpointed whole —
    the CT read is the expensive part, evidence replays offline.
    """
    key = {'offsets': [float(t) for t in THICK_OFFSETS],
           'threshold': THICK_T, 'quantiles': list(z_quantiles)}
    if os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') == key:
            return payload['nodes']
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    heights, _ = scrolls.row_heights(grid)
    band_rows = [row for row in range(grid.shape[0])
                 if np.isfinite(heights[row])
                 and any(abs(heights[row] - q) <= detect_v1.SUPPORT_BAND_VX
                         for q in z_quantiles)]
    centre_idx = int(np.where(THICK_OFFSETS == 0.0)[0][0])
    out = {'row': [], 'col': [], 'thick': [], 'sat': []}
    for k, row in enumerate(band_rows):
        mask = valid[row]
        if not mask.any():
            continue
        z = heights[row]
        try:
            cx, cy = centre.at(float(z))
        except ValueError:
            continue
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)
        radial = points[:, :2] - np.array([cx, cy])
        norm = np.linalg.norm(radial, axis=1)
        ok = norm > 1e-6
        if not ok.any():
            continue
        points, cols = points[ok], cols[ok]
        u = radial[ok] / norm[ok][:, None]
        stacked = np.concatenate([
            np.concatenate([points[:, :2] + t * u, points[:, 2:3]], 1)
            for t in THICK_OFFSETS])
        sampled = ct.values(stacked).reshape(len(THICK_OFFSETS), len(points))
        bright = sampled.astype(float) >= THICK_T
        up = np.cumprod(bright[centre_idx:], axis=0).sum(axis=0)
        down = np.cumprod(bright[centre_idx::-1], axis=0).sum(axis=0)
        thick = np.where(bright[centre_idx], up + down - 1, 0)
        sat = bright[centre_idx] & ((up == len(THICK_OFFSETS) - centre_idx)
                                    | (down == centre_idx + 1))
        out['row'].extend([int(row)] * len(cols))
        out['col'].extend(int(c) for c in cols)
        out['thick'].extend(int(t) for t in thick)
        out['sat'].extend(bool(s) for s in sat)
        if (k + 1) % 10 == 0 or k + 1 == len(band_rows):
            print(f'  row {k + 1}/{len(band_rows)} '
                  f'({len(out["col"])} nodes probed)', flush=True)
    payload = {'key': key, 'nodes': out}
    tmp = f'{ckpt}.{os.getpid()}.part'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp, ckpt)
    return out


def thickness_candidates(names, node_sets):
    """The declared thick family: doubling vs the segment's own median."""
    candidates = []
    stats = {}
    for name in names:
        nodes = node_sets[name]
        thick = np.array(nodes['thick'])
        positive = thick[thick > 0]
        med = float(np.median(positive)) if positive.size else 0.0
        sat_share = (float(np.mean(np.array(nodes['sat'])[thick > 0]))
                     if positive.size else None)
        stats[name] = {'nodes': int(thick.size),
                       'papyrus_nodes': int(positive.size),
                       'median_thickness_vx': med,
                       'saturated_share': sat_share}
        if med <= 0:
            print(f'{name}: mute (no papyrus thickness median)', flush=True)
            continue
        evidence, best = {}, {}
        for row, col, t in zip(nodes['row'], nodes['col'], nodes['thick']):
            if t < THICK_REL_FLOOR * med:
                continue
            s = t / med
            cell = (row, col // detect_v1.BLOCK)
            evidence[cell] = evidence.get(cell, 0.0) + s
            if s > best.get(cell, (None, -1.0))[1]:
                best[cell] = (col, s)
        tall = 0
        for cells, mass, top in detect_v1.differenced_clusters(
                evidence, best, {}):
            if cells is None:
                tall += 1
                continue
            candidates.append((name, top[0], best[top][0], 'thick', mass,
                               best[top][1]))
        print(f'{name}: median {med:.1f} vx, {len(candidates)} thick '
              f'candidates so far ({tall} tall clusters cut)', flush=True)
    return candidates, stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--zone-min', type=int, required=True,
                        help='declared zone mass threshold (CORPUS.md insert)')
    parser.add_argument('--report', required=True)
    parser.add_argument('--random-seeds', type=int, default=100)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    ct = CTVolume(args.cache, base=scroll.ct, level=scroll.level)
    names = corpus_map['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(args.report)),
                            'thick_cells_B')
    os.makedirs(ckpt_dir, exist_ok=True)
    node_sets = {}
    for name in names:
        print(f'{name}: probing radial thickness', flush=True)
        node_sets[name] = segment_thickness(
            ct, grids[name], centre, manifest['z_quantiles'],
            os.path.join(ckpt_dir, f'{name}.pkl'))

    candidates, stats = thickness_candidates(names, node_sets)

    report = {'declared_zone_min': args.zone_min,
              'map': os.path.abspath(args.map),
              'declaration': 'CORPUS.md insert 18.08.2026 (TOPO-046, '
                             'twenty-fourth session), committed before this '
                             'ran; T from the frozen probe_ct calibration',
              'channel': {'threshold': THICK_T,
                          'offsets_vx': [float(THICK_OFFSETS[0]),
                                         float(THICK_OFFSETS[-1])],
                          'rel_floor': THICK_REL_FLOOR,
                          'max_cluster_rows': detect_v1.MAX_CLUSTER_ROWS,
                          'per_segment': stats,
                          'candidates': len(candidates)},
              'bootstrap': {'resamples': 2000, 'seed': 20260815},
              'rows': {}}

    rng_pool = [(name, row, col)
                for name, grid in sorted(grids.items())
                for row, col in sheet_erl.windows_of(grid)]
    thresholds = (('primary', args.zone_min), ('half', args.zone_min // 2),
                  ('double', args.zone_min * 2))

    entry = {}
    for label, threshold in thresholds:
        zones = zone_records(corpus_map['zones'], threshold)
        n = len(zones)
        randoms = []
        for seed in range(args.random_seeds):
            rng = np.random.default_rng(seed)
            picks = rng.choice(len(rng_pool), size=min(n, len(rng_pool)),
                               replace=False)
            random_ranking = [(*rng_pool[i], 0.0) for i in picks]
            randoms.append(evaluate(random_ranking, zones, n)[0])
        baseline = {
            key: {'median': float(np.median([r[key] for r in randoms])),
                  'iqr': [float(np.percentile([r[key] for r in randoms], 25)),
                          float(np.percentile([r[key] for r in randoms], 75))]}
            for key in ('ap', 'recall_at_n')}
        ranking = detect_v1.merge_channels(candidates, top=4 * n)
        metrics, outcomes = evaluate(ranking, zones, n)
        variant = {'zone_min': threshold, 'n_zones': n, 'metrics': metrics,
                   'baseline_random': baseline,
                   'ci': bootstrap_ci(outcomes, zones, n)}
        if label == 'primary':
            iqr_hi = baseline['ap']['iqr'][1]
            variant['signal'] = bool(variant['ci']['ap_ci95'][0] > iqr_hi)
        entry[label] = variant
    report['rows']['thick'] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    p = entry['primary']
    verdict = 'SIGNAL' if p['signal'] else 'chance'
    print(f"thick (mass >= {args.zone_min}, {p['n_zones']} zones): "
          f"AP {p['metrics']['ap']:.4f} "
          f"[{p['ci']['ap_ci95'][0]:.4f}-{p['ci']['ap_ci95'][1]:.4f}] "
          f"vs random IQR {p['baseline_random']['ap']['iqr'][0]:.4f}-"
          f"{p['baseline_random']['ap']['iqr'][1]:.4f} -> {verdict}")
    for label in ('half', 'double'):
        e = entry[label]
        print(f"  sensitivity {label:6s} (>= {e['zone_min']}, "
              f"{e['n_zones']} zones): AP {e['metrics']['ap']:.4f} "
              f"[{e['ci']['ap_ci95'][0]:.4f}-{e['ci']['ap_ci95'][1]:.4f}]")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
