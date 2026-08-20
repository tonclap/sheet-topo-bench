"""TOPO-051: trace phase vs the CT layering on the corpus-B zones (U-018).

A corpus-B zone is a production trace diverging from the human-verified
banner: a trace that left its sheet has, by construction, lost phase against
the CT layering. The node evidence — offset 0 farther than half the local
half-period from every bright-run centre — is probe_phase's frozen
construction (TOPO-048), declared there before any corpus-B number existed
and not retuned here (PROTOCOL §5). The channel is prediction-free (reads
only the papyrus/air CT boundary and the mesh): a signal would make it the
THIRD independent real channel next to support (recto) and thick
(CT run length through the trace).

The declaration (CORPUS.md insert, 18.08.2026, twenty-seventh session,
committed before this ran) fixes one row — **phase**:

- row scope, probe geometry, zones, thresholds (20/10/40), the 100-seed
  random baseline and the by-zone bootstrap (2000 resamples, seed 20260815)
  are the thick run's, byte for byte;
- offsets -36..+36 vx step 1; runs >= T = 80.0 (frozen probe_ct
  calibration) of length >= 2 vx; node period = median spacing of run
  centres, mute if < 2 runs (share published);
- evidence iff distance from offset 0 to the nearest run centre >= 0.5 of
  the half-period (frozen by the TOPO-048 declaration), value = the ratio;
  cells (row, col // BLOCK), `differenced_clusters` with an EMPTY atlas,
  MAX_CLUSTER_ROWS = 14, `merge_channels` ranking.

Reading rule (TOPO-028's, verbatim): the row signals iff the lower edge of
its 95 % bootstrap AP interval sits strictly above the upper edge of the
random baseline's IQR at the primary threshold; a primary signal that dies
at both sensitivity thresholds reads as unstable, not as a signal.

Regression gates before the phase run (the run dies without a report): the
support and thick replays reproduce the published primary APs (0.0330 /
0.0306) to 1e-12.

Usage (from oneshot/real_errors/):

    python eval_phaseB.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --support-report ../../output/topo/real_paris4/eval_supportB.json \
        --thick-report ../../output/topo/real_paris4/eval_thicknessB.json \
        --zone-min 20 --report ../../output/topo/real_paris4/eval_phaseB.json
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
from probe_phase import node_stats, PHASE_DIAG_FLOOR                  # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_supportB import bootstrap_ci                                # noqa: E402
from eval_fusionB import support_from_checkpoints                     # noqa: E402
from eval_thicknessB import thickness_candidates                      # noqa: E402

PHASE_OFFSETS = np.arange(-36.0, 37.0)   # radial probe, 1 vx step


def segment_phase(ct, grid, centre, z_quantiles, ckpt):
    """Per-node phase stats for the support-scope rows, checkpointed whole."""
    key = {'offsets': [float(t) for t in PHASE_OFFSETS],
           'floor': PHASE_DIAG_FLOOR, 'quantiles': list(z_quantiles)}
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
    out = {'row': [], 'col': [], 'phase': [], 'runs': []}
    for k, row in enumerate(band_rows):
        mask = valid[row]
        if not mask.any():
            continue
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)
        z_med = np.median(points[:, 2])
        try:
            cx, cy = centre.at(float(z_med))
        except ValueError:
            continue
        radial = points[:, :2] - np.array([cx, cy])
        norm = np.linalg.norm(radial, axis=1)
        ok = norm > 1e-6
        if not ok.any():
            continue
        points, cols = points[ok], cols[ok]
        u = radial[ok] / norm[ok][:, None]
        stacked = np.concatenate([
            np.concatenate([points[:, :2] + t * u, points[:, 2:3]], 1)
            for t in PHASE_OFFSETS])
        sampled = ct.values(stacked).reshape(len(PHASE_OFFSETS), len(points))
        for j, col in enumerate(cols):
            _thick, _centred, phase, n_runs = node_stats(
                sampled[:, j].astype(float))
            out['row'].append(int(row))
            out['col'].append(int(col))
            out['phase'].append(phase)
            out['runs'].append(int(n_runs))
        if (k + 1) % 10 == 0 or k + 1 == len(band_rows):
            print(f'  row {k + 1}/{len(band_rows)} '
                  f'({len(out["col"])} nodes probed)', flush=True)
    payload = {'key': key, 'nodes': out}
    tmp = f'{ckpt}.{os.getpid()}.part'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp, ckpt)
    return out


def phase_candidates(names, node_sets):
    """The declared phase family: mid-gap nodes vs the layering."""
    candidates = []
    stats = {}
    for name in names:
        nodes = node_sets[name]
        phases = nodes['phase']
        n_nodes = len(nodes['col'])
        n_mute = sum(1 for p in phases if p is None)
        evidence, best = {}, {}
        n_evidence = 0
        for row, col, phase in zip(nodes['row'], nodes['col'], phases):
            if phase is None or phase < PHASE_DIAG_FLOOR:
                continue
            n_evidence += 1
            s = phase
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
            candidates.append((name, top[0], best[top][0], 'phase', mass,
                               best[top][1]))
        stats[name] = {'nodes': n_nodes, 'evidence_nodes': n_evidence,
                       'mute_share': round(n_mute / n_nodes, 4)
                       if n_nodes else None}
        print(f'{name}: {len(candidates)} phase candidates so far '
              f'({tall} tall clusters cut, mute {stats[name]["mute_share"]})',
              flush=True)
    return candidates, stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--support-report', required=True)
    parser.add_argument('--thick-report', required=True)
    parser.add_argument('--zone-min', type=int, required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--random-seeds', type=int, default=100)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.support_report, encoding='utf-8') as f:
        published_support = json.load(f)
    with open(args.thick_report, encoding='utf-8') as f:
        published_thick = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    ct = CTVolume(args.cache, base=scroll.ct, level=scroll.level)
    names = corpus_map['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}

    real_dir = os.path.dirname(os.path.abspath(args.report))
    zones_primary = zone_records(corpus_map['zones'], args.zone_min)
    n_primary = len(zones_primary)

    # ------------------------------------------------ regression gates
    support = support_from_checkpoints(
        names, os.path.join(real_dir, 'cells_supportB'))
    node_sets_thick = {}
    for name in names:
        with open(os.path.join(real_dir, 'thick_cells_B',
                               f'{name}.pkl'), 'rb') as f:
            node_sets_thick[name] = pickle.load(f)['nodes']
    thick, _ = thickness_candidates(names, node_sets_thick)
    for row_name, cands, published in (
            ('support', support,
             published_support['rows']['support']['primary']['metrics']['ap']),
            ('thick', thick,
             published_thick['rows']['thick']['primary']['metrics']['ap'])):
        metrics, _o = evaluate(detect_v1.merge_channels(
            cands, top=4 * n_primary), zones_primary, n_primary)
        if abs(metrics['ap'] - published) > 1e-12:
            raise SystemExit(f'{row_name} replay gate FAILED: '
                             f"{metrics['ap']} vs {published}")
        print(f"{row_name} replay gate OK: AP {metrics['ap']:.4f}")

    # ------------------------------------------------ the phase row
    ckpt_dir = os.path.join(real_dir, 'phase_cells_B')
    os.makedirs(ckpt_dir, exist_ok=True)
    node_sets = {}
    for name in names:
        print(f'{name}: probing radial phase', flush=True)
        node_sets[name] = segment_phase(
            ct, grids[name], centre, manifest['z_quantiles'],
            os.path.join(ckpt_dir, f'{name}.pkl'))
    candidates, stats = phase_candidates(names, node_sets)

    report = {'declared_zone_min': args.zone_min,
              'map': os.path.abspath(args.map),
              'declaration': 'CORPUS.md insert 18.08.2026 (TOPO-051, '
                             'twenty-seventh session), committed before '
                             'this ran; construction frozen by the '
                             'TOPO-048 declaration',
              'channel': {'offsets_vx': [float(PHASE_OFFSETS[0]),
                                         float(PHASE_OFFSETS[-1])],
                          'floor': PHASE_DIAG_FLOOR,
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
    report['rows']['phase'] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    p = entry['primary']
    verdict = 'SIGNAL' if p['signal'] else 'chance'
    print(f"phase (mass >= {args.zone_min}, {p['n_zones']} zones): "
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
