"""TOPO-039: the prediction-free vjump channel vs the corpus-B zones (H3.2).

Support (TOPO-028) is the one independently validated channel on real
labelling, but it reads the recto prediction — every real signal of the line
hangs on that single input. The vjump channel is computed from the mesh
geometry alone: a signal here would be a *second* independent channel on the
human-verified corpus B, and the first prediction-free one.

The declaration (CORPUS.md insert, 17.08.2026, twenty-first session,
committed before this ran) fixes one row — **vjump** — in the v6-declared
construction (detect_v6.py's protocol commit precedes the insert): node
evidence where the inter-row radial jump exceeds VJUMP_REL_FLOOR = 3.0
times the segment's own median, evidence value = the ratio, the standard
(row, block) cells, `differenced_clusters` with an EMPTY atlas (substrate
subtraction is undefined on a real substrate — same as prox/support in
TOPO-028), only the MAX_CLUSTER_ROWS = 20 cut (the v6 value). Ranking by
`merge_channels` (one family — monotone in mass). Zones, thresholds
(20/10/40), the 100-seed random baseline and the by-zone bootstrap (2000
resamples, seed 20260815) are TOPO-028's, byte for byte.

Reading rule (TOPO-028's, verbatim): the row signals iff the lower edge of
its 95 % bootstrap AP interval sits strictly above the upper edge of the
random baseline's IQR at the primary threshold; a primary signal that dies
at both sensitivity thresholds reads as unstable, not as a signal.

Usage:

    python eval_vjumpB.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --zone-min 20 --report ../../output/topo/real_paris4/eval_vjumpB.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402
import sheet_erl                                                      # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_supportB import bootstrap_ci                                # noqa: E402

VJUMP_REL_FLOOR = 3.0     # detect_v6.py §2, declared before any B number
MAX_CLUSTER_ROWS_V6 = 20  # detect_v6.py §1


def vjump_candidates(names, grids, centre):
    """The v6 vjump family, mesh-only: empty atlas, MAX_CLUSTER_ROWS cut."""
    detect_v1.MAX_CLUSTER_ROWS = MAX_CLUSTER_ROWS_V6
    candidates = []
    for name in names:
        r = detect_v1.radial_map(grids[name], centre)
        vj = np.abs(np.diff(r, axis=0))
        med = float(np.nanmedian(vj)) if np.isfinite(vj).any() else 0.0
        if med <= 0:
            print(f'{name}: mute (no vjump median)', flush=True)
            continue
        ratio = vj / med
        rows, cols = np.where(np.isfinite(ratio)
                              & (ratio > VJUMP_REL_FLOOR))
        evidence, best = {}, {}
        for row, col in zip(rows, cols):
            s = float(ratio[row, col])
            key = (int(row), int(col) // detect_v1.BLOCK)
            evidence[key] = evidence.get(key, 0.0) + s
            if s > best.get(key, (None, -1.0))[1]:
                best[key] = (int(col), s)
        tall = 0
        for cells, mass, top in detect_v1.differenced_clusters(
                evidence, best, {}):
            if cells is None:
                tall += 1
                continue
            candidates.append((name, top[0], best[top][0], 'vjump', mass,
                               best[top][1]))
        print(f'{name}: {len(candidates)} vjump candidates so far '
              f'({tall} tall clusters cut)', flush=True)
    return candidates


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
    names = corpus_map['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}

    candidates = vjump_candidates(names, grids, centre)

    report = {'declared_zone_min': args.zone_min,
              'map': os.path.abspath(args.map),
              'declaration': 'CORPUS.md insert 17.08.2026 (TOPO-039, '
                             'twenty-first session), committed before this '
                             'ran; channel constants from detect_v6.py',
              'channel': {'vjump_rel_floor': VJUMP_REL_FLOOR,
                          'max_cluster_rows': MAX_CLUSTER_ROWS_V6,
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
    report['rows']['vjump'] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    p = entry['primary']
    verdict = 'SIGNAL' if p['signal'] else 'chance'
    print(f"vjump (mass >= {args.zone_min}, {p['n_zones']} zones): "
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
