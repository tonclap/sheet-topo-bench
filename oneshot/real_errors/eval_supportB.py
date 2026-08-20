"""TOPO-028: the frozen support channel vs the corpus-B zones (U-014).

Corpus B is the one real labelling with no circularity to the models: its
zones are geometric distances to the human-verified GP-banner, no prediction
is read. The support channel samples recto — circular on corpora A/A2, clean
here. The declaration (CORPUS.md insert, 17.08.2026, committed before this
ran) fixes three rows — prox / support / prox+support — all byte-for-byte the
frozen detector: `support_cells` with SUPPORT_BAND_VX 160 and the scroll's
prediction threshold, `differenced_clusters` with an empty atlas (mesh-only,
same as the prox run: the substrate subtraction is undefined on a real
substrate, only the MAX_CLUSTER_ROWS cut applies), `merge_channels`
percentile-rank fusion. Zones, thresholds (20/10/40), the 100-seed random
baseline and the by-zone bootstrap (2000 resamples, seed 20260815) are the
prox run's.

Reading rule (declared before any number): a row signals iff the lower edge
of its 95 % bootstrap AP interval sits strictly above the upper edge of the
random baseline's IQR at the primary threshold; a primary signal that dies at
both sensitivity thresholds reads as unstable, not as a signal.

Support sampling is the slow part (recto chunks over four segments); each
segment checkpoints its evidence cells, so a killed run resumes rather than
resamples.

Usage:

    python eval_supportB.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --zone-min 20 --report ../../output/topo/real_paris4/eval_supportB.json
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
from eval_real import prox_candidates, evaluate, zone_records         # noqa: E402

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815


def support_candidates(names, grids, scroll, cache, z_quantiles, ckpt_dir):
    """The frozen support channel, mesh-only, with per-segment checkpoints.

    No atlas: same reasoning as the prox real run — the run object *is* the
    substrate — so `differenced_clusters` gets an empty atlas and only the
    MAX_CLUSTER_ROWS cut applies.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    prediction = None
    candidates = []
    for name in names:
        ckpt = os.path.join(ckpt_dir, f'{name}.pkl')
        payload = None
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('key') != {'segment': name,
                                      'quantiles': list(z_quantiles)}:
                payload = None
        if payload is None:
            if prediction is None:
                prediction = scrolls.open_prediction(scroll, cache,
                                                     max_chunks=256)
            evidence, best = detect_v1.support_cells(
                grids[name], prediction, z_quantiles, scroll.threshold)
            payload = {'key': {'segment': name,
                               'quantiles': list(z_quantiles)},
                       'evidence': evidence, 'best': best}
            with open(ckpt, 'wb') as f:
                pickle.dump(payload, f)
        tall = 0
        for cells, mass, top in detect_v1.differenced_clusters(
                payload['evidence'], payload['best'], {}):
            if cells is None:
                tall += 1
                continue
            candidates.append((name, top[0], payload['best'][top][0],
                               'support', mass, 1.0))
        print(f'{name}: {len(candidates)} support candidates so far '
              f'({tall} tall clusters cut)', flush=True)
    return candidates


def bootstrap_ci(outcomes, zones, n):
    """By-zone bootstrap of AP/recall, the real_ci.py decomposition inline.

    Validated against the direct AP before any interval is returned.
    """
    found_rank = {}
    for rank, (hit, *_rest) in enumerate(outcomes, 1):
        if hit is not None and hit not in found_rank:
            found_rank[hit] = rank
    ranks = sorted(found_rank.values())
    tp_at = {r: i for i, r in enumerate(ranks, 1)}
    contribution = np.array([
        tp_at[found_rank[k]] / found_rank[k] if k in found_rank else 0.0
        for k in range(len(zones))])
    recall_terms = np.array([
        1.0 if k in found_rank and found_rank[k] <= n else 0.0
        for k in range(len(zones))])
    ap_direct = sheet_erl.average_precision(outcomes, n)
    if abs(float(contribution.mean()) - ap_direct) > 1e-9:
        raise SystemExit(f'AP reconstruction mismatch: '
                         f'{contribution.mean()} vs {ap_direct} — no CI')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(zones), size=(BOOTSTRAP_RESAMPLES,
                                                len(zones)))
    aps = contribution[samples].mean(axis=1)
    recalls = recall_terms[samples].mean(axis=1)
    return {'ap_ci95': [float(np.percentile(aps, 2.5)),
                        float(np.percentile(aps, 97.5))],
            'recall_ci95': [float(np.percentile(recalls, 2.5)),
                            float(np.percentile(recalls, 97.5))]}


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

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(args.report)),
                            'cells_supportB')
    support = support_candidates(names, grids, scroll, args.cache,
                                 manifest['z_quantiles'], ckpt_dir)
    prox = prox_candidates(names, grids, centre)
    rows = {'prox': prox, 'support': support, 'prox_support': prox + support}

    report = {'declared_zone_min': args.zone_min,
              'map': os.path.abspath(args.map),
              'declaration': 'CORPUS.md insert 17.08.2026 (TOPO-028), '
                             'committed before this ran',
              'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES,
                            'seed': BOOTSTRAP_SEED},
              'rows': {}}

    rng_pool = [(name, row, col)
                for name, grid in sorted(grids.items())
                for row, col in sheet_erl.windows_of(grid)]
    thresholds = (('primary', args.zone_min), ('half', args.zone_min // 2),
                  ('double', args.zone_min * 2))

    baselines = {}
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
        baselines[label] = {
            key: {'median': float(np.median([r[key] for r in randoms])),
                  'iqr': [float(np.percentile([r[key] for r in randoms], 25)),
                          float(np.percentile([r[key] for r in randoms], 75))]}
            for key in ('ap', 'recall_at_n')}

    for row_name, candidates in rows.items():
        entry = {}
        for label, threshold in thresholds:
            zones = zone_records(corpus_map['zones'], threshold)
            n = len(zones)
            ranking = detect_v1.merge_channels(candidates, top=4 * n)
            metrics, outcomes = evaluate(ranking, zones, n)
            variant = {'zone_min': threshold, 'n_zones': n,
                       'metrics': metrics,
                       'baseline_random': baselines[label],
                       'ci': bootstrap_ci(outcomes, zones, n)}
            if label == 'primary':
                iqr_hi = baselines[label]['ap']['iqr'][1]
                variant['signal'] = bool(variant['ci']['ap_ci95'][0] > iqr_hi)
            entry[label] = variant
        report['rows'][row_name] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    # The summary comes out of the same numbers as the table — the
    # stale-summary lesson, applied by construction.
    for row_name in ('prox', 'support', 'prox_support'):
        p = report['rows'][row_name]['primary']
        verdict = 'SIGNAL' if p['signal'] else 'chance'
        print(f"{row_name:12s} (mass >= {args.zone_min}, {p['n_zones']} zones): "
              f"AP {p['metrics']['ap']:.4f} "
              f"[{p['ci']['ap_ci95'][0]:.4f}-{p['ci']['ap_ci95'][1]:.4f}] "
              f"vs random IQR {p['baseline_random']['ap']['iqr'][0]:.4f}-"
              f"{p['baseline_random']['ap']['iqr'][1]:.4f} -> {verdict}")
        for label in ('half', 'double'):
            e = report['rows'][row_name][label]
            print(f"  sensitivity {label:6s} (>= {e['zone_min']}, "
                  f"{e['n_zones']} zones): AP {e['metrics']['ap']:.4f} "
                  f"[{e['ci']['ap_ci95'][0]:.4f}-{e['ci']['ap_ci95'][1]:.4f}]")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
