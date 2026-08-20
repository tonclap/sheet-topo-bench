"""TOPO-054: frozen fusion of the three signal channels vs the corpus-B zones.

TOPO-047 measured that the frozen percentile fusion of support+thick HOLDS
the solo level (0.0346 vs 0.0330/0.0330); TOPO-050 named the fusion-wall
mechanism — dilution by an asymmetric noise pool. TOPO-053 added a third
signal channel, defect (AP 0.0134, pool 518). The three signal pools are
comparable (793/930/518), so the TOPO-050 mechanism predicts the three-way
fusion holds the solo level — this file turns the prediction into a
measurement.

Declaration: CORPUS.md insert of 19.08.2026 (thirtieth session), committed
BEFORE this ran. One rule row — **fusion3** (frozen `merge_channels` of the
united three-channel pool, top 4N). Alongside, outside the rule: the pairs
support+defect and thick+defect, same form.

The run is entirely offline: all three channels replay from their
per-segment checkpoints (`cells_supportB/*.pkl`, `thick_cells_B/*.pkl`,
`defect_cells_B/*.pkl`), no CT or recto reads. Regression gates (the run
dies without writing a report): the three solo replays must reproduce the
published primary APs (`eval_supportB.json` 0.0330, `eval_thicknessB.json`
0.0306, `eval_defectB.json` 0.0134) to 1e-12, and the recomputed defect
FLOOR must equal the published `stage1_price.floor` (gated inside
`collect_defectB_zones.defect_candidates_from_checkpoints`).

Reading rule — TOPO-028 verbatim (signal vs the random baseline). Solo-level
rule (declared before any number): best solo = support (highest published
primary AP). fusion3 **holds** the best-solo level iff the upper edge of the
95 % paired by-zone bootstrap interval of (fusion3 - support) AP is >= 0;
**drowns** iff that upper edge is strictly < 0; **above** (a subcase of
holds, reported separately) iff the lower edge is strictly > 0. Deltas vs
thick and defect are published alongside, outside the rule.

Usage (from oneshot/real_errors/):

    python eval_fusion3B.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids \
        --support-report ../../output/topo/real_paris4/eval_supportB.json \
        --thick-report ../../output/topo/real_paris4/eval_thicknessB.json \
        --defect-report ../../output/topo/real_paris4/eval_defectB.json \
        --report ../../output/topo/real_paris4/eval_fusion3B.json
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
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_supportB import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,       # noqa: E402
                           bootstrap_ci)
from eval_thicknessB import thickness_candidates                      # noqa: E402
from eval_fusionB import (support_from_checkpoints,                   # noqa: E402
                          zone_contributions, paired_delta_ci)
from collect_defectB_zones import defect_candidates_from_checkpoints  # noqa: E402

SOLOS = ('support', 'thick', 'defect')
BEST_SOLO = 'support'   # highest published primary AP — declared, not read


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--support-report', required=True)
    parser.add_argument('--thick-report', required=True)
    parser.add_argument('--defect-report', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--random-seeds', type=int, default=100)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(os.path.join(args.corpus, 'manifest.json'),
              encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.support_report, encoding='utf-8') as f:
        published_support = json.load(f)
    with open(args.thick_report, encoding='utf-8') as f:
        published_thick = json.load(f)
    with open(args.defect_report, encoding='utf-8') as f:
        published_defect = json.load(f)
    zone_min = published_support['declared_zone_min']
    if (published_thick['declared_zone_min'] != zone_min
            or published_defect['declared_zone_min'] != zone_min):
        raise SystemExit('published reports disagree on the zone threshold')

    scroll = scrolls.SCROLLS[manifest['scroll']]
    names = corpus_map['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}

    real_dir = os.path.dirname(os.path.abspath(args.report))
    support = support_from_checkpoints(
        names, os.path.join(real_dir, 'cells_supportB'))
    node_sets = {}
    for name in names:
        with open(os.path.join(real_dir, 'thick_cells_B',
                               f'{name}.pkl'), 'rb') as f:
            node_sets[name] = pickle.load(f)['nodes']
    thick, _stats = thickness_candidates(names, node_sets)
    defect, floor = defect_candidates_from_checkpoints(
        names, os.path.join(real_dir, 'defect_cells_B'),
        corpus_map['zones'], published_defect['stage1_price']['floor'])
    pools = {'support': support, 'thick': thick, 'defect': defect}

    # ---------------------------------------------- regression gates
    zones_primary = zone_records(corpus_map['zones'], zone_min)
    n_primary = len(zones_primary)
    published_ap = {
        'support': published_support['rows']['support']['primary']['metrics']['ap'],
        'thick': published_thick['rows']['thick']['primary']['metrics']['ap'],
        'defect': published_defect['rows']['defect']['primary']['metrics']['ap']}
    solo_outcomes = {}
    for solo in SOLOS:
        ranking = detect_v1.merge_channels(pools[solo], top=4 * n_primary)
        metrics, outcomes = evaluate(ranking, zones_primary, n_primary)
        if abs(metrics['ap'] - published_ap[solo]) > 1e-12:
            raise SystemExit(
                f"{solo} replay regression FAILED: {metrics['ap']} vs "
                f"published {published_ap[solo]} — nothing written")
        solo_outcomes[solo] = outcomes
        print(f"{solo} replay gate OK: AP {metrics['ap']:.4f} == published")

    # ------------------------------- the rule row and the two side pairs
    rankings = {
        'fusion3': detect_v1.merge_channels(support + thick + defect,
                                            top=4 * n_primary),
        'support+defect': detect_v1.merge_channels(support + defect,
                                                   top=4 * n_primary),
        'thick+defect': detect_v1.merge_channels(thick + defect,
                                                 top=4 * n_primary)}

    report = {'declared_zone_min': zone_min,
              'map': os.path.abspath(args.map),
              'declaration': 'CORPUS.md insert 19.08.2026 (TOPO-054, '
                             'thirtieth session), committed before this '
                             'ran; fully offline from the published '
                             'checkpoints',
              'gates': {f'{solo}_ap': published_ap[solo] for solo in SOLOS},
              'defect_floor': floor,
              'pool': {solo: len(pools[solo]) for solo in SOLOS},
              'best_solo': BEST_SOLO,
              'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES,
                            'seed': BOOTSTRAP_SEED},
              'rows': {}}

    rng_pool = [(name, row, col)
                for name, grid in sorted(grids.items())
                for row, col in sheet_erl.windows_of(grid)]
    thresholds = (('primary', zone_min), ('half', zone_min // 2),
                  ('double', zone_min * 2))
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

    solo_contrib = {solo: zone_contributions(outcomes, zones_primary,
                                             n_primary)
                    for solo, outcomes in solo_outcomes.items()}

    for row_name, ranking in rankings.items():
        entry = {}
        for label, threshold in thresholds:
            zones = zone_records(corpus_map['zones'], threshold)
            n = len(zones)
            metrics, outcomes = evaluate(ranking, zones, n)
            variant = {'zone_min': threshold, 'n_zones': n,
                       'metrics': metrics,
                       'baseline_random': baselines[label],
                       'ci': bootstrap_ci(outcomes, zones, n)}
            if label == 'primary':
                iqr_hi = baselines[label]['ap']['iqr'][1]
                variant['signal'] = bool(variant['ci']['ap_ci95'][0] > iqr_hi)
                contrib = zone_contributions(outcomes, zones, n)
                variant['delta_vs_solo'] = {
                    solo: {'delta_ap': float(contrib.mean()
                                             - solo_contrib[solo].mean()),
                           'delta_ci95': paired_delta_ci(contrib,
                                                         solo_contrib[solo])}
                    for solo in SOLOS}
                d_best = variant['delta_vs_solo'][BEST_SOLO]['delta_ci95']
                variant['holds_best_solo'] = bool(d_best[1] >= 0)
                variant['above_best_solo'] = bool(d_best[0] > 0)
            entry[label] = variant
        report['rows'][row_name] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    # The summary comes out of the same numbers as the table — the
    # stale-summary lesson, applied by construction.
    print(f"solo reference: support AP {published_ap['support']:.4f}, "
          f"thick AP {published_ap['thick']:.4f}, "
          f"defect AP {published_ap['defect']:.4f}")
    for row_name in rankings:
        p = report['rows'][row_name]['primary']
        verdict = 'SIGNAL' if p['signal'] else 'chance'
        rule = ''
        if row_name == 'fusion3':
            rule = ('; ABOVE best solo' if p['above_best_solo']
                    else ('; HOLDS best solo' if p['holds_best_solo']
                          else '; DROWNS below best solo'))
        print(f"{row_name:15s} (mass >= {zone_min}, {p['n_zones']} zones): "
              f"AP {p['metrics']['ap']:.4f} "
              f"[{p['ci']['ap_ci95'][0]:.4f}-{p['ci']['ap_ci95'][1]:.4f}] "
              f"vs random IQR {p['baseline_random']['ap']['iqr'][0]:.4f}-"
              f"{p['baseline_random']['ap']['iqr'][1]:.4f} -> {verdict}{rule}")
        for solo in SOLOS:
            d = p['delta_vs_solo'][solo]
            print(f"    vs {solo:8s} dAP {d['delta_ap']:+.4f} "
                  f"[{d['delta_ci95'][0]:+.4f}..{d['delta_ci95'][1]:+.4f}]")
        for label in ('half', 'double'):
            e = report['rows'][row_name][label]
            print(f"    sensitivity {label:6s} (>= {e['zone_min']}, "
                  f"{e['n_zones']} zones): AP {e['metrics']['ap']:.4f} "
                  f"[{e['ci']['ap_ci95'][0]:.4f}-{e['ci']['ap_ci95'][1]:.4f}]")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
