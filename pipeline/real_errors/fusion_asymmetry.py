"""TOPO-050: the fusion wall's mechanism — noise dilution or the percentile
form itself (U-023).

TOPO-047 measured that the wall is not universal: the frozen percentile
merge drowns prox+support on B to 0.0012 (support solo 0.0330) yet holds
support+thick at solo level (0.0346). Percentile rank is scale-invariant
within a channel, so mass scale cannot be the mechanism; the live candidates
are (a) dilution of the ranking prefix by a NOISE channel (prox on B is pure
chance, AP 0.0002, and its candidates occupy a share of every prefix) or
(b) the percentile form as such. The two are separable cheaply: thin the
prox pool and watch the frozen merge recover — dilution predicts monotone
recovery toward support solo; a form defect predicts no recovery. Entirely
offline from the B checkpoints, no CT or recto reads.

**Declared before any number:**

- **Gate (the run dies without writing a report):** the frozen
  ``merge_channels`` on the FULL prox+support pools reproduces the published
  prox_support primary AP (`eval_supportB.json`) to 1e-12, and the support
  solo replay reproduces 0.0330 likewise.
- Experiment: the prox pool is thinned uniformly at random to fractions
  **1.0 / 0.5 / 0.25 / 0.1 / 0.05**, 20 seeds per fraction
  (``default_rng(20260819 + seed_index)``), frozen percentile merge with
  the full support pool, AP against the primary zones (mass >= 20,
  178 zones), top 4N as in every B run. Median and IQR over seeds per
  fraction.
- **Reading rule:** mechanism = dilution iff the median AP rises
  monotonically with thinning and reaches >= 0.5 x support solo (0.0165)
  at fraction 0.05; mechanism = the percentile form iff the median AP at
  fraction 0.05 stays <= 0.1 x support solo (0.0033); anything between is
  recorded as a mixed mechanism, with numbers.

Usage (from pipeline/real_errors/):

    python fusion_asymmetry.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --support-report ../../output/topo/real_paris4/eval_supportB.json \
        --report ../../output/topo/real_paris4/fusion_asymmetry.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402
from eval_real import prox_candidates, evaluate, zone_records         # noqa: E402
from eval_fusionB import support_from_checkpoints                     # noqa: E402

FRACTIONS = (1.0, 0.5, 0.25, 0.1, 0.05)
SEEDS = 20
RNG_BASE = 20260819


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--support-report', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.support_report, encoding='utf-8') as f:
        published = json.load(f)
    zone_min = published['declared_zone_min']
    published_fused = published['rows']['prox_support']['primary']['metrics']['ap']
    published_support = published['rows']['support']['primary']['metrics']['ap']

    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    names = corpus_map['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}

    real_dir = os.path.dirname(os.path.abspath(args.report))
    support = support_from_checkpoints(
        names, os.path.join(real_dir, 'cells_supportB'))
    prox = prox_candidates(names, grids, centre)
    zones = zone_records(corpus_map['zones'], zone_min)
    n = len(zones)

    # ------------------------------------------------ gates
    m_full, _ = evaluate(detect_v1.merge_channels(prox + support, top=4 * n),
                         zones, n)
    if abs(m_full['ap'] - published_fused) > 1e-12:
        raise SystemExit(f"frozen-merge gate FAILED: {m_full['ap']} vs "
                         f"{published_fused} — nothing written")
    m_solo, _ = evaluate(detect_v1.merge_channels(support, top=4 * n),
                         zones, n)
    if abs(m_solo['ap'] - published_support) > 1e-12:
        raise SystemExit(f"support-solo gate FAILED: {m_solo['ap']} vs "
                         f"{published_support} — nothing written")
    print(f'gates OK: frozen merge {m_full["ap"]:.4f}, '
          f'support solo {m_solo["ap"]:.4f}')

    # ------------------------------------------------ thinning experiment
    results = {}
    for fraction in FRACTIONS:
        aps = []
        if fraction == 1.0:
            aps = [m_full['ap']] * 1
        else:
            k = max(1, int(round(len(prox) * fraction)))
            for s in range(SEEDS):
                rng = np.random.default_rng(RNG_BASE + s)
                picks = rng.choice(len(prox), size=k, replace=False)
                thinned = [prox[i] for i in sorted(picks)]
                m, _ = evaluate(detect_v1.merge_channels(
                    thinned + support, top=4 * n), zones, n)
                aps.append(m['ap'])
        results[str(fraction)] = {
            'prox_pool': max(1, int(round(len(prox) * fraction))),
            'ap_median': float(np.median(aps)),
            'ap_iqr': [float(np.percentile(aps, 25)),
                       float(np.percentile(aps, 75))],
            'n_seeds': len(aps)}
        print(f'fraction {fraction}: prox {results[str(fraction)]["prox_pool"]} '
              f'-> AP median {results[str(fraction)]["ap_median"]:.4f} '
              f'IQR {results[str(fraction)]["ap_iqr"]}')

    medians = [results[str(f)]['ap_median'] for f in FRACTIONS]
    monotone = all(medians[i] <= medians[i + 1] + 1e-9
                   for i in range(len(medians) - 1))
    tail = results[str(0.05)]['ap_median']
    if monotone and tail >= 0.5 * published_support:
        verdict = 'dilution'
    elif tail <= 0.1 * published_support:
        verdict = 'percentile_form'
    else:
        verdict = 'mixed'

    report = {
        'declaration': 'fusion_asymmetry.py header, committed before the '
                       'run (TOPO-050, twenty-seventh session)',
        'pools': {'prox': len(prox), 'support': len(support)},
        'published': {'frozen_merge': published_fused,
                      'support_solo': published_support},
        'fractions': {str(f): results[str(f)] for f in FRACTIONS},
        'reading_rule': 'dilution iff monotone recovery and AP(0.05) >= '
                        '0.5x support solo; percentile_form iff AP(0.05) <= '
                        '0.1x support solo; else mixed',
        'monotone_recovery': bool(monotone),
        'verdict': verdict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f'verdict (pre-declared rule): {verdict.upper()} '
          f'(monotone {monotone}, tail {tail:.4f} vs support '
          f'{published_support:.4f})')
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
