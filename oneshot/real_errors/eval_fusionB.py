"""TOPO-047: fusion of the two validated real channels vs the corpus-B zones.

After TOPO-046 the line holds two validated channels with disjoint inputs:
support (reads recto, AP 0.0330) and thick (reads only the papyrus/air CT
boundary, AP 0.0306), partially complementary on credited zones (18/51).
The frozen percentile fusion drowns channels (TOPO-016; on B: prox+support
0.0012) while a dev-trained logit repairs it (V7, TOPO-037) — but thick has
no dev channel to train on (the injector does not thicken material), so the
declared fusion forms are label-free (the V2 precedent) plus a
cross-fitted-on-B logit:

- **frozen**  — the frozen `merge_channels` percentile fusion (the v1 form);
- **rawmass** — the union pool sorted by raw mass, no normalisation;
- **zscore**  — per-family z-score of log1p(mass), then one sort;
- **loso**    — `detect_v5.fit_logit` construction on [family indicators,
  log1p(mass) x indicator, factor]; labels are PROTOCOL §3 hits against the
  primary-threshold zones; folds are leave-one-segment-out over the four B
  segments, so no zone is scored by weights that saw it. A training fold
  with no positive labels declares the row degenerate (outcome (c)).

The run is entirely offline: both channels replay byte-for-byte from their
per-segment checkpoints (`cells_supportB/*.pkl`, `thick_cells_B/*.pkl`), no
CT or recto reads. No frozen-detector parameter changes (PROTOCOL §5) —
the run answers a question, it does not tune v1.

Regression gates (the run dies without writing a report): the support-solo
and thick-solo replays must reproduce the published primary APs
(`eval_supportB.json` 0.0330, `eval_thicknessB.json` 0.0306) to 1e-12.

Reading rule (TOPO-028's, verbatim): a row signals iff the lower edge of its
95 % bootstrap AP interval (2000 resamples, seed 20260815) sits strictly
above the upper edge of the 100-seed random baseline's IQR at the primary
threshold; a primary signal that dies at both sensitivity thresholds reads
as unstable, not as a signal. LOSO labels always come from the primary
zones; sensitivity thresholds re-evaluate the same ranking.

Comparison rule (declared before any number): a fusion form beats both solos
iff the lower edges of the 95 % paired by-zone bootstrap intervals of
(fusion - support) AND (fusion - thick) AP sit strictly above zero at the
primary threshold — joint zone resample, same 2000 resamples and seed.

Usage (from oneshot/real_errors/):

    python eval_fusionB.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids \
        --support-report ../../output/topo/real_paris4/eval_supportB.json \
        --thick-report ../../output/topo/real_paris4/eval_thicknessB.json \
        --report ../../output/topo/real_paris4/eval_fusionB.json
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
import detect_v5                                                      # noqa: E402
import sheet_erl                                                      # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_supportB import (BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,       # noqa: E402
                           bootstrap_ci)
from eval_thicknessB import thickness_candidates                      # noqa: E402

FAMILIES = ('support', 'thick')


def support_from_checkpoints(names, ckpt_dir):
    """The published support pool, replayed from checkpoints only.

    Unlike eval_supportB.support_candidates this never falls back to a live
    recto read: a missing or stale checkpoint is an error, not a resample —
    the declaration promises an offline run.
    """
    candidates = []
    for name in names:
        with open(os.path.join(ckpt_dir, f'{name}.pkl'), 'rb') as f:
            payload = pickle.load(f)
        for cells, mass, top in detect_v1.differenced_clusters(
                payload['evidence'], payload['best'], {}):
            if cells is None:
                continue
            candidates.append((name, top[0], payload['best'][top][0],
                               'support', mass, 1.0))
    return candidates


def zone_contributions(outcomes, zones, n):
    """Per-zone AP/recall decomposition, validated against the direct AP.

    The bootstrap_ci decomposition, returned raw so two rows can be
    resampled jointly for a paired delta.
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
    ap_direct = sheet_erl.average_precision(outcomes, n)
    if abs(float(contribution.mean()) - ap_direct) > 1e-9:
        raise SystemExit(f'AP reconstruction mismatch: '
                         f'{contribution.mean()} vs {ap_direct} — no CI')
    return contribution


def paired_delta_ci(contrib_a, contrib_b):
    """95 % CI of the AP delta (a - b) under a joint by-zone resample."""
    n_zones = len(contrib_a)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, n_zones, size=(BOOTSTRAP_RESAMPLES, n_zones))
    deltas = (contrib_a[samples] - contrib_b[samples]).mean(axis=1)
    return [float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5))]


def feature_matrix(pool):
    """Declared LOSO features: family indicators, log1p(mass) x indicator,
    the candidate's factor. The candidate tuple is (seg, row, col, family,
    mass, factor) throughout the line."""
    X = np.zeros((len(pool), 5))
    for i, (_seg, _row, _col, fam, mass, factor) in enumerate(pool):
        j = FAMILIES.index(fam)
        X[i, j] = 1.0
        X[i, 2 + j] = np.log1p(mass)
        X[i, 4] = factor
    return X


def label_vector(pool, zones):
    """PROTOCOL §3's hit rule as labels: centre inside the zone rectangle
    grown by 50 % per axis (the sheet_erl.hits test, minus one-credit
    bookkeeping — labels are per-candidate, not per-ranking)."""
    by_seg = {}
    for r in zones:
        by_seg.setdefault(r['segment'], []).append(r)
    y = np.zeros(len(pool))
    for i, (seg, row, col, *_rest) in enumerate(pool):
        for r in by_seg.get(seg, ()):
            growth_r = (r['row_hi'] - r['row_lo']) * 0.25
            growth_c = (r['col_hi'] - r['col_lo']) * 0.25
            if (r['row_lo'] - growth_r <= row < r['row_hi'] + growth_r
                    and r['col_lo'] - growth_c <= col < r['col_hi'] + growth_c):
                y[i] = 1.0
                break
    return y


def ranking_from_scores(pool, scores, top):
    """Deterministic score ranking: ties broken by (segment, row, col)."""
    order = sorted(range(len(pool)),
                   key=lambda i: (-scores[i], pool[i][0], pool[i][1],
                                  pool[i][2]))
    return [(pool[i][0], pool[i][1], pool[i][2], float(scores[i]))
            for i in order[:top]]


def loso_scores(pool, zones):
    """Leave-one-segment-out logit scores; None if a fold degenerates."""
    y = label_vector(pool, zones)
    X = feature_matrix(pool)
    segs = np.array([c[0] for c in pool])
    scores = np.zeros(len(pool))
    folds = {}
    for name in sorted(set(segs)):
        test = segs == name
        y_train = y[~test]
        if y_train.sum() == 0:
            return None, {'degenerate_fold': name,
                          'train_positives': 0}
        model = detect_v5.fit_logit(X[~test], y_train)
        scores[test] = detect_v5.predict_logit(model, X[test])
        folds[name] = {'train_positives': int(y_train.sum()),
                       'test_candidates': int(test.sum()),
                       'weights': [float(w) for w in model[1]]}
    return scores, folds


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--support-report', required=True,
                        help='published eval_supportB.json (regression gate)')
    parser.add_argument('--thick-report', required=True,
                        help='published eval_thicknessB.json (regression gate)')
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
    zone_min = published_support['declared_zone_min']
    if published_thick['declared_zone_min'] != zone_min:
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
    pool = support + thick

    # ---------------------------------------------- regression gates
    zones_primary = zone_records(corpus_map['zones'], zone_min)
    n_primary = len(zones_primary)
    solo_rankings = {
        'support': detect_v1.merge_channels(support, top=4 * n_primary),
        'thick': detect_v1.merge_channels(thick, top=4 * n_primary)}
    published_ap = {
        'support': published_support['rows']['support']['primary']['metrics']['ap'],
        'thick': published_thick['rows']['thick']['primary']['metrics']['ap']}
    solo_outcomes = {}
    for row_name, ranking in solo_rankings.items():
        metrics, outcomes = evaluate(ranking, zones_primary, n_primary)
        if abs(metrics['ap'] - published_ap[row_name]) > 1e-12:
            raise SystemExit(
                f"{row_name} replay regression FAILED: {metrics['ap']} vs "
                f"published {published_ap[row_name]} — nothing written")
        solo_outcomes[row_name] = outcomes
        print(f"{row_name} replay gate OK: AP {metrics['ap']:.4f} == published")

    # ---------------------------------------------- the four declared forms
    masses = np.array([c[4] for c in pool])
    fams = np.array([c[3] for c in pool])
    log_mass = np.log1p(masses)
    z = np.zeros(len(pool))
    for fam in FAMILIES:
        of = fams == fam
        sd = log_mass[of].std()
        z[of] = (log_mass[of] - log_mass[of].mean()) / (sd if sd else 1.0)

    rankings = {'frozen': detect_v1.merge_channels(pool, top=4 * n_primary),
                'rawmass': ranking_from_scores(pool, masses, 4 * n_primary),
                'zscore': ranking_from_scores(pool, z, 4 * n_primary)}
    scores_loso, folds = loso_scores(pool, zones_primary)
    if scores_loso is None:
        print(f"loso: DEGENERATE ({folds}) — row not ranked")
    else:
        rankings['loso'] = ranking_from_scores(pool, scores_loso,
                                               4 * n_primary)

    report = {'declared_zone_min': zone_min,
              'map': os.path.abspath(args.map),
              'declaration': 'CORPUS.md insert 18.08.2026 (TOPO-047, '
                             'twenty-fifth session), committed before this '
                             'ran; fully offline from the published '
                             'checkpoints',
              'gates': {'support_ap': published_ap['support'],
                        'thick_ap': published_ap['thick']},
              'pool': {'support': len(support), 'thick': len(thick)},
              'loso_folds': folds,
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

    solo_contrib = {row_name: zone_contributions(outcomes, zones_primary,
                                                 n_primary)
                    for row_name, outcomes in solo_outcomes.items()}

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
                    for solo in FAMILIES}
                variant['beats_both_solos'] = bool(all(
                    variant['delta_vs_solo'][solo]['delta_ci95'][0] > 0
                    for solo in FAMILIES))
            entry[label] = variant
        report['rows'][row_name] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    # The summary comes out of the same numbers as the table — the
    # stale-summary lesson, applied by construction.
    print(f"solo reference: support AP {published_ap['support']:.4f}, "
          f"thick AP {published_ap['thick']:.4f}")
    for row_name in rankings:
        p = report['rows'][row_name]['primary']
        verdict = 'SIGNAL' if p['signal'] else 'chance'
        beats = 'BEATS BOTH SOLOS' if p['beats_both_solos'] else 'not above'
        print(f"{row_name:8s} (mass >= {zone_min}, {p['n_zones']} zones): "
              f"AP {p['metrics']['ap']:.4f} "
              f"[{p['ci']['ap_ci95'][0]:.4f}-{p['ci']['ap_ci95'][1]:.4f}] "
              f"vs random IQR {p['baseline_random']['ap']['iqr'][0]:.4f}-"
              f"{p['baseline_random']['ap']['iqr'][1]:.4f} -> {verdict}; "
              f"{beats}")
        for solo in FAMILIES:
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
