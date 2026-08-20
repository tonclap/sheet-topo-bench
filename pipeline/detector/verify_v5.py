"""TOPO-037: re-verify the v5 result on everything available WITHOUT held-out.

Owner's decision (17.08.2026, after session 20): the held-out exam waits at
least until 19.08; until then the learned-fusion result is attacked with
every check that spends no exam budget. The checks and their pass criteria
are DECLARED in about/tasks/TOPO-037.md (committed before this ran):

  V1  label-permutation control (leakage): 100 permutations, LOSO pipeline;
      95th percentile of permuted APs must sit below v5lu's lower CI (0.82).
  V2  label-free baselines (mechanism attribution): global raw-mass sort,
      per-family z-score of log1p(mass), per-family max-normalisation.
  V3  harsher split: 3 contiguous winding blocks, score out-of-block;
      paired bootstrap vs v1 must stay significantly positive on AP.
  V4  hyperparameter sensitivity: logit lambda 0.25 / 4.0.
  V5  feature ablations: no vjump; shared mass slope; intercepts only.
  V6  transfer to PHerc0139 (opened v1-iteration held-out, different scroll
      and prediction model): logit trained on the FULL Paris 4 dev pool,
      applied to the 0139 pool replayed from checkpoints; v1 regression
      gate to 6 decimals; corroboration if transfer AP >= frozen 0.6468.
  V7  real corpus B: does the learned fusion stop drowning support?
      Frozen-merge regression gate against eval_supportB.json; signal
      criterion = learned prox+support AP CI above the stored random IQR.

Self-gate: the harness's own LOSO reproduction of v5lu must match the
frozen detector_v5 report to 6 decimals before any check runs.

Usage (from pipeline/detector/):

    python verify_v5.py --topo ../../output/topo \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --report ../../output/topo/verify_v5_report.json \
        --out VERIFY_V5.md
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
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'real_errors'))
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v1 as v1                                                # noqa: E402
import detect_v3 as v3                                                # noqa: E402
import detect_v4 as v4                                                # noqa: E402
import detect_v5 as v5                                                # noqa: E402
from eval_real import prox_candidates, evaluate, zone_records         # noqa: E402
from eval_supportB import support_candidates, bootstrap_ci            # noqa: E402

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815
PERMUTATIONS = 100


def loso_scores(pool, X, y, l2):
    """detect_v5's leave-one-segment-out, parameterised by lambda and X."""
    old = v5.LOGIT_L2
    v5.LOGIT_L2 = l2
    try:
        scores = np.zeros(len(pool))
        for seg in sorted({c[0] for c in pool}):
            test = np.array([c[0] == seg for c in pool])
            model = v5.fit_logit(X[~test], y[~test])
            scores[test] = v5.predict_logit(model, X[test])
    finally:
        v5.LOGIT_L2 = old
    return scores


def ap_recall(rank, injections):
    n = len(injections)
    outcomes = sheet_erl.hits(rank, injections, {})
    ap = sheet_erl.average_precision(outcomes, n)
    recall = (sum(1 for h, *_ in outcomes[:n] if h is not None) / n
              if n else 0.0)
    return ap, recall, outcomes


def contribution_terms(outcomes, injections):
    """Per-injection AP contributions + recall indicators (paired bootstrap)."""
    n = len(injections)
    contrib = np.zeros(n)
    recall = np.zeros(n)
    tp = 0
    for i, (hit, *_rest) in enumerate(outcomes, 1):
        if hit is not None:
            tp += 1
            contrib[hit] = tp / i
            if i <= n:
                recall[hit] = 1.0
    return contrib, recall


def paired_delta(outcomes_a, outcomes_b, injections, samples):
    ca, _ = contribution_terms(outcomes_a, injections)
    cb, _ = contribution_terms(outcomes_b, injections)
    delta = (ca - cb)[samples].mean(axis=1)
    return (float(np.mean(delta)), float(np.percentile(delta, 2.5)),
            float(np.percentile(delta, 97.5)))


# Row order of the generated table is the order the checks are DECLARED in
# this file's docstring — not the report's key order. The report is stored
# with sort_keys=True, so iterating it directly would reorder the rows and
# make the shipped file irreproducible (caught by TOPO-060).
V2_ORDER = ('global_mass', 'family_zscore', 'family_maxnorm')
V5_ORDER = ('no_vjump', 'shared_mass_slope', 'intercepts_only')


def _ordered(mapping, order):
    """Declared keys first, then anything else, so a new key cannot vanish."""
    return [k for k in order if k in mapping] + \
           [k for k in mapping if k not in order]


def render_md(r):
    """VERIFY_V5.md purely from the report — no recomputation, no network.

    Split out of main() by TOPO-060 so verify.py can regenerate the shipped
    file offline and diff it, the way it already does for the ABLATION
    family. main() renders through this same function, so the two paths
    cannot drift.
    """
    lines = [
        '# Re-check of v5 without held-out (TOPO-037) — generated by verify_v5.py',
        '',
        '',
        '_The checks and their criteria were declared in about/tasks/TOPO-037.md before the run; '
        'dev anchors: v5lu AP '
        f"{r['v5lu_dev_ap']:.4f}, v1 {r['v1_dev_ap']:.4f}. Self-gate: the LOSO "
        'harness reproduces the frozen detector_v5 report to 6 decimals._',
        '',
        '| check | result | criterion | verdict |',
        '|---|---|---|---|',
        f"| V1 permutations ({PERMUTATIONS}) | median "
        f"{r['V1_permutation']['aps']['median']:.4f}, p95 "
        f"{r['V1_permutation']['aps']['p95']:.4f}, max "
        f"{r['V1_permutation']['aps']['max']:.4f} | p95 < 0.82 | "
        + ('**PASS**' if r['V1_permutation']['passed'] else '**FAIL — leakage**')
        + ' |',
    ]
    for label in _ordered(r['V2_label_free'], V2_ORDER):
        row = r['V2_label_free'][label]
        d = row['delta_vs_v1']
        lines.append(
            f"| V2 {label} | AP {row['ap']:.4f} "
            f"(Δv1 {d[0]:+.3f} [{d[1]:+.3f}..{d[2]:+.3f}]"
            + (' sig.' if row['significant'] else '')
            + ') | context | — |')
    d = r['V3_block_split']['delta_vs_v1']
    lines.append(
        f"| V3 block split | AP {r['V3_block_split']['ap']:.4f} "
        f"(Δv1 {d[0]:+.3f} [{d[1]:+.3f}..{d[2]:+.3f}]) | Δ significant > 0 | "
        + ('**PASS**' if r['V3_block_split']['passed'] else '**FAIL**') + ' |')
    for l2, row in r['V4_lambda']['rows'].items():
        lines.append(f"| V4 λ={l2} | AP {row['ap']:.4f} "
                     f"({row['shift']:+.4f}) | \\|shift\\| < 0.02 | "
                     + ('stable' if abs(row['shift']) < 0.02
                        else 'shifted') + ' |')
    for label in _ordered(r['V5_feature_ablation'], V5_ORDER):
        row = r['V5_feature_ablation'][label]
        lines.append(f"| V5 {label} | AP {row['ap']:.4f} | context | — |")
    t = r['V6_transfer_0139']
    lines.append(
        f"| V6 transfer to 0139 | AP {t['transfer_ap']:.4f} against v1 "
        f"{t['v1_ap']:.4f}; recall {t['transfer_recall']:.3f}; "
        + ', '.join(f"{k} {t['transfer_recall_by_type'][k]:.3f}"
                    for k in _ordered(t['transfer_recall_by_type'], 'SMH'))
        + ' | AP ≥ v1-0139 | '
        + ('**PASS**' if t['passed'] else '**BELOW — scroll-specific**')
        + ' |')
    b = r['V7_corpusB']
    lines.append(
        f"| V7 corpus B | learned fusion AP {b['learned_fusion_ap']:.4f} "
        f"[{b['learned_fusion_ci'][0]:.4f}..{b['learned_fusion_ci'][1]:.4f}] "
        f"against the frozen {b['frozen_merge_ap']:.4f} and support solo "
        f"{b['support_alone_ap']:.4f} | CI lower > random IQR "
        f"({b['random_iqr_hi']:.4f}) | "
        + ('**SIGNAL PRESERVED**' if b['passed'] else '**DROWNED**') + ' |')
    lines.append('')
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--topo', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    topo = args.topo

    # ---------------------------------------------------------------- dev pool
    corpus = os.path.join(topo, 'corpus_paris4')
    with open(os.path.join(corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    injections = [r for r in manifest['injections'] if r['winding_low'] < 100]
    by_id = {r['id']: r for r in injections}
    names = sorted({r['segment'] for r in injections})
    n = len(injections)

    ckpt_key = {'corpus': os.path.abspath(corpus), 'bands': 'dev',
                'detector': 'v2'}
    pool = []
    for name in names:
        with open(os.path.join(topo, 'ckpt_paris4_dev_v2', f'{name}.pkl'),
                  'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') != ckpt_key:
            raise SystemExit(f'{name}: dev checkpoint key mismatch')
        pool += [tuple(c) for c in payload['candidates']]
    v1_pool = [c for c in pool if c[3] in ('prox', 'rect', 'support')]

    with open(os.path.join(topo, 'probe_ct_paris4.json'),
              encoding='utf-8') as f:
        threshold = json.load(f)['calibration']['threshold']
    records, seen = [], set()
    with open(os.path.join(topo, 'probe_ct_paris4_windows.jsonl'),
              encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec['id'] not in seen and rec['id'] in by_id:
                seen.add(rec['id'])
                records.append(rec)
    ct_pool = v3.ct_candidates(records, by_id, threshold)
    u_pool = v4.union_pool(v1_pool, ct_pool)

    with open(os.path.join(topo, 'detector_v5_paris4.json'),
              encoding='utf-8') as f:
        v5_report = json.load(f)

    X_u = v5.feature_matrix(u_pool)
    y_u = v5.label_vector(u_pool, injections)
    X_1 = v5.feature_matrix(v1_pool)
    y_1 = v5.label_vector(v1_pool, injections)

    # Self-gate: reproduce the frozen v5lu AP to 6 decimals.
    scores_u = loso_scores(u_pool, X_u, y_u, l2=1.0)
    rank_u = v5.ranking_from_scores(u_pool, scores_u, top=4 * n)
    ap_u, recall_u, outcomes_u = ap_recall(rank_u, injections)
    frozen_ap = v5_report['metrics']['ap']
    if round(ap_u, 6) != round(frozen_ap, 6):
        raise SystemExit(f'self-gate FAILED: harness v5lu AP {ap_u:.6f} vs '
                         f'frozen {frozen_ap:.6f}')
    print(f'self-gate OK: v5lu AP {ap_u:.4f} == frozen {frozen_ap:.4f}')

    rank_v1 = v1.merge_channels(v1_pool, top=4 * n)
    ap_v1, _, outcomes_v1 = ap_recall(rank_v1, injections)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    report = {'declaration': 'about/tasks/TOPO-037.md, committed before run',
              'v5lu_dev_ap': ap_u, 'v1_dev_ap': ap_v1}

    # ------------------------------------------------ V1 permutation control
    perm_aps = []
    for seed in range(PERMUTATIONS):
        prng = np.random.default_rng(seed)
        y_perm = y_u[prng.permutation(len(y_u))]
        s = loso_scores(u_pool, X_u, y_perm, l2=1.0)
        perm_aps.append(ap_recall(
            v5.ranking_from_scores(u_pool, s, top=4 * n), injections)[0])
    p95 = float(np.percentile(perm_aps, 95))
    v5lu_ci_lo = 0.82   # lower CI bound from ABLATION_V5.md, declared anchor
    report['V1_permutation'] = {
        'aps': {'median': float(np.median(perm_aps)), 'p95': p95,
                'max': float(np.max(perm_aps))},
        'criterion': f'p95 < {v5lu_ci_lo}', 'passed': bool(p95 < v5lu_ci_lo)}
    print(f"V1 permutation: median {np.median(perm_aps):.4f}, p95 {p95:.4f} "
          f"-> {'PASS' if p95 < v5lu_ci_lo else 'FAIL (leakage!)'}")

    # ------------------------------------------------ V2 label-free baselines
    def rank_by(scores):
        return v5.ranking_from_scores(u_pool, scores, top=4 * n)

    mass = np.array([c[4] for c in u_pool], dtype=float)
    fams = np.array([c[3] for c in u_pool])
    z = np.zeros(len(u_pool))
    mx = np.zeros(len(u_pool))
    for fam in set(fams):
        of = fams == fam
        lm = np.log1p(mass[of])
        sd = lm.std() or 1.0
        z[of] = (lm - lm.mean()) / sd
        mx[of] = mass[of] / (mass[of].max() or 1.0)
    v2_rows = {}
    for label, s in (('global_mass', mass), ('family_zscore', z),
                     ('family_maxnorm', mx)):
        ap, recall, outc = ap_recall(rank_by(s), injections)
        d = paired_delta(outc, outcomes_v1, injections, samples)
        v2_rows[label] = {'ap': ap, 'recall_at_n': recall,
                          'delta_vs_v1': d, 'significant': bool(d[1] > 0)}
        print(f'V2 {label}: AP {ap:.4f} (delta vs v1 {d[0]:+.4f} '
              f'[{d[1]:+.4f}..{d[2]:+.4f}])')
    report['V2_label_free'] = v2_rows

    # ------------------------------------------------ V3 winding-block split
    winding_of = {}
    for r in injections:
        winding_of.setdefault(r['segment'], r['winding_low'])
    ordered = sorted(names, key=lambda s: winding_of[s])
    blocks = [ordered[i::3] for i in range(3)]  # NOT contiguous — see below
    # Contiguous blocks (declared): thirds of the winding-sorted segment list.
    k = (len(ordered) + 2) // 3
    blocks = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
    scores_block = np.zeros(len(u_pool))
    for block in blocks:
        test = np.array([c[0] in block for c in u_pool])
        if not test.any():
            continue
        model = v5.fit_logit(X_u[~test], y_u[~test])
        scores_block[test] = v5.predict_logit(model, X_u[test])
    ap_b, recall_b, outcomes_b = ap_recall(rank_by(scores_block), injections)
    d_b = paired_delta(outcomes_b, outcomes_v1, injections, samples)
    report['V3_block_split'] = {
        'blocks': [[winding_of[b[0]], winding_of[b[-1]]] for b in blocks],
        'ap': ap_b, 'recall_at_n': recall_b, 'delta_vs_v1': d_b,
        'criterion': 'paired delta AP vs v1 significantly > 0',
        'passed': bool(d_b[1] > 0)}
    print(f"V3 block split: AP {ap_b:.4f}, delta vs v1 {d_b[0]:+.4f} "
          f"[{d_b[1]:+.4f}..{d_b[2]:+.4f}] "
          f"-> {'PASS' if d_b[1] > 0 else 'FAIL'}")

    # ------------------------------------------------ V4 lambda sensitivity
    v4_rows = {}
    for l2 in (0.25, 4.0):
        s = loso_scores(u_pool, X_u, y_u, l2=l2)
        ap, _, _ = ap_recall(rank_by(s), injections)
        v4_rows[str(l2)] = {'ap': ap, 'shift': ap - ap_u}
        print(f'V4 lambda={l2}: AP {ap:.4f} (shift {ap - ap_u:+.4f})')
    report['V4_lambda'] = {'rows': v4_rows,
                           'stable': bool(all(abs(r['shift']) < 0.02
                                              for r in v4_rows.values()))}

    # ------------------------------------------------ V5 feature ablations
    def loso_on(X):
        s = np.zeros(len(u_pool))
        for seg in sorted({c[0] for c in u_pool}):
            test = np.array([c[0] == seg for c in u_pool])
            model = v5.fit_logit(X[~test], y_u[~test])
            s[test] = v5.predict_logit(model, X[test])
        return ap_recall(rank_by(s), injections)[0]

    lm_all = np.log1p(mass)[:, None]
    variants = {
        'no_vjump': X_u[:, :6],
        'shared_mass_slope': np.hstack([X_u[:, :3], lm_all, X_u[:, 6:7]]),
        'intercepts_only': X_u[:, :3]}
    v5_rows = {label: {'ap': loso_on(X)} for label, X in variants.items()}
    for label, row in v5_rows.items():
        print(f"V5 {label}: AP {row['ap']:.4f}")
    report['V5_feature_ablation'] = v5_rows

    # ------------------------------------------------ V6 transfer to 0139
    corpus_0139 = os.path.join(topo, 'corpus_0139')
    with open(os.path.join(corpus_0139, 'manifest.json'),
              encoding='utf-8') as f:
        man_0139 = json.load(f)
    inj_0139 = man_0139['injections']
    names_0139 = sorted({r['segment'] for r in inj_0139})
    n_0139 = len(inj_0139)
    key_0139 = {'corpus': os.path.abspath(corpus_0139), 'bands': 'all',
                'with_prediction': True}
    pool_0139 = []
    for name in names_0139:
        with open(os.path.join(topo, 'ckpt_0139', f'{name}.pkl'), 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') != key_0139:
            raise SystemExit(f'{name}: 0139 checkpoint key mismatch')
        pool_0139 += [tuple(c) for c in payload['candidates']]

    with open(os.path.join(topo, 'detector_v1_0139.json'),
              encoding='utf-8') as f:
        frozen_0139 = json.load(f)
    rank_0139_v1 = v1.merge_channels(pool_0139, top=4 * n_0139)
    ap_0139_v1, recall_0139_v1, _ = ap_recall(rank_0139_v1, inj_0139)
    if (round(ap_0139_v1, 6) != round(frozen_0139['metrics']['ap'], 6)
            or round(recall_0139_v1, 6)
            != round(frozen_0139['metrics']['recall_at_n'], 6)):
        raise SystemExit(
            f"0139 v1 regression FAILED: {ap_0139_v1:.6f} vs "
            f"{frozen_0139['metrics']['ap']:.6f}")
    print(f'0139 v1 regression OK: AP {ap_0139_v1:.4f}')

    model_dev = v5.fit_logit(X_1, y_1)   # full Paris 4 dev pool, no CV
    X_0139 = v5.feature_matrix(pool_0139)
    s_0139 = v5.predict_logit(model_dev, X_0139)
    rank_0139_v5 = v5.ranking_from_scores(pool_0139, s_0139, top=4 * n_0139)
    ap_t, recall_t, outc_t = ap_recall(rank_0139_v5, inj_0139)
    found = {inj_0139[h]['id'] for h, *_ in outc_t[:n_0139] if h is not None}
    by_type_t = {t: (sum(1 for r in inj_0139
                         if r['type'] == t and r['id'] in found)
                     / max(sum(1 for r in inj_0139 if r['type'] == t), 1))
                 for t in 'SMH'}
    report['V6_transfer_0139'] = {
        'v1_ap': ap_0139_v1, 'v1_recall': recall_0139_v1,
        'transfer_ap': ap_t, 'transfer_recall': recall_t,
        'transfer_recall_by_type': by_type_t,
        'criterion': 'transfer AP >= frozen v1 0139 AP',
        'passed': bool(ap_t >= ap_0139_v1)}
    print(f"V6 transfer 0139: AP {ap_t:.4f} vs v1 {ap_0139_v1:.4f}, "
          f"recall {recall_t:.3f} vs {recall_0139_v1:.3f}, by-type "
          + ', '.join(f'{k}={v:.3f}' for k, v in by_type_t.items())
          + f" -> {'PASS' if ap_t >= ap_0139_v1 else 'BELOW'}")

    # ------------------------------------------------ V7 real corpus B
    real_dir = os.path.join(topo, 'real_paris4')
    with open(os.path.join(real_dir, 'corpusB.json'), encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(os.path.join(real_dir, 'eval_supportB.json'),
              encoding='utf-8') as f:
        frozen_b = json.load(f)
    zone_min = frozen_b['declared_zone_min']
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    names_b = corpus_map['segments']
    grids_b = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
               for name in names_b}
    support_b = support_candidates(names_b, grids_b, scroll, args.cache,
                                   manifest['z_quantiles'],
                                   os.path.join(real_dir, 'cells_supportB'))
    prox_b = prox_candidates(names_b, grids_b, centre)
    zones_b = zone_records(corpus_map['zones'], zone_min)
    n_b = len(zones_b)

    frozen_row = frozen_b['rows']['prox_support']['primary']
    rank_frozen = v1.merge_channels(prox_b + support_b, top=4 * n_b)
    metrics_frozen, _ = evaluate(rank_frozen, zones_b, n_b)
    if round(metrics_frozen['ap'], 6) != round(frozen_row['metrics']['ap'], 6):
        raise SystemExit(
            f"B frozen-merge regression FAILED: {metrics_frozen['ap']:.6f} "
            f"vs {frozen_row['metrics']['ap']:.6f}")
    print(f"B frozen-merge regression OK: AP {metrics_frozen['ap']:.4f}")

    X_b = v5.feature_matrix(prox_b + support_b)
    s_b = v5.predict_logit(model_dev, X_b)
    rank_b = v5.ranking_from_scores(prox_b + support_b, s_b, top=4 * n_b)
    metrics_b, outcomes_b7 = evaluate(rank_b, zones_b, n_b)
    ci_b = bootstrap_ci(outcomes_b7, zones_b, n_b)
    random_iqr_hi = frozen_row['baseline_random']['ap']['iqr'][1]
    support_alone = frozen_b['rows']['support']['primary']['metrics']['ap']
    report['V7_corpusB'] = {
        'frozen_merge_ap': metrics_frozen['ap'],
        'support_alone_ap': support_alone,
        'learned_fusion_ap': metrics_b['ap'],
        'learned_fusion_ci': ci_b['ap_ci95'],
        'random_iqr_hi': random_iqr_hi,
        'criterion': 'learned-fusion AP CI lower bound > random IQR high',
        'passed': bool(ci_b['ap_ci95'][0] > random_iqr_hi)}
    print(f"V7 corpus B: learned fusion AP {metrics_b['ap']:.4f} "
          f"[{ci_b['ap_ci95'][0]:.4f}..{ci_b['ap_ci95'][1]:.4f}] vs frozen "
          f"merge {metrics_frozen['ap']:.4f}, support alone {support_alone:.4f}, "
          f"random IQR hi {random_iqr_hi:.4f} "
          f"-> {'SIGNAL KEPT' if report['V7_corpusB']['passed'] else 'DROWNED'}")

    # ---------------------------------------------------------------- output
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f'report at {args.report}')

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(render_md(report)))
        print(f'written to {args.out}')


if __name__ == '__main__':
    main()
