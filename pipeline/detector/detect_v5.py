"""Detector v5: learned fusion without GPU (TOPO-035, option 1 of TOPO-027).

The wall this file attacks was measured three times: both rank merges
redistribute the deficit between families (TOPO-016: v3uq buys AP +0.040 at
S -0.120, both significant), a third feature family carries new information to
the candidates and loses it in the merge (TOPO-026), and on the real corpus B
the frozen merge drowns the only independently validated channel (TOPO-028:
support 0.0330 -> 0.0012). The owner's decision of 17.08 (GPU_GATE.md, second
insert) opens exactly one branch: a calibrated ranker over the frozen
detectors' *existing* candidates — no GPU, no new sampling, no channel edits.

**Protocol, declared 17.08.2026 BEFORE any training run of this file
(the commit adding this file precedes the first run; every constant below is
fixed here, none is tuned on output):**

1. **Pools are replayed, not recomputed.** The v1 pool (prox/rect/support)
   comes from the detect_v2 checkpoints, the ct pool from the probe's stored
   windows, the union pool (ct folded into support) verbatim from detect_v4's
   `union_pool`. Regression gates abort the run if the replayed v1 or v3u
   percentile ranking does not reproduce the frozen reports to 6 decimals,
   and if the replayed v3uq quota ranking does not reproduce the detector_v4
   report to 6 decimals.
2. **Labels are the harness's own hit rule, without the greedy uniqueness:**
   a candidate is positive iff its (row, col) lies inside any scoped
   injection's rectangle grown by 50 % per axis (sheet_erl.hits, PROTOCOL
   §3). Mechanistic, from the manifest alone.
3. **Features — 7 per candidate, nothing else:** family one-hot (prox, rect,
   support; in the union pool ct is already folded into support), per-family
   log1p(mass) (the raw pre-percentile channel score, per-family so the
   model can calibrate each family's scale — the percentile merge's one
   degree of freedom, made explicit), and the vertical-jump factor (U-010's
   disambiguator; 1.0 outside prox). No percentiles, no positions, no
   segment identity.
4. **Models — both declared here with fixed hyperparameters, no search:**
   - *logit*: L2 logistic regression, lambda = 1.0, intercept unpenalised,
     features standardised on the training fold, IRLS capped at 100
     iterations, step tolerance 1e-10. Deterministic.
   - *boost*: gradient boosting of depth-1 stumps on log-loss, 200 rounds,
     learning rate 0.1, leaf regularisation 1.0, split thresholds at
     midpoints of consecutive sorted unique feature values of the training
     fold; ties broken to the lowest feature index, then lowest threshold.
     Deterministic.
   Each model trains on each pool: v5l/v5b on the v1 pool, v5lu/v5bu on the
   union pool — four learned configurations, mirroring v4's four.
5. **Splits are leave-one-segment-out over the dev segments:** a candidate's
   score is only ever produced by a model that saw neither its segment's
   candidates nor its segment's injections. The ranking sorts by descending
   out-of-fold probability, ties by (segment, row, col), top 4*N — the
   evaluation harness, bootstrap seed and resamples are v4's, unchanged.
6. **Ship rule (v3's and v4's, unchanged):** a learned form is a candidate
   for the held-out exam only if, against v1 on paired bootstrap, AP or M
   improves significantly AND none of AP / recall@N / S / M / H / plausible
   degrades significantly. Which single form (if any) goes to held-out is
   fixed in ABLATION_V5.md before any held-out run; this file never touches
   TOPO-025's budget.
7. **The band-run signature (TOPO-034) is NOT a feature here.** If TOPO-034
   yields a numeric signature it enters as its own declared ablation row
   later, not silently into this protocol.

Usage (from pipeline/detector/):

    python detect_v5.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids \
        --probe-report ../../output/topo/probe_ct_paris4.json \
        --probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --v1-report ../../output/topo/detector_v1_paris4.json \
        --v3-report ../../output/topo/detector_v3_paris4.json \
        --v4-report ../../output/topo/detector_v4_paris4.json \
        --bands dev --report ../../output/topo/detector_v5_paris4.json
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
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v1 as v1                                                # noqa: E402
import detect_v3 as v3                                                # noqa: E402
import detect_v4 as v4                                                # noqa: E402

FAMILIES = ('prox', 'rect', 'support')

LOGIT_L2 = 1.0
LOGIT_ITERS = 100
LOGIT_TOL = 1e-10

BOOST_ROUNDS = 200
BOOST_LR = 0.1
BOOST_LEAF_L2 = 1.0


def feature_matrix(pool):
    """§3's seven features; the candidate tuple is (seg, row, col, family,
    mass, factor) throughout the line."""
    X = np.zeros((len(pool), 7))
    for i, (_seg, _row, _col, fam, mass, factor) in enumerate(pool):
        j = FAMILIES.index(fam)
        X[i, j] = 1.0
        X[i, 3 + j] = np.log1p(mass)
        X[i, 6] = factor
    return X


def label_vector(pool, injections):
    """§2: positive iff inside any scoped injection's grown rectangle."""
    by_seg = {}
    for r in injections:
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


def fit_logit(X, y):
    n, d = X.shape
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.hstack([np.ones((n, 1)), (X - mu) / sd])
    w = np.zeros(d + 1)
    reg = LOGIT_L2 * np.eye(d + 1)
    reg[0, 0] = 0.0
    for _ in range(LOGIT_ITERS):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        grad = Xs.T @ (p - y) + reg @ w
        hess = (Xs.T * (p * (1.0 - p))) @ Xs + reg + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < LOGIT_TOL:
            break
    return ('logit', w, mu, sd)


def predict_logit(model, X):
    _, w, mu, sd = model
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return 1.0 / (1.0 + np.exp(-Xs @ w))


def fit_boost(X, y):
    """Depth-1 Newton stumps on log-loss; every constant from the header."""
    n, d = X.shape
    prior = min(max(float(y.mean()), 1e-6), 1.0 - 1e-6)
    f0 = float(np.log(prior / (1.0 - prior)))
    f = np.full(n, f0)
    stumps = []
    orders = [np.argsort(X[:, j], kind='stable') for j in range(d)]
    for _ in range(BOOST_ROUNDS):
        p = 1.0 / (1.0 + np.exp(-f))
        g, h = y - p, p * (1.0 - p)
        g_total, h_total = float(g.sum()), float(h.sum())
        base_gain = g_total ** 2 / (h_total + BOOST_LEAF_L2)
        best = None  # (gain, j, threshold, left value, right value)
        for j in range(d):
            order = orders[j]
            xj = X[order, j]
            gl = np.cumsum(g[order])[:-1]
            hl = np.cumsum(h[order])[:-1]
            valid = xj[1:] > xj[:-1]
            if not valid.any():
                continue
            thresholds = (xj[:-1] + xj[1:]) / 2.0
            gr, hr = g_total - gl, h_total - hl
            gain = (gl ** 2 / (hl + BOOST_LEAF_L2)
                    + gr ** 2 / (hr + BOOST_LEAF_L2) - base_gain)
            gain[~valid] = -np.inf
            k = int(np.argmax(gain))
            if best is None or gain[k] > best[0]:
                best = (float(gain[k]), j, float(thresholds[k]),
                        float(gl[k] / (hl[k] + BOOST_LEAF_L2)),
                        float(gr[k] / (hr[k] + BOOST_LEAF_L2)))
        if best is None or best[0] <= 0:
            break
        _, j, threshold, left, right = best
        stumps.append((j, threshold, left, right))
        f += BOOST_LR * np.where(X[:, j] <= threshold, left, right)
    return ('boost', f0, stumps)


def predict_boost(model, X):
    _, f0, stumps = model
    f = np.full(len(X), f0)
    for j, threshold, left, right in stumps:
        f += BOOST_LR * np.where(X[:, j] <= threshold, left, right)
    return 1.0 / (1.0 + np.exp(-f))


def crossval_scores(pool, y, fit, predict):
    """§5: leave-one-segment-out; scores are out-of-fold only."""
    X = feature_matrix(pool)
    scores = np.zeros(len(pool))
    for seg in sorted({c[0] for c in pool}):
        test = np.array([c[0] == seg for c in pool])
        model = fit(X[~test], y[~test])
        scores[test] = predict(model, X[test])
    return scores


def ranking_from_scores(pool, scores, top):
    order = sorted(range(len(pool)),
                   key=lambda i: (-scores[i], pool[i][0], pool[i][1],
                                  pool[i][2]))
    return [(pool[i][0], pool[i][1], pool[i][2], float(scores[i]))
            for i in order[:top]]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--probe-report', required=True)
    parser.add_argument('--probe-windows', required=True)
    parser.add_argument('--v2-checkpoints', required=True)
    parser.add_argument('--v1-report', required=True)
    parser.add_argument('--v3-report', required=True)
    parser.add_argument('--v4-report', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']

    def in_scope(record):
        if args.bands == 'dev':
            return record['winding_low'] < 100
        if args.bands == 'heldout':
            return record['winding_low'] >= 100
        return True

    injections = [r for r in manifest['injections'] if in_scope(r)]
    scoped = dict(manifest, injections=injections)
    by_id = {r['id']: r for r in injections}
    names = sorted({r['segment'] for r in injections})
    n = len(injections)

    type_counts = {t: sum(1 for r in injections if r['type'] == t)
                   for t in 'SMH'}
    quotas = {'prox': type_counts['S'] / n,
              'support': type_counts['M'] / n,
              'rect': type_counts['H'] / n}

    # v1 pool, replayed from the v2 checkpoints (front dropped) — v4's step.
    ckpt_key = {'corpus': os.path.abspath(args.corpus), 'bands': args.bands,
                'detector': 'v2'}
    pool = []
    for name in names:
        path = os.path.join(args.v2_checkpoints, f'{name}.pkl')
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') != ckpt_key:
            raise SystemExit(f'{path}: checkpoint key mismatch — refusing to '
                             f'replay candidates from a different run')
        pool += [tuple(c) for c in payload['candidates']]
    v1_pool = [c for c in pool if c[3] in ('prox', 'rect', 'support')]

    with open(args.probe_report, encoding='utf-8') as f:
        probe_report = json.load(f)
    threshold = probe_report['calibration']['threshold']
    records, seen = [], set()
    with open(args.probe_windows, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec['id'] in seen or rec['id'] not in by_id:
                continue
            seen.add(rec['id'])
            records.append(rec)
    if len(records) != n:
        raise SystemExit(f'probe windows hold {len(records)} of {n} scoped '
                         f'injections — the probe run is incomplete')
    ct_pool = v3.ct_candidates(records, by_id, threshold)
    v3u_pool = v4.union_pool(v1_pool, ct_pool)
    print(f'pools: v1 {len(v1_pool)}, ct {len(ct_pool)}, '
          f'v3u union {len(v3u_pool)}')

    pristine = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
                for name in names}
    zones = {name: sheet_erl.segment_zones(scoped, name) for name in names}
    erl_broken, *_ = sheet_erl.sheet_erl(pristine, zones, voxel_mm,
                                         args.row_step)

    def judged(rank):
        result = sheet_erl.evaluate_ranking(rank, scoped, pristine, voxel_mm,
                                            args.row_step,
                                            erl_broken=erl_broken)
        full = sheet_erl.hits(rank, injections, pristine)
        outcomes = full[:n]
        found = {injections[h]['id'] for h, *_ in outcomes if h is not None}
        by_type = {}
        for t in 'SMH':
            of_type = [r for r in injections if r['type'] == t]
            by_type[t] = (sum(1 for r in of_type if r['id'] in found)
                          / len(of_type) if of_type else None)
        plausible = [r for r in injections if r.get('plausible')]
        recall_plausible = (sum(1 for r in plausible if r['id'] in found)
                            / len(plausible)) if plausible else None
        by_band = {}
        for band in sorted({r['band'] for r in injections}):
            of_band = [r for r in injections if r['band'] == band]
            by_band[f'w{band * 10:03d}-{band * 10 + 9:03d}'] = round(
                sum(1 for r in of_band if r['id'] in found) / len(of_band), 4)
        per_rank = [None if h is None else injections[h]['id']
                    for h, *_rest in full]
        per_injection = {r['id']: None for r in injections}
        for i, (h, *_rest) in enumerate(full, 1):
            if h is not None:
                per_injection[injections[h]['id']] = i
        return {'metrics': result, 'recall_by_type': by_type,
                'recall_by_band': by_band,
                'recall_on_locally_plausible': recall_plausible,
                'per_rank_outcomes': per_rank,
                'per_injection_rank': per_injection}

    # §1's three regression gates.
    v1_block = judged(v1.merge_channels(v1_pool, top=4 * n))
    with open(args.v1_report, encoding='utf-8') as f:
        frozen = json.load(f)
    ours, theirs = v1_block['metrics'], frozen['metrics']
    if (round(ours['ap'], 6) != round(theirs['ap'], 6)
            or round(ours['recall_at_n'], 6)
            != round(theirs['recall_at_n'], 6)):
        raise SystemExit(
            f"v1 regression FAILED: replayed AP {ours['ap']:.6f} / recall "
            f"{ours['recall_at_n']:.6f} vs frozen {theirs['ap']:.6f} / "
            f"{theirs['recall_at_n']:.6f}")
    print(f"v1 regression OK: AP {ours['ap']:.4f} == frozen {theirs['ap']:.4f}")

    v3u_block = judged(v1.merge_channels(v3u_pool, top=4 * n))
    with open(args.v3_report, encoding='utf-8') as f:
        v3_report = json.load(f)
    v3u_frozen = v3_report['ablation_v3u_config']['metrics']
    if (round(v3u_block['metrics']['ap'], 6) != round(v3u_frozen['ap'], 6)
            or round(v3u_block['metrics']['recall_at_n'], 6)
            != round(v3u_frozen['recall_at_n'], 6)):
        raise SystemExit(
            f"v3u regression FAILED: replayed AP "
            f"{v3u_block['metrics']['ap']:.6f} vs v3 report "
            f"{v3u_frozen['ap']:.6f}")
    print(f"v3u regression OK: AP {v3u_block['metrics']['ap']:.4f} == "
          f"v3 report {v3u_frozen['ap']:.4f}")

    v3uq_block = judged(v4.merge_channels_quota(v3u_pool, top=4 * n,
                                                quotas=quotas))
    with open(args.v4_report, encoding='utf-8') as f:
        v4_report = json.load(f)
    v3uq_frozen = v4_report['metrics']
    if (round(v3uq_block['metrics']['ap'], 6) != round(v3uq_frozen['ap'], 6)
            or round(v3uq_block['metrics']['recall_at_n'], 6)
            != round(v3uq_frozen['recall_at_n'], 6)):
        raise SystemExit(
            f"v3uq regression FAILED: replayed AP "
            f"{v3uq_block['metrics']['ap']:.6f} vs v4 report "
            f"{v3uq_frozen['ap']:.6f}")
    print(f"v3uq regression OK: AP {v3uq_block['metrics']['ap']:.4f} == "
          f"v4 report {v3uq_frozen['ap']:.4f}")

    # §2–§5: labels, out-of-fold scores, rankings.
    learned_blocks, model_notes = {}, {}
    for tag, this_pool in (('v5l', v1_pool), ('v5lu', v3u_pool),
                           ('v5b', v1_pool), ('v5bu', v3u_pool)):
        fit, predict = ((fit_logit, predict_logit) if tag in ('v5l', 'v5lu')
                        else (fit_boost, predict_boost))
        y = label_vector(this_pool, injections)
        scores = crossval_scores(this_pool, y, fit, predict)
        learned_blocks[tag] = judged(
            ranking_from_scores(this_pool, scores, top=4 * n))
        model_notes[tag] = {
            'pool': 'v1' if this_pool is v1_pool else 'v3u',
            'model': 'logit' if tag in ('v5l', 'v5lu') else 'boost',
            'positives': int(y.sum()), 'candidates': len(this_pool)}
        m = learned_blocks[tag]['metrics']
        print(f"{tag}: AP {m['ap']:.4f}, recall@N {m['recall_at_n']:.3f} "
              f"({model_notes[tag]['positives']}/{len(this_pool)} positive)")

    report = {'detector': 'v5 = learned fusion over frozen candidate pools '
                          '(TOPO-035, option 1 of TOPO-027)',
              'lineage': 'rank merges redistribute the deficit between '
                         'families (TOPO-016/026/028); this file calibrates '
                         'family scores with leave-one-segment-out models '
                         'over replayed pools; protocol declared in the '
                         'header before the first run',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'hyperparameters': {
                  'logit': {'l2': LOGIT_L2, 'iters': LOGIT_ITERS,
                            'tol': LOGIT_TOL},
                  'boost': {'rounds': BOOST_ROUNDS, 'lr': BOOST_LR,
                            'leaf_l2': BOOST_LEAF_L2}},
              'models': model_notes,
              **learned_blocks['v5lu'],
              'ablation_v1_config': v1_block,
              'ablation_v3u_config': v3u_block,
              'ablation_v3uq_config': v3uq_block,
              'ablation_v5l_config': learned_blocks['v5l'],
              'ablation_v5b_config': learned_blocks['v5b'],
              'ablation_v5bu_config': learned_blocks['v5bu'],
              'ct_channel': {'threshold': threshold,
                             'candidates': len(ct_pool)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v1 (percentile)', v1_block),
                         ('v3u (percentile)', v3u_block),
                         ('v3uq (quota)', v3uq_block),
                         ('v5l (logit, v1 pool)', learned_blocks['v5l']),
                         ('v5lu (logit, union)', learned_blocks['v5lu']),
                         ('v5b (boost, v1 pool)', learned_blocks['v5b']),
                         ('v5bu (boost, union)', learned_blocks['v5bu'])):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + f", plausible={block['recall_on_locally_plausible']:.3f}")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
