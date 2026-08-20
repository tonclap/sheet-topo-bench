"""Detector v7: the cross-winding pair (H2.3) as ranker features (TOPO-040).

Why this file exists: TOPO-036 laid out v5lu's residual ranking deficit —
9 of the 12 covered-but-below-N misses are S (switches) at ranks 233-336,
right past the cutoff. The mechanism was declared in detect_v1 the day the
prox channel was born ("The ghost, declared up front"): contact is symmetric,
so when a trace jumps onto the neighbouring winding, BOTH arcs — the
corrupted one and its pristine twin one turn away — see d ~ 0, prox emits
both, and the false twin occupies a top-N slot. U-010 (the vjump factor)
already disambiguates them *individually*: the arc that moved is radially
displaced against its own rows, the ghost is not. What the ranker never saw
is the PAIR: that two candidates sit one winding apart at the same contact,
and which of the two carries the jump. After v6's floors failed (TOPO-038),
this is the main remaining ranking move, and it needs no regeneration —
everything runs on the frozen v5 pools.

**Protocol, declared 18.08.2026 (twenty-second session) BEFORE any run of
this file (the commit adding this file precedes the first run; every
constant is fixed here, none is tuned on this file's output):**

1. **Pools are v5lu's, replayed verbatim:** the v1 pool from the v2
   checkpoints, ct at the frozen T=80 folded into support by detect_v4's
   `union_pool`. No candidate is added, moved or removed — the pair is a
   FEATURE of existing candidates, not a channel.
2. **The pair rule (fixed, not tuned):** two prox candidates of the same
   segment form a pair iff |Δrow| <= PAIR_ROW_MAX = 4 and their winding
   coordinates differ by about one turn: |Δturn − 1| <= PAIR_TURN_TOL = 0.5.
   The winding coordinate of a candidate is read at its (row, col) from the
   corrupted grid's own `turn_profile` (the same walk prox scoring uses) —
   deterministic, mechanistic, no prediction. A candidate whose row cannot
   produce a turn coordinate (short row, outside the centre domain) is
   unpaired by definition.
3. **Features — v5's seven plus two, nothing else:**
   - `is_paired`: 1.0 iff the candidate has at least one partner (§2);
   - `pair_factor_margin`: own vjump factor minus the strongest partner's
     (0.0 if unpaired) — U-010's asymmetry made pairwise: the arc that
     moved carries the positive margin, its ghost the negative one.
   Both are 0.0 for every non-prox candidate.
4. **Model, splits, seed — v5's shipped form, unchanged:** L2 logit
   (lambda 1.0, IRLS 100, tol 1e-10), leave-one-segment-out, ranking by
   out-of-fold probability, ties by (segment, row, col), top 4N; bootstrap
   2000 resamples, seed 20260815 (the ablation summary's).
5. **Ship rule against v5lu (v6's §7, unchanged):** v7l is a candidate for
   any future exam only if, against the replayed v5lu on paired bootstrap,
   AP or M improves significantly AND none of AP / recall@N / S / M / H /
   plausible degrades significantly. The frozen form v5lu and its TOPO-025
   exam claim do NOT mutate; this file never touches held-out. The task's
   own target is S: the 9 below-N switches — the S column is read with the
   same discipline (significant gain / no significant loss), and outcome
   (b) of TOPO-040 — the pair does not separate — is a valid result.
6. **One regression gate aborts the run:** v5lu replayed (v2 checkpoints,
   frozen ct, union pool, LOSO logit) must reproduce the frozen detector_v5
   report's AP and recall@N to 6 decimals.
7. **Pair statistics are published:** number of prox candidates, paired
   candidates, pairs, and the label composition of pairs (both-positive /
   one-positive / both-negative) — the last is diagnostic output, computed
   AFTER the ranking, and feeds no decision inside this file.

Usage (from oneshot/detector/):

    python detect_v7.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --probe-report ../../output/topo/probe_ct_paris4.json \
        --probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --v5-report ../../output/topo/detector_v5_paris4.json \
        --bands dev --report ../../output/topo/detector_v7_paris4.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v3 as v3                                                # noqa: E402
import detect_v4 as v4                                                # noqa: E402
import detect_v5 as v5                                                # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)

# §2 — the pair rule's two constants.
PAIR_ROW_MAX = 4
PAIR_TURN_TOL = 0.5


def candidate_turns(pool, corpus, scroll, centre):
    """§2: winding coordinate of every prox candidate, read at (row, col)
    from the corrupted grid's own turn_profile. None where the row is mute."""
    by_seg = {}
    for i, c in enumerate(pool):
        if c[3] == 'prox':
            by_seg.setdefault(c[0], []).append(i)
    turns = {}
    for seg, idxs in sorted(by_seg.items()):
        grid = np.load(os.path.join(corpus, 'grids', f'{seg}.npy'))
        cache = {}
        for i in idxs:
            row, col = pool[i][1], pool[i][2]
            if row not in cache:
                mask = (grid[row, :, 0] != -1) & (grid[row, :, 1] != -1)
                cols = np.where(mask)[0]
                if len(cols) < 30:
                    cache[row] = (cols, None)
                else:
                    try:
                        _points, row_turns, _ = scrolls.turn_profile(
                            grid, row, mask, centre)
                        cache[row] = (cols, row_turns)
                    except Exception:
                        cache[row] = (cols, None)
            cols, row_turns = cache[row]
            if row_turns is None:
                continue
            at = np.where(cols == col)[0]
            if len(at):
                turns[i] = float(row_turns[at[0]])
    return turns


def pair_features(pool, turns):
    """§3: is_paired and pair_factor_margin per candidate; §2's rule."""
    partners = {i: [] for i in range(len(pool))}
    by_seg = {}
    for i, c in enumerate(pool):
        if c[3] == 'prox' and i in turns:
            by_seg.setdefault(c[0], []).append(i)
    n_pairs = 0
    for _seg, idxs in sorted(by_seg.items()):
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if abs(pool[i][1] - pool[j][1]) > PAIR_ROW_MAX:
                    continue
                if abs(abs(turns[i] - turns[j]) - 1.0) > PAIR_TURN_TOL:
                    continue
                partners[i].append(j)
                partners[j].append(i)
                n_pairs += 1
    extra = np.zeros((len(pool), 2))
    for i, mates in partners.items():
        if not mates:
            continue
        extra[i, 0] = 1.0
        strongest = max(pool[j][5] for j in mates)
        extra[i, 1] = pool[i][5] - strongest
    return extra, n_pairs


def crossval_scores_v7(pool, extra, y):
    """§4: v5's LOSO logit over the 7+2 feature matrix."""
    X = np.hstack([v5.feature_matrix(pool), extra])
    scores = np.zeros(len(pool))
    for seg in sorted({c[0] for c in pool}):
        test = np.array([c[0] == seg for c in pool])
        model = v5.fit_logit(X[~test], y[~test])
        scores[test] = v5.predict_logit(model, X[test])
    return scores


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--probe-report', required=True)
    parser.add_argument('--probe-windows', required=True)
    parser.add_argument('--v2-checkpoints', required=True)
    parser.add_argument('--v5-report', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    if manifest.get('mechanism', 'turn-shift') != 'turn-shift':
        raise SystemExit('detect_v7 dev iteration supports the turn-shift '
                         'mechanism (Paris 4) only; any held-out is a '
                         'separate owner decision, not this file')

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
        per_rank = [None if h is None else injections[h]['id']
                    for h, *_rest in full]
        per_injection = {r['id']: None for r in injections}
        for i, (h, *_rest) in enumerate(full, 1):
            if h is not None:
                per_injection[injections[h]['id']] = i
        return {'metrics': result, 'recall_by_type': by_type,
                'recall_on_locally_plausible': recall_plausible,
                'per_rank_outcomes': per_rank,
                'per_injection_rank': per_injection}

    # §1 pools, §6 harness gate.
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
    frozen_t = probe_report['calibration']['threshold']
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
    ct_pool = v3.ct_candidates(records, by_id, frozen_t)
    union = v4.union_pool(v1_pool, ct_pool)

    y = v5.label_vector(union, injections)
    scores_u = v5.crossval_scores(union, y, v5.fit_logit, v5.predict_logit)
    v5lu_block = judged(v5.ranking_from_scores(union, scores_u, top=4 * n))
    with open(args.v5_report, encoding='utf-8') as f:
        frozen_v5 = json.load(f)
    ours, theirs = v5lu_block['metrics'], frozen_v5['metrics']
    if (round(ours['ap'], 6) != round(theirs['ap'], 6)
            or round(ours['recall_at_n'], 6)
            != round(theirs['recall_at_n'], 6)):
        raise SystemExit(
            f"v5lu harness gate FAILED: replayed AP {ours['ap']:.6f} / "
            f"recall {ours['recall_at_n']:.6f} vs frozen {theirs['ap']:.6f} "
            f"/ {theirs['recall_at_n']:.6f}")
    print(f"v5lu harness gate OK: AP {ours['ap']:.4f} == frozen "
          f"{theirs['ap']:.4f}", flush=True)

    # §2–§4: turns, pairs, features, LOSO ranking.
    turns = candidate_turns(union, args.corpus, scroll, centre)
    extra, n_pairs = pair_features(union, turns)
    n_prox = sum(1 for c in union if c[3] == 'prox')
    n_paired = int(extra[:, 0].sum())
    print(f'pairs: {n_prox} prox candidates, {len(turns)} with a turn '
          f'coordinate, {n_paired} paired in {n_pairs} pairs', flush=True)

    scores_p = crossval_scores_v7(union, extra, y)
    v7l_block = judged(v5.ranking_from_scores(union, scores_p, top=4 * n))
    m = v7l_block['metrics']
    print(f"v7l: AP {m['ap']:.4f}, recall@N {m['recall_at_n']:.3f}",
          flush=True)

    # §7 diagnostic: label composition of pairs (after the ranking).
    paired_idx = [i for i in range(len(union)) if extra[i, 0] > 0]
    seen_pairs, comp = set(), {'both_positive': 0, 'one_positive': 0,
                               'both_negative': 0}
    for i in paired_idx:
        for j in paired_idx:
            if j <= i or union[i][0] != union[j][0]:
                continue
            if abs(union[i][1] - union[j][1]) > PAIR_ROW_MAX:
                continue
            if (i in turns and j in turns
                    and abs(abs(turns[i] - turns[j]) - 1.0) <= PAIR_TURN_TOL
                    and (i, j) not in seen_pairs):
                seen_pairs.add((i, j))
                k = int(y[i]) + int(y[j])
                comp['both_positive' if k == 2 else
                     'one_positive' if k == 1 else 'both_negative'] += 1

    report = {'detector': 'v7 = cross-winding pair features over the frozen '
                          'v5lu pool (TOPO-040, H2.3)',
              'lineage': "TOPO-036: 9/12 below-N misses are S at ranks "
                         "233-336 — detect_v1's declared prox ghost; the "
                         "pair rule and both features are declared in the "
                         "header before the first run",
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'pair_rule': {'row_max': PAIR_ROW_MAX,
                            'turn_tol': PAIR_TURN_TOL},
              'pair_stats': {'prox_candidates': n_prox,
                             'with_turn': len(turns),
                             'paired': n_paired, 'pairs': n_pairs,
                             'label_composition': comp},
              **v7l_block,
              'ablation_v5lu_config': v5lu_block}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v5lu (replayed)', v5lu_block),
                         ('v7l (pair features)', v7l_block)):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + f", plausible={block['recall_on_locally_plausible']:.3f}")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
