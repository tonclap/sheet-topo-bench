"""Detector v6: the coverage iteration (TOPO-038) — relaxed generation floors,
vjump as its own family, rich cluster features, learned ranker against v5lu.

Why this file exists: after TOPO-035 the fusion stopped being the bottleneck.
v5lu's recall@N (0.853) sits next to the union pool's coverage ceiling
(210/232 = 0.905), and TOPO-036 laid the deficit out: 22 injections have no
candidate at all (17 M, 5 S, 14 plausible — the U-011 blind spot, merger in
prediction holes), 12 more are covered but ranked just below N (9 S, ranks
233–336 — the prox ghost's zone). Every generation floor below was set under
the *percentile* merge, where each noisy candidate diluted the ranking
(U-012, quotas paying S for H). V2 of TOPO-037 showed that tax is abolished:
even raw mass beat the percentile, and the calibrated ranker demotes noise
itself. So the floors can be declaredly relaxed, and the pool can grow.

**Protocol, declared 17.08.2026 (twenty-first session) BEFORE any run of this
file (the commit adding this file precedes the first run; every constant is
fixed here, none is tuned on this file's output):**

1. **Relaxed floors, symmetric by construction.** Each floor relaxes on the
   corrupted grid and its pristine atlas alike — the differencing principle
   (same detector on both grids) is preserved, only the gate moves:
   - PROX_FLOOR 0.5 -> 0.25 (evidence = contact below ~7.5 vx, was ~5);
   - SINGLE_ROW_DAMP 0.25 -> 0.5;
   - MAX_CLUSTER_ROWS 14 -> 20 (an injection is <= ~11 rows by PROTOCOL §4;
     20 still bounds the tall natural zones that motivated the cap);
   - support surplus floor 0.5 -> 0.25;
   - ct threshold 80 -> 96 (TOPO-036's visibility sweep: relaxation works
     *upward* — the dark-gap gate dominates; T=96 reaches 6 of the 22
     uncovered. Dev-informed, declared here before the run; the ct surplus
     floor stays 0.5 — its relaxation measured zero effect);
   - the frozen detectors (detect_v1/v2/v3/v4/v5) are NOT edited; v6
     overrides the three module constants at runtime and carries its own
     copy of the segment channel pass for the two floors that are inlined
     literals there (support surplus, and the per-candidate stats below).
2. **Vjump becomes its own family** (was: a prox multiplier only). Node
   evidence where the inter-row radial jump exceeds VJUMP_REL_FLOOR = 3.0
   times the segment median (cluster_vjump's own "jumped arc" scale, U-010),
   evidence value = the ratio; pooled into the standard (row, block) cells,
   differenced against the pristine grid's own vjump atlas, standard cluster
   pass. Offline from the grids — no prediction involved: this channel
   cannot be blinded by prediction holes, which is where the 22 live.
3. **Rich cluster features ride the checkpoint.** Every candidate is a
   9-tuple (seg, row, col, family, mass, factor, n_cells, row_span,
   density): cluster size, row extent, and mass density (for rect: number
   of spans, bbox rows, bbox fill) — TOPO-027 named density, the v2
   checkpoints never stored it.
4. **Features — 14 per candidate:** family one-hot (prox, rect, support,
   ct, vjump), per-family log1p(mass), the vjump factor, log1p(n_cells),
   log1p(row_span), density. No percentiles, no positions, no segment
   identity.
5. **Models and splits are v5's, unchanged:** L2 logit (lambda=1.0, IRLS
   100 iters, tol 1e-10) and depth-1 Newton stump boosting (200 rounds,
   lr 0.1, leaf L2 1.0), leave-one-segment-out over the dev segments,
   ranking by out-of-fold probability, ties by (segment, row, col),
   top 4N. Bootstrap: 2000 resamples, seed 20260815 (v4/v5's).
6. **Configurations:** v6l / v6b (logit / boost, full pool, 14 features)
   are the two candidate forms. Two attribution rows are diagnostics, not
   shippable forms: v6l_novjump (pool without the vjump family) isolates
   the new channel, v6l_thin (11 features: one-hots, masses, factor)
   isolates the rich features.
7. **Ship rule against v5lu:** a form is a candidate for any future exam
   only if, against the replayed v5lu on paired bootstrap, AP or M improves
   significantly AND none of AP / recall@N / S / M / H / plausible degrades
   significantly. Among passing forms the max-AP one is fixed in
   ABLATION_V6.md. The frozen form v5lu and its TOPO-025 exam claim do NOT
   mutate; a successful v6 gets its own exam from the remaining held-out
   budget by a separate owner decision. This file never touches held-out.
8. **Two regression gates abort the run:**
   - *harness gate*: v5lu replayed exactly as detect_v5 does (v2
     checkpoints, frozen ct at T=80, union pool, LOSO logit) must
     reproduce the frozen detector_v5 report's AP and recall@N to 6
     decimals;
   - *generation gate*: the v6 generation pass run at the FROZEN floors on
     the first dev segment must reproduce that segment's v2 checkpoint
     candidates (prox/rect/support, 6-tuple projection) exactly — proving
     the copied channel pass is the frozen one until the floors move.
9. **The new pool's coverage ceiling** (credit rule of coverage_breakdown)
   is measured and published beside recall, total and per family.

Usage (from pipeline/detector/):

    python detect_v6.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --probe-report ../../output/topo/probe_ct_paris4.json \
        --probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --v5-report ../../output/topo/detector_v5_paris4.json \
        --checkpoint ../../output/topo/ckpt_paris4_dev_v6 \
        --bands dev --report ../../output/topo/detector_v6_paris4.json
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
import detect_v4 as v4                                                # noqa: E402
import detect_v5 as v5                                                # noqa: E402
import probe_ct                                                       # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)

FAMILIES = ('prox', 'rect', 'support', 'ct', 'vjump')

# §1 — the relaxed floors.
PROX_FLOOR_V6 = 0.25
SINGLE_ROW_DAMP_V6 = 0.5
MAX_CLUSTER_ROWS_V6 = 20
SUPPORT_SURPLUS_MIN_V6 = 0.25
CT_THRESHOLD_V6 = 96.0
CT_SURPLUS_MIN_V6 = 0.5
# §2 — the vjump family's floor, from cluster_vjump's own scale.
VJUMP_REL_FLOOR = 3.0

FROZEN_FLOORS = {'PROX_FLOOR': 0.5, 'SINGLE_ROW_DAMP': 0.25,
                 'MAX_CLUSTER_ROWS': 14}
RELAXED_FLOORS = {'PROX_FLOOR': PROX_FLOOR_V6,
                  'SINGLE_ROW_DAMP': SINGLE_ROW_DAMP_V6,
                  'MAX_CLUSTER_ROWS': MAX_CLUSTER_ROWS_V6}


def set_floors(floors):
    """Override the frozen module's generation floors at runtime (§1)."""
    for key, value in floors.items():
        setattr(v1, key, value)


def cluster_stats(cells, mass):
    rows = [r for r, _ in cells]
    n = len(cells)
    return n, max(rows) - min(rows) + 1, mass / n if n else 0.0


def segment_channels_v6(name, grid, pristine, centre, support,
                        support_surplus_min):
    """detect_v1.segment_channels with the support surplus floor as a knob
    and per-candidate cluster stats (§3); the channel pass is otherwise a
    faithful copy — the generation gate (§8) proves it at the frozen values.
    Emits 9-tuples; no front channel (dropped from every pool since v5).
    """
    atlas, _, _natural, _total = v1.strong_cells(pristine, centre)
    row_scores = {}
    evidence, best = v1.strong_cells(grid, centre, row_scores)[:2]

    r = v1.radial_map(grid, centre)
    vjump = np.abs(np.diff(r, axis=0))
    med = (float(np.nanmedian(vjump))
           if np.isfinite(vjump).any() else 0.0)
    out = []
    for cells, mass, top in v1.differenced_clusters(evidence, best, atlas):
        if cells is None:
            continue
        factor = v1.cluster_vjump(cells, vjump, med)
        n_cells, row_span, density = cluster_stats(cells, mass)
        out.append((name, top[0], best[top][0], 'prox', mass, factor,
                    n_cells, row_span, density))

    pristine_invalid = (pristine[..., 0] == -1) | (pristine[..., 1] == -1)
    for row, col, score, spans in v1.hole_components(grid):
        area = sum(hi - lo for _, lo, hi in spans)
        old = sum(int(pristine_invalid[rr, lo:hi].sum())
                  for rr, lo, hi in spans)
        if old * 2 > area:
            continue
        span_rows = max(rr for rr, _, _ in spans) - min(
            rr for rr, _, _ in spans) + 1
        bbox = span_rows * (max(hi for _, _, hi in spans)
                            - min(lo for _, lo, _ in spans))
        fill = area / bbox if bbox else 0.0
        out.append((name, row, col, 'rect', score, 1.0,
                    len(spans), span_rows, fill))

    prediction, z_quantiles, threshold = support
    evidence_s, best_s = v1.support_cells(grid, prediction, z_quantiles,
                                          threshold)
    atlas_s, _ = v1.support_cells(pristine, prediction, z_quantiles,
                                  threshold)
    surplus = {key: value - atlas_s.get(key, 0.0)
               for key, value in evidence_s.items()
               if value - atlas_s.get(key, 0.0) > support_surplus_min}
    for cells, mass, top in v1.differenced_clusters(surplus, best_s, {}):
        if cells is None:
            continue
        n_cells, row_span, density = cluster_stats(cells, mass)
        out.append((name, top[0], best_s[top][0], 'support', mass, 1.0,
                    n_cells, row_span, density))
    return out, vjump, med


def vjump_candidates(name, vjump_c, med_c, pristine, centre):
    """§2: the standalone vjump family. Evidence where the inter-row radial
    jump exceeds VJUMP_REL_FLOOR x the segment median (each grid against its
    own median), differenced against the pristine grid's own vjump atlas,
    standard cluster pass. The candidate row is the gap index (between grid
    rows i and i+1) — within the hit rule's grown rectangle for any window
    the jump borders. Prediction-free by construction."""
    if med_c <= 0:
        return []
    r_p = v1.radial_map(pristine, centre)
    vjump_p = np.abs(np.diff(r_p, axis=0))
    med_p = (float(np.nanmedian(vjump_p))
             if np.isfinite(vjump_p).any() else 0.0)

    def cells_of(vj, med):
        evidence, best = {}, {}
        if med <= 0:
            return evidence, best
        ratio = vj / med
        rows, cols = np.where(np.isfinite(ratio)
                              & (ratio > VJUMP_REL_FLOOR))
        for row, col in zip(rows, cols):
            s = float(ratio[row, col])
            key = (int(row), int(col) // v1.BLOCK)
            evidence[key] = evidence.get(key, 0.0) + s
            if s > best.get(key, (None, -1.0))[1]:
                best[key] = (int(col), s)
        return evidence, best

    evidence, best = cells_of(vjump_c, med_c)
    atlas, _ = cells_of(vjump_p, med_p)
    out = []
    for cells, mass, top in v1.differenced_clusters(evidence, best, atlas):
        if cells is None:
            continue
        n_cells, row_span, density = cluster_stats(cells, mass)
        out.append((name, top[0], best[top][0], 'vjump', mass,
                    best[top][1], n_cells, row_span, density))
    return out


def ct_candidates_v6(records, by_id, threshold, surplus_min):
    """The ct family at the relaxed threshold, from the probe's stored
    windows (the same offline path v3/v5 replay), with cluster stats."""
    out = []
    for rec in records:
        evidence, best = probe_ct.evidence_cells(rec['corrupted'], threshold)
        atlas, _ = probe_ct.evidence_cells(rec['pristine'], threshold)
        surplus = {key: value - atlas.get(key, 0.0)
                   for key, value in evidence.items()
                   if value - atlas.get(key, 0.0) > surplus_min}
        segment = by_id[rec['id']]['segment']
        for cells, mass, top in v1.differenced_clusters(surplus, best, {}):
            if cells is None:
                continue
            n_cells, row_span, density = cluster_stats(cells, mass)
            out.append((segment, top[0], best[top][0], 'ct', mass, 1.0,
                        n_cells, row_span, density))
    return out


def feature_matrix_v6(pool, thin=False):
    """§4's fourteen features (§6's thin variant: the first eleven)."""
    d = 11 if thin else 14
    X = np.zeros((len(pool), d))
    for i, c in enumerate(pool):
        _seg, _row, _col, fam, mass, factor, n_cells, row_span, density = c
        j = FAMILIES.index(fam)
        X[i, j] = 1.0
        X[i, 5 + j] = np.log1p(mass)
        X[i, 10] = factor
        if not thin:
            X[i, 11] = np.log1p(n_cells)
            X[i, 12] = np.log1p(row_span)
            X[i, 13] = density
    return X


def crossval_scores_v6(pool, y, fit, predict, thin=False):
    """§5: leave-one-segment-out, v5's split rule over 9-tuple pools."""
    X = feature_matrix_v6(pool, thin=thin)
    scores = np.zeros(len(pool))
    for seg in sorted({c[0] for c in pool}):
        test = np.array([c[0] == seg for c in pool])
        model = fit(X[~test], y[~test])
        scores[test] = predict(model, X[test])
    return scores


def grown_hit(r, row, col):
    growth_r = (r['row_hi'] - r['row_lo']) * 0.25
    growth_c = (r['col_hi'] - r['col_lo']) * 0.25
    return (r['row_lo'] - growth_r <= row < r['row_hi'] + growth_r
            and r['col_lo'] - growth_c <= col < r['col_hi'] + growth_c)


def covered_ids(pool, injections):
    by_seg = {}
    for r in injections:
        by_seg.setdefault(r['segment'], []).append(r)
    out = set()
    for c in pool:
        seg, row, col = c[0], c[1], c[2]
        for r in by_seg.get(seg, ()):
            if r['id'] not in out and grown_hit(r, row, col):
                out.add(r['id'])
    return out


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
    parser.add_argument('--checkpoint', required=True)
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
        raise SystemExit('detect_v6 dev iteration supports the turn-shift '
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

    # Shared: pristine grids, zones, broken ERL, the judging closure.
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

    # §8 harness gate: v5lu replayed must reproduce the frozen v5 report.
    ckpt_key_v2 = {'corpus': os.path.abspath(args.corpus),
                   'bands': args.bands, 'detector': 'v2'}
    v2_payloads = {}
    pool = []
    for name in names:
        path = os.path.join(args.v2_checkpoints, f'{name}.pkl')
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') != ckpt_key_v2:
            raise SystemExit(f'{path}: checkpoint key mismatch — refusing to '
                             f'replay candidates from a different run')
        v2_payloads[name] = payload
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

    import detect_v3 as v3
    ct_frozen = v3.ct_candidates(records, by_id, frozen_t)
    union = v4.union_pool(v1_pool, ct_frozen)
    y_u = v5.label_vector(union, injections)
    scores_u = v5.crossval_scores(union, y_u, v5.fit_logit, v5.predict_logit)
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

    prediction = scrolls.open_prediction(scroll, args.cache, max_chunks=256)
    support = (prediction, manifest['z_quantiles'], scroll.threshold)

    # §8 generation gate: frozen floors on the first dev segment must
    # reproduce its v2 checkpoint candidates (6-tuple projection).
    os.makedirs(args.checkpoint, exist_ok=True)
    gate_marker = os.path.join(args.checkpoint, 'generation_gate.ok')
    if not os.path.exists(gate_marker):
        gate_name = names[0]
        set_floors(FROZEN_FLOORS)
        corrupted = np.load(os.path.join(args.corpus, 'grids',
                                         f'{gate_name}.npy'))
        gate_out, _vj, _med = segment_channels_v6(
            gate_name, corrupted, pristine[gate_name], centre, support,
            support_surplus_min=0.5)
        got = [c[:6] for c in gate_out]
        want = [tuple(c) for c in v2_payloads[gate_name]['candidates']
                if c[3] in ('prox', 'rect', 'support')]
        if got != want:
            raise SystemExit(
                f'generation gate FAILED on {gate_name}: frozen-floor '
                f'candidates diverge from the v2 checkpoint '
                f'({len(got)} vs {len(want)}) — the copied channel pass '
                f'is not the frozen one')
        with open(gate_marker, 'w', encoding='utf-8') as f:
            f.write('generation gate passed: frozen-floor candidates equal '
                    'the v2 checkpoint on ' + gate_name + '\n')
        print(f'generation gate OK on {gate_name}: '
              f'{len(got)} candidates identical', flush=True)
    else:
        print('generation gate: marker present, skipping', flush=True)

    # The v6 generation pass (relaxed floors), per-segment checkpoints.
    set_floors(RELAXED_FLOORS)
    ckpt_key_v6 = {'corpus': os.path.abspath(args.corpus),
                   'bands': args.bands, 'detector': 'v6',
                   'floors': dict(RELAXED_FLOORS,
                                  SUPPORT_SURPLUS_MIN=SUPPORT_SURPLUS_MIN_V6,
                                  VJUMP_REL_FLOOR=VJUMP_REL_FLOOR)}
    gen_pool = []
    for name in names:
        ckpt = os.path.join(args.checkpoint, f'{name}.pkl')
        payload = None
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('key') != ckpt_key_v6:
                payload = None
        resumed = payload is not None
        if payload is None:
            corrupted = np.load(os.path.join(args.corpus, 'grids',
                                             f'{name}.npy'))
            seg_candidates, vjump_c, med_c = segment_channels_v6(
                name, corrupted, pristine[name], centre, support,
                support_surplus_min=SUPPORT_SURPLUS_MIN_V6)
            seg_candidates += vjump_candidates(name, vjump_c, med_c,
                                               pristine[name], centre)
            payload = {'key': ckpt_key_v6, 'candidates': seg_candidates}
            tmp = f'{ckpt}.{os.getpid()}.part'
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f)
            os.replace(tmp, ckpt)
        gen_pool += [tuple(c) for c in payload['candidates']]
        print(f'{name}: {len(gen_pool)} generated candidates so far'
              + (' (from checkpoint)' if resumed else ''), flush=True)

    ct_pool = ct_candidates_v6(records, by_id, CT_THRESHOLD_V6,
                               CT_SURPLUS_MIN_V6)
    full_pool = gen_pool + ct_pool
    fam_sizes = {fam: sum(1 for c in full_pool if c[3] == fam)
                 for fam in FAMILIES}
    print(f'v6 pool: {len(full_pool)} candidates {fam_sizes}', flush=True)

    # §9 coverage ceilings by the credit rule.
    cov_total = covered_ids(full_pool, injections)
    coverage = {'total': len(cov_total), 'n': n,
                'v5_union_reference': len(covered_ids(union, injections)),
                'by_family': {fam: len(covered_ids(
                    [c for c in full_pool if c[3] == fam], injections))
                    for fam in FAMILIES}}
    print(f"v6 coverage ceiling: {coverage['total']}/{n} "
          f"(v5 union {coverage['v5_union_reference']}/{n}); "
          f"by family {coverage['by_family']}", flush=True)

    # §6 configurations.
    y = v5.label_vector(full_pool, injections)
    pool_novjump = [c for c in full_pool if c[3] != 'vjump']
    y_nv = v5.label_vector(pool_novjump, injections)
    blocks, notes = {}, {}
    for tag, this_pool, this_y, fit, predict, thin in (
            ('v6l', full_pool, y, v5.fit_logit, v5.predict_logit, False),
            ('v6b', full_pool, y, v5.fit_boost, v5.predict_boost, False),
            ('v6l_novjump', pool_novjump, y_nv,
             v5.fit_logit, v5.predict_logit, False),
            ('v6l_thin', full_pool, y, v5.fit_logit, v5.predict_logit,
             True)):
        scores = crossval_scores_v6(this_pool, this_y, fit, predict,
                                    thin=thin)
        blocks[tag] = judged(
            v5.ranking_from_scores(this_pool, scores, top=4 * n))
        notes[tag] = {'candidates': len(this_pool),
                      'positives': int(this_y.sum()),
                      'features': 11 if thin else 14,
                      'model': 'boost' if tag == 'v6b' else 'logit'}
        m = blocks[tag]['metrics']
        print(f"{tag}: AP {m['ap']:.4f}, recall@N {m['recall_at_n']:.3f} "
              f"({notes[tag]['positives']}/{len(this_pool)} positive)",
              flush=True)

    report = {'detector': 'v6 = coverage iteration: relaxed generation '
                          'floors + vjump family + rich cluster features, '
                          'learned ranker against v5lu (TOPO-038)',
              'lineage': 'the pool-growth tax was abolished by TOPO-035/037 '
                         '(V2: raw mass beat the percentile); TOPO-036 laid '
                         'out the deficit (17/22 uncovered are M, ct relax '
                         'works upward); protocol declared in the header '
                         'before the first run',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'floors': dict(RELAXED_FLOORS,
                             SUPPORT_SURPLUS_MIN=SUPPORT_SURPLUS_MIN_V6,
                             CT_THRESHOLD=CT_THRESHOLD_V6,
                             CT_SURPLUS_MIN=CT_SURPLUS_MIN_V6,
                             VJUMP_REL_FLOOR=VJUMP_REL_FLOOR),
              'pool_sizes': dict(fam_sizes, total=len(full_pool)),
              'coverage': coverage,
              'models': notes,
              **blocks['v6l'],
              'ablation_v5lu_config': v5lu_block,
              'ablation_v6b_config': blocks['v6b'],
              'ablation_v6l_novjump_config': blocks['v6l_novjump'],
              'ablation_v6l_thin_config': blocks['v6l_thin']}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v5lu (replayed)', v5lu_block),
                         ('v6l', blocks['v6l']),
                         ('v6b', blocks['v6b']),
                         ('v6l_novjump', blocks['v6l_novjump']),
                         ('v6l_thin', blocks['v6l_thin'])):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + f", plausible={block['recall_on_locally_plausible']:.3f}")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
