"""Held-out v2 exam scorer for v5lu (TOPO-025).

The exam's question is not "does the detector transfer" — form v1 answered
that in the v1 exam (FREEZE_2026-08-14.md) — but "does the *dev gain* of the
learned fusion transfer": v5lu − v1 = +0.211 AP on dev (ABLATION_V5.md). The
composition of held-out v2, the training procedure and the reading rule are
declared in the PROTOCOL §6 insert of 19.08.2026; the commit adding that
insert (and this file) precedes corpus generation and any evaluation run.

What this file does, in order:

1. **Dev gate before held-out.** Rebuilds the dev union pool exactly as
   detect_v5 does (v2 checkpoints replayed with key checks, probe windows,
   v4.union_pool) and replays the LOSO v5lu evaluation; unless AP and
   recall@N reproduce the frozen ``detector_v5_paris4.json`` to 6 decimals,
   the run aborts without having opened a single held-out artefact.
2. **Exam model.** v5's ``fit_logit`` (hyperparameters byte-identical) on the
   FULL dev pool — the training fold is the whole dev corpus, features
   standardised on it. No LOSO in the exam application, no training or
   calibration on held-out (PROTOCOL §6 insert).
3. **Held-out gate.** Loads the exam corpus pool the same way and replays the
   v1 percentile ranking; unless it reproduces the exam corpus's own
   detect_v1 report to 6 decimals, the run aborts (pool identity broken).
4. **Scoring.** The dev-trained model scores the exam pool; ranking is by
   descending probability, ties by (segment, row, col), top 4N; judged by
   the v1 harness (sheet_erl), with PROTOCOL §3's ERL comparators (oracle,
   random windows) computed like detect_v1 does. The report is written in
   detect_v1's schema with ``ablation_v1_config`` attached (detect_v5's
   convention), so the paired bootstrap of the transfer question — the
   declared reading rule — reads both sides from this one file.

Also published here (declared expectation 2 of the insert): the ct-probe
mute share — the fraction of scoped injections whose probe window contains
zero probed pristine nodes (rows without a counting centre are mute).

Usage (from oneshot/detector/), exam corpus B2 shown:

    python exam_v5lu.py \
        --dev-corpus ../../output/topo/corpus_paris4 \
        --dev-v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --dev-probe-report ../../output/topo/probe_ct_paris4.json \
        --dev-probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --dev-v5-report ../../output/topo/detector_v5_paris4.json \
        --corpus ../../output/topo/corpus_0139_h2 --bands all \
        --v2-checkpoints ../../output/topo/ckpt_0139_h2_v2 \
        --probe-report ../../output/topo/probe_ct_0139_h2.json \
        --probe-windows ../../output/topo/probe_ct_0139_h2_windows.jsonl \
        --v1-report ../../output/topo/detector_v1_0139_h2.json \
        --grid-cache ../../output/figgrids \
        --report ../../output/topo/exam_v5lu_0139_h2.json
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
import detect_v1 as v1                                                # noqa: E402
import detect_v3 as v3                                                # noqa: E402
import detect_v4 as v4                                                # noqa: E402
import detect_v5 as v5                                                # noqa: E402


def scoped_manifest(corpus, bands):
    with open(os.path.join(corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)

    def in_scope(record):
        if bands == 'dev':
            return record['winding_low'] < 100
        if bands == 'heldout':
            return record['winding_low'] >= 100
        return True

    injections = [r for r in manifest['injections'] if in_scope(r)]
    return manifest, dict(manifest, injections=injections), injections


def load_pool(corpus, bands, v2_checkpoints, probe_report_path,
              probe_windows_path, injections):
    """The (v1 pool, ct pool, union pool) triple, replayed exactly as
    detect_v5.main does — checkpoint key checks and the completeness check
    on probe windows included. Returns the pools and the ct mute share."""
    by_id = {r['id']: r for r in injections}
    names = sorted({r['segment'] for r in injections})
    ckpt_key = {'corpus': os.path.abspath(corpus), 'bands': bands,
                'detector': 'v2'}
    pool = []
    for name in names:
        path = os.path.join(v2_checkpoints, f'{name}.pkl')
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') != ckpt_key:
            raise SystemExit(f'{path}: checkpoint key mismatch — refusing to '
                             f'replay candidates from a different run')
        pool += [tuple(c) for c in payload['candidates']]
    v1_pool = [c for c in pool if c[3] in ('prox', 'rect', 'support')]

    with open(probe_report_path, encoding='utf-8') as f:
        probe_report = json.load(f)
    threshold = probe_report['calibration']['threshold']
    records, seen = [], set()
    with open(probe_windows_path, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec['id'] in seen or rec['id'] not in by_id:
                continue
            seen.add(rec['id'])
            records.append(rec)
    if len(records) != len(injections):
        raise SystemExit(f'probe windows hold {len(records)} of '
                         f'{len(injections)} scoped injections — the probe '
                         f'run is incomplete')
    mute = sum(1 for rec in records
               if not (rec.get('pristine') or {}).get('row'))
    ct_pool = v3.ct_candidates(records, by_id, threshold)
    union = v4.union_pool(v1_pool, ct_pool)
    return v1_pool, ct_pool, union, threshold, mute / len(records)


def make_judge(scoped, injections, grid_cache, row_step):
    """detect_v1's judged() plus its §3 ERL comparators, over this corpus."""
    scroll = scrolls.SCROLLS[scoped['scroll']]
    voxel_mm = scoped['voxel_mm']
    names = sorted({r['segment'] for r in injections})
    pristine = {name: scrolls.segment_grid(name, scroll, grid_cache)
                for name in names}
    zones = {name: sheet_erl.segment_zones(scoped, name) for name in names}
    erl_broken, *_ = sheet_erl.sheet_erl(pristine, zones, voxel_mm, row_step)
    n = len(injections)

    def judged(rank):
        result = sheet_erl.evaluate_ranking(rank, scoped, pristine, voxel_mm,
                                            row_step, erl_broken=erl_broken)
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

    def erl_context():
        oracle_rank = [(r['segment'], (r['row_lo'] + r['row_hi']) // 2,
                        (r['col_lo'] + r['col_hi']) // 2, 1.0)
                       for r in injections]
        oracle = sheet_erl.evaluate_ranking(oracle_rank, scoped, pristine,
                                            voxel_mm, row_step,
                                            erl_broken=erl_broken)
        rng = np.random.default_rng(20260815)
        random_deltas = sorted(
            sheet_erl.evaluate_ranking(
                sheet_erl.random_ranking(rng, pristine, n), scoped, pristine,
                voxel_mm, row_step, erl_broken=erl_broken)['delta_erl_mm']
            for _ in range(10))
        return {
            'delta_erl_oracle_mm': oracle['delta_erl_mm'],
            'oracle_recall_at_n': oracle['recall_at_n'],
            'delta_erl_random_mm_median10':
                random_deltas[len(random_deltas) // 2],
            'delta_erl_random_mm_seeds': [round(d, 4) for d in random_deltas],
            'note': 'delta_erl_mm values are corpus-wide (all injected '
                    'segments, one number), against erl_broken_mm of the '
                    'same corpus; the normalised reading is '
                    'delta/oracle_delta',
        }

    return judged, erl_context


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dev-corpus', required=True)
    parser.add_argument('--dev-v2-checkpoints', required=True)
    parser.add_argument('--dev-probe-report', required=True)
    parser.add_argument('--dev-probe-windows', required=True)
    parser.add_argument('--dev-v5-report', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        required=True)
    parser.add_argument('--v2-checkpoints', required=True)
    parser.add_argument('--probe-report', required=True)
    parser.add_argument('--probe-windows', required=True)
    parser.add_argument('--v1-report', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    # ---- 1. Dev pool + gate 1 (before any held-out artefact is opened) ----
    _, dev_scoped, dev_injections = scoped_manifest(args.dev_corpus, 'dev')
    dev_v1_pool, dev_ct_pool, dev_union, dev_threshold, _dev_mute = load_pool(
        args.dev_corpus, 'dev', args.dev_v2_checkpoints,
        args.dev_probe_report, args.dev_probe_windows, dev_injections)
    print(f'dev pools: v1 {len(dev_v1_pool)}, ct {len(dev_ct_pool)}, '
          f'union {len(dev_union)}', flush=True)
    dev_judged, _ = make_judge(dev_scoped, dev_injections, args.grid_cache,
                               args.row_step)
    y_dev = v5.label_vector(dev_union, dev_injections)
    loso_scores = v5.crossval_scores(dev_union, y_dev, v5.fit_logit,
                                     v5.predict_logit)
    n_dev = len(dev_injections)
    loso_block = dev_judged(v5.ranking_from_scores(dev_union, loso_scores,
                                                   top=4 * n_dev))
    with open(args.dev_v5_report, encoding='utf-8') as f:
        frozen = json.load(f)
    ours, theirs = loso_block['metrics'], frozen['metrics']
    if (round(ours['ap'], 6) != round(theirs['ap'], 6)
            or round(ours['recall_at_n'], 6)
            != round(theirs['recall_at_n'], 6)):
        raise SystemExit(
            f"gate 1 FAILED: dev LOSO v5lu replay AP {ours['ap']:.6f} / "
            f"recall {ours['recall_at_n']:.6f} vs frozen {theirs['ap']:.6f} "
            f"/ {theirs['recall_at_n']:.6f}")
    print(f"gate 1 OK: dev LOSO v5lu AP {ours['ap']:.4f} == frozen "
          f"{theirs['ap']:.4f}", flush=True)

    # ---- 2. Exam model: full-dev training fold (PROTOCOL §6 insert) ----
    model = v5.fit_logit(v5.feature_matrix(dev_union), y_dev)
    print(f'exam model trained on the full dev pool: '
          f'{int(y_dev.sum())}/{len(dev_union)} positive', flush=True)

    # ---- 3. Exam pool + gate 2 ----
    _, scoped, injections = scoped_manifest(args.corpus, args.bands)
    n = len(injections)
    v1_pool, ct_pool, union, threshold, ct_mute = load_pool(
        args.corpus, args.bands, args.v2_checkpoints, args.probe_report,
        args.probe_windows, injections)
    print(f'exam pools: v1 {len(v1_pool)}, ct {len(ct_pool)}, '
          f'union {len(union)}; ct-probe mute share {ct_mute:.3f}',
          flush=True)
    judged, erl_context = make_judge(scoped, injections, args.grid_cache,
                                     args.row_step)
    v1_block = judged(v1.merge_channels(v1_pool, top=4 * n))
    with open(args.v1_report, encoding='utf-8') as f:
        v1_report = json.load(f)
    ours, theirs = v1_block['metrics'], v1_report['metrics']
    if (round(ours['ap'], 6) != round(theirs['ap'], 6)
            or round(ours['recall_at_n'], 6)
            != round(theirs['recall_at_n'], 6)):
        raise SystemExit(
            f"gate 2 FAILED: exam v1 replay AP {ours['ap']:.6f} / recall "
            f"{ours['recall_at_n']:.6f} vs detect_v1 report "
            f"{theirs['ap']:.6f} / {theirs['recall_at_n']:.6f}")
    print(f"gate 2 OK: exam v1 replay AP {ours['ap']:.4f} == detect_v1 "
          f"report {theirs['ap']:.4f}", flush=True)

    # ---- 4. Score, judge, write ----
    scores = v5.predict_logit(model, v5.feature_matrix(union))
    exam_block = judged(v5.ranking_from_scores(union, scores, top=4 * n))
    context = erl_context()

    report = {'detector': 'v5lu exam application (TOPO-025 held-out v2): '
                          'logit trained on the full dev union pool, '
                          'applied frozen to this corpus',
              'lineage': 'form fixed in ABLATION_V5.md (17.08); composition, '
                         'training and reading rule in the PROTOCOL §6 '
                         'insert of 19.08.2026, declared before generation '
                         'and before this run',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'hyperparameters': {'logit': {'l2': v5.LOGIT_L2,
                                            'iters': v5.LOGIT_ITERS,
                                            'tol': v5.LOGIT_TOL}},
              'training': {'corpus': os.path.abspath(args.dev_corpus),
                           'bands': 'dev',
                           'candidates': len(dev_union),
                           'positives': int(y_dev.sum())},
              'pools': {'v1': len(v1_pool), 'ct': len(ct_pool),
                        'union': len(union)},
              'ct_probe': {'threshold': threshold,
                           'mute_window_share': round(ct_mute, 4)},
              **exam_block,
              'erl_context': context,
              'ablation_v1_config': v1_block}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v1 (replay)', v1_block), ('v5lu (exam)',
                                                     exam_block)):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + (f", plausible={block['recall_on_locally_plausible']:.3f}"
                 if block['recall_on_locally_plausible'] is not None else ''))
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
