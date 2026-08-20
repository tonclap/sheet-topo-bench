"""Coverage breakdown of the union pool's uncovered injections (TOPO-036).

After TOPO-035 the fusion stopped being the bottleneck: v5lu's recall@N
(0.853) sits next to the union pool's coverage ceiling (210/232 = 0.905 by
the credit rule). The remaining detector deficit splits into two disjoint
populations, and this script lays both out:

- **no-candidate injections** (22 expected): no pool candidate lies inside
  the grown rectangle at all — a candidate *generation* deficit that no
  fusion can recover;
- **below-N injections** (12 expected): covered by the pool but ranked
  below N by v5lu — the residual *ranking* deficit.

Everything replays offline from frozen artifacts (v2 checkpoints, ct probe
windows, detector_v5 report); no frozen detector or number is touched.
Regression gates abort the run if the replayed coverage ceilings do not
reproduce ABLATION_V5's 199/232 (v1 pool) and 210/232 (union pool), or if
the v5lu miss split does not reproduce the frozen report.

The one cheap-extension estimate computable without regeneration is the ct
channel: the probe's stored windows carry the raw per-node profiles, so the
threshold sweep re-runs the exact candidate path (evidence_cells ->
surplus > SURPLUS_MIN -> differenced_clusters) at each T and counts which
uncovered injections would gain a candidate inside their grown rectangle.
This is a visibility measurement (an upper bound on cheap coverage gain),
not a tuning: v6's constants are declared in its own protocol (TOPO-038).
The prox/rect/support floors are NOT estimable here — the checkpoints store
post-floor candidates only, so relaxing those floors requires regeneration
(exactly TOPO-038's background run); stated in the report rather than
guessed.

Usage (from oneshot/detector/):

    python coverage_breakdown.py --corpus ../../output/topo/corpus_paris4 \
        --probe-report ../../output/topo/probe_ct_paris4.json \
        --probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --v5-report ../../output/topo/detector_v5_paris4.json \
        --bands dev --report ../../output/topo/coverage_breakdown_paris4.json
"""
import argparse
import json
import os
import pickle
import sys

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import detect_v1 as v1                                                # noqa: E402
import detect_v4 as v4                                                # noqa: E402
import probe_ct                                                       # noqa: E402

# Sweep grid around the frozen T=80 (probe calibration: papyrus median 121,
# air median 39). Both directions: T gates the gap (peak1 < T) and the
# flanks (>= T) at once, so "weaker" is not a single direction a priori.
CT_SWEEP = (64.0, 72.0, 80.0, 88.0, 96.0)
SURPLUS_RELAXED = 0.25   # one relaxed-floor row at the frozen T


def grown_hit(r, row, col):
    """PROTOCOL §3 / detect_v5 §2: inside the rectangle grown 50% per axis."""
    growth_r = (r['row_hi'] - r['row_lo']) * 0.25
    growth_c = (r['col_hi'] - r['col_lo']) * 0.25
    return (r['row_lo'] - growth_r <= row < r['row_hi'] + growth_r
            and r['col_lo'] - growth_c <= col < r['col_hi'] + growth_c)


def covered_ids(pool, injections):
    by_seg = {}
    for r in injections:
        by_seg.setdefault(r['segment'], []).append(r)
    out = set()
    for seg, row, col, *_rest in pool:
        for r in by_seg.get(seg, ()):
            if r['id'] not in out and grown_hit(r, row, col):
                out.add(r['id'])
    return out


def ct_pool_at(records, by_id, threshold, surplus_min):
    """detect_v3.ct_candidates with threshold and surplus floor as knobs."""
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
            out.append((segment, top[0], best[top][0], 'ct', mass, 1.0))
    return out


def breakdown(rows):
    by_type = {t: sum(1 for r in rows if r['type'] == t) for t in 'SMH'}
    by_band = {}
    for r in rows:
        key = f"w{r['band'] * 10:03d}-{r['band'] * 10 + 9:03d}"
        by_band[key] = by_band.get(key, 0) + 1
    plausible = sum(1 for r in rows if r.get('plausible'))
    return {'n': len(rows), 'by_type': by_type,
            'by_band': dict(sorted(by_band.items())),
            'plausible': plausible}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--probe-report', required=True)
    parser.add_argument('--probe-windows', required=True)
    parser.add_argument('--v2-checkpoints', required=True)
    parser.add_argument('--v5-report', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)

    def in_scope(record):
        if args.bands == 'dev':
            return record['winding_low'] < 100
        if args.bands == 'heldout':
            return record['winding_low'] >= 100
        return True

    injections = [r for r in manifest['injections'] if in_scope(r)]
    by_id = {r['id']: r for r in injections}
    names = sorted({r['segment'] for r in injections})
    n = len(injections)

    # Pools, replayed exactly as detect_v5 does.
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
    ct_pool = ct_pool_at(records, by_id, frozen_t, probe_ct.SURPLUS_MIN)
    union = v4.union_pool(v1_pool, ct_pool)

    # Regression gates: the replayed ceilings must reproduce ABLATION_V5.
    v1_cov = covered_ids(v1_pool, injections)
    union_cov = covered_ids(union, injections)
    if args.bands == 'dev' and (len(v1_cov), len(union_cov)) != (199, 210):
        raise SystemExit(f'coverage regression FAILED: v1 {len(v1_cov)}/232, '
                         f'union {len(union_cov)}/232 vs frozen 199/210')
    print(f'coverage OK: v1 {len(v1_cov)}/{n}, union {len(union_cov)}/{n}')

    # v5lu miss split from the frozen report (rank is None or > N).
    with open(args.v5_report, encoding='utf-8') as f:
        v5 = json.load(f)
    ranks = v5['per_injection_rank']
    missed = {i for i, rank in ranks.items() if rank is None or rank > n}
    no_candidate = sorted(i for i in missed if i not in union_cov)
    below_n = sorted(i for i in missed if i in union_cov)
    if args.bands == 'dev' and (len(no_candidate), len(below_n)) != (22, 12):
        raise SystemExit(f'miss split regression FAILED: '
                         f'{len(no_candidate)} no-candidate / '
                         f'{len(below_n)} below-N vs frozen 22/12')
    uncovered_union = [by_id[i] for i in sorted(set(by_id) - union_cov)]
    uncovered_v1 = [by_id[i] for i in sorted(set(by_id) - v1_cov)]
    print(f'v5lu misses {len(missed)}: {len(no_candidate)} without any '
          f'candidate, {len(below_n)} covered but below N')

    # ct visibility sweep over the no-candidate injections.
    target = {i: by_id[i] for i in no_candidate}
    sweep = []
    grid = [(t, probe_ct.SURPLUS_MIN) for t in CT_SWEEP]
    grid.append((frozen_t, SURPLUS_RELAXED))
    for t, surplus_min in grid:
        cand = ct_pool_at(records, by_id, t, surplus_min)
        got = covered_ids(cand, list(target.values()))
        sweep.append({'threshold': t, 'surplus_min': surplus_min,
                      'ct_pool': len(cand),
                      'covered_of_uncovered': sorted(got),
                      'n_covered': len(got),
                      'by_type': {tt: sum(1 for i in got
                                          if target[i]['type'] == tt)
                                  for tt in 'SMH'}})
        print(f"ct T={t:5.1f} surplus>{surplus_min}: pool {len(cand):4d}, "
              f"covers {len(got):2d}/{len(target)} of the uncovered "
              f"({sweep[-1]['by_type']})")

    report = {
        'question': 'TOPO-036: composition of the union pool\'s uncovered '
                    'injections and the cheap-extension upper bound',
        'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
        'scoped_injections': n,
        'pool_sizes': {'v1': len(v1_pool), 'ct': len(ct_pool),
                       'union': len(union)},
        'coverage': {'v1': len(v1_cov), 'union': len(union_cov)},
        'v5lu_misses': {'total': len(missed),
                        'no_candidate': no_candidate,
                        'below_n': [{'id': i, 'rank': ranks[i],
                                     'type': by_id[i]['type'],
                                     'band': by_id[i]['band'],
                                     'plausible': by_id[i].get('plausible')}
                                    for i in below_n]},
        'uncovered_union': {
            'summary': breakdown(uncovered_union),
            'rows': [{'id': r['id'], 'type': r['type'], 'band': r['band'],
                      'segment': r['segment'],
                      'plausible': r.get('plausible'),
                      'rows': [r['row_lo'], r['row_hi']],
                      'cols': [r['col_lo'], r['col_hi']]}
                     for r in uncovered_union]},
        'uncovered_v1': {'summary': breakdown(uncovered_v1),
                         'ids': [r['id'] for r in uncovered_v1]},
        'ct_sweep': {'frozen_threshold': frozen_t,
                     'frozen_surplus_min': probe_ct.SURPLUS_MIN,
                     'rows': sweep},
        'not_estimable_offline': 'prox/rect/support floor relaxation — the '
                                 'v2 checkpoints store post-floor candidates '
                                 'only; requires regeneration (TOPO-038)'}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, rows in (('union-uncovered', uncovered_union),
                        ('v1-uncovered', uncovered_v1)):
        s = breakdown(rows)
        print(f"{label}: n={s['n']}, types {s['by_type']}, "
              f"plausible {s['plausible']}, bands {s['by_band']}")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
