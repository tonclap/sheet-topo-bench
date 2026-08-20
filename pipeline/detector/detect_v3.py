"""Detector v3 ablation: v1 +/- the CT-intensity channel (TOPO-026).

The probe (probe_ct.py) came back positive by its pre-declared rule: CT
clusters cover 13/36 of the M windows v1 misses on dev (bar was 12), 0/77 of
H windows, with the threshold T calibrated on the pristine substrate before
any window was read. This file builds the channel and answers the only
question that matters after TOPO-023: does adding it move the *ranking*, or
does merge dilution eat the mechanism again?

**No new sampling happens here.** Outside an injected window the corrupted
grid is bit-identical to the pristine grid, CT sampling is deterministic, so
the per-cell surplus is identically zero there — every CT candidate the full
detect_v1-style scope could produce lives inside the probe's stored windows.
The channel's candidates are therefore computed *offline* from the probe's
JSONL (which stored corrupted and pristine node probes for all 232 dev
windows), and the v1 pool is replayed from the detect_v2 checkpoints whose v1
configuration already reproduced the frozen report bit-for-bit. The same
regression gate is applied again here: if the replayed v1 configuration does
not match the frozen detector_v1 report to 6 decimals, the run aborts.

Configurations from one candidate pool, mirroring detect_v2's design:

- ``v1``     prox + rect + support (the regression anchor);
- ``v3``     ct replaces support — the hypothesis-shaped form: ct sees the
             merger's seat *in the scan itself*, support saw it through the
             prediction and dragged prediction-hole false mass along;
- ``v3add``  all four — the dilution-risk form U-012 warns about.

Declared before the numbers: the deliverable is the paired-bootstrap table
(ablation_summary_v3.py); the channel ships only if a configuration moves M
or AP significantly (paired interval clear of zero) without a significant
loss elsewhere. TOPO-025's held-out budget stays untouched either way.

Usage (from pipeline/detector/):

    python detect_v3.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids \
        --probe-report ../../output/topo/probe_ct_paris4.json \
        --probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --v1-report ../../output/topo/detector_v1_paris4.json \
        --bands dev --report ../../output/topo/detector_v3_paris4.json
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
import probe_ct                                                       # noqa: E402


def ct_candidates(records, by_id, threshold):
    """The CT channel's pooled candidates, from the probe's stored windows.

    Same shape as every v1 channel: per-cell surplus of corrupted over
    pristine evidence, SURPLUS_MIN floor, the standard cluster pass. Windows
    are disjoint by the injector's construction (occupied spans), so
    concatenating per-window candidates is the segment-level result.
    """
    out = []
    for rec in records:
        evidence, best = probe_ct.evidence_cells(rec['corrupted'], threshold)
        atlas, _ = probe_ct.evidence_cells(rec['pristine'], threshold)
        surplus = {key: value - atlas.get(key, 0.0)
                   for key, value in evidence.items()
                   if value - atlas.get(key, 0.0) > probe_ct.SURPLUS_MIN}
        segment = by_id[rec['id']]['segment']
        for cells, mass, top in v1.differenced_clusters(surplus, best, {}):
            if cells is None:
                continue
            out.append((segment, top[0], best[top][0], 'ct', mass, 1.0))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--probe-report', required=True)
    parser.add_argument('--probe-windows', required=True)
    parser.add_argument('--v2-checkpoints', required=True)
    parser.add_argument('--v1-report', required=True)
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

    # v1 pool, replayed from the v2 checkpoints (front dropped).
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

    # CT channel from the probe's stored windows, at the probe's frozen T.
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
    ct_pool = ct_candidates(records, by_id, threshold)
    print(f'pools: v1 {len(v1_pool)} candidates '
          f'({len(pool) - len(v1_pool)} front dropped), ct {len(ct_pool)}')

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

    def config_block(config_pool):
        return judged(v1.merge_channels(config_pool, top=4 * n))

    v1_block = config_block(v1_pool)
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

    v3_block = config_block([c for c in v1_pool if c[3] != 'support']
                            + ct_pool)
    v3add_block = config_block(v1_pool + ct_pool)

    # v3u — the "gate/weight inside the channel" form the task text names for
    # exactly this outcome (mechanism works, an extra rank family dilutes):
    # ct candidates join the *support* channel instead of forming their own.
    # A ct cluster within one cell of a support cluster corroborates it (mass
    # summed — both masses are node counts, same unit); an unmatched ct
    # cluster enters as a support-channel candidate. Three channels remain,
    # so the merge geometry v1 was tuned on is untouched.
    def union_pool():
        support = [list(c) for c in v1_pool if c[3] == 'support']
        rest = [c for c in v1_pool if c[3] != 'support']
        for seg, row, col, _, mass, factor in ct_pool:
            match = None
            for cand in support:
                if (cand[0] == seg and abs(cand[1] - row) <= v1.CELL_ROW_GAP
                        and abs(cand[2] - col) <= v1.BLOCK):
                    match = cand
                    break
            if match is not None:
                match[4] += mass
            else:
                support.append([seg, row, col, 'support', mass, factor])
        return rest + [tuple(c) for c in support]

    v3u_block = config_block(union_pool())

    # v3w — the decomposition of v3u: corroboration weight only. Matched ct
    # mass still boosts the support cluster, but unmatched ct clusters are
    # dropped, so the support family keeps its size and the rect channel's H
    # candidates keep their top-4N slots. Separates "ct corroboration ranks
    # support better" from "extra candidates crowd the merge".
    def weight_pool():
        support = [list(c) for c in v1_pool if c[3] == 'support']
        rest = [c for c in v1_pool if c[3] != 'support']
        for seg, row, col, _, mass, factor in ct_pool:
            for cand in support:
                if (cand[0] == seg and abs(cand[1] - row) <= v1.CELL_ROW_GAP
                        and abs(cand[2] - col) <= v1.BLOCK):
                    cand[4] += mass
                    break
        return rest + [tuple(c) for c in support]

    v3w_block = config_block(weight_pool())

    report = {'detector': 'v3 = v1 +/- ct-intensity channel (TOPO-026)',
              'lineage': 'probe_ct positive 17.08 (13/36 missed-M covered, '
                         'bar 12/36); candidates replayed offline: v1 from '
                         'detect_v2 checkpoints, ct from probe windows',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              **v3_block,
              'ablation_v1_config': v1_block,
              'ablation_v3add_config': v3add_block,
              'ablation_v3u_config': v3u_block,
              'ablation_v3w_config': v3w_block,
              'ct_channel': {'threshold': threshold,
                             'candidates': len(ct_pool),
                             'probe_coverage': probe_report['coverage']}}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v1', v1_block), ('v3 (ct replaces support)',
                                            v3_block),
                         ('v3add (v1 + ct)', v3add_block),
                         ('v3u (ct merged into support)', v3u_block),
                         ('v3w (ct corroboration weight only)', v3w_block)):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + f", plausible={block['recall_on_locally_plausible']:.3f}")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
