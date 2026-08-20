"""Detector v4 ablation: quota merge vs percentile merge (TOPO-016, U-012).

TOPO-026 measured, on a third independent feature family, that what binds the
line is not the input but the percentile merge itself: v3u (ct folded into the
support family) gave the line's first significant AP gain (+0.023) and paid
for it with a significant H loss (-0.078) through exactly the mechanism U-012
names — the percentile merge hands each family a share of every ranking
prefix *proportional to its pool size* (dev pools: prox 102, rect 77,
support 125, ct 93; a family that grows crowds the others out of top-N).
This file implements the branch's named alternative: top-N quotas per family,
derived from the corpus manifest, not tuned on AP.

**Declared before any run of this file (the numbers below are from the
manifest and from already-frozen reports, not from v4 output):**

1. **Family→type mapping is mechanistic and fixed:** prox→S (contact of the
   trace with the neighbouring winding's annotation is the sheet-switch
   signature, U-008), rect→H (hole shape, U-009), support→M (trace without
   surface at the merger's seat, U-011; the ct evidence is the merger's
   physical signature in the scan and joins this family in the union pool,
   TOPO-026).
2. **Quotas are the dev-manifest type shares:** S 76 / M 79 / H 77 of N=232
   — i.e. prox 76/232, support 79/232, rect 77/232 of every ranking prefix.
   No other constants exist in the scheme; there is nothing to tune.
3. **The ranking is built by largest-deficit apportionment:** at position k
   (1-based) the family with the largest deficit share*k - taken among
   non-empty families wins the slot (ties: alphabetical family name) and
   contributes its next candidate; within a family candidates are ordered by
   descending mass, ties by (segment, row, col). When a family's pool is
   exhausted the remaining families fill its slots by the same rule.
4. **Configurations:** v1 and v3u under the frozen percentile merge (both are
   regression anchors: v1 must reproduce the frozen detector_v1 report to six
   decimals, v3u must reproduce detector_v3's ablation_v3u_config block), and
   the same two pools under the quota merge (v1q, v3uq).
5. **Ship rule (same as v3):** a quota form is a candidate for the held-out
   exam only if, against v1 on paired bootstrap, AP or M improves
   significantly AND none of AP / recall@N / S / M / H / plausible degrades
   significantly. Which single form goes to held-out is fixed in
   ABLATION_V4.md before any held-out run. TOPO-025's budget stays untouched
   either way.

No new sampling happens here: the v1 pool is replayed from the detect_v2
checkpoints, the ct pool from the probe's stored windows, exactly as
detect_v3 did. Frozen files (detect_v1.py, sheet_erl.py) are not touched;
the quota merge lives here, next to — not instead of — merge_channels.

Usage (from oneshot/detector/):

    python detect_v4.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids \
        --probe-report ../../output/topo/probe_ct_paris4.json \
        --probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --v1-report ../../output/topo/detector_v1_paris4.json \
        --v3-report ../../output/topo/detector_v3_paris4.json \
        --bands dev --report ../../output/topo/detector_v4_paris4.json
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
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v1 as v1                                                # noqa: E402
import detect_v3 as v3                                                # noqa: E402
import probe_ct                                                       # noqa: E402


def union_pool(v1_pool, ct_pool):
    """v3u's pool, reproduced verbatim from detect_v3 (ct folded into support)."""
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


def merge_channels_quota(candidates, top, quotas):
    """One ranking from several channels, by per-family prefix quotas.

    `quotas` maps family name -> share of every ranking prefix (shares over
    present families are renormalised implicitly by the deficit rule). Within
    a family candidates are ordered by descending mass, ties by (segment,
    row, col) — fully canonical, no dependence on insertion order.
    """
    families = {}
    for c in candidates:
        families.setdefault(c[3], []).append(c)
    for name in families:
        families[name].sort(key=lambda c: (-c[4], c[0], c[1], c[2]))
    queues = {name: iter(fam) for name, fam in families.items()}
    remaining = {name: len(fam) for name, fam in families.items()}
    taken = {name: 0 for name in families}
    ranked = []
    total = min(top, sum(remaining.values()))
    for k in range(1, total + 1):
        live = [name for name in sorted(families) if remaining[name] > 0]
        name = max(live, key=lambda f: (quotas.get(f, 0.0) * k - taken[f],
                                        # ties: alphabetical — max() keeps the
                                        # first of equals, live is sorted
                                        ))
        c = next(queues[name])
        remaining[name] -= 1
        taken[name] += 1
        ranked.append((c[0], c[1], c[2], 1.0 - (k - 1) / total))
    return ranked


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

    # Quotas from the manifest's type counts, via the declared mapping.
    type_counts = {t: sum(1 for r in injections if r['type'] == t)
                   for t in 'SMH'}
    quotas = {'prox': type_counts['S'] / n,
              'support': type_counts['M'] / n,
              'rect': type_counts['H'] / n}
    print(f"quotas from manifest ({args.bands}, N={n}): "
          + ', '.join(f'{f}={type_counts[t]}/{n}' for f, t in
                      (('prox', 'S'), ('support', 'M'), ('rect', 'H'))))

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
    ct_pool = v3.ct_candidates(records, by_id, threshold)
    v3u_pool = union_pool(v1_pool, ct_pool)
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

    v1q_block = judged(merge_channels_quota(v1_pool, top=4 * n, quotas=quotas))
    v3uq_block = judged(merge_channels_quota(v3u_pool, top=4 * n,
                                             quotas=quotas))

    report = {'detector': 'v4 = quota merge vs percentile merge '
                          '(TOPO-016, U-012)',
              'lineage': 'v3u pays H -0.078 for AP +0.023 through pool-size '
                         'proportional prefix shares; quotas from manifest '
                         'type counts via the declared mechanistic mapping '
                         'prox-S support-M rect-H; candidates replayed '
                         'offline: v1 from detect_v2 checkpoints, ct from '
                         'probe windows',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'quotas': {f: f'{type_counts[t]}/{n}' for f, t in
                         (('prox', 'S'), ('support', 'M'), ('rect', 'H'))},
              **v3uq_block,
              'ablation_v1_config': v1_block,
              'ablation_v3u_config': v3u_block,
              'ablation_v1q_config': v1q_block,
              'ct_channel': {'threshold': threshold,
                             'candidates': len(ct_pool)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v1 (percentile)', v1_block),
                         ('v3u (percentile)', v3u_block),
                         ('v1q (quota)', v1q_block),
                         ('v3uq (quota)', v3uq_block)):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + f", plausible={block['recall_on_locally_plausible']:.3f}")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
