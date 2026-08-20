"""TOPO-018: the corpus-B labelling density gate, as a run report.

The gate was declared in CORPUS.md (dated insert, session 18) BEFORE the
measurement: sector voxel metrics (VOI / adapted Rand) are computed only if
>= 80% of the domain nodes of the banner-carrying segments are covered
(finite d <= T_cover = 10 vx L2). This script recomputes the measurement
from the per-segment checkpoints — offline, no network — and writes the run
report the benchmark README binds its density numbers to.

Usage (from oneshot/real_errors/):
    python density_gate_b.py \
        --map ../../output/topo/real_paris4/corpusB.json \
        --out ../../output/topo/real_paris4/densityB.json
"""
import argparse
import json
import os

import numpy as np

import sys
# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

GATE = 0.80
T_COVER = 10.0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    cells_dir = os.path.join(os.path.dirname(os.path.abspath(args.map)),
                             'cells_corpusB')

    per_segment = {}
    ds = {}
    total = 0
    for name in corpus_map['segments']:
        ck = np.load(os.path.join(cells_dir, f'{name}.npz'))
        d = ck['d']
        ds[name] = d
        covered = int((d[np.isfinite(d)] <= T_COVER).sum())
        per_segment[name] = {'nodes': int(len(d)), 'covered': covered,
                             'share': covered / len(d)}
        total += len(d)

    def covered_at(t):
        return int(sum((d[np.isfinite(d)] <= t).sum() for d in ds.values()))

    covered = covered_at(T_COVER)
    report = {
        'declared_gate': GATE,
        't_cover': T_COVER,
        'total_nodes': total,
        'covered': covered,
        'share': covered / total,
        'passes': covered / total >= GATE,
        'per_segment': per_segment,
        'sensitivity': {str(t): {'covered': covered_at(t),
                                 'share': covered_at(t) / total}
                        for t in (T_COVER * 0.75, T_COVER * 1.25)},
    }
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"density gate: {covered}/{total} = {covered / total:.1%} "
          f"(gate >= {GATE:.0%}: {'PASS' if report['passes'] else 'FAIL'}) "
          f"-> {args.out}")


if __name__ == '__main__':
    main()
