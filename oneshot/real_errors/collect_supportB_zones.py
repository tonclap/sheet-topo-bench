"""TOPO-029: the corpus-B zones the frozen support channel credited, by rank.

`eval_supportB.json` (TOPO-028) stores only aggregate rows — no per-zone
ranks. This script replays the support row from the per-segment checkpoints
(`cells_supportB/*.pkl`, no network, no prediction reads) and writes the
credited-zone table for the manual classification (ZONE_CRITERIA_B.md,
declared before this ran). The replay is gated: its AP must reproduce the
published support/primary AP from `eval_supportB.json` exactly, or nothing is
written — the number the cards inherit is the number that was published.

Usage (from oneshot/real_errors/):

    python collect_supportB_zones.py \
        --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --eval-report ../../output/topo/real_paris4/eval_supportB.json \
        --out ../../output/topo/real_paris4/supportB_zones.json
"""
import argparse
import json
import os
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_supportB import support_candidates                          # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--eval-report', required=True,
                        help='published eval_supportB.json (regression gate)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.eval_report, encoding='utf-8') as f:
        published = json.load(f)
    zone_min = published['declared_zone_min']
    published_row = published['rows']['support']['primary']

    scroll = scrolls.SCROLLS[manifest['scroll']]
    names = corpus_map['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(args.eval_report)),
                            'cells_supportB')
    support = support_candidates(names, grids, scroll, args.cache,
                                 manifest['z_quantiles'], ckpt_dir)

    zones = zone_records(corpus_map['zones'], zone_min)
    n = len(zones)
    if n != published_row['n_zones']:
        raise SystemExit(f'zone count mismatch: {n} vs published '
                         f"{published_row['n_zones']} — nothing written")
    ranking = detect_v1.merge_channels(support, top=4 * n)
    metrics, outcomes = evaluate(ranking, zones, n)
    if abs(metrics['ap'] - published_row['metrics']['ap']) > 1e-12:
        raise SystemExit(f"AP regression failed: replay {metrics['ap']} vs "
                         f"published {published_row['metrics']['ap']} — "
                         f'nothing written')

    found_rank = {}
    for rank, (hit, *_rest) in enumerate(outcomes, 1):
        if hit is not None and hit not in found_rank:
            found_rank[hit] = rank
    credited = [dict(z, support_rank=found_rank[k])
                for k, z in enumerate(zones) if k in found_rank]
    credited.sort(key=lambda z: z['support_rank'])

    report = {'declared_zone_min': zone_min, 'n_zones': n,
              'ranking_top': 4 * n,
              'replayed_ap': metrics['ap'],
              'published_ap': published_row['metrics']['ap'],
              'n_credited': len(credited),
              'in_top_n': sum(1 for z in credited if z['support_rank'] <= n),
              'zones': credited}
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    # The summary comes out of the same numbers as the table.
    print(f"support replay: AP {metrics['ap']:.4f} == published "
          f"{published_row['metrics']['ap']:.4f}; {len(credited)} zones "
          f"credited in top-{4 * n} ({report['in_top_n']} in top-{n})")
    for z in credited[:10]:
        print(f"  rank {z['support_rank']:4d}  {z['segment']}  "
              f"rows {z['row_lo']}-{z['row_hi']} cols {z['col_lo']}-"
              f"{z['col_hi']}  mass {z['mass']}")
    print(f'report at {args.out}')


if __name__ == '__main__':
    main()
