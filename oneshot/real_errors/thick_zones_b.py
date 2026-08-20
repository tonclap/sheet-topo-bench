"""TOPO-046 follow-up: the corpus-B zones the thick channel credited, by rank.

`eval_thicknessB.json` stores only aggregate rows. This script replays the
thick row from the per-segment thickness checkpoints (`thick_cells_B/*.pkl`,
no network, no CT reads) and writes the credited-zone table, joined with

- the support-credited table (`supportB_zones.json`, TOPO-029) — do the two
  validated channels credit the same zones or different ones?
- the by-hand zone classes (`zone_labels_b.csv` for the 57 support-credited
  zones, `zone_labels_census_b.csv` for the 121 uncredited ones).

The replay is gated: its AP must reproduce the published thick/primary AP
from `eval_thicknessB.json` exactly, or nothing is written. The join is a
recorded post-hoc observation — the primary result of TOPO-046 is the
declared reading rule's verdict, not anything here.

Usage (from oneshot/real_errors/):

    python thick_zones_b.py \
        --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids \
        --eval-report ../../output/topo/real_paris4/eval_thicknessB.json \
        --support-zones ../../output/topo/real_paris4/supportB_zones.json \
        --out ../../output/topo/real_paris4/thickB_zones.json
"""
import argparse
import csv
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_thicknessB import thickness_candidates                      # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--eval-report', required=True,
                        help='published eval_thicknessB.json (regression gate)')
    parser.add_argument('--support-zones', required=True,
                        help='supportB_zones.json (TOPO-029)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(args.eval_report, encoding='utf-8') as f:
        published = json.load(f)
    zone_min = published['declared_zone_min']
    published_row = published['rows']['thick']['primary']

    names = corpus_map['segments']
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(args.eval_report)),
                            'thick_cells_B')
    node_sets = {}
    for name in names:
        with open(os.path.join(ckpt_dir, f'{name}.pkl'), 'rb') as f:
            node_sets[name] = pickle.load(f)['nodes']
    candidates, _stats = thickness_candidates(names, node_sets)

    zones = zone_records(corpus_map['zones'], zone_min)
    n = len(zones)
    if n != published_row['n_zones']:
        raise SystemExit(f'zone count mismatch: {n} vs published '
                         f"{published_row['n_zones']} — nothing written")
    ranking = detect_v1.merge_channels(candidates, top=4 * n)
    metrics, outcomes = evaluate(ranking, zones, n)
    if abs(metrics['ap'] - published_row['metrics']['ap']) > 1e-12:
        raise SystemExit(f"AP regression failed: replay {metrics['ap']} vs "
                         f"published {published_row['metrics']['ap']} — "
                         f'nothing written')

    found_rank = {}
    for rank, (hit, *_rest) in enumerate(outcomes, 1):
        if hit is not None and hit not in found_rank:
            found_rank[hit] = rank
    credited = [dict(z, thick_rank=found_rank[k])
                for k, z in enumerate(zones) if k in found_rank]
    credited.sort(key=lambda z: z['thick_rank'])

    with open(args.support_zones, encoding='utf-8') as f:
        support_by_id = {z['id']: z['support_rank']
                         for z in json.load(f)['zones']}
    labels = {}
    for name in ('zone_labels_b.csv', 'zone_labels_census_b.csv'):
        path = os.path.join(_HERE, name)
        with open(path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                labels[row['id']] = row['class']
    for z in credited:
        z['support_rank'] = support_by_id.get(z['id'])
        z['label'] = labels.get(z['id'])

    both = [z for z in credited if z['support_rank'] is not None]
    label_counts = {}
    for z in credited:
        label_counts[z['label']] = label_counts.get(z['label'], 0) + 1
    report = {'declared_zone_min': zone_min, 'n_zones': n,
              'ranking_top': 4 * n,
              'replayed_ap': metrics['ap'],
              'published_ap': published_row['metrics']['ap'],
              'n_credited': len(credited),
              'in_top_n': sum(1 for z in credited if z['thick_rank'] <= n),
              'also_support_credited': len(both),
              'label_counts': label_counts,
              'zones': credited}
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f"thick replay: AP {metrics['ap']:.4f} == published "
          f"{published_row['metrics']['ap']:.4f}; {len(credited)} zones "
          f"credited in top-{4 * n} ({report['in_top_n']} in top-{n}); "
          f"{len(both)} also support-credited")
    print(f'labels of credited zones: {label_counts}')
    for z in credited:
        also = (f" support_rank {z['support_rank']}"
                if z['support_rank'] is not None else '')
        print(f"  rank {z['thick_rank']:4d}  {z['id']}  {z['segment']}  "
              f"rows {z['row_lo']}-{z['row_hi']} mass {z['mass']}  "
              f"label {z['label']}{also}")
    print(f'report at {args.out}')


if __name__ == '__main__':
    main()
