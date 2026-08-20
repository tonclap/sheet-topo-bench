"""TOPO-055: the corpus-B zones the defect channel credited, by rank.

`eval_defectB.json` (TOPO-053) stores credited ids and ranks but no zone
rectangles — the band renderer needs them. This script replays the defect
row entirely offline from the per-segment checkpoints
(`defect_cells_B/*.pkl`, no CT reads: the node stats are in the payloads)
and writes the credited-zone table plus the defect-only subset (credited by
defect, by neither support nor thick — the TOPO-055 candidates).

Three regression gates, any failure writes nothing:

- the recomputed background FLOOR must reproduce the published
  `stage1_price.floor` exactly;
- the replayed ranking's primary AP must reproduce the published
  defect/primary AP to 1e-12;
- the credited id set must equal the published `credited` keys.

Usage (from pipeline/real_errors/):

    python collect_defectB_zones.py \
        --map ../../output/topo/real_paris4/corpusB.json \
        --eval-report ../../output/topo/real_paris4/eval_defectB.json \
        --support-zones ../../output/topo/real_paris4/supportB_zones.json \
        --thick-zones ../../output/topo/real_paris4/thickB_zones.json \
        --out ../../output/topo/real_paris4/defectB_zones.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import detect_v1                                                      # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_defectB import MAIN_COMBO, background_mask, node_density    # noqa: E402


def defect_candidates_from_checkpoints(names, ckpt_dir, zones_all,
                                       published_floor):
    """The published defect pool, replayed from checkpoints only.

    A missing checkpoint or an unfinished probe is an error, not a resample —
    the run that published the AP finished every band row.
    """
    node_sets = {}
    for name in names:
        with open(os.path.join(ckpt_dir, f'{name}.pkl'), 'rb') as f:
            payload = pickle.load(f)
        if payload['rows_done'] != payload['key']['rows']:
            raise SystemExit(f'{name}: checkpoint unfinished '
                             f"({payload['rows_done']}/"
                             f"{payload['key']['rows']} rows) — nothing "
                             f'written')
        node_sets[name] = payload['nodes']

    pooled_bg = []
    for name in names:
        nodes = node_sets[name]
        seg_zones = [z for z in zones_all if z['segment'] == name]
        bg = background_mask(nodes, seg_zones)
        d = node_density(nodes, MAIN_COMBO)
        alive = d[bg][~np.isnan(d[bg])]
        pooled_bg.extend(float(x) for x in alive)
    floor = float(np.percentile(np.array(pooled_bg), 99))
    if abs(floor - published_floor) > 1e-15:
        raise SystemExit(f'floor regression failed: replay {floor} vs '
                         f'published {published_floor} — nothing written')

    candidates = []
    for name in names:
        nodes = node_sets[name]
        d = node_density(nodes, MAIN_COMBO)
        evidence, best = {}, {}
        for row, col, value in zip(nodes['row'], nodes['col'], d):
            if np.isnan(value) or value <= floor:
                continue
            cell = (row, col // detect_v1.BLOCK)
            evidence[cell] = evidence.get(cell, 0.0) + value
            if value > best.get(cell, (None, -1.0))[1]:
                best[cell] = (col, value)
        for cells, mass, top in detect_v1.differenced_clusters(
                evidence, best, {}):
            if cells is None:
                continue
            candidates.append((name, top[0], best[top][0], 'defect', mass,
                               best[top][1]))
    return candidates, floor


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--eval-report', required=True,
                        help='published eval_defectB.json (regression gates)')
    parser.add_argument('--support-zones', required=True,
                        help='supportB_zones.json (credited ids to subtract)')
    parser.add_argument('--thick-zones', required=True,
                        help='thickB_zones.json (credited ids to subtract)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(args.eval_report, encoding='utf-8') as f:
        published = json.load(f)
    with open(args.support_zones, encoding='utf-8') as f:
        support_ids = {z['id'] for z in json.load(f)['zones']}
    with open(args.thick_zones, encoding='utf-8') as f:
        thick_ids = {z['id'] for z in json.load(f)['zones']}
    zone_min = published['declared_zone_min']
    published_row = published['rows']['defect']['primary']

    names = corpus_map['segments']
    ckpt_dir = os.path.join(
        os.path.dirname(os.path.abspath(args.eval_report)), 'defect_cells_B')
    candidates, floor = defect_candidates_from_checkpoints(
        names, ckpt_dir, corpus_map['zones'], published['stage1_price']['floor'])

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
    replay_credited = {f'Z{k:04d}': r for k, r in found_rank.items()}
    if replay_credited != published_row['credited']:
        raise SystemExit('credited id/rank set diverges from the published '
                         'report — nothing written')

    credited = [dict(z, defect_rank=found_rank[k])
                for k, z in enumerate(zones) if k in found_rank]
    credited.sort(key=lambda z: z['defect_rank'])
    defect_only = [z for z in credited
                   if z['id'] not in support_ids and z['id'] not in thick_ids]

    report = {'declared_zone_min': zone_min, 'n_zones': n,
              'ranking_top': 4 * n,
              'replayed_ap': metrics['ap'],
              'published_ap': published_row['metrics']['ap'],
              'floor': floor,
              'n_credited': len(credited),
              'in_top_n': sum(1 for z in credited if z['defect_rank'] <= n),
              'defect_only_ids': [z['id'] for z in defect_only],
              'zones': credited}
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    # The summary comes out of the same numbers as the table.
    print(f"defect replay: AP {metrics['ap']:.4f} == published "
          f"{published_row['metrics']['ap']:.4f} (floor {floor:.6f}); "
          f"{len(credited)} zones credited in top-{4 * n}, "
          f"{len(defect_only)} defect-only")
    for z in defect_only:
        print(f"  rank {z['defect_rank']:4d}  {z['id']}  {z['segment']}  "
              f"rows {z['row_lo']}-{z['row_hi']} cols {z['col_lo']}-"
              f"{z['col_hi']}  mass {z['mass']}")
    print(f'report at {args.out}')


if __name__ == '__main__':
    main()
