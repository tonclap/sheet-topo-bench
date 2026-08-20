"""TOPO-052 benefit probe: the disclination feature on the 22 uncovered windows.

Stage 2 of U-022 (stage 1 = probe_defect_cost.py, the pristine floor). Runs
ONLY through stage 1's gates; the floor is not chosen here — it is the
pooled pristine q99 of D at the main setting (CL 0.5, ring 3), read from
the stage-1 report. The background is <= 1 % by construction; this probe
asks the benefit question at that fixed price.

**Declared before the run, not tuned on the result:**

- Feature: probe_defect_cost's construction verbatim (cube 23^3 step 1,
  SIGMA1/SIGMA2, Westin linearity, ring transport R = 3 at CL 0.5;
  D = defective / eligible centres; mute below 100 eligible centres).
- Windows: the 22 uncovered-union injections of coverage_breakdown, grown
  by +-2 rows / +-8 columns (probe_ct's margins); every valid column (no
  stride — windows are compact; the stride was a cost-scope device).
- Node evidence iff **strictly D > FLOOR** (FLOOR = stage-1 pooled q99 at
  the main setting); mute nodes carry no evidence. Cell (row, col //
  BLOCK) value = count of evidencing nodes (the probe_phase convention);
  per window, corrupted cell count minus pristine cell count must exceed
  ``SURPLUS_MIN = 0.5``; surviving cells cluster via ``differenced_clusters``
  with an empty atlas.
- **Decision rule: the channel is worth building iff at least 8 of the 22
  windows contain a surviving cluster (TOPO-045's bar verbatim).** Below
  that U-022's ring construction closes negative and the question of the
  target's boundary (the ~0.905 ceiling) goes to the owner.
- Published alongside (no part of the rule): the type split (17 M / 5 S),
  the window count at the q95 / q99.5 floors of the main setting, at the
  neighbouring (CL, ring) settings with their own q99 floors, and the
  absolute (non-differential) window count at FLOOR — the deployability
  diagnostic (a real channel has no pristine twin; TOPO-049's lesson).

Checkpoints: per-window JSONL (append-only, resume skips finished ids)
storing raw per-node stats at every (CL, ring) setting — every floor
replays offline.

Usage (from oneshot/detector/):

    python probe_defect.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --coverage ../../output/topo/coverage_breakdown_paris4.json \
        --cost-report ../../output/topo/probe_defect_cost_paris4.json \
        --report ../../output/topo/probe_defect_paris4.json \
        --details ../../output/topo/probe_defect_windows.jsonl
"""
import argparse
import json
import os
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
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import detect_v1 as v1                                                # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)
from probe_ct import CTVolume                                         # noqa: E402
from probe_defect_cost import (                                        # noqa: E402
    node_defects, CUBE_OFFSETS, SIDE, CL_BARS, RING_RS, CL_MIN, RING_R,
    MIN_ELIGIBLE, BATCH_NODES)

SURPLUS_MIN = 0.5         # probe_ct's per-cell surplus floor
ROW_MARGIN = 2            # window growth (probe_ct's margins)
COL_MARGIN = 8
DECISION_MIN = 8          # of 22 — TOPO-045's bar verbatim


def probe_window(ct, grid, rows, col_range):
    """Raw per-node defect stats for one window of one grid."""
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    out = {'row': [], 'col': [], 'stats': []}
    for row in rows:
        if not 0 <= row < grid.shape[0]:
            continue
        mask = valid[row].copy()
        scope = np.zeros_like(mask)
        scope[max(col_range[0], 0):col_range[1]] = True
        mask &= scope
        if not mask.any():
            continue
        cols = np.where(mask)[0]
        points = grid[row][cols].astype(np.float64)
        for start in range(0, len(cols), BATCH_NODES):
            batch_pts = points[start:start + BATCH_NODES]
            batch_cols = cols[start:start + BATCH_NODES]
            stacked = (batch_pts[:, None, :] + CUBE_OFFSETS[None, :, :]
                       ).reshape(-1, 3)
            sampled = ct.values(stacked).astype(np.float64).reshape(
                len(batch_pts), SIDE, SIDE, SIDE)
            for j, col in enumerate(batch_cols):
                st = node_defects(sampled[j])
                out['row'].append(int(row))
                out['col'].append(int(col))
                out['stats'].append(
                    {f'{bar}/{r}': list(st[(bar, r)]) for bar, r in st})
    return out


def evidence_cells(nodes, combo_key, floor):
    """(evidence, best) of nodes with strictly D > floor at one setting."""
    evidence, best = {}, {}
    for row, col, st in zip(nodes['row'], nodes['col'], nodes['stats']):
        n_elig, n_def = st[combo_key]
        if n_elig < MIN_ELIGIBLE:
            continue
        d = n_def / n_elig
        if d <= floor:
            continue
        key = (row, col // v1.BLOCK)
        evidence[key] = evidence.get(key, 0.0) + 1.0
        if d > best.get(key, (None, -1.0))[1]:
            best[key] = (col, d)
    return evidence, best


def surviving_clusters(corr_cells, prist_cells, best):
    surplus = {key: value - prist_cells.get(key, 0.0)
               for key, value in corr_cells.items()
               if value - prist_cells.get(key, 0.0) > SURPLUS_MIN}
    n = 0
    for cells, mass, top in v1.differenced_clusters(surplus, best, {}):
        if cells is None:
            continue
        n += 1
    return n


def window_hits(records, combo_key, floor, differential=True):
    hits = []
    for rec in records:
        corr, corr_best = evidence_cells(rec['corrupted'], combo_key, floor)
        if differential:
            prist, _ = evidence_cells(rec['pristine'], combo_key, floor)
        else:
            prist = {}
        if surviving_clusters(corr, prist, corr_best) > 0:
            hits.append(rec['id'])
    return hits


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--coverage', required=True)
    parser.add_argument('--cost-report', required=True,
                        help='stage-1 report (floors per setting)')
    parser.add_argument('--report', required=True)
    parser.add_argument('--details', required=True,
                        help='per-window JSONL (append-only checkpoint)')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.coverage, encoding='utf-8') as f:
        uncovered = {r['id'] for r in
                     json.load(f)['uncovered_union']['rows']}
    with open(args.cost_report, encoding='utf-8') as f:
        cost = json.load(f)
    main_key = f'{CL_MIN}/{RING_R}'
    floor_main = cost['pooled'][main_key]['q99']
    if floor_main is None:
        raise SystemExit('stage-1 floor missing — gates were not passed')

    scroll = scrolls.SCROLLS[manifest['scroll']]
    ct = CTVolume(args.cache, base=scroll.ct, level=scroll.level)

    targets = [r for r in manifest['injections'] if r['id'] in uncovered]
    if len(targets) != len(uncovered):
        raise SystemExit('coverage ids missing from the manifest')

    done = set()
    if os.path.exists(args.details):
        with open(args.details, encoding='utf-8') as f:
            for line in f:
                try:
                    done.add(json.loads(line)['id'])
                except (ValueError, KeyError):
                    continue
    grids = {}

    def grid_pair(name):
        if name not in grids:
            corrupted = np.load(os.path.join(args.corpus, 'grids',
                                             f'{name}.npy'))
            pristine = scrolls.segment_grid(name, scroll, args.grid_cache)
            grids[name] = (corrupted, pristine)
        return grids[name]

    with open(args.details, 'a', encoding='utf-8', newline='\n') as details:
        for k, r in enumerate(targets):
            if r['id'] in done:
                continue
            corrupted, pristine = grid_pair(r['segment'])
            rows = range(r['row_lo'] - ROW_MARGIN, r['row_hi'] + ROW_MARGIN)
            cols = (r['col_lo'] - COL_MARGIN, r['col_hi'] + COL_MARGIN)
            record = {
                'id': r['id'], 'type': r['type'],
                'segment': r['segment'],
                'plausible': bool(r.get('plausible')),
                'corrupted': probe_window(ct, corrupted, rows, cols),
                'pristine': probe_window(ct, pristine, rows, cols),
            }
            details.write(json.dumps(record, ensure_ascii=False) + '\n')
            details.flush()
            print(f'window {k + 1}/{len(targets)} {r["id"]} ({r["type"]}): '
                  f'{len(record["corrupted"]["col"])} nodes', flush=True)

    records, seen = [], set()
    with open(args.details, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec['id'] in seen:
                continue
            seen.add(rec['id'])
            records.append(rec)

    hits_main = window_hits(records, main_key, floor_main)
    decision = len(hits_main) >= DECISION_MIN
    by_type = {}
    ids_by_type = {}
    for rec in records:
        by_type.setdefault(rec['type'], 0)
        ids_by_type.setdefault(rec['type'], set()).add(rec['id'])
    for h in hits_main:
        for t, ids in ids_by_type.items():
            if h in ids:
                by_type[t] += 1

    sensitivity = {}
    for q in ('q95', 'q99_5'):
        f_alt = cost['pooled'][main_key][q]
        if f_alt is not None:
            sensitivity[f'main@{q}'] = len(window_hits(records, main_key, f_alt))
    for bar in CL_BARS:
        for r in RING_RS:
            key = f'{bar}/{r}'
            if key == main_key:
                continue
            f_alt = (cost['pooled'].get(key) or {}).get('q99')
            if f_alt is not None:
                sensitivity[f'{key}@q99'] = len(
                    window_hits(records, key, f_alt))
    absolute_hits = window_hits(records, main_key, floor_main,
                                differential=False)

    report = {
        'probe': 'volume disclination benefit (U-022 stage 2): 22 uncovered '
                 'windows, node evidence strictly D > stage-1 pristine q99, '
                 'probe_ct surplus construction, empty atlas',
        'declaration': 'probe_defect.py header, committed before the run '
                       '(TOPO-052, twenty-eighth session)',
        'floor_main': floor_main,
        'rule': f'channel worth building iff >= {DECISION_MIN} of '
                f'{len(records)} windows carry a surviving cluster',
        'windows_hit': len(hits_main),
        'hit_ids': sorted(hits_main),
        'by_type': by_type,
        'decision_build_channel': bool(decision),
        'sensitivity_no_rule': sensitivity,
        'absolute_at_floor_no_rule': {
            'windows_hit': len(absolute_hits),
            'note': 'no pristine subtraction — the deployability '
                    'diagnostic (TOPO-049 lesson)'},
        'n_windows': len(records),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f'windows hit (differential, floor {floor_main}): '
          f'{len(hits_main)}/{len(records)} (rule >= {DECISION_MIN})')
    print(f'by type: {by_type}')
    print(f'decision (pre-declared): '
          f'{"BUILD" if decision else "NEGATIVE - the ring construction closes"}')
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
