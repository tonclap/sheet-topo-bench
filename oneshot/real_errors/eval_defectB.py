"""TOPO-053: disclination density of the CT normal field vs the corpus-B zones
(U-025).

TOPO-052 closed the differential form on synthetics (0/22 — injections never
touch the volume) with a clean attribution: the absolute form scores 8/22 at
a 0.71 % floor, i.e. defect lines of the layering's normal field *do* see
anomalous zones of the real substrate. This file asks the TOPO-028/046
question of that absolute form: does defect density land on the
human-verified corpus-B zones? The feature reads only CT — prediction-free
like thick; same scan as input, different read quantity (normal-field
topology vs bright-run thickness), so the row's claim is "third
prediction-free channel", not "input-disjoint from thick".

Declaration: CORPUS.md insert of 19.08.2026 (twenty-ninth session),
committed BEFORE any CT read under it. In brief:

- **Lattice:** band rows of ALL five z-quantiles of the corpus_paris4
  manifest (SUPPORT_BAND_VX = 160), row step 1 (median corpus-B zone is one
  row tall — a row stride would kill coverage), every 8th valid column
  (probe_defect_cost's COL_STRIDE verbatim). Pre-run sizing: 547 642 valid
  band-row nodes over the 4 segments -> ~68 000 probed nodes.
- **Per-node feature D — probe_defect_cost verbatim, imported not copied**
  (23^3 cube, SIGMA1/SIGMA2, Westin coherence, CL_MIN 0.5, ring transport
  R = 3, MIN_ELIGIBLE 100; the CL 0.25/0.75 x ring 2/4 sensitivity grid is
  published alongside and enters no rule).
- **Price probe first (the TOPO-049 lesson, TOPO-052's protocol):**
  background = lattice nodes OUTSIDE every zone rectangle grown by 50 % per
  axis (the credit rule's growth); FLOOR := pooled background q99 (q95 /
  q99.5 alongside). Degeneracy gates decide before the benefit stage:
  (v) median per-segment mute share > 0.5 -> the feature does not exist on
  this substrate; (b) pooled background median D > 0 -> the TOPO-049
  degenerate mode — the channel closes without a benefit read.
- **Channel stage (only past the gates):** evidence = node with D strictly
  > FLOOR, value D; (row, col // BLOCK) cells summing values, best per
  cell; `differenced_clusters` with an empty atlas (MAX_CLUSTER_ROWS 14,
  as every B channel); `merge_channels`, top 4N.
- **Zones, thresholds (20/10/40), the 100-seed random baseline and the
  by-zone bootstrap (2000 resamples, seed 20260815) — TOPO-028 verbatim.**
- **Reading rule — TOPO-028 verbatim:** the row signals iff the lower edge
  of its 95 % bootstrap AP interval sits strictly above the upper edge of
  the random baseline's IQR at the primary threshold; a primary signal that
  dies at both sensitivity thresholds reads as unstable, not as a signal.

Checkpoints: per-segment pickle with a per-row increment (payload carries
`rows_done`; a killed run resumes at the next unfinished band row — the
long-run lesson applied up front, not after the second crash).

Usage:

    python eval_defectB.py --map ../../output/topo/real_paris4/corpusB.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --zone-min 20 --report ../../output/topo/real_paris4/eval_defectB.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402
import sheet_erl                                                      # noqa: E402
from probe_ct import CTVolume                                         # noqa: E402
import probe_defect_cost as pdc                                       # noqa: E402
from eval_real import evaluate, zone_records                          # noqa: E402
from eval_supportB import bootstrap_ci                                # noqa: E402

MAIN_COMBO = (pdc.CL_MIN, pdc.RING_R)     # (0.5, 3) — the declared setting
CKPT_SAVE_EVERY = 5                       # band rows between checkpoint saves


def band_rows_of(grid, z_quantiles):
    heights, _ = scrolls.row_heights(grid)
    return [row for row in range(grid.shape[0])
            if np.isfinite(heights[row])
            and any(abs(heights[row] - q) <= detect_v1.SUPPORT_BAND_VX
                    for q in z_quantiles)]


def segment_defect_nodes_bands(ct, grid, z_quantiles, ckpt):
    """probe_defect_cost.segment_defect_nodes, widened to all five quantile
    bands and checkpointed per row-increment: `rows_done` marks how many band
    rows are already in `nodes`, so a killed run resumes mid-segment."""
    rows = band_rows_of(grid, z_quantiles)
    key = {'half': pdc.HALF, 'edge': pdc.EDGE, 'sigma1': pdc.SIGMA1,
           'sigma2': pdc.SIGMA2, 'cl_bars': list(pdc.CL_BARS),
           'ring_rs': list(pdc.RING_RS), 'stride': pdc.COL_STRIDE,
           'quantiles': [float(q) for q in z_quantiles], 'rows': len(rows)}
    payload = None
    if os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') != key:
            payload = None
    if payload is None:
        payload = {'key': key, 'rows_done': 0,
                   'nodes': {'row': [], 'col': [], 'stats': []}}
    out = payload['nodes']
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)

    def save():
        tmp = f'{ckpt}.{os.getpid()}.part'
        with open(tmp, 'wb') as f:
            pickle.dump(payload, f)
        os.replace(tmp, ckpt)

    for k in range(payload['rows_done'], len(rows)):
        row = rows[k]
        mask = valid[row]
        if mask.any():
            cols = np.where(mask)[0][::pdc.COL_STRIDE]
            points = grid[row][cols].astype(np.float64)
            for start in range(0, len(cols), pdc.BATCH_NODES):
                batch_pts = points[start:start + pdc.BATCH_NODES]
                batch_cols = cols[start:start + pdc.BATCH_NODES]
                stacked = (batch_pts[:, None, :]
                           + pdc.CUBE_OFFSETS[None, :, :]).reshape(-1, 3)
                sampled = ct.values(stacked).astype(np.float64).reshape(
                    len(batch_pts), pdc.SIDE, pdc.SIDE, pdc.SIDE)
                for j, col in enumerate(batch_cols):
                    out['row'].append(int(row))
                    out['col'].append(int(col))
                    out['stats'].append(pdc.node_defects(sampled[j]))
        payload['rows_done'] = k + 1
        if (k + 1) % CKPT_SAVE_EVERY == 0 or k + 1 == len(rows):
            save()
            print(f'  row {k + 1}/{len(rows)} ({len(out["col"])} nodes '
                  f'probed, checkpointed)', flush=True)
    return out


def background_mask(nodes, zones):
    """True where the node lies outside every zone rectangle grown 50 %."""
    rows = np.array(nodes['row'])
    cols = np.array(nodes['col'])
    outside = np.ones(len(rows), bool)
    for z in zones:
        growth_r = (z['row_hi'] - z['row_lo']) * 0.25
        growth_c = (z['col_hi'] - z['col_lo']) * 0.25
        inside = ((rows >= z['row_lo'] - growth_r)
                  & (rows < z['row_hi'] + growth_r)
                  & (cols >= z['col_lo'] - growth_c)
                  & (cols < z['col_hi'] + growth_c))
        outside &= ~inside
    return outside


def node_density(nodes, combo):
    """Per-node D at one (CL, ring) setting; NaN where mute."""
    d = np.full(len(nodes['stats']), np.nan)
    for i, st in enumerate(nodes['stats']):
        n_elig, n_def = st[combo]
        if n_elig >= pdc.MIN_ELIGIBLE:
            d[i] = n_def / n_elig
    return d


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--zone-min', type=int, required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--random-seeds', type=int, default=100)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(os.path.join(args.corpus, 'manifest.json'),
              encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    ct = CTVolume(args.cache, base=scroll.ct, level=scroll.level)
    names = corpus_map['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}
    zones_all = corpus_map['zones']

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(args.report)),
                            'defect_cells_B')
    os.makedirs(ckpt_dir, exist_ok=True)
    node_sets = {}
    for name in names:
        print(f'{name}: probing disclination density '
              f'({len(band_rows_of(grids[name], manifest["z_quantiles"]))} '
              f'band rows)', flush=True)
        node_sets[name] = segment_defect_nodes_bands(
            ct, grids[name], manifest['z_quantiles'],
            os.path.join(ckpt_dir, f'{name}.pkl'))

    # ---- Stage 1: price probe — background floor and degeneracy gates ----
    combos = [(bar, r) for bar in pdc.CL_BARS for r in pdc.RING_RS]
    per_segment = {}
    pooled_bg = {combo: [] for combo in combos}
    mute_rates = []
    for name in names:
        nodes = node_sets[name]
        seg_zones = [z for z in zones_all if z['segment'] == name]
        bg = background_mask(nodes, seg_zones)
        entry = {}
        for combo in combos:
            d = node_density(nodes, combo)
            d_bg = d[bg]
            mute_rate = (float(np.isnan(d_bg).mean())
                         if d_bg.size else None)
            alive = d_bg[~np.isnan(d_bg)]
            pooled_bg[combo].extend(float(x) for x in alive)
            entry[f'{combo[0]}/{combo[1]}'] = {
                'background_nodes': int(d_bg.size),
                'mute_rate': mute_rate,
                'median': float(np.median(alive)) if alive.size else None,
                'zero_share': (round(float((alive == 0).mean()), 4)
                               if alive.size else None),
                'q99': (float(np.percentile(alive, 99))
                        if alive.size else None)}
        per_segment[name] = entry
        main_entry = entry[f'{MAIN_COMBO[0]}/{MAIN_COMBO[1]}']
        mute_rates.append(main_entry['mute_rate'])
        print(f'{name}: {len(nodes["stats"])} nodes '
              f'({int(bg.sum())} background), mute '
              f'{main_entry["mute_rate"]}, median D {main_entry["median"]}, '
              f'q99 {main_entry["q99"]}', flush=True)

    d_main = np.array(pooled_bg[MAIN_COMBO])
    median_mute = float(np.median([m for m in mute_rates if m is not None]))
    pooled_median = float(np.median(d_main)) if d_main.size else None
    floor = (float(np.percentile(d_main, pdc.FLOOR_QUANTILE))
             if d_main.size else None)
    gate_mute_dead = median_mute > pdc.GATE_MUTE_MAX
    gate_median_dead = (pooled_median is not None) and (pooled_median > 0)

    report = {'declared_zone_min': args.zone_min,
              'map': os.path.abspath(args.map),
              'declaration': 'CORPUS.md insert 19.08.2026 (TOPO-053, '
                             'twenty-ninth session), committed before the '
                             'run; feature constants imported from '
                             'probe_defect_cost (TOPO-052)',
              'lattice': {'quantile_bands': 'all 5', 'row_stride': 1,
                          'col_stride': pdc.COL_STRIDE,
                          'nodes': {n: len(node_sets[n]['stats'])
                                    for n in names}},
              'stage1_price': {
                  'per_segment': per_segment,
                  'pooled_background': {
                      'nodes': int(d_main.size),
                      'median': pooled_median,
                      'zero_share': (round(float((d_main == 0).mean()), 4)
                                     if d_main.size else None),
                      'q95': (float(np.percentile(d_main, 95))
                              if d_main.size else None),
                      'q99': floor,
                      'q99_5': (float(np.percentile(d_main, 99.5))
                                if d_main.size else None)},
                  'floor': floor,
                  'gates': {'median_mute': median_mute,
                            'gate_mute_dead': bool(gate_mute_dead),
                            'pooled_median': pooled_median,
                            'gate_median_dead': bool(gate_median_dead)}},
              'bootstrap': {'resamples': 2000, 'seed': 20260815},
              'rows': {}}

    if gate_mute_dead or gate_median_dead:
        verdict = ('(v) DEGENERATE: '
                   + ('median mute share > 0.5' if gate_mute_dead
                      else 'pooled background median D > 0 (TOPO-049 mode)')
                   + ' — the channel stage is not read')
        report['verdict'] = verdict
        os.makedirs(os.path.dirname(os.path.abspath(args.report)),
                    exist_ok=True)
        with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(verdict)
        print(f'report at {args.report}')
        return

    # ---- Stage 2: the channel, evidence strictly D > FLOOR ----
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
        tall = 0
        for cells, mass, top in detect_v1.differenced_clusters(
                evidence, best, {}):
            if cells is None:
                tall += 1
                continue
            candidates.append((name, top[0], best[top][0], 'defect', mass,
                               best[top][1]))
        print(f'{name}: {len(candidates)} defect candidates so far '
              f'({tall} tall clusters cut)', flush=True)
    report['channel'] = {'floor': floor, 'setting': list(MAIN_COMBO),
                         'max_cluster_rows': detect_v1.MAX_CLUSTER_ROWS,
                         'candidates': len(candidates)}

    rng_pool = [(name, row, col)
                for name, grid in sorted(grids.items())
                for row, col in sheet_erl.windows_of(grid)]
    entry = {}
    for label, threshold in (('primary', args.zone_min),
                             ('half', args.zone_min // 2),
                             ('double', args.zone_min * 2)):
        zones = zone_records(zones_all, threshold)
        n = len(zones)
        randoms = []
        for seed in range(args.random_seeds):
            rng = np.random.default_rng(seed)
            picks = rng.choice(len(rng_pool), size=min(n, len(rng_pool)),
                               replace=False)
            random_ranking = [(*rng_pool[i], 0.0) for i in picks]
            randoms.append(evaluate(random_ranking, zones, n)[0])
        baseline = {
            key: {'median': float(np.median([r[key] for r in randoms])),
                  'iqr': [float(np.percentile([r[key] for r in randoms], 25)),
                          float(np.percentile([r[key] for r in randoms], 75))]}
            for key in ('ap', 'recall_at_n')}
        ranking = detect_v1.merge_channels(candidates, top=4 * n)
        metrics, outcomes = evaluate(ranking, zones, n)
        variant = {'zone_min': threshold, 'n_zones': n, 'metrics': metrics,
                   'baseline_random': baseline,
                   'ci': bootstrap_ci(outcomes, zones, n)}
        if label == 'primary':
            iqr_hi = baseline['ap']['iqr'][1]
            variant['signal'] = bool(variant['ci']['ap_ci95'][0] > iqr_hi)
            found_rank = {}
            for rank, (hit, *_rest) in enumerate(outcomes, 1):
                if hit is not None and hit not in found_rank:
                    found_rank[hit] = rank
            variant['zones_found'] = len(found_rank)
            variant['credited'] = {f'Z{k:04d}': r
                                   for k, r in sorted(found_rank.items())}
        entry[label] = variant
    report['rows']['defect'] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    p = entry['primary']
    verdict = 'SIGNAL' if p['signal'] else 'chance'
    print(f"defect (mass >= {args.zone_min}, {p['n_zones']} zones): "
          f"AP {p['metrics']['ap']:.4f} "
          f"[{p['ci']['ap_ci95'][0]:.4f}-{p['ci']['ap_ci95'][1]:.4f}] "
          f"vs random IQR {p['baseline_random']['ap']['iqr'][0]:.4f}-"
          f"{p['baseline_random']['ap']['iqr'][1]:.4f} -> {verdict}")
    for label in ('half', 'double'):
        e = entry[label]
        print(f"  sensitivity {label:6s} (>= {e['zone_min']}, "
              f"{e['n_zones']} zones): AP {e['metrics']['ap']:.4f} "
              f"[{e['ci']['ap_ci95'][0]:.4f}-{e['ci']['ap_ci95'][1]:.4f}]")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
