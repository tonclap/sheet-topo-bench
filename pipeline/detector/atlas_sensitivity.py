"""Is the substrate contact atlas stable under the contact threshold? TOPO-021.

The defect map of the production substrate (contact zones between adjacent
windings) is published at one threshold: contact = distance below ~5 vx
(CONTACT_REF 10 * (1 - PROX_FLOOR 0.5)). The 15.08 review asked the honest
question: if the threshold moves, does the map survive? This is a sensitivity
analysis on PRISTINE dev grids only — the frozen detector, its thresholds and
every held-out number are untouched.

Protocol, declared before the run (TOPO-021):

- Substrate: the 16 pristine dev grids of corpus_paris4 (winding_low < 100),
  the same grids the published atlas stands on.
- Per-point contact scores are computed once (they do not depend on the
  threshold); the threshold only gates which points count as evidence.
  Variants: contact < 3 vx (floor 0.7), < 5 vx (floor 0.5, the published
  reference), < 8 vx (floor 0.2).
- Atlas at a threshold = evidence cells (row, col-block) built exactly like
  detect_v1.strong_cells; clusters = the same union-find (±CELL_ROW_GAP rows,
  ±1 block), no tall-drop (natural zones may be tall; the map is the subject
  here, not a candidate filter).
- Stability measures, declared up front:
  1. cell level — Jaccard of evidence-cell sets against the 5 vx reference;
  2. cluster level — top-100 clusters by mass at each threshold; a cluster is
     *retained* at another threshold if >= 50 % of its cells lie on that
     threshold's top-100 cells dilated by ±1 row (the same on-atlas rule as
     detect_v1.differenced_clusters); report retention in both directions
     for 5<->3 and 5<->8;
  3. contact share of skeleton points at each threshold (0.61 % published).
- Verdict rule, declared before seeing any number: "стабильна" if all four
  retentions >= 0.8; "порого-зависима" if any retention < 0.5; anything
  between = "частично устойчива" (publish as a range / threshold-robust core).

Usage:
    python pipeline/detector/atlas_sensitivity.py \
        --corpus output/topo/corpus_paris4 --grid-cache output/figgrids \
        --cache output/figcache --checkpoint output/topo/atlas_sensitivity \
        --report output/topo/atlas_sensitivity.json
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
sys.path.insert(0, _HERE)
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402

# Contact thresholds under test, vx -> evidence floor on the score
# 1 - d / CONTACT_REF. 5 vx is the published reference.
THRESHOLDS_VX = (3.0, 5.0, 8.0)
REFERENCE_VX = 5.0
TOP_K = 100
RETAIN_MIN_STABLE = 0.8   # verdict rule, declared before the run
RETAIN_MAX_DEPENDENT = 0.5


def floor_of(vx):
    return 1.0 - vx / detect_v1.CONTACT_REF


def segment_atlases(name, scroll, grid_cache, centre):
    """Per-threshold evidence cells of one pristine grid, scores computed once.

    Returns ({vx: {(row, block): mass}}, {vx: strong_points}, total_points).
    """
    grid = scrolls.segment_grid(name, scroll, grid_cache)
    evidence = {vx: {} for vx in THRESHOLDS_VX}
    strong_pts = {vx: 0 for vx in THRESHOLDS_VX}
    total = 0
    for row in range(grid.shape[0]):
        mask = (grid[row, :, 0] != -1) & (grid[row, :, 1] != -1)
        cols, score = detect_v1.prox_scores(grid, row, mask, centre)
        finite = np.isfinite(score)
        total += int(finite.sum())
        for vx in THRESHOLDS_VX:
            strong = finite & (score >= floor_of(vx))
            strong_pts[vx] += int(strong.sum())
            ev = evidence[vx]
            for col, s in zip(cols[strong], score[strong]):
                key = (row, int(col) // detect_v1.BLOCK)
                ev[key] = ev.get(key, 0.0) + float(s)
    return evidence, strong_pts, total


def clusters_of(evidence):
    """Union-find clusters of evidence cells — detect_v1's connectivity
    (±CELL_ROW_GAP rows, ±1 block), no tall-drop, no differencing."""
    parent = {key: key for key in evidence}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for row, blk in evidence:
        for dr in range(-detect_v1.CELL_ROW_GAP, detect_v1.CELL_ROW_GAP + 1):
            for db in (-1, 0, 1):
                other = (row + dr, blk + db)
                if other != (row, blk) and other in evidence:
                    parent[find(other)] = find((row, blk))

    grouped = {}
    for key in evidence:
        grouped.setdefault(find(key), []).append(key)
    return [(cells, sum(evidence[c] for c in cells))
            for cells in grouped.values()]


def retention(top_a, top_b):
    """Share of top_a clusters retained on top_b's cells (±1 row dilation,
    >= 50 % of cells — the on-atlas rule of differenced_clusters)."""
    dilated = {(seg, r + dr, b)
               for seg, cells, _ in top_b for r, b in cells for dr in (-1, 0, 1)}
    kept = sum(1 for seg, cells, _ in top_a
               if sum((seg, r, b) in dilated for r, b in cells) * 2
               >= len(cells))
    return kept / len(top_a) if top_a else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    names = sorted({r['segment'] for r in manifest['injections']
                    if r['winding_low'] < 100})

    os.makedirs(args.checkpoint, exist_ok=True)
    per_segment = {}
    for name in names:
        ckpt = os.path.join(args.checkpoint, f'{name}.pkl')
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                per_segment[name] = pickle.load(f)
            print(f'{name}: from checkpoint', flush=True)
            continue
        payload = segment_atlases(name, scroll, args.grid_cache, centre)
        tmp = f'{ckpt}.{os.getpid()}.part'
        with open(tmp, 'wb') as f:
            pickle.dump(payload, f)
        os.replace(tmp, ckpt)
        per_segment[name] = payload
        print(f'{name}: done ({payload[2]} skeleton points)', flush=True)

    # Aggregate: global cell sets and global top-K clusters per threshold.
    report = {'thresholds_vx': list(THRESHOLDS_VX), 'reference_vx': REFERENCE_VX,
              'top_k': TOP_K, 'segments': len(names), 'per_threshold': {}}
    cell_sets, tops = {}, {}
    total_points = sum(per_segment[n][2] for n in names)
    for vx in THRESHOLDS_VX:
        cells = {(n, r, b) for n in names for r, b in per_segment[n][0][vx]}
        cell_sets[vx] = cells
        strong = sum(per_segment[n][1][vx] for n in names)
        all_clusters = []
        for n in names:
            for cl, mass in clusters_of(per_segment[n][0][vx]):
                all_clusters.append((n, cl, mass))
        all_clusters.sort(key=lambda t: -t[2])
        tops[vx] = all_clusters[:TOP_K]
        report['per_threshold'][f'{vx:g}vx'] = {
            'contact_points': strong,
            'contact_share': round(strong / max(total_points, 1), 5),
            'evidence_cells': len(cells),
            'clusters_total': len(all_clusters),
            'top_mass_min': round(all_clusters[:TOP_K][-1][2], 2)
            if len(all_clusters) >= TOP_K else None,
        }

    ref = REFERENCE_VX
    report['cell_jaccard_vs_ref'] = {
        f'{vx:g}vx': round(len(cell_sets[vx] & cell_sets[ref])
                           / max(len(cell_sets[vx] | cell_sets[ref]), 1), 4)
        for vx in THRESHOLDS_VX if vx != ref}
    retentions = {}
    for vx in THRESHOLDS_VX:
        if vx == ref:
            continue
        retentions[f'top{TOP_K}_{ref:g}vx_retained_at_{vx:g}vx'] = retention(
            tops[ref], tops[vx])
        retentions[f'top{TOP_K}_{vx:g}vx_retained_at_{ref:g}vx'] = retention(
            tops[vx], tops[ref])
    report['retention'] = {k: round(v, 3) for k, v in retentions.items()}

    # Descriptive extra (added after the verdict rule, does not feed it):
    # the threshold-robust core — reference top-K clusters retained at BOTH
    # other thresholds. If the map must be published, this is the part that
    # survives any of the three reasonable thresholds.
    others = [vx for vx in THRESHOLDS_VX if vx != ref]
    dilated = {}
    for vx in others:
        dilated[vx] = {(seg, r + dr, b) for seg, cells, _ in tops[vx]
                       for r, b in cells for dr in (-1, 0, 1)}
    core = sum(
        1 for seg, cells, _ in tops[ref]
        if all(sum((seg, r, b) in dilated[vx] for r, b in cells) * 2
               >= len(cells) for vx in others))
    report['robust_core_of_ref_topk'] = core

    values = list(retentions.values())
    if all(v >= RETAIN_MIN_STABLE for v in values):
        verdict = 'стабильна'
    elif any(v < RETAIN_MAX_DEPENDENT for v in values):
        verdict = 'порого-зависима'
    else:
        verdict = 'частично устойчива'
    report['verdict'] = verdict
    report['verdict_rule'] = (
        f'declared before the run: стабильна if all retentions >= '
        f'{RETAIN_MIN_STABLE}; порого-зависима if any < '
        f'{RETAIN_MAX_DEPENDENT}; else частично устойчива')

    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
