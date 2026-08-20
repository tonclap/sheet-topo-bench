"""Detector v6s: split thresholds — mass vs clustering (U-016, TOPO-041).

Why this file exists: v6 (TOPO-038) relaxed every generation floor
symmetrically and the prox pool *shrank* (102 -> 96): cells in the 0.25-0.5
band bridged separate clusters (merged candidates died on MAX_CLUSTER_ROWS or
lost their granularity) and inflated the pristine atlas the same way, so the
atlas veto grew too. The measured lesson: the floors carried candidate
granularity, not noise. U-016 names the untried branch — keep the clustering
topology at the frozen floor and let only the *mass* come from the relaxed
one; or keep the relaxed floor but restrict connectivity so bridges cannot
form. This file runs both declared forms. Only the prox family changes;
rect/support are replayed from the v2 checkpoints and ct stays at the frozen
T=80 — the change is isolated by construction.

**Protocol, declared 18.08.2026 (twenty-second session) BEFORE any run of
this file (the commit adding this file precedes the first run; every
constant is fixed here, none is tuned on this file's output):**

1. **Two floors, one scoring pass.** Per-point contact scores are the frozen
   detector's own (`prox_scores`, CONTACT_REF unchanged). Cells are
   accumulated at the relaxed floor 0.25 with `row_scores` captured; the
   frozen tier (score >= 0.5) is re-derived from the captured rows by the
   same accumulation loop, so the frozen-tier cells, masses and bests are
   bit-identical to what the frozen pass would produce (the generation gate
   below proves it). The relaxed-only tier holds each cell's mass from
   scores in [0.25, 0.5).
2. **Form v6s — seeded two-floor generation:**
   - *core clusters*: the frozen-tier cells clustered and vetoed exactly as
     detect_v1 does (CELL_ROW_GAP 2, +-1 block, MAX_CLUSTER_ROWS 14,
     majority-on-dilated-atlas veto against the frozen-tier pristine atlas).
     Survivors are the frozen prox candidates — same cells, same top, same
     best col.
   - *mass augmentation*: each surviving core cluster adds the relaxed-only
     mass of (a) its own cells and (b) cells adjacent to it under the same
     connectivity template that are adjacent to exactly one surviving
     cluster — an ambiguous cell (adjacent to two clusters) is dropped, so
     bridging is impossible by construction. Position (top cell, best col)
     and the vjump factor stay the core cluster's own.
   - *new clusters*: relaxed-only cells with no frozen-tier mass and not
     adjacent to ANY frozen-tier cell form their own components (frozen
     connectivity, MAX_CLUSTER_ROWS 14), vetoed by majority against the
     dilated *relaxed* pristine atlas (frozen + relaxed-only cells) — each
     tier is differenced against its own tier's atlas. These are the
     coverage candidates the 0.5 floor could never emit.
3. **Form v6r — restricted connectivity at the relaxed floor:** all cells at
   0.25 (frozen + relaxed-only mass summed), atlas = relaxed pristine cells,
   the same majority veto, but components connect only through
   (+-1 row, same block) or (same row, +-1 block) — no diagonals, row gap 1
   (v1: gap 2 with diagonals). MAX_CLUSTER_ROWS stays 14 (frozen; v6's 20 is
   not carried). Bridges need touching cells, so sparse 0.25 skirts stop
   merging distant cores.
4. **Pools:** for each form, the prox family above replaces the v2
   checkpoints' prox in the v1 pool; rect and support are the checkpoints'
   own; ct at the frozen T=80 is folded into support by detect_v4's
   `union_pool`, verbatim. Everything else about the pool is v5lu's.
5. **Features, model, splits, seed — v5's, unchanged:** 7 features, L2 logit
   (lambda 1.0, IRLS 100, tol 1e-10), leave-one-segment-out, ranking by
   out-of-fold probability, ties by (segment, row, col), top 4N; bootstrap
   2000 resamples, seed 20260815 (the ablation summary's).
6. **Ship rule against v5lu (v6's §7, unchanged):** a form is a candidate
   for any future exam only if, against the replayed v5lu on paired
   bootstrap, AP or M improves significantly AND none of AP / recall@N / S /
   M / H / plausible degrades significantly. The frozen form v5lu and its
   TOPO-025 exam claim do NOT mutate; this file never touches held-out.
7. **Two regression gates abort the run:**
   - *harness gate*: v5lu replayed (v2 checkpoints, frozen ct at T=80,
     union pool, LOSO logit) must reproduce the frozen detector_v5 report's
     AP and recall@N to 6 decimals;
   - *generation gate*: the core clusters of §2 on the first dev segment
     must reproduce that segment's v2 checkpoint prox candidates (6-tuple,
     mass and factor included) exactly — proving the re-derived frozen tier
     IS the frozen pass until the knobs move.
8. **Coverage ceilings** (credit rule of coverage_breakdown / detect_v6's
   `covered_ids`) are measured and published for both forms, total and for
   the prox family, beside the v5 union reference.
9. **Checkpoints carry the cell tiers**, not only candidates: per segment,
   the frozen-tier and relaxed-only evidence/atlas cell maps and bests are
   stored, so any future clustering variant replays offline without
   recomputing the scoring pass.

Usage (from oneshot/detector/):

    python detect_v6s.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --probe-report ../../output/topo/probe_ct_paris4.json \
        --probe-windows ../../output/topo/probe_ct_paris4_windows.jsonl \
        --v2-checkpoints ../../output/topo/ckpt_paris4_dev_v2 \
        --v5-report ../../output/topo/detector_v5_paris4.json \
        --checkpoint ../../output/topo/ckpt_paris4_dev_v6s \
        --bands dev --report ../../output/topo/detector_v6s_paris4.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v1 as v1                                                # noqa: E402
import detect_v3 as v3                                                # noqa: E402
import detect_v4 as v4                                                # noqa: E402
import detect_v5 as v5                                                # noqa: E402
from detect_v6 import covered_ids                                     # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)

# §1 — the two floors.
FLOOR_FROZEN = 0.5      # v1.PROX_FLOOR, restated here as the clustering tier
FLOOR_RELAXED = 0.25    # v6's relaxed floor, here the mass tier only


def cells_at_floor(row_scores, floor, ceiling=None):
    """The strong_cells accumulation loop re-run from captured row scores:
    (row, block) -> summed mass and best (col, score), taking only points
    with floor <= score (< ceiling if given). Iteration order is the frozen
    pass's own (rows ascending, cols in order), so at (0.5, None) the cells,
    masses and insertion order are bit-identical to strong_cells at the
    frozen floor — the generation gate stands on this."""
    evidence, best = {}, {}
    for row in sorted(row_scores):
        cols, score = row_scores[row]
        finite = np.isfinite(score)
        take = finite & (score >= floor)
        if ceiling is not None:
            take &= score < ceiling
        for col, s in zip(cols[take], score[take]):
            key = (row, int(col) // v1.BLOCK)
            evidence[key] = evidence.get(key, 0.0) + float(s)
            if float(s) > best.get(key, (None, -1.0))[1]:
                best[key] = (int(col), float(s))
    return evidence, best


def neighbourhood(key, row_gap, diagonals):
    row, blk = key
    for dr in range(-row_gap, row_gap + 1):
        for db in (-1, 0, 1):
            if (dr, db) == (0, 0):
                continue
            if not diagonals and dr != 0 and db != 0:
                continue
            yield (row + dr, blk + db)


def components(cells, row_gap, diagonals):
    """Union-find over a cell set with an explicit connectivity template."""
    parent = {key: key for key in cells}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for key in cells:
        for other in neighbourhood(key, row_gap, diagonals):
            if other in cells:
                parent[find(other)] = find(key)
    out = {}
    for key in cells:
        out.setdefault(find(key), []).append(key)
    return list(out.values())


def cluster_pass(evidence, best, atlas_keys, row_gap, diagonals,
                 max_rows=None):
    """differenced_clusters' survivor rule with the connectivity as a knob:
    majority-on-dilated-atlas veto, MAX_CLUSTER_ROWS height cap."""
    if max_rows is None:
        max_rows = v1.MAX_CLUSTER_ROWS
    dilated = {(r + dr, b) for r, b in atlas_keys for dr in (-1, 0, 1)}
    for cells in components(set(evidence), row_gap, diagonals):
        rows = [r for r, _ in cells]
        if max(rows) - min(rows) + 1 > max_rows:
            continue
        if sum(c in dilated for c in cells) * 2 > len(cells):
            continue
        mass = sum(evidence[c] for c in cells)
        top = max(cells, key=lambda c: best[c][1])
        yield cells, mass, top


def prox_v6s(name, tiers, vjump, med):
    """§2: seeded two-floor prox. Core clusters are the frozen pass verbatim
    (via v1.differenced_clusters at module defaults); relaxed-only mass joins
    a core cluster only when unambiguous; pure-relaxed components live apart
    from any frozen-tier cell and difference against the relaxed atlas."""
    ev_f, best_f = tiers['ev_frozen'], tiers['best_frozen']
    ev_r, best_r = tiers['ev_relaxed_only'], tiers['best_relaxed']
    atlas_f = tiers['atlas_frozen']
    atlas_relaxed = set(atlas_f) | set(tiers['atlas_relaxed_only'])

    core, out = [], []
    for cells, mass, top in v1.differenced_clusters(ev_f, best_f, atlas_f):
        if cells is None:
            continue
        core.append((cells, mass, top))

    claims = {}
    for idx, (cells, _mass, _top) in enumerate(core):
        seen = set(cells)
        for cell in cells:
            for other in neighbourhood(cell, v1.CELL_ROW_GAP, True):
                if other in seen or other in ev_f or other not in ev_r:
                    continue
                seen.add(other)
                claims.setdefault(other, set()).add(idx)
    extra = {}
    for cell, owners in claims.items():
        if len(owners) == 1:
            extra.setdefault(next(iter(owners)), []).append(cell)

    for idx, (cells, mass, top) in enumerate(core):
        mass += sum(ev_r.get(c, 0.0) for c in cells)
        mass += sum(ev_r[c] for c in extra.get(idx, ()))
        factor = v1.cluster_vjump(cells, vjump, med)
        out.append((name, top[0], best_f[top][0], 'prox', mass, factor))

    frozen_adjacent = set(ev_f)
    for key in ev_f:
        frozen_adjacent.update(neighbourhood(key, v1.CELL_ROW_GAP, True))
    fresh = {k: v for k, v in ev_r.items()
             if k not in frozen_adjacent}
    for cells, mass, top in cluster_pass(fresh, best_r, atlas_relaxed,
                                         v1.CELL_ROW_GAP, True):
        factor = v1.cluster_vjump(cells, vjump, med)
        out.append((name, top[0], best_r[top][0], 'prox', mass, factor))
    return out, len(core)


def prox_v6r(name, tiers, vjump, med):
    """§3: the relaxed floor everywhere, connectivity without diagonals and
    with row gap 1; each tier's mass summed, relaxed atlas, frozen height
    cap."""
    ev = dict(tiers['ev_frozen'])
    for key, value in tiers['ev_relaxed_only'].items():
        ev[key] = ev.get(key, 0.0) + value
    best = tiers['best_relaxed']
    atlas = set(tiers['atlas_frozen']) | set(tiers['atlas_relaxed_only'])
    out = []
    for cells, mass, top in cluster_pass(ev, best, atlas, 1, False):
        factor = v1.cluster_vjump(cells, vjump, med)
        out.append((name, top[0], best[top][0], 'prox', mass, factor))
    return out


def segment_tiers(grid, pristine, centre):
    """§1: one scoring pass per grid at the relaxed floor with row capture;
    the frozen tier re-derived from the captured rows."""
    old_floor = v1.PROX_FLOOR
    v1.PROX_FLOOR = FLOOR_RELAXED
    try:
        rows_c = {}
        ev_all_c, best_c = v1.strong_cells(grid, centre, rows_c)[:2]
        rows_p = {}
        ev_all_p, _best_p = v1.strong_cells(pristine, centre, rows_p)[:2]
    finally:
        v1.PROX_FLOOR = old_floor
    ev_f, best_f = cells_at_floor(rows_c, FLOOR_FROZEN)
    ev_r_only, _ = cells_at_floor(rows_c, FLOOR_RELAXED, FLOOR_FROZEN)
    atlas_f, _ = cells_at_floor(rows_p, FLOOR_FROZEN)
    atlas_r_only, _ = cells_at_floor(rows_p, FLOOR_RELAXED, FLOOR_FROZEN)

    r = v1.radial_map(grid, centre)
    vjump = np.abs(np.diff(r, axis=0))
    med = (float(np.nanmedian(vjump)) if np.isfinite(vjump).any() else 0.0)
    tiers = {'ev_frozen': ev_f, 'best_frozen': best_f,
             'ev_relaxed_only': ev_r_only, 'best_relaxed': best_c,
             'atlas_frozen': atlas_f, 'atlas_relaxed_only': atlas_r_only}
    return tiers, vjump, med


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--probe-report', required=True)
    parser.add_argument('--probe-windows', required=True)
    parser.add_argument('--v2-checkpoints', required=True)
    parser.add_argument('--v5-report', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    if manifest.get('mechanism', 'turn-shift') != 'turn-shift':
        raise SystemExit('detect_v6s dev iteration supports the turn-shift '
                         'mechanism (Paris 4) only; any held-out is a '
                         'separate owner decision, not this file')

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
        per_rank = [None if h is None else injections[h]['id']
                    for h, *_rest in full]
        per_injection = {r['id']: None for r in injections}
        for i, (h, *_rest) in enumerate(full, 1):
            if h is not None:
                per_injection[injections[h]['id']] = i
        return {'metrics': result, 'recall_by_type': by_type,
                'recall_on_locally_plausible': recall_plausible,
                'per_rank_outcomes': per_rank,
                'per_injection_rank': per_injection}

    # §7 harness gate: v5lu replayed must reproduce the frozen v5 report.
    ckpt_key_v2 = {'corpus': os.path.abspath(args.corpus),
                   'bands': args.bands, 'detector': 'v2'}
    v2_pool = []
    v2_prox = {}
    for name in names:
        path = os.path.join(args.v2_checkpoints, f'{name}.pkl')
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        if payload.get('key') != ckpt_key_v2:
            raise SystemExit(f'{path}: checkpoint key mismatch — refusing to '
                             f'replay candidates from a different run')
        v2_pool += [tuple(c) for c in payload['candidates']]
        v2_prox[name] = [tuple(c) for c in payload['candidates']
                         if c[3] == 'prox']
    v1_pool = [c for c in v2_pool if c[3] in ('prox', 'rect', 'support')]
    rest_pool = [c for c in v1_pool if c[3] != 'prox']

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
    ct_pool = v3.ct_candidates(records, by_id, frozen_t)
    union = v4.union_pool(v1_pool, ct_pool)
    y_u = v5.label_vector(union, injections)
    scores_u = v5.crossval_scores(union, y_u, v5.fit_logit, v5.predict_logit)
    v5lu_block = judged(v5.ranking_from_scores(union, scores_u, top=4 * n))
    with open(args.v5_report, encoding='utf-8') as f:
        frozen_v5 = json.load(f)
    ours, theirs = v5lu_block['metrics'], frozen_v5['metrics']
    if (round(ours['ap'], 6) != round(theirs['ap'], 6)
            or round(ours['recall_at_n'], 6)
            != round(theirs['recall_at_n'], 6)):
        raise SystemExit(
            f"v5lu harness gate FAILED: replayed AP {ours['ap']:.6f} / "
            f"recall {ours['recall_at_n']:.6f} vs frozen {theirs['ap']:.6f} "
            f"/ {theirs['recall_at_n']:.6f}")
    print(f"v5lu harness gate OK: AP {ours['ap']:.4f} == frozen "
          f"{theirs['ap']:.4f}", flush=True)

    # Scoring passes with per-segment checkpoints (§1, §9).
    os.makedirs(args.checkpoint, exist_ok=True)
    ckpt_key = {'corpus': os.path.abspath(args.corpus), 'bands': args.bands,
                'detector': 'v6s',
                'floors': {'frozen': FLOOR_FROZEN, 'relaxed': FLOOR_RELAXED}}
    prox_s, prox_r = [], []
    gate_done = False
    for name in names:
        ckpt = os.path.join(args.checkpoint, f'{name}.pkl')
        payload = None
        if os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('key') != ckpt_key:
                payload = None
        resumed = payload is not None
        if payload is None:
            corrupted = np.load(os.path.join(args.corpus, 'grids',
                                             f'{name}.npy'))
            tiers, vjump, med = segment_tiers(corrupted, pristine[name],
                                              centre)
            seg_s, n_core = prox_v6s(name, tiers, vjump, med)
            seg_r = prox_v6r(name, tiers, vjump, med)
            payload = {'key': ckpt_key, 'tiers': tiers, 'med': med,
                       'n_core': n_core, 'v6s': seg_s, 'v6r': seg_r}
            tmp = f'{ckpt}.{os.getpid()}.part'
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f)
            os.replace(tmp, ckpt)
        # §7 generation gate on the first segment: core candidates (mass
        # WITHOUT augmentation) must equal the v2 checkpoint's prox.
        if not gate_done:
            tiers = payload['tiers']
            corrupted = np.load(os.path.join(args.corpus, 'grids',
                                             f'{name}.npy'))
            r_map = v1.radial_map(corrupted, centre)
            vjump = np.abs(np.diff(r_map, axis=0))
            med = payload['med']
            got = []
            for cells, mass, top in v1.differenced_clusters(
                    tiers['ev_frozen'], tiers['best_frozen'],
                    tiers['atlas_frozen']):
                if cells is None:
                    continue
                factor = v1.cluster_vjump(cells, vjump, med)
                got.append((name, top[0], tiers['best_frozen'][top][0],
                            'prox', mass, factor))
            if got != v2_prox[name]:
                raise SystemExit(
                    f'generation gate FAILED on {name}: frozen-tier core '
                    f'candidates diverge from the v2 checkpoint '
                    f'({len(got)} vs {len(v2_prox[name])})')
            print(f'generation gate OK on {name}: {len(got)} core '
                  f'candidates identical to the v2 checkpoint', flush=True)
            gate_done = True
        prox_s += [tuple(c) for c in payload['v6s']]
        prox_r += [tuple(c) for c in payload['v6r']]
        print(f'{name}: v6s {len(prox_s)}, v6r {len(prox_r)} prox '
              f'candidates so far' + (' (from checkpoint)' if resumed else ''),
              flush=True)

    frozen_prox = [c for c in v1_pool if c[3] == 'prox']
    print(f'prox pools: frozen {len(frozen_prox)}, v6s {len(prox_s)}, '
          f'v6r {len(prox_r)}', flush=True)

    # §4 pools, §5 ranking, §8 coverage.
    blocks, notes, coverage = {}, {}, {}
    cov_reference = covered_ids(union, injections)
    coverage['v5_union_reference'] = {
        'total': len(cov_reference),
        'prox': len(covered_ids(frozen_prox, injections))}
    for tag, this_prox in (('v6s', prox_s), ('v6r', prox_r)):
        pool = v4.union_pool(rest_pool + this_prox, ct_pool)
        y = v5.label_vector(pool, injections)
        scores = v5.crossval_scores(pool, y, v5.fit_logit, v5.predict_logit)
        blocks[tag] = judged(v5.ranking_from_scores(pool, scores, top=4 * n))
        notes[tag] = {'candidates': len(pool), 'prox': len(this_prox),
                      'positives': int(y.sum()), 'model': 'logit'}
        coverage[tag] = {
            'total': len(covered_ids(pool, injections)), 'n': n,
            'prox': len(covered_ids(this_prox, injections))}
        m = blocks[tag]['metrics']
        print(f"{tag}: AP {m['ap']:.4f}, recall@N {m['recall_at_n']:.3f}, "
              f"coverage {coverage[tag]['total']}/{n} "
              f"(prox {coverage[tag]['prox']}); "
              f"({notes[tag]['positives']}/{len(pool)} positive)", flush=True)

    report = {'detector': 'v6s = split thresholds: clustering at the frozen '
                          'floor, mass from the relaxed one (U-016, '
                          'TOPO-041); v6r = relaxed floor with restricted '
                          'connectivity',
              'lineage': 'v6 relaxed the floors symmetrically and the prox '
                         'pool shrank 102 -> 96 (bridged clusters, inflated '
                         'atlas veto); U-016 splits mass from clustering; '
                         'protocol declared in the header before the first '
                         'run',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'floors': {'frozen': FLOOR_FROZEN, 'relaxed': FLOOR_RELAXED,
                         'max_cluster_rows': v1.MAX_CLUSTER_ROWS,
                         'v6r_row_gap': 1, 'v6r_diagonals': False,
                         'ct_threshold': frozen_t},
              'pool_sizes': {'frozen_prox': len(frozen_prox),
                             'v6s_prox': len(prox_s),
                             'v6r_prox': len(prox_r)},
              'coverage': coverage,
              'models': notes,
              **blocks['v6s'],
              'ablation_v5lu_config': v5lu_block,
              'ablation_v6r_config': blocks['v6r']}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v5lu (replayed)', v5lu_block),
                         ('v6s', blocks['v6s']), ('v6r', blocks['v6r'])):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + f", plausible={block['recall_on_locally_plausible']:.3f}")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
