"""Detector v2: v1 + the front-count channel (U-011 in its original form). TOPO-023.

Why this channel exists: the support channel (U-011 as shipped in v1) is a
proxy. It scores "trace without a surface" — an unsupported node — but cannot
tell the merger's seat (trace parked mid-gap *between two* predicted sheets)
from a prediction hole (no sheet anywhere near). Both read "unsupported"; only
one is a merger. U-011's original statement named the discriminating feature
and v1 never built it: count the *independent fronts* of the surface
prediction across the gap. This file builds it.

**The feature.** For each node the support sampling calls unsupported, probe
the surface prediction along the in-plane radial direction (away from and
toward the counting centre) at offsets FRONT_T, each side, with the same ±1 vx
z-slack the support channel uses. A side has a front if any probe fires at or
above the scroll's threshold. Evidence = unsupported AND flanked on *both*
sides: the trace sits in a gap that verifiably has a sheet on either side of
it — the merger's seat, and the splice ramps of a switch, which cross the
mid-gap the same way (that is not a leak: v1's support channel already banked
S recall from exactly those ramps). A prediction hole has no flanks and is
dropped — which is the entire point: the false mass that support accumulates
in ragged prediction zones never enters this channel.

**Differencing, unchanged in kind.** Like support, the channel is a per-cell
surplus over the pristine grid's own flanked-unsupported count (the substrate
carries flanked-unsupported nodes of its own where annotations ride off the
prediction), then the standard cluster pass. All v1 channels, constants,
pooling, merge and evaluation are imported from detect_v1 and untouched.

**The built-in regression.** The run evaluates two configurations from the
same candidate pool: `v1` (prox + rect + support — must reproduce the frozen
detector_v1 report's metrics bit-for-bit, else this file's refactor is broken
and the run aborts loudly) and `v2` (those plus front). The written report is
the v2 configuration in v1's schema, with the v1 configuration attached under
`ablation_v1_config` so the paired bootstrap (ablation_summary.py) reads both
sides from one file. Dev bands only; held-out v2 is TOPO-025's budget, not
this file's business.

Usage (from oneshot/detector/):

    python detect_v2.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --report ../../output/topo/detector_v2_paris4.json --bands dev \
        --checkpoint ../../output/topo/ckpt_paris4_dev_v2 \
        --v1-report ../../output/topo/detector_v1_paris4.json
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
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)

# In-plane radial probe offsets, voxels, each side of the node. The merger's
# seat is mid-gap: half a pitch to either sheet, ~5-15 vx (U-011; background
# pitch 21-45 vx with 14-40 vx annotation jitter). 3 vx clears the support
# sampler's ±2 vx in-plane tolerance so a side's own sheet edge does not read
# as a flank; 18 vx stays under one full background pitch so the probe does
# not reach *past* the gap's far sheet into the next winding.
FRONT_T = (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 18.0)
FRONT_SURPLUS_MIN = 0.5   # same per-cell surplus floor as the support channel


def front_cells(grid, prediction, z_quantiles, threshold, centre):
    """Evidence cells of the front-count channel: unsupported AND flanked.

    Mirrors support_cells' row scope (protocol height bands, SUPPORT_BAND_VX)
    and its unsupported determination sample for sample, then keeps only the
    nodes with a predicted front on both radial sides. Returns the same
    (evidence, best) shape as every other channel. Rows whose z has no
    counting centre (scrolls.Centre refuses to guess) are mute — the same
    semantics every v1 channel uses for a feature with nothing to stand on.
    """
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    heights, _ = scrolls.row_heights(grid)
    support_offsets = [np.array([dx, dy, dz], float)
                       for dx in (-2, -1, 0, 1, 2)
                       for dy in (-2, -1, 0, 1, 2)
                       for dz in (-1, 0, 1)]
    evidence, best = {}, {}
    stats = {'unsupported': 0, 'flanked': 0}
    for row in range(grid.shape[0]):
        z = heights[row]
        if not np.isfinite(z) or not any(
                abs(z - q) <= v1.SUPPORT_BAND_VX for q in z_quantiles):
            continue
        mask = valid[row]
        if mask.sum() < 30:
            continue
        try:
            cx, cy = centre.at(float(z))
        except ValueError:
            continue
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)

        # Unsupported determination: identical sampling to support_cells.
        stacked = (points[None, :, :]
                   + np.stack(support_offsets)[:, None, :])
        values = prediction.sample(stacked.reshape(-1, 3))
        peak = values.reshape(len(support_offsets), -1).max(axis=0)
        unsupported = peak < threshold
        if not unsupported.any():
            continue
        stats['unsupported'] += int(unsupported.sum())

        # Radial flank probe, unsupported nodes only. The direction is the
        # in-plane unit vector from the counting centre — the transverse of
        # the sheet, the axis along which the gap's two fronts lie.
        p = points[unsupported]
        radial = p[:, :2] - np.array([cx, cy])
        norm = np.linalg.norm(radial, axis=1)
        ok = norm > 1e-6
        if not ok.any():
            continue
        p, radial, norm = p[ok], radial[ok], norm[ok]
        u = radial / norm[:, None]
        node_cols = cols[unsupported][ok]

        flanked = np.ones(len(p), bool)
        for sign in (+1.0, -1.0):
            probes = []
            for t in FRONT_T:
                for dz in (-1.0, 0.0, 1.0):
                    shifted = p.copy()
                    shifted[:, :2] += sign * t * u
                    shifted[:, 2] += dz
                    probes.append(shifted)
            sampled = prediction.sample(np.concatenate(probes))
            fired = (sampled.reshape(len(probes), -1) >= threshold).any(axis=0)
            flanked &= fired
        stats['flanked'] += int(flanked.sum())
        for col in node_cols[flanked]:
            key = (row, int(col) // v1.BLOCK)
            evidence[key] = evidence.get(key, 0.0) + 1.0
            if best.get(key, (None, -1.0))[1] < 1.0:
                best[key] = (int(col), 1.0)
    return evidence, best, stats


def front_candidates(name, grid, pristine, prediction, z_quantiles, threshold,
                     centre, stats):
    """The front channel's pooled candidates of one corrupted grid: per-cell
    surplus over the pristine grid's own flanked-unsupported counts, then the
    standard cluster pass — the exact differencing shape the support channel
    uses, so a difference in outcome is a difference in the feature."""
    evidence, best, corrupted_stats = front_cells(
        grid, prediction, z_quantiles, threshold, centre)
    atlas, _, pristine_stats = front_cells(
        pristine, prediction, z_quantiles, threshold, centre)
    stats['front_unsupported'] = corrupted_stats['unsupported']
    stats['front_flanked'] = corrupted_stats['flanked']
    stats['front_flanked_pristine'] = pristine_stats['flanked']
    surplus = {key: value - atlas.get(key, 0.0)
               for key, value in evidence.items()
               if value - atlas.get(key, 0.0) > FRONT_SURPLUS_MIN}
    out = []
    masked = 0
    for cells, mass, top in v1.differenced_clusters(surplus, best, {}):
        if cells is None:
            masked += 1
            continue
        out.append((name, top[0], best[top][0], 'front', mass, 1.0))
    stats['masked_front_clusters'] = masked
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--checkpoint', default=None,
                        help='per-segment resume checkpoints, keyed by '
                             '(corpus, bands, detector=v2); a v1 checkpoint '
                             'directory will not be trusted')
    parser.add_argument('--v1-report', default=None,
                        help='frozen v1 report of the same corpus/bands; the '
                             'run aborts if its own v1 configuration does not '
                             'reproduce it bit-for-bit (AP, recall@N)')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)

    mechanism = manifest.get('mechanism', 'turn-shift')
    if mechanism != 'turn-shift':
        # The dev ablation lives on Paris 4, and this guard used to refuse
        # anything else outright. PROTOCOL §6 (19.08, before the exam)
        # declares the B2 chain WITH detect_v2 and names the price in
        # expectation (2): on PHerc0139 the centre-dependent channels run
        # half-mute — rows whose z has no counting centre go mute by the
        # standard semantics already in front_cells (the 2002bfb
        # precedent: mute instead of killing the exam). The guard therefore
        # warns instead of refusing; the mute share lands in the report.
        # The prox channel takes the cross-mesh path exactly as detect_v1
        # runs it on this corpus (neighbour contexts below).
        print(f"warning: mechanism {manifest['mechanism']!r} — "
              f'centre-dependent v2 channels run with partial-centre '
              f'muteness (PROTOCOL §6, expectation 2)', flush=True)
    by_winding, winding_of = {}, {}
    if mechanism == 'neighbour-mesh':
        by_winding = {low: seg for seg, low, high
                      in scrolls.labelled_segments(scroll) if low == high}
        winding_of = {seg: low for low, seg in by_winding.items()}

    def in_scope(record):
        if args.bands == 'dev':
            return record['winding_low'] < 100
        if args.bands == 'heldout':
            return record['winding_low'] >= 100
        return True

    injections = [r for r in manifest['injections'] if in_scope(r)]
    scoped = dict(manifest, injections=injections)
    names = sorted({r['segment'] for r in injections})
    n = len(injections)

    prediction = scrolls.open_prediction(scroll, args.cache, max_chunks=256)
    support = (prediction, manifest['z_quantiles'], scroll.threshold)

    # Both helpers verbatim from detect_v1.main — the prox channel of the
    # neighbour-mesh corpus compares against the same cross-mesh contexts
    # there and here, so the v1 regression gate stays meaningful.
    def row_context(grid):
        heights, valid = scrolls.row_heights(grid)
        spacing = float(np.nanmedian(np.abs(np.diff(heights))))
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = 19.0
        return heights, spacing, grid, valid

    def neighbour_pair(seg):
        w = winding_of[seg]
        corrupted_ctx, pristine_ctx = [], []
        for other in (by_winding.get(w - 1), by_winding.get(w + 1)):
            if other is None:
                continue
            clean = scrolls.segment_grid(other, scroll, args.grid_cache)
            path = os.path.join(args.corpus, 'grids', f'{other}.npy')
            broken = np.load(path) if os.path.exists(path) else clean
            pristine_ctx.append(row_context(clean))
            corrupted_ctx.append(row_context(broken))
        return corrupted_ctx, pristine_ctx

    candidates, probes = [], []
    background = {'prox': [], 'vjump': []}
    pristine = {}
    substrate = {'substrate_contact_points': 0, 'skeleton_points': 0,
                 'masked_prox_clusters': 0, 'masked_rect_components': 0,
                 'masked_support_clusters': 0, 'front_unsupported': 0,
                 'front_flanked': 0, 'front_flanked_pristine': 0,
                 'masked_front_clusters': 0}
    ckpt_key = {'corpus': os.path.abspath(args.corpus), 'bands': args.bands,
                'detector': 'v2'}
    if args.checkpoint:
        os.makedirs(args.checkpoint, exist_ok=True)
    for name in names:
        pristine[name] = scrolls.segment_grid(name, scroll, args.grid_cache)
        ckpt = (os.path.join(args.checkpoint, f'{name}.pkl')
                if args.checkpoint else None)
        payload = None
        if ckpt and os.path.exists(ckpt):
            with open(ckpt, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('key') != ckpt_key:
                payload = None
        resumed = payload is not None
        if payload is None:
            corrupted = np.load(os.path.join(args.corpus, 'grids',
                                             f'{name}.npy'))
            row_scores, stats = {}, {}
            r = v1.radial_map(corrupted, centre)
            vjump = np.abs(np.diff(r, axis=0))
            neighbours = (neighbour_pair(name)
                          if mechanism == 'neighbour-mesh' else None)
            seg_candidates = v1.segment_channels(
                name, corrupted, pristine[name], centre, row_scores, stats,
                vjump=vjump, support=support, neighbours=neighbours)
            seg_candidates += front_candidates(
                name, corrupted, pristine[name], prediction,
                manifest['z_quantiles'], scroll.threshold, centre, stats)
            seg_background = {'prox': [], 'vjump': []}
            of_segment = [rec for rec in injections if rec['segment'] == name]
            seg_probes = v1.probe_segment(of_segment, row_scores, vjump,
                                          seg_background)
            payload = {'key': ckpt_key, 'candidates': seg_candidates,
                       'probes': seg_probes, 'background': seg_background,
                       'stats': stats}
            if ckpt:
                tmp = f'{ckpt}.{os.getpid()}.part'
                with open(tmp, 'wb') as f:
                    pickle.dump(payload, f)
                os.replace(tmp, ckpt)
        candidates += payload['candidates']
        probes += payload['probes']
        for ch in background:
            background[ch].extend(payload['background'][ch])
        for key in substrate:
            substrate[key] += payload['stats'].get(key, 0)
        print(f'{name}: {len(candidates)} candidates so far'
              + (' (from checkpoint)' if resumed else ''), flush=True)

    substrate['contact_share'] = round(
        substrate['substrate_contact_points']
        / max(substrate['skeleton_points'], 1), 5)

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
        return (result, by_type, recall_plausible, by_band,
                per_rank, per_injection)

    def config_block(pool):
        ranking = v1.merge_channels(pool, top=4 * n)
        (result, by_type, recall_plausible, by_band,
         per_rank, per_injection) = judged(ranking)
        return {'metrics': result, 'recall_by_type': by_type,
                'recall_by_band': by_band,
                'recall_on_locally_plausible': recall_plausible,
                'per_rank_outcomes': per_rank,
                'per_injection_rank': per_injection}

    # Three configurations from one candidate pool. 'v2add' (all four
    # channels) is the naive form; the smoke run showed it dilutes exactly the
    # way U-012 warned (each channel gets equal percentile footing, front's
    # candidates displace top-N slots without adding finds). 'v2' is the
    # hypothesis-shaped form: front IS support minus the prediction holes
    # (its nodes are a subset of support's), so replacing support with front
    # tests "support's false mass costs us top-N slots" directly.
    v1_pool = [c for c in candidates if c[3] != 'front']
    v1_block = config_block(v1_pool)
    v2_block = config_block([c for c in candidates if c[3] != 'support'])
    v2add_block = config_block(candidates)

    if args.v1_report:
        with open(args.v1_report, encoding='utf-8') as f:
            frozen = json.load(f)
        ours, theirs = v1_block['metrics'], frozen['metrics']
        if (round(ours['ap'], 6) != round(theirs['ap'], 6)
                or round(ours['recall_at_n'], 6)
                != round(theirs['recall_at_n'], 6)):
            raise SystemExit(
                f"v1 regression FAILED: this run's v1 configuration reads "
                f"AP {ours['ap']:.6f} / recall {ours['recall_at_n']:.6f} "
                f"against the frozen {theirs['ap']:.6f} / "
                f"{theirs['recall_at_n']:.6f} — the refactor changed v1 "
                f"behaviour, nothing about the front channel is "
                f"interpretable until this is fixed")
        print(f"v1 regression OK: AP {ours['ap']:.4f} == frozen "
              f"{theirs['ap']:.4f}")

    front_only = [c for c in candidates if c[3] == 'front']
    front_probe = {
        'candidates': len(front_only),
        'per_type_window_hits': {},
    }
    # Mechanism record: does a merger window actually contain front evidence?
    by_cell = {}
    for name_, row, col, _, mass, _ in front_only:
        by_cell.setdefault(name_, []).append((row, col, mass))
    for t in 'SMH':
        of_type = [r for r in injections if r['type'] == t]
        hit = 0
        for r in of_type:
            cells = by_cell.get(r['segment'], [])
            if any(r['row_lo'] - 2 <= row < r['row_hi'] + 2
                   and r['col_lo'] - 8 <= col < r['col_hi'] + 8
                   for row, col, _ in cells):
                hit += 1
        front_probe['per_type_window_hits'][t] = (
            round(hit / len(of_type), 4) if of_type else None)

    report = {'detector': 'v2 = v1 + front-count channel (U-011 original '
                          'form, TOPO-023)',
              'lineage': 'v1 frozen 14.08 (FREEZE_2026-08-14.md); front '
                         'channel added on dev only, held-out untouched '
                         '(TOPO-025)',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              **v2_block,
              'ablation_v1_config': v1_block,
              'ablation_v2add_config': v2add_block,
              'front_probe': front_probe,
              'substrate': substrate}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, block in (('v1', v1_block), ('v2 (front replaces support)',
                                            v2_block),
                         ('v2add (all four)', v2add_block)):
        m, t = block['metrics'], block['recall_by_type']
        print(f"{label} on {args.bands}: AP {m['ap']:.4f}, "
              f"recall@N {m['recall_at_n']:.3f}, "
              + ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=n/a'
                          for k, v in t.items())
              + f", plausible={block['recall_on_locally_plausible']:.3f}")
    print(f"front channel: {front_probe['candidates']} candidates, "
          f"window hits by type {front_probe['per_type_window_hits']}")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
