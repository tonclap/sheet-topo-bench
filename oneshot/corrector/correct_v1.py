"""Corrector v1: the Zung detector->corrector step, measured honestly. TOPO-022.

The metric harness already models a *successful* repair: PROTOCOL §3's
delta-ERL@N replaces a hit injection's zone with a zero-width break at the
detected column (the cut heals the topology, the scar stays a run boundary).
What it deliberately does not model — and what a corrector must — is the cost
of being wrong: a cut at a false-positive window severs a healthy trace, and
severed healthy runs are exactly how ERL punishes vandalism. This script is
that model, applied to every window the corrector acts on, not only the hits:

- **hit window** (§3 rule, credited once per injection): the injection's zone
  is removed (repaired) and a one-column break over the injection's rows is
  added — byte-for-byte the cut of evaluate_ranking;
- **false window**: a one-column break over FALSE_CUT_ROWS rows centred on
  the window — the same scar geometry an injected window would get (heights
  <= ~11 rows by PROTOCOL §4), charged against a trace that was healthy.

Comparators, same charging rules: the **oracle** cuts at the true windows
(ceiling — every cut is a hit by construction) and **random windows** (floor —
cuts land on healthy trace, minus luck). Reported over a sweep of k, because
"how many detections to act on" is the corrector's only free parameter and
the honest answer is a curve, not a chosen point: as k grows past the
detector's precision, false cuts eat the repairs.

The candidate pool is read from the detect_v2 checkpoint directory (the same
frozen candidates the detector reports rank; deterministic, mesh-only, no
prediction sampling), restricted to the v1 channels so the corrector measures
the *frozen* detector, not the TOPO-023 experiment.

Usage (from oneshot/corrector/):

    python correct_v1.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids \
        --checkpoints ../../output/topo/ckpt_paris4_dev_v2 --bands dev \
        --report ../../output/topo/corrector_v1_paris4.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v1 as v1                                                # noqa: E402

FALSE_CUT_ROWS = 12      # scar height of a cut on healthy trace: the tallest
                         # injected window is ~11 rows (PROTOCOL §4), and the
                         # corrector cannot know its window was false — the
                         # scar must cost the same either way
RANDOM_SEEDS = 100


def corrected_erl(ranking_slice, injections, zones, grids, voxel_mm, row_step,
                  scar=True):
    """(ERL after the corrector acts on every window of the slice, helpful,
    harmful) under the charging rules of the module docstring.

    `scar=False` is the post-hoc sensitivity of 17.08.2026, added after the
    scar-model sweep came out negative at every k and is labelled as such in
    the report: a repair restores continuity outright (the zone is removed,
    no break is added — the resew is perfect), while a false cut still leaves
    its scar. The two models bracket a real resew tool: the scar model
    under-credits repairs (oracle ceiling +0.21 mm against the +12.4 mm
    distance from broken to pristine ERL), the no-scar model over-credits
    them. The conclusion is only trustworthy where both models agree on the
    sign.
    """
    outcomes = sheet_erl.hits(ranking_slice, injections, grids)
    cut_zones = {name: list(z) for name, z in zones.items()}
    helpful = harmful = 0
    for hit, segment, row, col in outcomes:
        if hit is not None:
            r = injections[hit]
            cut_zones[segment] = [z for z in cut_zones[segment]
                                  if z != (r['row_lo'], r['row_hi'],
                                           r['col_lo'], r['col_hi'])]
            if scar:
                cut_zones[segment].append((r['row_lo'], r['row_hi'],
                                           col, col + 1))
            helpful += 1
        else:
            half = FALSE_CUT_ROWS // 2
            cut_zones.setdefault(segment, []).append(
                (max(row - half, 0), row + half, col, col + 1))
            harmful += 1
    erl, *_ = sheet_erl.sheet_erl(grids, cut_zones, voxel_mm, row_step)
    return erl, helpful, harmful


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--checkpoints', required=True,
                        help='detect_v2 checkpoint directory holding the '
                             'frozen candidate pool')
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--report', required=True)
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--no-scar', action='store_true',
                        help='post-hoc sensitivity (17.08.2026): repairs '
                             'restore continuity outright, false cuts still '
                             'scar; see corrected_erl docstring')
    parser.add_argument('--skip-random', action='store_true',
                        help='reuse the scar run\'s random floor (hits are '
                             'negligible in random rankings, the floor is '
                             'insensitive to the repair model)')
    args = parser.parse_args()
    scar = not args.no_scar

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
    names = sorted({r['segment'] for r in injections})
    n = len(injections)

    candidates = []
    for name in names:
        path = os.path.join(args.checkpoints, f'{name}.pkl')
        with open(path, 'rb') as f:
            payload = pickle.load(f)
        candidates += [c for c in payload['candidates'] if c[3] != 'front']
    print(f'{len(candidates)} v1-channel candidates from '
          f'{len(names)} checkpoints')

    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}
    zones = {name: sheet_erl.segment_zones(
        dict(manifest, injections=injections), name) for name in names}
    erl_broken, total_mm, _ = sheet_erl.sheet_erl(grids, zones, voxel_mm,
                                                  args.row_step)
    erl_pristine, *_ = sheet_erl.sheet_erl(grids, {}, voxel_mm, args.row_step)

    ranking = v1.merge_channels(candidates, top=4 * n)

    sweep = []
    for k in (n // 2, n, 2 * n, 4 * n):
        erl, helpful, harmful = corrected_erl(
            ranking[:k], injections, zones, grids, voxel_mm, args.row_step,
            scar=scar)
        sweep.append({'k': k, 'delta_erl_mm': erl - erl_broken,
                      'helpful_cuts': helpful, 'harmful_cuts': harmful})
        print(f'k={k}: dERL {erl - erl_broken:+.3f} mm, '
              f'{helpful} repairs / {harmful} false cuts', flush=True)

    oracle_rank = [(r['segment'], (r['row_lo'] + r['row_hi']) // 2,
                    (r['col_lo'] + r['col_hi']) // 2, 1.0) for r in injections]
    erl_o, helpful_o, harmful_o = corrected_erl(
        oracle_rank, injections, zones, grids, voxel_mm, args.row_step,
        scar=scar)
    oracle = {'k': n, 'delta_erl_mm': erl_o - erl_broken,
              'helpful_cuts': helpful_o, 'harmful_cuts': harmful_o}
    print(f'oracle: dERL {oracle["delta_erl_mm"]:+.3f} mm')

    randoms = {}
    if not args.skip_random:
        pool = [(name, row, col) for name, grid in sorted(grids.items())
                for row, col in sheet_erl.windows_of(grid)]
        for k in (n // 2, n, 2 * n, 4 * n):
            deltas = []
            for seed in range(RANDOM_SEEDS):
                rng = np.random.default_rng(seed)
                picks = rng.choice(len(pool), size=min(k, len(pool)),
                                   replace=False)
                erl_r, *_ = corrected_erl([(*pool[i], 0.0) for i in picks],
                                          injections, zones, grids, voxel_mm,
                                          args.row_step, scar=scar)
                deltas.append(erl_r - erl_broken)
            randoms[str(k)] = {
                'median': float(np.median(deltas)),
                'iqr': [float(np.percentile(deltas, 25)),
                        float(np.percentile(deltas, 75))]}
            print(f'random k={k}: median dERL '
                  f'{randoms[str(k)]["median"]:+.3f} mm', flush=True)

    best = max(sweep, key=lambda s: s['delta_erl_mm'])
    report = {
        'corrector': ('v1 zone-model cuts (TOPO-022): hit = repair + scar, '
                      'false = scar on healthy trace' if scar else
                      'v1 zone-model cuts, NO-SCAR sensitivity (post hoc, '
                      '17.08.2026): hit = perfect resew, false = scar'),
        'repair_model': 'scar' if scar else 'no-scar',
        'false_cut_rows': FALSE_CUT_ROWS,
        'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
        'channel_pool': 'v1 (prox+rect+support), frozen',
        'erl_pristine_mm': erl_pristine, 'erl_broken_mm': erl_broken,
        'skeleton_total_mm': total_mm,
        'sweep': sweep, 'oracle': oracle, 'random': randoms,
        'summary': {
            'best_k': best['k'],
            'best_delta_erl_mm': best['delta_erl_mm'],
            'oracle_share': (best['delta_erl_mm'] / oracle['delta_erl_mm']
                             if oracle['delta_erl_mm'] > 0 else None)}}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    share = report['summary']['oracle_share']
    print(f"best k={best['k']}: dERL {best['delta_erl_mm']:+.3f} mm = "
          f"{share:.0%} of the oracle's {oracle['delta_erl_mm']:+.3f}"
          if share is not None else 'oracle delta is zero')
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
