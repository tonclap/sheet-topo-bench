"""Jackknife uncertainty for the published ΔERL@N numbers (TARGET item 5).

ΔERL@N is corpus-wide (one number per run, not one per injection), so the
injection bootstrap that widened AP and the recalls cannot reach it. The
resampling unit that exists above it is the segment: candidates are stored
per segment in the run's own checkpoints, the merge is deterministic, and
sheet_erl is a sum over segments — leave one segment out, re-merge, re-rank,
re-evaluate, and the spread of the leave-one-out statistics is a standard
jackknife. Nothing is re-detected: every number is rebuilt from the frozen
run's checkpoints, and the full-corpus reproduction is asserted against the
published report before any interval is printed.

Statistics jackknifed: delta_erl_mm and its fraction of the oracle ceiling
(the two forms HELDOUT_RESULTS publishes). 95 % CI = point ± 1.96 * SE_jack.

Usage:
    python pipeline/detector/jackknife_erl.py --topo output/topo \
        --grid-cache output/figgrids --out output/topo/erl_jackknife.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
sys.path.insert(0, _HERE)
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402
import detect_v1                                                      # noqa: E402

RUNS = {
    'dev': ('corpus_paris4', 'ckpt_paris4_dev', 'detector_v1_paris4.json',
            lambda r: r['winding_low'] < 100),
    'paris4_heldout': ('corpus_paris4', 'ckpt_paris4_heldout',
                       'detector_v1_paris4_heldout.json',
                       lambda r: r['winding_low'] >= 100),
    'pherc0139': ('corpus_0139', 'ckpt_0139', 'detector_v1_0139.json',
                  lambda r: True),
}


def evaluate(names, cand_of, injections, manifest, grids, voxel_mm):
    scoped = dict(manifest, injections=injections)
    n = len(injections)
    cands = [c for name in names for c in cand_of[name]]
    ranking = detect_v1.merge_channels(cands, top=4 * n)
    pristine = {name: grids[name] for name in names}
    zones = {name: sheet_erl.segment_zones(scoped, name) for name in names}
    erl_broken, *_ = sheet_erl.sheet_erl(pristine, zones, voxel_mm, 1)
    result = sheet_erl.evaluate_ranking(ranking, scoped, pristine, voxel_mm,
                                        1, erl_broken=erl_broken)
    oracle_rank = [(r['segment'], (r['row_lo'] + r['row_hi']) // 2,
                    (r['col_lo'] + r['col_hi']) // 2, 1.0) for r in injections]
    oracle = sheet_erl.evaluate_ranking(oracle_rank, scoped, pristine,
                                        voxel_mm, 1, erl_broken=erl_broken)
    delta = result['delta_erl_mm']
    ceiling = oracle['delta_erl_mm']
    return delta, (delta / ceiling if ceiling > 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topo', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    report = {}
    for key, (corpus, ckpt, published, in_scope) in RUNS.items():
        manifest_path = os.path.join(args.topo, corpus, 'manifest.json')
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        scroll = scrolls.SCROLLS[manifest['scroll']]
        injections = [r for r in manifest['injections'] if in_scope(r)]
        names = sorted({r['segment'] for r in injections})
        cand_of = {}
        for name in names:
            with open(os.path.join(args.topo, ckpt, f'{name}.pkl'), 'rb') as f:
                cand_of[name] = pickle.load(f)['candidates']
        grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
                 for name in names}
        voxel_mm = manifest['voxel_mm']

        full_delta, full_frac = evaluate(names, cand_of, injections, manifest,
                                         grids, voxel_mm)
        with open(os.path.join(args.topo, published), encoding='utf-8') as f:
            published_delta = json.load(f)['metrics']['delta_erl_mm']
        if abs(full_delta - published_delta) > 1e-9:
            raise SystemExit(
                f'{key}: reproduction mismatch {full_delta} vs published '
                f'{published_delta} — checkpoints do not rebuild the run, '
                f'no interval')

        thetas_d, thetas_f = [], []
        for name in names:
            rest = [n for n in names if n != name]
            inj = [r for r in injections if r['segment'] != name]
            d, frac = evaluate(rest, cand_of, inj, manifest, grids, voxel_mm)
            thetas_d.append(d)
            thetas_f.append(frac)
            print(f'{key}: -{name}: delta {d:+.4f}', flush=True)

        m = len(names)
        def jack(thetas, point):
            thetas = np.array([t for t in thetas if t is not None], float)
            se = float(np.sqrt((m - 1) / m
                               * ((thetas - thetas.mean()) ** 2).sum()))
            return {'point': point, 'se_jack': se,
                    'ci95': [point - 1.96 * se, point + 1.96 * se],
                    'leave_one_out': [round(float(t), 5) for t in thetas]}
        report[key] = {
            'segments': m,
            'delta_erl_mm': jack(thetas_d, full_delta),
            'delta_erl_fraction_of_oracle': jack(thetas_f, full_frac),
        }
        print(f'{key}: delta {full_delta:+.3f} ± {report[key]["delta_erl_mm"]["se_jack"]:.3f} (SE)',
              flush=True)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {s: v[s]['ci95'] for s in
                          ('delta_erl_mm', 'delta_erl_fraction_of_oracle')}
                      for k, v in report.items()}, indent=2))


if __name__ == '__main__':
    main()
