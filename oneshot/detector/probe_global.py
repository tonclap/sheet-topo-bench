"""TOPO-024 mechanism probe: do the wave-2 global features cover what v1 misses?

The question behind TOPO-024 is whether the global-coordinate features
(self-intersections, ray winding-order inversions — already built as §7
baselines) can lift the locally-plausible subgroup (recall 0.610 on dev).
The v2add ablation of TOPO-023 taught the cost side: an extra percentile
channel dilutes the merge significantly (dAP -0.135) even when its feature
works. So before any channel is built, this probe measures the benefit side
alone, at the mechanism level, with no ranking involved:

    Of the injections v1 MISSES (per_injection_rank absent or > N in the
    frozen report), what share has atlas-differenced global evidence inside
    its window? And of those v1 finds, ditto?

If the missed injections are not covered, no merge scheme can help and
TOPO-024's answer is negative for free. If they are covered, the follow-up
(how to merge without dilution) becomes worth an experiment. Windows are
grown by the same margin the TOPO-023 probe used (±2 rows, ±8 cols).

Usage (from oneshot/detector/):

    python probe_global.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --v2-report ../../output/topo/detector_v2_paris4.json --bands dev \
        --report ../../output/topo/probe_global_paris4.json
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
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import baselines                                                      # noqa: E402

ROW_MARGIN = 2
COL_MARGIN = 8


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--v2-report', required=True,
                        help='detect_v2 report whose ablation_v1_config '
                             'says which injections v1 finds')
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.v2_report, encoding='utf-8') as f:
        v1_ranks = json.load(f)['ablation_v1_config']['per_injection_rank']
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)

    def in_scope(record):
        if args.bands == 'dev':
            return record['winding_low'] < 100
        if args.bands == 'heldout':
            return record['winding_low'] >= 100
        return True

    injections = [r for r in manifest['injections'] if in_scope(r)]
    names = sorted({r['segment'] for r in injections})
    n = len(injections)
    found_by_v1 = {i for i, rank in v1_ranks.items()
                   if rank is not None and rank <= n}

    clusters = {ch: {} for ch in baselines.CHANNELS}
    for name in names:
        corrupted = np.load(os.path.join(args.corpus, 'grids', f'{name}.npy'))
        pristine = scrolls.segment_grid(name, scroll, args.grid_cache)
        for ch in baselines.CHANNELS:
            evidence, best = baselines.channel_cells(corrupted, centre, ch)
            atlas, _ = baselines.channel_cells(pristine, centre, ch)
            clusters[ch][name] = baselines.candidates_of(
                name, evidence, best, atlas=set(atlas))
        print(f'{name}: '
              + ', '.join(f'{ch}={len(clusters[ch][name])}'
                          for ch in baselines.CHANNELS), flush=True)

    def in_window(r, cand):
        _, row, col, _ = cand
        return (r['row_lo'] - ROW_MARGIN <= row < r['row_hi'] + ROW_MARGIN
                and r['col_lo'] - COL_MARGIN <= col < r['col_hi'] + COL_MARGIN)

    report = {'bands': args.bands, 'n_injections': n,
              'v1_found': len([r for r in injections
                               if r['id'] in found_by_v1]),
              'margins': [ROW_MARGIN, COL_MARGIN], 'coverage': {}}
    for ch in baselines.CHANNELS:
        entry = {}
        for label, subset in (
                ('all', injections),
                ('plausible', [r for r in injections if r.get('plausible')]),
                ('missed_by_v1', [r for r in injections
                                  if r['id'] not in found_by_v1]),
                ('plausible_missed_by_v1',
                 [r for r in injections if r.get('plausible')
                  and r['id'] not in found_by_v1]),
                ('M_missed_by_v1',
                 [r for r in injections if r['type'] == 'M'
                  and r['id'] not in found_by_v1])):
            hit = sum(1 for r in subset
                      if any(in_window(r, c)
                             for c in clusters[ch][r['segment']]))
            entry[label] = {'n': len(subset), 'covered': hit,
                            'share': round(hit / len(subset), 4)
                            if subset else None}
        n_clusters = sum(len(c) for c in clusters[ch].values())
        entry['clusters_total'] = n_clusters
        report['coverage'][ch] = entry
        print(f"{ch}: " + ', '.join(
            f"{k}={v['covered']}/{v['n']}" for k, v in entry.items()
            if isinstance(v, dict)))

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
