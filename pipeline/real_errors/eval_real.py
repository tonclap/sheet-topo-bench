"""Real-corpus evaluation: the frozen mesh-only detector vs the recto/m7 map.

The rules are CORPUS.md's, declared before this ran. Only the prox channel is
scored: rect has no labels by construction (disagreement points live on valid
nodes, holes have none), and support samples the model that makes half the
label. The two substrate subtractions are undefined here — the run object *is*
the substrate — so the candidates are exactly the natural contact clusters the
synthetic benchmark subtracts away; everything else (cell geometry, cluster
connectivity, MAX_CLUSTER_ROWS, percentile-rank merge) is byte-for-byte the
frozen detector, imported rather than copied.

The synthetic comparison row runs the *same* prox-only channel through the
same evaluation code on the synthetic corpus, so the table compares like with
like; the full three-channel synthetic numbers stay in the TOPO-009 report.

AP here answers a different question than on synthetics: not "does the
detector find planted errors" but "do mesh-geometric anomalies land where two
independent surface models argue". A drop is expected and publishable either
way (PROTOCOL §5). Delta-ERL is deliberately not computed: the zones are noisy
model-disagreement labels, not confirmed mesh errors, and a usefulness number
against them would be fiction.

Usage:

    python eval_real.py --map ../../output/topo/real_paris4/disagreement.json \
        --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --zone-min 40 --report ../../output/topo/real_paris4/eval.json
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
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'winding', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..', 'detector'))
sys.path.insert(0, os.path.join(_HERE, '..', 'metric'))
import scrolls                                                        # noqa: E402
import detect_v1                                                      # noqa: E402
import sheet_erl                                                      # noqa: E402


def prox_candidates(names, grids, centre, atlas_grids=None):
    """The frozen prox channel over a set of grids.

    With `atlas_grids` (the synthetic run) each grid is differenced against its
    pristine twin, exactly as in detect_v1.segment_channels; without (the real
    run) the atlas is empty and only the frozen MAX_CLUSTER_ROWS cut applies.
    """
    candidates = []
    for name in names:
        atlas = {}
        if atlas_grids is not None:
            atlas = detect_v1.strong_cells(atlas_grids[name], centre)[0]
        evidence, best = detect_v1.strong_cells(grids[name], centre)[:2]
        for cells, mass, top in detect_v1.differenced_clusters(
                evidence, best, atlas):
            if cells is None:
                continue
            candidates.append((name, top[0], best[top][0], 'prox', mass, 1.0))
        print(f'{name}: {len(candidates)} candidates so far', flush=True)
    return candidates


def evaluate(ranking, zones, n):
    outcomes = sheet_erl.hits(ranking, zones, None)
    recall = (sum(1 for h, *_ in outcomes[:n] if h is not None) / n) if n else 0.0
    return {'ap': sheet_erl.average_precision(outcomes, n),
            'recall_at_n': recall}, outcomes


def zone_records(zone_list, threshold):
    out = [dict(z, id=f'Z{i:04d}') for i, z in enumerate(zone_list)
           if z['mass'] >= threshold]
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--zone-min', type=int, required=True,
                        help='declared zone mass threshold (CORPUS.md insert)')
    parser.add_argument('--report', required=True)
    parser.add_argument('--random-seeds', type=int, default=100)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        disagreement = json.load(f)
    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    names = disagreement['segments']
    grids = {name: scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in names}

    # ------------------------------------------------ real corpus, three thresholds
    real_candidates = prox_candidates(names, grids, centre)
    report = {'declared_zone_min': args.zone_min, 'map': os.path.abspath(args.map),
              'channel': 'prox only (CORPUS.md: rect has no labels by '
                         'construction, support is circular)',
              'real': {}}
    rng_pool = None
    for label, threshold in (('primary', args.zone_min),
                             ('half', args.zone_min // 2),
                             ('double', args.zone_min * 2)):
        zones = zone_records(disagreement['zones'], threshold)
        n = len(zones)
        ranking = detect_v1.merge_channels(real_candidates, top=4 * n)
        metrics, outcomes = evaluate(ranking, zones, n)

        randoms = []
        if rng_pool is None:
            rng_pool = [(name, row, col)
                        for name, grid in sorted(grids.items())
                        for row, col in sheet_erl.windows_of(grid)]
        for seed in range(args.random_seeds):
            rng = np.random.default_rng(seed)
            picks = rng.choice(len(rng_pool), size=min(n, len(rng_pool)),
                               replace=False)
            random_ranking = [(*rng_pool[i], 0.0) for i in picks]
            randoms.append(evaluate(random_ranking, zones, n)[0])
        entry = {'zone_min': threshold, 'n_zones': n, 'metrics': metrics,
                 'baseline_random': {
                     key: {'median': float(np.median([r[key] for r in randoms])),
                           'iqr': [float(np.percentile([r[key] for r in randoms], 25)),
                                   float(np.percentile([r[key] for r in randoms], 75))]}
                     for key in ('ap', 'recall_at_n')}}
        if label == 'primary':
            # The candidate table: zones ranked by mass, cross-referenced with
            # the detector ranking — the intersection of two independent
            # witnesses is the real-error candidate list for open problem #4.
            found_rank = {}
            for rank, (hit, segment, row, col) in enumerate(outcomes, 1):
                if hit is not None and hit not in found_rank:
                    found_rank[hit] = rank
            entry['zones_found'] = len(found_rank)
            entry['candidates'] = [
                dict(z, detector_rank=found_rank.get(k))
                for k, z in enumerate(zones)]
        report['real'][label] = entry

    # ------------------------------------------------ synthetic comparison row
    injections = [r for r in manifest['injections'] if r['winding_low'] < 100]
    syn_names = sorted({r['segment'] for r in injections})
    corrupted = {name: np.load(os.path.join(args.corpus, 'grids', f'{name}.npy'))
                 for name in syn_names}
    # The map's segment list may be a subset of the dev segments (corpus B:
    # only banner-covered segments carry zones) — the synthetic row's pristine
    # atlas is loaded by its own list, not borrowed from the map's.
    atlas = {name: grids[name] if name in grids
             else scrolls.segment_grid(name, scroll, args.grid_cache)
             for name in syn_names}
    syn_candidates = prox_candidates(syn_names, corrupted, centre,
                                     atlas_grids=atlas)
    n = len(injections)
    syn_ranking = detect_v1.merge_channels(syn_candidates, top=4 * n)
    syn_metrics, syn_outcomes = evaluate(syn_ranking, injections, n)
    found = {injections[h]['id'] for h, *_ in syn_outcomes if h is not None}
    by_type = {t: (sum(1 for r in injections if r['type'] == t and r['id'] in found)
                   / max(sum(1 for r in injections if r['type'] == t), 1))
               for t in 'SMH'}
    report['synthetic_prox_only'] = {'n_injections': n, 'metrics': syn_metrics,
                                     'recall_by_type': by_type}

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    # The summary sentence comes out of the same numbers as the table — the
    # stale-summary lesson, applied by construction.
    p = report['real']['primary']
    print(f"real corpus (zone mass >= {args.zone_min}, {p['n_zones']} zones): "
          f"AP {p['metrics']['ap']:.4f}, recall@N {p['metrics']['recall_at_n']:.3f} "
          f"(random median {p['baseline_random']['ap']['median']:.4f})")
    for label in ('half', 'double'):
        e = report['real'][label]
        print(f"  sensitivity {label} (>= {e['zone_min']}, {e['n_zones']} zones): "
              f"AP {e['metrics']['ap']:.4f}, "
              f"recall@N {e['metrics']['recall_at_n']:.3f}")
    s = report['synthetic_prox_only']
    print(f"synthetic prox-only ({s['n_injections']} injections): "
          f"AP {s['metrics']['ap']:.4f}, "
          f"recall@N {s['metrics']['recall_at_n']:.3f}, "
          f"by type " + ', '.join(f'{t}={v:.3f}' for t, v in by_type.items()))
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
