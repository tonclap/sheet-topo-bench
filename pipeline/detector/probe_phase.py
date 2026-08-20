"""TOPO-048 mechanism probe: trace phase against the CT layering (U-018).

TOPO-045 measured that the "arc left its own smooth trend" axis has no power
(mesh roughness ~ the displacement signature itself); ANGLES_2026-08-18 §6
hands the W1 remainder (22 uncovered windows: 17 M, 5 S) to U-018 — read the
CT layering, not the mesh smoothness. The honest transfer to the synthetic
corpus: injections never touch the CT, so U-018's literal form (sheet-air
period count differs by one left vs right of a pinch-out) is undefined here,
exactly as thickness-doubling was undefined (TOPO-046). What transfers is
the *trace's position within the layering*: §2 measured that 16 of the 17
invisible M sit in natural-contact zones (peak1 at papyrus level), where the
injector parks the merged trace at the middle of the annotation gap — the
CENTRE of a thick bright run — while an honest trace rides its own sheet,
offset from the run centre by about half a sheet thickness. Local intensity
is identical for both (§2); what differs is position inside the run.

Per the TOPO-023/026 lesson, no channel is built here: the probe measures
the benefit side alone, differentially against the pristine grid at the same
cells (probe_ct's surplus construction — natural centred-trace zones cancel).

**Declared before the run, not tuned on the result:**

- Geometry: probe_ct verbatim — CT sampled along the in-plane radial unit
  vector from the counting centre at the node's own z; offsets -36..+36 vx,
  step 1 (73 samples; base >= 1.5 pitch at the observed 21-45 vx pitch).
- Binarisation ``T = 80.0`` — the frozen probe_ct calibration (TOPO-026),
  no retuning. Off-mask zeros read as air.
- Bright runs: maximal contiguous stretches of samples >= T, of length
  >= ``MIN_RUN_VX = 2`` (single-voxel flickers are noise). The node's run =
  the run containing offset 0; a node outside papyrus carries no evidence.
- Node thickness = its run's length (the TOPO-046 estimator on this base);
  centredness = |offset 0 - run centre| / (run length / 2), 0 = dead centre,
  1 = run edge.
- Reference thickness: per-segment median of thickness > 0 over the
  *pristine*-grid nodes of that segment's probed windows; segments with
  < 100 such nodes fall back to the pooled median. (Windows sit in
  anomalous zones, so this reference is biased high — both grids share it,
  and the bias direction is conservative: it understates evidence.)
- **Node evidence** iff thickness >= ``THICK_REL_FLOOR = 2.0`` x reference
  (the frozen TOPO-046 doubling floor) AND centredness <= ``CENTER_FLOOR =
  0.25``; value = thickness / reference. Cells (row, col // BLOCK); per
  window, corrupted cell value minus pristine cell value must exceed
  ``SURPLUS_MIN = 0.5`` (probe_ct's floor); surviving cells cluster via
  ``differenced_clusters`` with an empty atlas. Windows grown by (+-2 rows,
  +-8 cols) — the probe_ct margins.
- **Decision rule: the channel is worth building iff at least 8 of the 22
  uncovered-union windows contain a surviving cluster** (the probe_ct
  one-third convention, TOPO-045's bar verbatim). Below that the probe's
  answer is negative and the W1 remainder falls to U-022.
- Published alongside (no part of the rule): the type split (17 M / 5 S),
  centredness-floor sensitivity at 0.15 / 0.35, a phase diagnostic (node
  distance to the *nearest* run centre >= 0.5 of the half-period, period =
  median spacing of run centres, mute if < 2 runs), and mute shares.

Checkpoints: per-window JSONL (append-only, resume skips finished ids)
storing raw per-node stats — evidence replays offline at any floor.

Usage (from pipeline/detector/):

    python probe_phase.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --coverage ../../output/topo/coverage_breakdown_paris4.json \
        --report ../../output/topo/probe_phase_paris4.json \
        --details ../../output/topo/probe_phase_windows.jsonl
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
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import detect_v1 as v1                                                # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)
from probe_ct import CTVolume                                         # noqa: E402

OFFSETS = np.arange(-36.0, 37.0)   # radial probe, 1 vx step
THICK_T = 80.0            # frozen probe_ct calibration (TOPO-026), no retuning
MIN_RUN_VX = 2            # single-voxel flickers are noise
THICK_REL_FLOOR = 2.0     # frozen TOPO-046 doubling floor
CENTER_FLOOR = 0.25       # centredness bar (sensitivity 0.15/0.35 published)
PHASE_DIAG_FLOOR = 0.5    # diagnostic only, no part of the rule
SURPLUS_MIN = 0.5         # probe_ct's per-cell surplus floor
ROW_MARGIN = 2            # window growth (probe_ct's margins)
COL_MARGIN = 8
DECISION_MIN = 8          # of 22 — TOPO-045's bar verbatim
MIN_REF_NODES = 100       # per-segment reference fallback threshold


def node_stats(profile):
    """(thickness, centredness, nearest-centre phase, n_runs) of one node.

    ``profile`` is the 73-sample radial CT read; returns thickness 0 for a
    node outside papyrus. Phase is NaN when fewer than 2 runs define no
    period.
    """
    bright = profile >= THICK_T
    runs = []
    start = None
    for i, b in enumerate(bright):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start >= MIN_RUN_VX:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(bright) - start >= MIN_RUN_VX:
        runs.append((start, len(bright) - 1))
    centre_idx = int(np.where(OFFSETS == 0.0)[0][0])
    own = next((r for r in runs if r[0] <= centre_idx <= r[1]), None)
    if own is None:
        thickness, centred = 0, None
    else:
        thickness = own[1] - own[0] + 1
        half = max(thickness / 2.0, 0.5)
        centred = abs(centre_idx - (own[0] + own[1]) / 2.0) / half
    phase = None
    if len(runs) >= 2:
        centres = np.array([(a + b) / 2.0 for a, b in runs])
        period = float(np.median(np.diff(np.sort(centres))))
        if period > 0:
            phase = float(min(abs(centres - centre_idx)) / (period / 2.0))
    return thickness, centred, phase, len(runs)


def probe_window(ct, grid, centre, rows, col_range):
    """Raw per-node stats for one window of one grid (probe_ct geometry)."""
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    out = {'row': [], 'col': [], 'thick': [], 'centred': [], 'phase': [],
           'runs': []}
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
        points = grid[row][mask].astype(np.float64)
        z_med = np.median(points[:, 2])
        try:
            cx, cy = centre.at(float(z_med))
        except ValueError:
            continue
        radial = points[:, :2] - np.array([cx, cy])
        norm = np.linalg.norm(radial, axis=1)
        ok = norm > 1e-6
        if not ok.any():
            continue
        points, cols = points[ok], cols[ok]
        u = radial[ok] / norm[ok][:, None]
        stacked = np.concatenate([
            np.concatenate([points[:, :2] + t * u, points[:, 2:3]], 1)
            for t in OFFSETS])
        sampled = ct.values(stacked).reshape(len(OFFSETS), len(points))
        for j, col in enumerate(cols):
            thickness, centred, phase, n_runs = node_stats(
                sampled[:, j].astype(float))
            out['row'].append(int(row))
            out['col'].append(int(col))
            out['thick'].append(int(thickness))
            out['centred'].append(centred)
            out['phase'].append(phase)
            out['runs'].append(int(n_runs))
    return out


def evidence_cells(nodes, ref, centre_floor):
    """v1-shaped (evidence, best) at the declared doubling + centred rule."""
    evidence, best = {}, {}
    for row, col, thickness, centred in zip(nodes['row'], nodes['col'],
                                            nodes['thick'], nodes['centred']):
        if thickness < THICK_REL_FLOOR * ref or centred is None \
                or centred > centre_floor:
            continue
        s = thickness / ref
        key = (row, col // v1.BLOCK)
        evidence[key] = evidence.get(key, 0.0) + s
        if s > best.get(key, (None, -1.0))[1]:
            best[key] = (col, s)
    return evidence, best


def phase_cells(nodes):
    """Diagnostic evidence: node far from every run centre (mid-gap phase)."""
    evidence, best = {}, {}
    for row, col, phase in zip(nodes['row'], nodes['col'], nodes['phase']):
        if phase is None or phase < PHASE_DIAG_FLOOR:
            continue
        key = (row, col // v1.BLOCK)
        evidence[key] = evidence.get(key, 0.0) + 1.0
        if best.get(key, (None, -1.0))[1] < 1.0:
            best[key] = (col, 1.0)
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--coverage', required=True,
                        help='coverage_breakdown_paris4.json (uncovered ids)')
    parser.add_argument('--report', required=True)
    parser.add_argument('--details', required=True,
                        help='per-window JSONL (append-only checkpoint)')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.coverage, encoding='utf-8') as f:
        uncovered = {r['id'] for r in
                     json.load(f)['uncovered_union']['rows']}
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
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
                'corrupted': probe_window(ct, corrupted, centre, rows, cols),
                'pristine': probe_window(ct, pristine, centre, rows, cols),
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

    # -------- reference thickness: pristine nodes, per segment, declared rule
    by_seg = {}
    for rec in records:
        by_seg.setdefault(rec['segment'], []).extend(
            t for t in rec['pristine']['thick'] if t > 0)
    pooled = [t for vals in by_seg.values() for t in vals]
    pooled_med = float(np.median(pooled)) if pooled else 0.0
    refs = {seg: (float(np.median(vals)) if len(vals) >= MIN_REF_NODES
                  else pooled_med)
            for seg, vals in by_seg.items()}
    if pooled_med <= 0:
        raise SystemExit('degenerate reference: no pristine papyrus '
                         'thickness in any window — outcome (c)')

    # -------- verdicts at the declared floor + published sensitivities
    floors = {'primary': CENTER_FLOOR, 'tight': 0.15, 'loose': 0.35}
    per_window, covered = {}, {label: [] for label in floors}
    phase_covered, mute = [], {}
    for rec in records:
        ref = refs[rec['segment']]
        entry = {'type': rec['type'], 'plausible': rec['plausible'],
                 'reference_vx': ref}
        for label, floor in floors.items():
            corr_ev, corr_best = evidence_cells(rec['corrupted'], ref, floor)
            prist_ev, _ = evidence_cells(rec['pristine'], ref, floor)
            n = surviving_clusters(corr_ev, prist_ev, corr_best)
            entry[f'clusters_{label}'] = n
            if n > 0:
                covered[label].append(rec['id'])
        corr_ph, ph_best = phase_cells(rec['corrupted'])
        prist_ph, _ = phase_cells(rec['pristine'])
        entry['clusters_phase_diag'] = surviving_clusters(corr_ph, prist_ph,
                                                          ph_best)
        if entry['clusters_phase_diag'] > 0:
            phase_covered.append(rec['id'])
        thick_arr = np.array(rec['corrupted']['thick'])
        entry['nodes'] = int(thick_arr.size)
        entry['air_share'] = round(float(np.mean(thick_arr == 0)), 4) \
            if thick_arr.size else None
        entry['phase_mute_share'] = round(float(np.mean(
            [p is None for p in rec['corrupted']['phase']])), 4) \
            if thick_arr.size else None
        per_window[rec['id']] = entry
        mute[rec['id']] = entry['air_share']

    n_windows = len(records)
    decision = len(covered['primary']) >= DECISION_MIN

    def by_type(ids):
        return {t: sum(1 for i in ids if per_window[i]['type'] == t)
                for t in ('M', 'S')}

    report = {
        'probe': 'trace phase vs CT layering (U-018 localisation): thick '
                 'run >= 2x pristine median AND trace centred in the run; '
                 'surplus vs pristine cells, probe_ct construction',
        'declaration': 'probe_phase.py header, committed before the run '
                       '(TOPO-048, twenty-sixth session); T and doubling '
                       'floor frozen from TOPO-026/046',
        'n_windows': n_windows,
        'offsets_vx': [float(OFFSETS[0]), float(OFFSETS[-1])],
        'threshold': THICK_T, 'thick_rel_floor': THICK_REL_FLOOR,
        'center_floor': CENTER_FLOOR, 'surplus_min': SURPLUS_MIN,
        'references_vx': refs, 'pooled_reference_vx': pooled_med,
        'decision_rule': f'build the channel iff >= {DECISION_MIN} of '
                         f'{n_windows} windows carry a surviving cluster '
                         f'(declared in advance)',
        'decision_build_channel': bool(decision),
        'covered': {label: {'ids': sorted(ids), 'n': len(ids),
                            'by_type': by_type(ids)}
                    for label, ids in covered.items()},
        'phase_diag_covered': {'ids': sorted(phase_covered),
                               'n': len(phase_covered),
                               'by_type': by_type(phase_covered)},
        'per_window': per_window,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label in ('primary', 'tight', 'loose'):
        ids = covered[label]
        print(f'{label:8s} (centre <= {floors[label]}): {len(ids)}/'
              f'{n_windows} covered {by_type(ids)}')
    print(f'phase diagnostic (>= {PHASE_DIAG_FLOOR} half-period): '
          f'{len(phase_covered)}/{n_windows} {by_type(phase_covered)}')
    print(f'decision (pre-declared, bar {DECISION_MIN}/{n_windows}): '
          f'{"BUILD the channel" if decision else "NEGATIVE - no channel"}')
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
