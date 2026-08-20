"""TOPO-026 mechanism probe: does raw CT intensity cover the mergers v1 misses?

Every channel to date consumed the pair (mesh, surface prediction); TOPO-023/024
measured that pair's information boundary. CT intensity is the one PROTOCOL §3
input no channel has touched, and it is not circular to any prediction: an
injected merger parks the trace mid-gap (inject_errors.apply_window, fraction
0.5), so in the *scan itself* the trace sits in air (~50 inside the mask)
with papyrus fronts half a pitch away on both radial sides, while a healthy
trace sits on papyrus. The candidate node evidence is therefore

    in_air(node)  AND  papyrus flank on both radial sides,

the front channel's geometry (detect_v2.FRONT_T) with the prediction replaced
by the CT volume. Per the TOPO-023 lesson, no channel is built here: this
probe measures the benefit side alone — of the M injections v1 misses on dev
(36 of 79), how many windows contain at least one CT-evidence cluster that
survives the standard atlas differencing? Cross-talk to S and H windows is
measured alongside (an H window has no valid nodes, so it should read zero by
construction; an S window's splice ramps cross the mid-gap and may fire).

**Declared before the run, not tuned on the result:**

- Geometry: per node, CT is sampled along the in-plane radial unit vector
  from the counting centre only (sheets separate radially; z runs along the
  sheet, so no z slack is taken). Stage 1 = offsets ``STAGE1_T`` (±2 vx) —
  the node's own papyrus peak; stage 2 = offsets ``FLANK_T`` (3–18 vx, the
  v2 front constants verbatim) each side.
- Threshold ``T`` between air and papyrus: calibrated on the *pristine*
  substrate before any window is looked at — papyrus reference = stage-1 peak
  at trace nodes, air reference = minimum over the interior flank offsets
  ``AIR_T`` (5–15 vx, both sides, zeros = off-mask excluded); ``T`` is the
  midpoint of the two medians, one global number, published with both
  distributions. No quantity below depends on the injections.
- Node evidence: ``0 < stage1_peak < T`` (in air, but inside the scan mask)
  and both flank maxima ``>= T``. Cells, surplus floor (0.5), clustering and
  atlas differencing are v1's own (BLOCK, differenced_clusters), applied
  inside each injection window grown by (±2 rows, ±8 cols) — the margins
  probe_global used. Outside windows corrupted == pristine, so the surplus is
  identically zero there and windows are the entire evidence surface.
- **Decision rule: the channel is worth building iff at least one third of
  the missed-M windows (>= 12 of 36) contain a surviving CT cluster.** Below
  that the probe's answer is negative and TOPO-026 closes without a channel,
  like TOPO-024 did (its coverage was 8-11 of 70).

Checkpoints: per-window JSONL (append-only, resume skips finished ids) and a
per-segment pickle for the calibration pass — the CT cache is part-cold and
the network drops (net_retry is imported first thing).

Usage (from oneshot/detector/):

    python probe_ct.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --v2-report ../../output/topo/detector_v2_paris4.json --bands dev \
        --report ../../output/topo/probe_ct_paris4.json \
        --details ../../output/topo/probe_ct_paris4_windows.jsonl \
        --ckpt ../../output/topo/ckpt_ct_probe
"""
import argparse
import collections
import json
import os
import pickle
import sys
from concurrent import futures

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
import detect_v1 as v1                                                # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)
import absolute_winding_calibration as awc                            # noqa: E402

STAGE1_T = (-2.0, -1.0, 0.0, 1.0, 2.0)   # node's own radial neighbourhood, vx
FLANK_T = (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 18.0)   # detect_v2.FRONT_T
AIR_T = (5.0, 7.0, 9.0, 11.0, 13.0, 15.0)  # interior offsets for the air reference
ROW_MARGIN = 2            # window growth for the evidence surface (probe_global's)
COL_MARGIN = 8
SURPLUS_MIN = 0.5         # per-cell surplus floor, the support/front channels' own
BG_ROW_STRIDE = 20        # calibration subsample of the pristine substrate
BG_COL_STRIDE = 6
DECISION_MIN_SHARE = 1 / 3   # the pre-declared bar on missed-M coverage


class CTVolume(awc.ScanMask):
    """ScanMask that returns intensity values, vectorised, with an LRU cap.

    The parent already does the honest work — uncompressed-u1 check, 16 KB
    plane-range reads, disk cache, None-chunk = fill — and its ``inside()``
    stays byte-identical. This subclass adds a numpy gather over unique
    (chunk, plane) keys so a whole row's probes cost one plane lookup each,
    and bounds the in-memory plane dict (the parent grows it forever, which
    is fine for slices and not for a probe touching thousands of planes).
    """

    def __init__(self, cache, base, level, max_planes=30000):
        super().__init__(cache, base, level)
        self._planes = collections.OrderedDict()
        self.max_planes = max_planes

    def values(self, points_xyz, workers=16):
        pts = np.asarray(points_xyz, float)
        idx = np.rint(pts[:, ::-1]).astype(np.int64)              # xyz -> zyx
        out = np.zeros(len(idx), np.uint8)
        inside = np.ones(len(idx), bool)
        for axis in range(3):
            inside &= (idx[:, axis] >= 0) & (idx[:, axis] < self.shape[axis])
        if not inside.any():
            return out
        here = idx[inside]
        keys = np.stack([here[:, 0] // self.chunk, here[:, 1] // self.chunk,
                         here[:, 2] // self.chunk, here[:, 0] % self.chunk], 1)
        local_y = here[:, 1] % self.chunk
        local_x = here[:, 2] % self.chunk
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        need = [tuple(int(v) for v in k) for k in uniq
                if tuple(int(v) for v in k) not in self._planes]
        if len(need) > 1:
            with futures.ThreadPoolExecutor(min(workers, len(need))) as pool:
                list(pool.map(lambda k: self._plane(*k), need))
        vals = np.empty(len(here), np.uint8)
        for i, k in enumerate(uniq):
            key = tuple(int(v) for v in k)
            take = inverse == i
            vals[take] = self._plane(*key)[local_y[take], local_x[take]]
        out[inside] = vals
        while len(self._planes) > self.max_planes:
            self._planes.popitem(last=False)
        return out


def probe_rows(ct, grid, centre, row_range, col_range=None):
    """Radial CT probes for every valid node of the given rows.

    Returns per-node arrays: rows, cols, stage-1 peak, flank maxima (outward,
    inward) and the air reference (min over AIR_T both sides, zeros excluded,
    NaN where every interior probe is off-mask). One ``values`` call per row
    keeps the plane gather batched.
    """
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    heights, _ = scrolls.row_heights(grid)
    out = {'row': [], 'col': [], 'peak1': [], 'out': [], 'inn': [], 'air': []}
    air_idx = [FLANK_T.index(t) for t in AIR_T]
    for row in row_range:
        if not 0 <= row < grid.shape[0]:
            continue
        z = heights[row]
        if not np.isfinite(z):
            continue
        mask = valid[row].copy()
        if col_range is not None:
            scope = np.zeros_like(mask)
            scope[max(col_range[0], 0):col_range[1]] = True
            mask &= scope
        if not mask.any():
            continue
        try:
            cx, cy = centre.at(float(z))
        except ValueError:
            continue
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)
        radial = points[:, :2] - np.array([cx, cy])
        norm = np.linalg.norm(radial, axis=1)
        ok = norm > 1e-6
        if not ok.any():
            continue
        points, cols = points[ok], cols[ok]
        u = radial[ok] / norm[ok][:, None]
        offsets = list(STAGE1_T) + [t for t in FLANK_T] + [-t for t in FLANK_T]
        stacked = np.concatenate([
            np.concatenate([points[:, :2] + t * u, points[:, 2:3]], 1)
            for t in offsets])
        sampled = ct.values(stacked).reshape(len(offsets), len(points)).astype(float)
        n1 = len(STAGE1_T)
        nf = len(FLANK_T)
        peak1 = sampled[:n1].max(axis=0)
        prof_out = sampled[n1:n1 + nf]
        prof_in = sampled[n1 + nf:]
        interior = np.concatenate([prof_out[air_idx], prof_in[air_idx]])
        interior = np.where(interior > 0, interior, np.nan)
        with np.errstate(invalid='ignore'):
            air = np.nanmin(interior, axis=0)
        out['row'].extend([row] * len(cols))
        out['col'].extend(int(c) for c in cols)
        out['peak1'].extend(float(p) for p in peak1)
        out['out'].extend(float(p) for p in prof_out.max(axis=0))
        out['inn'].extend(float(p) for p in prof_in.max(axis=0))
        out['air'].extend(float(a) if np.isfinite(a) else None for a in air)
    return out


def evidence_cells(nodes, threshold):
    """v1-shaped (evidence, best) from probed nodes at a given threshold."""
    evidence, best = {}, {}
    for row, col, peak1, mo, mi in zip(nodes['row'], nodes['col'],
                                       nodes['peak1'], nodes['out'],
                                       nodes['inn']):
        if not (0 < peak1 < threshold and mo >= threshold and mi >= threshold):
            continue
        key = (row, col // v1.BLOCK)
        evidence[key] = evidence.get(key, 0.0) + 1.0
        if best.get(key, (None, -1.0))[1] < 1.0:
            best[key] = (col, 1.0)
    return evidence, best


def window_verdict(record, threshold):
    """Surviving CT clusters of one stored window at the given threshold."""
    evidence, best = evidence_cells(record['corrupted'], threshold)
    atlas_cells, _ = evidence_cells(record['pristine'], threshold)
    surplus = {key: value - atlas_cells.get(key, 0.0)
               for key, value in evidence.items()
               if value - atlas_cells.get(key, 0.0) > SURPLUS_MIN}
    clusters = masked = 0
    for cells, mass, top in v1.differenced_clusters(surplus, best, {}):
        if cells is None:
            masked += 1
            continue
        clusters += 1
    n_evidence = int(sum(evidence.values()))
    n_atlas = int(sum(atlas_cells.values()))
    return {'clusters': clusters, 'masked': masked,
            'evidence_nodes': n_evidence, 'pristine_nodes': n_atlas}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--v2-report', required=True,
                        help='frozen detect_v2 report; its ablation_v1_config '
                             'says which injections v1 finds')
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev')
    parser.add_argument('--report', required=True)
    parser.add_argument('--details', required=True,
                        help='per-window JSONL (append-only checkpoint)')
    parser.add_argument('--ckpt', required=True,
                        help='directory for per-segment calibration checkpoints')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(args.v2_report, encoding='utf-8') as f:
        v1_ranks = json.load(f)['ablation_v1_config']['per_injection_rank']
    scroll = scrolls.SCROLLS[manifest['scroll']]
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)
    ct = CTVolume(args.cache, base=scroll.ct, level=scroll.level)

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

    os.makedirs(args.ckpt, exist_ok=True)
    ckpt_key = {'corpus': os.path.abspath(args.corpus), 'bands': args.bands,
                'probe': 'ct', 'stage1': list(STAGE1_T),
                'flank': list(FLANK_T)}

    # ---- Pass 1: calibration on the pristine substrate ---------------------
    z_q = manifest['z_quantiles']
    papyrus, airs = [], []
    for name in names:
        ck = os.path.join(args.ckpt, f'calib_{name}.pkl')
        payload = None
        if os.path.exists(ck):
            with open(ck, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('key') != ckpt_key:
                payload = None
        if payload is None:
            grid = scrolls.segment_grid(name, scroll, args.grid_cache)
            heights, _ = scrolls.row_heights(grid)
            rows = [row for row in range(0, grid.shape[0], BG_ROW_STRIDE)
                    if np.isfinite(heights[row]) and any(
                        abs(heights[row] - q) <= v1.SUPPORT_BAND_VX
                        for q in z_q)]
            sub = grid.copy()
            keep = np.zeros(grid.shape[1], bool)
            keep[::BG_COL_STRIDE] = True
            sub[:, ~keep] = -1
            nodes = probe_rows(ct, sub, centre, rows)
            payload = {'key': ckpt_key, 'nodes': nodes}
            tmp = f'{ck}.{os.getpid()}.part'
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f)
            os.replace(tmp, ck)
        nodes = payload['nodes']
        papyrus.extend(p for p in nodes['peak1'] if p > 0)
        airs.extend(a for a in nodes['air'] if a is not None)
        print(f'calib {name}: {len(nodes["col"])} nodes '
              f'({len(papyrus)} papyrus / {len(airs)} air so far)', flush=True)

    papyrus_med = float(np.median(papyrus))
    air_med = float(np.median(airs))
    threshold = 0.5 * (papyrus_med + air_med)
    calib = {'papyrus': v1.quantiles(papyrus, (0.1, 0.25, 0.5, 0.75, 0.9)),
             'air': v1.quantiles(airs, (0.1, 0.25, 0.5, 0.75, 0.9)),
             'threshold': round(threshold, 2)}
    print(f'calibration: papyrus median {papyrus_med:.1f}, air median '
          f'{air_med:.1f}, T = {threshold:.1f}', flush=True)
    if papyrus_med <= air_med + 10:
        print('calibration WARNING: papyrus and air medians are not separated '
              '— the probe is answering "CT does not separate at this '
              'resolution" (outcome 3 of TOPO-026)', flush=True)

    # ---- Pass 2: every scoped injection window -----------------------------
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
        for k, r in enumerate(injections):
            if r['id'] in done:
                continue
            corrupted, pristine = grid_pair(r['segment'])
            rows = range(r['row_lo'] - ROW_MARGIN, r['row_hi'] + ROW_MARGIN)
            cols = (r['col_lo'] - COL_MARGIN, r['col_hi'] + COL_MARGIN)
            record = {
                'id': r['id'], 'type': r['type'],
                'plausible': bool(r.get('plausible')),
                'missed_by_v1': r['id'] not in found_by_v1,
                'corrupted': probe_rows(ct, corrupted, centre, rows, cols),
                'pristine': probe_rows(ct, pristine, centre, rows, cols),
            }
            details.write(json.dumps(record, ensure_ascii=False) + '\n')
            details.flush()
            print(f'window {k + 1}/{n} {r["id"]} ({r["type"]}): '
                  f'{len(record["corrupted"]["col"])} nodes', flush=True)

    # ---- Aggregate ---------------------------------------------------------
    records = []
    with open(args.details, encoding='utf-8') as f:
        seen = set()
        for line in f:
            rec = json.loads(line)
            if rec['id'] in seen:
                continue
            seen.add(rec['id'])
            records.append(rec)
    verdicts = {}
    for rec in records:
        verdicts[rec['id']] = dict(window_verdict(rec, threshold),
                                   type=rec['type'],
                                   plausible=rec['plausible'],
                                   missed=rec['missed_by_v1'])

    def coverage(subset):
        hit = sum(1 for x in subset if verdicts[x['id']]['clusters'] > 0)
        return {'n': len(subset), 'covered': hit,
                'share': round(hit / len(subset), 4) if subset else None}

    m_all = [r for r in injections if r['type'] == 'M']
    m_missed = [r for r in m_all if r['id'] not in found_by_v1]
    subsets = {
        'M_all': coverage(m_all),
        'M_missed_by_v1': coverage(m_missed),
        'M_found_by_v1': coverage([r for r in m_all
                                   if r['id'] in found_by_v1]),
        'M_plausible_missed': coverage([r for r in m_missed
                                        if r.get('plausible')]),
        'S_all': coverage([r for r in injections if r['type'] == 'S']),
        'H_all': coverage([r for r in injections if r['type'] == 'H']),
        'all_missed_by_v1': coverage([r for r in injections
                                      if r['id'] not in found_by_v1]),
    }
    decision = (subsets['M_missed_by_v1']['covered']
                >= DECISION_MIN_SHARE * subsets['M_missed_by_v1']['n'])
    report = {
        'probe': 'ct-intensity radial profile (TOPO-026); node evidence = '
                 'in-air trace flanked by papyrus both radial sides',
        'bands': args.bands, 'n_injections': n,
        'v1_found': len([r for r in injections if r['id'] in found_by_v1]),
        'calibration': calib,
        'margins': [ROW_MARGIN, COL_MARGIN],
        'decision_rule': f'build the channel iff M_missed_by_v1 coverage >= '
                         f'{DECISION_MIN_SHARE:.3f} (declared in advance)',
        'decision_build_channel': bool(decision),
        'coverage': subsets,
        'per_window': verdicts,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    for label, entry in subsets.items():
        print(f'{label}: {entry["covered"]}/{entry["n"]} '
              f'({entry["share"]})')
    print(f'decision (pre-declared, bar {DECISION_MIN_SHARE:.2f} of missed M): '
          f'{"BUILD the channel" if decision else "NEGATIVE - no channel"}')
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
