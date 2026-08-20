"""Detector v1: cross-trace proximity (U-008) and hole shape (U-009). TOPO-009.

Why this feature exists: v0's post-mortem. The ray ladder measured *gaps along a
ray*, where a switch collision (gap ~ 0) is indistinguishable from a polyline
return, and the return filter threw the signal away with the noise. U-008 states
the other source of information: a switch parks the trace where the *neighbouring
winding's own annotation* already lies. That is not a property of one row's
parameterisation — it is a relation between two traces, and the injector proved
it is well-defined by using it to build the corpus (`neighbour_targets_turn`).
The detector inverts the injector.

**The feature.** For each valid point of a grid row, walk the same row's polyline
one full accumulated turn outward and inward (the injector's own notion of
"neighbouring sheet" on Paris 4) and take the euclidean distance to the nearest
node of each neighbouring trace within a tight angular window. The score is
absolute contact: 1 - min(d_out, d_in) / 10 vx. A switch's trace *is* the
neighbour's annotation (d ~ 0-3 vx); the background floor is ~7-11 vx. Two
normalised variants died on the way to this (details in `prox_scores`): the
scroll's core is naturally asymmetric and the local pitch too unstable to be a
unit — every relative score saturated on healthy geometry.

**Differencing against the substrate — the load-bearing step.** The production
substrate carries the same signature naturally: ~0.7 % of skeleton points lie
within contact distance of the neighbouring winding's annotation (adjacent
windings genuinely running together — themselves candidate real errors, and a
sharper quantification of open problem № 4). Their strongest clusters outrank
every injection, so raw ranking scores 0.000 no matter how cleanly the points
separate. The benchmark's question is "find the error we *made*", and the
answer is a difference: the same detector runs on the pristine grid, its
evidence cells become the substrate contact atlas, and corrupted-grid clusters
that sit on the atlas are dropped. Likewise a hole component mostly invalid in
the pristine grid too is the mesh's own raggedness, not our injection. On the
smoke slice this single step took AP from 0.042 to 0.474.

**The ghost, declared up front.** The contact relation is symmetric: at a
switch both the corrupted arc and the pristine neighbour a turn away see d ~ 0.
Geometry alone cannot say which of the pair moved, so v1 emits both and eats
the false positive. The probe measures the vertical-jump channel (radial offset
against the rows above and below) as the candidate disambiguator for a later
iteration: the arc that moved is the one displaced against its own vertical
neighbours.

**Hole shape (U-009).** v0's hole channel ranked by size and drowned in natural
raggedness (production meshes lose runs of hundreds of columns). The injector's
holes differ in *shape*, not size: a rectangular window with edges aligned
across rows. Interior invalid runs are merged into connected components and
scored by bbox fill x edge alignment, with single-row components damped — a
single-row gap has no shape evidence at all.

Everything else — pooling of per-point evidence into (row, col-block) windows,
channel merge by within-channel percentile rank, the evaluation harness — is
v0's, unchanged, so a difference in the outcome is a difference in the feature.

Development runs restrict to the dev bands of PROTOCOL §6 (windings < 100 on
Paris 4); the held-out shot is spent once, later, not here.

Usage:

    python detect_v1.py --corpus ../../output/topo/corpus_paris4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --report ../../output/topo/detector_v1_paris4.json --bands dev
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
import scrolls                                                        # noqa: E402
import sheet_erl                                                      # noqa: E402

# The angular window around "one turn away" inside which a node counts as the
# neighbouring sheet. The injector's own tolerance (0.15) is too loose for a
# *distance*: at the tolerance edge the euclidean distance picks up a
# tangential component of ~r * 2*pi*0.15 (~90 vx at the core), which is what
# poisoned the asymmetry variant's background. 0.06 of a turn is ~6 columns.
ANGLE_TOL = 0.06
NEIGHBOUR_K = 24         # fixed vectorisation width; ~2x the nodes 2*ANGLE_TOL spans
MIN_SEPARATION = 24      # a node this close in columns is our own branch, not a
                         # neighbour: one turn is ~90+ columns on these meshes
# Contact reference, voxels. Background min-distance to the neighbouring trace
# has median ~11 vx (measured, pristine w010-027 row 383) — the sheets are that
# close and that jittery. A switch's trace *is* the neighbour's annotation
# (distance ~0), so the score rewards absolute contact, not relative closeness.
CONTACT_REF = 10.0
PROX_FLOOR = 0.5         # evidence = contact below ~5 vx; keeps pooling sparse
BLOCK = 8                # evidence cell width, columns — v0's, unchanged
CELL_ROW_GAP = 2         # cells this many rows apart cluster together (Hann
                         # edges dip below the floor without breaking the arc)
# An injected window is at most ~11 rows tall (height <= 2 mm, rows ~19 vx
# apart — PROTOCOL §4), so a contact zone spanning more rows than this is the
# substrate's own geometry: adjacent windings whose annotations genuinely run
# together over a tall patch. Those are the natural contact zones that outranked
# every injection when pooling was blind to vertical extent (smoke, 14.08).
MAX_CLUSTER_ROWS = 14
SINGLE_ROW_DAMP = 0.25   # a single-row hole has no shape evidence
BACKGROUND_STRIDE = 3    # every k-th row feeds the probe's background sample
# U-011, the support channel: rows sampled against the surface prediction are
# restricted to bands around the corpus's five protocol heights — that is where
# every injection lives by stratification (PROTOCOL §4), and it is what keeps
# the chunk traffic at the level the wave-2 protocol runs already cached.
SUPPORT_BAND_VX = 160.0


def side_distance(points, order, ts, turns, sign):
    """Distance from each point to the trace one full turn away, NaN where absent.

    `points` are one row's valid nodes in polyline order, `turns` their
    accumulated turn, `order`/`ts` the turn-sorted view. The distance is the
    minimum over *nodes* whose turn lies within ANGLE_TOL of `turns + sign` —
    node-to-node, because the injector's switch target is itself a node, and
    because segments between jittery nodes sweep the inter-sheet gap and
    under-read the distance (measured: segment-min median 11.0 vx against a
    radial pitch median of 16.5 on the same row).
    """
    n = len(points)
    target = turns + sign
    lo = np.searchsorted(ts, target - ANGLE_TOL)
    hi = np.searchsorted(ts, target + ANGLE_TOL)
    idx = np.minimum(lo[:, None] + np.arange(NEIGHBOUR_K), n - 1)
    usable = (lo[:, None] + np.arange(NEIGHBOUR_K)) < hi[:, None]
    node = order[idx]
    usable &= np.abs(node - np.arange(n)[:, None]) >= MIN_SEPARATION
    d = np.linalg.norm(points[:, None, :2] - points[node, :2], axis=-1)
    d[~usable] = np.inf
    dist = d.min(axis=1)
    dist[~np.isfinite(dist)] = np.nan
    return dist


def strong_cells(grid, centre, row_scores=None, neighbours=None):
    """Evidence cells of one grid: (row, block) -> (mass, best col, best score).

    The same accumulation serves two purposes: on a corrupted grid it feeds the
    candidate clustering, on the pristine grid it is the *substrate contact
    atlas* — the natural zones where adjacent windings' annotations already run
    within contact distance. Measured on the smoke dev slice, those zones cover
    ~0.7 % of skeleton points and their strongest clusters outrank every
    injection, because they are the same physical signature; the benchmark must
    therefore score detections of *new* contact, i.e. difference against the
    atlas.
    """
    evidence, best = {}, {}
    strong_points = total_points = 0
    for row in range(grid.shape[0]):
        mask = (grid[row, :, 0] != -1) & (grid[row, :, 1] != -1)
        cols, score = (prox_scores_mesh(grid, row, mask, neighbours)
                       if neighbours is not None
                       else prox_scores(grid, row, mask, centre))
        if row_scores is not None:
            row_scores[row] = (cols, score)
        finite = np.isfinite(score)
        total_points += int(finite.sum())
        strong = finite & (score >= PROX_FLOOR)
        strong_points += int(strong.sum())
        for col, s in zip(cols[strong], score[strong]):
            key = (row, int(col) // BLOCK)
            evidence[key] = evidence.get(key, 0.0) + float(s)
            if float(s) > best.get(key, (None, -1.0))[1]:
                best[key] = (int(col), float(s))
    return evidence, best, strong_points, total_points


def prox_scores(grid, row, mask, centre):
    """(cols, per-point contact score) of one row; NaN where the feature is mute.

    Score = 1 - min(d_out, d_in) / CONTACT_REF, clipped to [0, 1]: absolute
    contact with the neighbouring trace. Two scoring variants died before this
    one, on the record (PROTOCOL §9), both measured on the pristine substrate:

    - *asymmetry* |d_out - d_in| / (d_out + d_in) (0 clean, 1 switch, 0.5
      merger): background q99 was 0.94. The scroll's core is genuinely
      asymmetric — at r < 120 the outward pitch is ~10 vx while inward is
      40-90 — and at the 0.15-turn anchor tolerance the distance picks up a
      tangential component of ~90 vx. Natural geometry saturates the score.
    - *relative closeness* 1 - d / local_pitch: the local pitch itself has
      background quantiles 9.9-30.0 vx (q10-q90) — there is no stable unit to
      be relative to, and every compressed zone reads as half-pitch "merger".

    What survives is the absolute signature: a switch's trace *is* the
    neighbour's annotation (d ~ 0-3 vx), while the background min-distance
    floor is ~7-11 vx. The merger (d ~ half pitch ~ 5-15 vx) sits inside the
    background band — this feature does not claim it, and the report says so.
    """
    cols = np.where(mask)[0]
    if len(cols) < 30:
        return cols, np.full(len(cols), np.nan)
    points, turns, _ = scrolls.turn_profile(grid, row, mask, centre)
    points = points.astype(np.float64)
    order = np.argsort(turns, kind='stable')
    ts = turns[order]
    d_out = side_distance(points, order, ts, turns, +1.0)
    d_in = side_distance(points, order, ts, turns, -1.0)
    near = np.fmin(np.nan_to_num(d_out, nan=np.inf),
                   np.nan_to_num(d_in, nan=np.inf))
    score = np.full(len(cols), np.nan)
    finite = np.isfinite(near)
    score[finite] = np.clip(1.0 - near[finite] / CONTACT_REF, 0.0, 1.0)
    return cols, score


def prox_scores_mesh(grid, row, mask, neighbours):
    """Cross-mesh contact for single-winding grids (PHerc0139).

    The turn-shift walk above is Paris 4's inversion of the injector: the
    neighbouring sheet lives in the same grid, one accumulated turn along the
    row. PHerc0139's meshes span exactly one winding each, so `turns ± 1` has
    nothing to land on and the walk is mute by construction. The injector's
    second mechanism (`neighbour-mesh`) already names the inversion for this
    case: the neighbouring sheet is the adjacent winding's *own mesh*, matched
    by iso-z row, distance in the annotation plane. This function mirrors that
    mechanism node for node — same iso-z lookup, same xy-plane distance — and
    keeps every scoring constant (CONTACT_REF, PROX_FLOOR) unchanged, so the
    freeze covers one detector with two declared neighbour lookups, exactly
    like the injector it inverts.

    `neighbours` is a list of (row_heights, row_spacing, grid, valid) for the
    adjacent windings' meshes. NaN where the feature is mute (no neighbour row
    at this height) — the same semantics as the turn-shift walk at the ends of
    a Paris 4 mesh.
    """
    cols = np.where(mask)[0]
    if len(cols) < 30:
        return cols, np.full(len(cols), np.nan)
    points = grid[row][mask][:, :2].astype(np.float64)
    z = float(np.median(grid[row][mask][:, 2]))
    near = np.full(len(cols), np.inf)
    for heights, spacing, n_grid, n_valid in neighbours:
        if np.all(np.isnan(heights)):
            continue
        n_row = int(np.nanargmin(np.abs(heights - z)))
        if abs(heights[n_row] - z) > 2 * spacing:
            continue
        n_mask = n_valid[n_row]
        if n_mask.sum() < 10:
            continue
        cand = n_grid[n_row][n_mask][:, :2].astype(np.float64)
        d = np.linalg.norm(points[:, None, :] - cand[None, :, :],
                           axis=-1).min(axis=1)
        near = np.minimum(near, d)
    score = np.full(len(cols), np.nan)
    finite = np.isfinite(near)
    score[finite] = np.clip(1.0 - near[finite] / CONTACT_REF, 0.0, 1.0)
    return cols, score


def radial_map(grid, centre):
    """Radius from the counting centre, per node; NaN where invalid."""
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    r = np.full(grid.shape[:2], np.nan)
    for row in range(grid.shape[0]):
        m = valid[row]
        if m.sum() < 2:
            continue
        z = float(np.median(grid[row][m][:, 2]))
        try:
            cx, cy = centre.at(z)
        except ValueError:
            # PHerc0139's mesh-geometry centre is defined only over the z range
            # the innermost labelled turn reaches (scrolls.Centre refuses to
            # guess outside it). A row with no counting origin has no radius:
            # it stays NaN — mute, the same semantics every channel uses —
            # rather than crashing the run or inventing a centre.
            continue
        r[row, m] = np.hypot(grid[row, m, 0] - cx, grid[row, m, 1] - cy)
    return r


def hole_components(grid):
    """Interior invalid runs merged across rows: (row, col, score, spans).

    Fill of the bounding box and edge alignment separate an injected rectangle
    (fill 1.0, aligned edges) from the natural raggedness that blinded v0's
    size-ranked channel; `spans` — the component's (row, lo, hi) runs — let the
    caller difference the component against the pristine grid's own holes.
    """
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    parent = {}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    components = {}       # root -> list of (row, lo, hi) half-open col spans
    previous = []         # (root, lo, hi) of the row above
    for row in range(grid.shape[0]):
        cols = np.where(valid[row])[0]
        runs = []
        if len(cols) >= 2:
            gaps = np.where(np.diff(cols) > 1)[0]
            runs = [(int(cols[g]) + 1, int(cols[g + 1])) for g in gaps]
        current = []
        for lo, hi in runs:
            overlapping = [root for root, plo, phi in previous
                           if plo < hi and lo < phi]
            key = (row, lo)
            parent[key] = key
            if overlapping:
                for root in overlapping:
                    parent[find(root)] = key
            components[key] = [(row, lo, hi)]
            current.append((key, lo, hi))
        previous = current

    merged = {}
    for key, spans in components.items():
        merged.setdefault(find(key), []).extend(spans)

    out = []
    for spans in merged.values():
        rows = [r for r, _, _ in spans]
        starts = [lo for _, lo, _ in spans]
        ends = [hi for _, _, hi in spans]
        span_rows = max(rows) - min(rows) + 1
        area = float(sum(hi - lo for _, lo, hi in spans))
        bbox = span_rows * (max(ends) - min(starts))
        fill = area / bbox if bbox else 0.0
        align = 1.0 / (1.0 + float(np.std(starts)) + float(np.std(ends)))
        damp = 1.0 if span_rows > 1 else SINGLE_ROW_DAMP
        score = damp * fill * align * float(np.sqrt(area))
        row = (min(rows) + max(rows)) // 2
        col = (min(starts) + max(ends)) // 2
        out.append((row, col, score, spans))
    return out


def support_cells(grid, prediction, z_quantiles, threshold):
    """Evidence cells of the U-011 support channel: trace without a surface.

    A merger's trace sits mid-gap, where the surface prediction says there is
    no sheet — the one signature the contact channel cannot see, because half a
    pitch is inside the background band of inter-sheet distances. Score 1 for
    an unsupported node, pooled into the same (row, block) cells; natural
    unsupported zones (mesh outside the scan mask, prediction holes) are tall
    or match the pristine atlas, and die the same two deaths as natural
    contact.

    A node counts as supported if the prediction fires anywhere within ±2 vx
    in the plane and ±1 vx in z: the predicted sheet is thin and the
    annotation rides a voxel off it — nearest-voxel sampling alone loses a
    third of the *pristine* trace (measured: 0.67 supported raw, 0.87 with the
    neighbourhood), while half a pitch — the merger's seat — stays well
    outside the tolerance.
    """
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    heights, _ = scrolls.row_heights(grid)
    offsets = [np.array([dx, dy, dz], float)
               for dx in (-2, -1, 0, 1, 2)
               for dy in (-2, -1, 0, 1, 2)
               for dz in (-1, 0, 1)]
    evidence, best = {}, {}
    for row in range(grid.shape[0]):
        z = heights[row]
        if not np.isfinite(z) or not any(
                abs(z - q) <= SUPPORT_BAND_VX for q in z_quantiles):
            continue
        mask = valid[row]
        if mask.sum() < 30:
            continue
        cols = np.where(mask)[0]
        points = grid[row][mask].astype(np.float64)
        # One batched sample for all offsets: with one call per offset, a row
        # of an outer mesh touches more chunks than the reader's LRU holds and
        # every call decompresses the whole ring again — the 14.08 full run
        # stalled ~40 min on one such segment before this was batched.
        stacked = (points[None, :, :] + np.stack(offsets)[:, None, :])
        values = prediction.sample(stacked.reshape(-1, 3))
        peak = values.reshape(len(offsets), -1).max(axis=0)
        unsupported = peak < threshold
        for col in cols[unsupported]:
            key = (row, int(col) // BLOCK)
            evidence[key] = evidence.get(key, 0.0) + 1.0
            if best.get(key, (None, -1.0))[1] < 1.0:
                best[key] = (int(col), 1.0)
    return evidence, best


def differenced_clusters(evidence, best, atlas):
    """Clusters of evidence cells, differenced against an atlas of the same
    channel on the pristine grid. Yields (cells, mass, top) for survivors and
    (None, None, None) for each masked cluster, so callers can count both.
    """
    dilated = {(r + dr, b) for r, b in atlas for dr in (-1, 0, 1)}
    parent = {key: key for key in evidence}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for row, blk in evidence:
        for dr in range(-CELL_ROW_GAP, CELL_ROW_GAP + 1):
            for db in (-1, 0, 1):
                other = (row + dr, blk + db)
                if other != (row, blk) and other in evidence:
                    parent[find(other)] = find((row, blk))

    clusters = {}
    for key in evidence:
        clusters.setdefault(find(key), []).append(key)

    for cells in clusters.values():
        rows = [r for r, _ in cells]
        tall = max(rows) - min(rows) + 1 > MAX_CLUSTER_ROWS
        on_atlas = sum(c in dilated for c in cells) * 2 > len(cells)
        if tall or on_atlas:
            yield None, None, None
            continue
        mass = sum(evidence[c] for c in cells)
        top = max(cells, key=lambda c: best[c][1])
        yield cells, mass, top


def cluster_vjump(cells, vjump, med):
    """Cluster's vertical-jump factor: strongest cell jump over the segment
    median. The U-010 branch: an arc that *jumped* onto the neighbour's sheet
    is radially displaced against its own rows above and below, while a
    natural contact zone the atlas missed is vertically coherent. Ghosts are
    the target: the pristine neighbour of a switch sees the same contact but
    did not move, so its factor stays near 1.
    """
    peak = 0.0
    for row, blk in cells:
        window = vjump[max(row - 1, 0):row + 1,
                       blk * BLOCK:(blk + 1) * BLOCK]
        if window.size and np.isfinite(window).any():
            peak = max(peak, float(np.nanmax(window)))
    return peak / med if med > 0 and peak > 0 else 1.0


def segment_channels(name, grid, pristine, centre, row_scores, stats,
                     vjump=None, support=None, neighbours=None):
    """Pooled candidates of one corrupted grid, differenced against its pristine
    substrate; fills `row_scores` for the probe and `stats` for the report.

    Evidence cells (row, col-block) holding summed contact scores are clustered
    into connected components (±CELL_ROW_GAP rows, ±1 block); the candidate is
    the component's best single contribution, so narrow errors survive pooling.
    Two subtractions, both against the *pristine* grid — the benchmark scores
    detection of the error we made, not of the errors the substrate came with:

    - a prox cluster more than half of whose cells sit on (or one row off) the
      substrate contact atlas is the mesh's own touching zone, dropped;
    - a hole component more than half of whose area is invalid in the pristine
      grid too is the mesh's own raggedness, dropped.

    A cluster taller than MAX_CLUSTER_ROWS is dropped either way: an injection
    cannot be that tall by construction (PROTOCOL §4 heights), and the tallest
    natural zones are exactly the ones whose skirts leak past the atlas.
    """
    corrupted_ctx = pristine_ctx = None
    if neighbours is not None:
        corrupted_ctx, pristine_ctx = neighbours
    atlas, _, natural_points, total_points = strong_cells(
        pristine, centre, neighbours=pristine_ctx)
    stats['substrate_contact_points'] = natural_points
    stats['skeleton_points'] = total_points

    evidence, best = strong_cells(grid, centre, row_scores,
                                  neighbours=corrupted_ctx)[:2]

    med = (float(np.nanmedian(vjump)) if vjump is not None
           and np.isfinite(vjump).any() else 0.0)
    out = []
    masked_prox = masked_rect = 0
    for cells, mass, top in differenced_clusters(evidence, best, atlas):
        if cells is None:
            masked_prox += 1
            continue
        factor = cluster_vjump(cells, vjump, med) if vjump is not None else 1.0
        out.append((name, top[0], best[top][0], 'prox', mass, factor))

    pristine_invalid = (pristine[..., 0] == -1) | (pristine[..., 1] == -1)
    for row, col, score, spans in hole_components(grid):
        area = sum(hi - lo for _, lo, hi in spans)
        old = sum(int(pristine_invalid[r, lo:hi].sum()) for r, lo, hi in spans)
        if old * 2 > area:
            masked_rect += 1
            continue
        out.append((name, row, col, 'rect', score, 1.0))
    stats['masked_prox_clusters'] = masked_prox
    stats['masked_rect_components'] = masked_rect

    if support is not None:
        prediction, z_quantiles, threshold = support
        evidence_s, best_s = support_cells(grid, prediction, z_quantiles,
                                           threshold)
        atlas_s, _ = support_cells(pristine, prediction, z_quantiles, threshold)
        # Differencing here is a per-cell *count* subtraction, not a veto: the
        # pristine trace is itself 10-20 % unsupported in noisy regions
        # (prediction holes, off-mask), and a cluster veto killed every merger
        # that landed there (smoke autopsy 14.08: M004 read 61/140 unsupported
        # against a pristine 24/140 — a clear surplus, vetoed anyway). The
        # surplus of unsupported nodes over the pristine count is the signal.
        surplus = {key: value - atlas_s.get(key, 0.0)
                   for key, value in evidence_s.items()
                   if value - atlas_s.get(key, 0.0) > 0.5}
        masked_support = 0
        for cells, mass, top in differenced_clusters(surplus, best_s, {}):
            if cells is None:
                masked_support += 1
                continue
            out.append((name, top[0], best_s[top][0], 'support', mass, 1.0))
        stats['masked_support_clusters'] = masked_support
    return out


def merge_channels(candidates, top, weighted=False):
    """One ranking from several channels, by within-channel percentile rank.

    `weighted` multiplies each prox cluster's mass by its vertical-jump factor
    (U-010) before ranking; the rect channel is unaffected either way.
    """
    def score(c):
        return c[4] * c[5] if weighted and c[3] == 'prox' else c[4]

    ranked = []
    # sorted(): set iteration order is hash-seeded per process, and the final
    # stable sort keeps insertion order among percentile ties — so the same
    # candidates gave AP 0.6485 in one process and 0.6496 in another (0139
    # replay, 15.08). The scores are untouched; the tie order is simply made
    # canonical so the exam number is one number.
    for channel in sorted({c[3] for c in candidates}):
        of_channel = sorted((c for c in candidates if c[3] == channel),
                            key=score)
        n = len(of_channel)
        for i, c in enumerate(of_channel):
            ranked.append((c[0], c[1], c[2], (i + 1) / n))
    ranked.sort(key=lambda t: -t[3])
    return ranked[:top]


def quantiles(values, qs=(0.5, 0.9, 0.99, 0.999)):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    out = {f'q{q}': round(float(np.quantile(a, q)), 4) for q in qs}
    out['n'] = int(len(a))
    return out


def probe_segment(injections, row_scores, vjump, background):
    """Per-injection window maxima vs the background the same code produced.

    This is the mechanism record: whatever AP says, these numbers say whether
    U-008 separates at the point level, and whether the vertical jump could
    disambiguate the ghost.
    """
    for cols, score in list(row_scores.values())[::BACKGROUND_STRIDE]:
        finite = np.isfinite(score)
        if finite.any():
            background['prox'].extend(score[finite].tolist())
    finite = np.isfinite(vjump)
    if finite.any():
        background['vjump'].extend(
            vjump[finite][::BACKGROUND_STRIDE].tolist())

    out = []
    for r in injections:
        peak = 0.0
        for row in range(r['row_lo'], r['row_hi']):
            cols, score = row_scores.get(row, (None, None))
            if cols is None:
                continue
            inside = (cols >= r['col_lo']) & (cols < r['col_hi'])
            windowed = score[inside]
            finite = np.isfinite(windowed)
            if finite.any():
                peak = max(peak, float(np.nanmax(windowed)))
        jump = vjump[max(r['row_lo'] - 1, 0):r['row_hi'],
                     r['col_lo']:r['col_hi']]
        jump_peak = float(np.nanmax(jump)) if np.isfinite(jump).any() else None
        out.append({'id': r['id'], 'type': r['type'], 'prox_peak': round(peak, 4),
                    'vjump_peak_vx': round(jump_peak, 2)
                    if jump_peak is not None else None,
                    'plausible': r.get('plausible')})
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--bands', choices=('dev', 'heldout', 'all'),
                        default='dev',
                        help="dev = segments with winding_low < 100 (PROTOCOL "
                             "§6); 'heldout' = the complement (>= 100), "
                             "reserved for the single held-out shot on "
                             "Paris 4; 'all' = no filter (a whole-scroll "
                             "held-out substrate such as PHerc0139)")
    parser.add_argument('--row-step', type=int, default=1)
    parser.add_argument('--with-prediction', action='store_true',
                        help='enable the U-011 support channel (samples the '
                             'surface prediction over the protocol height '
                             'bands; needs network or a warm chunk cache)')
    parser.add_argument('--checkpoint', default=None,
                        help='directory for per-segment resume checkpoints: '
                             'each finished segment saves its outputs there, '
                             'and a rerun replays them instead of recomputing '
                             '— a run killed mid-flight (network drop, laptop '
                             'lid) resumes at the first unfinished segment. '
                             'Per-segment computation is deterministic, so a '
                             'resumed run and a straight run produce '
                             'identical reports; checkpoints are keyed by '
                             '(corpus, bands, with-prediction) and a '
                             'mismatched key is recomputed, not trusted')
    args = parser.parse_args()

    with open(os.path.join(args.corpus, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    scroll = scrolls.SCROLLS[manifest['scroll']]
    voxel_mm = manifest['voxel_mm']
    centre = scrolls.Centre(scroll, args.cache, args.grid_cache)

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

    support = None
    if args.with_prediction:
        support = (scrolls.open_prediction(scroll, args.cache, max_chunks=256),
                   manifest['z_quantiles'], scroll.threshold)

    mechanism = manifest.get('mechanism', 'turn-shift')
    by_winding, winding_of = {}, {}
    if mechanism == 'neighbour-mesh':
        by_winding = {low: seg for seg, low, high
                      in scrolls.labelled_segments(scroll) if low == high}
        winding_of = {seg: low for low, seg in by_winding.items()}

    def row_context(grid):
        heights, valid = scrolls.row_heights(grid)
        spacing = float(np.nanmedian(np.abs(np.diff(heights))))
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = 19.0
        return heights, spacing, grid, valid

    def neighbour_pair(seg):
        """(corrupted-side, pristine-side) contexts of the adjacent windings.

        The corrupted scan compares against the corpus's own copy of each
        neighbour where one exists: on Paris 4 the corrupted grid holds both
        halves of the contact relation in one mesh, and this reproduces that —
        including the declared ghost, where the neighbour of a switch sees the
        same contact. The atlas side is always pristine against pristine.
        """
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
                 'masked_support_clusters': 0}
    ckpt_key = {'corpus': os.path.abspath(args.corpus), 'bands': args.bands,
                'with_prediction': bool(args.with_prediction)}
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
            r = radial_map(corrupted, centre)
            vjump = np.abs(np.diff(r, axis=0))
            neighbours = (neighbour_pair(name)
                          if mechanism == 'neighbour-mesh' else None)
            seg_candidates = segment_channels(
                name, corrupted, pristine[name], centre, row_scores, stats,
                vjump=vjump, support=support, neighbours=neighbours)
            seg_background = {'prox': [], 'vjump': []}
            of_segment = [rec for rec in injections if rec['segment'] == name]
            seg_probes = probe_segment(of_segment, row_scores, vjump,
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
    ranking = merge_channels(candidates, top=4 * n)
    ranking_vj = merge_channels(candidates, top=4 * n, weighted=True)
    substrate['contact_share'] = round(
        substrate['substrate_contact_points']
        / max(substrate['skeleton_points'], 1), 5)

    zones = {name: sheet_erl.segment_zones(scoped, name) for name in names}
    erl_broken, *_ = sheet_erl.sheet_erl(pristine, zones, voxel_mm, args.row_step)

    def judged(rank):
        result = sheet_erl.evaluate_ranking(rank, scoped, pristine, voxel_mm,
                                            args.row_step, erl_broken=erl_broken)
        # hits() credits greedily in rank order, so the full list's prefix is
        # identical to hits(rank[:n]) — one pass serves recall@N, the
        # per-injection ranks and the per-window outcome record that the
        # bootstrap needs (a resample of injections re-labels windows credited
        # to dropped injections as false positives; without this record the
        # report's AP is a point with no width).
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

    (result, by_type, recall_plausible, by_band,
     per_rank, per_injection) = judged(ranking)
    (result_vj, by_type_vj, recall_plausible_vj, by_band_vj,
     per_rank_vj, per_injection_vj) = judged(ranking_vj)

    # PROTOCOL §3's two delta-ERL comparators, declared there from day one but
    # absent from every report until the 15.08 review caught it: the oracle
    # (cuts at the true windows — the ceiling any detector can reach) and
    # random windows (the floor). Without them "+0.29 mm" has no denominator:
    # the corpus ERL is ~hundreds of mm, and the honest reading of the
    # detector's delta is its fraction of the oracle's, not its absolute size.
    oracle_rank = [(r['segment'], (r['row_lo'] + r['row_hi']) // 2,
                    (r['col_lo'] + r['col_hi']) // 2, 1.0) for r in injections]
    oracle = sheet_erl.evaluate_ranking(oracle_rank, scoped, pristine, voxel_mm,
                                        args.row_step, erl_broken=erl_broken)
    rng = np.random.default_rng(20260815)
    random_deltas = sorted(
        sheet_erl.evaluate_ranking(
            sheet_erl.random_ranking(rng, pristine, n), scoped, pristine,
            voxel_mm, args.row_step, erl_broken=erl_broken)['delta_erl_mm']
        for _ in range(10))
    erl_context = {
        'delta_erl_oracle_mm': oracle['delta_erl_mm'],
        'oracle_recall_at_n': oracle['recall_at_n'],
        'delta_erl_random_mm_median10': random_deltas[len(random_deltas) // 2],
        'delta_erl_random_mm_seeds': [round(d, 4) for d in random_deltas],
        'note': 'delta_erl_mm values are corpus-wide (all injected segments, '
                'one number), against erl_broken_mm of the same corpus; the '
                'normalised reading is delta/oracle_delta',
    }

    probe = {'background_prox': quantiles(background['prox']),
             'background_vjump_vx': quantiles(background['vjump']),
             'injections': probes}
    for t in 'SM':
        peaks = [p['prox_peak'] for p in probes if p['type'] == t]
        jumps = [p['vjump_peak_vx'] for p in probes
                 if p['type'] == t and p['vjump_peak_vx'] is not None]
        probe[f'{t}_prox_peaks'] = quantiles(peaks, (0.1, 0.25, 0.5, 0.9))
        probe[f'{t}_vjump_peaks_vx'] = quantiles(jumps, (0.1, 0.25, 0.5, 0.9))

    report = {'detector': 'v1 cross-trace proximity (U-008) + hole shape (U-009)',
              'lineage': 'v0 ray-ladder AP 0.0000; U-008 taken as declared in '
                         'protocol/UNTRIED.md',
              'bands': args.bands, 'corpus': os.path.abspath(args.corpus),
              'scoped_injections': n,
              'metrics': result,
              'recall_by_type': by_type,
              'recall_by_band': by_band,
              'recall_on_locally_plausible': recall_plausible,
              'erl_context': erl_context,
              'per_rank_outcomes': per_rank,
              'per_injection_rank': per_injection,
              'vjump_weighted': {'metrics': result_vj,
                                 'recall_by_type': by_type_vj,
                                 'recall_by_band': by_band_vj,
                                 'recall_on_locally_plausible':
                                     recall_plausible_vj,
                                 'per_rank_outcomes': per_rank_vj,
                                 'per_injection_rank': per_injection_vj},
              'substrate': substrate,
              'probe': probe}
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f"v1 on {args.bands} bands: AP {result['ap']:.4f}, "
          f"recall@N {result['recall_at_n']:.3f}, "
          f"dERL {result['delta_erl_mm']:+.2f} mm")
    print(f"v1+vjump (U-010): AP {result_vj['ap']:.4f}, "
          f"recall@N {result_vj['recall_at_n']:.3f}")
    print("recall by type: " + ', '.join(
        f'{t}={v:.3f}' if v is not None else f'{t}=n/a'
        for t, v in by_type.items()))
    if by_band:
        worst = min(by_band, key=by_band.get)
        best_band = max(by_band, key=by_band.get)
        print("recall@N by winding band: " + ', '.join(
            f'{b}={v:.3f}' for b, v in by_band.items()))
        print(f"worst band {worst} = {by_band[worst]:.3f}, "
              f"best band {best_band} = {by_band[best_band]:.3f} "
              f"(PROTOCOL §8: worst published alongside best)")
    if recall_plausible is not None:
        print(f"recall on locally-plausible injections: {recall_plausible:.3f}")
    print(f"probe: background prox {probe['background_prox']}, "
          f"S peaks {probe['S_prox_peaks']}, M peaks {probe['M_prox_peaks']}")
    print(f"report at {args.report}")


if __name__ == '__main__':
    main()
