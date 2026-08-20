"""Inject known topological errors into correct tifxyz meshes (TOPO-007).

The oracle of the TOPO line is self-generating: take a mesh the field accepts as
correct, break it in a known way at a known place, and the detector either finds the
break or it does not. This script performs the breaking. Everything it does is written
to a manifest that is the single source of truth about where the errors are
(`oneshot/PROTOCOL.md` §4); the corpus on disk is a derived artefact.

Three error types, fixed by the protocol:

- ``S`` (sheet switch): over an arc, the trace leaves its own sheet and lands on the
  neighbouring winding's sheet, with a raised-cosine blend at both ends so the centre
  of the arc sits fully on the wrong sheet while the splice itself stays smooth. This
  is the locally-plausible error the target is about: no crease, no tear, just the
  wrong sheet.
- ``M`` (merger): over the same kind of window, the trace is pulled halfway into the
  gap — two sheets resolved as one front, the dominant failure mode measured by
  wave 2 (22-28 % of gaps).
- ``H`` (hole): a window of the grid is invalidated outright.

Two mechanisms find "the neighbouring sheet", recorded per injection:

- ``neighbour-mesh`` (PHerc0139): the corresponding point on the adjacent winding's
  own mesh — nearest valid point, in the plane of the same grid row.
- ``turn-shift`` (PHercParis4): the point on the *same* mesh row whose accumulated
  turn around the umbilicus differs by exactly one winding (the meshes span up to 18
  windings, so the neighbour lives in the same grid).

Determinism is a deliverable: one seed, one corpus, bit for bit. Grids are therefore
saved as raw ``.npy`` (no zip timestamps) and every random draw comes from a single
seeded generator.

Usage (development substrate, Paris 4 — held-out discipline lives in PROTOCOL §6):

    python inject_errors.py --scroll PHercParis4 \
        --grid-cache ../../output/figgrids --cache ../../output/figcache \
        --out ../../output/topo/corpus_paris4 --per-type 100 --seed 20260814
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

_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'wave2', 'figures')
sys.path.insert(0, _FIGURES)
import scrolls                                                        # noqa: E402

# Annotation-frame voxel pitch in mm. PHerc0139's is stated in its volume name
# (9.362 um); Paris 4's frame is level 2 of the 2.4 um scan, 4 * 2.4 = 9.6 um.
VOXEL_MM = {'PHercParis4': 9.6e-3, 'PHerc0139': 9.362e-3}

LAMBDA_MM = (0.5, 10.0)   # arc length, log-uniform (PROTOCOL §4)
HEIGHT_MM = (0.2, 2.0)    # window height, uniform (PROTOCOL §4)
TYPES = ('S', 'M', 'H')
# A found neighbour further than this many voxels is not a neighbouring winding but a
# listing gap or a torn sheet; the draw is rejected. Sheet pitch is ~21-22 voxels in
# both frames, so 3 pitches is generous without ever crossing two windings.
MAX_NEIGHBOUR_VX = 65.0


def hann(n):
    """Raised-cosine weights, 0 at both edges, 1 in the middle; length n >= 1."""
    if n == 1:
        return np.ones(1)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def valid_mask(grid):
    return (grid[..., 0] != -1) & (grid[..., 1] != -1)


def row_step_vx(grid, row, mask):
    """Median spacing between adjacent valid nodes along the row, in voxels."""
    points = grid[row][mask]
    if len(points) < 3:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    steps = steps[steps > 0]
    return float(np.median(steps)) if len(steps) else None


def row_spacing_vx(grid):
    """Median z distance between consecutive grid rows (~19 vx on these meshes)."""
    heights, _ = scrolls.row_heights(grid)
    spacing = np.nanmedian(np.abs(np.diff(heights)))
    return float(spacing) if np.isfinite(spacing) and spacing > 0 else 19.0


def patch_normals(grid, rows, cols):
    """Unit normals of the grid patch, NaN where geometry is invalid.

    Normals come from the cross product of the two grid tangents. The patch is read
    with a one-cell border so the finite differences are central where possible.
    """
    r0, r1 = max(rows.start - 1, 0), min(rows.stop + 1, grid.shape[0])
    c0, c1 = max(cols.start - 1, 0), min(cols.stop + 1, grid.shape[1])
    patch = grid[r0:r1, c0:c1].astype(np.float64)
    bad = ~valid_mask(grid[r0:r1, c0:c1])
    patch[bad] = np.nan
    dv = np.gradient(patch, axis=0)
    du = np.gradient(patch, axis=1)
    normal = np.cross(du, dv)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    with np.errstate(invalid='ignore'):
        return normal / length


def max_kink_deg(grid, rows, cols):
    """Largest angle between adjacent normals inside the window, degrees.

    This is the local-plausibility measure of PROTOCOL §4: an injection whose kink
    does not exceed the pristine mesh's median kink cannot be found by looking for a
    crease — which is the point of the target.
    """
    normals = patch_normals(grid, rows, cols)
    worst = 0.0
    for axis in (0, 1):
        a = np.take(normals, range(normals.shape[axis] - 1), axis=axis)
        b = np.take(normals, range(1, normals.shape[axis]), axis=axis)
        dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
        angles = np.degrees(np.arccos(dot))
        if np.any(np.isfinite(angles)):
            worst = max(worst, float(np.nanmax(angles)))
    return worst


def max_kink_smoothed_deg(grid, rows, cols, win=1):
    """TOPO-017 (corpus v2) measure: the same largest adjacent-normal angle,
    computed on a box-smoothed normal field.

    The window is grown by ``win`` nodes for smoothing support, the normal
    field is nanmean-averaged over the (2*win+1)^2 box component-wise and
    renormalised, then the v1 angle maximum is taken. Normal orientation is
    consistent within a patch (both come from the grid parameterisation), so
    plain averaging does not cancel; the angle step uses |dot| as in v1.
    Declared in PROTOCOL §4 (insert of 17.08.2026) before any v2 run.
    """
    r = slice(max(rows.start - win, 0), min(rows.stop + win, grid.shape[0]))
    c = slice(max(cols.start - win, 0), min(cols.stop + win, grid.shape[1]))
    normals = patch_normals(grid, r, c)
    padded = np.full((normals.shape[0] + 2 * win, normals.shape[1] + 2 * win,
                      3), np.nan)
    padded[win:win + normals.shape[0], win:win + normals.shape[1]] = normals
    stack = [padded[dr:dr + normals.shape[0], dc:dc + normals.shape[1]]
             for dr in range(2 * win + 1) for dc in range(2 * win + 1)]
    with np.errstate(invalid='ignore'):
        smoothed = np.nanmean(np.stack(stack), axis=0)
        length = np.linalg.norm(smoothed, axis=-1, keepdims=True)
        smoothed = smoothed / length
    worst = 0.0
    for axis in (0, 1):
        a = np.take(smoothed, range(smoothed.shape[axis] - 1), axis=axis)
        b = np.take(smoothed, range(1, smoothed.shape[axis]), axis=axis)
        dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
        angles = np.degrees(np.arccos(dot))
        if np.any(np.isfinite(angles)):
            worst = max(worst, float(np.nanmax(angles)))
    return worst


def pristine_kink_median(rng, grid, samples=40, size=8, smoothed=False):
    """Median max-kink over random windows of the untouched grid.

    With ``smoothed`` the v2 measure is evaluated on the SAME windows (one
    rng stream, the draw count unchanged — a no-flag run stays bit-for-bit
    the v1 manifest) and a (v1_median, smoothed_median) pair is returned.
    """
    mask = valid_mask(grid)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (None, None) if smoothed else None
    kinks, kinks_s = [], []
    for _ in range(samples):
        i = int(rng.integers(len(rows)))
        r, c = int(rows[i]), int(cols[i])
        window = (slice(max(r - size, 0), r + size),
                  slice(max(c - size, 0), c + size))
        kinks.append(max_kink_deg(grid, *window))
        if smoothed:
            kinks_s.append(max_kink_smoothed_deg(grid, *window))
    if smoothed:
        return float(np.median(kinks)), float(np.median(kinks_s))
    return float(np.median(kinks))


def neighbour_targets_mesh(own_points, neighbour_grid, z):
    """``neighbour-mesh``: nearest valid point on the neighbour's iso-z row."""
    row, mask = scrolls.iso_z_row(neighbour_grid, z)
    if row is None or mask.sum() < 10:
        return None
    candidates = neighbour_grid[row][mask]
    targets = np.empty_like(own_points)
    for i, point in enumerate(own_points):
        distances = np.linalg.norm(candidates[:, :2] - point[:2], axis=1)
        j = int(np.argmin(distances))
        if distances[j] > MAX_NEIGHBOUR_VX:
            return None
        targets[i] = candidates[j]
    return targets


def neighbour_targets_turn(grid, row, mask, centre, col_slice, sign):
    """``turn-shift``: the same row's point one full winding away.

    Works because Paris 4 meshes span whole ranges of windings: the neighbouring
    sheet is the same grid, one turn along axis 1. `turns` from the wave-2 profile is
    monotonic along the row up to noise; the target column is found by nearest turn.
    """
    points, turns, _ = scrolls.turn_profile(grid, row, mask, centre)
    cols = np.where(mask)[0]
    order = np.argsort(turns)
    turns_sorted, cols_sorted = turns[order], cols[order]
    targets, used = [], []
    for col in range(col_slice.start, col_slice.stop):
        if not mask[col]:
            targets.append(None)
            continue
        own_turn = turns[np.searchsorted(cols, col)]
        wanted = own_turn + sign
        k = int(np.clip(np.searchsorted(turns_sorted, wanted), 1,
                        len(turns_sorted) - 1))
        k = k if abs(turns_sorted[k] - wanted) < abs(turns_sorted[k - 1] - wanted) \
            else k - 1
        if abs(turns_sorted[k] - wanted) > 0.15:       # no sheet a full turn away
            return None
        targets.append(grid[row, int(cols_sorted[k])].copy())
        used.append(col)
    if not used:
        return None
    out = np.full((col_slice.stop - col_slice.start, 3), np.nan)
    for i, t in enumerate(targets):
        if t is not None:
            out[i] = t
    return out


def apply_window(grid, rows, cols, targets, fraction):
    """Blend grid points toward targets with a separable raised-cosine window.

    ``fraction`` is 1.0 for a sheet switch (centre lands on the neighbour sheet) and
    0.5 for a merger (centre sits mid-gap). Returns the peak displacement actually
    applied, for the manifest's self-check.
    """
    wr = hann(rows.stop - rows.start)
    wc = hann(cols.stop - cols.start)
    peak = 0.0
    for i, r in enumerate(range(rows.start, rows.stop)):
        for j, c in enumerate(range(cols.start, cols.stop)):
            target = targets[j]
            if target is None or np.any(np.isnan(target)):
                continue
            if grid[r, c, 0] == -1 or grid[r, c, 1] == -1:
                continue
            w = wr[i] * wc[j] * fraction
            if w <= 0:
                continue
            shift = (target - grid[r, c]) * w
            grid[r, c] += shift.astype(grid.dtype)
            peak = max(peak, float(np.linalg.norm(shift)))
    return peak


def build_cells(scroll, segments, grid_cache):
    """Stratification cells: (winding band of 10) x (5 z-quantile heights)."""
    heights = scrolls.protocol_heights(scroll, grid_cache, segments=segments)
    cells = {}
    for name, low, high in segments:
        band = low // 10
        for qi, z in enumerate(heights):
            cells.setdefault((band, qi), []).append((name, low, high, z))
    return cells, heights


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scrolls.add_argument(parser)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-type', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260814)
    parser.add_argument('--smoothed', action='store_true',
                        help='TOPO-017 corpus v2: also record the smoothed '
                             'plausibility measure (PROTOCOL §4 insert '
                             '17.08.2026) next to the v1 one; off by '
                             'default — a no-flag run reproduces the v1 '
                             'manifest bit-for-bit')
    args = parser.parse_args()

    scroll = scrolls.SCROLLS[args.scroll]
    voxel_mm = VOXEL_MM[scroll.key]
    rng = np.random.default_rng(args.seed)
    segments = scrolls.labelled_segments(scroll)
    by_winding = {low: name for name, low, high in segments if low == high}
    mechanism = 'neighbour-mesh' if by_winding else 'turn-shift'
    centre = (scrolls.Centre(scroll, args.cache, args.grid_cache)
              if mechanism == 'turn-shift' else None)

    cells, heights = build_cells(scroll, segments, args.grid_cache)
    cell_keys = sorted(cells)
    grids = {}          # segment name -> mutable grid
    occupied = {}       # (segment, row) -> list of (lo, hi) taken col spans
    injections, rejected = [], 0

    def grid_of(name):
        if name not in grids:
            grids[name] = scrolls.segment_grid(name, scroll, args.grid_cache).copy()
        return grids[name]

    for error_type in TYPES:
        placed = 0
        attempts = 0
        while placed < args.per_type:
            attempts += 1
            if attempts > args.per_type * 60:
                raise RuntimeError(
                    f'{error_type}: only {placed}/{args.per_type} placed after '
                    f'{attempts} draws — do not silently ship a thinner corpus')
            cell = cell_keys[(placed + attempts) % len(cell_keys)]
            name, low, high, z = cells[cell][int(rng.integers(len(cells[cell])))]
            grid = grid_of(name)
            row, mask = scrolls.iso_z_row(grid, z)
            if row is None or mask.sum() < 40:
                rejected += 1
                continue

            lam_mm = float(np.exp(rng.uniform(*np.log(LAMBDA_MM))))
            h_mm = float(rng.uniform(*HEIGHT_MM))
            step = row_step_vx(grid, row, mask)
            if step is None:
                rejected += 1
                continue
            ncols = max(4, int(round(lam_mm / voxel_mm / step)))
            nrows = max(1, int(round(h_mm / voxel_mm / row_spacing_vx(grid))))

            cols_valid = np.where(mask)[0]
            if len(cols_valid) <= ncols + 2:
                rejected += 1
                continue
            start = int(cols_valid[int(rng.integers(len(cols_valid) - ncols - 1))])
            col_slice = slice(start, start + ncols)
            row_slice = slice(max(row - nrows // 2, 0),
                              min(row + nrows - nrows // 2, grid.shape[0]))
            spans = occupied.setdefault((name, row), [])
            margin = ncols
            if any(col_slice.start < hi + margin and lo - margin < col_slice.stop
                   for lo, hi in spans):
                rejected += 1
                continue

            record = {
                'id': f'{error_type}{placed:03d}',
                'type': error_type, 'segment': name,
                'winding_low': low, 'winding_high': high,
                'band': cell[0], 'z_quantile': cell[1], 'z': float(z),
                'row': int(row), 'row_lo': row_slice.start, 'row_hi': row_slice.stop,
                'col_lo': col_slice.start, 'col_hi': col_slice.stop,
                'lambda_mm': round(lam_mm, 4), 'height_mm': round(h_mm, 4),
            }

            if error_type == 'H':
                window = grid[row_slice, col_slice]
                if not valid_mask(window).any():
                    rejected += 1
                    continue
                grid[row_slice, col_slice] = -1.0
                record.update(mechanism='invalidate', check='ok')
                assert not valid_mask(grid[row_slice, col_slice]).any()
            else:
                if mechanism == 'neighbour-mesh':
                    sign = int(rng.choice((-1, 1)))
                    neighbour = by_winding.get(low + sign)
                    if neighbour is None:
                        rejected += 1
                        continue
                    own = grid[row, col_slice].copy()
                    targets = neighbour_targets_mesh(
                        own, grid_of(neighbour), float(z))
                else:
                    sign = int(rng.choice((-1, 1)))
                    targets = neighbour_targets_turn(
                        grid, row, mask, centre, col_slice, sign)
                if targets is None:
                    rejected += 1
                    continue
                fraction = 1.0 if error_type == 'S' else 0.5
                before = grid[row_slice, col_slice].copy()
                peak = apply_window(grid, row_slice, col_slice, targets, fraction)
                if peak < 1.0:            # a "switch" that moved less than a voxel
                    grid[row_slice, col_slice] = before
                    rejected += 1
                    continue
                outside = grid[row_slice.stop:row_slice.stop + 1, col_slice]
                record.update(mechanism=mechanism, sign=sign,
                              peak_shift_vx=round(peak, 3), check='ok')
                del outside
                kink = max_kink_deg(grid, row_slice, col_slice)
                record['kink_deg'] = round(kink, 3)
                if args.smoothed:
                    record['kink_smoothed_deg'] = round(
                        max_kink_smoothed_deg(grid, row_slice, col_slice), 3)

            injections.append(record)
            spans.append((col_slice.start, col_slice.stop))
            placed += 1
        print(f'{error_type}: {placed} placed', flush=True)

    # Pristine plausibility floor, per touched segment, from a dedicated stream so
    # corpus content does not depend on how many draws were rejected above.
    pristine_rng = np.random.default_rng(args.seed + 1)
    pristine = {}
    pristine_s = {}
    for name in sorted(grids):
        clean = scrolls.segment_grid(name, scroll, args.grid_cache)
        if args.smoothed:
            pristine[name], pristine_s[name] = pristine_kink_median(
                pristine_rng, clean, smoothed=True)
        else:
            pristine[name] = pristine_kink_median(pristine_rng, clean)
    for record in injections:
        if 'kink_deg' in record:
            floor = pristine[record['segment']]
            record['plausible'] = bool(floor is not None
                                       and record['kink_deg'] <= floor)
            if args.smoothed:
                floor_s = pristine_s[record['segment']]
                record['plausible_smoothed'] = bool(
                    floor_s is not None
                    and record['kink_smoothed_deg'] <= floor_s)

    plausible = [r for r in injections if 'plausible' in r]
    share = (sum(r['plausible'] for r in plausible) / len(plausible)
             if plausible else 0.0)
    share_s = (sum(r['plausible_smoothed'] for r in plausible) / len(plausible)
               if args.smoothed and plausible else None)

    os.makedirs(os.path.join(args.out, 'grids'), exist_ok=True)
    for name, grid in sorted(grids.items()):
        np.save(os.path.join(args.out, 'grids', f'{name}.npy'), grid)
    manifest = {
        'protocol': 'oneshot/PROTOCOL.md v1',
        'scroll': scroll.key, 'seed': args.seed, 'per_type': args.per_type,
        'voxel_mm': voxel_mm, 'mechanism': mechanism,
        'lambda_mm': LAMBDA_MM, 'height_mm': HEIGHT_MM,
        'z_quantiles': [float(z) for z in heights],
        'rejected_draws': rejected,
        'pristine_kink_median_deg': pristine,
        'plausible_share': round(share, 4),
        'injections': injections,
    }
    if args.smoothed:
        manifest['smoothing_window'] = 3
        manifest['pristine_kink_median_smoothed_deg'] = pristine_s
        manifest['plausible_share_smoothed'] = round(share_s, 4)
    path = os.path.join(args.out, 'manifest.json')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1, sort_keys=True)
    extra = (f' (smoothed measure: {share_s:.2f})'
             if share_s is not None else '')
    print(f'{len(injections)} injections into {len(grids)} segments; '
          f'{rejected} draws rejected; locally-plausible share {share:.2f}'
          f'{extra}; manifest at {path}', flush=True)


if __name__ == '__main__':
    main()
