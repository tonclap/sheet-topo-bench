"""Everything that is a property of a scroll, in one place.

The measurement — how well the surface prediction resolves the gap between two
labelled sheets — is not about PHerc. Paris 4. But the code that made it was: paths,
pyramid level, segment naming and the umbilicus were all constants in
`absolute_winding_calibration.py`, so "run it on another scroll" was not an option a
reader had. This module makes the scroll a parameter and nothing else changes.

Four things differ between the two scrolls, and each one would have silently produced
a wrong number rather than an error:

- **the annotation frame is a different pyramid level.** The meshes are written in the
  voxel grid of the scan the prediction was computed on. For Paris 4 that grid is the
  prediction's level 2 (18946x8174x8174); for PHerc0139 it is level 0
  (20974x6621x6621). Sampling the wrong level does not fail — it reads a volume four
  times too coarse and counts fewer laminae.
- **segment names.** Paris 4 publishes winding *ranges*, `…-w010-027`, twice each, and
  only the `20260701…` copy carries a mesh in the prediction's frame. PHerc0139
  publishes single windings, `…-w023_2026…`, once each, 37 of them, 23 through 59
  without a gap.
- **the mesh directory.** `mesh/intermediate/tifxyz_original/` on Paris 4;
  `mesh/<segment-id>-on-20250728140407-9.362um.tifxyz/` on PHerc0139, where the name
  states the scan and the id is the segment's own. All 37 ship it (checked 13.08.2026).
- **the umbilicus.** Published for Paris 4 on the data server; for PHerc0139 there is
  none — `spiral_datasets/PHerc0139/` holds only `20250728140407/tracks/` (404 on
  `umbilicus.json`, checked 13.08.2026). It is not needed: see `Centre` below.

What does *not* differ, and is worth saying because it makes the two comparable: both
annotation frames are ~9.4-9.6 um per voxel, so sheet spacing is ~21-22 voxels in
both and the voxel-valued defaults (step, tolerances) carry over unchanged.

Usage — the self-check this module ships with:

    python scrolls.py --scroll PHerc0139 --cache ../../output/cache0139 \\
        --grid-cache ../../output/grids0139
"""
import argparse
import dataclasses
import io
import os
import re
import sys
import warnings

import numpy as np
import tifffile

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

# `absolute_winding_calibration.py` sits beside this file in the published repository
# and one directory over in the working one. Add the second location only if it exists,
# so the published copies are byte-identical to the ones that produced the numbers.
_STANDALONE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'standalone')
if os.path.isdir(_STANDALONE):
    sys.path.insert(0, _STANDALONE)
import absolute_winding_calibration as awc                            # noqa: E402


@dataclasses.dataclass(frozen=True)
class Scroll:
    """One scroll's data layout. Every field is a fact checked against the server."""

    key: str
    prediction: str
    ct: str
    segments: str
    level: str
    # Regex over a segment directory name, yielding the winding interval. Group 'low'
    # is required; 'high' is optional and defaults to 'low' for single-winding names.
    name_pattern: str
    # Directory under `segments/<name>/` holding the tifxyz mesh in the prediction's
    # frame. `{id}` expands to the part of the segment name before `-w`.
    mesh: str
    # Only segment copies starting with this prefix carry that mesh; None = no copies.
    copy_prefix: str | None
    umbilicus_url: str | None
    # Chunk cache directory for this scroll, relative to a cache root. Per-scroll and
    # never shared: chunk files are keyed by pyramid level and coordinates, which
    # collide between scrolls. The names are the ones the caches already have on disk.
    cache_name: str
    # The prediction is a binary mask (its name ends in a threshold), so any cut
    # between 0 and 255 is the same mask. Kept per-scroll because that is a claim about
    # a file rather than about masks in general: checked on both, 13.08.2026 — a ray
    # through PHerc0139's `m7-L0-th0.2` reads 0 or 255 and nothing else.
    threshold: int = 115

    def segment_url(self, name, suffix=''):
        return f'{awc.S3}{self.segments}{name}/{suffix}'

    def mesh_dir(self, name):
        return self.mesh.format(id=name.split('-w')[0])


PARIS4 = Scroll(
    key='PHercParis4',
    prediction=awc.PREDICTION,
    ct=awc.CT,
    segments=awc.SEGMENTS,
    level='2',
    name_pattern=r'-w(?P<low>\d{3})-(?P<high>\d{3})$',
    mesh=awc.CHEAP_MESH,
    copy_prefix=awc.SEGMENT_COPY,
    umbilicus_url=awc.DATA_SERVER + 'umbilicus.json',
    cache_name='figcache',
)

PHERC0139 = Scroll(
    key='PHerc0139',
    prediction=('PHerc0139/representations/predictions/surfaces/'
                '20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr/'),
    ct='PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/',
    segments='PHerc0139/segments/',
    level='0',
    name_pattern=r'-w(?P<low>\d{3})_',
    mesh='mesh/{id}-on-20250728140407-9.362um.tifxyz/',
    copy_prefix=None,
    umbilicus_url=None,
    cache_name='cache0139',
)

SCROLLS = {scroll.key: scroll for scroll in (PARIS4, PHERC0139)}


def add_argument(parser):
    parser.add_argument('--scroll', default=PARIS4.key, choices=sorted(SCROLLS),
                        help='which scroll to measure (default: %(default)s)')


def labelled_segments(scroll):
    """Segments whose names carry absolute windings, as (name, low, high).

    `high` is inclusive, as the mesh geometry showed it to be: a mesh named w010-027
    spans 18.0 turns, not 17. A single-winding name is the same convention with
    low == high, spanning 1.0 turn.
    """
    body = awc.fetch(f'{awc.S3}?list-type=2&max-keys=1000&delimiter=/&'
                     f'prefix={scroll.segments}', timeout=60).decode()
    if '<IsTruncated>true</IsTruncated>' in body:
        raise RuntimeError(f'{scroll.key}: segment listing was truncated; add '
                           f'pagination before trusting the winding coverage')
    seen = {}
    for prefix in re.findall(r'<Prefix>([^<]+)</Prefix>', body):
        if not prefix.endswith('/'):
            continue
        name = prefix.split('/')[-2]
        match = re.search(scroll.name_pattern, name)
        if not match:
            continue
        groups = match.groupdict()
        low = int(groups['low'])
        high = int(groups.get('high') or low)
        seen.setdefault((low, high), []).append(name)

    out, skipped = [], []
    for (low, high), names in sorted(seen.items()):
        usable = ([n for n in names if n.startswith(scroll.copy_prefix)]
                  if scroll.copy_prefix else names)
        if usable:
            out.append((usable[0], low, high))
        else:
            skipped.append(f'w{low:03d}-{high:03d}')
    if skipped:
        print(f'{scroll.key}: skipping {len(skipped)} winding range(s) with no '
              f'{scroll.copy_prefix} copy and therefore no mesh in the prediction '
              f'frame: {", ".join(skipped)}', flush=True)
    if not out:
        raise RuntimeError(f'{scroll.key}: no segment name matched '
                           f'{scroll.name_pattern!r} — the naming changed, and '
                           f'silently measuring nothing is the failure mode this '
                           f'project files bug reports about')
    return out


def segment_grid(name, scroll, cache):
    """The tifxyz grid with its shape kept: (rows, cols, 3), -1 where invalid."""
    # Keyed by segment name alone: names carry their own date-id and are unique across
    # scrolls, so one grid cache can hold both — and the Paris 4 grids already on disk
    # stay valid. The *chunk* caches must stay per-scroll: those are keyed by pyramid
    # level and coordinates, which do collide.
    path = os.path.join(cache, f'{name}.npz')
    if os.path.exists(path):
        with np.load(path) as data:
            return data['grid']
    os.makedirs(cache, exist_ok=True)
    channels = []
    for channel in 'xyz':
        url = scroll.segment_url(name, scroll.mesh_dir(name) + f'{channel}.tif')
        raw = awc.fetch(url, timeout=600)
        if raw is None:
            raise RuntimeError(f'{scroll.key}/{name}: no {channel}.tif at {url}')
        channels.append(tifffile.imread(io.BytesIO(raw)))
    grid = np.stack(channels, -1).astype(np.float32)
    tmp = f'{path}.{os.getpid()}.part.npz'
    np.savez_compressed(tmp, grid=grid)
    os.replace(tmp, path)
    return grid


def row_heights(grid):
    """Mean z of each grid row, NaN where the row has no valid point.

    A row with no valid point is normal — a mesh is a ragged strip in a rectangular
    grid — so numpy's "mean of empty slice" warning is noise here, not news, and is
    silenced rather than printed 37 times a slice.
    """
    valid = (grid[..., 0] != -1) & (grid[..., 1] != -1)
    with np.errstate(invalid='ignore'):
        heights = np.where(valid, grid[..., 2], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(heights, axis=1), valid


def iso_z_row(grid, z, max_dz=None):
    """The grid row nearest this height, its valid mask — or (None, None).

    `max_dz` is the generalisation that matters. Paris 4's 28 segments all span the
    full height of the fit domain, so taking the nearest row unconditionally was
    harmless there. PHerc0139's 37 segments do not: they cover different z ranges, and
    asking for a height a segment never reaches would return its topmost row and treat
    it as if it sat at the requested z — a fabricated rung on the ladder, with no error
    anywhere. Rows are ~19 voxels apart, so two row spacings is the default: near
    enough to be the same height, far enough not to reject a usable row.
    """
    per_row, valid = row_heights(grid)
    if np.all(np.isnan(per_row)):
        return None, None
    row = int(np.nanargmin(np.abs(per_row - z)))
    if max_dz is None:
        spacing = np.nanmedian(np.abs(np.diff(per_row)))
        max_dz = 2 * (spacing if np.isfinite(spacing) and spacing > 0 else 19.0)
    if abs(per_row[row] - z) > max_dz:
        return None, None
    return row, valid[row]


def fit_circle(points_xy):
    """Least-squares (Kasa) circle through 2-D points: returns (cx, cy, radius)."""
    x, y = points_xy[:, 0].astype(float), points_xy[:, 1].astype(float)
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    solution, *_ = np.linalg.lstsq(design, x ** 2 + y ** 2, rcond=None)
    cx, cy = solution[0] / 2, solution[1] / 2
    return cx, cy, float(np.sqrt(max(solution[2] + cx ** 2 + cy ** 2, 0.0)))


class Centre:
    """Where counting starts, at a given height.

    Two sources, one interface, because everything downstream only ever asks "where is
    the centre at this z".

    - **Published umbilicus** (Paris 4): control points interpolated in z, refusing to
      extrapolate — the existing behaviour, unchanged.
    - **Mesh geometry** (PHerc0139, and any scroll without a published umbilicus): a
      least-squares circle through the innermost labelled turn at that height.

    On the honesty of the second: a full turn is not a circle — its radius grows by one
    sheet spacing (~21 vx) over the turn against a radius of several hundred — so the
    fit is centred to roughly a third of a sheet spacing, not exactly. That is a claim
    that has to be checked rather than asserted, and the check cannot be the turn-span
    identity used elsewhere: accumulated turn over a closed loop is 1.0 for *any*
    centre inside it, so that test is blind here. The test with teeth is the ladder
    itself — a ray from the centre must cross each single-winding mesh exactly once,
    in winding order. `check()` below measures that, and `ray_vs_mesh.py` reports it
    per ray at run time.
    """

    def __init__(self, scroll, cache, grid_cache):
        self.scroll, self.cache, self.grid_cache = scroll, cache, grid_cache
        self.control = None
        self._fits = {}
        self._grid = None
        if scroll.umbilicus_url:
            self.control = awc.load_umbilicus(None, cache)
            self.source = f'published umbilicus ({len(self.control)} control points)'
        else:
            self.inner = min(labelled_segments(scroll), key=lambda item: item[1])
            self.source = (f'circle fitted to the innermost labelled turn '
                           f'(w{self.inner[1]:03d}, {self.inner[0]})')

    def at(self, z):
        """(x, y) of the centre at this height.

        Callers ask at each mesh row's own height, which differs from the slice height
        by up to a row spacing and differs again per segment, so the fit is memoised on
        the height rounded to a voxel: refitting a circle per segment would cost 37
        grid loads a slice and move the centre by less than the rounding.
        """
        if self.control is not None:
            return awc.umbilicus_at(self.control, z)
        z = round(float(z))
        if z not in self._fits:
            if self._grid is None:
                self._grid = segment_grid(self.inner[0], self.scroll, self.grid_cache)
            grid = self._grid
            row, valid = iso_z_row(grid, z)
            if row is None or valid.sum() < 10:
                raise ValueError(
                    f'{self.scroll.key}: the innermost labelled turn '
                    f'w{self.inner[1]:03d} does not reach z={z:.0f}, so there is '
                    f'nothing to centre on; counting from a guessed origin would be '
                    f'quietly wrong rather than absent')
            cx, cy, radius = fit_circle(grid[row][valid][:, :2])
            self._fits[z] = (cx, cy, radius)
        cx, cy, _ = self._fits[z]
        return float(cx), float(cy)

    def describe(self, z):
        centre = self.at(z)
        extra = ''
        if self.control is None:
            extra = f', inner turn radius {self._fits[round(float(z))][2]:.0f} vx'
        return (f'{self.scroll.key} z={z:.0f}: centre ({centre[0]:.1f}, '
                f'{centre[1]:.1f}) from {self.source}{extra}')


def ladder_curves(scroll, centre, direction, grid_cache, z, segments=None,
                  min_span=0.95, verbose=True):
    """The labelled-sheet ladder at one height: (points, winding) per segment.

    Both the geometry check and the metric need exactly this, and they used to build it
    twice with slightly different wording. One definition, one set of exclusions,
    printed rather than silent:

    - a segment that does not reach this height is absent, not stretched to it;
    - a segment whose mesh covers less than `min_span` of the turns its name claims is
      excluded loudly, because the reference rests on a rung being where its name says.

    On the gate being one-sided, which is a change forced by the second scroll and not
    a loosening to make it pass. Paris 4's meshes span their named turns to within
    0.2%, so the old symmetric 0.97-1.03 window was really testing one thing: does the
    mesh cover its name. PHerc0139's do not behave that way — its single-winding
    annotations overrun a full turn by a median 4-8% (w026 by 23%), i.e. neighbouring
    sheets are annotated with a deliberate overlap. That overrun is harmless here:
    every crossing carries its own interpolated winding, so the extra sliver at the end
    of w041 is labelled 42.0x and coincides with the start of w042 rather than
    contradicting it — and a ray that meets the same sheet twice at the seam is exactly
    what `ray_vs_mesh.py` counts as a revisit. Undershoot is not harmless: a mesh
    covering a third of its turn (w050 here) leaves a hole in the ladder that reads as
    an extra prediction run in a doubled gap.

    A short mesh is therefore excluded and a long one kept. On Paris 4 this changes
    nothing at all — no segment there exceeds 1.03 — which is why the earlier slice
    reproduces to the digit.
    """
    segments = segments or labelled_segments(scroll)
    curves, missing, short, ratios = [], [], [], []
    for name, low, high in segments:
        grid = segment_grid(name, scroll, grid_cache)
        row, valid = iso_z_row(grid, z)
        if row is None or valid.sum() < 10:
            missing.append(f'w{low:03d}')
            continue
        points, turns, _ = turn_profile(grid, row, valid, centre)
        named = high - low + 1
        ratio = abs(float(turns[-1] - turns[0])) / named
        ratios.append(ratio)
        if ratio < min_span:
            short.append(f'w{low:03d}-{high:03d} ({ratio:.2f})')
            continue
        # Label from the mesh's *inner* end outward, which is what the name means. The
        # inner end is whichever end of the grid the geometry says it is — the two
        # scrolls run opposite ways (see `traversal_direction`), so anchoring on the
        # grid's first point instead would put PHerc0139's labels a winding low.
        advance = (turns - turns[0]) * direction
        curves.append((points, low + advance - advance.min()))
    if verbose:
        print(f'{scroll.key} z={z:.0f}: {len(curves)} of {len(segments)} labelled '
              f'segments usable, median turns/named '
              f'{np.median(ratios) if ratios else float("nan"):.3f}'
              + (f'; {len(missing)} do not reach this height' if missing else '')
              + (f'; {len(short)} cover less than {min_span:.2f} of their named turns: '
                 f'{", ".join(short)}' if short else ''), flush=True)
    return curves


def ray_reach(curves, origin):
    """How far out a ray has to be traced to leave the labelled region entirely."""
    return max(float(np.hypot(points[:, 0] - origin[0],
                              points[:, 1] - origin[1]).max())
               for points, _ in curves)


def centre_from_control(control):
    """A `Centre`-shaped view of already-loaded umbilicus control points."""
    return _ControlCentre(control)


class _ControlCentre:
    def __init__(self, control):
        self.control = control

    def at(self, z):
        return awc.umbilicus_at(self.control, z)


def turn_profile(grid, row, mask, centre):
    """Cumulative turn around the centre along the row, in windings."""
    points = grid[row][mask]
    cx, cy = centre.at(float(np.median(points[:, 2])))
    angle = np.unwrap(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    return points, (angle - angle[0]) / (2 * np.pi), (cx, cy)


def ray_crossings(polyline, winding, origin, unit, max_radius):
    """Where the ray meets this polyline, as (radius, winding) pairs.

    Straight segment-by-segment intersection: for each consecutive pair of mesh points,
    solve for the point where the connecting segment crosses the ray's line, keep it if
    it lies inside the pair and at positive distance along the ray.
    """
    perp = np.array([-unit[1], unit[0]])
    side = (polyline[:, :2] - origin[:2]) @ perp
    along = (polyline[:, :2] - origin[:2]) @ unit
    # A crossing of the ray's *line* is a sign change in the perpendicular coordinate.
    change = np.flatnonzero(np.sign(side[:-1]) != np.sign(side[1:]))
    out = []
    for index in change:
        first, second = side[index], side[index + 1]
        if first == second:
            continue
        t = first / (first - second)
        radius = along[index] + t * (along[index + 1] - along[index])
        if 0 <= radius <= max_radius:
            out.append((float(radius),
                        float(winding[index] + t * (winding[index + 1] - winding[index]))))
    return out


def traversal_direction(scroll, centre, grid_cache, z, segments=None, quiet=False):
    """Does walking a mesh grid's axis 1 move towards higher windings?

    One global fact about the scroll — a fixed grid cannot change orientation with
    height — and settling it per segment was the third self-inflicted bug of the 13.08
    session. Two further ways of settling it died against the second scroll before this
    one stood up, and both died loudly, which is the whole point of writing the check
    down:

    - *radius at the two ends of the mesh.* Right for Paris 4's `w010-027` (254 -> 672
      voxels over 18 turns), useless for a single-turn mesh: PHerc0139's annotations
      overrun their turn by 4-25%, so the far end sits up to 90 degrees past the near
      one, where an out-of-round scroll's radius has nothing to do with the sheet step.
      That vote came out 77% inward — a 23% minority a fixed grid cannot have.
    - *the order of the ladder.* Appealing, since ordered crossings are what the
      direction is for, but on single-turn meshes it cannot see the answer: each rung
      is labelled by its own integer either way, and only the fraction flips. Measured:
      92.2% against 91.2%, a coin toss.

    What settles it is where a mesh's far end lands **relative to its neighbours**, and
    that needs no interpolation and no assumption about the scroll's shape: take the
    last point of a mesh row, read the angle it sits at, and compare its radius with
    the radius of the next mesh out and the previous mesh in, at that same angle, using
    their own points. If advancing the accumulated angle moves outward, the far end of
    mesh k lands on mesh k+1. Measured over all meshes at all heights: Paris 4 says
    outward 92%, PHerc0139 says inward 79%.

    A second, independent physical rule is kept as a cross-check: follow one mesh a
    full turn from its own start and compare the radius there with the radius at the
    start — same angle, so the shape cancels, leaving one lamina of separation with a
    sign. It agrees with the neighbour test on both scrolls: outward 99% on Paris 4,
    inward 89% on PHerc0139. Two different measurements, same answer per scroll.

    **The two scrolls genuinely differ, and that is a fact about the data rather than a
    bug here.** On Paris 4 a mesh grid advances outward as its accumulated angle grows;
    on PHerc0139 it advances inward. A name therefore anchors the opposite end of the
    turn on the two scrolls, so labels are anchored by geometry rather than by grid
    order: whichever end of the mesh is the inner one is labelled `low`, and the labels
    run outward from there. That reproduces the published names on both scrolls (median
    disagreement 0.07 windings and 0.00) without assuming which way the grid runs.

    What is *not* at stake, measured rather than argued: the sign does not move the
    result. Running a slice both ways gives identical shares to four decimal places on
    both scrolls (0139 z=5026, Paris 4 z=15694) — gaps are ordered by radius, not by
    label — and only the winding labels, and hence the band rows of the tables, shift.
    So the 8% and 21% minorities above bound a label, not a measurement.

    `z` may be several heights, and for a whole run it should be: the answer is a
    property of the scroll, not of a slice. Settling it per slice made a run hostage to
    its worst slice — at PHerc0139's lowest protocol height only 21 of 37 meshes reach
    the slice and two of those have a negative accumulated turn (w054 -0.06, w055
    -0.36: degenerate rows at the bottom edge of the labelled region).
    """
    segments = segments or labelled_segments(scroll)
    heights = [z] if np.isscalar(z) else list(z)
    neighbour_votes, step_votes, steps, ambiguous = [], [], [], 0
    for height in heights:
        rows = {}
        for name, low, high in segments:
            grid = segment_grid(name, scroll, grid_cache)
            row, valid = iso_z_row(grid, height)
            if row is None or valid.sum() < 10:
                continue
            try:
                points, turns, (cx, cy) = turn_profile(grid, row, valid, centre)
            except ValueError:                  # no centre at this height at all
                continue
            advance = turns - turns[0]
            angle = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
            radius = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
            rows[low] = (angle, radius, advance, high)
            # Cross-check: the sheet step a full turn along, at a fixed angle.
            walk, flip = (advance, 1.0) if advance[-1] > 0 else (-advance, -1.0)
            if walk[-1] >= 1.0:
                step = float(np.interp(1.0, walk, radius) - radius[0])
                step_votes.append(flip * np.sign(step) > 0)
                steps.append(abs(step))

        for low, (angle, radius, advance, high) in rows.items():
            outer = [k for k in sorted(rows) if k > high]
            inner = [k for k in sorted(rows) if k < low]
            if not outer or not inner:
                continue
            end = int(np.argmax(advance) if advance[-1] > 0 else np.argmin(advance))
            here = angle[end]
            near = []
            for key in (outer[0], inner[-1]):
                other_angle, other_radius = rows[key][:2]
                gap = np.abs(np.angle(np.exp(1j * (other_angle - here))))
                near.append((float(other_radius[np.argmin(gap)]), float(gap.min())))
            if max(gap for _, gap in near) > 0.15:   # no neighbour point at this angle
                ambiguous += 1
                continue
            to_outer = abs(radius[end] - near[0][0])
            to_inner = abs(radius[end] - near[1][0])
            if to_outer == to_inner:
                ambiguous += 1
                continue
            # The far end is in the direction of growing accumulated angle, so landing
            # on the outer neighbour means growing angle runs outward.
            neighbour_votes.append((to_outer < to_inner) == (advance[-1] > 0))

    if len(neighbour_votes) < 3:
        raise ValueError(f'{scroll.key}: only {len(neighbour_votes)} mesh-slice(s) have '
                         f'both an inner and an outer neighbour at the heights given; '
                         f'not enough to settle traversal')
    share = float(np.mean(neighbour_votes))
    outward = share > 0.5
    agreement = max(share, 1 - share)
    step_share = float(np.mean(step_votes)) if step_votes else float('nan')
    if not quiet:
        print(f'{scroll.key}: traversal settled against neighbouring meshes over '
              f'{len(neighbour_votes)} mesh-slice(s): growing accumulated angle runs '
              f'{"outward" if outward else "inward"}, {agreement:.0%} agree '
              f'({ambiguous} ambiguous). Cross-check (sheet step a turn along): '
              f'{max(step_share, 1 - step_share):.0%} of {len(step_votes)} say '
              f'{"outward" if step_share > 0.5 else "inward"}, median step '
              f'{np.median(steps):.1f} vx.', flush=True)
    if agreement < 0.7:
        raise ValueError(f'{scroll.key}: the neighbour test splits {agreement:.0%}/'
                         f'{1 - agreement:.0%}; the meshes do not agree on which way '
                         f'they run and a ladder built on that would be fiction')
    if (step_share > 0.5) != outward:
        print(f'{scroll.key}: WARNING — the two traversal tests disagree; labels are '
              f'anchored to the published names either way, and the measured shares do '
              f'not depend on the sign, but read the winding labels with that in mind.',
              flush=True)
    return 1.0 if outward else -1.0


def labelled_z_extent(scroll, grid_cache, segments=None):
    """(zmin, zmax) actually covered by the labelled meshes.

    The protocol asks for heights at quantiles of this extent, so it has to be read off
    the data rather than typed in: Paris 4 and PHerc0139 do not even share an axis
    length. Taken as the range where labelling exists at all, i.e. the union.
    """
    lows, highs = [], []
    for name, _, _ in segments or labelled_segments(scroll):
        per_row, _ = row_heights(segment_grid(name, scroll, grid_cache))
        if np.all(np.isnan(per_row)):
            continue
        lows.append(np.nanmin(per_row))
        highs.append(np.nanmax(per_row))
    if not lows:
        raise RuntimeError(f'{scroll.key}: no labelled mesh has a usable height')
    return float(min(lows)), float(max(highs))


def protocol_heights(scroll, grid_cache, count=5, segments=None):
    """The heights the pre-registered protocol runs on: `count` equidistant quantiles.

    Declared before any run, and deliberately not choosable afterwards: the previous
    write-up's weakest point was a slice that looked hand-picked. Equidistant quantiles
    of the labelled z-extent, endpoints excluded so no height sits on the edge where a
    single segment carries the whole slice.
    """
    low, high = labelled_z_extent(scroll, grid_cache, segments)
    fractions = (np.arange(count) + 0.5) / count
    return [float(low + f * (high - low)) for f in fractions]


def open_prediction(scroll, cache, **kwargs):
    return awc.Prediction(cache, base=scroll.prediction, level=scroll.level, **kwargs)


def open_scan_mask(scroll, cache):
    return awc.ScanMask(cache, base=scroll.ct, level=scroll.level)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_argument(parser)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--grid-cache', required=True)
    parser.add_argument('--heights', type=int, default=5)
    args = parser.parse_args()
    scroll = SCROLLS[args.scroll]

    segments = labelled_segments(scroll)
    windings = [(low, high) for _, low, high in segments]
    print(f'{scroll.key}: {len(segments)} labelled segments, '
          f'w{windings[0][0]:03d}..w{windings[-1][1]:03d}, '
          f'{sum(h - l + 1 for l, h in windings)} named windings')

    pred = open_prediction(scroll, args.cache)
    print(f'prediction level {scroll.level}: shape {pred.shape}, chunks {pred.chunks}')
    mask = open_scan_mask(scroll, args.cache)
    if list(mask.shape) != list(pred.shape):
        raise RuntimeError(
            f'{scroll.key}: the CT level {scroll.level} is {mask.shape} but the '
            f'prediction is {pred.shape}; the mask filter would reject the wrong '
            f'points, so the level is wrong for this scroll')
    print(f'CT level {scroll.level}: shape {mask.shape} — same frame, mask usable')

    low, high = labelled_z_extent(scroll, args.grid_cache, segments)
    heights = protocol_heights(scroll, args.grid_cache, args.heights, segments)
    print(f'labelled z-extent {low:.0f}..{high:.0f}; protocol heights '
          + ', '.join(f'{z:.0f}' for z in heights))

    centre = Centre(scroll, args.cache, args.grid_cache)
    print(f'\n{"z":>7s}  {"segments here":>13s}  centre')
    for z in heights:
        here = 0
        for name, _, _ in segments:
            row, valid = iso_z_row(segment_grid(name, scroll, args.grid_cache), z)
            here += bool(row is not None and valid.sum() >= 10)
        try:
            print(f'{z:7.0f}  {here:13d}  {centre.describe(z)}')
        except ValueError as error:
            print(f'{z:7.0f}  {here:13d}  no centre: {error}')


if __name__ == '__main__':
    main()
