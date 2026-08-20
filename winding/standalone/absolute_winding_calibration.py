"""Absolute winding calibration for PHerc. Paris 4 — self-contained.

Winding fields recovered from CT are relative: they get the differences between
windings right and leave the origin unknown. winding-sync calls absolute counts "not
yet calibrated" in its own README; constraint-gauge, the external benchmark for
winding generators, writes "winding may be relative, only differences are scored"
into its rules. So the offset is measured by nobody. This measures it.

    absolute winding = laminae counted from the umbilicus + 6      (+- 2)

valid over z ~ 3100..15700, with the regression slope as a built-in applicability
gate: it needs no absolute labels, and at the top of the fit domain (z = 18000) it
collapses to 0.465 on its own, correctly refusing the slice.

The spread was published as (+1 / -2) on five passing slices; two more inside the same
domain (z = 3500 and z = 4000) came out at C = +8 on 13.08.2026 and widened it to +-2.
Those two also showed the gate is necessary rather than sufficient: at z = 3500 the
slope passes at 1.018 while the second estimator reads +2.8 against the grid search's
+8, so a wide disagreement between the two estimators is itself a reason to distrust
a slice's constant. See README, "Results".

Ground truth is the 28 PHerc. Paris 4 segments whose names are absolute winding
intervals (w010-027 ... w128-129) — the numbering the official spiral fit and the rest
of the project work in. There is exactly **one** absolute ground truth behind it: the
fit takes absolute winding annotations as an input and no other input carries an
absolute number, so the segment names are the 59 hand-placed labels of
`abs_winding.json` propagated across windings 10-129. This does not measure two
sources agreeing; it measures a counter against the project's numbering.

COUNTING CONVENTION — the whole result is one integer, so the conventions that move
it by one have to be written down rather than left to the code:

- a lamina is a maximal run of mask at or above `--threshold`, and a run already in
  progress at the first sample counts. Rays start at the umbilicus, which is not
  inside a lamina, so this does not fire here — but it would for a different origin;
- the ray runs from the umbilicus control point (not from the first lamina) out to
  the target point, and the lamina containing the target is counted;
- winding numbers follow the segment names, which are 1-based: `w010` is the tenth
  winding, so the constant is stated in that convention.

Everything streams by range request: no GPU, no bulk download.

Usage:
    python absolute_winding_calibration.py --cache ./chunks --mesh-cache ./meshes

INPUT PROVENANCE. Everything is fetched automatically and anonymously, but from two
different hosts, which is worth stating because it is not guessable:

- the surface prediction and the segment meshes come from the S3 bucket
  `vesuvius-challenge-open-data`;
- the umbilicus comes from the **data server**, `dl.ash2txt.org`, as part of the
  spiral-input PHerc. Paris 4 dataset. It is not in the S3 bucket — that bucket has
  no `datasets/` prefix at all.

The umbilicus is load-bearing rather than incidental: the constant is defined
relative to where counting starts, so a different umbilicus is a different constant.
Pass `--umbilicus` to override with a local copy.
"""
import argparse
import collections
import concurrent.futures as futures
import io
import json
import os
import re
import urllib.error
import urllib.request

import numcodecs
import numpy as np
import tifffile

import sys
# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

S3 = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/'
PREDICTION = ('PHercParis4/representations/predictions/surfaces/'
              '20260411134726-surface-20260413141734-surface-recto-2um-ps256-'
              'L0-th0.45.zarr/')
# The raw scan the prediction was computed on, in the same frame: level 0 is byte-for-
# byte the same shape as the prediction's, so level 2 is the annotation frame voxel for
# voxel. Used only to ask whether a sampled point is inside the scan at all — see
# `ScanMask`.
CT = 'PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/'
LEVEL = '2'
SEGMENTS = 'PHercParis4/segments/'
# The cheap `…-45.532um.tifxyz` mesh is a trap: it is defined on scan 20260310170716,
# a different scan from the prediction's 20260411134726. Fitting a scale factor for it
# peaks at the nominal 4.75 — at 0.244 on-mask against a random-cloud baseline of
# 0.241, i.e. no signal. `tifxyz_original` is in the prediction's own frame.
CHEAP_MESH = 'mesh/intermediate/tifxyz_original/'
# 28 winding ranges are published, each in two copies; only this one ships the mesh
# above. (Two further directories, w046-052_jordi and w053-058_jordi, are alternative
# takes on ranges already covered, not additional ranges.)
SEGMENT_COPY = '20260701'
# The annotation corpus lives on the data server, not in the S3 bucket. Anonymous
# access verified 11.08.2026; the file served here is byte-identical to the one the
# published constant was measured on (sha256 c5f30b0d135c1d33…).
DATA_SERVER = 'https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/'
# Applicability gate. The slope is computed from the same rays with no absolute
# labels involved, so it is a self-check the tool can run anywhere: near 1 the count
# tracks truth and only the origin is unknown; at the top of the fit domain (z=18000)
# it falls to 0.465 and the constant must not be read off that slice.
SLOPE_GATE = (0.8, 1.2)
NULL_PERMUTATIONS = 200


def fetch(url, byte_range=None, timeout=300, attempts=4):
    """One GET. `None` means a 404 and nothing else.

    A zarr chunk that was never written is legitimately absent and reads as the fill
    value. Anything else — a timeout, DNS, a 500, a proxy — is not absence, and
    swallowing it would quietly return a volume of zeros and a confidently wrong
    constant. That silent-failure shape is the same one this project files bug reports
    about; it does not belong in its own tool.

    Retrying is not a weakening of that rule, and it earns its place: a single TLS
    handshake timeout killed two twenty-minute runs during the 13.08 correction work.
    After `attempts` tries the error is still raised.
    """
    request = urllib.request.Request(url)
    if byte_range:
        request.add_header('Range', 'bytes=%d-%d' % byte_range)
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout).read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except Exception:                                          # noqa: BLE001
            if attempt == attempts - 1:
                raise
    return None


class ScanMask:
    """Is a point inside the scanned volume at all?

    Added 13.08.2026, and it is a correction rather than a feature. The sampler below
    filtered mesh points only by the tifxyz invalid marker, so it happily scored points
    where the fitted segment surfaces run past the edge of the scroll — 21% of the mesh
    points at z = 15694 and 51% at z = 18000. A ray aimed at a point that is not on the
    scroll counts the wrong laminae, and that contamination was holding the published
    regression slope near 1. See README, "Correction, 13 August 2026 (ii)".

    The test is the volume's own: the CT is published as `…-masked.zarr`, so a voxel
    reading exactly 0 is outside the scan mask. Air *inside* the mask reads around 50,
    so this rejects only what was never scanned — no tunable threshold is involved.
    The CT is stored uncompressed, so one z-plane of a chunk is 16 KB of contiguous
    bytes at a computable offset and a whole slice of points costs a handful of range
    requests and no decode.
    """

    def __init__(self, cache, base=CT, level=LEVEL):
        self.cache, self.base, self.level = cache, base, level
        os.makedirs(cache, exist_ok=True)
        meta = json.loads(fetch(f'{S3}{base}{level}/.zarray', timeout=120).decode())
        if meta['compressor'] is not None or meta['dtype'] != '|u1':
            raise RuntimeError(f'the CT level is no longer raw uint8, so the byte '
                               f'offsets below would read garbage: {meta}')
        self.shape = meta['shape']
        self.chunk = meta['chunks'][0]
        self._planes = {}

    def _plane(self, cz, cy, cx, lz):
        key = (cz, cy, cx, lz)
        hit = self._planes.get(key)
        if hit is not None:
            return hit
        path = os.path.join(self.cache, f'ct{self.level}_{cz}_{cy}_{cx}_{lz}.bin')
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                buf = handle.read()
        else:
            size = self.chunk * self.chunk
            buf = fetch(f'{S3}{self.base}{self.level}/{cz}/{cy}/{cx}',
                        (lz * size, lz * size + size - 1), timeout=120)
            if buf is None:                       # never-written chunk: all fill value
                buf = bytes(size)
            elif len(buf) != size:
                raise RuntimeError(f'short read on CT chunk {cz}/{cy}/{cx}: '
                                   f'{len(buf)} of {size} bytes')
            tmp = f'{path}.{os.getpid()}.part'
            with open(tmp, 'wb') as handle:
                handle.write(buf)
            os.replace(tmp, path)
        plane = np.frombuffer(buf, np.uint8).reshape(self.chunk, self.chunk)
        self._planes[key] = plane
        return plane

    def inside(self, points_xyz, workers=16):
        points = np.asarray(points_xyz, float)
        index = np.rint(points[:, ::-1]).astype(np.int64)              # xyz -> zyx
        out = np.zeros(len(index), bool)
        need = {}
        for row, (z, y, x) in enumerate(index):
            if not all(0 <= v < s for v, s in zip((z, y, x), self.shape)):
                continue
            need.setdefault((z // self.chunk, y // self.chunk, x // self.chunk,
                             z % self.chunk), []).append(
                                 (row, y % self.chunk, x % self.chunk))
        missing = [key for key in need if key not in self._planes]
        if len(missing) > 1:
            with futures.ThreadPoolExecutor(min(workers, len(missing))) as pool:
                list(pool.map(lambda key: self._plane(*key), missing))
        for key, items in need.items():
            plane = self._plane(*key)
            for row, ly, lx in items:
                out[row] = plane[ly, lx] > 0
        return out


class Prediction:
    """Chunk-cached reader for one zarr pyramid level of the surface prediction."""

    def __init__(self, cache, base=PREDICTION, level=LEVEL, max_chunks=64,
                 fetch_workers=8):
        self.base, self.level, self.cache = base, level, cache
        os.makedirs(cache, exist_ok=True)
        header = self._fetch('.zarray')
        if header is None:
            raise RuntimeError(
                f'no .zarray at {S3}{base}{level}/ — the prediction moved or the '
                f'level is wrong; nothing below this point would be meaningful')
        meta = json.loads(header.decode())
        self.shape = meta['shape']
        self.chunks = meta['chunks']
        self.codec = numcodecs.get_codec(meta['compressor'])
        # A decoded chunk is 256^3 = 16.8 MB, so an unbounded cache reaches gigabytes
        # over a full run. Eviction costs a ~27 ms re-decode from disk; raise
        # max_chunks to trade memory for that.
        self._chunks = collections.OrderedDict()
        self.max_chunks = max_chunks
        self.fetch_workers = fetch_workers

    def _path(self, key):
        return os.path.join(self.cache, f'{self.level}_{key}'.replace('/', '_'))

    def _fetch(self, key):
        path = self._path(key)
        if os.path.exists(path):
            with open(path, 'rb') as handle:
                return handle.read()
        data = fetch(f'{S3}{self.base}{self.level}/{key}', timeout=300)
        if data is None:
            return None
        tmp = f'{path}.{os.getpid()}.part'             # atomic: concurrent fetchers
        with open(tmp, 'wb') as handle:                # must not read a half file
            handle.write(data)
        os.replace(tmp, path)
        return data

    def prefetch(self, keys):
        """Download the chunks not yet on disk, concurrently.

        A cold slice pulls ~500 chunks of ~1.3 MB, and that transfer is the wall
        clock — not the sampling (~48 s) and not the decode (27 ms each). The requests
        are independent and pure waiting, so threads are the whole fix: measured 75 s
        to 17.7 s on a cold 35-chunk workload.
        """
        missing = [key for key in keys
                   if key not in self._chunks and not os.path.exists(self._path(key))]
        if len(missing) < 2:
            return
        with futures.ThreadPoolExecutor(
                max_workers=min(self.fetch_workers, len(missing))) as pool:
            list(pool.map(self._fetch, missing))

    def chunk(self, cz, cy, cx):
        key = (cz, cy, cx)
        cached = self._chunks.get(key)
        if cached is not None:
            self._chunks.move_to_end(key)
            return cached
        raw = self._fetch(f'{cz}/{cy}/{cx}')
        block = (np.zeros(self.chunks, np.uint8) if raw is None else
                 np.frombuffer(self.codec.decode(raw), np.uint8).reshape(self.chunks))
        self._chunks[key] = block
        if len(self._chunks) > self.max_chunks:
            self._chunks.popitem(last=False)
        return block

    def sample(self, pts_xyz):
        """Nearest-voxel values for many points: one numpy gather per touched chunk."""
        pts = np.asarray(pts_xyz, float)
        idx = np.rint(pts[:, ::-1]).astype(np.int64)          # xyz -> zyx
        out = np.zeros(len(idx), np.int16)
        inside = np.ones(len(idx), bool)
        for axis in range(3):
            inside &= (idx[:, axis] >= 0) & (idx[:, axis] < self.shape[axis])
        if not inside.any():
            return out
        here = idx[inside]
        block_id = here // np.array(self.chunks, np.int64)
        local = here % np.array(self.chunks, np.int64)
        # Flatten the chunk coordinate before np.unique: sorting an (N, 3) array by
        # rows costs more than the gather it is meant to organise.
        grid = [-(-self.shape[a] // self.chunks[a]) for a in range(3)]
        flat = (block_id[:, 0] * grid[1] + block_id[:, 1]) * grid[2] + block_id[:, 2]
        keys, inverse = np.unique(flat, return_inverse=True)
        self.prefetch([f'{k // (grid[1] * grid[2])}/'
                       f'{k % (grid[1] * grid[2]) // grid[2]}/{k % grid[2]}'
                       for k in keys.tolist()])
        values = np.empty(len(here), np.int16)
        for k, key in enumerate(keys):
            take = inverse == k
            sub = local[take]
            cz, rest = divmod(int(key), grid[1] * grid[2])
            cy, cx = divmod(rest, grid[2])
            values[take] = self.chunk(cz, cy, cx)[sub[:, 0], sub[:, 1], sub[:, 2]]
        out[inside] = values
        return out


def load_umbilicus(path, cache_dir):
    """Umbilicus control points: a local file if given, otherwise the data server."""
    if path:
        with open(path, encoding='utf-8-sig') as handle:
            return json.load(handle)['control_points']
    cached = os.path.join(cache_dir, 'umbilicus.json')
    if not os.path.exists(cached):
        os.makedirs(cache_dir, exist_ok=True)
        data = fetch(DATA_SERVER + 'umbilicus.json', timeout=120)
        tmp = f'{cached}.{os.getpid()}.part'
        with open(tmp, 'wb') as handle:
            handle.write(data)
        os.replace(tmp, cached)
        print(f'fetched umbilicus.json from {DATA_SERVER}', flush=True)
    with open(cached, encoding='utf-8-sig') as handle:
        return json.load(handle)['control_points']


def umbilicus_at(control_points, z):
    """Umbilicus (x, y) at a given z, linearly interpolated between control points.

    Refuses to extrapolate. `np.interp` clamps to the end value outside the control
    range without saying so, and a silently wrong origin is a silently wrong constant
    — the whole result is defined relative to where counting starts. For PHerc.
    Paris 4 the control points span z 563..18240 (146 of them), so every slice used
    here is interpolated — though z=18000 sits in the *last* interval, 240 voxels
    below the top support, which is enough to rule out clamping but not to call it
    well anchored. A caller outside the range gets an error rather than a number.
    """
    zs = np.array([p['z'] for p in control_points], float)
    order = np.argsort(zs)
    zs = zs[order]
    if not zs[0] <= z <= zs[-1]:
        raise ValueError(
            f'z={z:.0f} is outside the umbilicus control range {zs[0]:.0f}..{zs[-1]:.0f}; '
            f'interpolation would silently clamp to the end point and the constant is '
            f'defined relative to the origin, so the result would be quietly wrong')
    xs = np.array([p['x'] for p in control_points], float)[order]
    ys = np.array([p['y'] for p in control_points], float)[order]
    return float(np.interp(z, zs, xs)), float(np.interp(z, zs, ys))


def count_runs(values, threshold):
    """Number of maximal runs at or above threshold — one run is one lamina.

    The prediction is a binary mask (0/255, `th0.45` in its name), so laminae are
    solid runs 4-5 voxels thick separated by zeros, and counting runs is exact to
    sub-voxel on short spans (median error 0.28-0.63 vx against human clicks).

    Counting rising edges is the same thing as walking the runs, and avoids a Python
    loop over every sample — 3.1 M iterations per slice at the default step.
    """
    above = (np.asarray(values) >= threshold).astype(np.int8)
    return int(np.count_nonzero(np.diff(above, prepend=np.int8(0)) == 1))


def count_to_point(pred, origin, point, threshold=115, step=0.5):
    """Laminae crossed walking from the umbilicus out to a point."""
    direction = point - origin
    length = float(np.linalg.norm(direction))
    if length < step:
        return 0
    direction = direction / length
    ts = np.arange(0.0, length + step, step)
    return count_runs(pred.sample(origin[None, :] + ts[:, None] * direction[None, :]),
                      threshold)


def labelled_segments():
    """The 28 segments whose names are absolute winding intervals."""
    body = fetch(f'{S3}?list-type=2&max-keys=1000&delimiter=/&prefix={SEGMENTS}',
                 timeout=60).decode()
    # Silently keeping the first page would drop segments and quietly narrow the
    # winding range the constant is fitted over. 81 prefixes today against a 1000 cap.
    if '<IsTruncated>true</IsTruncated>' in body:
        raise RuntimeError('segment listing was truncated; add pagination before '
                           'trusting the winding coverage')
    seen = collections.defaultdict(list)
    for prefix in re.findall(r'<Prefix>([^<]+)</Prefix>', body):
        if not prefix.endswith('/'):
            continue
        name = prefix.split('/')[-2]
        match = re.search(r'-w(\d{3})-(\d{3})$', name)
        if not match:
            continue
        seen[(int(match.group(1)), int(match.group(2)))].append(name)

    # Each winding range is published twice, and only the 20260701… copy carries the
    # `tifxyz_original` mesh this reads. Skipping the other copy explicitly, and
    # saying so, beats letting `segment_points` fail on a 404 halfway through a run.
    out, skipped = [], []
    for (low, high), names in sorted(seen.items()):
        usable = [n for n in names if n.startswith(SEGMENT_COPY)]
        if usable:
            out.append((usable[0], low, high))
        else:
            skipped.append(f'w{low:03d}-{high:03d}')
    if skipped:
        print(f'skipping {len(skipped)} winding range(s) with no {SEGMENT_COPY} copy '
              f'and therefore no mesh in the prediction frame: {", ".join(skipped)}',
              flush=True)
    return out


def segment_points(name, cache_dir):
    """Mesh points of one segment in the prediction's frame, cached on disk."""
    path = os.path.join(cache_dir, f'{name}.npy')
    if os.path.exists(path):
        return np.load(path)
    channels = [tifffile.imread(io.BytesIO(fetch(
        f'{S3}{SEGMENTS}{name}/{CHEAP_MESH}{channel}.tif', timeout=600)))
        for channel in 'xyz']
    points = np.stack(channels, -1).astype(np.float32)
    points = points[~((points[..., 0] == -1) & (points[..., 1] == -1))]
    tmp = f'{path}.{os.getpid()}.part'      # an interrupted save would otherwise
    np.save(tmp, points)                    # poison the cache for every later run
    os.replace(f'{tmp}.npy' if os.path.exists(f'{tmp}.npy') else tmp, path)
    return points


def report(rows):
    """Regression, per-point scoring, and the null control they are read against."""
    truth = np.array([r['truth'] for r in rows], float)
    counted = np.array([r['median_count'] for r in rows], float)
    slope, intercept = np.polyfit(truth, counted, 1)
    passed = SLOPE_GATE[0] <= slope <= SLOPE_GATE[1]
    print(f'\nsegments {len(rows)}  slope {slope:.3f}  intercept {intercept:+.1f}  '
          f'r {np.corrcoef(truth, counted)[0, 1]:.3f}')
    if not passed:
        print(f'  SLOPE GATE FAILED (outside {SLOPE_GATE[0]}-{SLOPE_GATE[1]}): '
              f'counting drifts on this slice; the constant below is not to be '
              f'trusted here.')

    # One flat array of counts, plus the row each count came from — the row index is
    # what lets the null below be a gather instead of 200 rebuilt arrays.
    counts = np.array([c for r in rows for c in r['counts']], float)
    row_of = np.repeat(np.arange(len(rows)), [len(r['counts']) for r in rows])
    row_low = np.array([r['low'] for r in rows], float)
    row_high = np.array([r['high'] for r in rows], float)
    lows, highs = row_low[row_of], row_high[row_of]

    def hit(c, slack=0.0):
        return float(np.mean((counts + c >= lows - slack)
                             & (counts + c <= highs + slack)))

    candidates = np.arange(-25.0, 11.0)
    best = float(candidates[int(np.argmax([hit(c) for c in candidates]))])
    print(f'per-point ({len(counts)} points): best constant C = {best:+.0f}  '
          f'inside {hit(best):.3f}  +-1 {hit(best, 1):.3f}  +-2 {hit(best, 2):.3f}')
    # A second estimate of the same constant from the same data, by a different
    # route: fix the slope at 1 and take the median offset of the segment medians.
    # (Not -intercept/slope, which is the x-intercept — with a slope other than 1 the
    # implied offset varies along the range and that number means nothing here.)
    # Printing both makes a disagreement surface here rather than in review.
    forced = float(-np.median(counted - truth))
    print(f'  C=0 baseline: inside {hit(0.0):.3f}')
    print(f'  second estimator (slope fixed at 1, median offset): {forced:+.1f}  '
          f'against the grid search\'s {best:+.0f}')

    # The claim is limited by these, so the tool has to print them: dispersion grows
    # outward, which is why this calibrates a field and not individual points.
    mid = (lows + highs) / 2.0
    for lo, hi in ((0, 60), (60, 105), (105, 10_000)):
        band = (mid >= lo) & (mid < hi)
        if band.sum() < 5:
            continue
        residual = counts[band] - mid[band]
        print(f'  windings {int(mid[band].min()):3d}-{int(mid[band].max()):3d}: '
              f'n={int(band.sum()):3d}  '
              f'median offset {np.median(residual):+.1f}  '
              f'iqr {np.subtract(*np.percentile(residual, [75, 25])):.1f}')

    # A hit rate means nothing until the same statistic is computed with the labels
    # destroyed. Shuffling which interval belongs to which segment keeps every
    # marginal intact and breaks only the association.
    shifted = counts[None, :] + candidates[:, None]
    rng = np.random.default_rng(0)
    order = np.arange(len(rows))
    null = np.empty(NULL_PERMUTATIONS)
    for trial in range(NULL_PERMUTATIONS):
        rng.shuffle(order)
        lo_s, hi_s = row_low[order][row_of], row_high[order][row_of]
        null[trial] = ((shifted >= lo_s) & (shifted <= hi_s)).mean(axis=1).max()
    print(f'shuffled-label null: mean {null.mean():.3f}  '
          f'p95 {np.percentile(null, 95):.3f}  max {null.max():.3f}')
    return dict(slope=float(slope), gate_passed=bool(passed), constant=best,
                constant_slope1=forced, hit=hit(best), within_one=hit(best, 1.0),
                baseline_c0=hit(0.0), null_p95=float(np.percentile(null, 95)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--umbilicus', help='local override; fetched if omitted')
    parser.add_argument('--cache', required=True, help='chunk cache directory')
    parser.add_argument('--mesh-cache', required=True)
    parser.add_argument('--z', type=float, default=15694.0)
    parser.add_argument('--z-band', type=float, default=3.0)
    parser.add_argument('--per-segment', type=int, default=16)
    parser.add_argument('--threshold', type=int, default=115)
    parser.add_argument('--step', type=float, default=0.5)
    parser.add_argument('--no-mask-filter', action='store_true',
                        help='score mesh points that lie outside the scan mask too. '
                             'This reproduces the runs published on 12.08.2026, which '
                             'is the only reason it exists — see ScanMask.')
    parser.add_argument('--out')
    args = parser.parse_args()

    os.makedirs(args.mesh_cache, exist_ok=True)
    pred = Prediction(args.cache)
    mask = None if args.no_mask_filter else ScanMask(args.cache)
    control = load_umbilicus(args.umbilicus, args.cache)
    ux, uy = umbilicus_at(control, args.z)
    origin = np.array([ux, uy, args.z])
    print(f'z={args.z:.0f} umbilicus=({ux:.0f}, {uy:.0f})'
          f'{"  [mask filter OFF]" if mask is None else ""}', flush=True)

    rows, targets, dropped, offered = [], [], 0, 0
    for name, low, high in labelled_segments():
        points = segment_points(name, args.mesh_cache)
        band = points[np.abs(points[:, 2] - args.z) <= args.z_band]
        if len(band) == 0:
            print(f'{name:34s} w{low:03d}-{high:03d}: no points at this z', flush=True)
            continue
        if mask is not None:
            # Filter the candidates, not the drawn sample: where a segment still has
            # on-mask geometry at this z, this keeps the full `--per-segment` sample
            # instead of thinning it, so segments do not silently lose weight in the
            # regression just because part of their surface left the scan.
            keep = mask.inside(band)
            offered += len(band)
            dropped += int((~keep).sum())
            band = band[keep]
            if len(band) == 0:
                print(f'{name:34s} w{low:03d}-{high:03d}: every mesh point at this z '
                      f'is outside the scan mask; segment skipped', flush=True)
                continue
        # Seeded per segment, not once for the run: adding or removing a segment
        # then leaves every other segment's sample untouched, so results stay
        # comparable across runs that see a different set.
        index = np.random.default_rng(0).choice(
            len(band), min(args.per_segment, len(band)), replace=False)
        for point in band[index]:
            targets.append((len(rows), point.astype(float)))
        rows.append(dict(name=name, low=low, high=high, truth=(low + high) / 2.0,
                         counts=[]))

    # Walk the rays in angular order around the umbilicus. Each count is independent,
    # so the order changes nothing in the result — but neighbouring rays share chunks,
    # and taking them together is what makes a bounded cache work. Measured on the
    # z=18000 slice with everything already on disk: 535 s segment-by-segment, where
    # nothing was downloaded and the whole cost was re-decoding evicted chunks.
    if mask is not None and offered:
        print(f'scan mask: {dropped} of {offered} candidate mesh points at this z '
              f'({dropped / offered:.1%}) lie outside the scanned volume and were not '
              f'scored', flush=True)
    targets.sort(key=lambda item: np.arctan2(item[1][1] - uy, item[1][0] - ux))
    for done, (row_index, point) in enumerate(targets, 1):
        rows[row_index]['counts'].append(
            count_to_point(pred, origin, point, args.threshold, args.step))
        if done % 100 == 0:
            print(f'  {done}/{len(targets)} rays', flush=True)

    for row in rows:
        counts = np.array(row['counts'], float)
        row['median_count'] = float(np.median(counts))
        print(f"{row['name']:34s} w{row['low']:03d}-{row['high']:03d}  counted median "
              f"{row['median_count']:6.1f}  offset "
              f"{row['median_count'] - row['truth']:+.1f}", flush=True)

    if len(rows) < 3:
        print()
        print(f'only {len(rows)} segments have points at z={args.z:.0f}; a regression '
              f'over fewer than 3 says nothing, so nothing is reported.')
        return
    summary = report(rows)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(dict(summary=summary, segments=rows), handle, indent=1,
                      default=float)


if __name__ == '__main__':
    main()
