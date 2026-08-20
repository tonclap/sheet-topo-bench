"""TOPO-014, corpus B, step 1: is the 2023 frame reducible to the 2026 frame?

The GP-banner (`20231231235900_GP.obj`) lives in the 2023 volume's frame
(`20230205180739`, 7.91 um), our detector and the 2026 production meshes in the
2026 scan's frame (`20260411134726`, 2.4 um; the working L2 is that /4). The
volpkg publishes no transform (`transforms/` holds one `.vckeep`), so corpus B
was gated on establishing the frame ourselves.

What dissolves the gate: the S3 bucket ships eleven 2023-era segments with the
*same* mesh reprojected into both frames (`…-on-20230205180739-7.91um.tifxyz`
and `…-on-20260411134726-2.4um.tifxyz`, checked 17.08.2026). The two grids of
one segment are resampled at different pitches, but share the uv
parametrisation, so scaled (row, col) indices give dense point correspondences
— tens of thousands per segment — and the 2023->2026 transform can be *fitted*
rather than reconstructed from scan geometry.

This script measures, per segment and pooled:

- a global affine fit (12 params) 2023 -> 2026 over all eleven segments,
  leave-one-segment-out cross-validated — the honest transferability number;
- the residual decomposed against the local sheet normal of the 2026 grid.
  The tangential component absorbs uv-resampling slack (the two grids need not
  place samples at identical surface points), so **the normal component is the
  figure of merit**: it is what displaces a discrepancy map across the gap;
- a per-segment affine as the non-rigidity gauge: how much a local fit beats
  the global one says whether corpus B needs a piecewise transform.

Declared read of the outcome, before the numbers: the corpus-B discrepancy
threshold will live at the contact scale (~5 vx L2 = 20 vx at 2.4 um). A
cross-validated *normal* residual comfortably under that (median < ~1/3 of
it) means the global affine carries corpus B; between 1/3 and the threshold
means a local (per-region) fit is required; above it the frame does not
reduce affinely and corpus B must either fit locally or be rejected with this
file as the measured reason (both outcomes close the TOPO-011 gate condition).

Needs `imagecodecs` (the reprojected tifs are LZW; the wave-2 mesh tifs were
raw, so the repo never needed it before).

Usage (from oneshot/real_errors/):

    python frame_2023.py --cache ../../output/frame2023 \
        --report ../../output/topo/corpusB_frame.json
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'wave2', 'figures'))
sys.path.insert(0, os.path.join(_HERE, '..'))
import scrolls                                                        # noqa: E402
import net_retry                                    # noqa: E402,F401  (patches awc.fetch)
import absolute_winding_calibration as awc                            # noqa: E402

FRAME_2023 = '20230205180739-7.91um'
FRAME_2026 = '20260411134726-2.4um'
SAMPLE_PER_SEGMENT = 60000
SEED = 20260817


def dual_frame_segments():
    """2023-era S3 segments carrying meshes in both frames, by listing."""
    import re
    body = awc.fetch(f'{awc.S3}?list-type=2&max-keys=1000&delimiter=/&'
                     f'prefix=PHercParis4/segments/', timeout=120).decode()
    names = [s for s in re.findall(
        r'<Prefix>PHercParis4/segments/([^<]+)/</Prefix>', body)
        if s.startswith('2023')]
    out = []
    for seg in names:
        body = awc.fetch(f'{awc.S3}?list-type=2&max-keys=40&delimiter=/&'
                         f'prefix=PHercParis4/segments/{seg}/mesh/',
                         timeout=120).decode()
        dirs = re.findall(
            r'<Prefix>PHercParis4/segments/[^/]+/mesh/([^<]+)/</Prefix>', body)
        if (any(FRAME_2023 in d for d in dirs)
                and any(FRAME_2026 in d for d in dirs)):
            out.append(seg)
    return out


def load_grid(cache, seg, frame):
    os.makedirs(cache, exist_ok=True)
    channels = []
    for c in 'xyz':
        path = os.path.join(cache, f'{seg}-on-{frame}-{c}.tif')
        if not os.path.exists(path):
            url = (f'{awc.S3}PHercParis4/segments/{seg}/mesh/'
                   f'{seg}-on-{frame}.tifxyz/{c}.tif')
            raw = awc.fetch(url, timeout=600)
            if raw is None:
                raise RuntimeError(f'{seg}: no {c}.tif at {url}')
            tmp = f'{path}.{os.getpid()}.part'
            with open(tmp, 'wb') as f:
                f.write(raw)
            os.replace(tmp, path)
        channels.append(tifffile.imread(path))
    return np.stack(channels, -1).astype(np.float32)


def correspondences(a, b, rng):
    """(points_2023, points_2026, normals_2026) via shared-uv bilinear lookup."""
    va = (a[..., 0] != -1) & (a[..., 1] != -1)
    vb = (b[..., 0] != -1) & (b[..., 1] != -1)
    ra, ca = a.shape[:2]
    rb, cb = b.shape[:2]
    rows_a, cols_a = np.where(va)
    if len(rows_a) > SAMPLE_PER_SEGMENT:
        take = rng.choice(len(rows_a), SAMPLE_PER_SEGMENT, replace=False)
        rows_a, cols_a = rows_a[take], cols_a[take]
    fr = rows_a * (rb - 1) / (ra - 1)
    fc = cols_a * (cb - 1) / (ca - 1)
    r0 = np.floor(fr).astype(int)
    c0 = np.floor(fc).astype(int)
    r1 = np.minimum(r0 + 1, rb - 1)
    c1 = np.minimum(c0 + 1, cb - 1)
    ok = vb[r0, c0] & vb[r0, c1] & vb[r1, c0] & vb[r1, c1]
    # normals need one more ring of valid neighbours
    r2 = np.minimum(r0 + 2, rb - 1)
    c2 = np.minimum(c0 + 2, cb - 1)
    ok &= vb[r2, c0] & vb[r0, c2]
    rows_a, cols_a = rows_a[ok], cols_a[ok]
    r0, c0, r1, c1, r2, c2 = r0[ok], c0[ok], r1[ok], c1[ok], r2[ok], c2[ok]
    wr = (fr[ok] - r0)[:, None]
    wc = (fc[ok] - c0)[:, None]
    pb = ((1 - wr) * (1 - wc) * b[r0, c0] + (1 - wr) * wc * b[r0, c1]
          + wr * (1 - wc) * b[r1, c0] + wr * wc * b[r1, c1])
    du = b[r2, c0] - b[r0, c0]
    dv = b[r0, c2] - b[r0, c0]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=1)
    good = norm > 1e-6
    n = n[good] / norm[good][:, None]
    return (a[rows_a, cols_a][good].astype(np.float64),
            pb[good].astype(np.float64), n)


def affine_fit(pa, pb):
    design = np.concatenate([pa, np.ones((len(pa), 1))], 1)
    x, *_ = np.linalg.lstsq(design, pb, rcond=None)
    return x


def residual_stats(pa, pb, normals, transform):
    design = np.concatenate([pa, np.ones((len(pa), 1))], 1)
    resid = design @ transform - pb
    total = np.linalg.norm(resid, axis=1)
    normal = np.abs(np.sum(resid * normals, axis=1))
    def q(v):
        return {'rms': round(float(np.sqrt((v ** 2).mean())), 3),
                'median': round(float(np.median(v)), 3),
                'q90': round(float(np.quantile(v, 0.90)), 3),
                'q99': round(float(np.quantile(v, 0.99)), 3),
                'max': round(float(v.max()), 3)}
    return {'n': len(pa), 'total_vx24': q(total), 'normal_vx24': q(normal)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    segments = dual_frame_segments()
    print(f'{len(segments)} dual-frame 2023-era segments: {segments}',
          flush=True)

    data = {}
    for seg in segments:
        a = load_grid(args.cache, seg, FRAME_2023)
        b = load_grid(args.cache, seg, FRAME_2026)
        pa, pb, n = correspondences(a, b, rng)
        data[seg] = (pa, pb, n)
        print(f'{seg}: grids {a.shape[:2]} -> {b.shape[:2]}, '
              f'{len(pa)} correspondences', flush=True)

    pooled_a = np.concatenate([pa for pa, _, _ in data.values()])
    pooled_b = np.concatenate([pb for _, pb, _ in data.values()])
    pooled_n = np.concatenate([n for _, _, n in data.values()])
    global_x = affine_fit(pooled_a, pooled_b)
    report = {
        'frames': {'from': FRAME_2023, 'to': FRAME_2026},
        'segments': segments,
        'sample_per_segment': SAMPLE_PER_SEGMENT,
        'seed': SEED,
        'note': 'residuals in 2.4um voxels; the working L2 frame is these /4; '
                'the tangential part absorbs uv-resampling slack, the normal '
                'part is the figure of merit for corpus B',
        'global_affine': {
            'matrix_rows_then_translation': [list(map(float, row))
                                             for row in global_x],
            'singular_values': [round(float(s), 5) for s in
                                np.linalg.svd(global_x[:3],
                                              compute_uv=False)],
            'iso_scale_expected': round(7.91 / 2.4, 5),
            'fit': residual_stats(pooled_a, pooled_b, pooled_n, global_x),
        },
        'leave_one_out': {},
        'per_segment_affine': {},
    }
    for seg in segments:
        rest_a = np.concatenate([pa for s, (pa, _, _) in data.items()
                                 if s != seg])
        rest_b = np.concatenate([pb for s, (_, pb, _) in data.items()
                                 if s != seg])
        x = affine_fit(rest_a, rest_b)
        pa, pb, n = data[seg]
        report['leave_one_out'][seg] = residual_stats(pa, pb, n, x)
        own = affine_fit(pa, pb)
        report['per_segment_affine'][seg] = residual_stats(pa, pb, n, own)

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)

    g = report['global_affine']['fit']
    print(f"global affine: normal residual median "
          f"{g['normal_vx24']['median']} vx(2.4um) = "
          f"{g['normal_vx24']['median'] / 4:.2f} vx(L2), "
          f"q99 {g['normal_vx24']['q99']}")
    worst = max(report['leave_one_out'].items(),
                key=lambda kv: kv[1]['normal_vx24']['median'])
    print(f"leave-one-out worst segment {worst[0]}: normal median "
          f"{worst[1]['normal_vx24']['median']} vx(2.4um)")
    local_med = np.median([v['normal_vx24']['median']
                           for v in report['per_segment_affine'].values()])
    print(f"per-segment affine normal median (median over segments): "
          f"{local_med:.3f} vx(2.4um) — the non-rigidity gauge")
    print(f'report at {args.report}')


if __name__ == '__main__':
    main()
