"""Every number of README.md, out of the shipped run reports — or exit 1.

Two passes, no network, seconds:

1. **Regenerate the shipped summary artifacts** from `runs/` and compare
   (newline-normalized): HELDOUT_RESULTS.md (including the v5lu exam section
   and its paired deltas) via `heldout_summary.py`, ABLATION_V2/V3/V4.md via
   their `ablation_summary*.py` byte-for-byte, ABLATION_V5/V6/V6S/V7.md
   line-by-line (those files carry dated verdict inserts on top of the
   generated summary, so every regenerated line must occur in the shipped
   file), VERIFY_V5.md/VERIFY_V8.md byte-for-byte via their scripts'
   `render_md` (pure functions of the shipped reports), the three by-eye
   zone tables (ZONES_124.md, ZONES_SUPPORT_B.md, ZONES_CENSUS_B.md) via
   `summarize_zones*.py` byte-for-byte, and the two
   bootstrap-CI JSONs via `real_ci.py`. One adaptation,
   documented here
   because it is the one thing that is not byte-identical to the original
   environment: the detector reports record the absolute corpus path of the
   machine that ran them, so `load_injection_info` is remapped to the shipped
   manifests by directory basename. The reports themselves are untouched.

2. **Bind the README numbers**: each check builds its expected literal string
   from the run report the README cites, and asserts the literal occurs in
   README.md. A number edited by hand without its source, or a source swapped
   without the text, fails the run — the wave-2 lesson (three published rows
   from a non-shipped script) applied by construction.

Usage:
    python verify.py
"""
import csv
import importlib
import io
import json
import os
import pathlib
import sys
import tempfile
from contextlib import contextmanager, nullcontext, redirect_stdout

for _stream in (sys.stdout, sys.stderr):      # cp1251 consoles vs Δ in labels
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs', 'topo')
sys.path.insert(0, os.path.join(HERE, 'oneshot', 'detector'))
sys.path.insert(0, os.path.join(HERE, 'oneshot', 'real_errors'))

FAILURES = []


def fail(label, detail):
    FAILURES.append(f'{label}: {detail}')
    print(f'FAIL {label}: {detail}')


def ok(label):
    print(f'  ok {label}')


def load(name):
    with open(os.path.join(RUNS, name), encoding='utf-8') as f:
        return json.load(f)


def normalized(path):
    with open(path, encoding='utf-8') as f:
        return f.read().replace('\r\n', '\n')


def remap_corpus(report):
    """The shipped manifest for the corpus a report recorded by machine path."""
    base = os.path.basename(str(report.get('corpus', '')).rstrip('/\\'))
    return dict(report, corpus=os.path.join(RUNS, base))


@contextmanager
def remapped_manifests(module):
    """`load_injection_info` reading the shipped manifests, for the call only.

    The detector reports record the absolute corpus path of the machine that
    ran them; `remap_corpus` points the reader at the shipped copy instead.
    Restored on the way out: left in place, the wrappers stack one per call
    and the next reader inherits a patch it never asked for.
    """
    import heldout_summary
    original = heldout_summary.load_injection_info

    def remapped(report):
        return original(remap_corpus(report))

    # ablation summaries bind the name at import: patch both namespaces.
    patched = [(heldout_summary, original)]
    if hasattr(module, 'load_injection_info'):
        patched.append((module, module.load_injection_info))
    try:
        for namespace, _ in patched:
            namespace.load_injection_info = remapped
        yield
    finally:
        for namespace, previous in patched:
            namespace.load_injection_info = previous


def run_module(module_name, argv, patch_injection_info=False):
    module = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [module_name] + argv
    try:
        with (remapped_manifests(module) if patch_injection_info
              else nullcontext()):
            with redirect_stdout(io.StringIO()):
                module.main()
    finally:
        sys.argv = old_argv


def compare_regen(label, generated, shipped):
    if normalized(generated) == normalized(shipped):
        ok(label)
    else:
        fail(label, f'{os.path.relpath(shipped, HERE)} differs from its '
                    f'regeneration out of runs/')


def compare_lines(label, generated, shipped):
    """Shipped ablation files carry dated verdict inserts appended to the
    generated summary; every regenerated line (whitespace-collapsed) must
    still occur verbatim in the shipped file."""
    gen = [' '.join(line.split())
           for line in normalized(generated).splitlines()]
    ship = {' '.join(line.split())
            for line in normalized(shipped).splitlines()}
    missing = [line for line in gen if line and line not in ship]
    if missing:
        fail(label, f'{len(missing)} regenerated line(s) missing from '
                    f'{os.path.relpath(shipped, HERE)}, first: {missing[0]!r}')
    else:
        ok(label)


def regenerate(tmp):
    print('pass 1: regenerate shipped summaries from runs/')
    out = os.path.join(tmp, 'HELDOUT_RESULTS.md')
    run_module('heldout_summary', ['--topo', RUNS, '--out', out],
               patch_injection_info=True)
    compare_regen('HELDOUT_RESULTS.md', out,
                  os.path.join(HERE, 'oneshot', 'HELDOUT_RESULTS.md'))

    for module, report, shipped in (
            ('ablation_summary', 'detector_v2_paris4.json', 'ABLATION_V2.md'),
            ('ablation_summary_v3', 'detector_v3_paris4.json',
             'ABLATION_V3.md'),
            ('ablation_summary_v4', 'detector_v4_paris4.json',
             'ABLATION_V4.md')):
        out = os.path.join(tmp, shipped)
        run_module(module, ['--report', os.path.join(RUNS, report),
                            '--out', out], patch_injection_info=True)
        compare_regen(shipped, out,
                      os.path.join(HERE, 'oneshot', 'detector', shipped))

    for module, report, shipped in (
            ('ablation_summary_v5', 'detector_v5_paris4.json',
             'ABLATION_V5.md'),
            ('ablation_summary_v6', 'detector_v6_paris4.json',
             'ABLATION_V6.md'),
            ('ablation_summary_v6s', 'detector_v6s_paris4.json',
             'ABLATION_V6S.md'),
            ('ablation_summary_v7', 'detector_v7_paris4.json',
             'ABLATION_V7.md')):
        out = os.path.join(tmp, shipped)
        run_module(module, ['--report', os.path.join(RUNS, report),
                            '--out', out], patch_injection_info=True)
        compare_lines(shipped, out,
                      os.path.join(HERE, 'oneshot', 'detector', shipped))

    # VERIFY_V5.md / VERIFY_V8.md: rendered by pure functions of their shipped
    # reports, so they regenerate and byte-compare like the rest.
    for module_name, report_name, shipped in (
            ('verify_v5', 'verify_v5_report.json', 'VERIFY_V5.md'),
            ('verify_v8', 'verify_v8_report.json', 'VERIFY_V8.md')):
        module = importlib.import_module(module_name)
        out = os.path.join(tmp, shipped)
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(module.render_md(load(report_name))))
        compare_regen(shipped, out,
                      os.path.join(HERE, 'oneshot', 'detector', shipped))

    # The three by-eye zone tables: the label CSVs and the zones_meta.json
    # provenance sidecars are all shipped, so the tables regenerate and
    # byte-compare like the rest.
    for module, labels_name, meta_dir, shipped in (
            ('summarize_zones', 'zone_labels.csv', 'zones_png',
             'ZONES_124.md'),
            ('summarize_zones_b', 'zone_labels_b.csv', 'zones_png_b',
             'ZONES_SUPPORT_B.md'),
            ('summarize_zones_b', 'zone_labels_census_b.csv',
             'zones_png_census_b', 'ZONES_CENSUS_B.md')):
        out = os.path.join(tmp, shipped)
        argv = ['--labels', os.path.join(HERE, 'oneshot', 'real_errors',
                                         labels_name),
                '--meta', os.path.join(RUNS, 'real_paris4', meta_dir,
                                       'zones_meta.json'),
                '--out', out]
        if module == 'summarize_zones_b':
            argv += ['--n-zones', '178']
        run_module(module, argv)
        compare_regen(shipped, out,
                      os.path.join(HERE, 'oneshot', 'real_errors', shipped))

    for eval_name, ci_name in (
            (os.path.join('real_paris4', 'eval.json'),
             os.path.join('real_paris4', 'eval_ci.json')),
            (os.path.join('real_paris4', 'eval_corpusB.json'),
             os.path.join('real_paris4', 'eval_corpusB_ci.json'))):
        out = os.path.join(tmp, os.path.basename(ci_name))
        run_module('real_ci', ['--eval', os.path.join(RUNS, eval_name),
                               '--out', out])
        with open(out, encoding='utf-8') as f:
            regen = json.load(f)
        shipped_ci = load(ci_name)
        regen.pop('source', None)
        stripped = dict(shipped_ci)
        stripped.pop('source', None)
        if regen == stripped:
            ok(ci_name)
        else:
            fail(ci_name, 'CI JSON differs from regeneration')


def headline_rows():
    """The three rows of the README headline table, from the reports."""
    import heldout_summary as hs
    rows = []
    for label, det in (
            ('dev (Paris 4, bands < 100)', 'detector_v1_paris4.json'),
            ('held-out A (Paris 4, bands ≥ 100)',
             'detector_v1_paris4_heldout.json'),
            ('held-out B (PHerc0139, whole scroll)', 'detector_v1_0139.json')):
        report = load(det)
        ci = hs.bootstrap_ci(report)
        m = report['metrics']
        share = m['delta_erl_mm'] / report['erl_context']['delta_erl_oracle_mm']
        rows.append(
            f"| {label} | {report['scoped_injections']} | "
            f"{m['ap']:.4f} [{ci['ap'][0]:.3f}–{ci['ap'][1]:.3f}] | "
            f"{m['recall_at_n']:.3f} | "
            f"{m['delta_erl_mm']:+.2f} mm ({share:.0%}) |")
    return rows


def _strip_mark(cell):
    return cell.replace(' **significant**', '')


def exam_checks():
    """The v5lu exam table (held-out v2) and its paired deltas."""
    import heldout_summary as hs
    checks = []
    b2_cells = None
    for label, v1_name, v5_name in (
            ('held-out A2 (Paris 4 ≥ 100, seed 20260819)',
             'detector_v1_paris4_h2.json', 'exam_v5lu_paris4_h2.json'),
            ('held-out B2 (PHerc0139, seed 20260819)',
             'detector_v1_0139_h2.json', 'exam_v5lu_0139_h2.json')):
        v1, v5 = load(v1_name), load(v5_name)
        ci1, ci5 = hs.bootstrap_ci(v1), hs.bootstrap_ci(v5)
        info = hs.load_injection_info(remap_corpus(v5))
        cells, _ = hs.paired_delta_lines(v5, info)
        if 'B2' in label:
            b2_cells = cells
        checks.append(
            ('exam row',
             f"| {label} | {v5['scoped_injections']} | "
             f"{v1['metrics']['ap']:.4f} "
             f"[{ci1['ap'][0]:.3f}–{ci1['ap'][1]:.3f}] | "
             f"{v5['metrics']['ap']:.4f} "
             f"[{ci5['ap'][0]:.3f}–{ci5['ap'][1]:.3f}] | "
             f"**{_strip_mark(cells[0])}** |"))
    checks.append(('exam B2 delta-M', f"ΔM {_strip_mark(b2_cells[3])}"))
    return checks


def v5_dev_checks():
    """The learned-fusion dev point and its paired deltas vs v1."""
    import heldout_summary as hs
    rep = load('detector_v5_paris4.json')
    info = hs.load_injection_info(remap_corpus(rep))
    ci = hs.bootstrap_ci(rep, info)
    cells, _ = hs.paired_delta_lines(rep, info)
    point = [_strip_mark(c).split(' [')[0] for c in cells]
    v8 = load('verify_v8_report.json')
    d = v8['paired_deltas']['ap']
    cb = load('coverage_breakdown_paris4.json')
    n = cb['scoped_injections']
    unc = cb['uncovered_union']['summary']
    return [
        ('v5lu dev AP',
         f"dev AP {rep['metrics']['ap']:.4f} "
         f"[{ci['ap'][0]:.3f}–{ci['ap'][1]:.3f}]"),
        ('v5lu dev deltas',
         f"ΔAP {_strip_mark(cells[0])}, ΔM {point[3]}, ΔH {point[4]}, "
         f"Δplausible {point[5]}"),
        ('v5lu fresh seed',
         f"v1 {v8['v1_ap']:.4f} → v5l {v8['v5l_ap']:.4f}, "
         f"ΔAP {d[0]:+.4f} [{d[1]:+.4f}–{d[2]:+.4f}]"),
        ('coverage ceiling',
         f"covers {cb['coverage']['v1']}/{n} injections, the union pool "
         f"{cb['coverage']['union']}/{n} = {cb['coverage']['union'] / n:.3f}"),
        ('coverage misses',
         f"Of the {unc['n']} dev injections with no candidate anywhere, "
         f"{unc['by_type']['M']} are mergers ({unc['plausible']} locally "
         f"plausible)"),
    ]


def v5_verify_checks():
    """The eight declared pre-exam checks of VERIFY_V5.

    These five numbers are the README's argument that v5lu is not an overfit.
    """
    r = load('verify_v5_report.json')
    perm = r['V1_permutation']['aps']['p95']
    t = r['V6_transfer_0139']
    shifts = [abs(row['shift']) for row in r['V4_lambda']['rows'].values()]
    if max(shifts) > 0.002:
        fail('v5 lambda bound',
             f'README claims shifts ≤ 0.002, report has {max(shifts):.4f}')
    return [
        ('v5 permutation p95', f"p95 {perm:.3f} < 0.82"),
        ('v5 block split', f"a hard block split ({r['V3_block_split']['ap']:.4f})"),
        ('v5 intercepts only',
         "per-family intercepts alone give "
         f"{r['V5_feature_ablation']['intercepts_only']['ap']:.4f}"),
        ('v5 cross-scroll transfer',
         "transfer to the other scroll "
         f"({t['transfer_ap']:.4f} vs {t['v1_ap']:.4f})"),
        ('v5 raw-mass attribution',
         "sorting by raw mass with no learning at all gives "
         f"{r['V2_label_free']['global_mass']['ap']:.4f}"),
    ]


def transfer_checks():
    """The transfer-composition numbers of the README's exam paragraph.

    They come from `transfer_breakdown.py`, which reads the two exam reports
    and nothing else. Its `totals()` is the single definition of these counts
    — the same one that prints them into ABLATION_V5.md — so a changed report
    moves the literal here and fails the run, and the two cannot drift apart.
    """
    import transfer_breakdown as tb
    tb.OUT = pathlib.Path(RUNS)

    def stats(v1_name, v5_name):
        counts = tb.totals(tb.load(v1_name), tb.load(v5_name))
        h_med = counts['by_type']['H']['medians']
        return {'misses': counts['misses'],
                'no_candidate': tuple(len(x) for x in counts['no_candidate']),
                'H_below': tuple(len(x)
                                 for x in counts['by_type']['H']['below_n']),
                'H_med': tuple(tb.fmt(m) for m in h_med)}

    a2 = stats('detector_v1_paris4_h2.json', 'exam_v5lu_paris4_h2.json')
    b2 = stats('detector_v1_0139_h2.json', 'exam_v5lu_0139_h2.json')
    gained = b2['no_candidate'][0] - b2['no_candidate'][1]
    return [
        ('transfer A2 H median',
         f"median rank of found H drops {a2['H_med'][0]} → {a2['H_med'][1]}"),
        ('transfer B2 misses',
         f"misses {b2['misses'][0]} → {b2['misses'][1]}"),
        ('transfer B2 new coverage',
         f"credits {gained} injections v1's pool never generates"),
        ('transfer B2 H below-N',
         f"H below-N {b2['H_below'][0]} → {b2['H_below'][1]}"),
        ('transfer B2 H rank',
         f"median H rank {b2['H_med'][0]} → {b2['H_med'][1]}"),
    ]


def v2_baseline_checks():
    a2 = load('baselines_paris4_h2.json')
    b2 = load('baselines_0139_h2.json')
    raw = max(rep[key]['metrics']['ap'] for rep in (a2, b2)
              for key in ('selfx_raw', 'rayinv_raw'))
    a2_atlas = max(a2[key]['metrics']['ap']
                   for key in ('selfx_atlas', 'rayinv_atlas'))
    b2_atlas = max(b2[key]['metrics']['ap']
                   for key in ('selfx_atlas', 'rayinv_atlas'))
    return [
        ('v2 baselines raw', f"raw baselines at or below AP {raw:.4f}"),
        ('v2 baselines atlas',
         f"reaches {a2_atlas:.4f} on A2 and {b2_atlas:.4f} on B2"),
    ]


def real_channel_checks():
    """The thick/defect/fusion rows and the fusion boundary numbers."""
    def primary(name, key):
        return load(os.path.join('real_paris4', name))['rows'][key]['primary']

    def row(title, r):
        base = r['baseline_random']['ap']
        return (f"| B — same zones | {r['n_zones']} | **{title}** | "
                f"**{r['metrics']['ap']:.4f} [{r['ci']['ap_ci95'][0]:.4f}–"
                f"{r['ci']['ap_ci95'][1]:.4f}]** | {base['median']:.4f} "
                f"[IQR {base['iqr'][0]:.4f}–{base['iqr'][1]:.4f}] | "
                f"**signal** |")

    thick = primary('eval_thicknessB.json', 'thick')
    defect = primary('eval_defectB.json', 'defect')
    fusion3 = primary('eval_fusion3B.json', 'fusion3')
    pair = load(os.path.join('real_paris4', 'eval_fusionB.json'))['rows']
    label_free = sorted(pair[k]['primary']['metrics']['ap']
                        for k in ('frozen', 'rawmass', 'zscore'))
    fa = load(os.path.join('real_paris4', 'fusion_asymmetry.json'))
    curve = fa['fractions']
    return [
        ('real corpora thick', row('thick', thick)),
        ('real corpora defect', row('defect', defect)),
        ('fusion pair range',
         f"support+thick ({label_free[0]:.4f}–{label_free[-1]:.4f})"),
        ('fusion3 level', f"fusion3 ({fusion3['metrics']['ap']:.4f})"),
        ('fusion loso',
         f"significantly *worse* than either solo "
         f"({pair['loso']['primary']['metrics']['ap']:.4f})"),
        ('fusion dilution pools',
         f"{fa['pools']['prox'] / fa['pools']['support']:.1f}× larger"),
        ('fusion dilution curve',
         f"{curve['1.0']['ap_median']:.4f} → {curve['0.05']['ap_median']:.4f} "
         f"at 5% of the pool"),
    ]


def _wilson_upper(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre + half) / denom


def _fisher_two_sided(a, b, c, d):
    """Exact two-sided Fisher for [[a, b], [c, d]] by summing all tables
    with probability ≤ the observed one (no scipy in requirements)."""
    from math import comb
    row1, col1, total = a + b, a + c, a + b + c + d

    def pmf(k):
        return (comb(col1, k) * comb(total - col1, row1 - k)
                / comb(total, row1))
    observed = pmf(a)
    return sum(pmf(k) for k in range(0, min(row1, col1) + 1)
               if pmf(k) <= observed * (1 + 1e-12))


def census_checks():
    with open(os.path.join(HERE, 'oneshot', 'real_errors',
                           'zone_labels_census_b.csv'),
              encoding='utf-8-sig') as f:
        census = [r['class'] for r in csv.DictReader(f)]
    with open(os.path.join(HERE, 'oneshot', 'real_errors',
                           'zone_labels_b.csv'), encoding='utf-8-sig') as f:
        credited = [r['class'] for r in csv.DictReader(f)]
    c_real, n_census = census.count('real_error'), len(census)
    b_real, n_credited = credited.count('real_error'), len(credited)
    upper = _wilson_upper(c_real, n_census)
    p = _fisher_two_sided(c_real, n_census - c_real,
                          b_real, n_credited - b_real)
    return [
        ('census yield',
         f"{c_real}/{n_census} real-error candidates [Wilson 95% "
         f"{c_real / n_census * 100:.1f}–{upper * 100:.1f}%] against "
         f"{b_real}/{n_credited} ({b_real / n_credited * 100:.1f}%)"),
        ('census fisher', f"p = {p:.4f}"),
    ]


def address_checks():
    with open(os.path.join(RUNS, 'real_paris4', 'zones_band_defect',
                           'band_labels_defect.csv'),
              encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    high = sorted(r['zone'] for r in rows
                  if r['control_class'] == 'crossing'
                  and r['control_confidence'] == 'high')
    low = sorted(r['zone'] for r in rows
                 if r['control_class'] == 'crossing'
                 and r['control_confidence'] == 'low')

    rank_keys = {'supportB_zones.json': 'support_rank',
                 'thickB_zones.json': 'thick_rank',
                 'defectB_zones.json': 'defect_rank'}

    def rank_of(name, zone_id):
        for z in load(os.path.join('real_paris4', name))['zones']:
            if z['id'] == zone_id:
                return z[rank_keys[name]]
        raise KeyError(f'{zone_id} absent from {name}')

    support = rank_of('supportB_zones.json', 'Z0136')
    thick = rank_of('thickB_zones.json', 'Z0136')
    defect = rank_of('defectB_zones.json', 'Z0136')
    return [
        ('address candidates high',
         f"confirmed at high confidence both passes ({', '.join(high)})"),
        ('address candidates low',
         f"with reduced control confidence ({', '.join(low)})"),
        ('Z0136 support rank', f"the original support credit (rank {support})"),
        ('Z0136 thick rank', f"a thick credit (rank {thick})"),
        ('Z0136 defect rank', f"defect channel ({defect})"),
    ]


def real_corpora_rows():
    a_ci = load(os.path.join('real_paris4', 'eval_ci.json'))
    a2 = load(os.path.join('real_paris4', 'eval_bothsilent.json'))
    b_ci = load(os.path.join('real_paris4', 'eval_corpusB_ci.json'))
    sb = load(os.path.join('real_paris4', 'eval_supportB.json'))
    sup = sb['rows']['support']['primary']
    fusion = sb['rows']['prox_support']['primary']

    def iqr(pair):
        return f'{pair[0]:.4f}–{pair[1]:.4f}'

    a2p = a2['real']['primary']
    # The "× over chance" cells are ratios of two shipped numbers, not prose:
    # computed here so a moved report moves the multiplier too. Read them with
    # the pool-asymmetry limit the README states next to the table.
    a_ratio = a_ci['ap'] / a_ci['baseline_random_ap_median']
    sup_ratio = (sup['metrics']['ap']
                 / sup['baseline_random']['ap']['median'])
    return [
        (f"| A — recto/m7 disagreement (mass ≥ 40) | {a_ci['n_zones']} | prox "
         f"| {a_ci['ap']:.4f} [{a_ci['ap_ci95'][0]:.4f}–"
         f"{a_ci['ap_ci95'][1]:.4f}] | "
         f"{a_ci['baseline_random_ap_median']:.4f} "
         f"[{iqr(a_ci['baseline_random_ap_iqr'])}] | "
         f"~{a_ratio:.1f}× over chance |"),
        (f"| A2 — both-silent consensus (mass ≥ 40) | — | prox | "
         f"{a2p['metrics']['ap']:.4f} | "
         f"{a2p['baseline_random']['ap']['median']:.4f} | weaker than A |"),
        (f"| B — human-verified GP-banner reference (mass ≥ 20) | "
         f"{b_ci['n_zones']} | prox | {b_ci['ap']:.4f} "
         f"[{b_ci['ap_ci95'][0]:.4f}–{b_ci['ap_ci95'][1]:.4f}] | "
         f"{b_ci['baseline_random_ap_median']:.4f} "
         f"[IQR {iqr(b_ci['baseline_random_ap_iqr'])}] | at chance |"),
        (f"| B — same zones | {sup['n_zones']} | **support** | "
         f"**{sup['metrics']['ap']:.4f} [{sup['ci']['ap_ci95'][0]:.4f}–"
         f"{sup['ci']['ap_ci95'][1]:.4f}]** | "
         f"{sup['baseline_random']['ap']['median']:.4f} "
         f"[IQR {iqr(sup['baseline_random']['ap']['iqr'])}] | "
         f"**~{sup_ratio:.0f}× over chance — signal** |"),
        (f"AP {sb['rows']['support']['half']['metrics']['ap']:.4f} at "
         f"mass ≥ 10, {sb['rows']['support']['double']['metrics']['ap']:.4f} "
         f"at ≥ 40"),
        (f"prox+support on B: AP {fusion['metrics']['ap']:.4f}, back at "
         f"chance"),
    ]


def literal_checks():
    checks = []
    for row in headline_rows():
        checks.append(('headline row', row))
    checks += exam_checks()
    checks += v5_dev_checks()
    checks += v5_verify_checks()
    checks += transfer_checks()
    checks += v2_baseline_checks()
    for row in real_corpora_rows():
        checks.append(('real corpora', row))
    checks += real_channel_checks()
    checks += census_checks()
    checks += address_checks()

    dev_b = load('baselines_paris4.json')
    ha_b = load('baselines_paris4_heldout.json')
    checks += [
        ('baseline raw ceiling',
         f"at or below AP {dev_b['rayinv_raw']['metrics']['ap']:.4f} raw"),
        ('baseline atlas dev',
         f"reaches {dev_b['selfx_atlas']['metrics']['ap']:.4f} on dev"),
        ('baseline atlas held-out',
         f"{ha_b['selfx_atlas']['metrics']['ap']:.4f} on held-out A"),
    ]

    v1 = load('detector_v1_paris4.json')
    sub = v1['substrate']
    checks += [
        ('atlas contact share',
         f"{sub['contact_share'] * 100:.2f}% of skeleton points in contact"),
        ('atlas contact clusters',
         f"{sub['masked_prox_clusters']:,} natural contact clusters".replace(
             ',', ' ')),
        ('self-intersection cells',
         f"{dev_b['cells']['selfx']['atlas_cells']:,} self-intersection "
         f"cells".replace(',', ' ')),
        ('ray-inversion cells',
         f"{dev_b['cells']['rayinv']['atlas_cells']:,} ray-inversion "
         f"cells".replace(',', ' ')),
    ]

    ct = load('probe_ct_paris4.json')
    missed = ct['coverage']['M_missed_by_v1']
    holes = ct['coverage']['H_all']
    glob = load('probe_global_paris4.json')
    gl = sorted(glob['coverage'][ch]['missed_by_v1']['covered']
                for ch in ('selfx', 'rayinv'))
    checks += [
        ('ct probe merger coverage',
         f"CT clusters cover {missed['covered']}/{missed['n']} v1-missed "
         f"merger windows, {holes['covered']}/{holes['n']} hole windows"),
        ('global probe coverage',
         f"probe: {gl[0]}–{gl[1]} of "
         f"{glob['coverage']['selfx']['missed_by_v1']['n']} missed"),
    ]

    cor = load('corrector_v1_paris4.json')
    noscar = load('corrector_v1_paris4_noscar.json')
    checks += [
        ('corrector random floor',
         f"{cor['random']['116']['median']:.0f} mm ERL at k=N/2, "
         f"{cor['random']['928']['median']:.0f} mm at k=4N"),
        ('corrector scar best',
         f"{cor['summary']['best_delta_erl_mm']:.1f} mm with a seam scar"),
        ('corrector noscar best',
         f"{noscar['summary']['best_delta_erl_mm']:+.1f} mm with ideal "
         f"reseaming"),
        ('corrector oracle',
         f"{noscar['summary']['oracle_share']:.0%} of the corrector oracle's "
         f"+{noscar['oracle']['delta_erl_mm']:.1f} mm"),
    ]

    epoch = load(os.path.join('real_paris4', 'corpusB.json'))['epoch_table']
    checks += [
        ('epoch 2026', f"diverge from the 2023 human-verified reference on "
                       f"{epoch['2026']['rate'] * 100:.1f}% of covered nodes"),
        ('epoch 2023', f"on {epoch['2023']['rate'] * 100:.1f}%"),
    ]

    with open(os.path.join(HERE, 'oneshot', 'real_errors', 'zone_labels.csv'),
              encoding='utf-8-sig') as f:
        labels = [row['class'] for row in csv.DictReader(f)]
    n = len(labels)

    def zshare(kind):
        return labels.count(kind), labels.count(kind) / n * 100
    real, real_pc = zshare('real_error')
    art, art_pc = zshare('model_artifact')
    fa, fa_pc = zshare('false_alarm')
    checks += [
        ('zones classified', "classified by hand, all of them"),
        ('zones real', f"{real} real error ({real_pc:.1f}%"),
        ('zones artifacts', f"{art} model artifacts ({art_pc:.1f}%)"),
        ('zones false alarms', f"{fa} false alarms ({fa_pc:.1f}%)"),
    ]

    atlas = load('atlas_sensitivity.json')
    checks.append(('atlas robust core',
                   f"threshold-stable core "
                   f"({atlas['robust_core_of_ref_topk']} of the top-"
                   f"{atlas['top_k']} clusters)"))

    # ------------------------- corpus-B zones by eye + the verification chain
    with open(os.path.join(HERE, 'oneshot', 'real_errors',
                           'zone_labels_b.csv'), encoding='utf-8-sig') as f:
        blabels = [row['class'] for row in csv.DictReader(f)]
    nb = len(blabels)

    def brow(kind, title):
        k = blabels.count(kind)
        return f"| {title} | {k} | {k / nb * 100:.1f}% |"
    checks += [
        ('zones B real candidates',
         brow('real_error', 'real_error candidates (single-slab reading)')),
        ('zones B banner_edge', brow('banner_edge', 'banner_edge')),
        ('zones B epoch_drift', brow('epoch_drift', 'epoch_drift')),
        ('zones B false_alarm', brow('false_alarm', 'false_alarm')),
    ]

    with open(os.path.join(RUNS, 'real_paris4', 'zones_png_b_confirm',
                           'confirm_labels_b.csv'),
              encoding='utf-8-sig') as f:
        crows = list(csv.DictReader(f, delimiter=';'))
    indep_real = [r for r in crows if r['independent'] == '1'
                  and r['control_class'] == 'real_error']
    # Bound unconditionally: if a slab ever does confirm, the literal changes
    # and the run fails, instead of the check quietly dropping out of the list
    # while the README keeps claiming the negative.
    checks.append(('confirm slabs negative',
                   'none of the four confirmed' if not indep_real
                   else 'independent slabs confirmed: '
                        + ', '.join(sorted(r['zone'] for r in indep_real))))
    with open(os.path.join(RUNS, 'real_paris4', 'zones_band_b',
                           'band_labels_b.csv'),
              encoding='utf-8-sig') as f:
        brows = list(csv.DictReader(f, delimiter=';'))
    crossing = sorted(r['zone'] for r in brows
                      if r['control_class'] == 'crossing')
    on_sheet = [r for r in brows if r['control_class'] == 'on_sheet']
    checks += [
        ('band artifacts',
         f"{ {3: 'three'}.get(len(on_sheet), len(on_sheet))} of the four "
         f"were slab-projection artifacts"),
        ('band confirmed zone',
         f"one — {crossing[0]} — is a confirmed real 2026 tracing error"
         if len(crossing) == 1 else f"crossing zones: {crossing}"),
        ('verified yield',
         f"{len(crossing)}/{nb}, not {blabels.count('real_error')}/{nb}"),
    ]

    dens = load(os.path.join('real_paris4', 'densityB.json'))
    seg_shares = sorted(v['share'] for v in dens['per_segment'].values())
    sens = sorted(v['share'] for v in dens['sensitivity'].values())
    checks += [
        ('density gate share', f"failed at **{dens['share'] * 100:.1f}%**"),
        ('density gate sensitivity',
         f"({sens[0] * 100:.1f}–{sens[1] * 100:.1f}% under T_cover × 0.75/"
         f"× 1.25; per segment {seg_shares[0] * 100:.1f}–"
         f"{seg_shares[-1] * 100:.1f}%)"),
    ]

    v1m = load(os.path.join('corpus_paris4', 'manifest.json'))
    v2m = load(os.path.join('corpus_paris4_v2', 'manifest.json'))
    checks.append(
        ('plausibility measure v2',
         f"at **{v2m['plausible_share_smoothed']:.2f} vs the declared "
         f"{v1m['plausible_share']:.2f}**"))
    return checks


def bind_readme():
    print('pass 2: bind README numbers to run reports')
    # Whitespace-collapsed on both sides: a literal must survive the README's
    # line wrapping, and the wrapping must not be able to hide a mismatch.
    readme = ' '.join(normalized(os.path.join(HERE, 'README.md')).split())
    for label, literal in literal_checks():
        literal = ' '.join(literal.split())
        if literal in readme:
            ok(f'{label}')
        else:
            fail(label, f'expected literal not in README: {literal!r}')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        regenerate(tmp)
    bind_readme()
    if FAILURES:
        print(f'\n{len(FAILURES)} failure(s)')
        raise SystemExit(1)
    print('\nall numbers verified against the shipped runs')


if __name__ == '__main__':
    main()
