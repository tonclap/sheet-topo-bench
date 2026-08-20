"""Held-out v2 exam driver (TOPO-025): the two declared chains, in order.

Operational glue only — every scientific decision lives in the frozen CLIs
and the PROTOCOL §6 insert / FREEZE_2026-08-19.md; this file just invokes
them with the declared arguments, stops loudly on the first failure, and
appends everything to one log. A rerun after a crash resumes through the
per-segment checkpoints of each stage.

The owner set the exam start at 20:00 MSK 19.08.2026, then moved it to
23:30 MSK the same day (decision of 19.08, ~18:45); the driver refuses to
start earlier (machine clock is MSK) so a stray launch cannot spend the
budget ahead of the declared time.

Usage (from oneshot/detector/):

    python run_exam_h2.py            # full exam, both chains + summary
"""
import datetime
import os
import subprocess
import sys

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

_HERE = os.path.dirname(os.path.abspath(__file__))
TOPO = os.path.join(_HERE, '..', '..', 'output', 'topo')
GRIDS = os.path.join(_HERE, '..', '..', 'output', 'figgrids')
CACHE = os.path.join(_HERE, '..', '..', 'output', 'figcache')
LOG = os.path.join(TOPO, 'exam_h2_run.log')

NOT_BEFORE = datetime.datetime(2026, 8, 19, 23, 30, 0)

CHAINS = (
    {'tag': 'paris4_h2', 'corpus': 'corpus_paris4_h2', 'bands': 'heldout'},
    {'tag': '0139_h2', 'corpus': 'corpus_0139_h2', 'bands': 'all'},
)

DEV_ARGS = [
    '--dev-corpus', os.path.join(TOPO, 'corpus_paris4'),
    '--dev-v2-checkpoints', os.path.join(TOPO, 'ckpt_paris4_dev_v2'),
    '--dev-probe-report', os.path.join(TOPO, 'probe_ct_paris4.json'),
    '--dev-probe-windows', os.path.join(TOPO, 'probe_ct_paris4_windows.jsonl'),
    '--dev-v5-report', os.path.join(TOPO, 'detector_v5_paris4.json'),
]


def run(title, argv):
    line = f'==== {datetime.datetime.now():%H:%M:%S} {title} ===='
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as log:
        log.write(line + '\n')
        log.flush()
        proc = subprocess.Popen([sys.executable] + argv, cwd=_HERE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace')
        for chunk in proc.stdout:
            log.write(chunk)
            log.flush()
            print(chunk, end='', flush=True)
        proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f'{title} FAILED (exit {proc.returncode}) — fix or '
                         f'resume; checkpoints make the rerun cheap')


def main():
    now = datetime.datetime.now()
    if now < NOT_BEFORE and os.environ.get('EXAM_NOW') != '1':
        raise SystemExit(f'exam start declared not before '
                         f'{NOT_BEFORE:%H:%M %d.%m.%Y} (now {now:%H:%M}); '
                         f'set EXAM_NOW=1 only with an explicit owner '
                         f'decision')
    for chain in CHAINS:
        tag, bands = chain['tag'], chain['bands']
        corpus = os.path.join(TOPO, chain['corpus'])
        v1_report = os.path.join(TOPO, f'detector_v1_{tag}.json')
        v2_report = os.path.join(TOPO, f'detector_v2_{tag}.json')
        probe_report = os.path.join(TOPO, f'probe_ct_{tag}.json')
        probe_windows = os.path.join(TOPO, f'probe_ct_{tag}_windows.jsonl')
        run(f'{tag}: detect_v1', [
            'detect_v1.py', '--corpus', corpus, '--grid-cache', GRIDS,
            '--cache', CACHE, '--bands', bands, '--with-prediction',
            '--checkpoint', os.path.join(TOPO, f'ckpt_{tag}'),
            '--report', v1_report])
        run(f'{tag}: detect_v2', [
            'detect_v2.py', '--corpus', corpus, '--grid-cache', GRIDS,
            '--cache', CACHE, '--bands', bands,
            '--checkpoint', os.path.join(TOPO, f'ckpt_{tag}_v2'),
            '--v1-report', v1_report, '--report', v2_report])
        run(f'{tag}: probe_ct', [
            'probe_ct.py', '--corpus', corpus, '--grid-cache', GRIDS,
            '--cache', CACHE, '--v2-report', v2_report, '--bands', bands,
            '--report', probe_report, '--details', probe_windows,
            '--ckpt', os.path.join(TOPO, f'ckpt_ct_{tag}')])
        run(f'{tag}: exam_v5lu', [
            'exam_v5lu.py', *DEV_ARGS, '--corpus', corpus, '--bands', bands,
            '--v2-checkpoints', os.path.join(TOPO, f'ckpt_{tag}_v2'),
            '--probe-report', probe_report, '--probe-windows', probe_windows,
            '--v1-report', v1_report, '--grid-cache', GRIDS,
            '--report', os.path.join(TOPO, f'exam_v5lu_{tag}.json')])
        run(f'{tag}: baselines', [
            'baselines.py', '--corpus', corpus, '--grid-cache', GRIDS,
            '--cache', CACHE, '--bands', bands,
            '--report', os.path.join(TOPO, f'baselines_{tag}.json')])
    run('heldout_summary (v2 section)', [
        'heldout_summary.py', '--topo', TOPO,
        '--out', os.path.join(_HERE, '..', 'HELDOUT_RESULTS.md')])
    print('EXAM COMPLETE', flush=True)


if __name__ == '__main__':
    main()
