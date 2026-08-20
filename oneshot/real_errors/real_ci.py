"""Bootstrap CIs for the real-corpus primary numbers (TARGET item 5).

The published claim of the real corpus is a pair of point numbers: prox-only
AP 0.0081 on 815 disagreement zones against a random-window median 0.0015
(`eval.json`, variant `primary`). Item 5 of the completion criterion demands a
measure of uncertainty on every published number; the random baseline already
carries an IQR over seeds, the detector's own AP and recall@N carry nothing.

Same machinery as heldout_summary.bootstrap_ci: the resampling unit is the
ground-truth item (here the disagreement zone), AP decomposes into one
contribution per credited zone (precision at its credited rank), the bootstrap
resamples the recorded contributions — nothing is re-run. The decomposition is
validated against the report's own AP before any interval is printed: if the
reconstruction does not match, the script refuses.

Usage:
    python oneshot/real_errors/real_ci.py \
        --eval output/topo/real_paris4/eval.json \
        --out output/topo/real_paris4/eval_ci.json
"""

import argparse
import json

import numpy as np

import sys
# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260815


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.eval, encoding='utf-8') as f:
        report = json.load(f)
    primary = report['real']['primary']
    zones = primary['candidates']
    n = primary['n_zones']
    assert len(zones) == n, (len(zones), n)

    ranks = sorted(z['detector_rank'] for z in zones
                   if z['detector_rank'] is not None)
    tp_at = {r: i for i, r in enumerate(ranks, 1)}
    contribution = np.array([
        tp_at[z['detector_rank']] / z['detector_rank']
        if z['detector_rank'] is not None else 0.0 for z in zones])
    recall_terms = np.array([
        1.0 if z['detector_rank'] is not None and z['detector_rank'] <= n
        else 0.0 for z in zones])

    ap_point = float(contribution.mean())
    recall_point = float(recall_terms.mean())
    if abs(ap_point - primary['metrics']['ap']) > 1e-9:
        raise SystemExit(
            f"AP reconstruction mismatch: {ap_point} vs "
            f"{primary['metrics']['ap']} — decomposition invalid, no CI")
    if abs(recall_point - primary['metrics']['recall_at_n']) > 1e-9:
        raise SystemExit(
            f"recall reconstruction mismatch: {recall_point} vs "
            f"{primary['metrics']['recall_at_n']}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    aps = contribution[samples].mean(axis=1)
    recalls = recall_terms[samples].mean(axis=1)
    out = {
        'source': args.eval,
        'resamples': BOOTSTRAP_RESAMPLES,
        'seed': BOOTSTRAP_SEED,
        'n_zones': n,
        'ap': primary['metrics']['ap'],
        'ap_ci95': [float(np.percentile(aps, 2.5)),
                    float(np.percentile(aps, 97.5))],
        'recall_at_n': primary['metrics']['recall_at_n'],
        'recall_ci95': [float(np.percentile(recalls, 2.5)),
                        float(np.percentile(recalls, 97.5))],
        'baseline_random_ap_median': primary['baseline_random']['ap']['median'],
        'baseline_random_ap_iqr': primary['baseline_random']['ap']['iqr'],
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
