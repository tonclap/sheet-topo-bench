"""TOPO-032: the census statistics, exactly as declared in ZONE_CRITERIA_B.md.

The insert of 17.08.2026 (nineteenth session) declared, before any census
card was rendered: the real_error share among the uncredited zones with a
95 % Wilson CI, and a two-sided Fisher exact test against the credited
4/57, threshold 0.05. This file computes those two numbers and nothing
else — the shares table itself comes from summarize_zones_b.py.

Usage (from oneshot/real_errors/):
    python census_stats.py --labels zone_labels_census_b.csv \
        --credited-real 4 --credited-n 57
"""
import argparse
import csv
import math

from scipy.stats import fisher_exact

import sys
# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')


def wilson(k, n, z=1.959963984540054):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--credited-real', type=int, required=True)
    parser.add_argument('--credited-n', type=int, required=True)
    args = parser.parse_args()

    with open(args.labels, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    k = sum(1 for r in rows if r['class'] == 'real_error')

    lo, hi = wilson(k, n)
    table = [[k, n - k],
             [args.credited_real, args.credited_n - args.credited_real]]
    _, p = fisher_exact(table, alternative='two-sided')

    print(f'census: {k}/{n} real_error = {k / n:.1%} '
          f'[Wilson 95% {lo:.1%}-{hi:.1%}]')
    print(f'credited: {args.credited_real}/{args.credited_n} = '
          f'{args.credited_real / args.credited_n:.1%}')
    print(f'Fisher exact two-sided: p = {p:.4f} '
          f'({"significant" if p < 0.05 else "not significant"} at 0.05)')


if __name__ == '__main__':
    main()
