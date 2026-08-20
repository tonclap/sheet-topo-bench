"""TOPO-032: the corpus-B zones the support channel did NOT credit.

The census counterpart of `collect_supportB_zones.py`: TOPO-029 measured the
real_error share among the 57 support-credited zones; this list is the other
side of the corpus — every zone the channel left uncredited — so the base
rate outside the credit can be measured with the same criteria and the
enrichment claim stops leaning on a different corpus (A) for its baseline.

Gates: the reconstructed zone list must be exactly the published corpus
(178 zones for the declared mass floor) and the credited ids must match
`supportB_zones.json` exactly, or nothing is written.

Usage (from oneshot/real_errors/):
    python collect_censusB_zones.py \
        --map ../../output/topo/real_paris4/corpusB.json \
        --support ../../output/topo/real_paris4/supportB_zones.json \
        --out ../../output/topo/real_paris4/censusB_zones.json
"""
import argparse
import json
import sys

# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

sys.path.insert(0, '.')
from eval_real import zone_records                                    # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True)
    parser.add_argument('--support', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.map, encoding='utf-8') as f:
        corpus_map = json.load(f)
    with open(args.support, encoding='utf-8') as f:
        support = json.load(f)

    zones = zone_records(corpus_map['zones'], support['declared_zone_min'])
    if len(zones) != support['n_zones']:
        raise SystemExit(f"zone count mismatch: {len(zones)} vs published "
                         f"{support['n_zones']} — nothing written")
    credited = {z['id'] for z in support['zones']}
    missing = credited - {z['id'] for z in zones}
    if missing:
        raise SystemExit(f'credited ids not in corpus: {missing} — '
                         f'nothing written')
    # support_rank '-' keeps the card header layout without leaking a rank.
    unc = [dict(z, support_rank='-') for z in zones if z['id'] not in credited]

    report = {'declared_zone_min': support['declared_zone_min'],
              'n_zones': support['n_zones'],
              'n_credited': len(credited),
              'n_uncredited': len(unc),
              'zones': unc}
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"census: {len(unc)} uncredited zones of {support['n_zones']} "
          f"({len(credited)} credited) -> {args.out}")


if __name__ == '__main__':
    main()
