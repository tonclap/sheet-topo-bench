# -*- coding: utf-8 -*-
"""Where the v5lu − v1 transfer comes from, by error family and by winding band.

Reads the finished exam reports only — it never runs anything against held-out:
  detector_v1_{paris4,0139}_h2.json   frozen v1 on the v2 corpora
  exam_v5lu_{paris4,0139}_h2.json     v5lu (exam_v5lu.py), same corpora

They live in `output/topo/` in the working repository and in `runs/topo/` in the
published package; whichever exists is found automatically, `--topo` overrides.

The question was declared before the exam: which types and which bands carry the
gain, and does the dev deficit breakdown reproduce on data nobody had opened
(dev: 34 misses = 22 with no candidate at all, 17 of them mergers, plus 12 found
below rank N, 9 of them sheet switches).

Deterministic; writes markdown to stdout. That markdown is the transfer section
of ABLATION_V5.md, so the printed strings below stay in Russian like the rest of
the generated records — READING_GUIDE.md translates them.
"""
import argparse
import json
import statistics
from pathlib import Path

import sys
# A Windows console defaults to a legacy code page, and the records below carry
# Cyrillic, Delta and the minus sign. Substitute the unrepresentable rather than
# raise: the numbers are the payload, and a UnicodeEncodeError would hide all of
# them behind the first one that does not fit.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(errors='replace')

ROOT = Path(__file__).resolve().parents[2]
# `output/topo` in the working repository, `runs/topo` in the published
# package, which ships the same reports read-only under a different name.
# Whichever exists is the one to read; --topo overrides both.
OUT = next((p for p in (ROOT / "output" / "topo", ROOT / "runs" / "topo")
            if p.is_dir()), ROOT / "output" / "topo")

PAIRS = [
    ("A2 (Paris 4 >= 100)", "detector_v1_paris4_h2.json", "exam_v5lu_paris4_h2.json"),
    ("B2 (PHerc0139)", "detector_v1_0139_h2.json", "exam_v5lu_0139_h2.json"),
]


def load(name):
    with open(OUT / name, encoding="utf-8") as f:
        return json.load(f)


def rank_map(rep):
    return dict(rep["per_injection_rank"].items())


def typ(inj_id):
    return inj_id[0]  # S / M / H


def fmt(x):
    return "—" if x is None else str(x)


def totals(v1, v5):
    """Miss, coverage and rank counts of one exam pair — the single definition.

    `breakdown` prints these numbers into ABLATION_V5.md and the package's
    `verify.py` binds README literals to them. Two copies of the counting rule
    would drift apart silently, with the verifier staying green on its own
    copy, so both callers read this one.
    """
    r1, r5 = rank_map(v1), rank_map(v5)
    ids = sorted(r1)
    n = v1["scoped_injections"]

    def no_candidate(r, subset):
        return [i for i in subset if r[i] is None]

    def below_n(r, subset):
        return [i for i in subset if r[i] is not None and r[i] > n]

    def medians(subset):
        both = [i for i in subset if r1[i] is not None and r5[i] is not None]
        if not both:
            return (None, None)
        return (statistics.median(r1[i] for i in both),
                statistics.median(r5[i] for i in both))

    def group(subset):
        none1, none5 = no_candidate(r1, subset), no_candidate(r5, subset)
        low1, low5 = below_n(r1, subset), below_n(r5, subset)
        return {"ids": subset,
                "no_candidate": (none1, none5),
                "below_n": (low1, low5),
                "misses": (len(none1) + len(low1), len(none5) + len(low5)),
                "medians": medians(subset)}

    out = group(ids)
    out["n"] = n
    out["by_type"] = {t: group([i for i in ids if typ(i) == t]) for t in "SMH"}
    return out


def breakdown(label, v1_name, v5_name):
    v1, v5 = load(v1_name), load(v5_name)
    r1, r5 = rank_map(v1), rank_map(v5)
    assert set(r1) == set(r5), "injection sets differ between the reports"
    counts = totals(v1, v5)
    ids, n = counts["ids"], counts["n"]
    assert n == v5["scoped_injections"] == len(ids)

    print(f"### {label} — N = {n}, recall@N cutoff = {n}")
    print()
    print(f"AP: v1 {v1['metrics']['ap']:.4f} -> v5lu {v5['metrics']['ap']:.4f}; "
          f"recall@N: {v1['metrics']['recall_at_n']:.3f} -> {v5['metrics']['recall_at_n']:.3f}")
    print()

    # --- by error family (S/M/H) ---
    print("| type | total | v1: no cand. | v1: below N | v1 recall@N | v5lu: no cand. | v5lu: below N | v5lu recall@N | median rank v1->v5lu (found by both) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in "SMH":
        g = counts["by_type"][t]
        if not g["ids"]:
            continue
        (v1_none, v5_none) = g["no_candidate"]
        (v1_below, v5_below) = g["below_n"]
        med1, med5 = g["medians"]
        rc1 = v1["recall_by_type"].get(t)
        rc5 = v5["recall_by_type"].get(t)
        print(f"| {t} | {len(g['ids'])} | {len(v1_none)} | {len(v1_below)} | {rc1:.3f} | "
              f"{len(v5_none)} | {len(v5_below)} | {rc5:.3f} | {fmt(med1)} -> {fmt(med5)} |")
    print()

    # --- miss breakdown, against the dev figures quoted above ---
    for slot, tag in ((0, "v1"), (1, "v5lu")):
        miss_none = counts["no_candidate"][slot]
        miss_below = counts["below_n"][slot]
        def by_t(lst):
            return {t: sum(1 for i in lst if typ(i) == t) for t in "SMH"}
        print(f"{tag}: misses {counts['misses'][slot]} = "
              f"{len(miss_none)} with no candidate {by_t(miss_none)} + "
              f"{len(miss_below)} below N {by_t(miss_below)}")
    print()

    # --- how ranks moved ---
    both = [i for i in ids if r1[i] is not None and r5[i] is not None]
    up = sum(1 for i in both if r5[i] < r1[i])
    down = sum(1 for i in both if r5[i] > r1[i])
    same = sum(1 for i in both if r5[i] == r1[i])
    gained = [i for i in ids if (r1[i] is None or r1[i] > n) and (r5[i] is not None and r5[i] <= n)]
    lost = [i for i in ids if (r1[i] is not None and r1[i] <= n) and (r5[i] is None or r5[i] > n)]
    def by_t(lst):
        return {t: sum(1 for i in lst if typ(i) == t) for t in "SMH"}
    print(f"Ranks (found by both, {len(both)}): up {up}, down {down}, unchanged {same}.")
    print(f"Entered the top-N under v5lu: {len(gained)} {by_t(gained)}; dropped out: {len(lost)} {by_t(lost)}.")
    print()

    # --- by winding band ---
    bands = sorted(set(v1["recall_by_band"]) | set(v5["recall_by_band"]))
    print("| band | recall@N v1 | recall@N v5lu |")
    print("|---|---|---|")
    for b in bands:
        print(f"| {b} | {v1['recall_by_band'].get(b, float('nan')):.3f} | {v5['recall_by_band'].get(b, float('nan')):.3f} |")
    print()
    print(f"Locally plausible: recall v1 {v1['recall_on_locally_plausible']:.3f} -> "
          f"v5lu {v5['recall_on_locally_plausible']:.3f}")
    print()


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topo', default=str(OUT),
                        help='directory holding the exam reports')
    OUT = Path(parser.parse_args().topo)
    print("## TOPO-044: breakdown of the v5lu - v1 transfer on held-out v2 (offline, from the exam reports)")
    print()
    for label, a, b in PAIRS:
        breakdown(label, a, b)


if __name__ == "__main__":
    main()
