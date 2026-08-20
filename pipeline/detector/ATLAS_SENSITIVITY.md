# Contact-atlas sensitivity to the threshold (TOPO-021)

_Hand-transcribed from `output/topo/atlas_sensitivity.json`
(`pipeline/detector/atlas_sensitivity.py` — the script writes JSON only, it
has no markdown renderer; every digit below was checked against the JSON
when this file was translated). The protocol and the verdict rule were
declared in the script's docstring before the run. Sensitivity analysis on
the CLEAN Paris 4 dev grids (16 segments); the frozen detector and the
held-out numbers are untouched._

## The numbers at each threshold

| contact threshold | share of skeleton points | evidence cells | clusters |
|---|---|---|---|
| < 3 vx | 0.127 % | 16 191 | 6 313 |
| < 5 vx (the published one) | 0.606 % | 60 618 | 14 054 |
| < 8 vx | 3.712 % | 271 701 | 22 576 |

The reference values reproduce: 0.606 % ≈ the published 0.61 % of points
in contact, 14 054 clusters ≈ 14 042 (`masked_prox_clusters` in
`detector_v1_paris4.json`; that count is over damaged copies of the same
grids).

## Overlap measures (declared before the run)

- Jaccard of evidence cells against 5 vx: **0.267** (3 vx), **0.223** (8 vx).
- Retention of the top-100 clusters by mass (rule "≥ 50 % of cells on the
  other threshold's top-100 with ±1-row dilation", as in
  `differenced_clusters`):
  - 5 vx → 3 vx: **0.24**; 3 vx → 5 vx: **0.73**;
  - 5 vx → 8 vx: **0.76**; 8 vx → 5 vx: **0.03**.

## Verdict (by the rule declared before the run)

**Threshold-dependent** (`порого-зависима` in the JSON; rule: stable if all
retentions ≥ 0.8; threshold-dependent if any < 0.5). The composition of the
map changes severalfold: the contact share grows ×4.8 on the 3→5 vx step and
×6.1 on the 5→8 vx step; the top clusters of the wide threshold barely
overlap the reference ones (0.03).

Consequence for TOPO-013 (per the task's Outcome): the substrate defect map
cannot be published as one number at one threshold. The publishable form is
a range over the three thresholds with an explicit sensitivity note, or the
threshold-stable core.

## The threshold-stable core (descriptive statistics, not part of the verdict)

Of the top-100 reference (5 vx) clusters, **15** survive at both neighbouring
thresholds. The asymmetry of directions (3 vx → 5 vx = 0.73 against
5 vx → 3 vx = 0.24) says the tight contact cores are robust to widening the
threshold, while the bulk of the reference map is skirts added by the
threshold itself.
