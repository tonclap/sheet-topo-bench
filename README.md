# sheet-topo-bench — a benchmark for topological errors of scroll surfaces

Sheet switch and merger are the two topological failure modes of Vesuvius
surface tracing: a trace that jumps to the neighbouring winding, and two
windings fused into one surface. They are open problems #3 and #4 of the
Scroll Prize, and the organizers' own phrasing — *"we don't always know which
part of the pipeline is limiting us"* — asks for measurement before repair.

> **Every generated table in this package is English** — the summary, the
> ablations, the verification tables, the zone classifications. What stayed
> Russian is the layer underneath: the protocol, the freeze records, the
> corpus and results write-ups, and the dated inserts. Those are the
> artifacts the runs actually wrote, and rewriting them afterwards would have
> cost the one property that makes them worth shipping — so they were
> translated nowhere and glossed instead.
> **[READING_GUIDE.md](READING_GUIDE.md) translates every column header,
> verdict word, metric and reading rule** they use. To confirm the headline
> numbers you need neither: `python verify.py` recomputes them offline in
> seconds.

This package is that measurement instrument, transplanted from connectomics
(Zung 2017; ERL from Januszewski 2018), where merge/split errors of thin
structures traced through dense volume are a ten-year-old industry:

- **an injector** that plants sheet switches, mergers and holes of known
  position into published segment meshes — ground truth that generates
  itself, no annotation needed;
- **a detector** (mesh + prediction + CT, no GPU) that ranks windows by the
  likelihood of a planted error: a frozen three-channel v1, and a learned
  CPU re-ranking over the frozen candidate pools (v5lu, a logistic
  regression) — each form frozen before its own untouched held-out
  generation and examined exactly once;
- **a metric — ERL for sheets**: expected run length along the surface,
  *how many millimetres of sheet are traced before a topological error* —
  the number that converts directly into "how much text reads in one piece";
- **three real-error corpora** built without GPU, with the negative results
  published alongside the positive ones;
- **every run that ever produced a published number**, failures included,
  and a verifier that recomputes each number of this README from the shipped
  run reports.

## The headline table

Copied from [pipeline/HELDOUT_RESULTS.md](pipeline/HELDOUT_RESULTS.md), which
`heldout_summary.py` generates from the shipped run reports; `verify.py`
regenerates it and diffs. Two exams, two frozen forms, each held-out
generation touched exactly once: v1 was frozen
([protocol/FREEZE_2026-08-14.md](protocol/FREEZE_2026-08-14.md)) before the
first held-out generation was opened (2026-08-15); the learned fusion v5lu was
declared — form in ABLATION_V5.md, composition, training and reading rule in
PROTOCOL §6, freeze in
[protocol/FREEZE_2026-08-19.md](protocol/FREEZE_2026-08-19.md) — before the
second, fresh held-out generation (seed 20260819) even existed, and examined
once (2026-08-20), after which the held-out v2 budget is spent.

| corpus | N | AP [95% CI] | recall@N | ΔERL@N (share of oracle) |
|---|---|---|---|---|
| dev (Paris 4, bands < 100) | 232 | 0.6442 [0.609–0.679] | 0.698 | +0.18 mm (86%) |
| held-out A (Paris 4, bands ≥ 100) | 68 | 0.6663 [0.569–0.750] | 0.750 | +0.11 mm (92%) |
| held-out B (PHerc0139, whole scroll) | 300 | 0.6468 [0.621–0.671] | 0.703 | +0.29 mm (92%) |

The interval of the dev point (0.609–0.679) overlaps both held-out
intervals: the correct reading is **no degradation off the development set**,
not "held-out is better". The oracle ceiling is ΔERL with cuts at the true
positions (PROTOCOL §3); random windows score ~0.

### The v5lu exam — does the dev gain transfer?

v5lu re-ranks the frozen candidate pools with a logistic regression on 7
features (family, per-family log1p mass, a vertical-jump factor) — dev AP
0.8558 vs v1's 0.6442, ΔAP +0.211 [+0.19–+0.24], the only iteration that
passed the declared shipping rule (see the fusion section below). The exam
question, declared before the run: does that gain transfer? Verdict rule,
also declared before the run: by the interval of the paired per-resample
ΔAP. The model was trained once on the full dev pool and applied frozen —
no training, calibration or threshold choice ever touched held-out.

| corpus | N | AP v1 | AP v5lu | paired ΔAP |
|---|---|---|---|---|
| held-out A2 (Paris 4 ≥ 100, seed 20260819) | 63 | 0.5886 [0.491–0.679] | 0.7030 [0.586–0.813] | **+0.114 [+0.09–+0.14]** |
| held-out B2 (PHerc0139, seed 20260819) | 300 | 0.6357 [0.608–0.660] | 0.8853 [0.855–0.914] | **+0.250 [+0.23–+0.27]** |

Both paired deltas are significantly above zero — **the dev gain
transfers**, and on B2 it exceeds the dev gain itself. The declared primary
check was the merger class: dev ΔM +0.216 reproduces on B2 as **ΔM +0.210
[+0.13–+0.29]** significant; B2 recall@N reaches 0.860 and ΔERL@N reaches
98% of the oracle ceiling. The composition of the win is measured
(`transfer_breakdown.py`, offline from the exam reports): on A2 the
candidate sets of v1 and v5lu credit identical injections — the union pool
(61) is smaller than N = 63, so the entire +0.114 is re-ranking within one
pool (median rank of found H drops 28 → 12) and every paired delta except AP
is exactly zero; on B2 the win is both coverage (misses 95 → 42; the union
pool's CT channel credits 9 injections v1's pool never generates) and
ranking (H below-N 26 → 1, median H rank 203 → 50 — the per-family
calibration doing exactly what the dev attribution said it does).

Baselines, from the same runs (PROTOCOL §7): every naive detector — random
windows, kink energy, mesh self-intersections, radial-ray inversions — stays
**at or below AP 0.0064 raw** on dev. Given our own substrate-atlas
differencing (the load-bearing idea of the detector, see below), the best of
them reaches 0.1214 on dev and 0.1720 on held-out A — still multiples under
the frozen v1, and further under v5lu. On the fresh v2 corpora the pattern
repeats: raw baselines at or below AP 0.0005; with atlas differencing the
best reaches 0.1608 on A2 and 0.0124 on B2, against v5lu's 0.7030 and
0.8853. A local-normal-jump detector is the planted-in-advance falsifier: it
must fail on locally-plausible errors, and it does.

## What the detector is

Three channels over the mesh's node grid in global coordinates (r, θ,
winding), each producing evidence cells, clustered, then fused by
within-channel percentile rank (`detect_v1.py`, the form of the first
held-out exam; the second exam ran the same frozen pools under the v5lu
re-ranking via `exam_v5lu.py`):

- **prox** — the trace touches the annotation of the neighbouring winding
  closer than the contact scale (< 5 vx);
- **rect** — the shape of holes torn by an injection differs from the mesh's
  natural raggedness;
- **support** — surplus of "trace without a predicted surface underneath"
  over the pristine count: a merger's seat is mid-gap, where the surface
  prediction says there is no sheet.

Everything stands on **differencing against the substrate's own atlas**: the
production meshes carry the same signatures naturally (0.61% of skeleton
points in contact, 14 042 natural contact clusters; 42 012 self-intersection
cells; 12 039 ray-inversion cells — a quantification of open problem #4 on
its own), and the differencing is what separates a planted error from them —
it lifts even the naive baselines by an order of magnitude (raw vs atlas
columns above). The atlas is thereby part of the corpus definition, declared
before any held-out run (PROTOCOL §4), and it is threshold-dependent — the
sensitivity analysis
([pipeline/detector/ATLAS_SENSITIVITY.md](pipeline/detector/ATLAS_SENSITIVITY.md))
says the published 5 vx map cannot be treated as *the* defect map: publish a
range or the threshold-stable core (15 of the top-100 clusters).

## The fusion wall — and what finally broke it

Four attempts to add a fourth signal or rebalance the three under rank-based
fusion, all evaluated by paired bootstrap on the same resamples, none shipped
([ABLATION_V2](pipeline/detector/ABLATION_V2.md),
[V3](pipeline/detector/ABLATION_V3.md),
[V4](pipeline/detector/ABLATION_V4.md)):

- **front counting** (v2): mechanism works (front evidence covers 64.6% of
  merger windows, 0% of hole windows), ranking does not move (ΔAP −0.009, in
  noise); adding it as a fourth channel is significantly harmful (ΔAP −0.135).
- **CT intensity** (v3): the probe is positive by its pre-declared rule
  (CT clusters cover 13/36 v1-missed merger windows, 0/77 hole windows —
  the feature does not read the prediction and does not go blind in its
  holes); merged into the support family it gives the line's first
  significant AP gain (+0.023) but pays ΔH −0.078 significantly.
- **manifest quotas** (v4): the largest significant AP gain of the line
  (+0.040 [+0.02–+0.06]) and H restored to 1.000 — paying ΔS −0.120
  significantly. Quotas carry no free constants (family→type mapping is
  mechanistic, quotas are the dev manifest's type shares) and still do not
  pass the shipping rule (no significant loss on any family).
- The measured conclusion: **the ranking prefix is a conserved quantity** —
  rank-based fusion schemes (percentile = share of pool size, quota = share
  of manifest) only move the deficit between the S/M/H families. Linking a
  third working family in needs score calibration.

### Learned fusion broke it — on CPU

That calibration turned out to cost a logistic regression, not a GPU
([ABLATION_V5](pipeline/detector/ABLATION_V5.md), protocol committed before
training): **v5lu, dev AP 0.8558 [0.816–0.892] vs v1's 0.6442 — ΔAP +0.211
[+0.19–+0.24], ΔM +0.216, ΔH +0.220, Δplausible +0.147, all significant,
with no significant loss on any family** — the first and only iteration to
pass the declared shipping rule. Before any held-out spending, the result
survived eight declared checks
([VERIFY_V5](pipeline/detector/VERIFY_V5.md),
[VERIFY_V8](pipeline/detector/VERIFY_V8.md)): label permutation (p95 0.794 <
0.82 — the features cannot fake the gain without the labels), a hard block
split (0.8578), λ-regularization sweeps (shifts ≤ 0.002), attribution
(per-family intercepts alone give 0.8255 — calibration is the load-bearing
part; vjump is not), transfer to the other scroll (0.8730 vs 0.6468),
restoration of the drowned real-corpus channel (below), and a fresh-seed
end-to-end rebuild on data that did not exist during development (v1 0.6172
→ v5l 0.8150, ΔAP +0.1978 [+0.1843–+0.2108]). One instructive attribution
number: sorting by raw mass with no learning at all gives 0.7436 — **the
percentile fusion is worse than no normalization whatsoever**; supervision
adds ~+0.11 over label-free normalizations. The held-out transfer of the
whole construction is the exam table above.

### The next wall is generation, not fusion — ceiling ≈ 0.905

With fusion calibrated, the deficit moved and was measured where it landed
([coverage breakdown in ABLATION_V5](pipeline/detector/ABLATION_V5.md),
regenerable from `runs/topo/coverage_breakdown_paris4.json`): the v1 pool
covers 199/232 injections, the union pool 210/232 = 0.905, and v5lu's
recall@N (0.853) sits essentially at that pool ceiling. Of the 22 dev
injections with no candidate anywhere, 17 are mergers (14 locally
plausible) — the known blind spot where the prediction is silent. Three
declared assaults on the ceiling all failed by their own pre-registered
rules and ship as measured boundaries: **v6** (relaxed floors + 14
features, [ABLATION_V6](pipeline/detector/ABLATION_V6.md)) — significant
losses everywhere except H; the floors carried candidate granularity, not
noise, and relaxing them shrank the prox pool 102 → 96 by cluster gluing;
**v6s/v6r** (separate mass-vs-clustering thresholds,
[ABLATION_V6S](pipeline/detector/ABLATION_V6S.md); both are configurations of
`detect_v6s.py` rather than separate files — v6r is the
restricted-connectivity one, flags `v6r_row_gap` and `v6r_diagonals`) —
de-glued the clusters
(pool 139) yet lifted the union ceiling only 210 → 211/232, ΔAP +0.0018
n.s.: the 17 uncovered mergers have no prox signal even at threshold 0.25;
**v7** (through-winding pair features against the declared prox ghost,
[ABLATION_V7](pipeline/detector/ABLATION_V7.md)) — the pairs exist as a
phenomenon (53/66 carry exactly one true candidate) and move the target
family by exactly 0.000. The ceiling stands as the boundary of the existing
generation channels.

## Real errors without GPU — three corpora

Synthetic injections risk teaching the detector an injection artifact, so the
protocol demands real labels (PROTOCOL §5). Every real labelling buildable
without GPU was built and run against the frozen detector
([pipeline/real_errors/CORPUS.md](pipeline/real_errors/CORPUS.md) — every rule
and constant declared before the corresponding run;
[RESULTS.md](pipeline/real_errors/RESULTS.md)):

| corpus | labels | frozen channel | AP | random base | verdict |
|---|---|---|---|---|---|
| A — recto/m7 disagreement (mass ≥ 40) | 815 | prox | 0.0081 [0.0068–0.0096] | 0.0015 [0.0011–0.0019] | ~5.6× over chance |
| A2 — both-silent consensus (mass ≥ 40) | — | prox | 0.0025 | 0.0011 | weaker than A |
| B — human-verified GP-banner reference (mass ≥ 20) | 178 | prox | 0.0002 [0.0000–0.0004] | 0.0003 [IQR 0.0001–0.0006] | at chance |
| B — same zones | 178 | **support** | **0.0330 [0.0262–0.0406]** | 0.0003 [IQR 0.0001–0.0006] | **~99× over chance — signal** |
| B — same zones | 178 | **thick** | **0.0306 [0.0232–0.0383]** | 0.0003 [IQR 0.0001–0.0006] | **signal** |
| B — same zones | 178 | **defect** | **0.0134 [0.0091–0.0177]** | 0.0003 [IQR 0.0001–0.0006] | **signal** |

Corpus B is the only labelling with no circularity to any model: its zones
are geometric distances of production meshes to the human-verified GP-banner
(59 million reference points, carried into the 2026 frame by an affine with
0.39 vx residual). **Three prediction-free channels see it**, each declared
with its reading rule committed before its run, each behind a regression
gate that replays the previous channels bit-for-bit first:

- **support** (declared in advance as the follow-up to the dev-corpus
  support channel): the label never reads a
  prediction, the channel reads recto — independent witnesses agreeing on
  where the 2026 trace left the sheet. Holds at both sensitivity thresholds
  (AP 0.0322 at mass ≥ 10, 0.0141 at ≥ 40).
- **thick** — radial CT run length through the node (calibration frozen
  from the CT probe, no new tuning): reads neither predictions, nor recto,
  nor the neighbouring mesh — not one shared input bit with support, at the
  same signal level.
- **defect** — disclination density of the CT normal field (structure
  tensor, sign transport around a ring): volume topology of the substrate
  itself, a 68 619-node lattice pass; ~40% of the support/thick level and
  complementary in addresses — of its 33 credited zones, **14 are seen by
  no other channel**, and Z0136 (below) gets its best rank (52) from it.

**One limit belongs next to the "×over chance" column, because the
comparison is not symmetric.** AP here is normalized by the number of zones
but summed over the whole ranking, and a longer ranking can only add
non-negative terms. The channels rank their full candidate pool — **712
windows** for all three — while the random baseline is drawn at the corpus
size, **178 windows** (`--random-seeds 100`, sampled without replacement
from the same window pool). So the baseline gets a quarter of the channel's
budget, and the multipliers above are upper bounds rather than fair ratios.
The bias runs in our favour and is not small in principle. What it does not
plausibly do is manufacture the verdict: the extra 534 ranks sit
deep, where each additional hit contributes on the order of 1/500 to AP,
against a gap of two orders of magnitude between 0.0330 and 0.0003. Read
the column as "clearly above chance", not as an exact multiple.

What fusion does to these channels is measured on both sides of the
boundary. The frozen percentile fusion drowns the live signal
(prox+support on B: AP 0.0012, back at chance), and the mechanism is
measured: **dilution** — the prox pool on this labelling is
7.2× larger pure noise, and thinning it at frozen fusion restores AP
monotonically (0.0012 → 0.0217 at 5% of the pool). Between signal channels
of comparable pools the wall does not reproduce: every declared label-free
form of support+thick (0.0333–0.0350) and the three-channel fusion3
(0.0308) **hold the best solo level but do not lift it** — the paired
intervals against solo all cross zero, and a supervised LOSO logit on 178
zones is significantly *worse* than either solo (0.0167). The learned
dev-fusion replayed on B restores the drowned channel to solo level
(0.0333 — the V7 check above). The boundary that ships with the
package: **fusion does not add to the best solo channel; with asymmetric
pools it drowns it; the value of extra channels is independent inputs and
address complementarity, not ranking gain.**

- The 124 zones of double evidence (models disagree AND the mesh detector
  fires) were **classified by hand, all of them**
  ([ZONES_124.md](pipeline/real_errors/ZONES_124.md)): 1 real error (0.8%,
  low confidence), 91 model artifacts (73.4%), 32 false alarms (25.8%).
- The corpus-B by-product stands on its own: **production meshes of 2026
  diverge from the 2023 human-verified reference on 18.8% of covered nodes;
  meshes of the reference's own epoch — on 2.1%**. A ninefold gap, measured
  with the reference in the middle.

In summary: this detector validates on synthetics and transfers
across scrolls there; of the real corpora buildable without GPU, the
model-derived labellings (A, A2) measure *something else* — model
disagreement systematics and prediction holes — while the human-verified
corpus B independently validates three prediction-free channels and locates
the real limits of fusing them.

### The support-credited B zones, by eye — and what it took to verify one

All 57 zones the frozen support channel credited on corpus B were
classified by hand against criteria declared before the first card was
rendered ([ZONE_CRITERIA_B.md](pipeline/real_errors/ZONE_CRITERIA_B.md),
labels in [zone_labels_b.csv](pipeline/real_errors/zone_labels_b.csv),
table generated by `summarize_zones_b.py` into
[ZONES_SUPPORT_B.md](pipeline/real_errors/ZONES_SUPPORT_B.md)):

| class | zones | share |
|---|---|---|
| real_error candidates (single-slab reading) | 4 | 7.0% |
| banner_edge | 8 | 14.0% |
| epoch_drift | 30 | 52.6% |
| false_alarm | 15 | 26.3% |

The majority class is the epoch systematics the 18.8% table above predicts;
the 4 single-slab candidates then went through two declared verification
protocols (both committed before rendering,
[RESULTS.md](pipeline/real_errors/RESULTS.md)):

- **independent axial slabs** on other zone rows (`confirm_zones_b.py`):
  none of the four confirmed — the readable independent windows all read
  as drift or noise;
- **a chain-normal band** along the traced row itself (`band_zones_b.py`,
  s = arc along the chain, n = in-plane normal, CT + prediction panels,
  banner as witness; `band_neighbour_rows.py` runs the same band on a
  single neighbouring row, `band_row_readout.py` reads the numbers out of
  it): **three of the four were slab-projection
  artifacts** (the chain holds its sheet along the whole zone), and
  **one — Z0136 — is a confirmed real 2026 tracing error**: over ~250 vx
  (~2.4 mm) of arc the sheet the row-680 chain was on leaves n = 0
  downward and does not return, taking the human-verified banner points
  with it, and a *different* band then arrives at n = 0 and stays; blind
  and control passes both high-confidence. **This is a claim about the
  trajectory, not about prediction being missing pointwise** — pointwise
  the row keeps 0.810 support against 0.932 and 0.772 on its `on_sheet`
  neighbours, which separates nothing (dated section of
  [RESULTS.md](pipeline/real_errors/RESULTS.md), 21.08.2026).

Two lessons: the verified address-level yield of
support on corpus B is **1/57, not 4/57** — and ±6 vx axial slab cards
overstate real errors (3 of 4 candidates dissolved), so the chain-normal
band is the mandatory verification step for any address-level claim from
this benchmark.

### The census control — support enriches real errors

The declared follow-up (`collect_censusB_zones.py`, protocol in
ZONE_CRITERIA_B.md) was then measured: all **121 uncredited** B zones were
classified by the same blind-batch + control procedure
([ZONES_CENSUS_B.md](pipeline/real_errors/ZONES_CENSUS_B.md)) and yielded
**0/121 real-error candidates [Wilson 95% 0.0–3.1%] against 4/57 (7.0%)
among the support-credited zones — Fisher's exact two-sided p = 0.0098**.
For the first time the comparison base comes from the same corpus, and it
turns "support finds zones" into "support *enriches* real errors". The
control pass also dissolved all 9 blind real-error candidates of the census
(the near-vertical-sheet slab-reading pattern), reproducing the
slab-overstatement lesson a second time.

**Three limits belong next to that p-value.** (1) It rests on **4 events in
total** — one reclassification moves it across 0.05, so read it as a
directional result on a small count, not a robust effect size. (2) Both
sides are counted at the **axial-slab level of reading**, which is what
makes the comparison apples-to-apples — but the numerator is the same 4 that
the later band verification reduced to 1 confirmed (Z0136). The test says
support enriches *slab-level candidates*; it does not say support finds four
real errors. (3) "Uncredited zones contain no real errors" would be the
wrong reading of the 0/121: those zones are uncredited **by support**, and
the defect channel later surfaced its own candidates from inside that same
set (previous section). The claim the number supports is narrow: *within
corpus B, at slab-level reading, the support channel's credited zones are
richer in real-error candidates than its uncredited ones.*

### The address list — Z0136 and four defect-channel candidates

**Z0136** carries four credits — three automated channels and the band —
and it is the band that is the decisive evidence: the chain-normal band (crossing, blind and control both
high-confidence — over ~250 vx of arc the sheet the chain was on leaves
n = 0 and does not return, taking the banner with it, and a different
band arrives; a trajectory reading, not a pointwise one). The other three are the
corroborating credits — the original support credit (rank 224), a thick
credit (rank 262), and the best rank of all from the defect channel (52).

Count the *inputs* rather than the credits and they are **three, not four**:
the band and support both read the recto prediction, thick and defect both
read the CT volume, and the human-verified banner is the third. The channels
are independent as features — support reads a prediction, thick and defect
read no prediction at all — but a reviewer counting disjoint data sources
should count three. Z0136 also passed the independent-slab protocol that
dissolved three of its four peers. It ships with a ready bug-report draft
(coordinates, both projections, blind readings, reproduction).

The 14 zones only the defect channel credits went through the same
two-pass band protocol (blind agents + zoom control, with the four
verified bands of the support round mixed in as quality controls):
**2 candidates
confirmed at high confidence both passes (Z0033, Z0078), 2 more with
reduced control confidence (Z0015, Z0121)**, 3 blind crossings dissolved
by the control zoom — the blind pass overstates, the control pass is
mandatory, third reproduction of that rule.

**These four are a weaker tier than Z0136.** They carry one band pass; they
have *not* been through the independent-slab check (the protocol that turned
4 support candidates into 1). And the 121-zone census, reading the same
zones from axial slab cards, classified all four as
`epoch_drift`/`false_alarm` — i.e. not errors. The two readings disagree,
and this package resolves the disagreement in the band's favour on the same
evidence that rescued Z0136 (the slab projects neighbouring sheets as
crossed; the band does not). That is a defensible rule, applied consistently
— but it is one method's word against another's, so the four ship as
candidates for someone with volume access to settle, not as findings. The
shipped address list is therefore **1 verified real 2026 tracing error + 4
candidates**, each with segment, row, column and band evidence in
[RESULTS.md](pipeline/real_errors/RESULTS.md) and the band label CSVs.

## Detect, then correct?

The connectomics playbook (Zung: detector → corrector) was priced by
building one (`correct_v1.py`,
[runs](runs/topo/corrector_v1_paris4.json)): blind cutting of random windows
is catastrophic (-16 mm ERL at k=N/2, -109 mm at k=4N), the detector shrinks
the harm by 1–2 orders of magnitude, and the sign at k≤N depends on the
reseaming model — -1.6 mm with a seam scar, +1.6 mm with ideal reseaming,
which is 13% of the corrector oracle's +12.4 mm. A false cut costs a
multiple of what a true repair recovers, so deployment needs precision far
above the detector's actual. Verdict: **do not deploy detect-and-cut; the
detector's use is triage for human eyes** — the manual pass the field is
trying to shorten (~25 h of annotation per winding × 31 windings ≈ 775
person-hours on PHerc. 1667, from the Scroll Prize's own segmentation
reporting; the only number in this README that comes from outside our runs
and is therefore not bound by `verify.py`).

## Compare your detector

The benchmark's interface is one function call away:

1. Rebuild a corpus from its shipped manifest. The manifest is the record,
   not an input file: it carries the seed, `per_type` and scroll the run
   used, and for each of its 300 injections the exact rectangle, mechanism,
   applied `peak_shift_vx` and self-check. Re-run the injector with those
   settings —
   `inject_errors.py --scroll PHercParis4 --seed 20260814 --per-type 100
   --cache … --grid-cache … --out …` — and diff your manifest against
   `runs/topo/corpus_paris4/manifest.json`: identical rectangles and shifts
   mean identical grids. (Seeds: 20260814 for the dev corpora, 20260819 for
   the second held-out generation, 20260817 for the v2 plausibility corpus.)
2. Emit your ranking as `(segment, row, col, score)` windows over the node
   grid.
3. Score it: `sheet_erl.hits` credits a window to an injection if its centre
   falls in the error rectangle grown by 50% per axis, once per injection,
   best rank wins; a window inside two grown rectangles is credited to the
   first still-uncredited injection **in manifest order**, not the nearest
   one (PROTOCOL §3, insert of 2026-08-20); `evaluate_ranking` returns AP, recall@N and
   ΔERL@N against the same oracle and random floors printed here.
4. Report AP with the by-injection bootstrap
   (2000 resamples, seed 20260815 — `heldout_summary.py` shows the exact
   procedure), your worst winding band (PROTOCOL §8), and the §7 baselines
   from your own run.

Held-out discipline if you want to compare against our held-out numbers:
PHerc0139 entire, and Paris 4 bands ≥ 100, one shot, no peeking (PROTOCOL
§6). Both our held-out generations (seeds 20260814 and 20260819) are spent;
the protocol's discipline for a new comparison is the one we followed for
the second exam — generate a fresh seed, declare composition and reading
rule before the run, and read the verdict off the paired interval.

## Reproduce

Light (no network, seconds) — every number in this README, from the shipped
run reports. Three packages, pinned to versions this has been run green on
(`requirements-verify.txt`); `scipy`, `imagecodecs` and `matplotlib` sit in
`requirements.txt` for the full pipeline and are not needed here:

```bash
pip install -r requirements-verify.txt
python verify.py
```

Run in bare virtual environments outside the source repository, on three
interpreters, 2026-08-20 — each **15/15 regenerations, 74/74 bindings,
exit 0**:

| interpreter | numpy | tifffile | numcodecs |
|---|---|---|---|
| CPython 3.10.6 (the development version) | 2.2.6 | 2025.5.10 | 0.13.1 |
| CPython 3.11.15 | 2.2.6 | 2025.5.10 | 0.13.1 |
| CPython 3.14.7 | 2.5.2 | 2026.8.16 | 0.16.5 |

The pins record what was verified, not a fragility: the same run is green on
tifffile 2023.8.12 as well, four years of that library apart. If it does not
reproduce for you, that is a bug in this package and worth an issue.

It regenerates the summary tables (`heldout_summary.py` — including the
v5lu exam section and its paired deltas, the `ablation_summary*.py` family
v2–v7, the three `summarize_zones*.py` zone tables, `real_ci.py`) from
`runs/` and the shipped label CSVs, and compares them against the shipped
markdown/JSON (byte-compare for the pure generators; containment for the
ablation files that carry dated verdict inserts), then checks every bound
number of this README against the run reports it came from.

Full (network, hours) — the pipeline end to end: fetch public meshes and
predictions, rebuild both synthetic corpora from manifest seeds, re-run the
frozen detector, baselines, metric, real-corpus builders and evaluations. The
commands, in dependency order, with expected costs, are in
[protocol/PROTOCOL.md](protocol/PROTOCOL.md) and each script's docstring (whose
`Usage:` paths need one substitution — [READING_GUIDE.md](READING_GUIDE.md)
states it); every
script is resumable via per-segment checkpoints, and the interruptions the
published runs actually hit are recorded in
[FREEZE](protocol/FREEZE_2026-08-14.md) inserts rather than cleaned up.

`requirements.txt` adds three packages for the full path, and not all three
are pinned on equal evidence — the file itself says which is which, per
package. The one that matters offline: **scipy** is pinned to verified
versions, and `census_stats.py` reproduces this README's census p-value on
two scipy majors to the last digit — as does `verify.py`, which recomputes it
without scipy at all.

CPU only, public data, no credentials.

## What is not shipped, and why

- **Corrupted grids of the corpora** (~420 MB `.npy`): regenerated by the
  injector from the seed and settings the manifest records. The check is a
  diff of the two manifests — every injection's rectangle, mechanism and
  applied shift is written there, so a divergent rebuild shows up as a
  differing row. We do not ship per-grid hashes; adding them to the injector
  is the obvious improvement and has not been made.
- **Scan/prediction chunk caches** (tens of GB): public data on
  dl.ash2txt.org, fetched on demand by the shipped readers.
- **Zone image cards** (120 MB corpus A; corpus B cards, confirmation
  slabs and chain-normal bands likewise): rendered from shipped maps by
  `render_zones.py` / `render_zones_b.py` / `confirm_zones_b.py` /
  `band_zones_b.py` / `band_neighbour_rows.py`; the classifications they
  fed are shipped in full
  ([ZONES_124.md](pipeline/real_errors/ZONES_124.md),
  [zone_labels.csv](pipeline/real_errors/zone_labels.csv),
  [ZONES_SUPPORT_B.md](pipeline/real_errors/ZONES_SUPPORT_B.md),
  [zone_labels_b.csv](pipeline/real_errors/zone_labels_b.csv), the
  confirmation and band label CSVs under `runs/topo/real_paris4/`).
- **A GPU pass** (spiral-fitting corpus C): out of scope by declared gate
  ([protocol/UNTRIED.md](protocol/UNTRIED.md) keeps the full list of roads
  not taken, with the conditions under which each would be). The learned
  fusion once listed here turned out not to need a GPU and ships above.

## Known blind spots

- **Mergers seated in prediction holes.** The support channel needs the
  prediction to speak; where it is silent the merger's mid-gap signature is
  invisible, and M recall (0.544 on dev) is v1's weakest family throughout.
  v5lu lifts M to 0.759 dev / 0.770 on B2 (the declared primary transfer
  check), but the residual uncovered mergers are exactly the ≈ 0.905
  generation ceiling above — calibration cannot rank a candidate that was
  never generated.
- **The rect channel dilutes H.** Adding channels costs the hole family
  (a lesson reproduced three times — see
  [protocol/UNTRIED.md](protocol/UNTRIED.md)); the shipped fusion accepts that
  trade, and the quota ablation shows what restoring H costs elsewhere.
- **The atlas is threshold-dependent** (verdict above): mesh-anomaly maps at
  a single contact threshold are not publishable as *the* defect map.
- **Corpus B sees windings 10–52 of one scroll**, with a conservative
  coverage cut (T_cover = 10 vx, declared fallback) that discards deep
  divergences — a bias against merger-like sites, recorded before the run.
- **Corpus B is not dense enough for voxel metrics.** The declared density
  gate (≥ 80% of banner-sector nodes covered) failed at **46.0%**
  (44.2–47.0% under T_cover × 0.75/× 1.25; per segment 5.8–78.0%), so
  VOI / adapted Rand are not computed — globally or per sector.
- **The local-plausibility measure is node-scale noisy.** Corpus v2
  (seed 20260817, both measures in the shipped manifest; the no-flag run
  reproduces the v1 manifest bit-for-bit) puts the smoothed-measure share
  of locally plausible injections at **0.63 vs the declared 0.52**, the
  whole shift in type S (hann splices are smoother than node-scale normal
  noise). The v1 number stands as declared; read per-group ablation rows
  knowing the S group is undercounted by measure noise.
- **ERL-for-sheet is injection-count-normalized** on synthetic corpora; on
  real substrate it has no ground-truth cuts to normalize against, so real
  corpora report AP/recall only.

## What was tried and did not survive

Kept because a measured negative is reusable: normalized contact forms (died
against the kernel's natural asymmetry), vertical-jump as a channel (neutral
on synthetics, zero on corpus B by its declared rule), front counting as a
channel (neutral as replacement, harmful as addition — twice), CT as a
replacement or addition (harmful), whole-scroll global channels
(complementarity probe: 8–11 of 70 missed injections — the local channels
with atlas differencing already carry the global information), both rank
fusion schemes (the wall above), relaxed-floor regeneration, separate
thresholds and through-winding pairs (the three ceiling assaults above), the
radial self-residual axis (0/22 by its declared rule — the substrate's
background is not smooth at signal scale), the trace-phase probe (5/22 at
the ≥ 8/22 rule; its diagnostic lived only in a differential unavailable to
a shippable channel, and the phase channel's floor catches the median honest
node), the ring disclination construction on synthetics (perfect background,
zero differential — volume topology is a substrate property, not co-located
with injections; the same construction then signed up as the real-corpus
defect channel, which is the interesting half of that result),
detect-and-cut correction (economics above), "both models silent means mesh
error" (refuted: those are prediction holes), and the model-disagreement
corpus as detector validation (it measures the models' argument, not the
mesh).

## Licence and disclosure

MIT ([LICENSE](LICENSE)). Written with AI assistance under human direction
and review ([AI_DISCLOSURE.md](AI_DISCLOSURE.md)).
