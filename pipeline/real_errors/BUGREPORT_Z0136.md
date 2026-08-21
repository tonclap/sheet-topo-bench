# Bug report: sheet crossing in trace `20260701183124-w010-027` (PHercParis4)

_The address-level report for Z0136, the one real 2026 tracing error this
benchmark verified (README, "The address list"). **The issue body is
everything between the two horizontal rules below**, in English, laid out
along the villa issue template
(`.github/ISSUE_TEMPLATE/issue.md`); this header and the checklist are not
part of it._

_**Sent 21.08.2026 as [villa#1549](https://github.com/ScrollPrize/villa/issues/1549)**
(open). The text below is what was posted, after four fixes made on the
issue itself: the working file's tail had been pasted into the body, the
verification box carried a Cyrillic «х» and so was not a checkbox, a field
label had lost its space, and the two neighbour-row figures had no caption.
Edits to the text from here on have to be made on the issue too, or the two
diverge._

_Destination: an issue in `ScrollPrize/villa`. The repository takes
data-quality reports of this kind (recent precedents in the same tracker:
#1468 on missing `area_vx2`, #1522 on mesh `scale`, #1504 on `derived_from`
pointers)._

## Before sending — what only the owner can do

1. **Attach the four figures.** They are not in the published package (it
   ships the maps; the cards are rendered on demand) and GitHub takes them
   by drag-and-drop into the issue body at the marked places. All four are
   staged in the order they appear, in
   `output/topo/real_paris4/issue_Z0136/`:
   - `1_band_row680.png` — the chain-normal band of the crossing row, the
     decisive evidence (`zones_band_b/Z0136_band.png`, `band_zones_b.py`);
   - `2_axial_slab_card.png` — the axial slab card with the banner overlay
     (`zones_png_b/Z0136.png`, `render_zones_b.py`). Its title line is
     clipped by the render; the caption in the body carries the same facts;
   - `3_band_row678_neighbour.png`, `4_band_row682_neighbour.png` — the same
     band on the two neighbouring rows, the evidence that the crossing is
     local to one row (`band_neighbour_rows.py`, TOPO-062). Optional: the
     body reads without them, and dropping them shortens the issue — but
     then the sentence about rows 678 and 682 is a claim without a picture.
2. **Write the "I was trying to" paragraph in your own words.** villa's AI
   guidelines require human-written commentary on LLM-assisted *PRs*; an
   issue is not a PR, but the same standard is the safer read here, and the
   precedent in this repository is villa#1546, where the first
   machine-written paragraph was caught and rewritten by hand before
   sending. The facts to draw on are listed under the placeholder; the
   sentences must be yours.
3. **Tick the verification checkbox** — it is a personal statement, and it
   is true: the run was on real published scroll data, not a toy example.
4. **Decide whether to cite the benchmark.** The body carries no link back
   to sheet-topo-bench. The package is public since 21.08.2026, so the link
   is now a working reproduction route rather than an advertisement; if you
   want it, the ready line is at the end of this file.

_A third, numeric signature (`band_signature.jsonl`) is deliberately absent
from the issue: it failed its own pre-declared separation rule, and the body
says so below rather than omitting it silently._

---
**Title** (goes in the title box, not the body):

`20260701183124-w010-027` (PHercParis4): row 680 leaves its sheet near
z ≈ 16400–16520 and rejoins on the neighbouring one ~250 vx (~2.4 mm)
further along

**In one sentence:** one row of the published 2026-07-01 trace
`20260701183124-w010-027` crosses between sheets — the sheet it was on drops
away from under it, and ~250 vx later the row is running on the next one.

**I was trying to:**

<!-- OWNER: two or three sentences in your own words. Facts to draw on:
     building a detector for topological errors in surface traces; needed
     to know whether its flagged windows on real 2026 meshes are genuine
     tracing errors, so every candidate was checked by eye against CT and
     the 2023 Grand Prize banner. This one survived. -->

**Using:**

- **Trace:** tifxyz grid of `20260701183124-w010-027`, anonymous HTTPS at
  `https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/segments/20260701183124-w010-027/mesh/intermediate/tifxyz_original/`
- **Frame (all coordinates below):** pyramid **level 2** of
  `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr`
  — 18946 × 8174 × 8174, ≈ 9.6 µm/vx, the grid these meshes are written in
  (×4 for full resolution).
- **Also read:** the recto prediction
  `20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr`
  at level 2 (`L0` is part of the name, not the level read), and the 2023
  banner `20231231235900_GP.obj` mapped into the same frame.
- **Code:** ours, not a villa binary.

**What happened:** rows 678–682, columns 936–951 (inclusive here; the
figure titles print the same window half-open), voxel box x 4784–4896,
y 4578–4937, z 16357–16501 — about 1.1 × 3.4 × 1.4 mm. Row 680 leaves its sheet inside this window and
returns on the neighbouring one. Rows 678 and 682, read the same way, stay
on theirs: the event is one row wide.

<!-- OWNER: optional — attach 3_band_row678_neighbour.png and
     4_band_row682_neighbour.png here -->

**What I expected or needed:** a row should stay on the sheet it started
on; otherwise everything built on the segment — flattening, rendering, ink work — inherits the swap as text
from two layers stitched into one image. What would help, in order: confirm
or reject the read; if it is real, say whether the segment is worth
re-tracing over these columns or just marking; and whether something in your
pipeline already catches this — we could not find such a check.

**Evidence / reproduction:**

**Figure 1 — sideways projection along the row, the decisive one.**

<!-- OWNER: attach 1_band_row680.png here -->

Walk row 680 along columns 912–975 in 1-vx steps and sample ±30 vx sideways
at its own height: CT on top, prediction below. The row itself is the line
n = 0 (magenta), the dashed verticals are the flagged columns, the cyan dots
are banner points within 3 vx of the row. A row on its sheet keeps that
sheet's bright band on n = 0. Here the band leaves n = 0 downward, takes the
banner dots with it and keeps going — 10–15 vx below the row by the right
dashed line; the row line is bare for a stretch (98 vx with no predicted
surface on it, 84 vx of dark CT); then a **different** band arrives at n = 0
and stays. Read twice, `crossing` at high confidence both times: a blind
first pass and an independent control pass at higher zoom.

The claim is about the trajectory, not about prediction being missing
pointwise: a crossing row hugs both sheets, so a pointwise test passes most
of the way (81% of the window here against 93% and 77% on rows 678 and 682)
and separates nothing — our own pre-declared pointwise signature failed on
exactly that and is not offered here.

**Figure 2 — axial slab, corroboration.**

<!-- OWNER: attach 2_axial_slab_card.png here -->

Column 938, z = 16432, ±6 vx slab: the trace steps across CT sheets while
the banner leads the two branches apart. On its own an axial reading can be
a projection artifact — of four axially flagged windows only this one
survived figure 1 — so it corroborates, it does not carry.

**Reproduction:** take row 680, columns 912–975, resample its polyline to
1-vx steps; sample the masked CT and the prediction, both at level 2, at the
row's own height along its in-plane normal, n ∈ [−30, +30] vx; project the
banner into the same frame. Expected: inside the box above the band leaves
n = 0 while the row goes straight, and rows 678 and 682 keep theirs.

- [ ] I personally encountered or reproduced this using the version and data
      stated above.

## Details

Found by a detector we were building: its support channel — trace nodes
with no prediction under them, against an atlas of undisturbed substrate —
ranked this window 224th of 712. All 57 windows that channel credited in
this sector were reviewed by eye (blind pass, then control), and figure 1
left exactly this one confirmed. Two further channels credit it too, CT
thickness (rank 262) and disclination density (rank 52), but they share
inputs with the first two: three data sources, not four.

Scope: one address, not a survey — the only crossing we call confirmed
among the windows we looked at here. Four further candidates from another
channel sit at a lower tier of evidence and are deliberately not in this
report.
---

_Ready line for the citation, if the owner wants it (see the checklist at
the top) — as the last line of the "Details" section:_

> The detector, the corpus and the generator of the band figure are public:
> <https://github.com/tonclap/sheet-topo-bench> (MIT; `verify.py`
> regenerates the maps and tables from the shipped checkpoints).

_The disclosure the receiving project's policy asks for is decided together
with it, at submission time._

_Caveat lifted 21.08.2026 by the re-release (`a888844`): the neighbour-row
figures (3 and 4) come from `band_neighbour_rows.py`, and that caller now
ships with the package alongside `band_zones_b.py`. The sentence in the
posted issue was updated to say so — the file and the issue are kept in
step deliberately._
