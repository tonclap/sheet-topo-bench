# Bug report: sheet crossing in trace `20260701183124-w010-027` (PHercParis4)

_The address-level report for Z0136, the one real 2026 tracing error this
benchmark verified (README, "The address list"). **The issue body is
everything between the two horizontal rules below**, in English; this header
is not part of it._

_Two figures belong with the issue and are **not** in the package — the
package ships the maps, the cards are rendered on demand. Produce them from
the working tree with `render_zones_b.py` and `band_zones_b.py`, or take
them from `output/topo/real_paris4/`:_

- `zones_band_b/Z0136_band.png` — the chain-normal band, the decisive evidence
- `zones_png_b/Z0136.png` — the axial slab card with the banner overlay

_A third, numeric signature (`band_signature.jsonl`) is deliberately absent:
it failed its own pre-declared separation rule, and the report says so below
rather than omitting it silently._

---

## Title

`20260701183124-w010-027` (PHercParis4): row ~680 leaves its sheet and runs
~250 vx through unsupported dark CT near z ≈ 16400–16520 (sheet crossing)

## Summary

While building a topological-error detector for surface traces
(sheet-topo-bench), we found — and verified in two independent projections —
one location where the 2026-07-01 GP trace `20260701183124-w010-027`
crosses between sheets instead of following its own lamina.

The decisive evidence is the chain-normal band (evidence 1 below), read
blind twice. Three automated channels also credit this window — surface
support (rank 224 of 712), radial CT thickness (262) and CT normal-field
disclination density (52) — but they are corroboration, not proof, and they
draw on three data sources between them, not four: support reads the recto
prediction that the band also uses, thickness and disclination both read the
CT volume, and the 2023 banner is the third, independent one.

- **Segment:** `20260701183124-w010-027` (PHercParis4, windings 10–27)
- **Mesh window:** rows 678–683, cols 936–952 of the tifxyz grid
  (row step ≈ 19 vx); the crossing is a *row-local* event centred on
  row 680 — adjacent rows follow the sheet correctly.
- **Voxel box of the flagged zone (level 0):** x 4784–4896, y 4578–4937,
  z 16357–16501.
- **Band extent (level 0):** the projection below is rendered over the wider
  chain segment cols 912–976, z 16399–16524 — hence the z range in the title,
  which describes the crossing along the band rather than the zone box.

## Evidence

1. **Chain-normal band projection (attached, `Z0136_band.png`).** The
   valid nodes of row 680 on cols 912–976 form a chain, resampled to 1-vx
   arc steps; CT and the recto prediction
   (`recto-2um-ps256-L0-th0.45`) are sampled along the in-plane normal
   (±30 vx) at chain height. In this projection a correctly traced row
   keeps its sheet's bright band on n = 0. Here, just past the left edge
   of the flagged window the recto band departs downward from n = 0
   together with the 2023 GP banner points (|Δz| ≤ 3 vx witness), the
   chain then runs ~250 vx of arc through dark CT with no recto support
   (trace in the inter-sheet gap), and at the right edge a *different*
   band arrives at n = 0 — i.e. the row rejoins on the neighbouring
   sheet. Read blind twice (independent first pass and control pass),
   both `crossing high`.
2. **Axial slab card (attached, `Z0136.png`).** In the ±6 vx slab of the
   central row the trace steps across CT laminae while the banner leads
   the two branches separately. (On its own the axial reading can be a
   projection artifact — of four axially-flagged zones only this one
   survived the band check; the band projection is the decisive one.)

A third, numeric witness was attempted and is **not** offered here: a
pointwise "no recto support and dark CT" signature along the band. It failed
its own pre-declared separation rule — 31 vx of unsupported run at Z0136
against 59 vx at Z0163, a zone the band check had already read as *not* an
error — so the evidence for this report is the two projections above, each
read blind twice, and nothing else.

## How it was found

The detector's *support* channel (surplus of trace nodes lacking recto
prediction support, differenced against a pristine-substrate atlas) ranked
this window 224 of 712 on the 178-zone corpus over the 2023-banner sector;
57 zones got support credit, and manual double-blind review of all 57
followed by the band check left exactly this one confirmed crossing. So the
find is machine-flagged, human-verified — and the two witnesses are
independent (the zone flagging never reads the prediction the support
channel uses).

## Reproduction

- Trace: `paths/20260701183124-w010-027` of
  `full-scrolls/Scroll1/PHercParis4.volpkg` (dl.ash2txt.org), tifxyz grid.
- Take row 680, cols 912–976; resample the polyline to 1-vx arc; sample
  the masked CT volume and `recto-2um-ps256-L0-th0.45` at chain height
  along the in-plane normal, n ∈ [−30, +30] vx; compare with the 2023 GP
  banner (`20231231235900_GP.obj`) projected into the same frame.
- Expected: the recto/banner band leaves n = 0 inside x 4784–4896,
  y 4578–4937, z 16357–16501 while the chain continues straight;
  neighbouring rows (678, 682) stay on the band.

---

_The issue body carries no link back to this benchmark: whether to cite
sheet-topo-bench is decided at submission time, together with the AI
disclosure the receiving project's policy asks for._
