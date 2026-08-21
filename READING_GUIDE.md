# Reading guide — how to check any number in this package

The README and every **generated** table are in English — the held-out
summary, the ablations, the verification tables, the zone classifications.
Their generators emit English, and `verify.py` regenerates each one from the
run reports, so the translation is checkable rather than asserted.

**The hand-written records stayed Russian**: the protocol, the two freeze
records, the corpus and results write-ups, the by-eye criteria, the roads not
taken, and the dated inserts appended below some generated tables. Those are
the working artifacts of the line that produced them and were never rewritten
for publication — rewriting them after the fact would have broken the one
property that makes them worth shipping: every file is the record the run
actually wrote, at the time it wrote it. A translation of a declared rule is
a retelling of it, and nothing here could check such a retelling the way
`verify.py` checks a number.

This guide is the bridge to that layer. With the glossary below, every table
in every cited file is readable without Russian: the tables are numbers, and
the column headers, verdict words and reading rules are all listed here.

If you only want to confirm the headline numbers, you do not need this file at
all — run `python verify.py` (seconds, no network) and it recomputes them from
the shipped run reports.

---

## Glossary — the words that appear in the tables

| Russian | English | Note |
|---|---|---|
| проверка | check | |
| результат | result | |
| критерий | criterion | the rule declared *before* the run |
| вердикт | verdict | |
| выборка | sample / corpus | |
| правдоп. / правдоподобные | locally plausible | injections that look normal locally — the hard subset |
| полоса (витков) | winding band | e.g. `w100-109` |
| худшая полоса | worst band | the per-band floor reported next to every headline recall |
| зона / зоны | zone(s) | a rectangle of the node grid on a real corpus |
| инъекция | injection | a planted error with known position |
| пул | pool | the candidate set a detector generated |
| ранг | rank | |
| масса | mass | cluster size, the evidence weight of a zone |
| сид | seed | |
| дев / дев-корпус | dev (development) corpus | |
| перенос | transfer | does a gain measured on dev survive off it |
| слияние | fusion | combining detector channels |
| перестановки | permutations | the label-permutation leakage control |
| блочный сплит | block split | the harsher, contiguous-block validation split |
| потолок | ceiling | oracle value, the best achievable |
| пол | floor | random baseline |
| база / случайная база | baseline / random baseline | printed next to every number |
| значимо | **significant** | the paired bootstrap interval excludes zero; the compact V2 rows of `VERIFY_V5.md` abbreviate it to ` sig.` |
| незначимо / н.з. | not significant | |
| устойчиво | stable | survives the sensitivity sweep |
| стабильна / частично устойчива / порого-зависима | stable / partly stable / threshold-dependent | the three declared verdicts of the atlas sensitivity sweep, written by the run into `atlas_sensitivity.json`; `ATLAS_SENSITIVITY.md` reads them in English |
| кратно | by a multiple (≥ 2×) | the owner's word in the TOPO-034 task; `band_signature.py` §5 turns it into the declared separation rule: Z0136's signature ≥ 2× each of the other three |
| исход (а) / (б) / (в) | outcome (a) / (b) / (c) | the pre-declared possible answers of a question-task; **a negative outcome still counts as the task being done**. Note the letters are Cyrillic: **`в` is the third one, (c)** — it is not a Latin `B`, which is the easy misreading |
| СИГНАЛ | SIGNAL | above the random baseline by the declared rule |
| случайность | at chance | indistinguishable from the random baseline |
| ПРОЙДЕНО / PASS | passed | |
| НЕ ПРОЙДЕНО / FAIL | failed | |
| правило отгрузки | shipping rule | significant AP gain **and** no significant loss on any family; the ablation summaries print its verdict as **PASSED** / **FAILED** — one pair of words in all five reports (ABLATION_V4/V5/V6/V6S/V7), unified 21.08.2026 |
| врезка | dated insert | a correction added in place, never a rewrite of what it corrects |
| гейт | gate | a precondition that had to pass before the run counted |
| замороженный | frozen | code/parameters fixed before the corpus was opened |
| чувствительность | sensitivity | the same measurement at a different threshold |
| ошибка / реальная ошибка | error / real (non-synthetic) tracing error | |
| подтверждена / не подтверждена | confirmed / not confirmed | "not confirmed" is explicitly **not** "refuted" |
| сессия N | working session N | the line was run in numbered sessions; the number dates a record relative to the others and carries no other meaning |
| владелец | the owner | the human who makes the irreversible decisions — what to freeze, what to publish, what to spend. Records name them where a step waited on a decision rather than on a result |

Four more conventions the run records use without introducing themselves:

- **`TOPO-nnn`, `U-nnn`, `DEV-nnn`** are this project's own task identifiers.
  Each names one declared step — a question asked, its rule, and its outcome —
  and they appear so that a result can be traced back to the step that
  declared it. They point at nothing outside this project, and no number
  depends on them; read them as footnote markers.
- **`TARGET.md`, `NEXT.md`, `STATUS.md`, `LINES.md`** are the project's own
  planning files. They are mentioned where a record explains why a step was
  taken, and they are deliberately **not** part of this package: they hold
  scheduling, not evidence. Nothing in the package depends on them.
- **`wave 2`** is the earlier line of this project — the winding-number and
  laminae measurement that preceded the benchmark; two of its modules ship
  here under `winding/`, and the detector reuses its global frame. Where a
  docstring says "the wave-2 lesson" it means one specific thing: wave 2
  published three table rows generated by a script that was not shipped with
  them, and nobody could tell until the script was gone. Every generated
  table and summary sentence in this package is produced by shipped code for
  that reason, and `verify.py` regenerates them.
- **`note` and `note_en`** appear side by side in every by-eye label CSV.
  `note` is the observation as the labeller wrote it, in Russian, at
  labelling time — the record, never rewritten. `note_en` is its English
  rendering, added afterwards as a separate column; the generated tables
  render `note_en`. Where the two could ever disagree, `note` is the one that
  was there when the class was decided.
- **`Znnnn`** is a zone identifier on a real corpus — a fixed rectangle of the
  node grid, stable across every file that mentions it (so `Z0136` is the same
  place in the label CSVs, the zone tables and the bug report).

## Metrics

- **AP** — average precision of the ranked window list against the planted
  injections. Hit rule: the window centre falls inside the error rectangle
  grown 50% per axis; one injection is credited once, best rank wins
  (`PROTOCOL.md` §3).
- **recall@N** — share of injections found within the first N windows, where
  N is the number of injections in that corpus.
- **S / M / H** — the three error types: **S** = sheet switch (the trace jumps
  to the neighbouring winding), **M** = merger (two windings fused into one
  surface), **H** = hole (a piece of surface lost). Per-family columns are
  recall within that family.
- **ΔERL@N** — millimetres of ERL (expected run length: how far along a sheet
  you get before a topological error) recovered by cutting at the detector's
  top N windows, against the oracle ceiling (cuts at the true positions) and
  the random floor. "(86% of oracle)" in a table row = 86% of the oracle
  ceiling; the Russian layer writes the same thing as "86% оракула".
- **[95% CI]** — bootstrap by injection, 2000 resamples, seed 20260815.
  **Paired** deltas use the *same* resamples for both rows, which is why a
  paired delta can be significant while the two rows' own intervals overlap.

## One path substitution

The `Usage:` line in each shipped script's docstring was written in the
working repository, where the run reports live under `output/topo/`. This
package ships the same reports read-only under `runs/topo/`. Substitute that
one path; the flags and the file names are unchanged. (`verify.py` and
`transfer_breakdown.py` find the shipped location themselves.)

## Which file answers which question

| File | What it holds | Language |
|---|---|---|
| `README.md` | the whole story, every headline number | English |
| `verify.py` | recomputes every README number offline | code |
| `protocol/PROTOCOL.md` | the evaluation protocol, declared before any run; §3 hit rule, §6 held-out discipline, §7 baselines, §8 per-band reporting | Russian |
| `pipeline/HELDOUT_RESULTS.md` | the headline table and the v5lu exam, with paired deltas | English, generated |
| `pipeline/detector/ABLATION_V2..V4.md` | the three rank-fusion attempts that did **not** ship | English, generated |
| `pipeline/detector/ABLATION_V5.md` | the learned fusion that did ship, plus the transfer breakdown | English generated table + Russian dated inserts |
| `pipeline/detector/ABLATION_V6/V6S/V7.md` | the three assaults on the generation ceiling, all negative | English generated tables; V6 also carries a Russian dated insert |
| `pipeline/detector/VERIFY_V5.md` | the eight pre-exam checks (permutation, block split, λ sweep, attribution, cross-scroll transfer, corpus B) | English, generated |
| `pipeline/detector/VERIFY_V8.md` | the fresh-seed end-to-end rebuild | English, generated |
| `pipeline/detector/ATLAS_SENSITIVITY.md` | what the substrate atlas does when the contact threshold moves; the source of the "publish a range, not *the* defect map" verdict | English, hand-transcribed from `atlas_sensitivity.json` (its header says so; the script has no markdown renderer) |
| `pipeline/real_errors/CORPUS.md` | how each real corpus was built; every rule declared before its run | Russian |
| `pipeline/real_errors/RESULTS.md` | all real-corpus results, positive and negative | Russian |
| `pipeline/real_errors/ZONE_CRITERIA.md`, `ZONE_CRITERIA_B.md` | the by-eye classification criteria, committed before the first zone card was rendered — the "rule declared before the run" artifact for the manual passes | Russian |
| `pipeline/real_errors/ZONES_*.md`, `*_labels*.csv` | the by-eye classifications, every zone | English tables; the CSVs keep the original `note` beside an English `note_en` |
| `pipeline/real_errors/BUGREPORT_Z0136.md` | the address-level report for the one confirmed real error; the issue body itself is English | English body, Russian notes around it |
| `protocol/FREEZE_2026-08-14/19.md` | what was frozen before each held-out generation was opened, and the incidents during the runs | Russian |
| `protocol/UNTRIED.md` | roads not taken and the condition under which each would be | Russian |

**The shipped code is English, and so is what it prints.** Every docstring
and comment in the package is English — a script's docstring is where its
protocol is declared, so that layer had to be readable on its own — and the
generators emit English tables: `heldout_summary.py`, the
`ablation_summary*.py` family, `verify_v5/v8.py`, `transfer_breakdown.py`,
`summarize_zones*.py`. The only Russian left in a `.py` file is a declared
verdict token quoted in the docstring that declares it
(`atlas_sensitivity.py`, `band_signature.py`): those tokens are what the runs
wrote into their shipped reports, so translating them would have made code
and report disagree. Both are glossed above.

## The three reading rules that explain most verdicts

1. **A baseline stands next to every number.** A number without its baseline
   was not considered evidence, and several claims died precisely here — see
   the "what did not survive" section of the README.
2. **The rule is declared before the run.** Protocols, thresholds and verdict
   rules are committed first; when a result then failed its own rule, the
   negative shipped rather than the rule moving.
3. **Corrections are inserts, not rewrites.** A wrong reading stays where it
   was, with a dated insert next to it. So a file can contain a claim *and*
   its later withdrawal — the withdrawal is the current state.
