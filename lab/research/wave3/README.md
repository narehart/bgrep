# Research wave 3: miss autopsy and D-class case mining

Date: 2026-07-15

## Headline numbers

- **Lexical ceiling: 93.08%** (3591/3858 gold lines). Even if every lexically-visible
  gold line (IDF-filtered against per-repo document frequency) were captured perfectly,
  line-weighted recall tops out at 93.08% — ~6.9% of gold-line mass is structurally
  unreachable by *any* term-matching objective because no query term appears anywhere
  near it in the source. (E7 miss autopsy.)
- **The length-penalty mechanism.** Mining the 26 single-dominant-function D-class miss
  cases: gold is longer than the chosen (returned) function in 25/26 (96%), and **gold is
  >3x longer than chosen in 22/26 (85%)** (median 16x, up to 90x). Median within-file
  idf-density rank of the gold function is **22.5** (of ~50 functions/file) — this is not
  a near-miss population fixable by bumping top-k; the packing objective's implicit bias
  against long-but-genuinely-relevant functions is the dominant, load-bearing failure mode.
  (E13 D-class case mining.)

## Source reports

- [`e7-miss-autopsy.md`](./e7-miss-autopsy.md) — read-only analysis of the corrected
  lexical-visibility recall ceiling (IDF-filtered vs. stopword-only).
- [`e13-dclass-case-mining.md`](./e13-dclass-case-mining.md) — three-stage mining of
  D-class (wrong-function) misses: per-instance D-mass screening (300 instances), a
  feature battery over the top-30 D-mass cases, and aggregation across the 26 clean
  single-dominant-function cases.
- [`data/stage1_dmass.json`](./data/stage1_dmass.json), [`data/stage2_features.json`](./data/stage2_features.json)
  — supporting data from the E13 pipeline (stage3 aggregate rows were not archived here;
  see the report for the aggregated table).

## Derived experiment queue (status as of 2026-07-15)

| Experiment | Description | Status |
|---|---|---|
| E12 | Padding / top-k depth sweep | eval running |
| E14 | Length-normalization of the packing objective (sub-linear token-count penalty) | implementing |
| E15 | Preamble/import-block handling | implementing |
| E16 | Sibling-function inclusion | queued |

## Meta-lesson

Case-mining the actual misses out-produced literature scanning. Wave 2 ran six
prior-literature-driven experiments off SOTA localization/passage-selection papers: five
came back NEG, one NULL. In contrast, the E7 miss autopsy (a few hours of direct,
read-only inspection of what bgrep actually failed to retrieve) immediately surfaced a
concrete, gate-able mechanism — the length penalty — that E13's case mining then
confirmed and quantified with hard numbers (22/26, median rank 22.5). When the literature
isn't converging, look at the misses directly before reaching for another paper.
