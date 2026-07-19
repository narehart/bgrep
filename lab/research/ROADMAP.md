# Experiment roadmap from the 2026-07 arXiv mining

Convergent #1 (both reports): NEIGHBORHOOD-FIRST retrieval for large repos
(anchor with rare-term hits -> structural expansion -> rank within region).
Experiment in flight. Targets: monorepo dilution, JS/TS.

Queued, evidence-ranked:
1. Pandora-index global expansion stopping (replaces per-source caps; prophet
   1/2-guarantee; TASR pattern) — targets cap starvation (JS/TS dominant lever).
2. Matroid slack reallocation (soft per-source floors + pooled marginal-gain
   budget) — composes with 1.
3. Heat-kernel diffusion small-t + inverse-only edges for hub-heavy repos —
   principled version of our empirical hub fixes; targets @1.
4. CHA interface->implementation index for Go — gated on Go miss study
   confirming interface-dispersion class.
5. MVT patch-leaving rule for packing depth (leave a file when marginal
   relevant-tokens-per-token < running average) — next packing iteration if
   survival-scheduling underdelivers.
G4 (NL-only issues): field-wide ceiling externally confirmed (arXiv:2507.18319)
— attack indirectly via space-shrinking (#neighborhood), no silver bullet.

## Status update (2026-07, #4 campaign ledger)

The region campaign (#4) has since tested the literature-prior direction this
roadmap encodes, and it lost: every literature-derived packing/selection
mechanism tried on the exact 300-instance harness came back negative or null
(E1 block-boundary packing NEG, E2 adaptive stopping NEG, E3 salience NEG,
E5 chunk-rank NULL, plus w_name NEG/reverted) — 0-for-6 against priors. The
adopted wins (guarded span padding `--pad-lines 5` + sub-linear length
normalization `--len-exp 0.85`, engine defaults since PR #40) came from miss
autopsies and case mining (E7/E13), not from this roadmap. The queued items
below remain UNVALIDATED leads, kept for the record; neighborhood-first
shipped earlier as a flag-off experiment (see lab/README.md).
