# Research wave 4: post-adoption miss autopsy

Date: 2026-07-18

## Headline numbers

- **D-class remains the pool.** Re-running the E7 miss autopsy against the now-adopted
  engine (guarded padding `--pad-lines 5` + length-normalization `--len-exp 0.85`) shows
  all five miss categories moved in the expected direction (A up, B down, C up, D down, E
  flat) on both datasets — but D (wrong-function-in-file) is still by far the largest
  fixable pool: **27.8pp fixable on Lite, 33.0pp fixable on Verified**. The adoption
  helped, but did not come close to closing the gap it targets.
- **Adoption's out-of-sample strength is ~60-65% of its in-sample per-class effect.**
  Comparing the pre/post deltas measured on Lite (the tuning set) against the matched
  pre/post pair on Verified (the held-out set), each per-class delta on Verified retains
  roughly 60-65% of the magnitude seen on Lite (e.g. D: −6.29pp Lite vs −3.21pp Verified;
  B: −2.72pp Lite vs −3.35pp Verified is an exception running the other way) — a real but
  partial generalization, not an artifact of tuning-set overfit, and not a free repeat of
  the full in-sample gain either.
- **The 0.85 length exponent is under-damped for the long tail.** Case mining the worst
  remaining D-class remnants shows 5 of E13's 8 worst cases (length ratios 3-45x) are
  essentially unmoved by `--len-exp 0.85` — a mild sub-linear exponent measurably helps
  the moderate cases but doesn't reach the extreme tail. This points at either a lower
  exponent or a floor-based alternative penalty form, not a sign flip.
- **New phenomena surfaced post-adoption**: a degenerate header-fallback failure mode (in
  `pytest-7490`), chronic hub-file attractors that win file-selection on generic
  vocabulary alone (e.g. `core/expr.py`, `application.py`), and a material Verified-only
  weakness — **E (missed-file) lexical-visibility on Verified is only ~60%**, well below
  Lite's ~80%, meaning the never-attacked file-selection front is measurably harder to
  reach via pure lexical signal out-of-sample than the tuning set suggested.

## Source reports

- [`e7b-post-adoption-autopsy.md`](./e7b-post-adoption-autopsy.md) — full read-only
  A-E miss autopsy of the adopted engine, run against both the Lite tuning set (300
  instances) and the Verified held-out set (406/407 usable instances), plus a matched
  pre-adoption run on the same Verified set for a true apples-to-apples pre/post
  comparison.
- [`data/miss_autopsy_raw_lite.json`](./data/miss_autopsy_raw_lite.json),
  [`data/miss_autopsy_raw_verified.json`](./data/miss_autopsy_raw_verified.json),
  [`data/miss_autopsy_raw_verified_old.json`](./data/miss_autopsy_raw_verified_old.json)
  — per-line/per-instance raw classification records backing the report (all under 5MB).

## Dual-dataset gating discipline (now in force)

Prior waves gated experiments on Lite (300 instances) alone. This autopsy is the first to
run the full A-E classification on both Lite *and* the Verified held-out set side by side,
and it found the two datasets agree on direction but disagree on magnitude (D's fixable
pool is bigger on Verified than Lite; E's visibility floor is much lower on Verified).
Going forward, any experiment claiming a fix to D, B, or E must be gated on **both**
datasets before being called a win — a Lite-only gate is no longer sufficient evidence
that an effect generalizes, and a result that only shows up on Lite should be treated as
a tuning-set artifact until confirmed on Verified.

## Ranked experiment queue

1. **E14b — deeper length-exponent sweep** (in flight). Targets the pure-length subset of
   D's remaining pool. `--len-exp 0.85` measurably helped but left the 3-45x-ratio tail
   essentially untouched; sweep lower exponents (0.7/0.6) or an additive-floor penalty
   form, regated on both Lite and Verified since over-correcting length is a real risk
   (a small number of cases correctly prefer the shorter function).
2. **E16 — sibling-class/same-class locality boost within an already-anchored file**.
   Targets D directly (the ~27.8pp Lite / ~33.0pp Verified fixable pool, the single
   largest remaining bucket on both datasets) via a bounded, same-class-only marginal-gain
   bonus in `pack_regions`' greedy loop — smaller blast radius than the reverted `w_name`
   symbol-weighting experiment.
3. **E15b — preamble/import-block force-include, with the combo2 eviction lesson applied**.
   Targets a narrow, near-zero-cost slice of D (E13 measured ~13.3% of D-mass repo-wide
   from missing import/constant preambles, still unaddressed post-adoption, confirmed in
   both `sphinx-11445` and the newly-found `pytest-7490` degenerate-fallback case). Must
   apply the eviction-safety lesson learned from the `e-combo2` worktree (force-including
   a preamble must never crowd out an already-selected higher-value region) before this
   is gate-ready.

## Meta-lesson

The adopted padding + length-normalization change is a real, validated, direction-correct
win (E7 -> E7b: A up, D down, on both datasets) — but it is a partial fix to a problem
that is still dominated by the same D-class mechanism E13 first identified in wave 3.
Nothing in this autopsy changes the ranked priority from wave 3 (D is still the pool);
it does add two hard constraints for wave 5: gate on Verified as well as Lite, and treat
the length-exponent fix as under-damped for the tail rather than as a closed chapter.
