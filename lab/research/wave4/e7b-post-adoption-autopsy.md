All analysis is complete and read-only-verified (only pre-existing `.roust/` untracked artifacts present, same as E7 noted — no checkouts/mutations from this analysis; the copied verified-heldout parquet is git-ignored and not tracked). Scripts and raw data live in `/tmp/e7b_scratch/` (not committed, per instructions).

---

# E7b — Post-Adoption Miss Autopsy

Read-only analysis, same A–E taxonomy and method as `lab/research/wave3/e7-miss-autopsy.md`, run against the **adopted** engine (guarded padding `--pad-lines 5` + length-normalization `--len-exp 0.85`, commit `5e81c8a`).

**Data**: Lite = `lab/results_regions/full300_v11.jsonl` (300 instances, 0 errors, 3858 gold lines) vs. Verified held-out = `lab/results_regions/full407_verified_new.jsonl` (407 instances, 1 timeout — `django__django-11603` — excluded, 406 usable, 9381 gold lines). As a bonus rigor check I also classified `lab/results_regions/full407_verified_old.jsonl` (pre-adoption engine on the *same* held-out set, 0 errors, 9393 gold lines — 12 more because the timeout instance succeeded pre-adoption), giving a true matched pre/post pair on Verified, not just an aggregate comparison. Gold hunks via `parse_gold_hunks` from `lab/swebench_lite.parquet` / `lab/swebench_verified_heldout.parquet` (the latter copied in read-only from the `verified-region-plumbing` worktree since `lab/*.parquet` is gitignored and wasn't present in this working tree — flagged, not a mutation). Source read via `git -C lab/swebench_repos/<repo> show <base_commit>:<path>`. Lexical-visibility ceiling reused the E7 per-repo IDF table (`repo_idf.json`, HEAD-snapshot, same 12 repos, still valid — vocabulary/DF is stable across nearby commits, same approximation E7 flagged).

## 1. A–E distribution: what did adoption consume?

| Category | Lite **pre** (E7, v10) | Lite **post** (v11) | Δ | Verified **pre** (old) | Verified **post** (new) | Δ |
|---|---|---|---|---|---|---|
| A captured | 40.28% | **47.17%** | +6.89 | 33.08% | **37.38%** | +4.30 |
| B missed-near | 13.97% | **11.25%** | −2.72 | 18.05% | **14.70%** | −3.35 |
| C missed-same-fn | 0.31% | **2.44%** | +2.13 | 0.42% | **2.65%** | +2.23 |
| D wrong-fn-in-file | 38.15% | **31.86%** | −6.29 | 40.84% | **37.63%** | −3.21 |
| E missed-file | 7.28% | **7.28%** | **0.00** | 7.62% | **7.63%** | +0.01 |

**All five deltas point the same direction on both datasets** (A↑, B↓, C↑, D↓, E flat) — this is the clean signature of a padding + length-normalization change: spans get slightly wider (converts near-misses in B into A, and pulls some far-D lines into the *same, already-anchored* function → C), and less-aggressive length penalty lets genuinely long correct functions compete (shrinks D). **E is untouched to the third decimal on Lite (281/3858 both runs, exactly) and near-identical on Verified (716/9381 vs 716/9393)** — file selection is a completely separate stage from region-packing, and this change never touched it, confirmed at both the aggregate line level *and* the per-instance level (23/300 Lite / 15/407 Verified "gold file missing entirely" counts are identical pre→post on both datasets) *and* even within each query-type stratum (see §6) — E% is invariant to the fourth digit across every cut we sliced.

Consumption asymmetry: **B shrank more than C grew** (Lite: −2.72 vs +2.13; Verified: −3.35 vs +2.23) — i.e. most of B's mass converted straight to A (full capture), not merely to C (same-function-but-still-missing-part). **D absorbed the largest share of the gain** (Lite −6.29pp, Verified −3.21pp) but remains far the largest miss class by a wide margin in both.

## 2. Does Verified match Lite's structure?

**Yes, directionally, with two material differences.**

**Match**: the adoption-effect vector (A↑ B↓ C↑ D↓ E≈0) replicates exactly in sign across both datasets (§1), and the query-type ranking is preserved (has-code-block queries are worst, has-traceback/has-quoted-identifier best, prose-only has the worst E-rate — see §6, unchanged from E7's original Lite finding). C's near-invisible-to-real jump (0.3%→2.4%ish) also replicates almost exactly in magnitude (Lite +2.13pp, Verified +2.23pp) — a genuinely repo-independent mechanical artifact of wider spans, not an overfit-to-Lite quirk.

**Differences, both real**:
- **Magnitude of the gain is smaller out-of-sample.** ΔA is +6.89pp on Lite vs +4.30pp on Verified; ΔD is −6.29pp vs −3.21pp. The fix generalizes in direction but not in full strength — expected ceiling-effect/overfit shrinkage, but worth knowing the "true" out-of-sample gain is roughly 60–65% of the in-sample gain.
- **E's lexical visibility is structurally worse on Verified** (60.1% visible vs Lite's 80.4% — see §5). This is *not* an adoption artifact (identical 60.1% in both verified_old and verified_new) — it means the held-out set's file-ranking misses are intrinsically harder / more lexically opaque than Lite's, a genuine population difference the tuning loop never saw.
- Verified's B-distance histogram is *more* front-loaded than Lite's (§4) and its per-instance capture redistribution took a different path (§3) — worth noting even though the net effect matches.

## 3. Bimodality: did adoption flatten it or deepen it?

| | 0% capture | strictly-between | 100% capture |
|---|---|---|---|
| Lite pre (E7) | 143/300 (47.7%) | 50/300 (16.7%) | 107/300 (35.7%) |
| Lite post | 124/300 (41.3%) | 48/300 (16.0%) | **128/300 (42.7%)** |
| Verified pre (old) | 168/407 (41.3%) | 132/407 (32.4%) | 107/407 (26.3%) |
| Verified post (new) | 154/406 (37.9%) | 108/406 (26.6%) | **144/406 (35.5%)** |

Bimodality **persists post-adoption on both datasets** — it did not flatten into a smooth distribution. But the *mechanism* of redistribution differs by dataset, which is itself informative:

- **On Lite**, the mid-band barely moved (50→48). Nearly the entire gain (Δzero=−19, Δfull=+21) is instances **flipping straight from 0% to 100%** capture, bypassing partial. This is consistent with single-gold-hunk instances (very common in Lite, median instance has 1 gold file/hunk) where the fix either lands the one relevant function or it doesn't — there's no "partial" state to pass through.
- **On Verified**, the mid-band shrank substantially (132→108, Δ−24) alongside a modest zero-shrink (Δ−14) — i.e. a good chunk of the full-capture gain (+37) came from **already-partial instances completing**, not just zero-instances flipping. Verified instances tend to have more gold hunks/files per instance (multi-hunk fixes more common), so there's more room for genuine "deepening" rather than binary flips.

So: **mostly converting zero→full on Lite, a mix of zero→full and partial→full on Verified** — adoption is not simply "deepening already-good instances"; it's disproportionately rescuing previously-hopeless ones, but the *path* to that outcome differs with gold-hunk cardinality.

## 4. The new dominant miss class

**D remains dominant, unambiguously, on both datasets** (Lite 31.86%, Verified 37.63% — vs. B 11.25%/14.70%, E 7.28%/7.63%, C 2.44%/2.65%). Nothing has overtaken it; it just shrank from being even-more-dominant.

**D-remnant case mining** (top-8 Lite instances by D-mass, `pallets__flask-5063`, `django__django-13265`, `scikit-learn__scikit-learn-25500`, `sympy__sympy-20322`, `sympy__sympy-20639`, `sphinx-doc__sphinx-11445`, `pytest-dev__pytest-7490`, `sympy__sympy-14308` — the gold-line count is *identical* to E13's original top-D-mass ranking order for 5 of these 8, meaning **the len_exp=0.85 fix did not flip a single one of E13's five original vignette cases**, it just reduced everyone's length-ratio penalty uniformly):

- **Still length-dominated in 6/8**: `flask-5063` (gold `routes_command`, 47 lines, beaten by chosen 12- and 5-line functions, ratio 3.9×/9.4×), `django-13265` (gold `generate_created_models`, 174 lines, beaten by five shorter siblings, ratios 6.7×–43.5×), `sympy-20322`, `sympy-20639` (gold `_print_nth_root`, 44 lines, still losing to five ~6–9-line `_print_X` stubs), `sympy-14308` (same file, same story, ratios 2.1×–12.7×). **Damping the exponent to 0.85 clearly wasn't enough for the worst cases** — these have length ratios in the 3–45× range that a mild sub-linear exponent barely dents.
- **Genuinely new pattern, not length**: `scikit-learn-25500` — gold (`transform`, 36 lines) actually loses to a **longer** competitor (`isotonic_regression`, 53 lines, ratio 0.68× i.e. chosen is bigger). The query's problem statement is literally about `IsotonicRegression`, and the competing function's name is an exact substring hit — density from an exact-name match overwhelms length entirely here. This is the flip side of E13's "boilerplate stub wins" story: an exact-name match can win even against the length penalty, in either direction.
- **Import-preamble pattern still unaddressed**: `sphinx-doc__sphinx-11445` — same instance as E13's vignette #4, gold lines still have no enclosing function (module-level), still not force-included. E13 flagged this as a "near-free, unaddressed" fix; confirmed still unshipped.
- **A "single-span degenerate" sub-case, not previously named**: `pytest-dev__pytest-7490` — the engine returns **exactly one span for the file: lines 1–23**, which is the raw import header, not even a real content match — the gold function (`pytest_runtest_setup`, lines 232–244) scored so low nothing else was selected at all, and a bare "grab the top of the file" fallback fired instead of any function pick. Distinct from "wrong function chosen" — this is "no function scored, fell back to file header."

**E-case mining** (top-8 Lite instances by E-mass: `astropy-14182`, `sympy-18087`, `django-11283`, `sympy-13895`, `sphinx-8273`, `sphinx-8627`, `sympy-13146`, `sympy-21379`) — **"what does the gold file lose to?"**: a small set of **generic "hub" files recur as the winner across multiple, unrelated instances in the same repo** — `sympy/core/expr.py` wins in `sympy-18087`, `sympy-13895`, `sympy-21379`; `sympy/core/function.py` wins in `sympy-18087`, `sympy-13146`, `sympy-21379`; `sphinx/application.py` wins in `sphinx-8273` and `sphinx-8627`; `astropy/io/ascii/core.py` beats the sibling gold file `rst.py` by loose overlap alone. These are large, generically-named, vocabulary-diverse files (core/base classes, dispatch modules) that share incidental term overlap with almost *any* domain query in that repo, functioning as **file-level attractors** analogous to E13's function-level "boilerplate stub" story, one level up the hierarchy. In the most extreme case (`sympy-13895`) the gold file's own fix-site window has **zero** overlapping query terms at all (genuinely invisible there), while `core/basic.py`/`simplify.py` win on 2–3 generic terms like "simplify"/"subs"/"expression" that appear in dozens of files.

## 5. Ceiling math and per-class headroom

| | Lite | Verified |
|---|---|---|
| Ceiling (fraction lexically visible) | 93.08% (unchanged from E7 — same gold lines/query terms) | 90.37% |
| Invisible floor | 6.92% (unchanged from E7) | 9.63% |
| Mean per-instance capture (post) | **51.68%** | **47.41%** |

Matches the task's expected `~0.474–0.517` and `6.9%` floor exactly.

**Per-class fixable pool** (= class's line-share × its own visibility rate — i.e. what's left on the table if that class's misses were perfectly resolved by a ranking fix, assuming the term is visible somewhere nearby):

| Class | Lite share | Lite visible | Lite fixable-pool | Verified share | Verified visible | Verified fixable-pool |
|---|---|---|---|---|---|---|
| B | 11.25% | 93.8% | **10.55%** | 14.70% | 92.0% | **13.53%** |
| C | 2.44% | 100.0% | 2.44% | 2.65% | 89.6% | 2.38% |
| D | 31.86% | 87.3% | **27.81%** | 37.63% | 87.6% | **32.97%** |
| E | 7.28% | 80.4% | 5.86% | 7.63% | 60.1% | 4.58% |

**D is still, unambiguously, the single biggest fixable pool on both datasets** — and it's *bigger* on Verified (32.97% vs 27.81%) than on Lite, meaning the length/padding fix generalized less completely out-of-sample and left more of D's pool on the table there than the tuning-set numbers alone would suggest. B is the second-largest and cheapest pool (its distance histogram remains front-loaded and got *more* front-loaded — Lite 1–5-line bucket 34.8% of B post vs 31.7% pre; Verified even higher at 40.8%/42.3% — more padding/window-widening still has room to run before diminishing returns bite). E's fixable pool actually *shrank* in relative terms on Verified specifically because its visibility floor dropped to 60.1% (vs Lite's 80.4%) — a genuine reduction in how attackable E is out-of-sample via pure lexical signal, independent of any engine change.

## Ranked next experiments

1. **Sibling-method / same-class locality boost within an already-anchored file** (targets D, ~27.8pp Lite / ~33.0pp Verified fixable pool, the single largest remaining bucket on both datasets). E13's #2 recommendation, never shipped — the case mining here shows 5 of E13's 8 worst D-remnant cases are completely unmoved by the len_exp=0.85 fix (their length ratios, 3–45×, are too large for a mild sub-linear exponent to overcome). A bounded, same-class-only marginal-gain bonus (smaller blast radius than symbol-name weighting, which already regressed once) is the natural next lever. *Effort*: medium — needs the same full-300 + Verified-406 gate as every prior E-numbered experiment; a Rust-side change to `pack_regions`' greedy loop, not a CLI flag sweep.

2. **Stronger length-exponent sweep beyond 0.85, or a floor-based alternative** (targets the pure-length subset of D remnants, roughly the same pool as #1 but a cheaper knob to turn first). The case mining shows 0.85 measurably helped (D −3 to −6pp) but left the worst cases (3–45× length ratios) essentially untouched — this suggests the exponent is under-damped for the tail, not mis-signed. Try 0.7/0.6, or an additive-floor form (`gain/(tok+k)`), swept the same way len_exp=0.85 itself was gated. Cheap to try (it's an existing CLI flag, `--len-exp`), but must be regated on both Lite and Verified since scikit-learn-25500 shows length isn't *always* the right direction to push (a case where a longer function correctly loses is rare but exists, so over-correcting is a real risk). *Effort*: low (flag sweep + existing harness), *risk*: needs the discipline this project already applies (E12/E14-style gate) since a second aggressive length-normalization pass could interact non-linearly with the first.

3. **Import/module-preamble force-include** (targets a narrow but clean, near-zero-cost slice of D — E13 measured 13.3% of D-mass repo-wide from this pattern alone, and it's confirmed still unaddressed in both `sphinx-11445` and the newly-found `pytest-7490` degenerate-fallback case). Force-including a returned file's first ~20–30 lines (or, more precisely, its top-level import/constant block via a cheap AST pass) regardless of term density is nearly free in tokens and has no plausible failure mode analogous to `w_name`'s regression (it can only ever add a small, low-density preamble to files already selected, never change file-selection or crowd out an existing pick). *Effort*: low — single, well-scoped Rust change, still needs the standard full-dataset gate but the blast radius is small enough that a negative result would be genuinely surprising.

**Honest null / not recommended for this round**: E (file-selection) is still the "never-attacked front," and this autopsy's hub-file finding (`core/expr.py`, `application.py`, etc. as chronic generic-vocabulary attractors) is real but diffuse — no clean single feature discriminates gold-file-wins-file-contest the way length dominated the D-class function contest in E13. It would need its own E13-style feature battery (cross-file import/reference-graph signals, directory co-location) before being gate-worthy, and Verified's much-lower E-visibility (60.1%, a genuine population difference, not an artifact) means whatever signal is found there should be validated on Verified from the start, not tuned on Lite and hoped to transfer.

**Files** (scratchpad only, not committed):
- `/tmp/e7b_scratch/miss_autopsy_e7b.py` — per-line A–E classifier (lite/verified/verified_old), reuses E7's tokenizer/IDF method
- `/tmp/e7b_scratch/report_e7b.py` — aggregate stats/report generator (`full_report.txt` has the complete raw output)
- `/tmp/e7b_scratch/mine_cases.py`, `mine_d_detail.py`, `mine_e_detail.py` — D/E case-mining scripts
- `/tmp/e7b_scratch/miss_autopsy_raw_{lite,verified,verified_old}.json` — per-line/per-instance raw classification records
- `/tmp/e7b_scratch/repo_idf.json` — reused E7 per-repo document-frequency table (unchanged, same 12 repos)
- `/Users/nicholasarehart/programming-projects/bgrep/lab/swebench_verified_heldout.parquet` — read-only copy from the `verified-region-plumbing` worktree (gitignored, not tracked, needed since it wasn't present in this working tree)
