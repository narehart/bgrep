E7 Miss Autopsy complete. Pure read-only analysis — no engine changes, no checkouts (only `git show`/`git ls-tree`/`git cat-file` object-DB reads), nothing committed.

**Files (scratchpad only, not committed):**
- `/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/8da5679e-b9a2-4564-be57-3b22e4a483df/scratchpad/miss_autopsy.py` — main per-line A-E classifier + lexical-visibility tokenizer
- `/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/8da5679e-b9a2-4564-be57-3b22e4a483df/scratchpad/build_repo_idf.py` — per-repo document-frequency builder (git ls-tree + cat-file --batch, read-only)
- `/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/8da5679e-b9a2-4564-be57-3b22e4a483df/scratchpad/repo_idf.json`, `miss_autopsy_raw.json` — intermediate data
- `/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/8da5679e-b9a2-4564-be57-3b22e4a483df/scratchpad/miss_autopsy_report.py` — stats/aggregation
- `/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/8da5679e-b9a2-4564-be57-3b22e4a483df/scratchpad/E7_MISS_AUTOPSY_REPORT.md` — full report (pasted below)

Verified read-only: `git status` on the repo and all 12 `lab/swebench_repos/*` clones shows only pre-existing untracked `.roust/` cache artifacts (created by concurrent background eval activity, not by this analysis — I never invoked the roust CLI, only `git show`/`ls-tree`/`cat-file`).

Full report:

---

# E7 — Miss Autopsy

Read-only analysis. Data: `lab/results_regions/full300_v10.jsonl` (300 SWE-bench Lite instances, deterministic baseline regions), gold hunks from `lab/swebench_lite.parquet` via `parity/region_eval.py`'s hunk parser, source read via `git -C lab/swebench_repos/<repo> show <base_commit>:<path>` (object-DB reads only — no checkout/reset/clean, no working-tree mutation). 300/300 instances had usable results (0 errors, 0 skips); every instance had exactly 1 gold file (300 gold files total across 300 instances) and all 300 gold files are Python (`.py`) — SWE-bench Lite's 12 repos are all pure-Python projects, so the spec's "count non-Python separately" bucket is empty by construction, not by omission. Total gold lines across all instances: **3858**.

Baseline sanity check: mean-of-per-instance line recall recomputes to **0.4564** (median 0.261), matching the spec's quoted ≈0.456 baseline. Note this is the *instance-weighted* mean of per-instance fractions; the pooled *line-weighted* fraction (below) is lower (0.403) because higher-gold-line-count instances pull it down more than the per-instance average does.

## A–E distribution

| Category | Line-weighted (n=3858) | Instance-weighted (mean of per-instance %) |
|---|---|---|
| A — captured | 1554 (40.28%) | 45.64% |
| B — missed-near (±20 lines, returned file) | 539 (13.97%) | 14.42% |
| C — missed-same-function (>20 lines, same AST function) | 12 (0.31%) | 0.25% |
| D — missed-in-file-far (different function, returned file) | 1472 (38.15%) | 32.03% |
| E — missed-file (gold file not returned at all) | 281 (7.28%) | 7.67% |

23/300 instances (7.7%) have the gold file missing entirely from the returned set (all of that instance's gold lines fall in E) — matches the 300−277=23 "not all gold files retrieved" count from the existing `region_eval_report.json` baseline, cross-confirming the file-selection stage is unchanged between that report and this one.

Per-instance capture-fraction distribution: 143/300 instances (47.7%) at 0% capture, 107/300 (35.7%) at 100% capture — the distribution is strongly bimodal, not smoothly spread: bgrep on a given instance is disproportionately likely to either fully nail the gold region or fully miss it, with only ~17% of instances landing strictly between 0 and 1.

**C is nearly empty (12 lines, 0.3%).** This says something structural about how bgrep already builds regions: it appears to chunk along function/class boundaries, so partial-function coverage (captured part of a function, missed another part of the *same* function >20 lines away) essentially doesn't happen — a function is either brought in whole (or near-whole, within the ±20 window) or skipped whole. The missed mass is not "almost right, wrong end of the function" (C) — it's "returned the file but selected the wrong function in it" (D, 38.15%) or "didn't return the file" (E, 7.28%).

## B distance histogram (539 missed-near lines)

| Distance from nearest returned span | Count | % of B |
|---|---|---|
| 1–5 | 171 | 31.7% |
| 6–10 | 157 | 29.1% |
| 11–15 | 104 | 19.3% |
| 16–20 | 107 | 19.9% |

Mean distance 9.2, median 9 — roughly flat-to-front-loaded, no sharp cliff. This is consistent with a fixable window/boundary-padding problem: a substantial share of B (the 1–10 bucket, 60.8% of B, ~8.5% of all gold lines) would convert straight to A with a modest (+10-line) span-padding rule, with diminishing but still-real returns out to the ±20 cutoff.

## Lexical-visibility ceiling

Tokenizer: lowercase, split on non-alphanumerics (which also splits snake_case, since `_` isn't in the word-char class used), plus a camelCase-boundary splitter. Two term-filtering variants were computed:

1. **Per-repo IDF filter (spec-preferred)**: for each of the 12 repos, built a document-frequency table over all `.py` files at the repo clone's current HEAD (via `git ls-tree` + `git cat-file --batch`, read-only, ~10k files total, 16s) and excluded any term appearing in >25% of that repo's files. **This is the number quoted below.**
2. **Stopword-only fallback** (English + Python-keyword stopword list, no corpus statistics) — computed for comparison only, see caveat below.

Window: gold line ± 15 lines (31-line window), read from the base_commit source via `git show`.

| | % of ALL gold lines | % of MISSED gold lines |
|---|---|---|
| Lexically **visible** (IDF-filtered) | 93.08% (3591/3858) | 89.28% (2057/2304) |
| Lexically **invisible** (IDF-filtered) | 6.92% (267/3858) | 10.72% (247/2304) |

**Corrected ceiling: 93.08%.** If every lexically-visible gold line were captured perfectly, line-weighted recall would be 93.08%, not 100% — i.e. ~6.9% of all gold-line mass is structurally unreachable by *any* term-matching objective, no matter how well-tuned, because no query term appears anywhere near it in the source.

Visibility by category (IDF-filtered):

| Category | n | visible | invisible |
|---|---|---|---|
| A | 1554 | 98.7% | 1.3% |
| B | 539 | 95.9% | 4.1% |
| C | 12 | 100.0% | 0.0% |
| D | 1472 | 88.5% | 11.5% |
| E | 281 | 80.4% | 19.6% |

Invisibility rate climbs monotonically A→B→D→E — i.e. the harder the miss class, the more likely it's *also* lexically invisible, but even in the worst bucket (E, missed-file) 80% of gold lines are still lexically visible, meaning most E misses are a **file-ranking** failure (the right file existed and had matching terms, but a competing file's terms scored better on the file-level retrieval step) rather than a lexical-coverage failure.

**Flagged assumption / methodological caveat**: the stopword-only variant (no corpus IDF) gives a much higher, less credible 97.41% ceiling — inspecting the actual overlap tokens driving it showed the top hits were generic high-frequency code/English words and even bare repo names (`error`, `value`, `name`, `model`, `field`, `array`, `data`, `sympy`, `django`, `default`, `check`, `file`...), which appear in nearly every file/window by chance and don't represent genuine retrieval signal. The spec's own escape hatch ("if cheap") turned out to be affordable here (~10k files, 16s total across all 12 repos), so this report uses the IDF-filtered 93.08% number as the ceiling and reports the 97.41% stopword-only number only as a cautionary comparison, not as the headline figure. Two remaining approximations in the IDF-filtered number, both load-bearing but judged acceptable: (a) document frequency was computed once per repo at the clone's current HEAD rather than separately at each instance's exact `base_commit` (re-tokenizing ~10k files × up to 300 distinct base_commits was not cheap; vocabulary/DF is stable across nearby commits within one repo's history); (b) only `.py` files were counted as "the repo's files" for the >25% denominator (non-code files are extremely unlikely to share identifier vocabulary and would only dilute the threshold).

## Query-type conditioning

Classification (priority order for the single "primary type" column: has-traceback > has-code-block > has-quoted-identifier > prose-only; the non-exclusive raw flag counts are given below the table since types overlap heavily in practice):

| Query type | n instances | mean capture-frac | A% | B% | C% | D% | E% |
|---|---|---|---|---|---|---|---|
| has-traceback | 71 | 56.32% | 53.2% | 12.4% | 0.6% | 28.1% | 5.7% |
| has-code-block | 108 | 30.86% | 24.1% | 15.5% | 0.5% | 53.2% | 6.6% |
| has-quoted-identifier | 108 | 53.81% | 50.8% | 13.1% | 0.0% | 29.6% | 6.5% |
| prose-only | 13 | 42.22% | 36.0% | 15.1% | 0.0% | 27.1% | 21.8% |

Non-exclusive raw counts (an instance can match more than one; only 300 instances total): has_traceback=71, has_code_block=156, has_quoted_identifier=282.

**Traceback and quoted-identifier queries do best** (~54-56% mean capture) — both give the retriever a literal, low-ambiguity string to match against (a file/line reference or an exact identifier), which shrinks D (wrong function in the right file) to ~28-30%. **Code-block queries do worst** (30.86% mean capture, D balloons to 53.2%) — counter to the "more literal text = easier" intuition, this is likely because embedded code blocks in SWE-bench problem statements are usually the *reproduction snippet* the user ran (which invokes the buggy code path, often from a test/example angle) or the *stack trace's surrounding context*, not the fix site itself; a code block full of matching-looking tokens can out-rank the actual defect location in the same file. Prose-only is a small, noisy sample (n=13) but has by far the worst E-rate (21.8%) — with no literal anchor at all, file-level ranking degrades most.

## Interpretation (honest read)

The single largest fixable pool is **D (38.15% line-weighted, 88.5% of it lexically visible)**: bgrep already retrieves the right file and the right kind of content signal is present nearby, but the region-selection step picks the wrong function/block within that file. This needs an **in-file ranking signal beyond raw term match** — e.g. def-use confluence (does this function call/get-called-by something matching the query), call-graph proximity to already-selected regions, or symbol-definition weighting for the *specific* function the query's terms most precisely name (astropy-12907 is a clean example: `_cstack` matches nothing about the query's surface terms as strongly as neighboring functions that happened to get selected instead). **B (13.97%, distance-histogram front-loaded)** is comparatively cheap to fix: a modest span-padding/window-widening rule around already-selected regions would convert the bulk of the 1-10-line-distance mass straight to A with no new signal required. **C is real but tiny (0.31%)** — not worth dedicated engineering; the region-builder's function-level chunking already prevents most partial-function misses. **E (7.28% lines, 23/300 instances with the gold file missing outright, 80.4% lexically visible)** is a **file-ranking** problem, not a lexical-coverage problem — the fix-site file usually contains matching terms, it just loses the file-selection contest to other candidate files; this calls for cross-file signals (import/reference graph from already-strong files, directory/module co-location with top-ranked hits) rather than more aggressive per-file term weighting. Finally, the **~6.9% lexically-invisible floor** is not addressable by *any* term-matching objective, however tuned — closing it requires genuinely non-lexical signals: version-control history/co-change association between the query's referenced symbols and the true fix site, "natural" code-salience priors (recently touched, high blame-churn, or structurally central code), or def-use confluence that doesn't route through surface tokens at all.
