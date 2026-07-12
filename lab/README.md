# bgrep lab — retrieval experiments

Goal: LLM-agent code retrieval with **recall 1.00, missed-task 0.00, ≥70% fewer
tokens than grep**, tested on the [archex](https://github.com/Mathews-Tom/archex)
head-to-head benchmark (real repos at pinned versions, archex's own metric and
tiktoken accounting).

## Results (2026-07-10)

| lane | mean tokens/task | file recall | missed-task |
|---|---|---|---|
| raw grep + read matched files | 1,180,370 | 1.00 | 0.00 |
| disciplined grep (read ±25 lines around matches) | 764,759 | 1.00 | 0.00 |
| oracle (read exactly the expected files) | 26,395 | 1.00 | 0.00 |
| archex (embeddings, published artifacts) | 6,433 | 0.947 | 0.158 |
| **bgrep pipeline (packed regions)** | **8,317** | **1.00** | **0.00** |

- Dev set: 19/19 tasks at recall 1.00; savings vs raw grep mean 95.7%, min 76.0%.
- Held-out set (21 `loc_*` fault-localization tasks, frozen config): 21/21 at
  recall 1.00; savings mean 95.2%, min 74.9%.
- vs the *disciplined* grep control the pipeline still saves ≥72% on every task
  (mean 94.7%) — discipline removes only ~35% of grep's cost because keyword
  matches are everywhere in a large repo.

## The pipeline (lanes.py — pure Python, no models, no embeddings)

1. **BM25F** over identifier subtokens (camelCase/snake_case split, Porter-lite
   stemming), path tokens as a weighted field, implementation-file prior
   (tests/docs/examples 0.3×).
2. **1-hop structural expansion** of the whole retrieved set over the
   import/same-package graph, with **RM3 pseudo-relevance feedback** carrying
   evidence from hits to their lexically-quiet neighbors; guarantees for
   direct-import dependencies and discriminative path matches (path-df < 10%);
   evidence-threshold cutoff.
3. **Greedy weighted-coverage region packing** under an 8k token budget,
   evidence-proportional per-file allowances.

Every component was added to fix a concrete observed miss (see HYPOTHESES.md
and git history). Negative results: global personalized PageRank loses to
truncated 1-hop diffusion (hub sink); every fixed-count cutoff caused
whack-a-mole between tasks; per-seed quotas starve globally-strong candidates.

## Known limits (measured)

- **Vocabulary brittleness**: with helper keywords dropped, natural paraphrases
  keep 16/19 tasks at recall 1.00 (mean 0.939); adversarial paraphrases that
  avoid the key terms drop to 14/19 (mean 0.833). Consistent with CORE-Bench
  (2026): the fix is **anchor preservation** — callers should pass raw task
  text (error strings, API names, identifiers) rather than cleaned summaries;
  query *cleanup* hurts even dense retrievers. bgrep's caller is an LLM agent,
  which can supply anchor-rich multi-variant queries nearly for free.
- **Depth vs breadth is budget-bound**: bundles pack ~15-26 files into 8k
  tokens. Evidence-proportional allocation helps only marginally (+3%); depth
  is a caller-set budget knob, not a packing-cleverness problem. Literature
  (SWE-bench oracle-collapsed ablation; SWE-Explore degradation study) says
  recall-first / precision-second is the right trade for modern models.
- Recall 1.00 is empirical over 40 tasks / 17 repos / 5 languages, not a
  guarantee. archex comprehension tasks have 1–5 expected files; harder
  issue-to-edit benchmarks (SWE-bench) remain unmeasured for this pipeline.

## Literature context (deep-research pass, 2026-07-10, adversarially verified)

- One-shot indexed retrieval (BM25 *or* dense) fails at repo scale: BM25
  recovers all gold files in <46% of SWE-bench instances at realistic budgets;
  strong embedders collapse on issue-to-edit localization (CORE-Bench);
  non-agentic retrievers score near random on SWE-Explore while agentic
  explorers reach only ~0.65 file-recall.
- Graph expansion measurably lifts *file-level* localization (+4.4–6.7 pts,
  RepoGraph, ICLR 2025) — the exact mechanism and granularity this pipeline
  exploits. AST/region chunking beats whole files at any budget (cAST, EMNLP
  2025; SWE-bench Table 6).
- GrepRAG (2025): optimized grep-style pipelines beat indexed RAG under equal
  budgets with ~35× latency advantage — supporting the no-embedding design.

## SWE-bench Lite file localization (2026-07-11)

Query = raw issue text, no LLM, no embeddings, no training. Gold = files edited
by the merged fix. 300 instances, zero errors, fully deterministic.

Ablation (File@k = all gold files within top-k of the ranked list):

| lane | @1 | @5 | @10 | @all | mean tokens |
|---|---|---|---|---|---|
| v1 pipeline | .470 | .733 | .803 | .890 | 10,041 |
| + vendor/minified exclusion, pack safety | .463 | .737 | .813 | .893 | 8,461 |
| + comments field (REJECTED) | .357 | .670 | .773 | .867 | 8,453 |
| + history via RRF fusion (REJECTED) | .337 | .643 | .757 | .890 | 8,457 |
| **+ history as monotone additions (final)** | **.463** | **.737** | **.813** | **.910** | **8,492** |
| + def-symbol anchors, flat promotion (superseded) | .463 | .737 | .823 | .917 | 8,515 |
| **+ def-symbol anchors, tiered** | **.463** | **.737** | **.827** | **.920** | **8,515** |
| + test-bridge with head tier (superseded) | .463 | .737 | .810 | .927 | 8,544 |
| **+ tail-only test+docs bridges (FROZEN v7)** | **.463** | **.737** | **.827** | **.923** | **8,549** |

Final: 91.0% of instances get every gold file, ~8.5k-token bundle, 530ms query.
Verified SOTA context (SweRank, arXiv:2505.07849): trained embedders reach
Acc@10 90.9 (137M) / 94.2 (7B); +LLM rerank 96.0; BM25 61.7. bgrep holds the
training-free/model-free point: File@10 81.3 (+19.6 over BM25), @all 91.0.

Negative results (documented, reproducible): comments field hurts issue-style
queries (-8 net instances) despite its API-search pedigree; fusing history into
the ranking head craters @1 (.463 -> .337); co-change test-bridges are impossible
by construction on repos with 1:1 test:impl coupling (django). The recurring
design law, twice observed: auxiliary channels must be MONOTONE — they may add
candidates, never reorder the lexical head.

Anchor campaign (2026-07-11, evidence-first): file-name anchors measured DEAD
(+2.4pp ceiling — BM25F path field already consumes them); code-block 5-gram
fingerprinting measured DEAD (0/56); definition-symbol anchors measured ALIVE
(25/56 failures have a gold file defining a symbol the issue names). Built with
a rarity gate (symbol defined in ≤3 impl files) and tiered head rights: only
backtick-quoted anchors (strength ≥2.0) may enter the top-10 (cap 2, insert at
rank 8, top-7 sovereign); weaker anchors append at rank 12+. @10 .813→.827,
@all .910→.920, zero @all losses, top-7 invariant 300/300. Config FROZEN after
this change: Lite has served as dev set through five iterations; SWE-bench
Verified is reserved as the untouched held-out test.

Bridge campaign (2026-07-11): the repo is a self-describing system — tests and
docs are NL->code mappings written by the developers. Headroom diagnostics:
test-file lexical bridge (issue matches test vocabulary; test imports gold)
14/52 @10 + 6/24 @all; docs bridge (issue matches doc page; page references
gold) 10/52 + 3/24 after fixing a filter bug that had excluded docs/ from its
own measurement; literal error-string matching 0/52 — modern f-string
interpolation means quoted runtime messages never exist verbatim in source.
Shipped tail-only (positions 14/16, top-10 byte-invariant 300/300): @all
.920->.923. Head-tier test-bridge tried and reverted (-5 net @10: tests import
their subject plus utilities, and import-linkage alone cannot tell them apart).
Ceiling-vs-conversion lesson: a channel that can reach gold must still win its
capped slot against the channel's other candidates — reachability ceilings
(.943) are permissive, conversion (+1) is an intra-channel ranking problem.

## SWE-bench Verified — held-out validation (2026-07-11)

Frozen v7 config, predictions pre-registered in PREDICTIONS.md (commit f5f6b0e)
before the run finished. 500 instances, zero errors.

| subset | n | @1 | @5 | @10 | @all |
|---|---|---|---|---|---|
| **held-out (minus Lite overlap)** | 407 | .354 | .649 | **.794** | **.921** |
| Lite overlap (contaminated) | 93 | .505 | .774 | .817 | .914 |
| full Verified | 500 | .382 | .672 | .798 | .920 |

Recall (.921) matches the dev set (.923) on never-seen instances — the
recall-first architecture generalizes. Head precision (@1/@5) came in below
the pre-registered intervals; the Lite-vs-held-out @1 gap (15pp) confounds
Lite's easier instance selection with dev-set tuning and is reported as
non-transferring. Grep-parity latency, ~8.5k-token bundles, zero learned
parameters throughout.

Post-Verified @1 investigation: BRTracer-style stack-trace boosting (verified
+10pp Top-1 in its home corpus, ICSME 2014) was headroom-measured on Lite
before building: only 17% of issues contain a traceback, gold is in-trace for
just 45% of those (CrashLocator's symptom-vs-cause ceiling, worse here), net
naive @1 bound +5 instances (~+1.7pp) with 3 displacement risks, and zero @10
headroom — below the pre-committed +10-instance build bar. Not built
(diag_stacktrace.py). Conclusion: every non-learned precision-class head
signal is now measured — path anchors (consumed by BM25F), symbol anchors
(shipped, +1.4pp @10), error strings (dead: f-string interpolation), stack
traces (below bar). Head precision beyond @1=.354 held-out requires semantics
that repo mining cannot supply; that boundary is this project's measured edge
of no-learning retrieval.

## Run it

```bash
git clone https://github.com/Mathews-Tom/archex && cd archex && uv sync
uv run python <this dir>/driver.py                     # all 19 dev tasks, all lanes
uv run python <this dir>/driver.py --tasks httpx_pooling --lanes bm25_ppr_pack
uv run python <this dir>/driver.py --questions-json paraphrases.json --variant adversarial
```

Result JSONs: `results/` (dev), `results_holdout/` (frozen-config loc_*),
`results_grepdisc/` (disciplined-grep control), `results_para_*/` (paraphrase
robustness), `results_weighted_pack/` (allocation ablation).

## SWE-bench Multilingual — zero-shot baseline (2026-07-11)

Frozen v7 pipeline + extension-visibility fix only (no language-specific
engineering beyond existing regexes). First non-LLM localization numbers on
this benchmark. n=300 evaluated (0 harness errors; the 17 lombok instances
that previously errored on a non-UTF-8 byte in `git log` output are now
recovered -- see `errors="replace"` hardening in `history.py`):

| language | n | @5 | @10 | @all |
|---|---|---|---|---|
| Java | 43 | .558 | .651 | .767 |
| C | 41 | .732 | .756 | .878 |
| PHP | 43 | .628 | .721 | .791 |
| Go | 41 | .537 | .683 | .780 |
| Rust | 43 | .465 | .628 | .767 |
| JS/TS | 43 | .256 | .535 | .744 |
| Ruby | 44 | .591 | .705 | .795 |
| ALL | 300 | .537 | .663 | .773 |

The language-agnostic lexical core carries .78 all-gold across nine languages.
Tree-sitter priority, from measured gaps: Ruby (no .rb parsing exists at all),
JS/TS (worst @5; re-export-heavy import graphs), Rust (@1 .163). Java and C
nearly match Python with lexical signals alone.

Ruby regex-tier support (2026-07-12): require_relative + lib-root require +
Rails-autoload constant resolution + Ruby def-index. @all .705->.795 (+4/-0);
adding .gemspec/.rake to the corpus initially cost @1 (project-name-dense
metadata outranking code) — fixed with a 0.5 metadata prior; one residual @1
instance (.erb site-template scaffold) accepted rather than tuned away. JS/TS
import-resolution improvements (re-export chains, tsconfig aliases, index
files) measured FLAT on the 43-instance slice — an 11-miss study shows JS/TS
failures are selection-scheduling and monorepo-seeding problems, not parsing
problems; the dominant lever is the per-source additions cap starving pooled
gold files (adaptive selection scheduling queued).

## Region-quality baseline and channel-aware packing (2026-07-11)

Ground truth: the exact lines edited by each gold patch (SWE-bench Lite, 300).
Baseline packing (keyword-coverage only): fix-line recall mean 0.26, median 0.00
— right files, wrong slices; 66% of instances had zero fix-line coverage.
Channel-aware packing v2 (the packer now honors WHY each file was selected:
anchored symbols' definition blocks are packed; deeper allocation to top
evidence files): mean 0.46, median 0.26, +0.4% tokens, file rankings byte-
identical (parity green). On the right-files-found subset: 0.49 / median 0.50.
archex expected_regions (informational, 9 tasks): 0.13 -> 0.19. Note: bgrep-rs
does not yet carry the v2 packing — its parity gate covers file rankings only;
porting + a region-metric gate is queued.

## Neighborhood-first retrieval (2026-07-12, experiment, flag off by default)

For repos >3000 files, seed with rare-term + symbol-anchor hits, expand 2 hops
over import+dir edges (cap 800), rank only within the region (LARGER pattern,
arXiv:2605.16352). Engaged only on babel (16.5k files) in the SWE-bench
Multilingual JS/TS slice: all-gold 2/5 -> 3/5; provably byte-identical on all
38 sub-threshold instances and inert on every Lite repo (max 2.2k files).
Promising but n=5; validation at scale on Multi-SWE-bench's 580 JS/TS
instances (incl. material-ui, 27.6k files) in progress.

## Survival-scheduled packing (2026-07-12, REJECTED — negative result)

Scheduling deep-vs-broad packing by score-concentration confidence (DSpark-inspired
adaptive verification) failed at estimator calibration, exactly as QPP
literature warns: flat class byte-identical (guard worked), peaked class net
-6.5pp with symmetric counts but asymmetric magnitudes — two instances fell
1.0 -> 0.0 when the scheduler confidently deep-packed the wrong files. Score
concentration measures confidence, not correctness. Patch + subset data in
lab/experiments/. Possible salvage (roadmap footnote): gate deep-packing on
anchor-strength signals only (the measured near-certain class), not score
concentration.

## Neighborhood-first at scale (2026-07-12, REJECTED)

Multi-SWE-bench JS/TS validation (580 instances; material-ui = 174 instances at
10.4k files) reversed the babel n=5 result decisively — engaged-group all-gold
43.1% -> 29.9% (+4/-27). Mechanism: the 800-file region cap saturates on every
engaged instance (2-hop expansion from ~15 seeds exceeds it immediately),
making the "neighborhood" a truncation artifact that excludes fix files. The
n=5 babel win was noise; the scale gate worked as designed. Salvage hypotheses
(roadmap, unproven): repo-size-proportional region caps; confidence-filtered
expansion instead of fixed hop/cap truncation. SEPARATE FINDING: 3
sub-threshold svelte instances returned non-identical lists flag-on vs
flag-off (should be inert) — determinism investigation below.

## Assembly-order scheduling (2026-07-12, NULL RESULT)

Global score-ordering of the below-head segment (motivated by Go/JS-TS miss
studies showing pool-ranked candidates starved by guarantee-phase assembly)
moved NOTHING — Go 42 instances and Ruby/JS 87 instances identical at every k,
zero gained/lost, despite 81/87 lists reordering below the head. Conclusion:
membership, not order, binds — the misses sit at pool ranks 21-47, beyond any
reordering's reach. The miss studies located the failures correctly but
misattributed the mechanism. Live levers remain: additions capacity
(Pandora-index roadmap item) and add_score evidence quality. lanes3.py
preserved as the experiment artifact.
