# Research wave 2 — region-quality literature scans

Date: 2026-07-14

Three scout reports investigating roust's region/line-level gap: file-level localization is strong (92.3% all-gold, SWE-bench Lite) but region/function/line quality lags (FUNCTION 41.0% vs Agentless 52.0%; only 45.6% gold-line capture once inside the right file; ContextBench 0.274 vs 0.374 best-agent).

## Reports

- **[sota-localization-anatomy.md](sota-localization-anatomy.md)** — *Given roust beats Agentless at FILE but loses at FUNCTION, what mechanical step converts a correct file into a correct function/line in published systems, and how much is deterministic scaffolding vs. LLM judgment?*
- **[budgeted-passage-selection.md](budgeted-passage-selection.md)** — *roust's greedy budgeted-coverage packer always fills its token budget at 0.45% precision — what does the span/passage-selection literature say about fixing the objective (not just the ranking signal) for this failure mode?*
- **[fine-grained-fault-localization.md](fine-grained-fault-localization.md)** — *What non-textual, execution-free, training-free signals (change-history/recency, AST/structural salience, query-structure-aware IR) predict where in a file an edit lands, independent of query-term density?*

## Convergent finding

Across all three independent scans, roust's region/line gap decomposes into the same three failures, plus two cheap orthogonal per-line signals:

1. **A granularity problem.** Pack function-boundary units, not arbitrary greedy-coverage windows. LocAgent's own ablation shows removing its entity-granular search index costs -19.3pp file Acc@5, while removing graph traversal costs only -2.2pp — the deterministic function/class-level index is the dominant lever, not the agentic hop-by-hop reasoning. SweRankEmbed confirms the same lesson from a different angle: a trained bi-encoder scoring at function-chunk granularity hits 82% Function Acc@10 with zero LLM calls at inference, beating LocAgent-Claude-3.5's agentic 77.4%.
2. **An objective problem.** Saturated-coverage/representativeness plus diversity beats raw term-coverage density. In the one same-optimizer-family study (greedy, marginal-gain-per-token, knapsack-constrained — the same shape roust already uses), swapping the objective from term-coverage-only to a submodular composite (Lin–Bilmes saturated coverage + diversity) yields +0.051 F1 over the naive coverage-only packer, at fewer tokens.
3. **A stopping problem.** Adaptive halting on the marginal-gain curve, not filling to a fixed token budget. Adaptive-k (EMNLP 2025) gets up to 10x fewer tokens than full-context while retaining ~70% of relevant passages, using only the already-computed score distribution — no new signal, no training, a pure stopping-rule change that composes cleanly with fixes #1 and #2.

Plus two cheap, orthogonal per-line signals surfaced independently of the above: **GLANCE**-family structural salience (token-count x call-count, or cognitive-complexity variants — fully deterministic, reuses roust's existing tree-sitter parse, ~230k LOC/sec) as a within-file line-ranking score orthogonal to term density; and **Linespots**-style exponential-recency-decay per-line historical-fault density from git history (24% of the lines Bugspots needs to find the first true fault) — with an explicit warning, corroborated by a 2026 replication study, that the recency decay must be keyed from query/report time, not fix-commit time, or it silently leaks future information.

None of these four packages requires training or execution; all are additive to roust's existing greedy/lexical pipeline rather than replacements for it.
