# roust

**Recall-first code retrieval for coding agents.**

Point an agent at grep and it has to iterate on search terms, reading through
a lot of matches to find what it needs. Point it at roust and it gets a
single, ranked, token-budgeted bundle of the relevant code back in one call,
with no embeddings, no LLM calls, no API keys, and no training. Validated on
407 held-out SWE-bench Verified instances (92.1% all-gold-files, never tuned
on) and on the archex head-to-head benchmark (40/40 tasks at recall 1.00).
roust is a ranking-and-packing pipeline over plain lexical, structural, and
version-control signals — it reads like a very disciplined `grep` session,
compressed into one process call.

## Install

roust is a single Rust binary. Every install path below builds or ships the
same `roust-rs` engine — there is no separate Python implementation.

```bash
# PyPI (wheel built by maturin, bundles the Rust binary):
pip install roust
# or
uv tool install roust
# or
pipx install roust
```

```bash
# From source, via cargo:
cargo install --path roust-rs
```

```bash
# From source, via pip (builds the wheel locally with maturin):
git clone https://github.com/narehart/roust && cd roust && pip install .
```

`git` should be on `PATH` if you want the commit-history signal (roust
degrades gracefully without it).

Developing against `roust-rs/`: `uv run roust` does not rebuild automatically
when `roust-rs/src` changes -- after any Rust edit, run `uv sync
--reinstall-package roust` before relying on `uv run roust` again (`roust
--version` embeds a git SHA + dirty flag so a stale build is identifiable;
see `lab/tokenbench/README.md`'s engine-provenance guard for the automated
version of this check).

The first `roust` call against a repo builds an index — a few hundred
milliseconds to a few seconds depending on repo size. The index is cached
under `<repo>/.roust/` (add that directory to your `.gitignore`) and
refreshes automatically whenever indexed files change, so every call after
the first is a cache hit.

## Usage

```
roust QUERY [PATH]
```

`QUERY` can be a natural-language question or raw issue/error text; `PATH`
defaults to `.`. Default output on stdout is a token-budgeted, packed bundle
of the most relevant code regions; a one-line stats summary always goes to
stderr, so stdout stays clean for piping.

A real session, run against `encode/httpx`:

```bash
$ roust "connection pooling" ~/code/httpx
[... ~8.4k tokens of packed file regions on stdout ...]
roust: 25 files, 8366 tokens (indexed 57 files, index 9ms, query 160ms, cache hit)
```

Other flags:

```bash
# Ranked file paths only, one per line -- for fast localization
roust "connection pooling" ~/code/httpx --files-only

# Machine-readable output: files, packed regions, bundle text, timing stats
roust "connection pooling" ~/code/httpx --json

# Cap the file count (0 = no cap, the default)
roust "connection pooling" ~/code/httpx --k 5

# Change the token budget for the packed bundle (default: 8192)
roust "connection pooling" ~/code/httpx --budget 4096

# Force a fresh index build even if a cache entry exists
roust "connection pooling" ~/code/httpx --reindex

# Skip the on-disk cache entirely (neither reads nor writes .roust/)
roust "connection pooling" ~/code/httpx --no-cache

# Disable individual signal channels (all on by default)
roust "connection pooling" ~/code/httpx --no-history     # git commit-message field + co-change frontier
roust "connection pooling" ~/code/httpx --no-docs        # *.rst/*.txt/*.md docs-bridge
roust "connection pooling" ~/code/httpx --no-anchors     # definition-symbol anchor channel
roust "connection pooling" ~/code/httpx --no-testbridge  # test-file lexical bridge

# Dump the full diagnostic record (roust.core.Explain) as JSON to stderr
roust "connection pooling" ~/code/httpx --explain
```

Exit codes: `0` = results found (this includes low-confidence matches, see
below -- roust still returns its best guess), `1` = no query term matched
anything in the indexed corpus vocabulary at all, `2` = usage error.

### Low-confidence matches

roust always returns a budget-filled bundle for any query that matches at
least one term somewhere in the repo -- it doesn't refuse to answer just
because the match is weak. To make that weak-match case visible instead of
silent, `--json` output's `stats` includes:

- `top_score`: the strongest candidate file's raw (pre-normalization) BM25F
  score for this query -- comparable across queries and repos, unlike the
  0-1 normalized scores used for ranking.
- `matched_query_terms` / `total_query_terms`: how many of the query's terms
  exist anywhere in the indexed corpus vocabulary (body text, comments,
  docs pages, commit messages, or path components).
- `low_confidence: true`, present only when the calibrated criterion trips
  (`top_score` below a fixed threshold, or fewer than 45% of query terms
  found in the corpus vocabulary) -- also appended as `[low-confidence
  match]` to the stderr summary line.

The thresholds were calibrated empirically against all 300 SWE-bench Lite
(query, repo) pairs -- 0 false trips on that real-query population is the
hard constraint -- checked against ~30 gibberish/off-topic queries across 3
repos. Because real BM25F scores scale with query length and repo size,
this signal is calibrated for realistic-size repositories; a tiny
few-file toy repo can legitimately score below the threshold even on a
genuinely on-topic query.

## Using with coding agents

This is the point of the tool: an agent that reaches for `roust` before
`grep` gets the files it needs in one shot, without needing to iterate on
search terms across a much larger result set.

### Claude Code

Add to your project's `CLAUDE.md`:

```markdown
## Code search

Before using grep/find/glob to explore this repo, run roust first:

- `roust "<question or issue text>" --files-only` to localize which files
  are relevant.
- `roust "<question or issue text>"` to get a packed bundle of the actual
  relevant code, ready to read.

Pass the raw question or issue text as the query -- don't summarize or
clean it up first. Include error messages, stack traces, file paths, and
backtick-quoted symbol/function names verbatim; roust uses those as
high-precision anchors. Only fall back to grep for a literal string match
roust's bundle doesn't cover.
```

And allowlist it in `.claude/settings.json` so it runs without a permission
prompt:

```json
{
  "permissions": {
    "allow": ["Bash(roust *)"]
  }
}
```

### Cursor

Add to `.cursorrules`:

```
Before grepping this repo, run `roust "<question or issue text>" --files-only`
(or without --files-only for a packed code bundle) in the terminal to find
relevant files. Pass the raw question/issue text as the query, including
error strings and backtick-quoted symbol names -- don't paraphrase it first.
```

### Aider

Invoke it from chat with `/run`:

```
/run roust "TypeError in connection pool cleanup" --files-only
```

And add a line to `CONVENTIONS.md`:

```
Search this repo with `roust "<raw question or error text>"` before grep --
it returns a token-budgeted bundle of the relevant code directly.
```

### OpenAI Codex CLI / generic agents

Add to `AGENTS.md`:

```markdown
## Code search

Run `roust "<question or issue text>" --files-only` to localize relevant
files, or `roust "<question or issue text>"` for a ready-to-read code
bundle. Pass the raw question/issue text verbatim as the query (error
messages, paths, and backtick-quoted symbols included) rather than a
cleaned-up paraphrase.
```

### MCP

No MCP server yet (it's on the roadmap) -- roust is shell-first by design
today, since every agent already has a shell and `roust` is a single
subprocess call with structured `--json` output when you need it.

> **Query tips for agents**
> - Pass the raw issue/question text verbatim as the query.
> - Include error messages, stack traces, and symbol names -- don't strip
>   them out.
> - Don't summarize the question into clean prose first: measured on
>   adversarial paraphrases that drop key terms, task recall falls from
>   1.00 to 0.833 (14/19 tasks) -- summarization removes the anchors roust
>   relies on. See "Known limits" in `lab/README.md`.

## How it works

- **BM25F** over identifier subtokens (camelCase/snake_case split, Porter-lite
  stemming), with path tokens as a separate weighted field and an
  implementation-file prior (tests/docs/examples down-weighted).
- **1-hop structural expansion** over the import/same-package graph, with RM3
  pseudo-relevance feedback carrying evidence from lexical hits to their
  quiet neighbors.
- **Commit-message channel**: git history text folded in as a monotone
  addition-only signal (never reorders the lexical head).
- **Definition-symbol anchors**: rarity-gated (symbol defined in ≤3 impl
  files), tiered promotion so only strong anchors can enter the top ranks.
- **Test/docs bridges**: tests and docs are treated as developer-written
  natural-language-to-code mappings, appended tail-only.
- **Greedy weighted-coverage region packing** under a token budget, so the
  final bundle is code regions, not whole files, chosen to maximize coverage
  per token.

Every component above was added to fix a concrete, measured miss, and every
number in this README is reproduced in the pipeline's research log,
including negative results and a pre-registered held-out validation run:
see [`lab/README.md`](lab/README.md).

## Scoreboard

Given the same task and the same agent (tokenbench v2, live Sonnet 4.5, each method as the agent's only tool), roust solves **93.3%** of tasks, grep **26.7%**, embedding-RAG **80.0%** (9-trial mean) — n=15, a partial run (see below). roust is **not** the most accurate retriever available: trained retrievers (see *Localization accuracy* below) score higher on published localization benchmarks. What roust offers is the best result you can get for free — no model, no embeddings, no API key, no training.

### Agent-loop outcomes (our protocol)

| System | Solves | Median turns | Tokens / attempt | $ / attempt | $ / successful run |
|---|---|---|---|---|---|
| **roust** | 93.3% | 9 | 308,184 | $0.95 | $0.93 |
| grep | 26.7% | 30 | 239,600 | $0.76 | $0.53 |
| embedding-RAG | 80.0% (9-trial mean ± 4.4pp) | 20 | 695,833 | $2.14 | $1.80 |
| roust + grep (both) | 57.1% | 28 | 595,234 | $1.83 | $1.60 |
| grep + stopping prompt | 20.0% | — | 52,576 | $0.17 | $0.18 |
| roust + stopping prompt | 66.7% | — | 241,027 | $0.74 | $0.63 |

- roust costs more per attempt than grep (308k vs 240k tokens) and wins on solve rate anyway. grep is cheap because it gives up: 73.3% of its runs hit the turn cap and produce nothing.
- `$ / successful run` remains a lower bound on cost-to-answer for a single attempt. The full repeat-run campaign (#16, `results_repeats.jsonl`) measured the rest: for roust and grep, failures are **stable across trials** (p≈0 — the retry term is meaningless; roust's one miss failed 10/10), while embedding-RAG's failures are genuinely stochastic, giving **E[cost to first success] = $2.50** over its solvable set (all 15 instances, per-instance p̂ 0.11–1.00).
- Giving the agent grep *alongside* roust makes it worse (93.3% → 57.1%): replace grep, don't supplement it ([#5](https://github.com/narehart/roust/issues/5)).
- embedding-RAG's **Solves** cell is a 9-trial mean, `lab/tokenbench/results_repeats.jsonl`; its other columns (median turns, tokens, $) are the trial-0 measurement, `lab/tokenbench/results.jsonl`.
- Outcome volatility (measured across 9 identical trial repeats, `lab/tokenbench/results_repeats.jsonl`): embedding-RAG bounces 73.3–86.7% across 9 identical runs (mean 80.0% ± 4.4pp); roust reproduced 93.3% exactly with 0 outcome flips across all repeats, and its single failure (django-16400) failed 10/10 trials — a capability gap, not variance (p < 0.30 at 95%, rule of three). grep's failures were stable across both its trials.

### Localization accuracy (published protocol)

| System | File-level | Metric | Free? | Source |
|---|---|---|---|---|
| SweRankEmbed-Large + LLM rerank | 96.0 | Acc@10 | no (trained + LLM) | arXiv:2505.07849 |
| SweRankEmbed-Large | 94.2 | Acc@10 | no (trained) | arXiv:2505.07849 |
| LocAgent | 94.16 | file acc | no (LLM) | arXiv:2503.09089 |
| **roust** | 92.3 | Agentless-metric FILE | yes | `lab/results_regions/agentless_metric_v3.json` |
| SweRankEmbed-Small | 90.9 | Acc@10 | no (trained) | arXiv:2505.07849 |
| OrcaLoca | 83.33 | file-match | no (LLM) | arXiv:2502.00350 |
| Agentless GPT-4o | 69.7 | Agentless-metric FILE | no (LLM) | arXiv:2407.01489 |
| BM25 | 61.7 | Acc@10 | yes | arXiv:2505.07849 |
| CoSIL | 60.7 | Top-1 | no (LLM) | arXiv:2503.22424 |
| archex | — | — | no (embeddings) | — |

— = not measured by us (see gaps below).

The File-level column mixes several different metrics (Acc@10 / Top-1 / file-match / Agentless-metric FILE) and is **not** comparable straight down the column — each row names its own. roust's Agentless-metric scores on Lite are FILE 92.3% / FUNCTION 41.0% (exact) / LINE 35.7%; Agentless (GPT-4o) for comparison is 69.7 / 52.0 / 35.3. Region precision (gold lines returned / total lines returned, i.e. "how much of the packed context is actually the fix") is 0.45% mean — roust trades precision for recall by design, packing ~1,150 lines of surrounding context per instance under the 8192-token budget ([#4](https://github.com/narehart/roust/issues/4)).

Cold index build, Rust vs Python engine: httpx 145ms vs 522ms; django 1.8s vs 7.6s — prose-only, no committed benchmark artifact ([#15](https://github.com/narehart/roust/issues/15)).

### ContextBench (human-annotated gold context, their evaluator)

[ContextBench](https://github.com/EuniAI/ContextBench) (arXiv:2602.05892) scores
retrieved context against human-annotated "necessary context" line regions.
roust was run one-shot (`--json --budget 8192`, single call, no model) on the
**Python subset of their curated 500-instance Verified benchmark (266 tasks, 19
repos, 266/266 evaluated, 0 skipped)** and scored with ContextBench's own
evaluator, unmodified ([#3](https://github.com/narehart/roust/issues/3)):

| Granularity | roust recall | roust precision | Claude Sonnet 4.5 agent recall | precision |
|---|---|---|---|---|
| file | **0.679** | 0.060 | 0.720 | 0.665 |
| block | 0.346 | 0.040 | 0.449 | 0.420 |
| line | 0.274 | 0.053 | 0.374 | 0.344 |

Protocols differ and the comparison is not apples-to-apples: the published
baselines are **multi-turn LLM agents** (read, navigate, then select context)
on the full 500-task 8-language set; roust is a **single sub-2-second call
with no model, no API key, and no training**, on the Python 266. Read it as:
one free one-shot call recovers ~94% of the file-level recall of the best
agent, and its precision is ~10x lower because roust deliberately packs a
full 8192-token recall-first bundle rather than a minimal answer — the same
recall-over-precision trade documented in
[#4](https://github.com/narehart/roust/issues/4). ContextBench's efficiency
metrics (AUC-Coverage/Redundancy) are N/A for a one-step trajectory.
Adapter + protocol: `lab/contextbench/`; aggregate:
`lab/contextbench/results_python.json`.

### What still needs work

- ~~Line-level 35.7% and function-level 44.3% (a proxy, not the exact metric)~~ measured exactly ([#2](https://github.com/narehart/roust/issues/2)): FUNCTION 39.7% (exact, was a 44.3% proxy) and LINE 29.3% (was 35.7%) from a fresh 300-instance run of the shipped engine; a `w_name` sweep on the exact harness ([#4](https://github.com/narehart/roust/issues/4)) then showed the symbol-name weighting itself caused the LINE drop — reverting it (w_name=0.0) restores FUNCTION 41.0% (exact) and LINE 35.7% (`lab/results_regions/agentless_metric_v3.json`). FUNCTION is still the weakest cell vs Agentless GPT-4o's 52.0 — [#3](https://github.com/narehart/roust/issues/3)
- archex has never been measured by us on any of our benches — [#1](https://github.com/narehart/roust/issues/1)
- ~~True cost-per-success~~ measured via repeat runs ([#16](https://github.com/narehart/roust/issues/16)): roust solves 14/15 deterministically at ~$1/answer with one real capability gap (django-16400, 0/10); embedding-RAG reaches everything eventually at $2.50/first-success — see `results_repeats.jsonl`
- Latency has no committed benchmark artifact — [#15](https://github.com/narehart/roust/issues/15)

### How these were measured

*Agent-loop outcomes* is our agent-loop harness (live Sonnet 4.5, each method as the agent's only tool, measured to task completion) — a partial run, 58 of 120 planned pairs, stopped at an $80 spend cap, so n=15 (14 for embedding-RAG). *Localization accuracy* is published Acc@k-style numbers from each system's own paper, on its own harness — a different protocol, not comparable to the agent-loop numbers. Full artifacts and the research log (including the retracted "95% fewer tokens than grep" claim, which came from a v1 one-shot protocol and does not hold in the agent loop — [#6](https://github.com/narehart/roust/issues/6)) are in `lab/README.md`.

## Limits

- **File-level, not line-level.** roust localizes to files and packs regions
  within them; it doesn't point at a specific line or diff hunk.
- **@1 precision is the measured weak spot.** Top-1 file accuracy on the
  held-out SWE-bench Verified set is .354 -- if you need "the one file",
  read further down the ranked list, don't trust rank 1 alone.
- **Natural-language issues with no identifiers are the hard class.** Every
  non-semantic retrieval method (roust included) leans on identifiers, paths,
  and error strings as anchors; a vague prose description with none of those
  gives the pipeline little to grab onto.
- **Python gets the full signal set** (import graph, definition-symbol
  index). Other languages (JS/TS, Go, Rust, Java, etc.) get a best-effort
  subset -- lexical/BM25F, paths, and history still apply, but there's no
  import-graph or def-index expansion yet.

## Roadmap

- ~~Rust port~~ **complete and shipped as the only engine**: `roust-rs/` was
  brought to feature-parity with the (now-deleted) Python v0.2 engine
  (channel-aware packing, on-disk cache with incremental updates,
  deterministic seed) — bundle-level parity gate **PASSED 300/300 exact**
  on SWE-bench Lite (report in `parity/rust_gate_300_v3.json`) before the
  Python engine was removed. Cold 3.6–4.2× faster than the old Python
  engine (httpx 145ms vs 522ms, django 1.8s vs 7.6s); warm/incremental
  2–3×. Build from source: `cd roust-rs && cargo build --release`.
- ~~Publish to PyPI~~ **done** — `pip install roust` ships the maturin-built
  wheel wrapping the Rust binary.
- MCP server.
- Incremental index updates (avoid full reindex on every change).
- Homebrew tap.

---

Research artifacts -- benchmark JSONLs, diagnostics, and pre-registered
held-out predictions -- live in [`lab/`](lab/README.md). `lab/` is a frozen
Python research sandbox (including `lab/lanes2.py`, the oracle the parity
gates were built against) -- it is never the source of truth for shipped
behavior, which is `roust-rs/` end to end.

License: MIT.

## History

Formerly `bgrep`; renamed to avoid collision with the binary-grep tool of
that name.
