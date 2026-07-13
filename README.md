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

Exit codes: `0` = results found, `1` = no results, `2` = usage error.

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

Given the same task and the same agent (tokenbench v2, live Sonnet 4.5, each method as the agent's only tool), roust solves **93.3%** of tasks, grep **26.7%**, embedding-RAG **71.4%** — n=15, a partial run (see below). roust is **not** the most accurate retriever available: trained retrievers (see *Localization accuracy* below) score higher on published localization benchmarks. What roust offers is the best result you can get for free — no model, no embeddings, no API key, no training.

### Agent-loop outcomes (our protocol)

| System | Solves | Median turns | Tokens / attempt | $ / attempt | $ / successful run |
|---|---|---|---|---|---|
| **roust** | 93.3% | 9 | 308,184 | $0.95 | $0.93 |
| grep | 26.7% | 30 | 239,600 | $0.76 | $0.53 |
| embedding-RAG | 71.4% | 20 | 695,833 | $2.14 | $1.80 |
| roust + grep (both) | 57.1% | 28 | 595,234 | $1.83 | $1.60 |
| grep + stopping prompt | 20.0% | — | 52,576 | $0.17 | $0.18 |
| roust + stopping prompt | 66.7% | — | 241,027 | $0.74 | $0.63 |

- roust costs more per attempt than grep (308k vs 240k tokens) and wins on solve rate anyway. grep is cheap because it gives up: 73.3% of its runs hit the turn cap and produce nothing.
- `$ / successful run` is a **lower bound** — it excludes the failed attempts paid for along the way. True cost-per-success is unmeasured ([#16](https://github.com/narehart/roust/issues/16)).
- Giving the agent grep *alongside* roust makes it worse (93.3% → 57.1%): replace grep, don't supplement it ([#5](https://github.com/narehart/roust/issues/5)).

### Localization accuracy (published protocol)

| System | File-level | Metric | Free? | Source |
|---|---|---|---|---|
| SweRankEmbed-Large + LLM rerank | 96.0 | Acc@10 | no (trained + LLM) | arXiv:2505.07849 |
| SweRankEmbed-Large | 94.2 | Acc@10 | no (trained) | arXiv:2505.07849 |
| LocAgent | 94.16 | file acc | no (LLM) | arXiv:2503.09089 |
| **roust** | 92.3 | Agentless-metric FILE | yes | `lab/results_regions/agentless_metric.json` |
| SweRankEmbed-Small | 90.9 | Acc@10 | no (trained) | arXiv:2505.07849 |
| OrcaLoca | 83.33 | file-match | no (LLM) | arXiv:2502.00350 |
| Agentless GPT-4o | 69.7 | Agentless-metric FILE | no (LLM) | arXiv:2407.01489 |
| BM25 | 61.7 | Acc@10 | yes | arXiv:2505.07849 |
| CoSIL | 60.7 | Top-1 | no (LLM) | arXiv:2503.22424 |
| archex | — | — | no (embeddings) | — |

— = not measured by us (see gaps below).

The File-level column mixes several different metrics (Acc@10 / Top-1 / file-match / Agentless-metric FILE) and is **not** comparable straight down the column — each row names its own. roust's Agentless-metric scores on Lite are FILE 92.3% / FUNCTION 44.3% (a proxy, not the exact metric) / LINE 35.7%; Agentless (GPT-4o) for comparison is 69.7 / 52.0 / 35.3.

Cold index build, Rust vs Python engine: httpx 145ms vs 522ms; django 1.8s vs 7.6s — prose-only, no committed benchmark artifact ([#15](https://github.com/narehart/roust/issues/15)).

### What still needs work

- Line-level 35.7% and function-level 44.3% (a proxy, not the exact metric) — the weakest cells, and where the next work goes — [#2](https://github.com/narehart/roust/issues/2), [#3](https://github.com/narehart/roust/issues/3)
- archex has never been measured by us on any of our benches — [#1](https://github.com/narehart/roust/issues/1)
- True cost-per-success is unmeasured (needs repeat runs to get per-task success probability) — [#16](https://github.com/narehart/roust/issues/16)
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
