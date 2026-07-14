# tokenbench v2: agent-token-usage benchmark (grep vs roust vs roust+grep vs rag+grep)

Measures the tokens an LLM agent actually burns to **localize** the fix
files for a SWE-bench Lite issue, under four retrieval toolbelts, holding
everything else (model, system prompt skeleton, turn cap, temperature,
success criterion) fixed. This is the harness behind the "solves it for
1/Nth the tokens" claim, so the design goal is that the grep arm is argued
in good faith, not a strawman, and that roust's own arm isn't given an
unfair head start either.

## STATUS: v1 pilot ran and is REJECTED for a design bug; this is the v2 rebuild

## Why v2 (v1 pilot findings)

v1 (archived in `v1/`, including its `results.jsonl` and `transcripts/`)
ran 39/90 (instance, condition) pairs before hitting its `--budget-cap-usd
20` cap, with three conditions (grep / roust / rag) that **injected**
retrieval context into the first user message. Three findings killed that
design:

1. **Injection gets rebilled every turn.** In a multi-turn tool-use loop
   the whole conversation (including turn 1's injected bundle) is resent
   on every subsequent API call -- that's how Anthropic actually bills
   input tokens. A ~8.5k-token roust bundle injected at turn 1 and resent
   across ~15 turns is billed roughly 15x, not once. Observed: roust's
   median `tiktoken_total_tokens` was 188,221 vs grep's 54,275 -- **roust
   looked 3.5x worse than plain grep**, purely as an artifact of how the
   injected condition was billed, not because retrieval was unhelpful.
2. **Every condition maxed out `max_turns=15`** (median `turns_used` = 15
   in all three conditions; `hit_turn_cap` true for 12-13/13 rows in every
   condition). Success was 0-8%. The v1 pilot was measuring "can an agent
   converge and stop before an arbitrary cutoff", not "does retrieval help
   localization" -- the turn cap dominated every other variable.
3. **The matched-success cell was empty.** No single instance was solved
   by all three v1 conditions, so the one truly apples-to-apples token
   comparison the whole benchmark exists to produce could not be computed
   at all.

v2 fixes all three directly (see "Methodology" below): retrieval becomes a
tool call (never injected, never rebilled on every turn), `max_turns`
raises to 30, and the system prompt explicitly rewards stopping early.

## Methodology

**Same model** (`claude-sonnet-4-5-20250929`), **same system-prompt
skeleton, same turn cap (`max_turns=30`), same temperature (0)**, across
all four arms -- that identity is the fairness property this whole
benchmark rests on. The prompt text that legitimately differs by arm is
minimal and mechanical: the one paragraph listing which tools exist
(different arms have different tools, by design) and a one-sentence
"call X first" opening-move hint; the stop-early instruction, the `FILES:`
answer format, and every guardrail are shared verbatim (see
`agent.build_system_prompt`).

**NO INJECTION ANYWHERE.** The first (and only human-authored) user
message is identical across all four arms: the bare issue text. All
retrieval is a tool the agent chooses to call, with a query the agent
composes itself (not hardcoded to the raw issue text -- an agent can
narrow/refine its query after a miss, same as a human would).

**Arms** (`agent.ARMS`):

- **A `grep`**: `run_command` (whitelist: rg/grep/find/ls/head/sed-range),
  `read_file(path, start, end)`.
- **B `roust`**: `roust(query)` + `read_file`. **No grep** -- roust is
  *the* search tool for this arm, by design (tests roust standing fully on
  its own, not as a supplement to grep).
- **C `roust_grep`**: `roust(query)` + `run_command` + `read_file` -- the
  agent chooses when to reach for which.
- **D `rag_grep`**: `rag_search(query)` + `run_command` + `read_file`.

`roust(query)` (`roust_tool.py`) shells out to the real, shipped CLI --
`uv run roust --json --budget 8192 "<query>" <repo>`, invoked with
`cwd=<bgrep repo root>` so `uv run` resolves roust's own project
environment -- and returns the packed bundle text verbatim as the tool
result. This measures the actual shipped tool, not a reimplementation.

`rag_search(query)` (`rag_tool.py`) retrieves the top-`k` chunks
(`all-MiniLM-L6-v2`, 40-line non-overlapping windows, cosine similarity)
against the query. **`k` is raised from v1's 12 to 24** so the returned
bundle lands at roughly the same token budget as roust's fixed 8192 --
v1's top-12 gave only ~4k tokens, which starved the RAG baseline relative
to roust's budget and made that comparison unfair (see v1's README
"RAG bundle is smaller than roust's"). Calibrated pre-run against two real
(repo, query) pairs:

    django/django,   k=24 -> 8539 tiktoken tokens
    astropy/astropy, k=24 -> 8808 tiktoken tokens

`summarize.py`'s fairness-audit table reports the **actual** mean/median
tokens returned per `roust`/`rag_search` call across the real run (not
just this pre-run calibration), broken out per arm, so the budget match is
auditable against real data.

**`max_turns = 30`** (v1's 15 was demonstrably too tight -- every v1 row
hit it).

**Prompt fix -- reward stopping.** The shared system-prompt tail now
reads: *"Answer with FILES: as soon as you are confident. Additional
exploration costs tokens and does not improve your score. Do not verify
beyond what you need -- once the evidence converges on the right file(s),
stop investigating and answer immediately; do not spend further turns
double-checking a conclusion you already trust."* (v1's grep condition, in
its one closely-read transcript, found the exact bug on turn 13 of 15 and
then burned its remaining turns re-verifying instead of answering.)

**Same instance set as v1**: `common.load_instances(stride=10)` over
SWE-bench Lite (300 -> 30 instances), matching the existing
`lab/swebench_driver2.py --sample` convention -- kept identical to v1 for
comparability across the rebuild.

**Success**: unchanged from v1 -- parse the agent's final message for a
`FILES:` line (lenient: strips backticks/quotes/markdown emphasis,
tolerates comma- or whitespace-separated lists, takes the last such line
if the model restates it). Success iff every gold file (from the
instance's patch, via `diff --git a/<path> b/`) is present in the parsed
list -- extra, non-gold files in the answer do not count against success.
Exhausting `max_turns` without a `FILES:` line is scored as failure with
its tokens still counted.

**Tokens**: unchanged from v1 -- both `tiktoken_*_tokens` (cl100k_base
over actual message text sent/received, summed across all turns -- the
headline metric, since it reflects the real cost of resending a growing
conversation history every turn) and `api_*_tokens` (Anthropic-reported
`usage.input_tokens`/`output_tokens`, used for the `$` figure) are
recorded on every result row. **v2 adds** `tool_call_log`: one entry per
tool call (`{"tool", "tokens", "retrieval", ...}`), the basis for the
per-arm fairness audit.

**Cost**: Sonnet 4.5 standard published pricing, $3/MTok input, $15/MTok
output (unchanged from v1; see v1/README.md for the citation/date and the
note that this is a conservative upper bound relative to Anthropic's
current promotional $2/$10 rate).

**Guardrails kept from v1** (unchanged, see `sandbox_tools.py`):
`read_file` hard-capped at 400 lines/call; tool output truncated at 8000
chars/call; `run_command` whitelist (`rg`/`grep`/`find`/`ls`/`head`/`sed`
range-read) executed with `shell=False` (pipes/redirects/chaining are
inert, not dangerous, and the tool description says so up front so the
agent doesn't waste a turn expecting shell semantics); both tools confined
to the repo root (no absolute paths, no `..` escape).

## Layout

```
lab/tokenbench/
  common.py        instance loading (SWE-bench Lite, stride sample), repo
                    clone/checkout, tiktoken cl100k_base counting, lenient
                    `FILES:` line parsing + success scoring (unchanged from v1)
  sandbox_tools.py  run_command whitelist + read_file, now exported as
                    individual tool schemas (RUN_COMMAND_TOOL, READ_FILE_TOOL)
                    so agent.py can compose per-arm toolbelts
  roust_tool.py     roust(query) AGENT TOOL (v2: was injected context in v1) --
                     shells out to the real `uv run roust --json --budget 8192` CLI
  rag_index.py      embedding infra (chunking, index build/cache, cosine
                    top-k retrieval) -- unchanged from v1
  rag_tool.py       rag_search(query) AGENT TOOL (v2: was injected context in
                    v1) -- wraps rag_index.py, k raised 12->24 to budget-match roust
  agent.py          the shared Anthropic tool-use loop: per-arm system
                    prompt/toolbelt composition, turn cap, per-turn tiktoken +
                    API-usage accounting, per-tool-call token logging, JSONL
                    transcript logging
  mock_client.py    scripted fake Anthropic client for wiring validation
                    without a key/network/spend (NOT used for the real run)
  run_bench.py      orchestration CLI: sweeps instances x arms, resume-safe,
                    cost-capped
  summarize.py      reads results.jsonl -> success rate / token / cost
                    tables, matched-success cell (all 4, with pairwise
                    A-B/A-C/C-D fallback), fairness audit
  rag_cache/        on-disk embedding cache keyed by (repo, commit) --
                    shared with v1 (embeds ALL chunks; k only changes how
                    many of the cached embeddings are returned, so no need
                    to re-embed for the higher v2 k)
  .venv/            dedicated venv (anthropic, sentence-transformers,
                    tiktoken, pandas/pyarrow) -- shared with v1
  results.jsonl     (created by a real run) one row per (instance, arm)
  transcripts/      (created by a real run) one JSONL file per
                    (instance, arm) with every request/response
  v1/               ARCHIVED v1 harness + its results.jsonl/transcripts/
                    README.md, kept for the record (see "Why v2" above);
                    not imported by anything in v2
```

## Validation performed before the real run

- `common.py`/`sandbox_tools.py` unit checks (parse_files_line,
  files_match, run_command whitelist/rejection) re-run against v2's copies
  -- pass (identical logic to v1, which was already validated live against
  a real checkout).
- End-to-end wiring, all four arms, via `mock_client.py` (scripted fake
  Anthropic client, zero LLM spend) for one real instance
  (`astropy__astropy-12907`): confirmed real tool dispatch --
  **`roust_tool.roust_search` genuinely shells out to `uv run roust`**
  (observed a real 8439-token bundle, 873 files indexed, cache=miss) and
  **`rag_tool.rag_search` genuinely retrieves via the real embedding
  index** (observed a real 7738-token, 20-chunk result at k=24) -- message
  threading, per-turn JSONL transcript logging, `tool_call_log`
  token-per-call accounting, `FILES:` parsing, and resume-safety all
  behaved as designed. (Mock LLM responses are scripted and not
  representative of real model behavior -- this validates the harness
  plumbing, not success rates.)
- `--budget-cap-usd` stop behavior and resume-safety: inherited unchanged
  from v1's validated `run_bench.py` logic (same cost-tracking and
  already-done-pairs skip code path).

## Running

```bash
cd /Users/nicholasarehart/programming-projects/bgrep
ANTHROPIC_API_KEY=... lab/tokenbench/.venv/bin/python3 lab/tokenbench/run_bench.py \
    --stride 10 --budget-cap-usd 80 \
    --out lab/tokenbench/results.jsonl \
    --transcripts-dir lab/tokenbench/transcripts
lab/tokenbench/.venv/bin/python3 lab/tokenbench/summarize.py \
    --results lab/tokenbench/results.jsonl
```

Resume-safe (rerun the same command to pick up where a partial/capped run
left off); pass `--arms grep,roust` etc. to run a subset;
`--budget-cap-usd` to change the spend cap (default $20, this run used
$80 per the $100-authorized budget).

## Run status: 58/120 pairs completed, stopped at the $80 cap

The real run executed 2026-07-13, gated on the coordination check below.
`--budget-cap-usd 80` (of the $100 authorized) was reached mid-sweep, as
designed -- the harness stopped cleanly rather than exceeding it (final
spend $81.13, one row over the cap since the check is between rows, not
mid-row). **15 instances got the full grep/roust/roust_grep/rag_grep
sweep, plus one 16th instance partially (grep + roust only)** -- 58/58
result rows have a matching transcript, zero harness errors. See
`summarize.py` output (reproduced in the delivered report) for the actual
numbers; headline: roust alone succeeded on 14/15 instances (93%) with a
median of 8 turns and zero turn-cap hits, a stark contrast to v1's 0-8%
success / universal turn-cap-out. Unlike v1, this is a real,
budget-limited partial sample, not a design failure -- convergence is
happening; the cap simply bounds how much of the 30-instance set was
affordable within the authorized spend. Rerun with a higher
`--budget-cap-usd` (resume-safe) to extend coverage.

## Known accounting limitations

- **No prompt caching.** The harness sets no `cache_control`; every turn
  re-bills the full transcript at $3/MTok. Real deployments cache the
  prefix. A cache-adjusted recomputation (`cache_cost_analysis.py`) shows
  costs drop ~72-81% for all arms -- but the roust/grep ratio WIDENS from
  1.48x to 1.80x, because caching bills writes at a premium ($3.75) and
  reads at a discount ($0.30), and roust's few-big-bundles shape puts 2.5x
  more of its tokens in the write bucket (write/read 0.167 vs grep 0.068).
  roust's 2x advantage over rag_grep narrows to 1.35x but survives.
  Negative result recorded so nobody re-litigates "just enable caching."
- **Turn-cap floor.** 73-80% of grep runs hit the 30-turn cap and stop
  mid-task; grep's measured $/attempt is a truncated floor, not a
  completion cost. Both limitations bias AGAINST roust, so the published
  roust-vs-grep comparison is conservative.

## Coordination note

At harness-build time, `src/roust/` and `roust-rs/` had uncommitted
changes from a concurrent gitignore-fix agent. Since `roust_tool.py` shells
out to the real installed CLI, running the benchmark mid-edit could mix
tool versions across rows within a single run. The real run was gated on
`git status --short src/roust/ roust-rs/` returning clean AND
`.venv-pkg/bin/roust --help` succeeding before any (instance, arm) pair was
executed.

## Engine provenance guard (stale-`uv run roust`-wheel incident)

`uv run roust` (what `roust_tool.py` shells out to) serves a cached
build of the `roust-rs` binary that does **not** rebuild automatically when
`roust-rs/src` changes -- confirmed 2026-07-13: after merging engine
changes, the venv binary stayed stale until `uv sync --reinstall-package
roust` was run. Sibling failure mode to the "coordination note" above and to
issue #8. **Any Rust engine change requires `uv sync --reinstall-package
roust` before the next bench run** -- otherwise the bench may silently
measure the old engine.

`run_bench.py` now enforces this automatically at startup, before any
(instance, arm) pair runs: it reads the running binary's embedded provenance
(`uv run roust --version` -> `roust 0.2.0 (<git sha>, clean|dirty)`, wired
via `roust-rs/build.rs`) and compares it against the repo's current
`roust-rs/`-scoped git state. A sha mismatch or any dirtiness (embedded or
current) is a hard `SystemExit` naming the fix command; `--allow-stale-engine`
downgrades this to a loud warning for deliberate stale-engine testing. Every
result row records the parsed `engine_version` string, so provenance is
in the data forever, not just in a one-time gate check.
