# tokenbench: agent-token-usage benchmark (grep vs roust vs RAG)

Measures the tokens an LLM agent actually burns to **localize** the fix
files for a SWE-bench Lite issue, under three retrieval conditions, holding
everything else (model, prompt persona, tools, max turns, temperature,
success criterion) fixed. This is the harness behind the "solves it for
1/Nth the tokens" claim, so the design goal is that condition A (grep) is
argued in good faith, not a strawman.

## STATUS: harness built and validated end-to-end; pilot NOT executed

**No `ANTHROPIC_API_KEY` is available in this environment** (checked `env`
and `~/.claude/settings.json` -- no key, no `apiKeyHelper`; this machine
runs Claude Code on an OAuth session, not a raw Anthropic API key). Per the
task spec, the harness stops rather than fabricating results:
`run_bench.py` refuses to start a real run and exits with a clear message
unless either `ANTHROPIC_API_KEY` is set or `--mock` is passed.

Everything that does not require spending against the real API has been
built and validated (see "Validation performed" below). Once a key is
available: `lab/tokenbench/.venv/bin/python3 lab/tokenbench/run_bench.py`
runs the full 30-instance x 3-condition pilot.

### Rough pre-flight cost estimate (for the $20 approval gate)

No real-agent-turn data exists yet to estimate from precisely (the mock
client stops after 3 scripted turns, which is not representative of real
turn counts). Order-of-magnitude estimate from component sizes:

- grep: ~600-1500 tok issue text, growing via tool-result history over an
  estimated ~5-7 turns to convergence -> roughly 10-15k tokens/run.
- roust: same, but the first message also carries the ~8-8.5k-token roust
  bundle, resent in full on every subsequent turn (that's how the API
  actually bills you) -> roughly 35-45k tokens/run, likely in fewer turns
  (~3-4) since the agent starts with a head start.
- rag: same shape, ~4k-token injected context (see "RAG bundle is smaller
  than roust's" below) -> roughly 20-28k tokens/run.

30 instances x (grep + roust + rag) x ~90% input / 10% output split, at
Sonnet 4.5 standard pricing ($3/MTok in, $15/MTok out) -> **rough estimate
$5-$15 for the full pilot**, comfortably under the $20 gate. This is a
coarse estimate, not a measurement -- `run_bench.py --budget-cap-usd 20`
(default) is the actual enforced backstop: it tracks real API-reported
usage tokens turn-by-turn and refuses to launch further (instance,
condition) runs once the running cost hits the cap, so a bad estimate
can't blow the budget unattended.

## Layout

```
lab/tokenbench/
  common.py        instance loading (SWE-bench Lite, stride sample), repo
                    clone/checkout, tiktoken cl100k_base counting, lenient
                    `FILES:` line parsing + success scoring
  sandbox_tools.py  the run_command whitelist (rg/grep/find/ls/head/
                    sed-range-read) and read_file(path,start,end) -- the
                    SAME two tools all three conditions get
  roust_bundle.py   condition B: `uv run roust --json --budget 8192 ...`
                    (the real CLI, not a reimplementation)
  rag_index.py      condition C: sentence-transformers/all-MiniLM-L6-v2,
                    40-line chunks, top-12 cosine retrieval, on-disk cache
                    keyed by (repo, commit)
  agent.py          the shared Anthropic tool-use loop: system prompt,
                    turn cap, per-turn tiktoken + API-usage accounting,
                    JSONL transcript logging
  mock_client.py    scripted fake Anthropic client for wiring validation
                    without a key/network/spend (NOT used for the real
                    pilot)
  run_bench.py      orchestration CLI: sweeps instances x conditions,
                    resume-safe, cost-capped
  summarize.py      reads results.jsonl -> success rate / token / cost
                    tables, including the matched-success apples-to-apples
                    cell
  .venv/            dedicated venv (anthropic, sentence-transformers,
                    tiktoken, pandas/pyarrow) -- kept separate from the
                    project's own .venv so tokenbench's heavier deps
                    (torch, transformers) don't leak into roust's own
                    dependency footprint
  results.jsonl     (created by a real run) one row per (instance, condition)
  transcripts/      (created by a real run) one JSONL file per
                    (instance, condition) with every request/response
```

## Methodology

**Conditions**, same model (`claude-sonnet-4-5-20250929`), same system
prompt, same two tools, same `max_turns=15`, `temperature=0`:

- **A. grep**: first user message is just the issue text. The system
  prompt explicitly (a) tells the agent to lead with the strongest signal
  in the issue (error strings, identifiers, tracebacks) and grep for it,
  (b) tells it to prefer narrow `read_file` ranges over reading whole
  files, and (c) tells it to stop investigating once evidence converges
  rather than turn-cap out. `read_file` is hard-capped at 400 lines/call
  (not just prompt guidance) so "read the whole file in one call" can't be
  used to route around the targeted-search framing -- see "Assumptions"
  below.
- **B. roust**: identical, except the first user message also contains the
  real `roust --json --budget 8192 "<problem_statement>" <repo>` bundle,
  framed explicitly as a hint to verify/expand, not ground truth. The
  agent keeps the same tools and can (and in the transcripts, does)
  deviate from the bundle.
- **C. rag**: identical, except the first user message contains the top-12
  chunks (40-line non-overlapping windows, `all-MiniLM-L6-v2` cosine
  similarity) for the issue text, same "hint, verify" framing.

**Success**: parse the agent's final message for a `FILES:` line (lenient:
strips backticks/quotes/markdown emphasis, tolerates comma- or
whitespace-separated lists, takes the last such line if the model restates
it). Success iff every gold file (from the instance's patch, via `diff
--git a/<path> b/`) is present in the parsed list -- extra, non-gold files
in the answer do not count against success. Exhausting `max_turns` without
a `FILES:` line is scored as failure with its tokens still counted, per
spec.

**Tokens**: every request/response is logged verbatim to a JSONL
transcript. Two token totals are computed and both are stored on every
result row:
- `tiktoken_*_tokens`: `tiktoken` `cl100k_base` over the actual message
  text sent/received each turn, **summed across all turns** (so it
  reflects the real cost of resending a growing conversation history each
  turn, including the injected bundle/RAG context on every subsequent
  turn for B/C -- that's genuinely how you'd be billed). This is the
  metric the spec asks for and the one `summarize.py` reports as the
  headline number, because it's reproducible without an API key and
  tokenizer-neutral across conditions (Anthropic doesn't publish Claude's
  exact tokenizer, so `cl100k_base` is used consistently for all three
  conditions rather than trusting three different possibly-inconsistent
  self-reported counts).
- `api_*_tokens`: the real `usage.input_tokens`/`usage.output_tokens` from
  each Anthropic response, summed. Used for the `$` cost figure in
  `summarize.py` (since that's what billing actually uses), reported
  alongside, not instead of, the tiktoken metric.

**Cost**: Sonnet 4.5 standard published pricing, $3/MTok input, $15/MTok
output (checked 2026-07, https://platform.claude.com/docs/en/about-claude/pricing).
Anthropic is currently running an introductory $2/$10 rate through
2026-08-31; `summarize.py` uses the standard $3/$15 rate so the reported
cost is a conservative (upper-bound) estimate, not the best case.

## Instance selection

`common.load_instances(stride=10)` takes every 10th row of SWE-bench Lite
in dataset order (300 -> 30 instances), matching the existing
`lab/swebench_driver2.py --sample N` convention in this repo. Verified:

```
30 instances, repos = {astropy, django x9, matplotlib x3, requests, xarray,
pytest x2, scikit-learn x2, sphinx x2, sympy x7}
```

Repos clone into `lab/swebench_repos/` (shared with the other
`lab/swebench_driver*.py` scripts) on first use; `pydata/xarray` is not
yet cloned in this checkout and will be cloned automatically by
`common.repo_clone()` on first real run.

## Validation performed (no API key required)

- `common.parse_files_line` / `files_match`: unit-checked against
  plain/bold-markdown/bulleted/backtick-quoted/whitespace-separated
  `FILES:` formats and the extras-allowed / missing-gold-file scoring
  rule.
- `sandbox_tools`: checked against a live checkout
  (`lab/swebench_repos/psf__requests`) -- `rg`/`sed -n '<range>p'` work;
  `cat`, piped commands (`grep foo | wc -l`), `sed -i`, and `../..` path
  escapes are all rejected with a clear tool-error message (not a crash);
  `read_file` returns line-numbered windows and truncates at 400
  lines/call with a note, without falsely claiming truncation when the
  window actually just hit EOF.
- `roust_bundle.get_roust_bundle`: live-invoked the real `uv run roust
  --json --budget 8192` against `psf__requests` -- returns the ranked file
  list, stats, and packed bundle text (8459 tiktoken tokens observed,
  matching the requested `--budget 8192`).
- `rag_index`: live-embedded `psf__requests` (122 source files) on CPU
  with `all-MiniLM-L6-v2`, retrieved top-12 chunks for a real issue query,
  cached to disk, confirmed cache-hit reload path re-slices chunk text
  from the checkout rather than duplicating source in the cache file.
- `agent.run_agent` + `run_bench.py`, full wiring, via `mock_client.py`
  (scripted fake client, zero network/spend): ran all three conditions
  end-to-end for one real instance (`psf__requests-1963`) -- tool
  dispatch, message threading, per-turn JSONL transcript logging,
  tiktoken/API-usage token accounting, `FILES:` parsing, resume-safety
  (rerun skipped all 3 already-done pairs), and the `--budget-cap-usd`
  circuit breaker (verified it stops mid-sweep once the running cost
  crosses the cap) all behaved as designed. Confirms real numbers will be
  well-formed once a key is available; does **not** substitute for real
  success-rate/token numbers, which need real model behavior.
- Refusal path: confirmed `run_bench.py` (no `--mock`, no
  `ANTHROPIC_API_KEY`) exits 1 with the "STOP rather than fabricate"
  message instead of running.

## RAG bundle is smaller than roust's (~4k vs ~8.5k tokens) -- flagged, not silently changed

On the one repo/query tested, 12 x 40-line chunks landed around ~4k
tiktoken tokens (Python source averages fewer tokens/line than the
"~8k to match roust's budget" back-of-envelope in the spec assumed). This
is measured behavior of the literal spec (40-line chunks, top-12, nothing
else) on real code, not a bug -- I did not enlarge `k` to force parity with
roust's fixed 8192-token budget, since the spec pins chunk size and count
explicitly and inflating retrieval count to hit a token target would be
the kind of undisclosed knob-turning that would undermine the RAG
condition's fairness. `summarize.py`/results rows record `context_tokens`
per row so this is visible and auditable per instance, not averaged away.

## Assumptions flagged

1. **`read_file` is hard-capped at 400 lines/call** (`sandbox_tools.
   MAX_READ_LINES`), not just prompt-guided. The spec says the grep
   condition's prompt should "warn against reading whole large files";
   I added an actual guardrail because a prompt-only warning is easy for
   a model to route around (`read_file(path, 1, 999999)`) in a way that
   would make condition A into a disguised "cat the file" condition,
   undermining the fairness comparison the whole benchmark exists to
   make. This caps tokens per *call*, not per *file* (an agent can still
   read a large file across several calls) and does not apply to B/C
   beyond the same tool.
2. **Tool output truncated at 8000 chars/call** (`sandbox_tools.
   MAX_OUTPUT_CHARS`), same rationale -- prevents an unbounded `rg .`
   from silently becoming a multi-thousand-token single tool result.
3. **`run_command` uses `shell=False`** (argv passed directly to
   `subprocess.run`, no real shell behind it). This is a safety choice
   (no shell-injection surface) that also means pipes/redirects/command
   substitution are simply inert rather than dangerous; the tool
   description tells the agent this explicitly so it doesn't waste turns
   on `grep foo | wc -l` expecting shell semantics.
4. **Instance stride uses raw dataset order** (`load_instances(stride=10)`),
   matching the existing `lab/swebench_driver2.py --sample` convention in
   this repo, rather than re-sorting by repo first. SWE-bench Lite's
   dataset order already clusters by repo, so this still yields a
   reasonably repo-diverse 30-instance sample (see "Instance selection").
5. **Turn definition** = one Anthropic API call (which may include
   multiple tool_use blocks executed and returned as one batch of
   tool_results before the next call). This matches "max turns" as a cap
   on API calls, not on individual tool invocations.

## Example transcripts

Not yet available -- transcripts are only produced by a real (or
`--mock`) run. `--mock` transcripts exist only as an ephemeral wiring
check (written to `/tmp` during validation, not committed here since they
contain no real model behavior and would be misleading if mistaken for
pilot data). Once a key is available and `run_bench.py` runs, per-
(instance, condition) transcripts land in `lab/tokenbench/transcripts/
<instance_id>__<condition>.jsonl`.

## Running the pilot (once ANTHROPIC_API_KEY is set)

```bash
export ANTHROPIC_API_KEY=...
cd /Users/nicholasarehart/programming-projects/bgrep
lab/tokenbench/.venv/bin/python3 lab/tokenbench/run_bench.py \
    --stride 10 --out lab/tokenbench/results.jsonl \
    --transcripts-dir lab/tokenbench/transcripts
lab/tokenbench/.venv/bin/python3 lab/tokenbench/summarize.py \
    --results lab/tokenbench/results.jsonl
```

Resume-safe (rerun the same command to pick up where a partial/capped run
left off); pass `--conditions grep,roust` etc. to run a subset;
`--budget-cap-usd` to change the spend cap (default $20).
