#!/usr/bin/env python3
"""
ESTIMATE: recompute tokenbench $/attempt under a prompt-caching accounting
model, using REAL per-turn Anthropic API usage numbers already recorded in
the transcripts (transcripts/*.jsonl, repeat_transcripts/*.jsonl). No API
calls are made; this is a pure re-aggregation of existing usage.input_tokens
/ usage.output_tokens fields under a different price schedule + a caching
assumption about which bytes are "new" vs "already-cached prefix" each turn.

Read-only: does not touch/modify anything under lab/tokenbench/.

--- Pricing (Sonnet 4.5, stated in the task; verified against
    lab/tokenbench/common.py::PRICE_INPUT_PER_MTOK=3.0 /
    PRICE_OUTPUT_PER_MTOK=15.0 for the no-cache baseline -- matches) ---
    no-cache:    input $3.00/MTok,  output $15.00/MTok   (measured baseline,
                 reproduces common.py/summarize.py row_cost exactly)
    cache WRITE: $3.75/MTok  (1.25x base input -- Anthropic's standard
                 write-premium ratio)
    cache READ:  $0.30/MTok  (0.1x base input -- Anthropic's standard
                 read-discount ratio)
    output:      $15.00/MTok always (caching does not affect output pricing)

--- Turn-level cache model ---
Each transcript line `t` records `response.usage.{input_tokens,output_tokens}`
which is the REAL, Anthropic-measured size of the ENTIRE conversation sent as
input for API call `t` (the harness has no cache_control, so every turn's
request_messages is user-msg + full prior turn history + new tool_result,
re-sent from scratch and billed in full -- this is exactly what
`ai_t = usage.input_tokens` already is).

Under caching, we assume the model would send an identical byte stream per
turn (same conversation growth), but the PREFIX common to turn t-1 and turn
t is served from cache instead of re-billed as fresh input:

  turn 1:            all ai_1 tokens are new -> billed at cache-WRITE $3.75
  turn t>1:  prefix = ai_(t-1)   (everything already sent as of the previous
                                   turn's call -- billed at cache-READ $0.30)
             new    = max(0, ai_t - ai_(t-1))  (the new suffix appended this
                                   turn: prior turn's assistant text/tool_use
                                   + this turn's tool_result -- billed at
                                   cache-WRITE $3.75, since it is being seen/
                                   cached for the first time)
  output_t:  always ao_t tokens at $15/MTok, unaffected by caching.

  cost_t = prefix*0.30/1e6 + new*3.75/1e6 + ao_t*15/1e6         (t>1)
  cost_1 = ai_1*3.75/1e6 + ao_1*15/1e6

This is the harness's own no-cache accounting (ai_t billed in full every
turn at $3/MTok flat) with the SAME per-turn token boundaries, just resliced
into read/write buckets at a different price. It assumes: (a) perfect cache
hits on the exact prefix (no cache-eviction/TTL misses -- real deployments
would occasionally miss and re-pay the write premium), (b) tokenization is
stable enough that ai_t - ai_(t-1) is a fair proxy for "bytes added this
turn" (true up to BPE re-tokenization noise at chunk boundaries, which the
guard `max(0, ...)` suppresses when it goes slightly negative).

--- Data pooling ---
Trial 0: transcripts/{instance}__{arm}.jsonl        (paired with results.jsonl)
Trial 1: repeat_transcripts/{instance}__{arm}__t1.jsonl (paired with
         results_repeats.jsonl; roust/grep/rag_grep only, no roust_grep)
Both trials are pooled per arm (attempt-level rows, not instance-averaged)
for roust/grep/rag_grep, matching the task's "both trials pooled where
available" instruction. roust_grep is trial-0-only and is NOT one of the
three arms asked for, so it is loaded but not reported in the headline table.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

TB_DIR = Path("/Users/nicholasarehart/programming-projects/bgrep/lab/tokenbench")

PRICE_IN_NOCACHE = 3.00
PRICE_OUT = 15.00
PRICE_CACHE_WRITE = 3.75
PRICE_CACHE_READ = 0.30

ARMS_OF_INTEREST = ["grep", "roust", "rag_grep"]


def load_results(path: Path) -> dict[tuple[str, str, int], dict]:
    """(instance_id, arm, trial) -> result row."""
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        trial = r.get("trial", 0)
        out[(r["instance_id"], r["arm"], trial)] = r
    return out


def turn_usages(transcript_path: Path) -> list[tuple[int, int]]:
    """[(input_tokens, output_tokens), ...] sorted by turn, from real API usage."""
    lines = [json.loads(l) for l in transcript_path.read_text().splitlines() if l.strip()]
    lines.sort(key=lambda r: r["turn"])
    return [(l["response"]["usage"]["input_tokens"], l["response"]["usage"]["output_tokens"])
            for l in lines]


def nocache_cost(usages: list[tuple[int, int]]) -> float:
    ai = sum(u[0] for u in usages)
    ao = sum(u[1] for u in usages)
    return ai / 1e6 * PRICE_IN_NOCACHE + ao / 1e6 * PRICE_OUT


def cached_cost(usages: list[tuple[int, int]]) -> float:
    total = 0.0
    prev_ai = None
    for ai_t, ao_t in usages:
        if prev_ai is None:
            total += ai_t / 1e6 * PRICE_CACHE_WRITE
        else:
            prefix = prev_ai
            new = max(0, ai_t - prev_ai)
            total += prefix / 1e6 * PRICE_CACHE_READ + new / 1e6 * PRICE_CACHE_WRITE
        total += ao_t / 1e6 * PRICE_OUT
        prev_ai = ai_t
    return total


def cache_breakdown(usages: list[tuple[int, int]]) -> tuple[int, int, int]:
    """(write_tokens, read_tokens, output_tokens) under the same turn model
    as cached_cost(), for diagnosing WHY the reduction% differs by arm."""
    write_tok = read_tok = out_tok = 0
    prev = None
    for ai_t, ao_t in usages:
        out_tok += ao_t
        if prev is None:
            write_tok += ai_t
        else:
            read_tok += prev
            write_tok += max(0, ai_t - prev)
        prev = ai_t
    return write_tok, read_tok, out_tok


def collect(transcripts_dir: Path, suffix_strip: str, trial: int, results: dict) -> list[dict]:
    """Return per-attempt rows: instance_id, arm, trial, nocache$, cached$, success."""
    out = []
    for f in sorted(transcripts_dir.glob("*.jsonl")):
        base = f.name[:-len(".jsonl")]
        if suffix_strip and base.endswith(suffix_strip):
            base = base[: -len(suffix_strip)]
        instance_id, arm = base.rsplit("__", 1)
        usages = turn_usages(f)
        if not usages:
            continue
        row = results.get((instance_id, arm, trial))
        success = bool(row.get("success")) if row else None
        excluded = row.get("status", "ok") in {"api_error", "aborted_over_budget"} if row else False
        write_tok, read_tok, out_tok = cache_breakdown(usages)
        out.append({
            "instance_id": instance_id, "arm": arm, "trial": trial,
            "nocache": nocache_cost(usages), "cached": cached_cost(usages),
            "success": success, "excluded": excluded,
            "n_turns": len(usages),
            "write_tok": write_tok, "read_tok": read_tok, "out_tok": out_tok,
        })
    return out


def main() -> None:
    results0 = load_results(TB_DIR / "results.jsonl")
    results1 = load_results(TB_DIR / "results_repeats.jsonl")

    rows = []
    rows += collect(TB_DIR / "transcripts", "", 0, results0)
    rows += collect(TB_DIR / "repeat_transcripts", "__t1", 1, results1)

    # sanity check: reproduce results.jsonl's own row_cost() (3.0/15.0 flat)
    # for a handful of trial-0 rows to confirm our nocache_cost() matches the
    # harness's existing accounting exactly (not just "close").
    print("=== sanity check: our nocache_cost() vs results.jsonl row_cost() (trial 0) ===")
    max_abs_diff = 0.0
    n_checked = 0
    for r in rows:
        if r["trial"] != 0:
            continue
        res = results0.get((r["instance_id"], r["arm"], 0))
        if not res:
            continue
        their_cost = (res.get("api_input_tokens", 0) or 0) / 1e6 * PRICE_IN_NOCACHE \
            + (res.get("api_output_tokens", 0) or 0) / 1e6 * PRICE_OUT
        diff = abs(their_cost - r["nocache"])
        max_abs_diff = max(max_abs_diff, diff)
        n_checked += 1
    print(f"checked {n_checked} rows, max |diff| = ${max_abs_diff:.6f} "
          f"({'MATCH' if max_abs_diff < 1e-6 else 'MISMATCH'})\n")

    by_arm = defaultdict(list)
    for r in rows:
        if r["excluded"]:
            continue  # infra hiccups, not task outcomes -- same convention as summarize.py
        by_arm[r["arm"]].append(r)

    print(f"{'arm':11} {'n':>4} {'succ%':>7} {'$/attempt':>10} {'$/attempt':>10} {'reduction':>10} "
          f"{'$/success':>10} {'$/success':>10}")
    print(f"{'':11} {'':>4} {'':>7} {'no-cache':>10} {'cached':>10} {'%':>10} "
          f"{'no-cache':>10} {'cached':>10}")

    summary = {}
    for a in ARMS_OF_INTEREST:
        arows = by_arm.get(a, [])
        n = len(arows)
        succ = [r for r in arows if r["success"]]
        nocache_mean = stats.mean(r["nocache"] for r in arows)
        cached_mean = stats.mean(r["cached"] for r in arows)
        reduction = (1 - cached_mean / nocache_mean) * 100
        nocache_succ = stats.mean(r["nocache"] for r in succ) if succ else float("nan")
        cached_succ = stats.mean(r["cached"] for r in succ) if succ else float("nan")
        succ_pct = len(succ) / n * 100 if n else float("nan")
        summary[a] = dict(n=n, nocache_mean=nocache_mean, cached_mean=cached_mean,
                           nocache_succ=nocache_succ, cached_succ=cached_succ,
                           succ_pct=succ_pct, n_succ=len(succ))
        print(f"{a:11} {n:>4} {succ_pct:>6.1f}% {nocache_mean:>10.3f} {cached_mean:>10.3f} "
              f"{reduction:>9.1f}% {nocache_succ:>10.3f} {cached_succ:>10.3f}")

    print()
    g, r_, rg = summary["grep"], summary["roust"], summary["rag_grep"]
    print("=== headline ratios ===")
    print(f"roust/grep  $/attempt  no-cache: {r_['nocache_mean']/g['nocache_mean']:.3f}x   "
          f"cached: {r_['cached_mean']/g['cached_mean']:.3f}x")
    print(f"roust/grep  $/success  no-cache: {r_['nocache_succ']/g['nocache_succ']:.3f}x   "
          f"cached: {r_['cached_succ']/g['cached_succ']:.3f}x")
    print(f"rag_grep/roust $/attempt no-cache: {rg['nocache_mean']/r_['nocache_mean']:.3f}x   "
          f"cached: {rg['cached_mean']/r_['cached_mean']:.3f}x")
    print(f"rag_grep/grep  $/attempt no-cache: {rg['nocache_mean']/g['nocache_mean']:.3f}x   "
          f"cached: {rg['cached_mean']/g['cached_mean']:.3f}x")

    print("\n=== why the reduction% differs by arm: mean new(write)/read tokens per attempt ===")
    print(f"{'arm':11} {'n':>4} {'avg_write_tok':>14} {'avg_read_tok':>13} {'avg_out_tok':>12} "
          f"{'write/read':>11}")
    for a in ARMS_OF_INTEREST:
        arows = by_arm.get(a, [])
        n = len(arows)
        aw = sum(r["write_tok"] for r in arows) / n
        ar = sum(r["read_tok"] for r in arows) / n
        ao = sum(r["out_tok"] for r in arows) / n
        print(f"{a:11} {n:>4} {aw:>14.0f} {ar:>13.0f} {ao:>12.0f} {aw/ar:>10.3f}x")

    print()
    for a in ARMS_OF_INTEREST:
        print(f"  {a}: n={summary[a]['n']}  n_success={summary[a]['n_succ']}")


if __name__ == "__main__":
    main()
