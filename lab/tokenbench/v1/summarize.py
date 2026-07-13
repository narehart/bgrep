#!/usr/bin/env python3
"""Reads results.jsonl and prints the headline summary tables:
per-condition success rate, median/mean tiktoken total tokens, tool calls,
wall-clock; tokens AT MATCHED SUCCESS (the apples-to-apples cell: instances
all three conditions solved); and $ cost via current Sonnet pricing on the
real API-reported usage tokens.

Usage: lab/tokenbench/.venv/bin/python3 lab/tokenbench/summarize.py \\
           [--results lab/tokenbench/results.jsonl]
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

TOKENBENCH_DIR = Path(__file__).resolve().parent
CONDITIONS = ["grep", "roust", "rag"]
PRICE_INPUT_PER_MTOK = 3.0
PRICE_OUTPUT_PER_MTOK = 15.0


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _fmt(x, nd=0):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def row_cost(row: dict) -> float:
    ai = row.get("api_input_tokens", 0) or 0
    ao = row.get("api_output_tokens", 0) or 0
    return ai / 1e6 * PRICE_INPUT_PER_MTOK + ao / 1e6 * PRICE_OUTPUT_PER_MTOK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(TOKENBENCH_DIR / "results.jsonl"))
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    rows = load_rows(path)

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    instances = sorted({r["instance_id"] for r in rows})
    print(f"=== tokenbench summary: {len(instances)} instances, {len(rows)} rows, "
          f"file={path} ===\n")

    print(f"{'condition':10} {'n':>4} {'success%':>9} {'med_tok':>9} {'mean_tok':>9} "
          f"{'med_tools':>10} {'mean_wall_s':>12} {'total_cost$':>12}")
    for c in CONDITIONS:
        crows = by_cond.get(c, [])
        if not crows:
            print(f"{c:10} (no rows)")
            continue
        n = len(crows)
        succ = sum(1 for r in crows if r.get("success")) / n * 100
        toks = [r.get("tiktoken_total_tokens", 0) for r in crows if "tiktoken_total_tokens" in r]
        tools = [r.get("tool_calls", 0) for r in crows if "tool_calls" in r]
        walls = [r.get("wall_clock_s", 0) for r in crows if "wall_clock_s" in r]
        cost = sum(row_cost(r) for r in crows)
        print(f"{c:10} {n:>4} {succ:>8.1f}% {_fmt(stats.median(toks) if toks else None):>9} "
              f"{_fmt(stats.mean(toks) if toks else None, 0):>9} "
              f"{_fmt(stats.median(tools) if tools else None):>10} "
              f"{_fmt(stats.mean(walls) if walls else None, 1):>12} "
              f"{cost:>11.3f}")

    # apples-to-apples cell: instances where ALL THREE conditions succeeded
    success_by_instance: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("success"):
            success_by_instance[r["instance_id"]].add(r["condition"])
    matched = [iid for iid, conds in success_by_instance.items()
               if all(c in conds for c in CONDITIONS)]
    print(f"\n=== tokens AT MATCHED SUCCESS (n={len(matched)} instances all 3 conditions solved) ===")
    if not matched:
        print("(no instance was solved by all three conditions -- cannot report the matched-success cell)")
    else:
        print(f"{'condition':10} {'n':>4} {'med_tok':>9} {'mean_tok':>9} {'med_tools':>10} {'mean_wall_s':>12}")
        for c in CONDITIONS:
            crows = [r for r in by_cond.get(c, []) if r["instance_id"] in matched]
            toks = [r.get("tiktoken_total_tokens", 0) for r in crows]
            tools = [r.get("tool_calls", 0) for r in crows]
            walls = [r.get("wall_clock_s", 0) for r in crows]
            print(f"{c:10} {len(crows):>4} {_fmt(stats.median(toks) if toks else None):>9} "
                  f"{_fmt(stats.mean(toks) if toks else None, 0):>9} "
                  f"{_fmt(stats.median(tools) if tools else None):>10} "
                  f"{_fmt(stats.mean(walls) if walls else None, 1):>12}")

    total_cost = sum(row_cost(r) for r in rows)
    print(f"\ntotal estimated cost across all rows: ${total_cost:.3f} "
          f"(Sonnet 4.5 standard pricing: ${PRICE_INPUT_PER_MTOK}/MTok in, "
          f"${PRICE_OUTPUT_PER_MTOK}/MTok out, real API-usage tokens -- NOT the tiktoken metric)")

    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"\n{len(errors)} rows errored (counted as failures per spec):")
        for r in errors[:10]:
            print(f"  {r['instance_id']:44} {r['condition']:6} {r['error'][:120]}")


if __name__ == "__main__":
    main()
