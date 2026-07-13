#!/usr/bin/env python3
"""Reads results.jsonl and prints the headline v2 summary tables:
per-arm success rate, median/mean tiktoken total tokens, tool calls, turns,
wall-clock, cost; tokens AT MATCHED SUCCESS (instances ALL FOUR arms
solved, falling back to the pairwise A-vs-B / A-vs-C / C-vs-D cells if that
set is empty); and the fairness audit -- mean tokens returned per tool
call, per arm, broken out by tool (so the roust-budget-8192 vs
rag_search-k=24 match is auditable against real run data, not just the
pre-run calibration in rag_tool.py).

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
ARMS = ["grep", "roust", "roust_grep", "rag_grep"]
# A/B/C/D labels per the v2 spec, used only for the pairwise matched-success fallback.
ARM_LETTER = {"grep": "A", "roust": "B", "roust_grep": "C", "rag_grep": "D"}
PAIRWISE_CELLS = [("grep", "roust"), ("grep", "roust_grep"), ("roust_grep", "rag_grep")]
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


def _print_token_table(title: str, by_arm_rows: dict[str, list[dict]]) -> None:
    print(title)
    print(f"{'arm':11} {'n':>4} {'med_tok':>9} {'mean_tok':>9} {'med_tools':>10} {'med_turns':>10} "
          f"{'mean_wall_s':>12}")
    for a in ARMS:
        rows = by_arm_rows.get(a, [])
        if not rows:
            continue
        toks = [r.get("tiktoken_total_tokens", 0) for r in rows]
        tools = [r.get("tool_calls", 0) for r in rows]
        turns = [r.get("turns_used", 0) for r in rows]
        walls = [r.get("wall_clock_s", 0) for r in rows]
        print(f"{a:11} {len(rows):>4} {_fmt(stats.median(toks)):>9} {_fmt(stats.mean(toks), 0):>9} "
              f"{_fmt(stats.median(tools)):>10} {_fmt(stats.median(turns)):>10} "
              f"{_fmt(stats.mean(walls), 1):>12}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(TOKENBENCH_DIR / "results.jsonl"))
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    rows = load_rows(path)

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r.get("arm", r.get("condition", "?"))].append(r)

    instances = sorted({r["instance_id"] for r in rows})
    print(f"=== tokenbench v2 summary: {len(instances)} instances, {len(rows)} rows, "
          f"file={path} ===\n")

    print(f"{'arm':11} {'n':>4} {'success%':>9} {'med_tok':>9} {'mean_tok':>9} "
          f"{'med_tools':>10} {'med_turns':>10} {'hit_cap%':>9} {'mean_wall_s':>12} {'total_cost$':>12}")
    for a in ARMS:
        arows = by_arm.get(a, [])
        if not arows:
            print(f"{a:11} (no rows)")
            continue
        n = len(arows)
        succ = sum(1 for r in arows if r.get("success")) / n * 100
        toks = [r.get("tiktoken_total_tokens", 0) for r in arows if "tiktoken_total_tokens" in r]
        tools = [r.get("tool_calls", 0) for r in arows if "tool_calls" in r]
        turns = [r.get("turns_used", 0) for r in arows if "turns_used" in r]
        walls = [r.get("wall_clock_s", 0) for r in arows if "wall_clock_s" in r]
        hit_cap = sum(1 for r in arows if r.get("hit_turn_cap")) / n * 100
        cost = sum(row_cost(r) for r in arows)
        print(f"{a:11} {n:>4} {succ:>8.1f}% {_fmt(stats.median(toks) if toks else None):>9} "
              f"{_fmt(stats.mean(toks) if toks else None, 0):>9} "
              f"{_fmt(stats.median(tools) if tools else None):>10} "
              f"{_fmt(stats.median(turns) if turns else None):>10} "
              f"{hit_cap:>8.1f}% "
              f"{_fmt(stats.mean(walls) if walls else None, 1):>12} "
              f"{cost:>11.3f}")

    # matched-success cell: instances where ALL FOUR arms succeeded
    success_by_instance: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("success"):
            success_by_instance[r["instance_id"]].add(r.get("arm", r.get("condition")))
    matched_all = [iid for iid, arms in success_by_instance.items() if all(a in arms for a in ARMS)]

    print(f"\n=== tokens AT MATCHED SUCCESS: all four arms (n={len(matched_all)} instances) ===")
    if matched_all:
        by_arm_matched = {a: [r for r in by_arm.get(a, []) if r["instance_id"] in matched_all] for a in ARMS}
        _print_token_table("", by_arm_matched)
    else:
        print("(no instance was solved by all four arms -- falling back to pairwise matched cells)")
        for a1, a2 in PAIRWISE_CELLS:
            matched = [iid for iid, arms in success_by_instance.items() if a1 in arms and a2 in arms]
            label = f"{ARM_LETTER[a1]}({a1}) vs {ARM_LETTER[a2]}({a2})"
            print(f"\n--- {label}: n={len(matched)} instances both solved ---")
            if not matched:
                print("(empty -- no instance solved by both)")
                continue
            by_arm_pair = {a: [r for r in by_arm.get(a, []) if r["instance_id"] in matched] for a in (a1, a2)}
            for a in (a1, a2):
                arows = by_arm_pair[a]
                toks = [r.get("tiktoken_total_tokens", 0) for r in arows]
                tools = [r.get("tool_calls", 0) for r in arows]
                print(f"  {a:11} n={len(arows):>3} med_tok={_fmt(stats.median(toks) if toks else None):>7} "
                      f"mean_tok={_fmt(stats.mean(toks) if toks else None, 0):>7} "
                      f"med_tools={_fmt(stats.median(tools) if tools else None)}")

    # fairness audit: mean tokens returned per tool call, per arm, broken out by tool
    print("\n=== fairness audit: mean tiktoken tokens returned PER TOOL CALL, by arm x tool ===")
    print(f"{'arm':11} {'tool':12} {'n_calls':>8} {'mean_tok':>9} {'median_tok':>11}")
    per_arm_tool_toks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        arm = r.get("arm", r.get("condition"))
        for call in r.get("tool_call_log", []) or []:
            per_arm_tool_toks[(arm, call["tool"])].append(call["tokens"])
    for a in ARMS:
        tool_names = sorted({t for (aa, t) in per_arm_tool_toks if aa == a})
        for t in tool_names:
            toks = per_arm_tool_toks[(a, t)]
            print(f"{a:11} {t:12} {len(toks):>8} {_fmt(stats.mean(toks), 0):>9} {_fmt(stats.median(toks)):>11}")

    total_cost = sum(row_cost(r) for r in rows)
    print(f"\ntotal estimated cost across all rows: ${total_cost:.3f} "
          f"(Sonnet 4.5 standard pricing: ${PRICE_INPUT_PER_MTOK}/MTok in, "
          f"${PRICE_OUTPUT_PER_MTOK}/MTok out, real API-usage tokens -- NOT the tiktoken metric)")

    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"\n{len(errors)} rows errored (counted as failures per spec):")
        for r in errors[:10]:
            print(f"  {r['instance_id']:44} {r.get('arm','?'):11} {r['error'][:120]}")


if __name__ == "__main__":
    main()
