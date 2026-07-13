#!/usr/bin/env python3
"""Reads results.jsonl and prints the headline v2 summary tables:
per-arm success rate, median/mean tiktoken total tokens, tool calls, turns,
wall-clock, cost; tokens AT MATCHED SUCCESS (instances ALL of the run's arms
solved, falling back to pairwise matched cells if that set is empty); and
the fairness audit -- mean tokens returned per tool call, per arm, broken
out by tool (so the roust-budget-8192 vs rag_search-k=24 match is auditable
against real run data, not just the pre-run calibration in rag_tool.py).

Arms are AUTODISCOVERED from the data (`sorted({row["arm"] for row in
rows})`), not hardcoded -- issue #17: the old hardcoded 4-arm v2 list made
summarize.py silently print "(no rows)" for every arm of results_forced.jsonl
(arms grep_forced/roust_forced) with no error, which looks like "no data"
rather than "wrong arm list". Pass --arms to require specific arms be
present; a requested arm with zero rows is now a loud failure, not a blank
table.

Rows with status in {"api_error", "aborted_over_budget"} (see agent.py FIX
2/FIX 3) are infrastructure hiccups, not task outcomes -- they're EXCLUDED
from the success-rate denominator and reported separately as a count so
they can't silently corrupt p_i.

Usage: lab/tokenbench/.venv/bin/python3 lab/tokenbench/summarize.py \\
           [--results lab/tokenbench/results.jsonl] [--arms grep,roust]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

TOKENBENCH_DIR = Path(__file__).resolve().parent
# The known v2 quartet: when the data's discovered arms are exactly this
# set, we keep the curated A/B/C/D labels and pairwise cells from the v2
# spec (so known-run output is byte-for-byte what it always was). Any other
# arm set (e.g. results_forced.jsonl's grep_forced/roust_forced) falls back
# to a generic all-pairs comparison labeled by arm name.
KNOWN_V2_ARM_LETTER = {"grep": "A", "roust": "B", "roust_grep": "C", "rag_grep": "D"}
KNOWN_V2_PAIRWISE_CELLS = [("grep", "roust"), ("grep", "roust_grep"), ("roust_grep", "rag_grep")]

# Statuses that mark a row as an infrastructure hiccup rather than a genuine
# task outcome (agent.py FIX 2 / FIX 3). Excluded from success-rate
# denominators; reported separately as a count. Rows with no "status" key
# (pre-fix results.jsonl / results_forced.jsonl) default to "ok" -> counted,
# so old files reproduce their old numbers exactly.
EXCLUDED_STATUSES = {"api_error", "aborted_over_budget"}

PRICE_INPUT_PER_MTOK = 3.0
PRICE_OUTPUT_PER_MTOK = 15.0


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def row_arm(row: dict) -> str:
    return row.get("arm", row.get("condition", "?"))


def is_excluded(row: dict) -> bool:
    return row.get("status", "ok") in EXCLUDED_STATUSES


def _fmt(x, nd=0):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def row_cost(row: dict) -> float:
    ai = row.get("api_input_tokens", 0) or 0
    ao = row.get("api_output_tokens", 0) or 0
    return ai / 1e6 * PRICE_INPUT_PER_MTOK + ao / 1e6 * PRICE_OUTPUT_PER_MTOK


def _print_token_table(title: str, arms: list[str], by_arm_rows: dict[str, list[dict]]) -> None:
    print(title)
    print(f"{'arm':11} {'n':>4} {'med_tok':>9} {'mean_tok':>9} {'med_tools':>10} {'med_turns':>10} "
          f"{'mean_wall_s':>12}")
    for a in arms:
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
    ap.add_argument("--arms", default=None,
                     help="comma-separated arm names that MUST be present with >0 rows (fail loudly "
                          "if any are missing/empty); default: autodiscover every arm found in --results")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    rows = load_rows(path)

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[row_arm(r)].append(r)

    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        missing = [a for a in arms if not by_arm.get(a)]
        if missing:
            raise SystemExit(
                f"--arms requested {missing} but {path} has zero rows for "
                f"{'them' if len(missing) > 1 else 'it'} "
                f"(arms actually present: {sorted(by_arm)}). Refusing to print an empty table silently "
                f"(issue #17) -- check the arm name(s) or the --results file."
            )
    else:
        arms = sorted(by_arm)
        if set(arms) == set(KNOWN_V2_ARM_LETTER):
            # Preserve the canonical v2 A/B/C/D ordering (not alphabetical)
            # so a known-arm run's table looks exactly like it always did.
            arms = list(KNOWN_V2_ARM_LETTER)

    # by_arm split into counted (status not in EXCLUDED_STATUSES -- a genuine
    # task outcome) vs excluded (api_error / aborted_over_budget -- an
    # infrastructure hiccup, not evidence the task failed). Success-rate
    # denominators use ONLY counted rows; excluded rows are reported as a
    # separate count per arm and overall.
    counted_by_arm: dict[str, list[dict]] = {a: [r for r in by_arm.get(a, []) if not is_excluded(r)] for a in arms}
    excluded_by_arm: dict[str, list[dict]] = {a: [r for r in by_arm.get(a, []) if is_excluded(r)] for a in arms}

    instances = sorted({r["instance_id"] for r in rows})
    print(f"=== tokenbench v2 summary: {len(instances)} instances, {len(rows)} rows, "
          f"file={path} ===\n")

    print(f"{'arm':11} {'n':>4} {'excl':>5} {'success%':>9} {'med_tok':>9} {'mean_tok':>9} "
          f"{'med_tools':>10} {'med_turns':>10} {'hit_cap%':>9} {'mean_wall_s':>12} {'total_cost$':>12}")
    for a in arms:
        arows = counted_by_arm.get(a, [])
        n_excl = len(excluded_by_arm.get(a, []))
        if not arows:
            print(f"{a:11} {'0':>4} {n_excl:>5} (no non-excluded rows)")
            continue
        n = len(arows)
        succ = sum(1 for r in arows if r.get("success")) / n * 100
        toks = [r.get("tiktoken_total_tokens", 0) for r in arows if "tiktoken_total_tokens" in r]
        tools = [r.get("tool_calls", 0) for r in arows if "tool_calls" in r]
        turns = [r.get("turns_used", 0) for r in arows if "turns_used" in r]
        walls = [r.get("wall_clock_s", 0) for r in arows if "wall_clock_s" in r]
        hit_cap = sum(1 for r in arows if r.get("hit_turn_cap")) / n * 100
        cost = sum(row_cost(r) for r in by_arm.get(a, []))  # cost totals include excluded rows -- they still spent $
        print(f"{a:11} {n:>4} {n_excl:>5} {succ:>8.1f}% {_fmt(stats.median(toks) if toks else None):>9} "
              f"{_fmt(stats.mean(toks) if toks else None, 0):>9} "
              f"{_fmt(stats.median(tools) if tools else None):>10} "
              f"{_fmt(stats.median(turns) if turns else None):>10} "
              f"{hit_cap:>8.1f}% "
              f"{_fmt(stats.mean(walls) if walls else None, 1):>12} "
              f"{cost:>11.3f}")
    total_excluded = sum(len(v) for v in excluded_by_arm.values())
    if total_excluded:
        by_status: dict[str, int] = defaultdict(int)
        for a in arms:
            for r in excluded_by_arm.get(a, []):
                by_status[r.get("status", "?")] += 1
        print(f"('n' above excludes {total_excluded} row(s) with status in {sorted(EXCLUDED_STATUSES)} "
              f"({dict(by_status)}) from success%/token/turn/tool stats; 'excl' is that count per arm.)")

    # matched-success cell: instances where ALL of this run's arms succeeded
    # (only counted, non-excluded rows can ever have success=True -- see
    # agent.py: status != "ok" forces success=False -- so no extra filtering
    # is needed here beyond using `rows`/`success`).
    success_by_instance: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("success"):
            success_by_instance[r["instance_id"]].add(row_arm(r))
    matched_all = [iid for iid, s in success_by_instance.items() if all(a in s for a in arms)]

    print(f"\n=== tokens AT MATCHED SUCCESS: all {len(arms)} arm(s) ({', '.join(arms)}) "
          f"(n={len(matched_all)} instances) ===")
    if matched_all:
        by_arm_matched = {a: [r for r in counted_by_arm.get(a, []) if r["instance_id"] in matched_all]
                           for a in arms}
        _print_token_table("", arms, by_arm_matched)
    else:
        print("(no instance was solved by all arms -- falling back to pairwise matched cells)")
        if set(arms) == set(KNOWN_V2_ARM_LETTER):
            pairwise_cells = KNOWN_V2_PAIRWISE_CELLS
            arm_letter = KNOWN_V2_ARM_LETTER
        else:
            pairwise_cells = list(itertools.combinations(arms, 2))
            arm_letter = {a: a for a in arms}
        for a1, a2 in pairwise_cells:
            matched = [iid for iid, s in success_by_instance.items() if a1 in s and a2 in s]
            label = f"{arm_letter[a1]}({a1}) vs {arm_letter[a2]}({a2})"
            print(f"\n--- {label}: n={len(matched)} instances both solved ---")
            if not matched:
                print("(empty -- no instance solved by both)")
                continue
            by_arm_pair = {a: [r for r in counted_by_arm.get(a, []) if r["instance_id"] in matched]
                           for a in (a1, a2)}
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
        for call in r.get("tool_call_log", []) or []:
            per_arm_tool_toks[(row_arm(r), call["tool"])].append(call["tokens"])
    for a in arms:
        tool_names = sorted({t for (aa, t) in per_arm_tool_toks if aa == a})
        for t in tool_names:
            toks = per_arm_tool_toks[(a, t)]
            print(f"{a:11} {t:12} {len(toks):>8} {_fmt(stats.mean(toks), 0):>9} {_fmt(stats.median(toks)):>11}")

    total_cost = sum(row_cost(r) for r in rows)
    print(f"\ntotal estimated cost across all rows: ${total_cost:.3f} "
          f"(Sonnet 4.5 standard pricing: ${PRICE_INPUT_PER_MTOK}/MTok in, "
          f"${PRICE_OUTPUT_PER_MTOK}/MTok out, real API-usage tokens -- NOT the tiktoken metric)")

    errors = [r for r in rows if r.get("error") and not is_excluded(r)]
    if errors:
        print(f"\n{len(errors)} rows errored (status='error', counted as failures per spec):")
        for r in errors[:10]:
            print(f"  {r['instance_id']:44} {row_arm(r):11} {r['error'][:120]}")
    if total_excluded:
        print(f"\n{total_excluded} row(s) excluded from success%/token stats "
              f"(api_error/aborted_over_budget -- infrastructure hiccups, not task failures):")
        for r in [r for r in rows if is_excluded(r)][:10]:
            print(f"  {r['instance_id']:44} {row_arm(r):11} status={r.get('status')} "
                  f"trial={r.get('trial', 0)} {(r.get('error') or '')[:100]}")


if __name__ == "__main__":
    main()
