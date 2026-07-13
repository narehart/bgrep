#!/usr/bin/env python3
"""Main driver for the tokenbench pilot: runs the grep / roust / RAG
conditions over a stratified SWE-bench Lite sample and writes one row per
(instance, condition) to results.jsonl.

Usage (real run, needs ANTHROPIC_API_KEY):
    lab/tokenbench/.venv/bin/python3 lab/tokenbench/run_bench.py \\
        --stride 10 --out lab/tokenbench/results.jsonl

Usage (wiring validation, no API key / no spend):
    lab/tokenbench/.venv/bin/python3 lab/tokenbench/run_bench.py \\
        --stride 10 --limit 1 --mock --out /tmp/mock_results.jsonl

Resume-safe: (instance_id, condition) pairs already present in --out are
skipped on rerun.

Cost safety valve: tracks running $ spend (Anthropic-reported usage tokens
x current Sonnet pricing) and stops launching new (instance, condition)
runs once --budget-cap-usd is reached, so a mis-estimated pilot can't blow
past the approved budget unattended.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import MAX_TURNS, MODEL, run_agent  # noqa: E402
from common import checkout, count_tokens, load_instances, repo_clone  # noqa: E402
from rag_index import format_rag_bundle, retrieve as rag_retrieve  # noqa: E402
from roust_bundle import get_roust_bundle  # noqa: E402

TOKENBENCH_DIR = Path(__file__).resolve().parent

# Current published Sonnet 4.5 pricing (standard tier, <=200K context):
# https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-07).
# Anthropic is running an introductory $2/$10 rate through 2026-08-31; we use
# the standard $3/$15 rate here so the cost estimate/cap is conservative
# (an upper bound, not the best case).
PRICE_INPUT_PER_MTOK = 3.0
PRICE_OUTPUT_PER_MTOK = 15.0

CONDITIONS = ["grep", "roust", "rag"]


def get_extra_context(condition: str, instance: dict, repo_path: Path, budget: int) -> tuple[str | None, dict]:
    if condition == "grep":
        return None, {}
    if condition == "roust":
        payload = get_roust_bundle(instance["problem_statement"], repo_path, budget=budget)
        return (payload["bundle"] or None), {"roust_files": payload["files"], "roust_stats": payload["stats"]}
    if condition == "rag":
        results = rag_retrieve(repo_path, instance["repo"], instance["base_commit"],
                                instance["problem_statement"], k=12)
        bundle = format_rag_bundle(results) if results else None
        return bundle, {"rag_files": sorted({r["path"] for r in results})}
    raise ValueError(condition)


def make_client(mock: bool):
    if mock:
        from mock_client import MockClient

        return MockClient()
    import anthropic

    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def already_done(out_path: Path) -> set[tuple[str, str]]:
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
                done.add((row["instance_id"], row["condition"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10, help="take every Nth SWE-bench Lite instance")
    ap.add_argument("--limit", type=int, default=0, help="cap number of instances after striding (0 = no cap)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--instances-file", default=None, help="newline-separated instance_ids to run only those")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--roust-budget", type=int, default=8192)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=str(TOKENBENCH_DIR / "results.jsonl"))
    ap.add_argument("--transcripts-dir", default=str(TOKENBENCH_DIR / "transcripts"))
    ap.add_argument("--budget-cap-usd", type=float, default=20.0)
    ap.add_argument("--mock", action="store_true",
                     help="use a scripted fake Anthropic client (no network, no spend) to validate "
                          "harness wiring end-to-end")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition '{c}', choose from {CONDITIONS}")

    if not args.mock:
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set in the environment and --mock was not passed. "
                "Per spec: STOP rather than fabricate results. Set the key or pass --mock to "
                "validate harness wiring without spending."
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transcripts_dir = Path(args.transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(stride=args.stride)
    if args.instances_file:
        wanted = {ln.strip() for ln in Path(args.instances_file).read_text().splitlines() if ln.strip()}
        instances = [i for i in instances if i["instance_id"] in wanted]
    if args.limit:
        instances = instances[: args.limit]

    done = already_done(out_path)
    client = make_client(args.mock)

    running_cost_usd = 0.0
    for row in _load_rows(out_path):
        running_cost_usd += _row_cost(row)

    total_pairs = len(instances) * len(conditions)
    todo = [(i, c) for i in instances for c in conditions if (i["instance_id"], c) not in done]
    print(f"{len(instances)} instances x {len(conditions)} conditions = {total_pairs} pairs, "
          f"{len(done)} done, {len(todo)} to run (mock={args.mock}, model={args.model}, "
          f"max_turns={args.max_turns}, budget_cap=${args.budget_cap_usd:.2f}, "
          f"already spent ~${running_cost_usd:.2f})", flush=True)

    with out_path.open("a") as out_fh:
        for k, (inst, condition) in enumerate(todo, 1):
            if running_cost_usd >= args.budget_cap_usd:
                print(f"STOPPING: running cost ~${running_cost_usd:.2f} has reached "
                      f"--budget-cap-usd {args.budget_cap_usd:.2f}. "
                      f"{len(todo) - k + 1} (instance, condition) pairs not run.", flush=True)
                break
            t_start = time.perf_counter()
            try:
                repo_path = repo_clone(inst["repo"])
                checkout(repo_path, inst["base_commit"])
                extra_context, extra_meta = get_extra_context(condition, inst, repo_path, args.roust_budget)
                context_tokens = count_tokens(extra_context or "")

                log_path = transcripts_dir / f"{inst['instance_id']}__{condition}.jsonl"
                with log_path.open("w") as log_fh:
                    result = run_agent(
                        client, inst, condition, repo_path, extra_context, log_fh,
                        max_turns=args.max_turns, model=args.model,
                    )
                result["context_tokens"] = context_tokens
                result["context_meta"] = extra_meta
                result["setup_s"] = round(time.perf_counter() - t_start - result["wall_clock_s"], 2)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "instance_id": inst["instance_id"], "condition": condition, "repo": inst["repo"],
                    "gold_files": inst["gold_files"], "error": f"{type(exc).__name__}: {exc}"[:500],
                    "success": False,
                }
            out_fh.write(json.dumps(result) + "\n")
            out_fh.flush()
            running_cost_usd += _row_cost(result)
            status = "OK" if result.get("success") else ("ERR" if result.get("error") else "FAIL")
            print(f"[{k}/{len(todo)}] {inst['instance_id']:44} {condition:6} {status:4} "
                  f"tiktoken_tot={result.get('tiktoken_total_tokens', '-')} "
                  f"turns={result.get('turns_used', '-')} "
                  f"tools={result.get('tool_calls', '-')} "
                  f"~${running_cost_usd:.2f} cum", flush=True)
    print("done", flush=True)


def _load_rows(out_path: Path):
    if not out_path.exists():
        return
    for line in out_path.read_text().splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass


def _row_cost(row: dict) -> float:
    ai = row.get("api_input_tokens", 0) or 0
    ao = row.get("api_output_tokens", 0) or 0
    return ai / 1e6 * PRICE_INPUT_PER_MTOK + ao / 1e6 * PRICE_OUTPUT_PER_MTOK


if __name__ == "__main__":
    main()
