#!/usr/bin/env python3
"""Main driver for the tokenbench v2 run: runs the four arms (grep / roust /
roust_grep / rag_grep) over the same stride-10 SWE-bench Lite sample as v1
and writes one row per (instance, arm) to results.jsonl.

Usage (real run, needs ANTHROPIC_API_KEY):
    lab/tokenbench/.venv/bin/python3 lab/tokenbench/run_bench.py \\
        --stride 10 --out lab/tokenbench/results.jsonl

Usage (wiring validation, no API key / no spend):
    lab/tokenbench/.venv/bin/python3 lab/tokenbench/run_bench.py \\
        --stride 10 --limit 1 --mock --out /tmp/mock_results.jsonl

Resume-safe: (instance_id, arm, trial) triples already present in --out are
skipped on rerun. --trial (default 0) distinguishes repeat runs of the same
(instance, arm) pair -- each trial gets its own transcript file and its own
row, so repeat-run campaigns (see issue #16) don't overwrite prior trials'
transcripts or get no-op'd by resume-dedup.

Cost safety valves:
  * between pairs: tracks running $ spend (Anthropic-reported usage tokens
    x current Sonnet pricing) and stops launching new (instance, arm, trial)
    runs once --budget-cap-usd is reached, so a mis-estimated run can't blow
    past the approved budget unattended.
  * within a pair: --max-cost-per-pair bounds a single run's own turn loop
    (see agent.py) so one abnormally expensive run can't itself blow past
    the per-pair estimate 4x+ over before the between-pairs check ever fires
    again (confirmed in the #18 pilot: one pair cost $4.19 against a $0.95
    estimate). A run that aborts this way is recorded as
    status="aborted_over_budget", NOT success=False -- it must not corrupt
    the task success rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import ARMS, MAX_COST_PER_PAIR_USD, MAX_TURNS, MODEL, ROUST_BUDGET, run_agent  # noqa: E402
from common import checkout, load_instances, repo_clone, row_cost  # noqa: E402
from rag_tool import TOP_K as RAG_TOP_K  # noqa: E402

TOKENBENCH_DIR = Path(__file__).resolve().parent

ARM_NAMES = list(ARMS.keys())  # grep, roust, roust_grep, rag_grep


def make_client(mock: bool):
    if mock:
        from mock_client import MockClient

        return MockClient()
    import anthropic

    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


def already_done(out_path: Path) -> set[tuple[str, str, int]]:
    """(instance_id, arm, trial) triples already present in --out. Rows
    written before the trial dimension existed have no "trial" key and are
    treated as trial 0 -- consistent with the schema default, and with the
    fact that every pre-trial row WAS an (implicit) trial 0."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
                done.add((row["instance_id"], row["arm"], row.get("trial", 0)))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10, help="take every Nth SWE-bench Lite instance")
    ap.add_argument("--limit", type=int, default=0, help="cap number of instances after striding (0 = no cap)")
    ap.add_argument("--arms", default=",".join(ARM_NAMES))
    ap.add_argument("--instances-file", default=None, help="newline-separated instance_ids to run only those")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--roust-budget", type=int, default=ROUST_BUDGET)
    ap.add_argument("--rag-k", type=int, default=RAG_TOP_K)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default=str(TOKENBENCH_DIR / "results.jsonl"))
    ap.add_argument("--transcripts-dir", default=str(TOKENBENCH_DIR / "transcripts"))
    ap.add_argument("--budget-cap-usd", type=float, default=20.0)
    ap.add_argument("--trial", type=int, default=0,
                     help="trial index for this invocation (issue #16 repeat-run campaigns); "
                          "included in the result row, the transcript filename, and the "
                          "resume-dedup key so repeat trials of the same (instance, arm) pair "
                          "neither overwrite each other's transcript nor get skipped as 'already done'")
    ap.add_argument("--max-cost-per-pair", type=float, default=MAX_COST_PER_PAIR_USD,
                     help="abort a single (instance, arm) run if its cumulative cost exceeds this "
                          "(checked inside the turn loop, not just between pairs); recorded as "
                          "status='aborted_over_budget', not success=False")
    ap.add_argument("--mock", action="store_true",
                     help="use a scripted fake Anthropic client (no network, no spend) to validate "
                          "harness wiring without spending")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_NAMES:
            raise SystemExit(f"unknown arm '{a}', choose from {ARM_NAMES}")

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
        running_cost_usd += row_cost(row.get("api_input_tokens", 0), row.get("api_output_tokens", 0))

    total_pairs = len(instances) * len(arms)
    todo = [(i, a) for i in instances for a in arms if (i["instance_id"], a, args.trial) not in done]
    print(f"{len(instances)} instances x {len(arms)} arms = {total_pairs} pairs (trial={args.trial}), "
          f"{len(done)} done, {len(todo)} to run (mock={args.mock}, model={args.model}, "
          f"max_turns={args.max_turns}, budget_cap=${args.budget_cap_usd:.2f}, "
          f"max_cost_per_pair=${args.max_cost_per_pair:.2f}, "
          f"already spent ~${running_cost_usd:.2f})", flush=True)

    with out_path.open("a") as out_fh:
        for k, (inst, arm) in enumerate(todo, 1):
            if running_cost_usd >= args.budget_cap_usd:
                print(f"STOPPING: running cost ~${running_cost_usd:.2f} has reached "
                      f"--budget-cap-usd {args.budget_cap_usd:.2f}. "
                      f"{len(todo) - k + 1} (instance, arm) pairs not run.", flush=True)
                break
            t_start = time.perf_counter()
            try:
                repo_path = repo_clone(inst["repo"])
                checkout(repo_path, inst["base_commit"])

                log_path = transcripts_dir / f"{inst['instance_id']}__{arm}__t{args.trial}.jsonl"
                with log_path.open("w") as log_fh:
                    result = run_agent(
                        client, inst, arm, repo_path, log_fh,
                        max_turns=args.max_turns, model=args.model,
                        roust_budget=args.roust_budget, rag_k=args.rag_k,
                        max_cost_per_pair=args.max_cost_per_pair,
                    )
                result["setup_s"] = round(time.perf_counter() - t_start - result["wall_clock_s"], 2)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "instance_id": inst["instance_id"], "arm": arm, "repo": inst["repo"],
                    "gold_files": inst["gold_files"], "error": f"{type(exc).__name__}: {exc}"[:500],
                    "success": False, "status": "error",
                }
            result["trial"] = args.trial
            out_fh.write(json.dumps(result) + "\n")
            out_fh.flush()
            running_cost_usd += row_cost(result.get("api_input_tokens", 0), result.get("api_output_tokens", 0))
            disp_status = ("OK" if result.get("success")
                            else result.get("status", "error").upper() if result.get("status", "ok") != "ok"
                            else ("ERR" if result.get("error") else "FAIL"))
            print(f"[{k}/{len(todo)}] {inst['instance_id']:44} {arm:11} {disp_status:9} "
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


if __name__ == "__main__":
    main()
