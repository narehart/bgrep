#!/usr/bin/env python3
"""Agentless-style "% Correct Location" metric on the FULL SWE-bench test split.

A thin fork of `lab/agentless_metric_verified.py` for the paper eval wave:
ALL scoring logic (FILE / FUNCTION-exact / LINE / region-precision, and the
unified errors-count-as-wrong-at-all-levels/v1 convention) is IMPORTED from
that module, not duplicated -- see its docstring for the definitions.
Differences are plumbing only:

  - `--predictions` accepts MULTIPLE JSONL paths (the per-shard reports a
    sharded `parity/region_eval_full.py` run produces); they are
    concatenated with a duplicate-instance-id guard.
  - Gold parquet defaults to `lab/swebench_full.parquet` (from
    `scripts/fetch_swebench_full.py`); `--expect-n` defaults to 2294 (the
    full test split) -- pass `--expect-n 0` to skip the completeness assert
    while scoring partial shard sets mid-run.
  - `--repos-dir` overrides the clone directory used for the read-only
    `git show` AST walks (default `lab/swebench_repos/` under this
    checkout). `git show <sha>:<path>` reads the object database only --
    it neither reads nor touches working trees, so scoring may point at
    clones a concurrent eval is checking out.

Usage:
    python lab/agentless_metric_full.py \\
        --predictions lab/results_regions/full2294_default_s*of8.jsonl \\
        --out lab/results_regions/agentless_metric_full_default.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lab"))
import agentless_metric_verified as amv  # noqa: E402  (shared scoring logic)

DEFAULT_GOLD_PARQUET = REPO_ROOT / "lab" / "swebench_full.parquet"
DEFAULT_OUT = REPO_ROOT / "lab" / "results_regions" / "agentless_metric_full.json"
EXPECTED_N = 2294


def load_merged_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for p in paths:
        for rec in amv.load_records(p):
            iid = rec["instance_id"]
            if iid in seen:
                raise SystemExit(f"duplicate instance_id {iid!r} (second copy in {p}) -- "
                                 f"overlapping shard reports?")
            seen.add(iid)
            records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, nargs="+", required=True,
                     help="region_eval_full JSONL report(s); pass every shard of a run")
    ap.add_argument("--gold-parquet", type=Path, default=DEFAULT_GOLD_PARQUET)
    ap.add_argument("--expect-n", type=int, default=EXPECTED_N,
                     help=f"assert len(records) == this (default {EXPECTED_N}, the full "
                          f"test split); 0 skips (partial shard-set scoring)")
    ap.add_argument("--repos-dir", type=Path, default=None,
                     help="clone directory for the read-only `git show` AST walks "
                          "(default: lab/swebench_repos under this checkout)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.repos_dir is not None:
        amv.SWEBENCH_REPOS = args.repos_dir

    records = load_merged_records(args.predictions)
    if args.expect_n:
        assert len(records) == args.expect_n, \
            f"expected {args.expect_n} merged records, got {len(records)}"

    engine_shas = {r["engine_sha"] for r in records if r["engine_sha"] is not None}
    engine_dirty = {r["engine_dirty"] for r in records if r["engine_dirty"] is not None}
    bm25_flags = {r.get("bm25_only", False) for r in records}
    if len(bm25_flags) > 1:
        raise SystemExit("mixed bm25_only values across merged records -- these shards "
                         "are not from the same arm")

    import pandas as pd
    df = pd.read_parquet(args.gold_parquet)
    patch_by_id = {row["instance_id"]: row["patch"] for _, row in df.iterrows()}
    unknown = [r["instance_id"] for r in records if r["instance_id"] not in patch_by_id]
    if unknown:
        raise SystemExit(f"{len(unknown)} record(s) not present in the gold parquet, "
                         f"e.g. {unknown[:3]}")

    n_region_eval_errors = sum(1 for r in records if r["error"] is not None)
    file_correct_subset = [r for r in records if r.get("all_gold_files_retrieved", False)]

    print("Computing exact FUNCTION-level metric + region precision via read-only "
          "`git show` (this walks every gold + returned .py file's AST)...", file=sys.stderr)

    def block(recs: list[dict]) -> dict:
        return {
            "n": len(recs),
            "file": amv.compute_file_level(recs),
            "function": amv.compute_function_level_exact(recs, patch_by_id),
            "line": amv.compute_line_level(recs),
            "region_precision": amv.compute_region_precision(recs, patch_by_id),
        }

    all_block = block(records)
    subset_block = block(file_correct_subset)

    def rel(p: Path) -> str:
        return str(p.relative_to(REPO_ROOT)) if p.is_absolute() and p.is_relative_to(REPO_ROOT) else str(p)

    report: dict = {
        "convention": "errors-count-as-wrong-at-all-levels/v1: FILE, FUNCTION, and LINE share "
                      "the same denominator (n = all loaded records); engine errors and "
                      "git_show failures count as WRONG and are reported separately "
                      "(*_counted_wrong / n_engine_errors keys).",
        "source": {
            "predictions": [rel(p) for p in args.predictions],
            "gold": rel(args.gold_parquet),
            "n_instances": len(records),
            "n_region_eval_errors": n_region_eval_errors,
            "engine_shas_seen": sorted(engine_shas),
            "engine_dirty_seen": sorted(engine_dirty),
            "bm25_only": bool(bm25_flags and next(iter(bm25_flags))),
            "pipeline": "shipped roust-rs engine, --budget 8192, confidence-scheduled packing "
                        "(parity/region_eval_full.py Part A, no-LLM), FULL SWE-bench test "
                        "split (paper eval wave)",
        },
        "all_instances": all_block,
        "file_correct_subset": {
            **subset_block,
            "note": "restricted to instances where FILE-level was already correct -- isolates "
                    "region/line/function quality from file recall",
        },
        "agentless_gpt4o_published": amv.AGENTLESS_GPT4O,
        "agentless_note": "AGENTLESS_GPT4O numbers are Agentless's published SWE-bench LITE "
                          "localization rates (arXiv:2407.01489 Table 1) -- context only; no "
                          "published Agentless full-split localization numbers exist to "
                          "compare against directly.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}", file=sys.stderr)

    amv.print_table(report)


if __name__ == "__main__":
    main()
