#!/usr/bin/env python3
"""Region-quality baseline measurement for bgrep.

Measures how well bgrep's *returned regions* (not just returned files) cover
two independent ground truths:

  A. SWE-bench Lite gold-patch hunks (binding/strong signal). For each of
     the 300 SWE-bench Lite instances: parse the gold `patch` field's
     hunk headers to get the OLD-file (pre-fix, i.e. base_commit) line
     ranges the fix touched, check out the repo at base_commit, run the
     installed bgrep CLI on the problem_statement, and measure what
     fraction of those gold hunk lines fall inside the regions bgrep
     returned.

  B. archex expected_regions (informational). For the subset of the 19
     archex lab/results/*.json comprehension tasks whose task yaml defines
     expected_regions, run bgrep on the task question and compute a
     weighted region line-recall against those hand-labeled regions.

This is a read-only measurement script (aside from `git checkout -f` on the
shared swebench_repos clones, which is expected/required to get the repo
into the state the gold patch was written against). It does not modify any
other file in this repo.

Usage:
    uv run --project <archex-venv-dir> python parity/region_eval.py \\
        [--part a|b|all] [--limit N] [--report PATH.json] [--timeout SECONDS]

Part A requires pandas+pyarrow (to read the Lite parquet) and Part B
requires PyYAML (to read the archex task fixtures) -- both are present in
the archex project's venv, hence the `uv run --project` invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed paths (this is a one-shot measurement script, not a reusable CLI --
# defaults mirror parity/harness.py's _SCRATCH conventions).
# ---------------------------------------------------------------------------

_SCRATCH = Path(
    "/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/"
    "3ab12e71-fab2-4a81-bb2b-84700d211ef2/scratchpad"
)

BGREP_BIN = Path("/Users/nicholasarehart/programming-projects/bgrep/.venv-pkg/bin/bgrep")

LITE_PARQUET = _SCRATCH / "bgrep_lab" / "swebench_lite.parquet"
LITE_REPOS = _SCRATCH / "bgrep_lab" / "swebench_repos"

ARCHEX_RESULTS = Path("/Users/nicholasarehart/programming-projects/bgrep/lab/results")
ARCHEX_TASKS_DIR = _SCRATCH / "archex" / "benchmarks" / "tasks"
ARCHEX_REPOS = _SCRATCH / "bgrep_lab" / "repos"

BUDGET = 8192
DEFAULT_TIMEOUT_S = 180
PROGRESS_EVERY = 25


# ---------------------------------------------------------------------------
# swebench_driver guard (spec requirement: check before any repo-checkout work)
# ---------------------------------------------------------------------------


def swebench_driver_guard() -> str | None:
    # Additive opt-out: BGREP_REGION_EVAL_SKIP_DRIVER_GUARD=1 lets a caller
    # who has independently confirmed the running swebench_driver process(es)
    # operate on a disjoint repo set (e.g. multilingual repos under a
    # different scratch tree) proceed anyway -- Part A only ever touches
    # LITE_REPOS (SWE-bench Lite checkouts), so it cannot race a driver
    # confined to other repos. Default behavior (guard active) is unchanged.
    if os.environ.get("BGREP_REGION_EVAL_SKIP_DRIVER_GUARD"):
        return None
    try:
        proc = subprocess.run(["pgrep", "-f", "swebench_driver"],
                               capture_output=True, text=True)
    except OSError:
        return None
    pids = proc.stdout.strip()
    if proc.returncode == 0 and pids:
        return f"swebench_driver process(es) running (pids: {pids.replace(chr(10), ',')})"
    return None


# ---------------------------------------------------------------------------
# bgrep CLI invocation
# ---------------------------------------------------------------------------


def run_bgrep(
    query: str, repo_path: Path, timeout: float, pack_uniform: bool = False
) -> tuple[dict | None, str | None]:
    """Runs the installed bgrep CLI in --json mode. Returns (parsed_json, error).
    `pack_uniform` passes bgrep's --pack-uniform escape hatch through, for A/B
    comparison against confidence-scheduled packing (additive; default False
    preserves the exact invocation this function always used)."""
    argv = [str(BGREP_BIN), "--json", "--budget", str(BUDGET), query, str(repo_path)]
    if pack_uniform:
        argv.insert(1, "--pack-uniform")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except OSError as exc:
        return None, f"failed to spawn bgrep: {exc}"
    if proc.returncode != 0:
        return None, f"exit {proc.returncode}: stderr[:300]={proc.stderr[:300]!r}"
    stdout = proc.stdout.strip()
    if not stdout:
        return None, "empty stdout"
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"bad JSON: {exc}"
    if not isinstance(obj, dict) or "regions" not in obj:
        return None, "JSON output has no \"regions\" key"
    return obj, None


def checkout(repo_path: Path, sha: str) -> None:
    r = subprocess.run(["git", "checkout", "-f", "-q", sha], cwd=repo_path,
                        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"checkout {sha} in {repo_path} failed: {r.stderr.strip()[:300]}")
    subprocess.run(["git", "clean", "-fdq"], cwd=repo_path, capture_output=True,
                    text=True, timeout=300)


# ---------------------------------------------------------------------------
# region-span helpers
# ---------------------------------------------------------------------------


def line_in_spans(line: int, spans: list[list[int]]) -> bool:
    return any(s <= line <= e for s, e in spans)


def bucket_of(recall: float) -> str:
    if recall <= 0.0:
        return "0"
    if recall >= 1.0:
        return "1.0"
    if recall <= 0.25:
        return "0-0.25"
    if recall <= 0.5:
        return "0.25-0.5"
    if recall <= 0.75:
        return "0.5-0.75"
    return "0.75-1.0"


BUCKET_ORDER = ["0", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0", "1.0"]


def mean_median(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.median(values)


# ===========================================================================
# PART A: SWE-bench Lite gold-patch hunks
# ===========================================================================

# Matches unified-diff old/new file header lines. `a/`/`b/` prefixes are
# git's default; /dev/null marks a pure file creation (old side) or
# deletion (new side).
_FILE_OLD_RE = re.compile(r"^--- (?:a/(.+)|(/dev/null))\s*$")
_FILE_NEW_RE = re.compile(r"^\+\+\+ (?:b/(.+)|(/dev/null))\s*$")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")


def parse_gold_hunks(patch_text: str) -> dict[str, list[tuple[int, int]]]:
    """Returns {old_path: [(start, end_inclusive), ...]} -- the OLD-file
    (pre-fix / base_commit) line ranges each hunk touched.

    Two categories of hunk are deliberately excluded (flagged assumption --
    the spec says "extract the OLD-file line ranges from hunk headers",
    which is undefined for these two cases since there are no old lines):

      * Files whose old side is /dev/null (pure file creation): the file
        does not exist at base_commit, so bgrep can never return it; it is
        not counted as a "gold file" at all for this instance.
      * Hunks with old-side count==0 (pure insertion into an existing file,
        `@@ -N,0 +...`): contribute zero old-file lines, so they cannot be
        "covered" or "touched" in the sense the spec's metrics define
        (fraction of *lines* covered; a hunk with 0 lines can't satisfy
        "hunk-touched: >=1 line covered"). Excluded from both the line-set
        and the hunk-touched denominator.
    """
    hunks: dict[str, list[tuple[int, int]]] = {}
    cur_old_path: str | None = None
    for line in patch_text.splitlines():
        m = _FILE_OLD_RE.match(line)
        if m:
            cur_old_path = m.group(1)  # None if /dev/null (pure creation)
            continue
        if _FILE_NEW_RE.match(line):
            continue
        m = _HUNK_RE.match(line)
        if m and cur_old_path:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            if count == 0:
                continue
            end = start + count - 1
            hunks.setdefault(cur_old_path, []).append((start, end))
    return hunks


def load_lite_rows(limit: int, stride: int = 1, offset: int = 0) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(LITE_PARQUET)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "patch": row["patch"],
            "problem_statement": row["problem_statement"],
        })
    if stride > 1:
        rows = rows[offset::stride]
    if limit:
        rows = rows[:limit]
    return rows


def eval_lite_instance(row: dict, timeout: float, pack_uniform: bool = False) -> dict:
    instance_id = row["instance_id"]
    gold_hunks = parse_gold_hunks(row["patch"])
    gold_files = sorted(gold_hunks.keys())
    rec: dict = {
        "instance_id": instance_id,
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "n_gold_files": len(gold_files),
        "n_gold_hunks": sum(len(v) for v in gold_hunks.values()),
        "error": None,
    }
    if not gold_files:
        rec["error"] = "no old-file hunk lines in gold patch (pure file creation(s) only)"
        return rec

    repo_path = LITE_REPOS / row["repo"].replace("/", "__")
    try:
        checkout(repo_path, row["base_commit"])
    except (RuntimeError, OSError) as exc:
        rec["error"] = f"checkout failed: {exc}"
        return rec

    obj, err = run_bgrep(row["problem_statement"], repo_path, timeout, pack_uniform=pack_uniform)
    if err:
        rec["error"] = err
        return rec

    regions: dict[str, list[list[int]]] = obj.get("regions", {})
    files_in_regions = set(regions.keys())

    # (1) hunk-file-covered
    covered_files = [f for f in gold_files if f in files_in_regions]
    rec["hunk_file_covered"] = len(covered_files) / len(gold_files)
    rec["all_gold_files_retrieved"] = len(covered_files) == len(gold_files)

    # (2) hunk line recall (union of gold lines per file, weighted by
    # how many of those lines fall inside that file's returned spans)
    total_lines = 0
    covered_lines = 0
    for f, ranges in gold_hunks.items():
        line_set: set[int] = set()
        for s, e in ranges:
            line_set.update(range(s, e + 1))
        spans = regions.get(f, [])
        total_lines += len(line_set)
        covered_lines += sum(1 for ln in line_set if line_in_spans(ln, spans))
    rec["hunk_line_recall"] = covered_lines / total_lines if total_lines else None

    # (3) hunk-touched: fraction of individual gold hunks with >=1 line covered
    total_hunks = 0
    touched_hunks = 0
    for f, ranges in gold_hunks.items():
        spans = regions.get(f, [])
        for s, e in ranges:
            total_hunks += 1
            if any(line_in_spans(ln, spans) for ln in range(s, e + 1)):
                touched_hunks += 1
    rec["hunk_touched"] = touched_hunks / total_hunks if total_hunks else None

    # (4) tokens of bundle
    stats = obj.get("stats", {})
    rec["tokens"] = stats.get("bundle_tokens")

    # (5) confidence-scheduled packing diagnostics (additive; absent/None on
    # a bgrep binary predating this stats field, handled as "flat" below by
    # aggregate_lite's .get(..., "flat") default so old reports and new ones
    # both aggregate without KeyErrors).
    rec["schedule"] = stats.get("schedule")
    rec["deep_files_count"] = stats.get("deep_files_count")
    rec["skeleton_files_count"] = stats.get("skeleton_files_count")

    return rec


def aggregate_lite(records: list[dict]) -> dict:
    ok = [r for r in records if r["error"] is None]
    subset = [r for r in ok if r["all_gold_files_retrieved"]]

    def metric_block(recs: list[dict]) -> dict:
        hfc = [r["hunk_file_covered"] for r in recs]
        hlr = [r["hunk_line_recall"] for r in recs if r["hunk_line_recall"] is not None]
        ht = [r["hunk_touched"] for r in recs if r["hunk_touched"] is not None]
        tok = [r["tokens"] for r in recs if r["tokens"] is not None]
        hfc_m, hfc_med = mean_median(hfc)
        hlr_m, hlr_med = mean_median(hlr)
        ht_m, ht_med = mean_median(ht)
        tok_m, tok_med = mean_median(tok)
        buckets = {b: 0 for b in BUCKET_ORDER}
        for v in hlr:
            buckets[bucket_of(v)] += 1
        return {
            "n": len(recs),
            "hunk_file_covered": {"mean": hfc_m, "median": hfc_med},
            "hunk_line_recall": {"mean": hlr_m, "median": hlr_med},
            "hunk_touched": {"mean": ht_m, "median": ht_med},
            "tokens": {"mean": tok_m, "median": tok_med},
            "hunk_line_recall_buckets": buckets,
        }

    # Additive: peaked-vs-flat split of hunk_line_recall (confidence-scheduled
    # packing diagnostics) plus mean deep/skeleton file counts per instance.
    # `schedule` is None on a report predating this field, or "flat" whenever
    # --pack-uniform was passed -- both grouped under "flat" here since
    # neither ever ran the peaked packer.
    peaked = [r for r in ok if r.get("schedule") == "peaked"]
    flat = [r for r in ok if r.get("schedule") != "peaked"]
    deep_counts = [r["deep_files_count"] for r in ok if r.get("deep_files_count") is not None]
    skel_counts = [r["skeleton_files_count"] for r in ok if r.get("skeleton_files_count") is not None]
    deep_m, deep_med = mean_median(deep_counts)
    skel_m, skel_med = mean_median(skel_counts)

    return {
        "n_total_instances": len(records),
        "n_ok": len(ok),
        "n_errors": len(records) - len(ok),
        "n_all_gold_files_retrieved": len(subset),
        "all_instances": metric_block(ok),
        "all_gold_files_retrieved_subset": metric_block(subset),
        "confidence_schedule_split": {
            "n_peaked": len(peaked),
            "n_flat": len(flat),
            "peaked": metric_block(peaked),
            "flat": metric_block(flat),
            "deep_files_count": {"mean": deep_m, "median": deep_med},
            "skeleton_files_count": {"mean": skel_m, "median": skel_med},
        },
    }


def run_part_a(
    limit: int, timeout: float, quiet: bool, stride: int = 1, offset: int = 0,
    pack_uniform: bool = False,
) -> dict:
    reason = swebench_driver_guard()
    if reason:
        print(f"REFUSED to run Part A: {reason}", file=sys.stderr)
        return {"skipped": True, "skip_reason": reason}

    rows = load_lite_rows(limit, stride=stride, offset=offset)
    records = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        rec = eval_lite_instance(row, timeout, pack_uniform=pack_uniform)
        records.append(rec)
        if not quiet and (i % PROGRESS_EVERY == 0 or i == len(rows)):
            elapsed = time.time() - t0
            print(f"[part A {i}/{len(rows)}] {row['instance_id']:45} "
                  f"elapsed={elapsed:.0f}s "
                  f"({'ERR' if rec['error'] else 'ok'})", flush=True)
    agg = aggregate_lite(records)
    agg["skipped"] = False
    agg["records"] = records
    return agg


def print_part_a_report(agg: dict) -> None:
    print("\n" + "=" * 78)
    print("PART A: SWE-bench Lite gold-patch hunk coverage")
    print("=" * 78)
    if agg.get("skipped"):
        print(f"SKIPPED: {agg['skip_reason']}")
        return
    print(f"instances loaded:                {agg['n_total_instances']}")
    print(f"instances evaluated (no error):  {agg['n_ok']}")
    print(f"instances with error/skip:       {agg['n_errors']}")
    print(f"  (of which: all-gold-files-retrieved subset: "
          f"{agg['n_all_gold_files_retrieved']})")

    for label, key in [("ALL evaluated instances", "all_instances"),
                        ("subset: ALL gold files retrieved", "all_gold_files_retrieved_subset")]:
        b = agg[key]
        print(f"\n--- {label} (n={b['n']}) ---")
        for metric in ("hunk_file_covered", "hunk_line_recall", "hunk_touched", "tokens"):
            m = b[metric]
            mean_s = f"{m['mean']:.4f}" if m['mean'] is not None else "n/a"
            med_s = f"{m['median']:.4f}" if m['median'] is not None else "n/a"
            print(f"  {metric:20} mean={mean_s:>10}  median={med_s:>10}")
        print("  hunk_line_recall distribution buckets:")
        for bkt in BUCKET_ORDER:
            print(f"    {bkt:12} {b['hunk_line_recall_buckets'][bkt]}")

    split = agg.get("confidence_schedule_split")
    if split:
        print(f"\n--- confidence-scheduled packing split "
              f"(peaked n={split['n_peaked']}, flat n={split['n_flat']}) ---")
        dfc, sfc = split["deep_files_count"], split["skeleton_files_count"]
        print(f"  deep_files_count      mean={dfc['mean']}  median={dfc['median']}")
        print(f"  skeleton_files_count  mean={sfc['mean']}  median={sfc['median']}")
        for label in ("peaked", "flat"):
            b = split[label]
            hlr = b["hunk_line_recall"]
            mean_s = f"{hlr['mean']:.4f}" if hlr["mean"] is not None else "n/a"
            med_s = f"{hlr['median']:.4f}" if hlr["median"] is not None else "n/a"
            print(f"  {label:8} (n={b['n']:3}) hunk_line_recall mean={mean_s:>10}  median={med_s:>10}")

    errors = [r for r in agg["records"] if r["error"]]
    if errors:
        print(f"\nerrored/skipped instances ({len(errors)}):")
        for r in errors:
            print(f"  - {r['instance_id']}: {r['error']}")


# ===========================================================================
# PART B: archex expected_regions (informational)
# ===========================================================================


def find_archex_repo(repo_slug: str) -> Path | None:
    owner_name = repo_slug.replace("/", "__")
    matches = sorted(ARCHEX_REPOS.glob(f"{owner_name}@*"))
    return matches[0] if matches else None


def eval_archex_task(task_id: str, timeout: float) -> dict:
    yaml_path = ARCHEX_TASKS_DIR / f"{task_id}.yaml"
    rec: dict = {"task_id": task_id, "error": None}
    if not yaml_path.exists():
        rec["error"] = f"no task yaml found at {yaml_path}"
        return rec

    import yaml as pyyaml
    data = pyyaml.safe_load(yaml_path.read_text())
    expected_regions = data.get("expected_regions")
    if not expected_regions:
        rec["error"] = "task yaml has no expected_regions"
        return rec

    repo_path = find_archex_repo(data["repo"])
    if repo_path is None:
        rec["error"] = f"no repo checkout found for {data['repo']}"
        return rec

    obj, err = run_bgrep(data["question"], repo_path, timeout)
    if err:
        rec["error"] = err
        return rec

    regions: dict[str, list[list[int]]] = obj.get("regions", {})

    # Weighted region line-recall: each expected_regions entry contributes
    # its own line-recall (covered lines / region length), weighted by its
    # `weight` field (default 1.0). This is a per-region weighted average,
    # not a global line-union recall (flagged assumption: the spec's yaml
    # schema has no documented example combining multiple weighted regions
    # for one task, so "weighted region line-recall" is interpreted at the
    # natural region granularity).
    total_weight = 0.0
    weighted_sum = 0.0
    per_region = []
    for er in expected_regions:
        path = er["path"]
        start, end = er["start_line"], er["end_line"]
        weight = er.get("weight", 1.0)
        span_len = end - start + 1
        spans = regions.get(path, [])
        covered = sum(1 for ln in range(start, end + 1) if line_in_spans(ln, spans))
        region_recall = covered / span_len if span_len else None
        per_region.append({
            "path": path, "start_line": start, "end_line": end, "weight": weight,
            "recall": region_recall, "in_returned_files": path in regions,
        })
        if region_recall is not None:
            weighted_sum += weight * region_recall
            total_weight += weight

    rec["repo"] = data["repo"]
    rec["question"] = data["question"]
    rec["n_expected_regions"] = len(expected_regions)
    rec["weighted_region_line_recall"] = weighted_sum / total_weight if total_weight else None
    rec["tokens"] = obj.get("stats", {}).get("bundle_tokens")
    rec["per_region"] = per_region
    return rec


def run_part_b(timeout: float, quiet: bool) -> dict:
    task_files = sorted(ARCHEX_RESULTS.glob("*.json"))
    task_ids = [f.stem for f in task_files]
    records = []
    for i, tid in enumerate(task_ids, 1):
        rec = eval_archex_task(tid, timeout)
        records.append(rec)
        if not quiet:
            status = rec["error"] if rec["error"] else \
                f"recall={rec['weighted_region_line_recall']:.4f}"
            print(f"[part B {i}/{len(task_ids)}] {tid:35} {status}", flush=True)

    with_regions = [r for r in records if r["error"] is None]
    recalls = [r["weighted_region_line_recall"] for r in with_regions
               if r["weighted_region_line_recall"] is not None]
    mean_r, med_r = mean_median(recalls)
    return {
        "n_total_tasks": len(records),
        "n_with_expected_regions": len(with_regions),
        "n_without_expected_regions": len(records) - len(with_regions),
        "mean_weighted_region_line_recall": mean_r,
        "median_weighted_region_line_recall": med_r,
        "records": records,
    }


def print_part_b_report(res: dict) -> None:
    print("\n" + "=" * 78)
    print("PART B: archex expected_regions (informational)")
    print("=" * 78)
    print(f"lab/results/*.json tasks found:      {res['n_total_tasks']}")
    print(f"  with expected_regions in task yaml: {res['n_with_expected_regions']}")
    print(f"  WITHOUT expected_regions (skipped): {res['n_without_expected_regions']}")
    print(f"  NOTE: spec assumed all 19 task yamls define expected_regions; "
          f"only {res['n_with_expected_regions']} of {res['n_total_tasks']} actually do "
          f"(checked via grep over benchmarks/tasks/*.yaml). Reporting on the subset that does.")
    mean_s = f"{res['mean_weighted_region_line_recall']:.4f}" \
        if res['mean_weighted_region_line_recall'] is not None else "n/a"
    med_s = f"{res['median_weighted_region_line_recall']:.4f}" \
        if res['median_weighted_region_line_recall'] is not None else "n/a"
    print(f"\nmean weighted region line-recall:   {mean_s}")
    print(f"median weighted region line-recall: {med_s}")
    print("\nper-task:")
    for r in res["records"]:
        if r["error"]:
            print(f"  - {r['task_id']:35} SKIPPED: {r['error']}")
        else:
            print(f"  - {r['task_id']:35} recall={r['weighted_region_line_recall']:.4f}  "
                  f"n_regions={r['n_expected_regions']}  tokens={r['tokens']}")


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["a", "b", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="cap Part A instance count (0 = all 300)")
    ap.add_argument("--stride", type=int, default=1,
                     help="Part A: take every Nth instance (after sorting from the parquet's "
                          "natural row order), e.g. --stride 5 for a 60-instance fast-iteration "
                          "subset of the 300; applied before --limit (default: 1, no subsampling)")
    ap.add_argument("--offset", type=int, default=0,
                     help="Part A: starting offset into the stride (default: 0)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                     help="per-task bgrep subprocess timeout in seconds")
    ap.add_argument("--report", type=Path, default=None, help="write full JSON report to PATH")
    ap.add_argument("--quiet", action="store_true", help="suppress per-task progress lines")
    ap.add_argument("--pack-uniform", action="store_true",
                     help="Part A: pass bgrep's --pack-uniform escape hatch through to every "
                          "invocation, for A/B comparison against confidence-scheduled packing "
                          "(default: off, i.e. confidence scheduling is exercised as bgrep's "
                          "own default)")
    args = ap.parse_args()

    if not BGREP_BIN.exists():
        raise SystemExit(f"bgrep binary not found at {BGREP_BIN}")

    report: dict = {}
    if args.part in ("a", "all"):
        report["part_a"] = run_part_a(args.limit, args.timeout, args.quiet,
                                       stride=args.stride, offset=args.offset,
                                       pack_uniform=args.pack_uniform)
        print_part_a_report(report["part_a"])
    if args.part in ("b", "all"):
        report["part_b"] = run_part_b(args.timeout, args.quiet)
        print_part_b_report(report["part_b"])

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nfull report written to {args.report}")


if __name__ == "__main__":
    main()
