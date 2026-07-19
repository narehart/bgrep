#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# SHARED-CLONE HAZARD (issue #41): this script MUTATES lab/swebench_repos/
# (git checkout of each instance's base commit before invoking the engines).
# That directory has historically been ONE physical set of clones shared
# across every worktree of this repo -- two concurrent evals (or an eval plus
# anything else reading those clones) in different worktrees race each
# other's checkouts and silently corrupt results. Discipline: when anything
# else might touch the clones, run from a worktree whose lab/swebench_repos/
# is a PRIVATE copy (`cp -R` of the main repo's clones), never the main
# repo's shared directory.
# ---------------------------------------------------------------------------
"""Full-bundle parity checker: Python `roust` engine vs Rust `roust` engine.

`parity/harness.py` is the binding 300/300 gate, but it only compares
``returned_files`` (the ranked file list) between a candidate and the frozen
lanes2-v7 expectations file -- it never looks at the packed REGIONS inside
the ``--json`` bundle. Region-packing v2 (nested block spans, idf-weighted
coverage, anchor-forced regions -- see ``roust-rs/PARITY_NOTES.md`` item 13)
is exactly the part of ``src/roust/core.py`` that is newer than the frozen
``lab/lanes2.py`` oracle the file-list gate was built against, so the region
behavior of the shipped Python engine vs the Rust port has never been
automatically verified.

This script runs BOTH engines' full ``--json`` bundle output (files, packed
regions, bundle text, stats) over a sample of SWE-bench Lite instances and
diffs the parsed JSON recursively, not just the file list:

    Engine A (Python): `uv run roust --json --budget 8192 QUERY REPO_PATH`
                        (cwd = this project's root, so `uv run` resolves it)
    Engine B (Rust):    `roust-rs/target/release/roust --json --budget 8192
                          QUERY REPO_PATH`

Per (query, repo) pair, verdict is one of:

    EXACT                           parsed JSON identical modulo volatile
                                    fields (index_ms, query_ms, cache).
    FILES-MATCH-BUT-REGIONS-DIFFER  file list + order match, but regions,
                                    bundle text, bundle_tokens, or any other
                                    structural field differs. THE INTERESTING
                                    FAILURE this script exists to catch.
    DIFFER                          the returned file list itself differs
                                    (set and/or order).
    ERROR                           one or both engines failed to produce
                                    parseable output (crash, timeout, bad
                                    JSON) -- reported separately, not folded
                                    into the three verdicts above.

Task loading (instance -> query + repo + base_commit) and the checked-out
directory management (git checkout -f -q + git clean -fdq) are reused
directly from ``parity/harness.py`` rather than reimplemented here, so this
script's corpus is defined exactly the same way the binding gate's is.

Usage
-----
    python parity/bundle_parity.py [--limit N] [--stride N] [--budget N]
        [--report PATH.json] [--timeout SECONDS]

Exit status: 0 iff every pair compared is EXACT or FILES-MATCH-BUT-REGIONS-
DIFFER with no ERRORs; 1 if any DIFFER or ERROR occurred (this script does
not gate anything -- it is an evidence-gathering tool for the "is it safe to
delete src/roust" question -- but a non-zero exit is still useful for CI/
scripting).
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402  (reuse task loading / checkout / driver guard)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LITE_EXPECTED = REPO_ROOT / "lab" / "results_swebench" / "abl_bridges_v7.jsonl"
DEFAULT_LITE_PARQUET = REPO_ROOT / "lab" / "swebench_lite.parquet"
DEFAULT_LITE_REPOS = REPO_ROOT / "lab" / "swebench_repos"
DEFAULT_RUST_BIN = REPO_ROOT / "roust-rs" / "target" / "release" / "roust"

DEFAULT_TIMEOUT_S = 300
DEFAULT_STRIDE = 5  # 300 lite instances / 5 = 60 pairs, spread across every repo
DIFF_LINE_CAP = 60

# Fields whose values legitimately vary run-to-run (wall-clock timings, and
# on-disk cache hit/miss state) and therefore must not participate in the
# EXACT/not-EXACT decision.
VOLATILE_STATS_KEYS = {"index_ms", "query_ms", "cache"}


# ---------------------------------------------------------------------------
# Task loading (reuses parity/harness.py's machinery)
# ---------------------------------------------------------------------------


def load_tasks(limit: int, stride: int, lite_expected: Path, lite_parquet: Path,
               lite_repos: Path) -> list["harness.Task"]:
    ns = types.SimpleNamespace(
        lite_expected=lite_expected,
        lite_parquet=lite_parquet,
        lite_queries_json=None,
        lite_repos=lite_repos,
        limit=0,
    )
    tasks = harness.load_lite_tasks(ns)
    if stride > 1:
        tasks = tasks[::stride]
    if limit:
        tasks = tasks[:limit]
    return tasks


# ---------------------------------------------------------------------------
# Engine invocation
# ---------------------------------------------------------------------------


def run_engine(argv: list[str], cwd: Path, timeout: float) -> tuple[dict | None, str | None, float]:
    """Runs one engine invocation. Returns (parsed_json, error, elapsed_s).
    A no-results run (exit 1, empty stdout) is contract-valid and normalized
    to an empty-bundle object rather than treated as an error."""
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s", time.perf_counter() - t0
    except OSError as exc:
        return None, f"failed to spawn: {exc}", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    if proc.returncode == 1 and not proc.stdout.strip():
        return ({"query": None, "files": [], "regions": {}, "bundle": "",
                  "stats": {"bundle_tokens": 0}}, None, elapsed)
    if proc.returncode not in (0, 1):
        return None, f"exit {proc.returncode}: stderr[:300]={proc.stderr[:300]!r}", elapsed
    try:
        obj = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return None, f"bad JSON: {exc}; stdout[:300]={proc.stdout[:300]!r}", elapsed
    if not isinstance(obj, dict):
        return None, f"top-level JSON is not an object: {type(obj).__name__}", elapsed
    return obj, None, elapsed


def python_argv(query: str, repo_path: Path, budget: int) -> list[str]:
    return ["uv", "run", "roust", "--json", "--budget", str(budget), query, str(repo_path)]


def rust_argv(rust_bin: Path, query: str, repo_path: Path, budget: int) -> list[str]:
    return [str(rust_bin), "--json", "--budget", str(budget), query, str(repo_path)]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def file_list(obj: dict) -> list[str]:
    return [f["path"] if isinstance(f, dict) else f for f in obj.get("files", [])]


def normalize(obj: dict) -> dict:
    """Deep-copies obj and strips volatile stats fields so the remainder can
    be compared for byte-for-byte equality (files, order, regions, bundle
    text, bundle_tokens, files_indexed, query -- everything load-bearing)."""
    obj = json.loads(json.dumps(obj))
    stats = obj.get("stats", {})
    for k in list(stats.keys()):
        if k in VOLATILE_STATS_KEYS:
            stats.pop(k)
    return obj


def deep_diff(a, b, path: str = "$") -> list[str]:
    """Generic recursive diff over parsed JSON values. Returns readable
    "path: python=... vs rust=..." lines. Special-cases the bundle field (a
    potentially large multi-line string) to emit a unified line diff instead
    of a giant repr."""
    lines: list[str] = []
    if path.endswith(".bundle") and isinstance(a, str) and isinstance(b, str):
        if a != b:
            lines.append(f"  bundle text differs at {path}:")
            udiff = list(difflib.unified_diff(
                a.splitlines(), b.splitlines(),
                fromfile="python.bundle", tofile="rust.bundle", lineterm="",
            ))
            lines.extend(f"    {ln}" for ln in udiff[:DIFF_LINE_CAP])
        return lines
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                lines.append(f"  missing in python at {path}.{k}: rust={b[k]!r}")
            elif k not in b:
                lines.append(f"  missing in rust at {path}.{k}: python={a[k]!r}")
            else:
                lines.extend(deep_diff(a[k], b[k], f"{path}.{k}"))
        return lines
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            lines.append(f"  length mismatch at {path}: python={len(a)} rust={len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            lines.extend(deep_diff(x, y, f"{path}[{i}]"))
        return lines
    if a != b:
        lines.append(f"  value mismatch at {path}: python={a!r} rust={b!r}")
    return lines


def region_span_diffs(py_obj: dict, rs_obj: dict, files: list[str]) -> list[str]:
    """Per-file region span ([start,end] pairs) diff, in the exact terms the
    task asks for: file, start line, end line."""
    lines: list[str] = []
    py_regions = py_obj.get("regions", {})
    rs_regions = rs_obj.get("regions", {})
    for f in files:
        py_spans = py_regions.get(f)
        rs_spans = rs_regions.get(f)
        if py_spans != rs_spans:
            lines.append(f"  region spans differ for {f}:")
            lines.append(f"    python: {py_spans}")
            lines.append(f"    rust:   {rs_spans}")
    return lines


def compare(py_obj: dict, rs_obj: dict) -> tuple[str, list[str]]:
    py_files = file_list(py_obj)
    rs_files = file_list(rs_obj)

    if py_files != rs_files:
        diff_lines = ["file list/order differs:"]
        diff_lines.extend(difflib.unified_diff(
            py_files, rs_files, fromfile="python.files", tofile="rust.files", lineterm="",
        )[:DIFF_LINE_CAP])
        return "DIFFER", diff_lines

    py_norm = normalize(py_obj)
    rs_norm = normalize(rs_obj)
    if py_norm == rs_norm:
        return "EXACT", []

    # File lists (and order) match; the divergence is in regions, bundle
    # text, token counts, or some other structural field. Report both a
    # region-specific view (spec's explicit ask: file/start/end/text) and a
    # generic recursive diff (spec's "any other structural field") so
    # nothing is silently swallowed.
    diff_lines = region_span_diffs(py_obj, rs_obj, py_files)
    diff_lines.extend(deep_diff(py_norm, rs_norm))
    if not diff_lines:
        diff_lines = ["(structural diff detected but no field-level diff produced -- "
                       "see raw JSON in report)"]
    return "FILES-MATCH-BUT-REGIONS-DIFFER", diff_lines


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="cap total pairs compared (0 = no cap)")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                     help=f"take every Nth lite instance (default {DEFAULT_STRIDE}, "
                          "spreads the sample across every repo since the expected-"
                          "results file is grouped by repo)")
    ap.add_argument("--budget", type=int, default=8192, help="--budget passed to both engines")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                     help="per-engine-invocation subprocess timeout in seconds")
    ap.add_argument("--report", type=Path,
                     default=REPO_ROOT / "parity" / "bundle_parity_report.json",
                     help="write full JSON report to PATH")
    ap.add_argument("--quiet", action="store_true", help="suppress per-pair progress lines")
    ap.add_argument("--rust-bin", type=Path, default=DEFAULT_RUST_BIN)
    ap.add_argument("--lite-expected", type=Path, default=DEFAULT_LITE_EXPECTED)
    ap.add_argument("--lite-parquet", type=Path, default=DEFAULT_LITE_PARQUET)
    ap.add_argument("--lite-repos", type=Path, default=DEFAULT_LITE_REPOS)
    args = ap.parse_args()

    if not args.rust_bin.is_file():
        print(f"error: rust binary not found at {args.rust_bin} "
              f"(build it: cd roust-rs && cargo build --release)", file=sys.stderr)
        sys.exit(2)

    reason = harness.swebench_driver_guard()
    if reason:
        print(f"REFUSED: {reason}", file=sys.stderr)
        sys.exit(1)

    tasks = load_tasks(args.limit, args.stride, args.lite_expected, args.lite_parquet,
                        args.lite_repos)
    if not tasks:
        print("error: no tasks loaded", file=sys.stderr)
        sys.exit(2)

    print(f"comparing {len(tasks)} (query, repo) pairs "
          f"(stride={args.stride}, budget={args.budget})", file=sys.stderr)

    results = []
    counts = {"EXACT": 0, "FILES-MATCH-BUT-REGIONS-DIFFER": 0, "DIFFER": 0, "ERROR": 0}

    for i, task in enumerate(tasks, 1):
        entry = {
            "instance_id": task.task_id,
            "repo": None,
            "verdict": None,
            "error": None,
            "diff": [],
            "python_elapsed_s": None,
            "rust_elapsed_s": None,
        }
        try:
            harness.checkout(task.repo_path, task.base_commit)
        except (RuntimeError, OSError) as exc:
            entry["verdict"] = "ERROR"
            entry["error"] = f"checkout failed: {exc}"
            counts["ERROR"] += 1
            results.append(entry)
            _progress(args, i, len(tasks), task.task_id, "ERROR")
            continue

        entry["repo"] = str(task.repo_path)

        py_obj, py_err, py_elapsed = run_engine(
            python_argv(task.query, task.repo_path, args.budget), REPO_ROOT, args.timeout)
        rs_obj, rs_err, rs_elapsed = run_engine(
            rust_argv(args.rust_bin, task.query, task.repo_path, args.budget), REPO_ROOT,
            args.timeout)
        entry["python_elapsed_s"] = round(py_elapsed, 2)
        entry["rust_elapsed_s"] = round(rs_elapsed, 2)

        if py_err or rs_err:
            entry["verdict"] = "ERROR"
            entry["error"] = f"python: {py_err!r}; rust: {rs_err!r}"
            counts["ERROR"] += 1
            results.append(entry)
            _progress(args, i, len(tasks), task.task_id, "ERROR")
            continue

        verdict, diff_lines = compare(py_obj, rs_obj)
        entry["verdict"] = verdict
        entry["diff"] = diff_lines
        entry["python_cache"] = py_obj.get("stats", {}).get("cache")
        entry["rust_cache"] = rs_obj.get("stats", {}).get("cache")
        entry["python_bundle_tokens"] = py_obj.get("stats", {}).get("bundle_tokens")
        entry["rust_bundle_tokens"] = rs_obj.get("stats", {}).get("bundle_tokens")
        counts[verdict] += 1
        results.append(entry)
        _progress(args, i, len(tasks), task.task_id, verdict)

        if verdict != "EXACT" and not args.quiet:
            for line in diff_lines:
                print(f"      {line}")

    print("\n=== summary ===")
    print(f"pairs compared: {len(tasks)}")
    for k in ("EXACT", "FILES-MATCH-BUT-REGIONS-DIFFER", "DIFFER", "ERROR"):
        print(f"  {k:32} {counts[k]}")

    safe_to_delete = counts["FILES-MATCH-BUT-REGIONS-DIFFER"] == 0 and counts["DIFFER"] == 0 \
        and counts["ERROR"] == 0
    print(f"\nVERDICT: {'SAFE' if safe_to_delete else 'NOT SAFE'} to delete src/roust on this "
          f"evidence ({counts['EXACT']}/{len(tasks)} exact)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "n_pairs": len(tasks),
            "stride": args.stride,
            "budget": args.budget,
            "counts": counts,
            "safe_to_delete_src_roust": safe_to_delete,
            "results": results,
        }, indent=2))
        print(f"full report written to {args.report}")

    sys.exit(0 if counts["DIFFER"] == 0 and counts["ERROR"] == 0 else 1)


def _progress(args, i: int, n: int, task_id: str, verdict: str) -> None:
    if args.quiet:
        return
    print(f"[{i}/{n}] {task_id:45} {verdict}", flush=True)


if __name__ == "__main__":
    main()
