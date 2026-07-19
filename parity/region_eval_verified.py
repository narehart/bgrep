#!/usr/bin/env python3
"""Region-eval Part A, re-run on SWE-bench VERIFIED held-out instances.

Modeled directly on `parity/region_eval2.py` (same gold-hunk parsing via
`parse_gold_hunks`, same checkout-then-invoke loop against a PRIVATE repo
checkout, same per-instance record shape: `hunk_file_covered`,
`hunk_line_recall`, `hunk_touched`, `all_gold_files_retrieved`, `tokens`,
plus the persisted `regions` dict and engine provenance `engine_sha` /
`engine_dirty`) -- but the instance source is a SWE-bench VERIFIED parquet
(default: the held-out subset, i.e. Verified minus the Lite overlap) instead
of `region_eval2.py`'s hardcoded Lite parquet.

This script exists for the out-of-sample validation campaign (issue #4
follow-on): confirm or refute, on instances the engine's defaults were never
tuned against, the region-packing gains measured on Lite (the adopted
padding + length-normalization defaults, commit 5e81c8a).

Path differences from region_eval2.py:
  - Gold parquet is a CLI flag (`--gold-parquet`), not a hardcoded constant --
    defaults to `lab/swebench_verified_heldout.parquet` (the held-out subset
    computed once and cached to a durable path, same convention
    `swebench_lite.parquet` uses for Lite).
  - Repo checkouts still read from `lab/swebench_repos/` (REPO_ROOT-relative,
    same as region_eval2.py) -- in THIS worktree that directory is a private
    `cp -R` of the main repo's clones (issue #41 standard), so a Verified run
    here cannot race or corrupt a concurrent Lite run elsewhere.
  - `--pad-lines` / `--len-exp` are ALWAYS forwarded to the roust binary
    (no "0 means omit the flag" sentinel like region_eval2.py's CLI has).
    region_eval2.py's sentinel trick exists because 0/1.0 used to be the
    engine's own pre-adoption defaults and it needed an "unset" value to
    fall back to whatever the shipped binary's default of the moment was;
    that ambiguity is undesirable here where the whole point is a byte-exact
    A/B between two NAMED formulas (new defaults vs the old formula), so
    this script's flags default to the current engine defaults (5 / 0.85)
    and are unconditionally passed through -- there is no implicit-default
    mode. To reproduce the pre-adoption ("old formula") arm, pass
    `--pad-lines 0 --len-exp 1.0` explicitly.

Usage:
    python parity/region_eval_verified.py [--limit N] [--timeout SECONDS] \\
        --report lab/results_regions/full407_verified_new.jsonl
    python parity/region_eval_verified.py --pad-lines 0 --len-exp 1.0 \\
        --report lab/results_regions/full407_verified_old.jsonl

Output: one JSON object per line (JSONL), one line per instance, written as
each instance completes (partial runs are resumable-by-eye). A final summary
line is NOT written to the JSONL -- the aggregate prints to stdout only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_eval import parse_gold_hunks, line_in_spans, swebench_driver_guard  # noqa: E402
from region_eval2 import check_engine_provenance  # noqa: E402  (blocking provenance guard)

REPO_ROOT = Path(__file__).resolve().parent.parent

ROUST_BIN = REPO_ROOT / "roust-rs" / "target" / "release" / "roust"
VERIFIED_HELDOUT_PARQUET = REPO_ROOT / "lab" / "swebench_verified_heldout.parquet"
SWEBENCH_REPOS = REPO_ROOT / "lab" / "swebench_repos"

BUDGET = 8192
DEFAULT_TIMEOUT_S = 180
PROGRESS_EVERY = 25
DEFAULT_PAD_LINES = 5    # current engine default (E12, adopted commit 5e81c8a)
DEFAULT_LEN_EXP = 0.85   # current engine default (E14, adopted commit 5e81c8a)

# Additional flags appended to every roust invocation. Empty here (this
# script always runs the default engine); parity/region_eval_full.py sets
# this module global to its BM25_ONLY_FLAGS for the same-harness BM25 arm.
EXTRA_ENGINE_FLAGS: list[str] = []


def engine_version_string() -> str:
    proc = subprocess.run([str(ROUST_BIN), "--version"], capture_output=True, text=True, timeout=30)
    return proc.stdout.strip() or proc.stderr.strip()


def run_roust(query: str, repo_path: Path, timeout: float, pad_lines: int,
              len_exp: float) -> tuple[dict | None, str | None]:
    argv = [str(ROUST_BIN), "--json", "--budget", str(BUDGET), query, str(repo_path),
            "--pad-lines", str(pad_lines), "--len-exp", str(len_exp),
            *EXTRA_ENGINE_FLAGS]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except OSError as exc:
        return None, f"failed to spawn roust: {exc}"
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


def load_verified_rows(gold_parquet: Path, limit: int) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(gold_parquet)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "patch": row["patch"],
            "problem_statement": row["problem_statement"],
        })
    # deterministic order (same convention as region_eval2.py's parquet
    # iteration order, which is the parquet's own row order); sort by
    # (repo, instance_id) so shared clones are checked out sequentially.
    rows.sort(key=lambda r: (r["repo"], r["instance_id"]))
    if limit:
        rows = rows[:limit]
    return rows


def eval_verified_instance(row: dict, timeout: float, pad_lines: int, len_exp: float) -> dict:
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
        "regions": {},
        "engine_sha": None,
        "engine_dirty": None,
    }
    if not gold_files:
        rec["error"] = "no old-file hunk lines in gold patch (pure file creation(s) only)"
        return rec

    repo_path = SWEBENCH_REPOS / row["repo"].replace("/", "__")
    if not repo_path.exists():
        rec["error"] = f"repo checkout not found: {repo_path}"
        return rec
    try:
        checkout(repo_path, row["base_commit"])
    except (RuntimeError, OSError) as exc:
        rec["error"] = f"checkout failed: {exc}"
        return rec

    obj, err = run_roust(row["problem_statement"], repo_path, timeout, pad_lines, len_exp)
    if err:
        rec["error"] = err
        return rec

    regions: dict[str, list[list[int]]] = obj.get("regions", {})
    rec["regions"] = regions
    files_in_regions = set(regions.keys())

    stats = obj.get("stats", {})
    rec["engine_sha"] = stats.get("engine_sha")
    rec["engine_dirty"] = stats.get("engine_dirty")

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
    rec["tokens"] = stats.get("bundle_tokens")

    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="cap instance count (0 = all)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--report", type=Path, required=True, help="JSONL output path")
    ap.add_argument("--gold-parquet", type=Path, default=VERIFIED_HELDOUT_PARQUET,
                     help="SWE-bench Verified parquet to evaluate against "
                          "(default: the pre-computed held-out subset, "
                          "Verified minus Lite overlap)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--pad-lines", type=int, default=DEFAULT_PAD_LINES,
                     help=f"passthrough to roust's --pad-lines (E12); always forwarded "
                          f"(default {DEFAULT_PAD_LINES}, the current engine default); pass "
                          f"`--pad-lines 0` to reproduce the pre-adoption old formula")
    ap.add_argument("--len-exp", type=float, default=DEFAULT_LEN_EXP,
                     help=f"passthrough to roust's --len-exp (E14); always forwarded "
                          f"(default {DEFAULT_LEN_EXP}, the current engine default); pass "
                          f"`--len-exp 1.0` to reproduce the pre-adoption old formula")
    ap.add_argument("--allow-stale-engine", action="store_true",
                     help="override the blocking engine-provenance guard (logs a loud warning "
                          "instead of refusing) when the roust binary's embedded sha/dirty state "
                          "does not match this worktree's roust-rs/ HEAD/dirty state -- NOT "
                          "recommended for real results")
    args = ap.parse_args()

    if not ROUST_BIN.exists():
        raise SystemExit(f"roust binary not found at {ROUST_BIN}")

    reason = swebench_driver_guard()
    if reason:
        raise SystemExit(f"REFUSED to run: {reason}")

    # Blocking engine-provenance check (shared with region_eval2.py): refuses
    # to run if ROUST_BIN's embedded sha/dirty state does not match this
    # worktree's roust-rs/ HEAD + dirty state. NOTE: region_eval2's guard
    # resolves ROUST_BIN/REPO_ROOT from ITS OWN module constants, which point
    # at the same worktree tree as this script's (__file__-relative), so the
    # comparison is against the correct private checkout.
    version = check_engine_provenance(args.allow_stale_engine)
    print(f"engine version: {version}", file=sys.stderr)
    print(f"gold parquet: {args.gold_parquet}", file=sys.stderr)
    print(f"pad_lines={args.pad_lines} len_exp={args.len_exp}", file=sys.stderr)

    rows = load_verified_rows(args.gold_parquet, args.limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_err = 0
    t0 = time.time()
    with args.report.open("w") as fh:
        for i, row in enumerate(rows, 1):
            rec = eval_verified_instance(row, args.timeout, args.pad_lines, args.len_exp)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            if rec["error"] is None:
                n_ok += 1
            else:
                n_err += 1
            if not args.quiet and (i % PROGRESS_EVERY == 0 or i == len(rows)):
                elapsed = time.time() - t0
                print(f"[{i}/{len(rows)}] {row['instance_id']:45} elapsed={elapsed:.0f}s "
                      f"({'ERR' if rec['error'] else 'ok'}) ok={n_ok} err={n_err}", flush=True,
                      file=sys.stderr)

    print(f"\nengine version: {version}", file=sys.stderr)
    print(f"wrote {len(rows)} records ({n_ok} ok, {n_err} errors) to {args.report}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
