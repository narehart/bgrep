#!/usr/bin/env python3
"""Region-eval Part A, re-run against the SHIPPED Rust engine, PERSISTING spans.

This is a lightweight variant of `parity/region_eval.py`'s Part A: same gold-
hunk parsing (`parse_gold_hunks`), same checkout-then-invoke loop, same
per-instance aggregate fields (`hunk_file_covered`, `hunk_line_recall`,
`hunk_touched`, `all_gold_files_retrieved`, `tokens`) -- but each record ALSO
keeps the raw `regions` dict the engine returned (`obj["regions"]`), plus the
engine's self-reported provenance (`engine_sha`, `engine_dirty` from the
`--json` bundle's `stats` block). `region_eval.py`'s Part A discards the
regions dict after computing the aggregate fractions (see its lines ~268-311);
this script exists so downstream tools (lab/agentless_metric.py) can compute
the EXACT function-level Agentless metric, which needs the actual predicted
region spans, not just aggregate coverage fractions.

Path differences from region_eval.py (whose defaults point at a stale
scratchpad tree and a stale `.venv-pkg` Python shim binary):
  - ROUST_BIN   -> roust-rs/target/release/roust (the shipped Rust engine)
  - LITE_PARQUET / LITE_REPOS -> lab/swebench_lite.parquet, lab/swebench_repos/
    (the durable, checked-in-path data), same convention parity/bundle_parity.py
    uses for its own defaults.

Part B (archex expected_regions) is out of scope here -- not needed for the
function-level metric -- so it is not reproduced in this script.

Usage:
    python parity/region_eval2.py [--limit N] [--timeout SECONDS] \\
        --report lab/results_regions/full300_v8.jsonl

Output: one JSON object per line (JSONL), one line per instance, written as
each instance completes (so a partial run is still usable / resumable-by-eye).
A final summary line prefixed "# " is NOT written to the JSONL (kept pure
JSONL, one record per instance) -- the aggregate summary instead prints to
stdout only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_eval import parse_gold_hunks, line_in_spans, swebench_driver_guard  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# SHARED-CLONE HAZARD (issue #41): this script MUTATES lab/swebench_repos/
# (git checkout -f + git clean -fdq per instance). That directory has
# historically been ONE physical set of clones shared across every worktree
# of this repo -- two concurrent evals (or an eval plus anything else reading
# those clones) in different worktrees race each other's checkouts and
# silently corrupt results. Discipline: when anything else might touch the
# clones, run from a worktree whose lab/swebench_repos/ is a PRIVATE copy
# (`cp -R` of the main repo's clones), never the main repo's shared
# directory.
# ---------------------------------------------------------------------------
ROUST_BIN = REPO_ROOT / "roust-rs" / "target" / "release" / "roust"
LITE_PARQUET = REPO_ROOT / "lab" / "swebench_lite.parquet"
LITE_REPOS = REPO_ROOT / "lab" / "swebench_repos"

BUDGET = 8192
DEFAULT_TIMEOUT_S = 180
PROGRESS_EVERY = 25


def engine_version_string() -> str:
    proc = subprocess.run([str(ROUST_BIN), "--version"], capture_output=True, text=True, timeout=30)
    return proc.stdout.strip() or proc.stderr.strip()


# roust --version prints e.g. "roust 0.2.0 (c591d75, clean)" -- see
# roust-rs/build.rs (engine provenance embed, `git rev-parse --short HEAD`)
# and roust-rs/src/main.rs (clap version wiring).
_ENGINE_VERSION_RE = re.compile(r"^roust (\S+) \(([0-9a-f]+|unknown), (clean|dirty)\)\s*$")


def check_engine_provenance(allow_stale: bool) -> str:
    """BLOCKING engine-provenance guard, mirroring lab/tokenbench/run_bench.py's
    check_engine_provenance: before any instance is evaluated, compare the
    ROUST_BIN binary's embedded git SHA + roust-rs-scoped dirty flag
    (`--version`) against this worktree's current HEAD and roust-rs/ dirty
    state, and refuse to run (SystemExit) on any mismatch/dirtiness unless
    `allow_stale` (--allow-stale-engine) is set, in which case a loud warning
    is printed and the run proceeds. Guards against silently scoring a stale
    target/release/roust left over from an earlier commit -- exactly the class
    of provenance bug the per-record engine_sha/engine_dirty fields exist to
    make auditable after the fact; this check makes it a hard failure up
    front instead.

    Returns the full `--version` line for logging."""
    version_line = engine_version_string()
    m = _ENGINE_VERSION_RE.match(version_line)
    if not m:
        msg = (f"engine provenance guard: could not parse '{ROUST_BIN} --version' "
               f"output: {version_line!r}. Rebuild with `cargo build --release` in "
               f"roust-rs/, or pass --allow-stale-engine to override.")
        if not allow_stale:
            raise SystemExit(msg)
        print(f"WARNING (--allow-stale-engine): {msg}", file=sys.stderr, flush=True)
        return version_line

    _pkg_version, engine_sha, engine_dirty_label = m.groups()

    repo_sha = None
    repo_dirty = None
    try:
        sha_r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10)
        if sha_r.returncode == 0:
            repo_sha = sha_r.stdout.strip()
        dirty_r = subprocess.run(["git", "status", "--porcelain", "--", "roust-rs/"],
                                  cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        if dirty_r.returncode == 0:
            repo_dirty = bool(dirty_r.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        pass

    sha_mismatch = repo_sha is not None and engine_sha != "unknown" and engine_sha != repo_sha
    stale = sha_mismatch or engine_dirty_label == "dirty" or bool(repo_dirty)

    if stale:
        reasons = []
        if sha_mismatch:
            reasons.append(f"binary sha {engine_sha!r} != worktree HEAD {repo_sha!r}")
        if engine_dirty_label == "dirty":
            reasons.append("binary was built from a dirty roust-rs/ tree")
        if repo_dirty:
            reasons.append("worktree roust-rs/ currently has uncommitted changes")
        msg = (
            f"engine provenance guard: stale/dirty roust engine detected "
            f"({'; '.join(reasons)}). The binary at {ROUST_BIN} ({version_line!r}) does "
            f"not match this worktree's current roust-rs/ state -- rebuild with "
            f"`cargo build --release` in roust-rs/ (and commit/stash roust-rs/ changes), "
            f"then re-run. Override with --allow-stale-engine (not recommended for real "
            f"results)."
        )
        if allow_stale:
            print(f"WARNING (--allow-stale-engine): {msg}", file=sys.stderr, flush=True)
        else:
            raise SystemExit(msg)

    return version_line


def run_roust(query: str, repo_path: Path, timeout: float, pad_lines: int = 0,
              len_exp: float = 1.0) -> tuple[dict | None, str | None]:
    argv = [str(ROUST_BIN), "--json", "--budget", str(BUDGET), query, str(repo_path)]
    if pad_lines != 0:
        # only appended for non-default (0) values here -- 0 is this
        # script's own "flags off" sentinel, kept distinct from roust-rs/
        # src/main.rs's own `--pad-lines` default (5, post-adoption). Not
        # passing `--pad-lines` at all (this script's default invocation)
        # lets the roust binary fall back to ITS OWN default, i.e. the
        # shipped-engine default of the moment (5 post-adoption, 0 pre-).
        argv += ["--pad-lines", str(pad_lines)]
    if len_exp != 1.0:
        # only appended for non-default (1.0) values here -- 1.0 is this
        # script's own "flags off" sentinel, kept distinct from roust-rs/
        # src/main.rs's own `--len-exp` default (0.85, post-adoption). Not
        # passing `--len-exp` at all (this script's default invocation) lets
        # the roust binary fall back to ITS OWN default, i.e. the
        # shipped-engine default of the moment (0.85 post-adoption, 1.0
        # pre-).
        argv += ["--len-exp", str(len_exp)]
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


def load_lite_rows(limit: int) -> list[dict]:
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
    if limit:
        rows = rows[:limit]
    return rows


def eval_lite_instance(row: dict, timeout: float, pad_lines: int = 0, len_exp: float = 1.0) -> dict:
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

    repo_path = LITE_REPOS / row["repo"].replace("/", "__")
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
    ap.add_argument("--limit", type=int, default=0, help="cap instance count (0 = all 300)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--report", type=Path, required=True, help="JSONL output path")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--pad-lines", type=int, default=0,
                     help="passthrough to roust's --pad-lines (E12); 0 (default, and the only "
                          "value this script treats as 'omit the flag') means the roust binary "
                          "uses ITS OWN default (5 post-adoption); any other value is forwarded "
                          "as-is (there is no way to force an explicit `--pad-lines 0` through "
                          "this script's CLI -- invoke the roust binary directly for that)")
    ap.add_argument("--len-exp", type=float, default=1.0,
                     help="passthrough to roust's --len-exp (E14/issue #14); 1.0 (default, and "
                          "the only value this script treats as 'omit the flag') means the roust "
                          "binary uses ITS OWN default (0.85 post-adoption); any other value is "
                          "forwarded as-is (there is no way to force an explicit `--len-exp 1.0` "
                          "through this script's CLI -- invoke the roust binary directly for that)")
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

    version = check_engine_provenance(args.allow_stale_engine)
    print(f"engine version: {version}", file=sys.stderr)

    rows = load_lite_rows(args.limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_err = 0
    t0 = time.time()
    with args.report.open("w") as fh:
        for i, row in enumerate(rows, 1):
            rec = eval_lite_instance(row, args.timeout, args.pad_lines, args.len_exp)
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
