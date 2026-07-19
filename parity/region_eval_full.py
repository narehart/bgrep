#!/usr/bin/env python3
"""Region-eval Part A on the FULL SWE-bench test split (2,294 instances).

Modeled directly on `parity/region_eval_verified.py` (same gold-hunk parsing
via `parse_gold_hunks`, same checkout-then-invoke loop, same per-instance
record shape: `hunk_file_covered`, `hunk_line_recall`, `hunk_touched`,
`all_gold_files_retrieved`, `tokens`, plus the persisted `regions` dict and
engine provenance `engine_sha` / `engine_dirty`; same BLOCKING driver +
engine-provenance guards; same always-forward `--pad-lines` / `--len-exp`
semantics -- see that script's docstring for the full rationale of each).
Differences:

  - Gold parquet defaults to `lab/swebench_full.parquet` (produced by
    `scripts/fetch_swebench_full.py`; instance list committed as
    `lab/swebench_full_instances.txt`).
  - `--shard K/N` (1-based) runs a resumable slice of the split: rows are
    sorted by (repo, instance_id) as always, then shard K takes the
    CONTIGUOUS balanced slice rows[(K-1)*len//N : K*len//N]. Contiguous (not
    strided) so a shard stays within as few repos as possible -- checkouts
    are the expensive, stateful part. `--limit` applies AFTER the shard
    slice. Every shard writes its own `--report` JSONL; concatenate shard
    JSONLs (order irrelevant) for scoring by `lab/agentless_metric_full.py`.
  - `--repos-dir` overrides the clone directory (default
    `lab/swebench_repos/` under this checkout, same as region_eval2.py).
  - `--bm25-only` appends the engine's full channel-ablation flag set
    (`--no-history --no-docs --no-anchors --no-testbridge`) to every roust
    invocation: the same-harness BM25 arm. What remains is the BM25F
    lexical core (Okapi body field + path-token field + comment/NL field +
    implementation-file prior) feeding the identical region packer. What
    this does NOT (cannot) disable: the import-graph/same-directory
    structural expansion inside `select_files` (its co-change input IS
    removed by --no-history) -- there is no engine flag for it. NOTE:
    `personalized_pagerank` needs no flag: it is dead code, never invoked
    by the CLI (see roust-rs/src/core.rs). This flag set is the documented
    definition of "BM25-only" for the paper; see lab/paper/EVAL_PLAN.md.

CHECKOUT DISCIPLINE (issue #41 standard, same as region_eval_verified.py):
this script mutates the working trees under --repos-dir (`git checkout -f`
+ `git clean -fdq` per instance). NEVER point concurrent shards, or a shard
and any other eval, at the same clone directory: give each concurrently
running shard its own private copy of the clones and pass it via
--repos-dir. Adjacent contiguous shards SHARE boundary repos, so "different
shards" is not isolation. The swebench_driver process guard below refuses
to start while a driver is running (override:
BGREP_REGION_EVAL_SKIP_DRIVER_GUARD=1 only with independently confirmed
disjoint repo sets).

Usage:
    python parity/region_eval_full.py --shard 1/8 \\
        --repos-dir /path/to/private_clones_shard1 \\
        --report lab/results_regions/full2294_default_s1of8.jsonl
    python parity/region_eval_full.py --shard 1/8 --bm25-only \\
        --repos-dir /path/to/private_clones_shard1 \\
        --report lab/results_regions/full2294_bm25_s1of8.jsonl

Output: JSONL, one object per instance, written as each instance completes.
Aggregates print to stderr only (never into the JSONL).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_eval import swebench_driver_guard  # noqa: E402
from region_eval2 import check_engine_provenance  # noqa: E402  (blocking provenance guard)
import region_eval_verified as rev  # noqa: E402  (shared eval loop internals)

REPO_ROOT = Path(__file__).resolve().parent.parent

ROUST_BIN = REPO_ROOT / "roust-rs" / "target" / "release" / "roust"
FULL_PARQUET = REPO_ROOT / "lab" / "swebench_full.parquet"
DEFAULT_REPOS_DIR = REPO_ROOT / "lab" / "swebench_repos"

DEFAULT_TIMEOUT_S = 180
PROGRESS_EVERY = 25

# The same-harness BM25 arm: every engine channel with a CLI ablation flag,
# disabled. See module docstring for exactly what remains active.
BM25_ONLY_FLAGS = ["--no-history", "--no-docs", "--no-anchors", "--no-testbridge"]


def parse_shard(spec: str, n_rows: int) -> tuple[int, int]:
    """'K/N' (1-based) -> (start, end) contiguous balanced slice bounds."""
    try:
        k_s, n_s = spec.split("/", 1)
        k, n = int(k_s), int(n_s)
    except ValueError:
        raise SystemExit(f"--shard must look like K/N (e.g. 3/8), got {spec!r}")
    if not (1 <= k <= n):
        raise SystemExit(f"--shard K/N requires 1 <= K <= N, got {spec!r}")
    return (k - 1) * n_rows // n, k * n_rows // n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                     help="cap instance count AFTER shard slicing (0 = all)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--report", type=Path, required=True, help="JSONL output path")
    ap.add_argument("--gold-parquet", type=Path, default=FULL_PARQUET,
                     help="gold parquet to evaluate (default: the full SWE-bench "
                          "test split from scripts/fetch_swebench_full.py)")
    ap.add_argument("--shard", default="1/1",
                     help="K/N (1-based): run the K-th contiguous balanced slice of "
                          "the (repo, instance_id)-sorted rows (default 1/1 = all)")
    ap.add_argument("--repos-dir", type=Path, default=DEFAULT_REPOS_DIR,
                     help="clone directory to check out instances in; concurrent "
                          "shards MUST each get their own private copy (issue #41)")
    ap.add_argument("--bm25-only", action="store_true",
                     help=f"same-harness BM25 arm: append {' '.join(BM25_ONLY_FLAGS)} "
                          f"to every roust invocation (see docstring for the exact "
                          f"definition of what remains active)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--pad-lines", type=int, default=rev.DEFAULT_PAD_LINES,
                     help="passthrough to roust's --pad-lines; ALWAYS forwarded "
                          "(same no-sentinel semantics as region_eval_verified.py)")
    ap.add_argument("--len-exp", type=float, default=rev.DEFAULT_LEN_EXP,
                     help="passthrough to roust's --len-exp; ALWAYS forwarded")
    ap.add_argument("--allow-stale-engine", action="store_true",
                     help="override the blocking engine-provenance guard (loud warning "
                          "instead of refusal) -- NOT recommended for real results")
    args = ap.parse_args()

    if not ROUST_BIN.exists():
        raise SystemExit(f"roust binary not found at {ROUST_BIN}")
    if not args.repos_dir.is_dir():
        raise SystemExit(f"--repos-dir not found: {args.repos_dir}")

    reason = swebench_driver_guard()
    if reason:
        raise SystemExit(f"REFUSED to run: {reason}")
    version = check_engine_provenance(args.allow_stale_engine)

    # Reuse region_eval_verified's loop internals wholesale, repointed at
    # this run's clone directory and (optionally) the BM25-only flag set.
    # rev.run_roust / rev.eval_verified_instance read these module globals.
    rev.SWEBENCH_REPOS = args.repos_dir
    if args.bm25_only:
        rev.EXTRA_ENGINE_FLAGS = BM25_ONLY_FLAGS

    print(f"engine version: {version}", file=sys.stderr)
    print(f"gold parquet: {args.gold_parquet}", file=sys.stderr)
    print(f"repos dir: {args.repos_dir}", file=sys.stderr)
    print(f"pad_lines={args.pad_lines} len_exp={args.len_exp} "
          f"bm25_only={args.bm25_only}", file=sys.stderr)

    rows = rev.load_verified_rows(args.gold_parquet, limit=0)
    start, end = parse_shard(args.shard, len(rows))
    rows = rows[start:end]
    print(f"shard {args.shard}: rows [{start}:{end}] of the sorted split "
          f"({len(rows)} instances)", file=sys.stderr)
    if args.limit:
        rows = rows[: args.limit]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_err = 0
    t0 = time.time()
    with args.report.open("w") as fh:
        for i, row in enumerate(rows, 1):
            rec = rev.eval_verified_instance(row, args.timeout, args.pad_lines, args.len_exp)
            rec["shard"] = args.shard
            rec["bm25_only"] = args.bm25_only
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            if rec["error"] is None:
                n_ok += 1
            else:
                n_err += 1
            if not args.quiet and (i % PROGRESS_EVERY == 0 or i == len(rows)):
                elapsed = time.time() - t0
                print(f"[{i}/{len(rows)}] {row['instance_id']:45} elapsed={elapsed:.0f}s "
                      f"({'ERR' if rec['error'] else 'ok'}) ok={n_ok} err={n_err}",
                      flush=True, file=sys.stderr)

    print(f"\nengine version: {version}", file=sys.stderr)
    print(f"wrote {len(rows)} records ({n_ok} ok, {n_err} errors) to {args.report}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
