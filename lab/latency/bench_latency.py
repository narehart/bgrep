#!/usr/bin/env python3
"""Latency benchmark for roust (#15): cold/warm index build and query timing
across repo sizes.

For each target repo, measures:
  - cold index: remove <repo>/.roust/, run one --json query, take
    stats.index_ms + end-to-end wall time; repeat 3x, report median.
  - warm index (cache hit): immediately re-run the same query, 5x, median.
  - query time: with a warm cache, run 20 queries cycling through a fixed
    list of ~10 realistic (problem-statement-like) phrases; report p50/p95
    of stats.query_ms and of end-to-end wall time (subprocess overhead
    included -- that's what an agent actually experiences).

Repos are expected to already be plain directories (this script does not
clone or copy anything -- the caller is responsible for pointing it at
disposable copies, never at a live checkout).

Usage:
    python3 bench_latency.py --engine /path/to/roust \
        --repo name=path [--repo name2=path2 ...] \
        --out lab/latency/latency_v1.json
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Fixed list of ~10 realistic, problem-statement-like query phrases, reused
# across repos so the query-time measurement isn't skewed by hand-picking
# terms per codebase.
QUERIES = [
    "fix off-by-one error in loop boundary",
    "add support for custom authentication headers",
    "handle unicode decoding error in response body",
    "connection pooling timeout not respected",
    "improve error message when validation fails",
    "memory leak when processing large files",
    "race condition in concurrent request handling",
    "deprecate old configuration option safely",
    "unexpected behavior when parsing malformed input",
    "add caching layer for repeated queries",
]


def run_query(engine, query, repo_path):
    args = [engine, query, str(repo_path), "--json"]
    t0 = time.perf_counter()
    proc = subprocess.run(args, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    # 0 = results found (incl. low-confidence), 1 = no query term matched at
    # all -- both still emit a JSON body on stdout. 2 is a real usage error.
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"roust exited {proc.returncode} on query {query!r} against "
            f"{repo_path}: {proc.stderr}"
        )
    data = json.loads(proc.stdout)
    return data, wall_ms


def dir_size_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in (".git", ".roust")]
        for name in files:
            fp = os.path.join(root, name)
            if os.path.islink(fp):
                continue
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def percentile(values, p):
    """Nearest-rank percentile; deterministic on small n, no numpy dep."""
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100.0 * (len(s) - 1))))
    return s[k]


def bench_repo(engine, repo_path):
    repo_path = Path(repo_path)
    roust_dir = repo_path / ".roust"

    # Cold index: nuke the cache, run one query, repeat 3x -> median.
    cold_index_ms, cold_wall_ms = [], []
    data = None
    for _ in range(3):
        shutil.rmtree(roust_dir, ignore_errors=True)
        data, wall_ms = run_query(engine, QUERIES[0], repo_path)
        cold_index_ms.append(data["stats"]["index_ms"])
        cold_wall_ms.append(wall_ms)

    files_indexed = data["stats"]["files_indexed"]

    # Warm index (cache hit): immediately re-run, 5x -> median.
    warm_index_ms, warm_wall_ms = [], []
    for _ in range(5):
        data, wall_ms = run_query(engine, QUERIES[0], repo_path)
        warm_index_ms.append(data["stats"]["index_ms"])
        warm_wall_ms.append(wall_ms)

    # Query time: warm cache, 20 queries cycling the fixed phrase list.
    query_ms, query_wall_ms = [], []
    for i in range(20):
        q = QUERIES[i % len(QUERIES)]
        data, wall_ms = run_query(engine, q, repo_path)
        query_ms.append(data["stats"]["query_ms"])
        query_wall_ms.append(wall_ms)

    return {
        "files_indexed": files_indexed,
        "repo_size_bytes": dir_size_bytes(repo_path),
        "cold_index": {
            "index_ms_median": statistics.median(cold_index_ms),
            "wall_ms_median": statistics.median(cold_wall_ms),
            "index_ms_samples": cold_index_ms,
            "wall_ms_samples": [round(w, 1) for w in cold_wall_ms],
        },
        "warm_index": {
            "index_ms_median": statistics.median(warm_index_ms),
            "wall_ms_median": statistics.median(warm_wall_ms),
            "index_ms_samples": warm_index_ms,
            "wall_ms_samples": [round(w, 1) for w in warm_wall_ms],
        },
        "query": {
            "query_ms_p50": percentile(query_ms, 50),
            "query_ms_p95": percentile(query_ms, 95),
            "wall_ms_p50": round(percentile(query_wall_ms, 50), 1),
            "wall_ms_p95": round(percentile(query_wall_ms, 95), 1),
            "query_ms_samples": query_ms,
            "wall_ms_samples": [round(w, 1) for w in query_wall_ms],
        },
    }


def get_machine_info():
    uname_m = platform.uname().machine
    cpu = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True,
    ).stdout.strip()
    return {"uname_m": uname_m, "cpu": cpu}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="path to the roust binary")
    ap.add_argument(
        "--repo", action="append", required=True, dest="repos",
        metavar="name=path", help="repo to benchmark, repeatable",
    )
    ap.add_argument("--out", required=True, help="output JSON artifact path")
    args = ap.parse_args()

    engine = str(Path(args.engine).resolve())
    version = subprocess.run(
        [engine, "--version"], capture_output=True, text=True
    ).stdout.strip()

    repo_results = {}
    for spec in args.repos:
        name, path = spec.split("=", 1)
        print(f"benchmarking {name} ({path})...", file=sys.stderr)
        repo_results[name] = bench_repo(engine, path)
        r = repo_results[name]
        print(
            f"  files_indexed={r['files_indexed']} "
            f"cold_index_ms_median={r['cold_index']['index_ms_median']} "
            f"warm_index_ms_median={r['warm_index']['index_ms_median']} "
            f"query_ms_p50={r['query']['query_ms_p50']} "
            f"query_ms_p95={r['query']['query_ms_p95']}",
            file=sys.stderr,
        )

    artifact = {
        "engine_version": version,
        "machine": get_machine_info(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queries": QUERIES,
        "repos": repo_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")

    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
