#!/usr/bin/env python3
"""Fetch the FULL SWE-bench test split (princeton-nlp/SWE-bench, split=test,
2,294 instances) for the paper eval wave (issue #4 follow-on).

Writes two artifacts:
  - lab/swebench_full.parquet          -- all columns, all rows of the test
    split (same keep-every-column convention as the committed
    lab/swebench_lite.parquet / lab/swebench_verified_heldout.parquet gold
    parquets; parity/region_eval_full.py and lab/agentless_metric_full.py
    read only repo / instance_id / base_commit / patch / problem_statement)
  - lab/swebench_full_instances.txt    -- sorted instance-ID list, one per
    line (committed; the durable record of exactly which instances the
    full-bench numbers cover, same convention as
    lab/swebench_verified_heldout_instances.txt)

and then VERIFIES repo coverage: every distinct `repo` in the split must have
a clone at <clones-dir>/<org>__<name> (the 12 SWE-bench clones under
lab/swebench_repos/). Any uncovered repo is reported with its instance count
and the script exits nonzero -- the full-bench harness cannot run instances
whose repo we do not have.

No `datasets` dependency: the split is fetched as the Hub's parquet
conversion via plain HTTPS (requests + pyarrow, both in .venv-pkg), the same
requests-only HF access pattern lab/mswe_adapter.py established.

  metadata: https://datasets-server.huggingface.co/size?dataset=<ds>
  shards:   https://huggingface.co/api/datasets/<ds>/parquet/<config>/<split>

Usage:
    .venv-pkg/bin/python scripts/fetch_swebench_full.py --dry-run   # metadata only
    .venv-pkg/bin/python scripts/fetch_swebench_full.py             # full download
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET = "princeton-nlp/SWE-bench"
CONFIG = "default"
SPLIT = "test"
EXPECTED_ROWS = 2294  # the published SWE-bench test-split size

DEFAULT_OUT = REPO_ROOT / "lab" / "swebench_full.parquet"
DEFAULT_INSTANCES_OUT = REPO_ROOT / "lab" / "swebench_full_instances.txt"
DEFAULT_CLONES_DIR = REPO_ROOT / "lab" / "swebench_repos"

SIZE_URL = "https://datasets-server.huggingface.co/size?dataset={ds}"
SHARDS_URL = "https://huggingface.co/api/datasets/{ds}/parquet/{config}/{split}"


def fetch_split_metadata(dataset: str, config: str, split: str) -> dict:
    """Returns the datasets-server size record for (config, split):
    {"num_rows": ..., "num_bytes_parquet_files": ...}."""
    import requests
    r = requests.get(SIZE_URL.format(ds=dataset), timeout=60)
    r.raise_for_status()
    for rec in r.json()["size"]["splits"]:
        if rec["config"] == config and rec["split"] == split:
            return rec
    raise SystemExit(f"split {config}/{split} not found in size response for {dataset}")


def fetch_shard_urls(dataset: str, config: str, split: str) -> list[str]:
    import requests
    r = requests.get(SHARDS_URL.format(ds=dataset, config=config, split=split), timeout=60)
    r.raise_for_status()
    urls = r.json()
    if not isinstance(urls, list) or not urls:
        raise SystemExit(f"no parquet shards listed for {dataset} {config}/{split}: {urls!r}")
    return urls


def download_split(urls: list[str]):
    """Downloads every parquet shard and returns (concatenated pyarrow Table,
    total_downloaded_bytes)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    tables = []
    total_bytes = 0
    for i, url in enumerate(urls, 1):
        print(f"  downloading shard {i}/{len(urls)}: {url}", file=sys.stderr)
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        total_bytes += len(r.content)
        tables.append(pq.read_table(io.BytesIO(r.content)))
    return pa.concat_tables(tables), total_bytes


def coverage_report(repos: dict[str, int], clones_dir: Path) -> list[tuple[str, int]]:
    """Prints per-repo instance counts vs the clone directory; returns the
    list of (repo, n_instances) whose clone is MISSING."""
    missing: list[tuple[str, int]] = []
    print(f"\nrepo coverage vs {clones_dir}:", file=sys.stderr)
    for repo in sorted(repos):
        slug = repo.replace("/", "__")
        have = (clones_dir / slug).is_dir()
        mark = "ok " if have else "MISSING"
        print(f"  {mark:8} {repo:35} {repos[repo]:5d} instances", file=sys.stderr)
        if not have:
            missing.append((repo, repos[repo]))
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                     help="print split metadata (row count, parquet bytes, shard list) "
                          "and exit without downloading")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--instances-out", type=Path, default=DEFAULT_INSTANCES_OUT)
    ap.add_argument("--clones-dir", type=Path, default=DEFAULT_CLONES_DIR,
                     help="directory of SWE-bench repo clones (<org>__<name>) to "
                          "verify coverage against")
    args = ap.parse_args()

    meta = fetch_split_metadata(DATASET, CONFIG, SPLIT)
    urls = fetch_shard_urls(DATASET, CONFIG, SPLIT)
    print(f"{DATASET} {CONFIG}/{SPLIT}: {meta['num_rows']} rows, "
          f"{meta['num_bytes_parquet_files'] / 1e6:.1f} MB parquet on the Hub, "
          f"{len(urls)} shard(s)", file=sys.stderr)
    if meta["num_rows"] != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} rows in the {SPLIT} split, Hub reports "
                         f"{meta['num_rows']} -- refusing to fetch a split that does not "
                         f"match the published SWE-bench test set")
    if args.dry_run:
        for u in urls:
            print(f"  shard: {u}", file=sys.stderr)
        print("dry run: no download performed", file=sys.stderr)
        return

    table, total_bytes = download_split(urls)
    if table.num_rows != EXPECTED_ROWS:
        raise SystemExit(f"downloaded {table.num_rows} rows, expected {EXPECTED_ROWS}")

    import pyarrow.parquet as pq
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)

    ids = sorted(table.column("instance_id").to_pylist())
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate instance_id values in the downloaded split")
    args.instances_out.parent.mkdir(parents=True, exist_ok=True)
    args.instances_out.write_text("\n".join(ids) + "\n")

    repos: dict[str, int] = {}
    for repo in table.column("repo").to_pylist():
        repos[repo] = repos.get(repo, 0) + 1
    missing = coverage_report(repos, args.clones_dir)

    print(f"\ndownloaded {total_bytes / 1e6:.1f} MB; wrote {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB, {table.num_rows} rows, "
          f"{len(repos)} distinct repos) and {args.instances_out} ({len(ids)} ids)",
          file=sys.stderr)
    if missing:
        n_lost = sum(n for _, n in missing)
        print(f"COVERAGE FAILURE: {len(missing)} repo(s) with no clone "
              f"({n_lost} instances unrunnable): "
              + ", ".join(f"{r} ({n})" for r, n in missing), file=sys.stderr)
        raise SystemExit(1)
    print("coverage OK: every repo in the split has a clone", file=sys.stderr)


if __name__ == "__main__":
    main()
