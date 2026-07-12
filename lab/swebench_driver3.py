"""SWE-bench Lite file-level localization with the bgrep pipeline (lanes2 fork).

Same task as swebench_driver.py, but runs lanes2 and can optionally turn on
the git-history semantic signals: commit-message field (--history, also
enables co-change frontier expansion, including test-bridge edges) and
comment/docstring field (--comments). With no flags, recall is identical to
swebench_driver.py's lanes recall (lanes2's unconditional vendor-exclusion /
pack-safety fixes can shift corpus file counts and packed token counts, but
not recall on non-vendor gold files -- see lanes2.py's module docstring).

--anchors turns on the definition-symbol anchor channel (lanes2.
extract_symbol_anchors): identifiers named verbatim in the issue text that
are rarely-defined (<=3 defining files) in the repo get promoted into
select_files()'s output at position 7, per lanes2._apply_anchor_promotions.
This never changes the body ranking's top-7.

For each instance: checkout repo@base_commit, run the pipeline with the RAW
problem statement as the query (anchor preservation; no helper keywords), and
score file-level recall of the gold-patch-edited files plus packed tokens.

Writes one JSON line per instance (resume-safe: already-done instances skipped).

--testbridge turns on the test-file lexical bridge channel (lanes2.
select_files use_testbridge / lanes2._apply_testbridge_promotions): impl
files reachable via the import graph from the top-3 BM25-matching test files
get promoted (tail-only, position 14, cap 3), ranked by a specificity score
that deprioritizes candidates imported by many test files.

--docsbridge turns on the docs lexical bridge channel (lanes2.select_files
use_docsbridge / lanes2._apply_docsbridge_promotions): impl files resolved
from dotted-path / Sphinx-directive code references on the top-3 BM25-
matching *.rst/*.txt/*.md doc pages get promoted (tail-only, position 16,
cap 2). Requires Corpus(build_docs=True), which is passed automatically when
this flag is set.

Both --testbridge and --docsbridge are tail-only channels (insert at
position >=14, never above): the top-10 returned_files ordering with either
or both flags on is identical to the ordering with both off.

Usage:  uv run python swebench_driver2.py [--limit N] [--history] [--comments] [--anchors]
                                           [--testbridge] [--docsbridge]
                                           [--instances-file PATH | --sample N] --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

import lanes3 as L  # noqa: E402
from history import mine_history  # noqa: E402
from archex.reporting import count_tokens  # noqa: E402

PARQUET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/"
    "resolve/main/data/test-00000-of-00001.parquet"
)
REPO_CACHE = LAB_DIR / "swebench_repos"
_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/", re.M)


def load_instances(parquet_url: str = PARQUET_URL, parquet_cache: Path | None = None) -> list[dict]:
    import pandas as pd  # provided by archex venv

    cache = parquet_cache if parquet_cache is not None else LAB_DIR / "swebench_lite.parquet"
    if not cache.exists():
        print(f"downloading SWE-bench parquet from {parquet_url}...", flush=True)
        urllib.request.urlretrieve(parquet_url, cache)
    df = pd.read_parquet(cache)
    out = []
    for _, row in df.iterrows():
        gold = sorted(set(_DIFF_FILE_RE.findall(row["patch"])))
        out.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
            "gold_files": gold,
        })
    # group by repo so the shared clone checks out sequentially
    out.sort(key=lambda r: (r["repo"], r["instance_id"]))
    return out


def repo_clone(slug: str) -> Path:
    dest = REPO_CACHE / slug.replace("/", "__")
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {slug} (full history)...", flush=True)
    r = subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{slug}.git", str(dest)],
        capture_output=True, text=True, timeout=3600,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"clone {slug} failed: {r.stderr.strip()[:300]}")
    return dest


def checkout(repo: Path, sha: str) -> None:
    r = subprocess.run(["git", "checkout", "-f", "-q", sha], cwd=repo,
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"checkout {sha} failed: {r.stderr.strip()[:300]}")
    subprocess.run(["git", "clean", "-fdq"], cwd=repo, capture_output=True,
                   text=True, timeout=300, encoding="utf-8", errors="replace")


def _list_current_files(repo_path: Path, extensions: tuple = L.CODE_EXTENSIONS) -> set[str]:
    """Cheap mirror of Corpus's file-collection filter (extension, .git,
    size cap) without reading/tokenizing file contents -- just enough to
    let mine_history() drop history entries for files that no longer exist
    at this checkout."""
    files: set[str] = set()
    for p in repo_path.rglob("*"):
        if not p.is_file() or p.suffix not in extensions:
            continue
        rel = str(p.relative_to(repo_path))
        if rel.startswith(".git/") or "/.git/" in rel:
            continue
        try:
            if p.stat().st_size > L.MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files.add(rel)
    return files


def run_instance(
    inst: dict, repo_path: Path, use_history: bool, use_comments: bool, use_anchors: bool = False,
    use_testbridge: bool = False, use_docsbridge: bool = False, extensions: tuple = L.CODE_EXTENSIONS,
    use_neighborhood: bool = False, use_global_order: bool = False,
) -> dict:
    t0 = time.perf_counter()

    # IMPORTANT correctness detail: history must be mined AT THE INSTANCE'S
    # CHECKOUT (i.e. strictly after checkout(repo, base_commit) has already
    # run, which the caller guarantees). `git log` with no explicit range
    # walks back from HEAD, and HEAD is base_commit at this point, so
    # mine_history() can only ever see commits <= base_commit -- no future
    # leakage of history that postdates the instance's gold patch.
    history_msgs: dict[str, str] | None = None
    cochange: dict[str, dict[str, int]] | None = None
    meta: dict[str, dict] | None = None
    mine_ms = 0.0
    if use_history:
        t_mine = time.perf_counter()
        current_files = _list_current_files(repo_path, extensions=extensions)
        history_msgs, cochange, meta = mine_history(repo_path, current_files=current_files)
        mine_ms = (time.perf_counter() - t_mine) * 1000

    corpus = L.Corpus(repo_path, history_msgs=history_msgs, use_comments=use_comments,
                       build_docs=use_docsbridge, extensions=extensions)
    build_ms = (time.perf_counter() - t0) * 1000
    terms = L.query_terms(inst["problem_statement"], [])
    anchors: list[tuple[str, float]] | None = None
    if use_anchors:
        anchors = L.extract_symbol_anchors(inst["problem_statement"], corpus)
    t1 = time.perf_counter()
    files, scores = L.select_files(corpus, terms, use_ppr=True, cochange=cochange, anchors=anchors,
                                    use_testbridge=use_testbridge, use_docsbridge=use_docsbridge,
                                    use_neighborhood=use_neighborhood, use_global_order=use_global_order)
    spans, bundle = L.pack_regions(corpus, files, terms, scores, 8192, count_tokens)
    query_ms = (time.perf_counter() - t1) * 1000
    packed_files = [f for f in files if f in spans]
    gold = [g for g in inst["gold_files"] if g.endswith(tuple(extensions))]
    fset = set(packed_files)
    present = [g for g in gold if g in fset]
    recall = len(present) / len(gold) if gold else 1.0
    cochange_additions = list(L.LAST_EXPLAIN.get("cochange_additions", []))
    anchor_promotions = list(L.LAST_EXPLAIN.get("anchor_promotions", []))
    testbridge = list(L.LAST_EXPLAIN.get("testbridge", []))
    docsbridge = list(L.LAST_EXPLAIN.get("docsbridge", []))
    neighborhood = dict(L.LAST_EXPLAIN.get("neighborhood", {}))
    return {
        "instance_id": inst["instance_id"],
        "repo": inst["repo"],
        "gold_files": gold,
        "n_gold": len(gold),
        "present": present,
        "missing": [g for g in gold if g not in fset],
        "recall": recall,
        "all_present": len(present) == len(gold),
        "n_returned": len(packed_files),
        "returned_files": packed_files,  # ordered: lexical picks then additions
        "tokens_packed": count_tokens(bundle),
        "corpus_files": corpus.n_docs,
        "docs_files": corpus.n_docs_files,
        "build_ms": round(build_ms),
        "query_ms": round(query_ms),
        "mine_ms": round(mine_ms),
        "signals": {"history": use_history, "comments": use_comments, "anchors": use_anchors,
                    "testbridge": use_testbridge, "docsbridge": use_docsbridge,
                    "extensions": "extended" if extensions == L.EXTENDED_EXTENSIONS else "default",
                    "neighborhood": use_neighborhood, "global_order": use_global_order},
        "cochange_additions": cochange_additions,
        "anchor_promotions": anchor_promotions,
        "testbridge": testbridge,
        "docsbridge": docsbridge,
        "neighborhood": neighborhood,
        "meta_available": bool(meta),  # mined but not yet used for scoring
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--history", action="store_true",
                     help="mine git history at each checkout; pass commit-message field + co-change edges")
    ap.add_argument("--comments", action="store_true",
                     help="extract comment/docstring NL field (use_comments=True)")
    ap.add_argument("--anchors", action="store_true",
                     help="promote definition-symbol anchor files (identifiers named verbatim "
                          "in the issue that are rarely-defined in the repo) into the top ranks")
    ap.add_argument("--testbridge", action="store_true",
                     help="promote impl files reachable via the import graph from the top-3 "
                          "BM25-matching test files (lanes2.select_files use_testbridge)")
    ap.add_argument("--docsbridge", action="store_true",
                     help="promote impl files resolved from dotted-path/Sphinx-directive code "
                          "references on the top-3 BM25-matching doc pages (lanes2.select_files "
                          "use_docsbridge); also enables Corpus(build_docs=True)")
    ap.add_argument("--instances-file", default=None,
                     help="newline-separated instance_ids to run only those")
    ap.add_argument("--sample", type=int, default=0,
                     help="take every Nth instance (by the (repo, instance_id) sort order "
                          "load_instances() produces) instead of --instances-file; "
                          "e.g. --sample 15 on 300 instances yields 20")
    ap.add_argument("--parquet-url", default=PARQUET_URL,
                     help="override the SWE-bench parquet dataset URL (default: SWE-bench Lite)")
    ap.add_argument("--parquet-cache", default=None,
                     help="override the local parquet cache path (default: swebench_lite.parquet in LAB_DIR)")
    ap.add_argument("--neighborhood", action="store_true",
                     help="neighborhood-first retrieval for large (>3000-file) repos "
                          "(lanes2.select_files use_neighborhood): seed-anchor + structural-"
                          "expand a region, then mask body bm25 to it before ranking; "
                          "no-op on repos at or below the threshold")
    ap.add_argument("--extensions", choices=["default", "extended"], default="default",
                     help="'extended' adds PHP/Ruby/C-family extensions (lanes2.EXTENDED_EXTENSIONS) "
                          "to both the Corpus file walk and gold-file filtering, for the multilingual "
                          "baseline; 'default' (lanes2.CODE_EXTENSIONS) is byte-identical to prior runs")
    ap.add_argument("--global-order", action="store_true",
                     help="lanes3 experiment: assemble the below-lex_picks segment (guarantee "
                          "candidates + pool fill + anchor-tail/testbridge/docsbridge-tail) by a "
                          "single comparable score instead of phase-assembly order (lanes3."
                          "select_files use_global_order). Default off -- byte-identical to lanes2 "
                          "when off.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    extensions = L.EXTENDED_EXTENSIONS if args.extensions == "extended" else L.CODE_EXTENSIONS

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                pass

    parquet_cache = Path(args.parquet_cache) if args.parquet_cache else None
    instances = load_instances(args.parquet_url, parquet_cache)
    if args.instances_file:
        wanted = {
            ln.strip() for ln in Path(args.instances_file).read_text().splitlines() if ln.strip()
        }
        instances = [i for i in instances if i["instance_id"] in wanted]
    elif args.sample:
        instances = instances[:: args.sample]
    if args.limit:
        instances = instances[: args.limit]
    todo = [i for i in instances if i["instance_id"] not in done]
    print(f"{len(instances)} instances, {len(done)} done, {len(todo)} to run "
          f"(history={args.history} comments={args.comments} anchors={args.anchors} "
          f"testbridge={args.testbridge} docsbridge={args.docsbridge} extensions={args.extensions} "
          f"neighborhood={args.neighborhood} global_order={args.global_order})",
          flush=True)

    with out_path.open("a") as fh:
        for k, inst in enumerate(todo, 1):
            try:
                repo = repo_clone(inst["repo"])
                checkout(repo, inst["base_commit"])
                res = run_instance(inst, repo, args.history, args.comments, args.anchors,
                                    args.testbridge, args.docsbridge, extensions, args.neighborhood,
                                    args.global_order)
            except Exception as exc:
                res = {"instance_id": inst["instance_id"], "repo": inst["repo"],
                       "error": str(exc)[:300]}
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            r = res.get("recall")
            print(f"[{k}/{len(todo)}] {inst['instance_id']:44} "
                  f"recall={r if r is not None else 'ERR'} "
                  f"tok={res.get('tokens_packed', '-')} "
                  f"mine_ms={res.get('mine_ms', '-')}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
