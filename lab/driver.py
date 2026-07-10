"""Run hypothesis lanes + archex raw baselines on the headtohead task set.

Reuses archex's own task loader, metric functions, and tiktoken counting so all
numbers are directly comparable with the published harness. Repos are cloned
once into a persistent cache (full repo, pinned tag/commit) and every lane runs
against the same corpus.

Usage (from the archex checkout, so `archex` is importable):
    uv run python <this file> --tasks httpx_pooling --lanes bm25,bm25_ppr,bm25_ppr_pack
    uv run python <this file>                      # all 19 tasks, all lanes
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_DIR))

from archex.benchmark.runner import load_selected_tasks  # noqa: E402
from archex.benchmark.strategies import (  # noqa: E402
    compute_precision,
    compute_recall,
    compute_required_file_metrics,
    count_file_tokens,
    extract_keywords,
    run_raw_files,
    run_raw_ripgrep,
)
from archex.reporting import count_tokens  # noqa: E402

import lanes as L  # noqa: E402

HEADTOHEAD_TASKS = [
    "celery_task_dispatch", "click_decorators", "django_middleware",
    "django_orm_queries", "express_error_handling", "express_middleware",
    "fastapi_dependency_injection", "fastapi_routing", "flask_blueprints",
    "gin_routing", "go_gin_middleware", "httpx_pooling", "mini_redis_async",
    "pydantic_validators", "pytest_fixtures", "react_hooks",
    "requests_sessions", "rust_tokio_runtime", "sqlalchemy_sessions",
]

REPO_CACHE = LAB_DIR / "repos"


def clone_repo(slug: str, commit: str) -> Path:
    dest = REPO_CACHE / f"{slug.replace('/', '__')}@{commit}"
    if (dest / ".git").exists() or dest.exists() and any(dest.iterdir()):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{slug}.git"
    shallow = subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", "--branch", commit, url, str(dest)],
        capture_output=True, text=True, timeout=900,
    )
    if shallow.returncode == 0:
        return dest
    for prefix in ("v",):  # tags like v18.2.0
        alt = subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", prefix + commit, url, str(dest)],
            capture_output=True, text=True, timeout=900,
        )
        if alt.returncode == 0:
            return dest
    full = subprocess.run(["git", "clone", "--quiet", url, str(dest)],
                          capture_output=True, text=True, timeout=1800)
    if full.returncode != 0:
        raise RuntimeError(f"clone failed for {slug}: {full.stderr.strip()}")
    co = subprocess.run(["git", "checkout", "--quiet", commit], cwd=dest,
                        capture_output=True, text=True, timeout=120)
    if co.returncode != 0:
        raise RuntimeError(f"checkout {commit} failed for {slug}: {co.stderr.strip()}")
    return dest


def _metrics(task, repo_path: Path, files: list[str], tokens: int, wall_ms: float,
             strategy: str, cold_ms: float, extra: dict | None = None) -> dict:
    fset = set(files)
    (rfr, mfr, mtr, all_present, present, missing) = compute_required_file_metrics(
        fset, task.expected_files)
    return {
        "task_id": task.task_id,
        "strategy": strategy,
        "tokens_total": tokens,
        "tool_calls": 1,
        "files_accessed": len(files),
        "recall": compute_recall(fset, task.expected_files),
        "precision": compute_precision(fset, task.expected_files),
        "required_file_recall": rfr,
        "missed_required_file_rate": mfr,
        "missed_required_task_rate": mtr,
        "all_required_files_present": all_present,
        "required_files_present": present,
        "required_files_missing": missing,
        "result_files": files,
        "wall_time_ms": wall_ms,
        "cold_start_ms": cold_ms,
        **(extra or {}),
    }


def run_lane(lane: str, task, repo_path: Path, corpus_cache: dict) -> dict:
    if lane == "raw_ripgrep":
        t0 = time.perf_counter()
        r = run_raw_ripgrep(task, repo_path)
        d = r.model_dump(mode="json")
        return {k: d[k] for k in (
            "task_id", "tokens_total", "tool_calls", "files_accessed", "recall",
            "precision", "required_file_recall", "missed_required_file_rate",
            "missed_required_task_rate", "all_required_files_present",
            "required_files_present", "required_files_missing", "result_files",
            "wall_time_ms")} | {"strategy": "raw_ripgrep", "cold_start_ms": 0.0}
    if lane == "raw_files":
        r = run_raw_files(task, repo_path)
        d = r.model_dump(mode="json")
        return {k: d[k] for k in (
            "task_id", "tokens_total", "tool_calls", "files_accessed", "recall",
            "precision", "required_file_recall", "missed_required_file_rate",
            "missed_required_task_rate", "all_required_files_present",
            "required_files_present", "required_files_missing", "result_files",
            "wall_time_ms")} | {"strategy": "raw_files", "cold_start_ms": 0.0}

    key = str(repo_path)
    if key not in corpus_cache:
        t0 = time.perf_counter()
        corpus_cache[key] = L.Corpus(repo_path)
        corpus_cache[key]._build_ms = (time.perf_counter() - t0) * 1000
    corpus = corpus_cache[key]
    cold_ms = getattr(corpus, "_build_ms", 0.0)

    keywords = extract_keywords(task.question, task.keywords)
    terms = L.query_terms(task.question, keywords)

    t0 = time.perf_counter()
    if lane == "bm25":
        files, scores = L.select_files(corpus, terms, use_ppr=False)
        tokens = count_file_tokens(repo_path, files)
        wall = (time.perf_counter() - t0) * 1000
        return _metrics(task, repo_path, files, tokens, wall, lane, cold_ms)
    if lane == "bm25_ppr":
        files, scores = L.select_files(corpus, terms, use_ppr=True)
        tokens = count_file_tokens(repo_path, files)
        wall = (time.perf_counter() - t0) * 1000
        return _metrics(task, repo_path, files, tokens, wall, lane, cold_ms)
    if lane == "bm25_ppr_pack":
        files, scores = L.select_files(corpus, terms, use_ppr=True)
        spans, bundle = L.pack_regions(
            corpus, files, terms, scores, task.token_budget, count_tokens)
        packed_files = [f for f in files if f in spans]
        tokens = count_tokens(bundle)
        wall = (time.perf_counter() - t0) * 1000
        return _metrics(task, repo_path, packed_files, tokens, wall, lane, cold_ms,
                        extra={"regions": {f: s for f, s in spans.items()}})
    raise ValueError(f"unknown lane {lane}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(HEADTOHEAD_TASKS))
    ap.add_argument("--lanes", default="raw_ripgrep,raw_files,bm25,bm25_ppr,bm25_ppr_pack")
    ap.add_argument("--tasks-dir", default="benchmarks/tasks")
    ap.add_argument("--out", default=str(LAB_DIR / "results"))
    args = ap.parse_args()

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    lanes_ = [x.strip() for x in args.lanes.split(",") if x.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = {t.task_id: t for t in load_selected_tasks(Path(args.tasks_dir))}
    corpus_cache: dict = {}

    for tid in task_ids:
        task = all_tasks[tid]
        print(f"=== {tid} ({task.repo}@{task.commit})", flush=True)
        repo_path = clone_repo(task.repo, task.commit)
        results = []
        for lane in lanes_:
            try:
                res = run_lane(lane, task, repo_path, corpus_cache)
            except Exception as exc:  # keep the run going, record the failure
                res = {"task_id": tid, "strategy": lane, "error": str(exc)}
            results.append(res)
            tok = res.get("tokens_total", -1)
            rec = res.get("recall", -1)
            miss = res.get("required_files_missing", res.get("error", ""))
            print(f"  {lane:16} tokens={tok:>9} recall={rec:>5} missing={miss}", flush=True)
        (out_dir / f"{tid}.json").write_text(json.dumps(
            {"task_id": tid, "repo": task.repo, "question": task.question,
             "results": results}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
