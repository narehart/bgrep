#!/usr/bin/env python3
"""Language-agnostic parity harness for the bgrep retrieval pipeline.

Runs any command that implements the bgrep retrieval CONTRACT against the
stored benchmark tasks (SWE-bench Lite 300 + the archex comprehension/loc
task sets) and compares its ranked file list to the stored expected result
for that task. This is the acceptance gate for a from-scratch reimplementation
(e.g. a Rust port): the port passes iff it reproduces the validated Python
pipeline's file rankings on every SWE-bench Lite task.

CONTRACT under test
--------------------
The candidate is invoked via a shell-quoted TEMPLATE string containing the
literal placeholders ``{query}`` and ``{repo_path}``, e.g.::

    bgrep-rs --json --budget 8192 {query} {repo_path}

The template is tokenized with ``shlex.split`` *before* substitution, so the
query text (which may be arbitrarily long, multi-line, and contain quotes)
is always passed as a single argv element -- never re-interpreted by a shell.
The candidate must print JSON to stdout containing at least::

    {"files": ["relative/path/one.py", "relative/path/two.py", ...]}

ranked best-first, repo-relative. Entries may also be objects with a "path"
key: ``{"files": [{"path": "a.py", ...}, ...]}``. Any other stdout content
(logging, banners) is tolerated as long as a JSON object with a "files" key
can be located somewhere in stdout (see ``_extract_json``).

Suites
------
lite    (binding gate) 300 SWE-bench Lite instances. Expected rankings come
        from results_swebench/abl_bridges_v7.jsonl (the frozen lanes2 v7
        config: history+anchors+testbridge+docsbridge on, keywords=[]).
        Queries (problem_statement) + base_commit are joined in from the
        SWE-bench Lite parquet (pandas) or a pre-exported --lite-queries-json.
        Each repo is `git checkout -f -q <base_commit>` before its instance
        runs; this harness is single-threaded/sequential, which is by itself
        sufficient to satisfy "no parallel checkouts of the same repo". As an
        extra safety net, refuses to run this suite at all if another
        swebench_driver process is already checking out the same shared
        clones (`pgrep -f swebench_driver`).
archex  (informational only) 19 comprehension + 21 holdout loc tasks under
        results/*.json and results_holdout/*.json. Expected ranking is the
        bm25_ppr_pack strategy's result_files for that task. These were
        produced by the OLDER lanes.py config, not the frozen v7 lanes2
        config the Lite suite gates on, so this suite is reported but never
        used to decide GATE PASS/FAIL.

Usage
-----
    python harness.py --cmd "TEMPLATE with {query} and {repo_path}" \\
        [--suite lite|archex|all] [--limit N] [--report PATH.json] \\
        [--gate exact|set|top10]

Exit status: 0 iff the gate verdict is PASS, 1 otherwise (including a
refused/skipped lite suite).
"""

from __future__ import annotations

import argparse
import difflib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults. These point at the scratchpad lab artifacts produced by the
# retrieval-lab work; override any of them via CLI flags to run this harness
# against a different data location (e.g. a checked-out copy of the lab
# outputs, or a CI cache).
# ---------------------------------------------------------------------------

_SCRATCH = Path(
    "/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/"
    "3ab12e71-fab2-4a81-bb2b-84700d211ef2/scratchpad/bgrep_lab"
)

DEFAULT_LITE_EXPECTED = _SCRATCH / "results_swebench" / "abl_bridges_v7.jsonl"
DEFAULT_LITE_PARQUET = _SCRATCH / "swebench_lite.parquet"
DEFAULT_LITE_REPOS = _SCRATCH / "swebench_repos"
DEFAULT_ARCHEX_RESULTS = _SCRATCH / "results"
DEFAULT_ARCHEX_HOLDOUT = _SCRATCH / "results_holdout"
DEFAULT_ARCHEX_REPOS = _SCRATCH / "repos"

DEFAULT_TIMEOUT_S = 300
DIFF_LINE_CAP = 30


# ---------------------------------------------------------------------------
# Task / result data model
# ---------------------------------------------------------------------------


@dataclass
class Task:
    suite: str
    task_id: str
    query: str
    repo_path: Path
    expected: list[str]
    base_commit: str | None = None  # lite only; None => no checkout needed


@dataclass
class TaskResult:
    task_id: str
    suite: str
    expected: list[str]
    actual: list[str] | None
    error: str | None
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.error is None and self.actual is not None

    @property
    def exact_match(self) -> bool:
        return self.ok and self.actual == self.expected

    @property
    def set_match(self) -> bool:
        return self.ok and set(self.actual) == set(self.expected)

    @property
    def top10_match(self) -> bool:
        # Order-insensitive within the top 10: what matters for a retrieval
        # gate is which files made the cut, not their relative order inside
        # the head -- exact/full-order equality is already covered by
        # exact_match. Flagged assumption (spec didn't define "match").
        return self.ok and set(self.actual[:10]) == set(self.expected[:10])


@dataclass
class SuiteResult:
    name: str
    binding: bool
    skipped: bool = False
    skip_reason: str = ""
    results: list[TaskResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_exact(self) -> int:
        return sum(r.exact_match for r in self.results)

    @property
    def n_set(self) -> int:
        return sum(r.set_match for r in self.results)

    @property
    def n_top10(self) -> int:
        return sum(r.top10_match for r in self.results)

    @property
    def n_errors(self) -> int:
        return sum(not r.ok for r in self.results)


# ---------------------------------------------------------------------------
# Candidate invocation
# ---------------------------------------------------------------------------


def build_argv(template: str, query: str, repo_path: str) -> list[str]:
    """Tokenize TEMPLATE with shell-quoting rules, then substitute the
    {query}/{repo_path} placeholders into whichever argv tokens contain them.
    Never invokes a shell, so arbitrarily long/quoted/multi-line query text
    is always passed as exactly one argv element per occurrence."""
    tokens = shlex.split(template)
    argv = []
    for tok in tokens:
        if "{query}" in tok or "{repo_path}" in tok:
            tok = tok.replace("{query}", query).replace("{repo_path}", repo_path)
        argv.append(tok)
    return argv


def _extract_json(stdout: str) -> dict:
    """Parse the candidate's stdout into the contract JSON object. Tries a
    whole-output parse first (the common case), then falls back to scanning
    for a balanced {...} object containing a "files" key, to tolerate
    candidates that also emit banners/log lines to stdout."""
    stdout = stdout.strip()
    if not stdout:
        raise ValueError("empty stdout")
    try:
        obj = json.loads(stdout)
        if isinstance(obj, dict) and "files" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    idx = 0
    candidates: list[dict] = []
    while True:
        brace = stdout.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(stdout, brace)
            if isinstance(obj, dict) and "files" in obj:
                candidates.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx = brace + 1
    if candidates:
        return candidates[-1]
    raise ValueError("no JSON object with a \"files\" key found in stdout")


def extract_files(obj: dict) -> list[str]:
    files = obj.get("files")
    if files is None:
        raise ValueError("JSON output has no \"files\" key")
    out: list[str] = []
    for item in files:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and "path" in item:
            out.append(item["path"])
        else:
            raise ValueError(f"unrecognized \"files\" entry: {item!r}")
    return out


def run_task(cmd_template: str, task: Task, timeout: float) -> TaskResult:
    argv = build_argv(cmd_template, task.query, str(task.repo_path))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return TaskResult(task.task_id, task.suite, task.expected, None,
                           f"timed out after {timeout}s", time.perf_counter() - t0)
    except OSError as exc:
        return TaskResult(task.task_id, task.suite, task.expected, None,
                           f"failed to spawn candidate: {exc}", time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return TaskResult(
            task.task_id, task.suite, task.expected, None,
            f"exit {proc.returncode}: stderr[:300]={proc.stderr[:300]!r}", elapsed,
        )
    try:
        obj = _extract_json(proc.stdout)
        actual = extract_files(obj)
    except (ValueError, TypeError) as exc:
        return TaskResult(task.task_id, task.suite, task.expected, None,
                           f"bad output: {exc}", elapsed)
    return TaskResult(task.task_id, task.suite, task.expected, actual, None, elapsed)


# ---------------------------------------------------------------------------
# Lite suite loading
# ---------------------------------------------------------------------------


def swebench_driver_guard() -> str | None:
    """Returns a non-None refusal reason if another swebench_driver process
    is running (it may be mid-checkout on the same shared swebench_repos/
    clones this suite also checks out)."""
    try:
        proc = subprocess.run(["pgrep", "-f", "swebench_driver"],
                               capture_output=True, text=True)
    except OSError:
        return None  # no pgrep on this platform; nothing to guard with
    pids = proc.stdout.strip()
    if proc.returncode == 0 and pids:
        return f"swebench_driver process(es) running (pids: {pids.replace(chr(10), ',')})"
    return None


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_lite_queries(parquet_path: Path, queries_json: Path | None) -> dict[str, dict]:
    """Returns {instance_id: {"repo", "base_commit", "problem_statement"}}.
    Uses a pre-exported JSON if given (list of objects with those keys plus
    instance_id), otherwise reads the parquet directly via pandas."""
    if queries_json is not None:
        rows = json.loads(queries_json.read_text())
        return {r["instance_id"]: r for r in rows}
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise SystemExit(
            "pandas (with a parquet engine: pyarrow or fastparquet) is not "
            "usable in this interpreter, and no --lite-queries-json was "
            "given. Either run this harness with a python that has pandas "
            "+ pyarrow (e.g. the archex venv), or pre-export queries via:\n"
            "  python -c \"import pandas as pd, json; df = pd.read_parquet('"
            f"{parquet_path}'); json.dump([{{'instance_id': r.instance_id, "
            "'repo': r.repo, 'base_commit': r.base_commit, "
            "'problem_statement': r.problem_statement} "
            "for r in df.itertuples()], open('queries.json', 'w'))\"\n"
            "then pass --lite-queries-json queries.json\n"
            f"(underlying error: {exc})"
        ) from exc
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        out[row["instance_id"]] = {
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
        }
    return out


def checkout(repo_path: Path, sha: str) -> None:
    r = subprocess.run(["git", "checkout", "-f", "-q", sha], cwd=repo_path,
                        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"checkout {sha} in {repo_path} failed: {r.stderr.strip()[:300]}")
    subprocess.run(["git", "clean", "-fdq"], cwd=repo_path, capture_output=True,
                    text=True, timeout=300)


def load_lite_tasks(args) -> list[Task]:
    expected_rows = load_jsonl(args.lite_expected)
    queries = load_lite_queries(args.lite_parquet, args.lite_queries_json)
    tasks: list[Task] = []
    skipped_no_query = 0
    for row in expected_rows:
        if "returned_files" not in row:
            continue  # instance errored when the expectations file was generated
        q = queries.get(row["instance_id"])
        if q is None:
            skipped_no_query += 1
            continue
        repo_path = args.lite_repos / q["repo"].replace("/", "__")
        tasks.append(Task(
            suite="lite",
            task_id=row["instance_id"],
            query=q["problem_statement"],
            repo_path=repo_path,
            expected=row["returned_files"],
            base_commit=q["base_commit"],
        ))
    if skipped_no_query:
        print(f"warning: {skipped_no_query} lite instance(s) had no matching "
              f"query row (instance_id not found in parquet/queries-json), skipped",
              file=sys.stderr)
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


# ---------------------------------------------------------------------------
# archex suite loading (informational)
# ---------------------------------------------------------------------------


def find_archex_repo(repos_dir: Path, repo_slug: str) -> Path | None:
    owner_name = repo_slug.replace("/", "__")
    matches = sorted(repos_dir.glob(f"{owner_name}@*"))
    return matches[0] if matches else None


def load_archex_tasks(args) -> list[Task]:
    tasks: list[Task] = []
    dirs = [args.archex_results, args.archex_holdout]
    for d in dirs:
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.json")):
            data = json.loads(fp.read_text())
            expected = None
            for r in data.get("results", []):
                if r.get("strategy") == "bm25_ppr_pack":
                    expected = r.get("result_files")
                    break
            if expected is None:
                print(f"warning: {fp} has no bm25_ppr_pack strategy result, skipped",
                      file=sys.stderr)
                continue
            repo_path = find_archex_repo(args.archex_repos, data["repo"])
            if repo_path is None:
                print(f"warning: no repo checkout found for {data['repo']} "
                      f"(task {data['task_id']}), skipped", file=sys.stderr)
                continue
            tasks.append(Task(
                suite="archex",
                task_id=data["task_id"],
                query=data["question"],
                repo_path=repo_path,
                expected=expected,
                base_commit=None,
            ))
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


# ---------------------------------------------------------------------------
# Suite runners
# ---------------------------------------------------------------------------


def run_lite_suite(args) -> SuiteResult:
    reason = swebench_driver_guard()
    if reason:
        print(f"REFUSED to run lite suite: {reason}", file=sys.stderr)
        return SuiteResult(name="lite", binding=True, skipped=True, skip_reason=reason)
    tasks = load_lite_tasks(args)
    results = []
    for i, task in enumerate(tasks, 1):
        try:
            checkout(task.repo_path, task.base_commit)
        except (RuntimeError, OSError) as exc:
            results.append(TaskResult(task.task_id, "lite", task.expected, None,
                                       f"checkout failed: {exc}", 0.0))
            continue
        res = run_task(args.cmd, task, args.timeout)
        results.append(res)
        _progress(args, "lite", i, len(tasks), res)
    return SuiteResult(name="lite", binding=True, results=results)


def run_archex_suite(args) -> SuiteResult:
    tasks = load_archex_tasks(args)
    results = []
    for i, task in enumerate(tasks, 1):
        res = run_task(args.cmd, task, args.timeout)
        results.append(res)
        _progress(args, "archex", i, len(tasks), res)
    return SuiteResult(name="archex", binding=False, results=results)


def _progress(args, suite: str, i: int, n: int, res: TaskResult) -> None:
    if args.quiet:
        return
    status = "EXACT" if res.exact_match else ("SET" if res.set_match else ("ERR" if not res.ok else "MISS"))
    print(f"[{suite} {i}/{n}] {res.task_id:50} {status:6} ({res.elapsed_s:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def diff_for(res: TaskResult) -> list[str]:
    expected = res.expected
    actual = res.actual if res.actual is not None else [f"<ERROR: {res.error}>"]
    lines = list(difflib.unified_diff(
        expected, actual, fromfile="expected", tofile="actual", lineterm="",
    ))
    return lines[:DIFF_LINE_CAP]


def print_suite_report(suite: SuiteResult) -> None:
    tag = "BINDING GATE" if suite.binding else "informational"
    print(f"\n=== suite: {suite.name} ({tag}) ===")
    if suite.skipped:
        print(f"  SKIPPED: {suite.skip_reason}")
        return
    print(f"  tasks run:        {suite.n}")
    print(f"  errors:           {suite.n_errors}")
    print(f"  exact-match:      {suite.n_exact}/{suite.n}")
    print(f"  set-match:        {suite.n_set}/{suite.n}")
    print(f"  first-10-match:   {suite.n_top10}/{suite.n}")
    mismatches = [r for r in suite.results if not r.exact_match]
    if mismatches:
        print(f"  mismatching tasks ({len(mismatches)}):")
        for r in mismatches:
            print(f"    - {r.task_id}" + (f"  ERROR: {r.error}" if r.error else ""))
            if r.ok:
                for line in diff_for(r):
                    print(f"        {line}")


def suite_to_dict(suite: SuiteResult) -> dict:
    return {
        "name": suite.name,
        "binding": suite.binding,
        "skipped": suite.skipped,
        "skip_reason": suite.skip_reason,
        "tasks_run": suite.n,
        "errors": suite.n_errors,
        "exact_match": suite.n_exact,
        "set_match": suite.n_set,
        "top10_match": suite.n_top10,
        "tasks": [
            {
                "task_id": r.task_id,
                "ok": r.ok,
                "error": r.error,
                "exact_match": r.exact_match,
                "set_match": r.set_match,
                "top10_match": r.top10_match,
                "expected": r.expected,
                "actual": r.actual,
                "diff": diff_for(r) if not r.exact_match else [],
                "elapsed_s": r.elapsed_s,
            }
            for r in suite.results
        ],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmd", required=True,
                     help="candidate command template, e.g. "
                          "'bgrep-rs --json --budget 8192 {query} {repo_path}'")
    ap.add_argument("--suite", choices=["lite", "archex", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="cap tasks per suite")
    ap.add_argument("--report", type=Path, default=None, help="write full JSON report to PATH")
    ap.add_argument("--gate", choices=["exact", "set", "top10"], default="exact",
                     help="which match strictness decides GATE PASS/FAIL on the lite suite")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                     help="per-task subprocess timeout in seconds")
    ap.add_argument("--quiet", action="store_true", help="suppress per-task progress lines")

    ap.add_argument("--lite-expected", type=Path, default=DEFAULT_LITE_EXPECTED)
    ap.add_argument("--lite-parquet", type=Path, default=DEFAULT_LITE_PARQUET)
    ap.add_argument("--lite-repos", type=Path, default=DEFAULT_LITE_REPOS)
    ap.add_argument("--lite-queries-json", type=Path, default=None,
                     help="pre-exported instance_id -> {repo,base_commit,problem_statement} "
                          "JSON, used instead of reading the parquet with pandas")

    ap.add_argument("--archex-results", type=Path, default=DEFAULT_ARCHEX_RESULTS)
    ap.add_argument("--archex-holdout", type=Path, default=DEFAULT_ARCHEX_HOLDOUT)
    ap.add_argument("--archex-repos", type=Path, default=DEFAULT_ARCHEX_REPOS)

    args = ap.parse_args()

    suites: list[SuiteResult] = []
    if args.suite in ("lite", "all"):
        suites.append(run_lite_suite(args))
    if args.suite in ("archex", "all"):
        suites.append(run_archex_suite(args))

    for s in suites:
        print_suite_report(s)

    lite = next((s for s in suites if s.name == "lite"), None)
    if lite is None:
        verdict = "GATE: N/A (lite suite not run)"
        passed = False
    elif lite.skipped:
        verdict = f"GATE: SKIPPED ({lite.skip_reason})"
        passed = False
    elif lite.n == 0:
        verdict = "GATE: N/A (0 lite tasks run)"
        passed = False
    else:
        metric = {"exact": lite.n_exact, "set": lite.n_set, "top10": lite.n_top10}[args.gate]
        passed = metric == lite.n
        verdict = f"GATE: {'PASS' if passed else 'FAIL'} " \
                  f"(lite {args.gate}-match {metric}/{lite.n})"

    print(f"\n{verdict}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "cmd": args.cmd,
            "gate_mode": args.gate,
            "verdict": verdict,
            "passed": passed,
            "suites": [suite_to_dict(s) for s in suites],
        }, indent=2))
        print(f"full report written to {args.report}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
