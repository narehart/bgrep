"""End-to-end smoke test: builds a tiny synthetic repo, runs the real bgrep
CLI against it via subprocess (python -m bgrep.cli, exercising the installed
package exactly as a user would), and checks the three output modes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "routing.py").write_text(
        '"""Routing logic: maps incoming requests to handlers."""\n\n'
        "from pkg.handlers import handle_request\n\n\n"
        "class Router:\n"
        '    """Dispatches incoming requests to the correct registered handler."""\n\n'
        "    def __init__(self):\n"
        "        self.routes = {}\n\n"
        "    def add_route(self, path, handler):\n"
        "        self.routes[path] = handler\n\n"
        "    def dispatch(self, path, request):\n"
        "        handler = self.routes.get(path, handle_request)\n"
        "        return handler(request)\n"
    )
    (pkg / "handlers.py").write_text(
        '"""Request handlers invoked by the router."""\n\n'
        "from pkg.utils import normalize_path\n\n\n"
        "def handle_request(request):\n"
        "    path = normalize_path(request.path)\n"
        '    return {"status": 200, "path": path}\n'
    )
    (pkg / "utils.py").write_text(
        '"""Shared utility helpers."""\n\n\n'
        "def normalize_path(path):\n"
        '    return path.strip("/").lower()\n'
    )
    return repo


def _run_cli(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "bgrep.cli", *args],
        cwd=cwd, capture_output=True, text=True, timeout=60,
    )


def test_bundle_output(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_cli(["how does the router dispatch requests to handlers", str(repo), "--no-cache"])
    assert r.returncode == 0, r.stderr
    assert "### " in r.stdout
    assert "routing.py" in r.stdout
    # stats footer must go to stderr, never stdout
    assert "bgrep:" in r.stderr
    assert "bgrep:" not in r.stdout


def test_json_output(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_cli(["how does the router dispatch requests to handlers", str(repo), "--no-cache", "--json"])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["query"] == "how does the router dispatch requests to handlers"
    assert payload["files"], "expected at least one ranked file"
    assert all("path" in f and "score_rank" in f for f in payload["files"])
    assert isinstance(payload["regions"], dict)
    assert "routing.py" in payload["bundle"] or any(
        f["path"].endswith("routing.py") for f in payload["files"]
    )
    stats = payload["stats"]
    for key in ("files_indexed", "index_ms", "query_ms", "bundle_tokens", "cache"):
        assert key in stats
    assert stats["cache"] in ("hit", "miss")


def test_files_only_output(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_cli(["how does the router dispatch requests to handlers", str(repo), "--no-cache", "--files-only"])
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert lines, "expected at least one file path"
    for ln in lines:
        assert not ln.startswith("### ")
        assert "\t" not in ln


def test_no_results_exit_code(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r = _run_cli(["zzz_nonexistent_query_term_xyzzy_plugh", str(repo), "--no-cache"])
    assert r.returncode == 1, r.stderr


def test_usage_error_exit_code(tmp_path: Path) -> None:
    r = _run_cli(["some query", str(tmp_path / "does-not-exist")])
    assert r.returncode == 2


def test_cache_hit_on_second_run(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    r1 = _run_cli(["how does the router dispatch requests", str(repo), "--json"])
    assert r1.returncode == 0, r1.stderr
    assert json.loads(r1.stdout)["stats"]["cache"] == "miss"
    r2 = _run_cli(["how does the router dispatch requests", str(repo), "--json"])
    assert r2.returncode == 0, r2.stderr
    assert json.loads(r2.stdout)["stats"]["cache"] == "hit"
