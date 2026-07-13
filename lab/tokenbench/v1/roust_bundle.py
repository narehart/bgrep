"""Condition B: fetch the roust bundle via the real CLI (subprocess), not a
reimplementation -- this measures the actual shipped tool.

    uv run roust --json --budget 8192 "<problem_statement>" <repo>

invoked with cwd=<bgrep repo root> so `uv run` resolves the project's own
environment regardless of what venv the tokenbench harness itself runs
under.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../bgrep
ROUST_TIMEOUT_S = 300


def get_roust_bundle(problem_statement: str, repo_path: Path, budget: int = 8192) -> dict:
    """Returns {"bundle": str, "files": [str, ...], "stats": {...}}.
    Empty bundle/files on roust's exit-1 ('no results') case."""
    cmd = ["uv", "run", "roust", "--json", "--budget", str(budget),
           problem_statement, str(repo_path)]
    r = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
        timeout=ROUST_TIMEOUT_S, encoding="utf-8", errors="replace",
    )
    if r.returncode not in (0, 1):
        raise RuntimeError(f"roust failed (exit {r.returncode}): {r.stderr.strip()[:500]}")
    if not r.stdout.strip():
        return {"bundle": "", "files": [], "stats": {}}
    payload = json.loads(r.stdout)
    return {
        "bundle": payload.get("bundle", ""),
        "files": [f["path"] for f in payload.get("files", [])],
        "stats": payload.get("stats", {}),
    }
