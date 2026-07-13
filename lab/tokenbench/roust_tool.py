"""Condition B/C: roust exposed as an agent-callable TOOL, not injected
context. This is the core v1->v2 fix -- see README.md '## Why v2' for the
injection-billing bug this replaces.

The agent calls roust(query) with a query of ITS OWN CHOOSING (not
necessarily the raw issue text -- the agent can refine, e.g. narrow to an
error string or symbol name after an initial miss). Each call shells out to
the real CLI:

    uv run roust --json --budget 8192 "<query>" <repo>

invoked with cwd=<bgrep repo root> so `uv run` resolves the project's own
environment regardless of what venv the tokenbench harness itself runs
under. Returns the packed bundle text as the tool result -- exactly what a
real integration would see -- and nothing else is added to the
conversation on the agent's behalf.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../bgrep
ROUST_TIMEOUT_S = 300
DEFAULT_BUDGET = 8192

ROUST_TOOL = {
    "name": "roust",
    "description": (
        "Search the repository with roust, a recall-first code-retrieval tool purpose-built "
        "for LLM agents. Pass a natural-language query (the issue text, an error string, a "
        "symbol/class/function name, or a refined query after a previous call) and it returns "
        "a token-budgeted bundle of the files/regions it judges most relevant to that query, "
        "ranked, with the actual source packed in. This is your primary search tool -- prefer "
        "it over guessing file paths. You can call it more than once with a different/narrower "
        "query if the first bundle doesn't converge on an answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "natural-language search query -- issue text, error string, or symbol name",
            }
        },
        "required": ["query"],
    },
}


def roust_search(query: str, repo_path: Path, budget: int = DEFAULT_BUDGET) -> tuple[str, dict]:
    """Returns (tool_result_text, meta). meta carries files/stats for
    per-row auditing (context_meta) without polluting the text the agent
    sees."""
    cmd = ["uv", "run", "roust", "--json", "--budget", str(budget), query, str(repo_path)]
    try:
        r = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=ROUST_TIMEOUT_S, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"roust: timed out after {ROUST_TIMEOUT_S}s -- try a narrower query.", {"error": "timeout"}
    if r.returncode not in (0, 1):
        err = r.stderr.strip()[:500]
        return f"roust: tool error (exit {r.returncode}): {err}", {"error": err}
    if not r.stdout.strip():
        return "roust: no results for that query -- try different terms.", {"files": [], "stats": {}}
    try:
        payload = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        return f"roust: could not parse tool output ({exc})", {"error": str(exc)}
    bundle = payload.get("bundle", "") or "roust: no results for that query -- try different terms."
    meta = {"files": [f["path"] for f in payload.get("files", [])], "stats": payload.get("stats", {})}
    return bundle, meta
