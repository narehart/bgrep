"""Condition D: RAG exposed as an agent-callable TOOL (rag_search), mirroring
roust_tool.py's shape -- same "the agent decides when/what to query"
principle the v1->v2 rebuild applies to all retrieval.

`k` is raised from v1's 12 to `TOP_K` (24) so the returned bundle is ~8000
tiktoken tokens, matching roust's --budget 8192 (v1's top-12 x 40-line
chunks landed at only ~4k tokens -- see v1/README.md 'RAG bundle is smaller
than roust's' -- which under-fed this arm relative to roust's budget and
made the comparison unfair). Calibrated against two real (repo, query)
pairs at harness-build time:

    django/django,   k=24 -> 8539 tiktoken tokens
    astropy/astropy, k=24 -> 8808 tiktoken tokens

i.e. k=24 lands consistently within ~5% of roust's 8192-token budget on
real repos/queries. `summarize.py` reports the actual mean tokens returned
per rag_search call across the real run (not just this calibration) so the
budget-match claim is auditable against real data, not just this pre-check.
"""

from __future__ import annotations

from pathlib import Path

from rag_index import format_rag_bundle, retrieve

TOP_K = 24

RAG_SEARCH_TOOL = {
    "name": "rag_search",
    "description": (
        "Search the repository with semantic (embedding) similarity. Pass a natural-language "
        "query (the issue text, an error string, a symbol/class/function name, or a refined "
        "query after a previous call) and it returns the top matching code chunks, ranked by "
        "cosine similarity, with their file path/line range and source text. Semantic "
        "similarity is not the same as 'this file needs to change' -- verify hits with your "
        "other tools before finalizing an answer. You can call it more than once with a "
        "different/narrower query if the first results don't converge on an answer."
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


def rag_search(query: str, repo_path: Path, repo_slug: str, commit: str, k: int = TOP_K) -> tuple[str, dict]:
    """Returns (tool_result_text, meta) -- same shape as roust_tool.roust_search
    so agent.py can treat both retrieval tools uniformly."""
    results = retrieve(repo_path, repo_slug, commit, query, k=k)
    if not results:
        return "rag_search: no results for that query -- try different terms.", {"files": []}
    bundle = format_rag_bundle(results)
    meta = {"files": sorted({r["path"] for r in results})}
    return bundle, meta
