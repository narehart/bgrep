#!/usr/bin/env python3
"""Reference shim used only to validate harness.py itself.

Implements the bgrep retrieval CONTRACT (see harness.py's docstring) on top
of the validated Python pipeline, wired up exactly the way
lab/swebench_driver2.py's run_instance() wires it for the frozen config that
produced results_swebench/abl_bridges_v7.jsonl:

    history=True, comments=False, anchors=True, testbridge=True,
    docsbridge=True, keywords=[]

i.e. Corpus(history_msgs=..., use_comments=False, build_docs=True),
query_terms(query, []), extract_symbol_anchors(query, corpus),
select_files(..., use_ppr=True, cochange=..., anchors=..., use_testbridge=True,
use_docsbridge=True), pack_regions(..., budget_tokens=8192, count_tokens).

REPO_PATH is assumed to already be checked out to the instance's base_commit
(the harness does that checkout before invoking this shim, exactly as
swebench_driver2.py's main() loop does before calling run_instance()) --
mine_history() walks `git log` from HEAD, so it must run strictly after that
checkout to avoid leaking future history.

Since a byte-for-byte match against the pipeline that PRODUCED the stored
expectations is the whole point of this shim, it is deliberately NOT a
generic reimplementation: it imports lab/lanes2.py and lab/history.py
directly rather than re-deriving their logic.

Usage: shim_reference.py QUERY REPO_PATH
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent / "lab"
sys.path.insert(0, str(LAB_DIR))

import lanes2 as L  # noqa: E402
from history import mine_history  # noqa: E402

import tiktoken  # noqa: E402

_ENCODER = tiktoken.get_encoding("cl100k_base")
_BUDGET_TOKENS = 8192


def count_tokens(text: str) -> int:
    """Same encoding/call as archex.reporting.count_tokens, inlined so this
    shim doesn't need the archex package installed -- only tiktoken, which
    bgrep already depends on."""
    return len(_ENCODER.encode(text, disallowed_special=()))


def _list_current_files(repo_path: Path) -> set[str]:
    """Verbatim mirror of swebench_driver2.py's _list_current_files: a cheap
    pre-filter so mine_history() can drop history entries for files that no
    longer exist at this checkout."""
    files: set[str] = set()
    for p in repo_path.rglob("*"):
        if not p.is_file() or p.suffix not in L.CODE_EXTENSIONS:
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


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: shim_reference.py QUERY REPO_PATH", file=sys.stderr)
        sys.exit(2)
    query, repo_path_s = sys.argv[1], sys.argv[2]
    repo_path = Path(repo_path_s)

    current_files = _list_current_files(repo_path)
    history_msgs, cochange, _meta = mine_history(repo_path, current_files=current_files)

    corpus = L.Corpus(repo_path, history_msgs=history_msgs, use_comments=False, build_docs=True)
    terms = L.query_terms(query, [])
    anchors = L.extract_symbol_anchors(query, corpus)
    files, scores = L.select_files(
        corpus, terms, use_ppr=True, cochange=cochange, anchors=anchors,
        use_testbridge=True, use_docsbridge=True,
    )
    spans, bundle = L.pack_regions(corpus, files, terms, scores, _BUDGET_TOKENS, count_tokens)
    packed_files = [f for f in files if f in spans]

    print(json.dumps({"files": packed_files}))


if __name__ == "__main__":
    main()
