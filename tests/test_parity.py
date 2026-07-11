"""Parity test: bgrep.core (+ bgrep.history) must return IDENTICAL ranked
file lists to lab/lanes2.py (+ lab/history.py) -- the frozen v7 source of
truth this package was ported from -- on a real checked-out repo.

Only runs when BGREP_PARITY_REPO points at a repo checkout; skipped
otherwise, since it needs a real multi-hundred-file codebase (not a tiny
fixture) to meaningfully exercise BM25 + graph expansion + all the tail-tier
bridge channels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ENV = "BGREP_PARITY_REPO"
_LAB_DIR = Path(__file__).resolve().parents[1] / "lab"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_REPO_ENV),
    reason=f"set {_REPO_ENV}=/path/to/checked-out/repo to run parity tests",
)

# Frozen-v7-flavored queries: deliberately mundane "how does X work" issue
# text, mirroring how swebench_driver2.py feeds raw problem statements in.
QUERIES = [
    "how does the middleware chain work",
    "how does connection pooling and keep-alive work",
    "how are incoming requests routed to handlers",
    "how does error handling and retry logic work",
    "how does authentication and session management work",
]


def _load_lab_lanes2():
    if str(_LAB_DIR) not in sys.path:
        sys.path.insert(0, str(_LAB_DIR))
    import lanes2  # noqa: PLC0415 (lab/lanes2.py, not a package)

    return lanes2


def test_select_files_parity() -> None:
    repo_path = Path(os.environ[_REPO_ENV]).resolve()
    assert repo_path.is_dir(), f"{_REPO_ENV}={repo_path} is not a directory"

    lab_lanes2 = _load_lab_lanes2()

    import bgrep.core as core
    import bgrep.history as bgrep_history

    current_files = {
        str(p.relative_to(repo_path))
        for p in repo_path.rglob("*")
        if p.is_file() and p.suffix in core.CODE_EXTENSIONS
    }
    msgs, cochange, _meta = bgrep_history.mine_history(repo_path, current_files=current_files)

    lab_corpus = lab_lanes2.Corpus(repo_path, history_msgs=msgs, use_comments=False, build_docs=True)
    core_corpus = core.Corpus(repo_path, history_msgs=msgs, use_comments=False, build_docs=True)

    assert lab_corpus.files == core_corpus.files, "corpus file sets diverged between lanes2 and core"

    for query in QUERIES:
        lab_terms = lab_lanes2.query_terms(query, [])
        core_terms = core.query_terms(query, [])
        assert core_terms == lab_terms, f"query_terms diverged for: {query!r}"

        lab_anchors = lab_lanes2.extract_symbol_anchors(query, lab_corpus)
        core_anchors = core.extract_symbol_anchors(query, core_corpus)
        assert core_anchors == lab_anchors, f"extract_symbol_anchors diverged for: {query!r}"

        lab_files, _lab_scores = lab_lanes2.select_files(
            lab_corpus, lab_terms, use_ppr=True, cochange=cochange, anchors=lab_anchors,
            use_testbridge=True, use_docsbridge=True,
        )
        core_files, _core_scores, _explain = core.select_files(
            core_corpus, core_terms, use_ppr=True, cochange=cochange, anchors=core_anchors,
            use_testbridge=True, use_docsbridge=True,
        )
        assert core_files == lab_files, f"ranked file list diverged for query: {query!r}"
