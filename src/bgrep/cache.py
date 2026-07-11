"""On-disk index cache for bgrep, stored under ``<repo>/.bgrep/``.

Corpus construction (a full file walk + read + tokenize pass over every
candidate file in the repo) is the expensive part of a bgrep run; this module
caches that work (plus the import graph and, optionally, mined git history)
keyed on:

  - the repo's git HEAD sha (``git rev-parse HEAD``; ``"nogit"`` if the path
    isn't a git repo at all), and
  - a fast ``os.walk`` **stat-only** pass (no file reads) hashing
    ``(relpath, mtime_ns, size)`` for every candidate-extension file, so an
    uncommitted working-tree edit invalidates the cache even though HEAD
    hasn't moved.

The pickle also carries an explicit ``CACHE_VERSION``; bumping it invalidates
every existing cache file regardless of key match, for use whenever the
pickled shape (Corpus's attributes, the edges type, the history tuple shape)
changes in a way that would make an old pickle unsafe to unpickle into the
current code. ``bgrep.core.Corpus`` unconditionally excludes ``.bgrep/`` from
its own file walk, so this cache directory is never itself indexed.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
from pathlib import Path

from bgrep.core import CODE_EXTENSIONS, Corpus, _DOCS_EXTENSIONS, build_import_graph
from bgrep.history import mine_history

CACHE_VERSION = 1
CACHE_DIRNAME = ".bgrep"
_INDEX_FILENAME = "index.pkl"
_PRUNE_DIRS = {".git", CACHE_DIRNAME}

HistoryTuple = tuple  # (msgs: dict[str, str], cochange: dict[str, dict[str, int]], meta: dict[str, dict])


def _git_head_sha(repo_path: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    if r.returncode != 0:
        return "nogit"
    sha = r.stdout.strip()
    return sha if sha else "nogit"


def _fingerprint_files(repo_path: Path, with_docs: bool) -> str:
    """Stat-only os.walk pass (no file reads) over every candidate-extension
    file, hashing sorted (relpath, mtime_ns, size) triples. Deliberately
    cheaper and coarser than Corpus's own walk (no vendor-regex / oversize /
    long-line filtering) -- a false-negative "changed" hash just costs an
    extra reindex, which is always safe; it can never cause a stale hit."""
    exts = set(CODE_EXTENSIONS)
    if with_docs:
        exts |= set(_DOCS_EXTENSIONS)
    entries: list[tuple[str, int, int]] = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1] not in exts:
                continue
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = os.path.relpath(full, repo_path)
            entries.append((rel, st.st_mtime_ns, st.st_size))
    entries.sort()
    h = hashlib.sha1()
    for rel, mtime_ns, size in entries:
        h.update(rel.encode("utf-8", "surrogateescape"))
        h.update(b"\0")
        h.update(str(mtime_ns).encode())
        h.update(b"\0")
        h.update(str(size).encode())
        h.update(b"\n")
    return h.hexdigest()


def _cache_key(repo_path: Path, with_history: bool, with_docs: bool) -> str:
    sha = _git_head_sha(repo_path)
    fp = _fingerprint_files(repo_path, with_docs)
    return f"{sha}:{fp}:h{int(with_history)}:d{int(with_docs)}"


def _cache_path(repo_path: Path) -> Path:
    return repo_path / CACHE_DIRNAME / _INDEX_FILENAME


def _load(repo_path: Path, key: str):
    path = _cache_path(repo_path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError,
            ImportError, IndexError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != CACHE_VERSION or payload.get("key") != key:
        return None
    try:
        return payload["corpus"], payload["edges"], payload["history"]
    except KeyError:
        return None


def _save(repo_path: Path, key: str, corpus: Corpus, edges: dict, history) -> None:
    cache_dir = repo_path / CACHE_DIRNAME
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "key": key,
            "corpus": corpus,
            "edges": edges,
            "history": history,
        }
        tmp_path = _cache_path(repo_path).with_suffix(".pkl.tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(_cache_path(repo_path))
    except OSError:
        # Cache directory not writable (read-only checkout, permissions,
        # disk full, ...): degrade to "no cache" rather than fail the query.
        pass


def load_or_build(
    repo_path: Path,
    with_history: bool = True,
    with_docs: bool = True,
    use_cache: bool = True,
    force_reindex: bool = False,
) -> tuple[Corpus, dict[str, set[str]], HistoryTuple | None, bool]:
    """Load a cached (Corpus, import-graph edges, history) triple for
    `repo_path` if the on-disk cache matches the current HEAD sha + file
    fingerprint, else build it fresh and (unless use_cache is False) save it.

    Returns (corpus, edges, history_or_None, cache_hit). `history_or_None` is
    the 3-tuple mine_history() returns (msgs, cochange, meta), or None when
    with_history is False.

    `edges` (bgrep.core.build_import_graph(corpus)) is cached alongside the
    corpus even though bgrep.core.select_files() -- kept byte-for-byte
    equivalent to lab/lanes2.py -- always rebuilds its own edges internally
    when use_ppr/use_testbridge require them (that rebuild is pure in-memory
    regex work over corpus.text, not file I/O, so it's cheap once the corpus
    itself is cached); the cached edges are exposed here for callers that
    want the graph without re-deriving it.
    """
    key = _cache_key(repo_path, with_history, with_docs)

    if use_cache and not force_reindex:
        hit = _load(repo_path, key)
        if hit is not None:
            corpus, edges, history = hit
            return corpus, edges, history, True

    history: HistoryTuple | None = None
    if with_history:
        current_files = {
            str(p.relative_to(repo_path))
            for p in repo_path.rglob("*")
            if p.is_file() and p.suffix in CODE_EXTENSIONS
            and not (str(p.relative_to(repo_path)).startswith((".git/", f"{CACHE_DIRNAME}/")))
        }
        history = mine_history(repo_path, current_files=current_files)

    history_msgs = history[0] if history else None
    corpus = Corpus(repo_path, history_msgs=history_msgs, use_comments=False, build_docs=with_docs)
    edges = build_import_graph(corpus)

    if use_cache:
        _save(repo_path, key, corpus, edges, history)

    return corpus, edges, history, False
