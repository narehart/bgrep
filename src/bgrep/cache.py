"""On-disk index cache for bgrep, stored under ``<repo>/.bgrep/``.

Corpus construction (a full file walk + read + tokenize pass over every
candidate file in the repo) is the expensive part of a bgrep run; this module
caches that work (plus the import graph and, optionally, mined git history)
keyed on:

  - the repo's git HEAD sha (``git rev-parse HEAD``; ``"nogit"`` if the path
    isn't a git repo at all), and
  - the ``with_history``/``with_docs`` flags a query was run with.

Unlike keying on a hash of every file's stat (which forces a full rebuild on
ANY change, however small), the cache instead stores a manifest --
``{relpath: (mtime_ns, size)}`` for every candidate-extension file (code, plus
docs extensions when ``with_docs``) -- alongside the pickled Corpus. On load,
a fresh stat-only walk is diffed against that manifest to classify the
working tree since the cache was written:

  - unchanged: no work at all, the cached Corpus is returned as-is.
  - modified only (the common agent edit-loop case: existing files' content
    changed, nothing added or removed): the cached Corpus is PATCHED in
    place -- each modified file's old contributions are subtracted and its
    new contributions added back (Corpus.update_files / update_docs_files),
    and the import graph is repaired for just the files that changed
    (update_import_graph_for_files) -- instead of rebuilding from scratch.
  - HEAD moved, or any file was added/removed: a conservative FULL REBUILD.
    Module-index prefix resolution and import edges can shift globally when
    the file SET changes (a new file can shadow/absorb an existing module
    prefix, a removed file can orphan edges elsewhere), so add/remove is
    deliberately not special-cased the way pure content edits are.

A successful incremental patch is designed to be byte-for-byte
observationally identical to a fresh build over the same on-disk content
(same corpus.bm25/select_files output for any query) -- see the docstrings
on Corpus.update_files / update_import_graph_for_files in bgrep.core for the
subtract-then-add invariant this relies on. If a patch attempt turns out to
be shaped like an add/remove after all (e.g. an edit shrinks a file below
MAX_FILE_BYTES's oversized-exclusion threshold, or empties it out entirely),
Corpus.update_files/update_docs_files decline (returning False, having made
NO changes) and the caller falls back to a full rebuild.

The pickle also carries an explicit ``CACHE_VERSION``; bumping it invalidates
every existing cache file regardless of key match, for use whenever the
pickled shape (Corpus's attributes, the edges type, the history tuple shape,
the manifest schema) changes in a way that would make an old pickle unsafe to
unpickle into the current code. ``bgrep.core.Corpus`` unconditionally
excludes ``.bgrep/`` from its own file walk, so this cache directory is never
itself indexed.
"""

from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path

from bgrep.core import (
    CODE_EXTENSIONS,
    Corpus,
    _DOCS_EXTENSIONS,
    build_import_graph,
    update_import_graph_for_files,
)
from bgrep.history import mine_history

CACHE_VERSION = 2
CACHE_DIRNAME = ".bgrep"
_INDEX_FILENAME = "index.pkl"
_PRUNE_DIRS = {".git", CACHE_DIRNAME}

HistoryTuple = tuple  # (msgs: dict[str, str], cochange: dict[str, dict[str, int]], meta: dict[str, dict])
Manifest = dict  # {relpath: (mtime_ns: int, size: int)}


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


def _scan_manifest(repo_path: Path, with_docs: bool) -> Manifest:
    """Stat-only os.walk pass (no file reads) over every candidate-extension
    file, returning {relpath: (mtime_ns, size)}. Deliberately cheaper and
    coarser than Corpus's own walk (no vendor-regex / oversize / long-line
    filtering) -- a spuriously-"changed" entry just costs an extra reindex
    of that one file (or, for add/remove, a full rebuild), which is always
    safe; it can never cause a stale hit."""
    exts = set(CODE_EXTENSIONS)
    if with_docs:
        exts |= set(_DOCS_EXTENSIONS)
    manifest: Manifest = {}
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
            manifest[rel] = (st.st_mtime_ns, st.st_size)
    return manifest


def _classify_changes(
    repo_path: Path, with_docs: bool, manifest: Manifest
) -> tuple[str, Manifest, list[str]]:
    """Stat-walk the repo's current candidate files and diff against
    `manifest` (the snapshot the cached Corpus was built from). Returns
    (verdict, new_manifest, modified) where verdict is one of "unchanged"
    (nothing differs), "modified" (only existing files' mtime/size changed --
    incremental-update eligible), or "full" (any file added or removed --
    conservative full rebuild required)."""
    current = _scan_manifest(repo_path, with_docs)
    old_files = manifest.keys()
    new_files = current.keys()
    if old_files - new_files or new_files - old_files:
        return "full", current, []
    modified = [rel for rel in current if manifest[rel] != current[rel]]
    if not modified:
        return "unchanged", current, []
    return "modified", current, modified


def _cache_key(repo_path: Path, with_history: bool, with_docs: bool) -> str:
    sha = _git_head_sha(repo_path)
    return f"{sha}:h{int(with_history)}:d{int(with_docs)}"


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
        return payload["corpus"], payload["edges"], payload["history"], payload["manifest"]
    except KeyError:
        return None


def _save(
    repo_path: Path, key: str, corpus: Corpus, edges: dict, history, manifest: Manifest
) -> None:
    cache_dir = repo_path / CACHE_DIRNAME
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "key": key,
            "corpus": corpus,
            "edges": edges,
            "history": history,
            "manifest": manifest,
        }
        tmp_path = _cache_path(repo_path).with_suffix(".pkl.tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(_cache_path(repo_path))
    except OSError:
        # Cache directory not writable (read-only checkout, permissions,
        # disk full, ...): degrade to "no cache" rather than fail the query.
        pass


def _try_incremental_update(corpus: Corpus, edges: dict, modified: list[str]) -> bool:
    """Attempt to patch `corpus`/`edges` in place for `modified` (files whose
    content changed but whose relpath set didn't). All-or-nothing: if either
    field's update is declined (Corpus.update_files/update_docs_files return
    False, having made no changes to that field), returns False immediately.
    A partial patch (code side succeeded, docs side declined) can leave
    `corpus` mutated but inconsistent with a fresh build -- callers MUST
    discard `corpus`/`edges` entirely (never save or return them) whenever
    this returns False, falling back to a full rebuild instead."""
    code_rels = [r for r in modified if Path(r).suffix in CODE_EXTENSIONS]
    docs_rels = [r for r in modified if Path(r).suffix in _DOCS_EXTENSIONS]

    old_text = {rel: corpus.text[rel] for rel in code_rels}
    if code_rels and not corpus.update_files(code_rels):
        return False
    if docs_rels and not corpus.update_docs_files(docs_rels):
        return False
    if code_rels:
        update_import_graph_for_files(corpus, edges, old_text)
    return True


def _build_fresh(
    repo_path: Path, with_history: bool, with_docs: bool
) -> tuple[Corpus, dict[str, set[str]], HistoryTuple | None]:
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
    return corpus, edges, history


def load_or_build(
    repo_path: Path,
    with_history: bool = True,
    with_docs: bool = True,
    use_cache: bool = True,
    force_reindex: bool = False,
) -> tuple[Corpus, dict[str, set[str]], HistoryTuple | None, bool]:
    """Load a cached (Corpus, import-graph edges, history) triple for
    `repo_path`, else build it fresh and (unless use_cache is False) save it.

    Returns (corpus, edges, history_or_None, cache_hit). `history_or_None` is
    the 3-tuple mine_history() returns (msgs, cochange, meta), or None when
    with_history is False. `cache_hit` is True whenever a fresh Corpus build
    was avoided -- i.e. both the "unchanged" (no work) and "incremental"
    (file-level patch) cases from load_or_build_ex's finer-grained
    `update_kind` collapse to True here; only "full" is False. Thin wrapper
    around load_or_build_ex() for callers (bgrep.cli) that only need the
    coarse hit/miss distinction.

    `edges` (bgrep.core.build_import_graph(corpus)) is cached alongside the
    corpus even though bgrep.core.select_files() -- kept byte-for-byte
    equivalent to lab/lanes2.py -- always rebuilds its own edges internally
    when use_ppr/use_testbridge require them (that rebuild is pure in-memory
    regex work over corpus.text, not file I/O, so it's cheap once the corpus
    itself is cached); the cached edges are exposed here for callers that
    want the graph without re-deriving it.
    """
    corpus, edges, history, cache_hit, _update_kind = load_or_build_ex(
        repo_path,
        with_history=with_history,
        with_docs=with_docs,
        use_cache=use_cache,
        force_reindex=force_reindex,
    )
    return corpus, edges, history, cache_hit


def load_or_build_ex(
    repo_path: Path,
    with_history: bool = True,
    with_docs: bool = True,
    use_cache: bool = True,
    force_reindex: bool = False,
) -> tuple[Corpus, dict[str, set[str]], HistoryTuple | None, bool, str]:
    """Same as load_or_build, plus a 5th `update_kind` element -- one of
    "unchanged" (cache hit, no work), "incremental" (an existing cached
    Corpus was patched in place for modified files, no full rebuild) or
    "full" (a fresh Corpus was built from scratch) -- exposed for
    tests/tooling that need to assert which path was actually taken.
    load_or_build() itself collapses this to the boolean `cache_hit`.
    """
    key = _cache_key(repo_path, with_history, with_docs)

    if use_cache and not force_reindex:
        hit = _load(repo_path, key)
        if hit is not None:
            corpus, edges, history, manifest = hit
            verdict, new_manifest, modified = _classify_changes(repo_path, with_docs, manifest)
            if verdict == "unchanged":
                return corpus, edges, history, True, "unchanged"
            if verdict == "modified" and _try_incremental_update(corpus, edges, modified):
                if use_cache:
                    _save(repo_path, key, corpus, edges, history, new_manifest)
                return corpus, edges, history, True, "incremental"
            # verdict == "full", or the incremental patch was declined
            # (shaped like an add/remove after all): fall through to a full
            # rebuild. `corpus`/`edges` above may be partially mutated by a
            # declined patch attempt and must not be reused below.

    corpus, edges, history = _build_fresh(repo_path, with_history, with_docs)
    manifest = _scan_manifest(repo_path, with_docs)
    if use_cache:
        _save(repo_path, key, corpus, edges, history, manifest)
    return corpus, edges, history, False, "full"
