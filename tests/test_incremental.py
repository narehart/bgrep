"""Incremental index-update tests for roust.cache.

Core acceptance bar (see roust.cache's module docstring): an
incrementally-patched Corpus must be OBSERVATIONALLY IDENTICAL to a
fresh-built Corpus over the same on-disk content -- same select_files()
output (files AND scores) for any query. These tests build a synthetic repo,
apply a scripted sequence of content-only edits, and after EACH edit compare
the cache's load_or_build_ex() result against an independently fresh-built
Corpus, while also asserting which update path (roust.cache's "unchanged" /
"incremental" / "full" `update_kind`) was actually taken.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from roust import cache as cache_mod
from roust.core import (
    Corpus,
    build_import_graph,
    extract_symbol_anchors,
    query_terms,
    select_files,
    update_import_graph_for_files,
)

QUERIES = [
    "how does the router dispatch requests to handlers",
    "how are widgets validated before creation",
    "how is application configuration loaded",
]

_INIT = '"""Widget package: models, validation, handlers, routing, and config."""\n'

_MODELS = (
    '"""Domain models for widgets."""\n\n\n'
    "class Widget:\n"
    '    """A widget resource with a name and a price."""\n\n'
    "    def __init__(self, name, price):\n"
    "        self.name = name\n"
    "        self.price = price\n\n"
    "    def describe(self):\n"
    '        return f"{self.name}: {self.price}"\n'
)

_VALIDATORS_V1 = (
    '"""Validation helpers for widget models."""\n\n'
    "from pkg.models import Widget\n\n\n"
    "def validate_widget(widget):\n"
    '    """Validate that a widget has a positive price."""\n'
    "    if widget.price <= 0:\n"
    '        raise ValueError("widget price must be positive")\n'
    "    return True\n"
)

_VALIDATORS_V2 = (
    '"""Validation helpers for widget models."""\n\n'
    "from pkg.models import Widget\n\n\n"
    "def validate_widget(widget):\n"
    '    """Validate that a widget has a positive price and a name."""\n'
    "    if widget.price <= 0:\n"
    '        raise ValueError("widget price must be positive")\n'
    "    if not widget.name:\n"
    '        raise ValueError("widget name must not be empty")\n'
    "    return True\n"
)

_UTILS_V1 = (
    '"""Shared string utility helpers."""\n\n\n'
    "def normalize_name(name):\n"
    "    return name.strip().lower()\n"
)

_UTILS_V2 = (
    '"""Shared string utility helpers."""\n\n\n'
    "def normalize_name(name):\n"
    "    return name.strip().lower()\n\n\n"
    "def slugify(name):\n"
    '    """Convert a name into a url-safe slug."""\n'
    '    return normalize_name(name).replace(" ", "-")\n'
)

_CONFIG = (
    '"""Static configuration loading."""\n\n'
    "DEFAULT_TIMEOUT = 30\n\n\n"
    "def load_config():\n"
    '    return {"timeout": DEFAULT_TIMEOUT}\n'
)

_HANDLERS_V1 = (
    '"""Request handlers for widget operations."""\n\n'
    "from pkg.models import Widget\n"
    "from pkg.validators import validate_widget\n\n\n"
    "def handle_create_widget(request):\n"
    '    widget = Widget(request["name"], request["price"])\n'
    "    validate_widget(widget)\n"
    "    return widget.describe()\n"
)

_HANDLERS_V2 = (
    '"""Request handlers for widget operations."""\n\n'
    "from pkg.models import Widget\n"
    "from pkg.validators import validate_widget\n"
    "from pkg.utils import normalize_name\n\n\n"
    "def handle_create_widget(request):\n"
    '    widget = Widget(normalize_name(request["name"]), request["price"])\n'
    "    validate_widget(widget)\n"
    "    return widget.describe()\n"
)

_ROUTER_V1 = (
    '"""Routing logic: maps incoming requests to handlers."""\n\n'
    "from pkg.handlers import handle_create_widget\n\n\n"
    "class Router:\n"
    '    """Dispatches incoming requests to the correct registered handler."""\n\n'
    "    def __init__(self):\n"
    '        self.routes = {"create_widget": handle_create_widget}\n\n'
    "    def dispatch(self, path, request):\n"
    "        handler = self.routes.get(path)\n"
    "        return handler(request)\n"
)

_ROUTER_V2 = (
    '"""Routing logic: maps incoming requests to handlers."""\n\n'
    "from pkg.handlers import handle_create_widget\n\n\n"
    "class Router:\n"
    '    """Dispatches incoming requests to the correct registered handler."""\n\n'
    "    def __init__(self):\n"
    '        self.routes = {"create_widget": handle_create_widget}\n\n'
    "    def add_route(self, path, handler):\n"
    "        self.routes[path] = handler\n\n"
    "    def dispatch(self, path, request):\n"
    "        handler = self.routes.get(path)\n"
    "        if handler is None:\n"
    '            raise KeyError(f"no handler registered for {path}")\n'
    "        return handler(request)\n"
)

_SERVICE_V1 = (
    '"""Top-level service wiring router and config together."""\n\n'
    "from pkg.router import Router\n"
    "from pkg.config import load_config\n\n\n"
    "class Service:\n"
    '    """Application service entry point."""\n\n'
    "    def __init__(self):\n"
    "        self.router = Router()\n"
    "        self.config = load_config()\n\n"
    "    def handle(self, path, request):\n"
    "        return self.router.dispatch(path, request)\n"
)

_SERVICE_V2 = (
    '"""Top-level service wiring router and config together."""\n\n'
    "from pkg.router import Router\n"
    "from pkg.config import load_config\n\n\n"
    "class Service:\n"
    '    """Application service entry point."""\n\n'
    "    def __init__(self):\n"
    "        self.router = Router()\n"
    "        self.config = load_config()\n\n"
    "    def handle(self, path, request):\n"
    "        return self.router.dispatch(path, request)\n\n"
    "    def timeout(self):\n"
    '        return self.config["timeout"]\n'
)

_FILES_V1 = {
    "pkg/__init__.py": _INIT,
    "pkg/models.py": _MODELS,
    "pkg/validators.py": _VALIDATORS_V1,
    "pkg/utils.py": _UTILS_V1,
    "pkg/config.py": _CONFIG,
    "pkg/handlers.py": _HANDLERS_V1,
    "pkg/router.py": _ROUTER_V1,
    "pkg/service.py": _SERVICE_V1,
}


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    for rel, text in _FILES_V1.items():
        (repo / rel).write_text(text)
    return repo


def _write(repo: Path, rel: str, text: str) -> None:
    # Guarantee an observable mtime change even on coarse-resolution
    # filesystems: write, then explicitly bump mtime forward.
    p = repo / rel
    p.write_text(text)
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))


def _touch(repo: Path, rel: str) -> None:
    p = repo / rel
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))


def _select(corpus: Corpus, query: str) -> tuple[list[str], dict[str, float]]:
    terms = query_terms(query, [])
    anchors = extract_symbol_anchors(query, corpus)
    files, scores, _explain = select_files(
        corpus, terms, use_ppr=True, cochange=None, anchors=anchors,
        use_testbridge=True, use_docsbridge=False,
    )
    return files, scores


def _assert_matches_fresh_build(repo: Path, cached_corpus: Corpus) -> None:
    """Assert `cached_corpus` (however it was produced -- unchanged/patched/
    rebuilt) gives IDENTICAL select_files() output to an independently
    fresh-built Corpus over the current on-disk content, for every query."""
    fresh = Corpus(repo, history_msgs=None, use_comments=False, build_docs=False)
    assert cached_corpus.files == fresh.files
    for query in QUERIES:
        c_files, c_scores = _select(cached_corpus, query)
        f_files, f_scores = _select(fresh, query)
        assert c_files == f_files, f"file list diverged for query: {query!r}"
        c_rounded = {k: round(v, 9) for k, v in c_scores.items()}
        f_rounded = {k: round(v, 9) for k, v in f_scores.items()}
        assert c_rounded == f_rounded, f"scores diverged for query: {query!r}"


def test_incremental_property(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    router_v1 = _FILES_V1["pkg/router.py"]

    # Seed the cache with an initial full build.
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "full"
    assert cache_hit is False
    _assert_matches_fresh_build(repo, corpus)

    # Step 1: modify a function body (validators.py) -- no import/def change.
    _write(repo, "pkg/validators.py", _VALIDATORS_V2)
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "incremental", "step 1 (modify function body) should patch, not rebuild"
    assert cache_hit is True
    _assert_matches_fresh_build(repo, corpus)

    # Step 2: modify imports in a file (handlers.py starts importing utils).
    _write(repo, "pkg/handlers.py", _HANDLERS_V2)
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "incremental", "step 2 (modify imports) should patch, not rebuild"
    assert "pkg/utils.py" in edges.get("pkg/handlers.py", set()), "new import edge missing"
    _assert_matches_fresh_build(repo, corpus)

    # Step 3: modify a file to add a new def (utils.py gains slugify()).
    _write(repo, "pkg/utils.py", _UTILS_V2)
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "incremental", "step 3 (add a def) should patch, not rebuild"
    assert corpus.def_index.get("slugify") == ["pkg/utils.py"]
    _assert_matches_fresh_build(repo, corpus)

    # Step 4: touch a file without content change (config.py).
    _touch(repo, "pkg/config.py")
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "incremental", "step 4 (mtime-only touch) should patch, not rebuild"
    _assert_matches_fresh_build(repo, corpus)

    # Step 5: modify two files at once (router.py and service.py).
    _write(repo, "pkg/router.py", _ROUTER_V2)
    _write(repo, "pkg/service.py", _SERVICE_V2)
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "incremental", "step 5 (two files at once) should patch, not rebuild"
    _assert_matches_fresh_build(repo, corpus)

    # Step 6: revert a file (router.py back to its original content).
    _write(repo, "pkg/router.py", router_v1)
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "incremental", "step 6 (revert) should patch, not rebuild"
    assert "add_route" not in corpus.text["pkg/router.py"]
    _assert_matches_fresh_build(repo, corpus)


def test_add_file_triggers_full_rebuild(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cache_mod.load_or_build_ex(repo, with_history=False, with_docs=False, use_cache=True)

    (repo / "pkg" / "extra.py").write_text(
        '"""A brand new module."""\n\n\n'
        "def extra_fn():\n"
        "    return 1\n"
    )
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "full", "adding a file must force a full rebuild"
    assert cache_hit is False
    assert "pkg/extra.py" in corpus.files
    _assert_matches_fresh_build(repo, corpus)


def test_remove_file_triggers_full_rebuild(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cache_mod.load_or_build_ex(repo, with_history=False, with_docs=False, use_cache=True)

    (repo / "pkg" / "utils.py").unlink()
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "full", "removing a file must force a full rebuild"
    assert cache_hit is False
    assert "pkg/utils.py" not in corpus.files
    _assert_matches_fresh_build(repo, corpus)


def test_unchanged_repo_is_pure_cache_hit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    cache_mod.load_or_build_ex(repo, with_history=False, with_docs=False, use_cache=True)
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "unchanged"
    assert cache_hit is True


def test_docs_field_incremental_update(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text(
        "# Guide\n\nSee pkg.router.Router for dispatch details.\n"
    )
    cache_mod.load_or_build_ex(repo, with_history=False, with_docs=True, use_cache=True)

    _write(repo, "docs/guide.md", "# Guide\n\nSee pkg.service.Service for the entry point.\n")
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=True, use_cache=True,
    )
    assert kind == "incremental", "docs-only content edit should patch, not rebuild"
    assert "service" in corpus.docs_text["docs/guide.md"]
    fresh = Corpus(repo, history_msgs=None, use_comments=False, build_docs=True)
    assert corpus.docs_tf == fresh.docs_tf
    assert corpus.docs_df == fresh.docs_df


def test_reverse_import_edge_preserved_when_only_one_side_still_imports(tmp_path: Path) -> None:
    """The tricky case in update_import_graph_for_files: A and B import each
    other; A is edited to stop importing B, but B (unchanged) still imports
    A. The (A, B) edge must survive since B independently still authors it.
    A subsequent edit removing B's import too must then fully remove it."""
    repo = tmp_path / "repo2"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text('"""pkg."""\n')
    a_v1 = (
        '"""Module A."""\n\n'
        "from pkg.b import helper_b\n\n\n"
        "def helper_a():\n"
        "    return helper_b() + 1\n"
    )
    b_v1 = (
        '"""Module B."""\n\n'
        "from pkg.a import helper_a\n\n\n"
        "def helper_b():\n"
        "    return 1\n\n\n"
        "def other_b():\n"
        "    return helper_a() + 2\n"
    )
    (repo / "pkg" / "a.py").write_text(a_v1)
    (repo / "pkg" / "b.py").write_text(b_v1)

    corpus = Corpus(repo, history_msgs=None, use_comments=False, build_docs=False)
    edges = build_import_graph(corpus)
    assert "pkg/b.py" in edges["pkg/a.py"]
    assert "pkg/a.py" in edges["pkg/b.py"]

    # A drops the import of B; B is untouched and still imports A.
    a_v2 = '"""Module A, standalone now."""\n\n\ndef helper_a():\n    return 42\n'
    old_text = {"pkg/a.py": corpus.text["pkg/a.py"]}
    (repo / "pkg" / "a.py").write_text(a_v2)
    assert corpus.update_files(["pkg/a.py"])
    update_import_graph_for_files(corpus, edges, old_text)

    assert "pkg/a.py" in edges.get("pkg/b.py", set()), "B's own import of A must survive"
    assert "pkg/b.py" in edges.get("pkg/a.py", set()), "edge must stay symmetric"

    fresh1 = Corpus(repo, history_msgs=None, use_comments=False, build_docs=False)
    fresh_edges1 = build_import_graph(fresh1)
    assert dict(edges) == dict(fresh_edges1)

    # Now B also drops its import of A -- the edge must fully disappear.
    b_v2 = (
        '"""Module B, standalone now."""\n\n\n'
        "def helper_b():\n"
        "    return 1\n\n\n"
        "def other_b():\n"
        "    return 2\n"
    )
    old_text2 = {"pkg/b.py": corpus.text["pkg/b.py"]}
    (repo / "pkg" / "b.py").write_text(b_v2)
    assert corpus.update_files(["pkg/b.py"])
    update_import_graph_for_files(corpus, edges, old_text2)

    assert "pkg/b.py" not in edges.get("pkg/a.py", set())
    assert "pkg/a.py" not in edges.get("pkg/b.py", set())

    fresh2 = Corpus(repo, history_msgs=None, use_comments=False, build_docs=False)
    fresh_edges2 = build_import_graph(fresh2)
    assert dict(edges) == dict(fresh_edges2)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _make_git_repo(tmp_path: Path) -> Path:
    """Like _make_repo, but a real git repo (init + commit) so
    roust.cache._scan_manifest exercises the git-ls-files-first candidate
    enumeration (roust.core._candidate_files) rather than the raw-walk
    fallback the other tests in this module hit (their repos are never
    git-initialized)."""
    repo = _make_repo(tmp_path)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=test@test.invalid", "-c", "user.name=test", "add", "-A")
    _git(repo, "-c", "user.email=test@test.invalid", "-c", "user.name=test", "commit", "-q", "-m", "test: initial commit")
    return repo


def test_gitignored_file_creation_does_not_trigger_rebuild(tmp_path: Path) -> None:
    """A .gitignore'd file appearing on disk (e.g. a build artifact, or --
    the motivating case -- a .venv/ package file) must never flip the cache
    verdict away from "unchanged": _scan_manifest enumerates candidates via
    the same git-ls-files-first source Corpus itself indexes, so an ignored
    file was never a manifest entry to begin with. A subsequent edit to an
    actually-tracked file must still be detected and patched incrementally."""
    repo = _make_git_repo(tmp_path)

    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "full"
    assert "pkg/router.py" in corpus.files

    (repo / ".gitignore").write_text("ignored_pkg/\n")
    ignored_dir = repo / "ignored_pkg"
    ignored_dir.mkdir()
    (ignored_dir / "junk.py").write_text(
        '"""Should never be indexed -- lives under a gitignored directory."""\n\n\n'
        "def junk_fn():\n"
        "    return 0\n"
    )

    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "unchanged", "an ignored file appearing on disk must not trigger a rebuild"
    assert cache_hit is True
    assert "ignored_pkg/junk.py" not in corpus.files

    # A tracked-file edit must still be detected and incrementally patched.
    _write(repo, "pkg/validators.py", _VALIDATORS_V2)
    corpus, edges, history, cache_hit, kind = cache_mod.load_or_build_ex(
        repo, with_history=False, with_docs=False, use_cache=True,
    )
    assert kind == "incremental", "a tracked-file edit must still be detected"
    assert cache_hit is True
    assert "ignored_pkg/junk.py" not in corpus.files
    _assert_matches_fresh_build(repo, corpus)
