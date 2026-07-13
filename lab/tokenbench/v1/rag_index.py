"""Condition C: real embedding-based top-k chunk retrieval.

sentence-transformers/all-MiniLM-L6-v2 (CPU), chunking source files into
non-overlapping ~40-line windows, cosine-similarity top-12 chunks against
the issue text -- roughly matching roust's 8192-token default budget (see
README.md for the actual token count observed).

Per-(repo, commit) embedding caches are written to lab/tokenbench/rag_cache/
so re-running an instance (or a condition-C-only rerun) doesn't re-embed the
whole repo every time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHUNK_LINES = 40
TOP_K = 12
MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent / "rag_cache"

# SWE-bench Lite is all-Python; restrict to source files so the index isn't
# dominated by fixtures/data/vendored assets. (Matches the spirit of roust's
# own CODE_EXTENSIONS filter, kept independent/self-contained here.)
CODE_EXTENSIONS = (".py",)
MAX_FILE_BYTES = 1_500_000
SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".tox", ".eggs", "build", "dist"}

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def _iter_source_files(repo_path: Path):
    for p in repo_path.rglob("*"):
        if not p.is_file() or p.suffix not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIR_NAMES or part.startswith(".") for part in p.relative_to(repo_path).parts[:-1]):
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES or p.stat().st_size == 0:
                continue
        except OSError:
            continue
        yield p


def chunk_repo(repo_path: Path) -> list[dict]:
    """Non-overlapping CHUNK_LINES-line windows over every source file."""
    chunks: list[dict] = []
    for p in _iter_source_files(repo_path):
        rel = str(p.relative_to(repo_path))
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        if not lines:
            continue
        for i in range(0, len(lines), CHUNK_LINES):
            window = lines[i : i + CHUNK_LINES]
            text = "\n".join(window).strip()
            if not text:
                continue
            chunks.append({
                "path": rel,
                "start_line": i + 1,
                "end_line": min(i + CHUNK_LINES, len(lines)),
                "text": "\n".join(window),
            })
    return chunks


def _cache_paths(repo_slug: str, commit: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{repo_slug.replace('/', '__')}__{commit[:12]}"
    return CACHE_DIR / f"{key}.npy", CACHE_DIR / f"{key}.meta.json"

def build_or_load_index(repo_path: Path, repo_slug: str, commit: str, use_cache: bool = True):
    """Returns (chunks: list[dict], embeddings: np.ndarray [n, d])."""
    emb_path, meta_path = _cache_paths(repo_slug, commit)
    if use_cache and emb_path.exists() and meta_path.exists():
        embeddings = np.load(emb_path)
        chunks = json.loads(meta_path.read_text())["chunks"]
        if len(chunks) == embeddings.shape[0]:
            return chunks, embeddings
    chunks = chunk_repo(repo_path)
    if not chunks:
        return [], np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    if use_cache:
        np.save(emb_path, embeddings)
        meta_path.write_text(json.dumps({
            "repo": repo_slug, "commit": commit, "chunk_lines": CHUNK_LINES,
            "model": MODEL_NAME,
            "chunks": [{"path": c["path"], "start_line": c["start_line"], "end_line": c["end_line"]} for c in chunks],
        }))
        # store chunk text separately is wasteful to reload; re-read from repo on cache hit instead.
    return chunks, embeddings


def _reattach_text(repo_path: Path, chunks: list[dict]) -> list[dict]:
    """Cache hits store only (path, start, end); re-slice text from disk at
    the current checkout (cheap, avoids duplicating full source in cache)."""
    if chunks and "text" in chunks[0]:
        return chunks
    by_path: dict[str, list[str]] = {}
    out = []
    for c in chunks:
        lines = by_path.get(c["path"])
        if lines is None:
            fp = repo_path / c["path"]
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines() if fp.exists() else []
            by_path[c["path"]] = lines
        window = lines[c["start_line"] - 1 : c["end_line"]]
        out.append({**c, "text": "\n".join(window)})
    return out


def retrieve(repo_path: Path, repo_slug: str, commit: str, query: str, k: int = TOP_K,
             use_cache: bool = True) -> list[dict]:
    chunks, embeddings = build_or_load_index(repo_path, repo_slug, commit, use_cache=use_cache)
    if not chunks:
        return []
    chunks = _reattach_text(repo_path, chunks)
    model = _get_model()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    sims = embeddings @ q_emb
    top_idx = np.argsort(-sims)[:k]
    results = []
    for i in top_idx:
        c = chunks[int(i)]
        results.append({
            "path": c["path"], "start_line": c["start_line"], "end_line": c["end_line"],
            "text": c["text"], "score": float(sims[int(i)]),
        })
    return results


def format_rag_bundle(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"### {r['path']}:{r['start_line']}-{r['end_line']} (score={r['score']:.3f})\n"
                      f"```\n{r['text']}\n```")
    return "\n\n".join(parts)
