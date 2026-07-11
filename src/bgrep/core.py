"""bgrep retrieval core -- a packaged port of lab/lanes2.py (frozen v7 config).

This module MUST stay retrieval-logic-identical to lab/lanes2.py. The only
intentional differences from lanes2.py are:

  - No LAST_EXPLAIN module global for debugging: select_files() returns an
    Explain dataclass instance instead (see Explain below).
  - Token counting is a local lazy-import wrapper (get_token_counter) instead
    of importing archex.reporting.count_tokens (bgrep has no archex
    dependency); it uses the identical tiktoken cl100k_base encoding.
  - The Corpus file walk additionally excludes the on-disk index cache
    directory (.bgrep/), which does not exist in the lab environment.

See lab/lanes2.py's module docstring for the retrieval-design rationale
(lanes, history signals, vendor/pack-safety fixes) -- it is not repeated here.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CODE_EXTENSIONS = (".py", ".ts", ".js", ".go", ".rs", ".java", ".kt", ".cs", ".swift", ".tsx", ".jsx")
MAX_FILE_BYTES = 2_000_000

# On-disk index cache directory (bgrep.cache); excluded from the corpus like
# .git so it never shows up as a candidate file or pollutes token counts.
_CACHE_DIR_NAME = ".bgrep"


def _is_cache_or_git_path(rel: str) -> bool:
    return (
        rel.startswith(".git/") or "/.git/" in rel
        or rel.startswith(f"{_CACHE_DIR_NAME}/") or f"/{_CACHE_DIR_NAME}/" in rel
    )


# ---------------------------------------------------------------- token counting

def get_token_counter():
    """Return a count_tokens(text: str) -> int callable using tiktoken's
    cl100k_base encoding, matching archex.reporting.count_tokens exactly.
    tiktoken is imported lazily so importing bgrep.core doesn't pay its
    (small but nonzero) import cost when token counting isn't needed."""
    import tiktoken

    encoder = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(encoder.encode(text, disallowed_special=()))

    return count_tokens


# ---------------------------------------------------------------- tokenization

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "import", "return",
    "self", "def", "class", "not", "none", "true", "false", "let", "const",
    "var", "function", "func", "type", "struct", "impl", "use", "pub", "new",
    "int", "str", "string", "bool", "void", "null", "nil", "err", "error",
}


def stem(t: str) -> str:
    """Conservative Porter-style suffix stripping. Both index and query pass
    through this, so only *consistency* matters, not linguistic correctness:
    validators/validate -> validat, dependencies/dependency -> dependenci,
    routing/route/router -> rout."""
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "i"
    elif t.endswith("sses"):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        t = t[:-1]
    if t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    if t.endswith("er") and len(t) > 5:
        t = t[:-2]
    elif t.endswith("or") and len(t) > 6:
        t = t[:-2]
    if t.endswith("y") and len(t) > 4:
        t = t[:-1] + "i"
    elif t.endswith("e") and len(t) > 4:
        t = t[:-1]
    return t


def subtokens(word: str) -> list[str]:
    """Split an identifier into lowercase subtokens (snake_case + camelCase)."""
    parts: list[str] = []
    for chunk in word.split("_"):
        if not chunk:
            continue
        parts.extend(m.group(0).lower() for m in _CAMEL_RE.finditer(chunk))
    return [stem(p) for p in parts if len(p) > 2 and p not in _STOP]


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _IDENT_RE.finditer(text):
        w = m.group(0)
        low = w.lower()
        if len(low) > 2 and low not in _STOP:
            out.append(stem(low))
        subs = subtokens(w)
        if len(subs) > 1 or (subs and subs[0] != stem(low)):
            out.extend(subs)
    return out


def query_terms(question: str, keywords: list[str]) -> list[str]:
    """Query = question tokens + task keywords, subtoken-expanded, deduped."""
    seen: set[str] = set()
    terms: list[str] = []
    for t in tokenize(question) + [s for k in keywords for s in ([stem(k.lower())] + subtokens(k))]:
        if t not in seen and len(t) > 2 and t not in _STOP:
            seen.add(t)
            terms.append(t)
    return terms


# ---------------------------------------------------------------- corpus + BM25

_TESTLIKE_RE = re.compile(
    r"(^|/)(tests?|testing|spec|specs|benches|benchmarks?|examples?|fixtures?|mocks?|docs?|__tests__|e2e"
    r"|docs_src|tutorials?|samples?|demos?|playground|scripts?|integration|t)(/|$)"
    r"|(^|/)(test_|conftest)|_test\.(py|go|rs|ts|js)$|\.test\.|\.spec\.",
    re.I,
)

# Vendor/minified/generated artifacts: not human-authored source, so excluded
# from the corpus unconditionally (same rationale as the test-file prior --
# a document-level property, not a signal that should be gate-able).
_VENDOR_RE = re.compile(
    r"(vendor|vendored|third_party|node_modules|\.min\.(js|css)$|bundle\.js$)",
    re.I,
)
_MAX_LINE_CHARS = 3000


def impl_prior(rel: str) -> float:
    """Document prior: implementation files are a priori more relevant to
    'how does X work' retrieval than tests/benches/examples/docs."""
    return 0.3 if _TESTLIKE_RE.search(rel) else 1.0


def path_tokens(rel: str) -> set[str]:
    toks: set[str] = set()
    for part in re.split(r"[/\\.\-]", rel):
        low = part.lower()
        if len(low) > 2 and low not in _STOP:
            toks.add(stem(low))
        toks.update(subtokens(part))
    return toks


# ---------------------------------------------------------------- NL/comment extraction

_PY_DEF_RE = re.compile(r"^\s*(?:class|def)\s+(\w+)", re.M)
_GO_DEF_RE = re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)", re.M)
_RS_DEF_RE = re.compile(r"^\s*(?:pub\s+)?fn\s+(\w+)|^\s*(?:pub\s+)?struct\s+(\w+)", re.M)
_JS_DEF_RE = re.compile(r"^\s*(?:export\s+)?(?:function|class)\s+(\w+)", re.M)

_PY_DOCSTRING_RE = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.S)
_PY_COMMENT_RE = re.compile(r"#(.*)$", re.M)
_C_BLOCK_COMMENT_RE = re.compile(r"/\*(.*?)\*/", re.S)
_C_LINE_COMMENT_RE = re.compile(r"//(.*)$", re.M)


def extract_comments(rel: str, text: str) -> str:
    """Best-effort NL-channel extraction: docstrings + '#' comments for Python,
    // and /* */ comments for everything else. Not string-context-aware (a '#'
    or '//' inside a string literal is misread as a comment); acceptable noise
    for a bag-of-words signal."""
    parts: list[str] = []
    if rel.endswith(".py"):
        for m in _PY_DOCSTRING_RE.finditer(text):
            parts.append(m.group(1) or m.group(2) or "")
        for m in _PY_COMMENT_RE.finditer(text):
            parts.append(m.group(1))
    else:
        for m in _C_BLOCK_COMMENT_RE.finditer(text):
            parts.append(m.group(1))
        for m in _C_LINE_COMMENT_RE.finditer(text):
            parts.append(m.group(1))
    return "\n".join(parts)


def _def_re_for(rel: str) -> re.Pattern | None:
    """Which definition-symbol regex (if any) applies to `rel`, by extension --
    factored out of Corpus.__init__'s per-file if/elif chain so Corpus.
    update_files() (incremental patch) can reuse the identical mapping when
    subtracting/re-adding a modified file's def_index contributions."""
    if rel.endswith(".py"):
        return _PY_DEF_RE
    elif rel.endswith(".go"):
        return _GO_DEF_RE
    elif rel.endswith(".rs"):
        return _RS_DEF_RE
    elif rel.endswith((".js", ".ts", ".jsx", ".tsx")):
        return _JS_DEF_RE
    return None


def _def_syms(def_re: re.Pattern, text: str) -> set[str]:
    syms: set[str] = set()
    for m in def_re.finditer(text):
        for g in m.groups():
            if g:
                syms.add(g)
    return syms


class Corpus:
    def __init__(
        self,
        repo_path: Path,
        history_msgs: dict[str, str] | None = None,
        use_comments: bool = False,
        build_docs: bool = False,
    ):
        self.repo_path = repo_path
        self.files: list[str] = []
        self.text: dict[str, str] = {}
        self.ptoks: dict[str, set[str]] = {}
        self.tf: dict[str, Counter[str]] = {}
        self.doclen: dict[str, int] = {}
        self.df: Counter[str] = Counter()
        self.use_comments = use_comments
        self.com_tf: dict[str, Counter[str]] = {}
        self.com_df: Counter[str] = Counter()
        self.def_index: dict[str, list[str]] = defaultdict(list)
        for p in sorted(repo_path.rglob("*")):
            if not p.is_file() or p.suffix not in CODE_EXTENSIONS:
                continue
            rel = str(p.relative_to(repo_path))
            if _is_cache_or_git_path(rel):
                continue
            if _VENDOR_RE.search(rel):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            text_lines = text.splitlines()
            if text_lines and max(len(ln) for ln in text_lines) > _MAX_LINE_CHARS:
                continue
            toks = tokenize(text)
            if not toks:
                continue
            self.files.append(rel)
            self.text[rel] = text
            self.ptoks[rel] = path_tokens(rel)
            counts = Counter(toks)
            self.tf[rel] = counts
            self.doclen[rel] = len(toks)
            for term in counts:
                self.df[term] += 1
            if use_comments:
                com_toks = tokenize(extract_comments(rel, text))
                if com_toks:
                    ctf = Counter(com_toks)
                    self.com_tf[rel] = ctf
                    for term in ctf:
                        self.com_df[term] += 1
            if impl_prior(rel) == 1.0:
                def_re = _def_re_for(rel)
                if def_re is not None:
                    for sym in _def_syms(def_re, text):
                        self.def_index[sym].append(rel)
        self.n_docs = len(self.files)
        self.avg_len = (sum(self.doclen.values()) / self.n_docs) if self.n_docs else 1.0
        self.n_com_docs = len(self.com_tf)

        # commit-message field: only for files present in this corpus. Scored
        # by its own standalone BM25 (msg_bm25 below) with proper per-field
        # length normalization -- doclen/avg_len over the msg field only, not
        # the body field. This field is fused into ranking via RRF in
        # select_files(), not blended additively into bm25().
        self.msg_tf: dict[str, Counter[str]] = {}
        self.msg_df: Counter[str] = Counter()
        self.msg_doclen: dict[str, int] = {}
        if history_msgs:
            for rel in self.files:
                msg = history_msgs.get(rel)
                if not msg:
                    continue
                mtoks = tokenize(msg)
                if not mtoks:
                    continue
                mtf = Counter(mtoks)
                self.msg_tf[rel] = mtf
                self.msg_doclen[rel] = len(mtoks)
                for term in mtf:
                    self.msg_df[term] += 1
        self.n_msg_docs = len(self.msg_tf)
        self.msg_avg_len = (sum(self.msg_doclen.values()) / self.n_msg_docs) if self.n_msg_docs else 1.0

        # docs field: *.rst/*.txt/*.md pages, indexed in a wholly separate
        # field from the code corpus above (own tf/df/doclen, own BM25 --
        # see docs_bm25). Test-path pages are excluded (a doc page living
        # under tests/ is test scaffolding, not user-facing documentation).
        # Only built when build_docs=True: doc pages are typically only a
        # few hundred files, but repos with huge changelogs/translations can
        # have thousands, so this is opt-in and capped.
        self.docs_files: list[str] = []
        self.docs_text: dict[str, str] = {}
        self.docs_tf: dict[str, Counter[str]] = {}
        self.docs_df: Counter[str] = Counter()
        self.docs_len: dict[str, int] = {}
        if build_docs:
            doc_paths: list[Path] = []
            for p in sorted(repo_path.rglob("*")):
                if not p.is_file() or p.suffix not in _DOCS_EXTENSIONS:
                    continue
                rel = str(p.relative_to(repo_path))
                if _is_cache_or_git_path(rel):
                    continue
                if _DOCS_EXCLUDE_RE.search(rel):
                    continue
                doc_paths.append(p)
            for p in doc_paths[:_MAX_DOCS_FILES]:
                rel = str(p.relative_to(repo_path))
                try:
                    if p.stat().st_size > _MAX_DOCS_FILE_BYTES:
                        continue
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                dtoks = tokenize(text)
                if not dtoks:
                    continue
                self.docs_files.append(rel)
                self.docs_text[rel] = text
                dcounts = Counter(dtoks)
                self.docs_tf[rel] = dcounts
                self.docs_len[rel] = len(dtoks)
                for term in dcounts:
                    self.docs_df[term] += 1
        self.n_docs_files = len(self.docs_files)
        self.docs_avg_len = (sum(self.docs_len.values()) / self.n_docs_files) if self.n_docs_files else 1.0

    # ------------------------------------------------------------ incremental update
    #
    # Both methods below exist for bgrep.cache's incremental-update path (the
    # common agent edit-loop case: a file's CONTENT changed but its relpath
    # set did not -- no add/remove). Each re-derives a modified file's
    # contribution to the corpus from scratch (subtract old, add new) using
    # the identical per-file logic __init__ uses, so a successfully patched
    # Corpus is observationally identical to a fresh build over the same
    # on-disk content. Neither method touches ptoks (the path, and therefore
    # path_tokens, is unchanged by a content-only edit) or the msg_* fields
    # (commit history is keyed on git HEAD, which incremental updates require
    # to be unchanged -- see bgrep.cache).
    #
    # Both are all-or-nothing: every file is pre-checked against __init__'s
    # own inclusion criteria BEFORE any mutation happens, so a False return
    # leaves the Corpus completely unmodified and the caller is free to
    # discard it and fall back to a full rebuild.

    def update_files(self, rels: list[str]) -> bool:
        """Patch this Corpus in place for `rels` -- files already present in
        self.files whose on-disk content has changed. Re-reads each file
        directly from self.repo_path and applies exactly __init__'s
        per-file inclusion criteria (MAX_FILE_BYTES, _MAX_LINE_CHARS,
        non-empty tokenization) to the new content; if any file's new
        content fails a criterion (or can no longer be read), this is
        shaped like an add/remove (the file would enter/leave the corpus
        under a fresh build) and this method makes NO changes and returns
        False -- callers must fall back to a full rebuild. Returns True,
        having refreshed df/tf/doclen/text/def_index (and com_tf/com_df if
        use_comments) plus avg_len/n_com_docs, on full success."""
        new_text: dict[str, str] = {}
        new_toks: dict[str, list[str]] = {}
        for rel in rels:
            p = self.repo_path / rel
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    return False
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            lines = text.splitlines()
            if lines and max(len(ln) for ln in lines) > _MAX_LINE_CHARS:
                return False
            toks = tokenize(text)
            if not toks:
                return False
            new_text[rel] = text
            new_toks[rel] = toks

        for rel in rels:
            # --- subtract old contributions (self.text[rel] is still old here)
            for term in self.tf[rel]:
                self.df[term] -= 1
                if self.df[term] <= 0:
                    del self.df[term]
            del self.tf[rel]
            del self.doclen[rel]
            if self.use_comments:
                old_ctf = self.com_tf.pop(rel, None)
                if old_ctf:
                    for term in old_ctf:
                        self.com_df[term] -= 1
                        if self.com_df[term] <= 0:
                            del self.com_df[term]
            def_re = _def_re_for(rel) if impl_prior(rel) == 1.0 else None
            if def_re is not None:
                for sym in _def_syms(def_re, self.text[rel]):
                    lst = self.def_index.get(sym)
                    if lst and rel in lst:
                        lst.remove(rel)

            # --- add new contributions
            counts = Counter(new_toks[rel])
            self.tf[rel] = counts
            self.doclen[rel] = len(new_toks[rel])
            for term in counts:
                self.df[term] += 1
            self.text[rel] = new_text[rel]
            if self.use_comments:
                com_toks = tokenize(extract_comments(rel, new_text[rel]))
                if com_toks:
                    ctf = Counter(com_toks)
                    self.com_tf[rel] = ctf
                    for term in ctf:
                        self.com_df[term] += 1
            if def_re is not None:
                for sym in _def_syms(def_re, new_text[rel]):
                    self.def_index[sym].append(rel)

        self.n_com_docs = len(self.com_tf)
        self.avg_len = (sum(self.doclen.values()) / self.n_docs) if self.n_docs else 1.0
        return True

    def update_docs_files(self, rels: list[str]) -> bool:
        """Analogous to update_files() but for the docs field (*.rst/*.txt/
        *.md pages collected when this Corpus was built with
        build_docs=True). Every rel must already be a member of
        self.docs_files. Returns False (no changes) if any file's new
        content would flip its __init__ inclusion verdict (now oversized, or
        now tokenizes to nothing) or can no longer be read -- callers must
        fall back to a full rebuild."""
        new_text: dict[str, str] = {}
        new_toks: dict[str, list[str]] = {}
        for rel in rels:
            p = self.repo_path / rel
            try:
                if p.stat().st_size > _MAX_DOCS_FILE_BYTES:
                    return False
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return False
            toks = tokenize(text)
            if not toks:
                return False
            new_text[rel] = text
            new_toks[rel] = toks

        for rel in rels:
            for term in self.docs_tf[rel]:
                self.docs_df[term] -= 1
                if self.docs_df[term] <= 0:
                    del self.docs_df[term]
            counts = Counter(new_toks[rel])
            self.docs_tf[rel] = counts
            self.docs_len[rel] = len(new_toks[rel])
            for term in counts:
                self.docs_df[term] += 1
            self.docs_text[rel] = new_text[rel]

        self.docs_avg_len = (sum(self.docs_len.values()) / self.n_docs_files) if self.n_docs_files else 1.0
        return True

    def bm25(
        self,
        terms: list[str],
        k1: float = 1.2,
        b: float = 0.75,
        path_weight: float = 2.5,
        use_prior: bool = True,
        comment_weight: float = 0.5,
    ) -> dict[str, float]:
        """BM25F-style: body field (Okapi) + path field (binary match, weighted),
        multiplied by the implementation-file document prior. When use_comments
        (NL/comment field) was supplied to the Corpus, it contributes an
        additional unnormalized-length BM25 term per matching file, weighted by
        comment_weight (comment text is body-like, so no separate field-length
        normalization). The commit-message field is NOT blended in here -- see
        msg_bm25() and select_files()'s rank fusion."""
        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            df = self.df.get(term)
            if df:
                idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
                for rel in self.files:
                    tf = self.tf[rel].get(term)
                    if not tf:
                        continue
                    denom = tf + k1 * (1 - b + b * self.doclen[rel] / self.avg_len)
                    scores[rel] += idf * (tf * (k1 + 1) / denom)
                for rel in self.files:
                    if term in self.ptoks[rel]:
                        scores[rel] += path_weight * idf
            if self.use_comments and self.com_tf and self.n_com_docs:
                cdf = self.com_df.get(term)
                if cdf:
                    idf_com = math.log(1.0 + (self.n_com_docs - cdf + 0.5) / (cdf + 0.5))
                    for rel, ctf_counter in self.com_tf.items():
                        ctf = ctf_counter.get(term)
                        if ctf:
                            scores[rel] += comment_weight * idf_com * (ctf * (k1 + 1) / (ctf + k1))
        if use_prior:
            return {rel: s * impl_prior(rel) for rel, s in scores.items()}
        return dict(scores)

    def msg_bm25(
        self,
        terms: list[str],
        k1: float = 1.2,
        b: float = 0.5,
        use_prior: bool = True,
    ) -> dict[str, float]:
        """Standalone Okapi BM25 over the commit-message field only, with its
        own per-field length normalization (msg_doclen/msg_avg_len, computed
        over the msg field alone -- not the body field's doclen/avg_len).
        Lower b than the body field (0.5 vs 0.75): commit-message document
        length is churn-driven (how many commits touched the file), not prose
        length, so it should be penalized more gently than body length.
        Returns {} if no history_msgs were supplied to this Corpus. Consumed
        by select_files()'s rank fusion, never blended into bm25()."""
        if not self.msg_tf or not self.n_msg_docs:
            return {}
        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            mdf = self.msg_df.get(term)
            if not mdf:
                continue
            idf = math.log(1.0 + (self.n_msg_docs - mdf + 0.5) / (mdf + 0.5))
            for rel, mtf_counter in self.msg_tf.items():
                mtf = mtf_counter.get(term)
                if not mtf:
                    continue
                denom = mtf + k1 * (1 - b + b * self.msg_doclen[rel] / self.msg_avg_len)
                scores[rel] += idf * (mtf * (k1 + 1) / denom)
        if use_prior:
            return {rel: s * impl_prior(rel) for rel, s in scores.items()}
        return dict(scores)

    def docs_bm25(self, terms: list[str], k1: float = 1.2, b: float = 0.75) -> dict[str, float]:
        """Standard Okapi BM25 over the docs field (*.rst/*.txt/*.md pages
        collected when this Corpus was built with build_docs=True), with its
        own per-field length normalization (docs_len/docs_avg_len). No path
        field, no impl-file prior -- these are doc pages, not code files.
        Returns {} if no docs were indexed. Consumed by select_files()'s
        docs-bridge channel (see _apply_docsbridge_promotions), never
        blended into bm25()."""
        if not self.docs_tf or not self.n_docs_files:
            return {}
        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            ddf = self.docs_df.get(term)
            if not ddf:
                continue
            idf = math.log(1.0 + (self.n_docs_files - ddf + 0.5) / (ddf + 0.5))
            for rel, dtf_counter in self.docs_tf.items():
                dtf = dtf_counter.get(term)
                if not dtf:
                    continue
                denom = dtf + k1 * (1 - b + b * self.docs_len[rel] / self.docs_avg_len)
                scores[rel] += idf * (dtf * (k1 + 1) / denom)
        return dict(scores)


# ---------------------------------------------------------------- definition-symbol anchors

_ANCHOR_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
_CODE_SPAN_RE = re.compile(r"```.*?```|`[^`\n]+`", re.S)


def extract_symbol_anchors(question: str, corpus: Corpus) -> list[tuple[str, float]]:
    """Definition-symbol anchor channel: an identifier that appears verbatim in
    the issue text AND is defined (class/def/func/struct/fn) by only a handful
    of files in the repo is strong file-identity evidence -- 25/56 measured
    File@10 failures had a gold file defining a symbol named verbatim in the
    issue. A repo-wide rarity gate (<=3 defining files) is essential: generic
    names like __init__/write/value are defined everywhere and would be pure
    noise. Candidates are matched against the RAW (unstemmed, uncased) question
    text since identifier casing is itself part of the identity signal."""
    code_spans = [(m.start(), m.end()) for m in _CODE_SPAN_RE.finditer(question)]

    def in_code(pos: int) -> bool:
        return any(a <= pos < b for a, b in code_spans)

    occurrences: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for m in _ANCHOR_IDENT_RE.finditer(question):
        s = m.group(0)
        if s not in occurrences:
            order.append(s)
        occurrences[s].append(m.start())

    best: dict[str, float] = {}
    def_counts: dict[str, int] = {}
    for s in order:
        if s.lower() in _STOP:
            continue
        files = corpus.def_index.get(s)
        if not files or len(files) > 3:
            continue
        strength = 2.0 if any(in_code(p) for p in occurrences[s]) else 1.0
        if s != s.lower() or "_" in s:
            strength += 0.5
        for f in files:
            if strength > best.get(f, -1.0):
                best[f] = strength
                def_counts[f] = len(files)
    return sorted(best.items(), key=lambda kv: (kv[1], -def_counts[kv[0]]), reverse=True)


# ---------------------------------------------------------------- import graph

_PY_FROM_RE = re.compile(r"^\s*from\s+([\w\.]+)\s+import\s+(\([^)]*\)|[^\n]+)", re.M)
_PY_PLAIN_IMPORT_RE = re.compile(r"^\s*import\s+([\w\., ]+)", re.M)
_JS_IMPORT_RE = re.compile(
    r"""(?:from\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)|import\s*\(\s*['"]([^'"]+)['"])"""
)
_RS_MOD_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+(\w+)\s*;", re.M)
_RS_USE_RE = re.compile(r"^\s*(?:pub\s+)?use\s+(?:crate|super|self)::([\w:]+)", re.M)
_GO_IMPORT_RE = re.compile(r'"([\w\./\-]+)"')


def _py_module_index(files: list[str]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for rel in files:
        if not rel.endswith(".py"):
            continue
        mod = rel[:-3].replace("/", ".")
        idx[mod] = rel
        if mod.endswith(".__init__"):
            idx[mod[: -len(".__init__")]] = rel
    return idx


def _file_import_targets(
    rel: str, text: str, pyidx: dict[str, str], fileset: set[str]
) -> set[str]:
    """The set of files that `rel`'s OWN text authors an import edge to
    (pre-symmetrization) -- i.e. the per-file body of build_import_graph's
    main loop, factored out so bgrep.cache's incremental-update path
    (update_import_graph_for_files) can recompute a single changed file's
    authored edges without re-parsing the whole corpus. Must stay exactly in
    sync with build_import_graph's loop body -- best-effort per language,
    unresolved imports ignored."""
    targets: set[str] = set()

    def resolve_py_module(mod: str) -> str:
        """Resolve a possibly-relative module spec to an absolute dotted path."""
        if not mod.startswith("."):
            return mod
        level = len(mod) - len(mod.lstrip("."))
        rest = mod.lstrip(".")
        pkg_parts = list(Path(rel).parent.parts)
        # 'from .' = current package; each extra dot goes up one package
        pkg_parts = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
        return ".".join([*pkg_parts, *(rest.split(".") if rest else [])])

    def add_module(mod: str) -> None:
        # exact, then progressively shorter prefixes (import x.y.z -> x/y.py etc.)
        parts = [p for p in mod.split(".") if p]
        for i in range(len(parts), 0, -1):
            hit = pyidx.get(".".join(parts[:i]))
            if hit:
                if hit != rel:
                    targets.add(hit)
                return

    if rel.endswith(".py"):
        for m in _PY_FROM_RE.finditer(text):
            mod = resolve_py_module(m.group(1))
            add_module(mod)
            # `from X import y` where y is itself a submodule
            for name in m.group(2).strip("()").replace("\n", " ").split(","):
                name = name.strip().split(" as ")[0].strip("*# \t")
                if name and "." not in name:
                    sub = pyidx.get(f"{mod}.{name}")
                    if sub and sub != rel:
                        targets.add(sub)
        for m in _PY_PLAIN_IMPORT_RE.finditer(text):
            for spec in m.group(1).split(","):
                mod = spec.strip().split(" as ")[0].strip()
                if mod:
                    add_module(mod)
    elif rel.endswith((".js", ".ts", ".jsx", ".tsx")):
        base = Path(rel).parent
        for m in _JS_IMPORT_RE.finditer(text):
            spec = next(g for g in m.groups() if g)
            if not spec.startswith("."):
                continue
            cand = os.path.normpath(str(base / spec))
            for suffix in ("", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"):
                if cand + suffix in fileset:
                    if cand + suffix != rel:
                        targets.add(cand + suffix)
                    break
    elif rel.endswith(".rs"):
        base = Path(rel).parent
        for m in _RS_MOD_RE.finditer(text):
            name = m.group(1)
            for cand in (str(base / f"{name}.rs"), str(base / name / "mod.rs")):
                if cand in fileset and cand != rel:
                    targets.add(cand)
        for m in _RS_USE_RE.finditer(text):
            head = m.group(1).split("::")[0]
            for cand in (str(base / f"{head}.rs"), str(base / head / "mod.rs"),
                         f"src/{head}.rs", f"src/{head}/mod.rs"):
                if cand in fileset and cand != rel:
                    targets.add(cand)
    elif rel.endswith(".go"):
        # Go: same-package (same dir) linkage dominates; imports resolved by dir suffix.
        for m in _GO_IMPORT_RE.finditer(text):
            pkg = m.group(1)
            tail = pkg.rsplit("/", 1)[-1]
            for other in fileset:
                if other != rel and other.endswith(".go") and Path(other).parent.name == tail:
                    targets.add(other)
    return targets


def build_import_graph(corpus: Corpus) -> dict[str, set[str]]:
    """Undirected import edges between repo files, plus same-directory edges added
    separately by the diffusion step. Best-effort per language; unresolved imports ignored."""
    edges: dict[str, set[str]] = defaultdict(set)
    pyidx = _py_module_index(corpus.files)
    fileset = set(corpus.files)
    for rel in corpus.files:
        for target in _file_import_targets(rel, corpus.text[rel], pyidx, fileset):
            edges[rel].add(target)
            edges[target].add(rel)
    return edges


def update_import_graph_for_files(
    corpus: Corpus, edges: dict[str, set[str]], old_text: dict[str, str]
) -> None:
    """Incrementally patch `edges` (mutated in place) for a batch of files
    whose content changed but whose relpath set is unchanged -- see
    bgrep.cache's incremental-update path. `old_text` is {rel: text} holding
    each changed file's PRE-edit text; corpus.text[rel] must already hold
    the POST-edit text for every rel in `old_text` by the time this is
    called (see Corpus.update_files, which must run first).

    Recomputes each changed file's own authored-edge set
    (_file_import_targets) from both its old and new text. An edge
    (rel, t) is removed only if rel was its SOLE author -- t's own current
    text, re-checked on demand, doesn't independently author it back -- so
    an edge created by an UNCHANGED file Y importing a changed file rel is
    left untouched even though rel's content changed, since Y's text (the
    sole source of that edge) didn't."""
    fileset = set(corpus.files)
    pyidx = _py_module_index(corpus.files)

    def authored(rel: str, text: str) -> set[str]:
        return _file_import_targets(rel, text, pyidx, fileset)

    new_authored = {rel: authored(rel, corpus.text[rel]) for rel in old_text}
    old_authored = {rel: authored(rel, text) for rel, text in old_text.items()}

    def other_authors(t: str, rel: str) -> bool:
        if t in old_text:
            return rel in new_authored[t]
        return rel in authored(t, corpus.text[t])

    touched: set[str] = set(old_text)
    for rel in old_text:
        removed = old_authored[rel] - new_authored[rel]
        added = new_authored[rel] - old_authored[rel]
        touched |= removed | added
        for t in removed:
            if not other_authors(t, rel):
                edges.get(rel, set()).discard(t)
                edges.get(t, set()).discard(rel)
        for t in added:
            edges[rel].add(t)
            edges[t].add(rel)

    for k in touched:
        if k in edges and not edges[k]:
            del edges[k]


def personalized_pagerank(
    seeds: dict[str, float],
    edges: dict[str, set[str]],
    same_dir: dict[str, list[str]],
    alpha: float = 0.15,
    iters: int = 25,
    same_dir_weight: float = 0.35,
) -> dict[str, float]:
    """Random walk with restart. Import edges weight 1, same-directory edges weight
    same_dir_weight. Restart distribution = normalized seed scores."""
    total = sum(seeds.values())
    if total <= 0:
        return {}
    restart = {k: v / total for k, v in seeds.items()}
    rank = dict(restart)
    for _ in range(iters):
        nxt: dict[str, float] = defaultdict(float)
        for node, mass in rank.items():
            if mass <= 1e-12:
                continue
            nbrs = edges.get(node, set())
            dir_nbrs = same_dir.get(str(Path(node).parent), [])
            weights: list[tuple[str, float]] = [(n, 1.0) for n in nbrs]
            weights += [(n, same_dir_weight) for n in dir_nbrs if n != node and n not in nbrs]
            wsum = sum(w for _, w in weights)
            if wsum <= 0:
                nxt[node] += (1 - alpha) * mass  # dangling: hold mass
            else:
                for n, w in weights:
                    nxt[n] += (1 - alpha) * mass * (w / wsum)
        for k, v in restart.items():
            nxt[k] += alpha * v
        rank = dict(nxt)
    return rank


# ---------------------------------------------------------------- selection

@dataclass
class Explain:
    """Diagnostic record of how select_files() reached its result, returned
    directly instead of lanes2.py's LAST_EXPLAIN module global (which was a
    debugging convenience unsafe to keep in a library that may be called
    concurrently). Field population mirrors lanes2's LAST_EXPLAIN dict:
    lex_picks/anchor_promotions/testbridge/docsbridge are always populated;
    sources/pool/additions/cochange_additions/msg_additions are only
    populated on the use_ppr=True path (they stay at their empty defaults
    otherwise, exactly as LAST_EXPLAIN omitted those keys on the non-ppr
    path)."""

    lex_picks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    pool: list[tuple[str, float, float]] = field(default_factory=list)
    additions: list[str] = field(default_factory=list)
    cochange_additions: list[str] = field(default_factory=list)
    msg_additions: list[str] = field(default_factory=list)
    anchor_promotions: list[tuple[str, float, str, str]] = field(default_factory=list)
    testbridge: list[tuple[str, str, str]] = field(default_factory=list)
    docsbridge: list[tuple[str, str, int]] = field(default_factory=list)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mx = max(scores.values())
    return {k: v / mx for k, v in scores.items()} if mx > 0 else scores


def _apply_anchor_promotions(
    out: list[str], anchors: list[tuple[str, float]] | None
) -> tuple[list[str], list[tuple[str, float, str, str]]]:
    """Split anchor promotion into two independent tiers by strength, since a
    300-instance ablation showed the gains come from high-strength (>=2.0,
    backticked-identifier) anchors while the losses come from weak (1.0)
    anchors displacing rank 8-10 files:

    - "head" tier (strength >= 2.0, cap 2): promoted into `out` at position 7
      exactly as before -- files not present anywhere are inserted there;
      files already present but ranked below position 10 are moved up. Files
      already in the top 10 are left untouched.
    - "tail" tier (strength < 2.0, cap 2): never touches the top-10 at all.
      A file absent from `out` entirely is inserted at position 12 (or
      appended if `out` is shorter than that); a file present anywhere
      (including below position 10) is left exactly where it is.

    Position 0-6 (the body ranking's top-7) is never reordered by either
    tier: head promotions are either wholly new or drawn from position >=10,
    and tail promotions never touch anything above position 10."""
    if not anchors:
        return out, []
    promotions: list[tuple[str, float, str, str]] = []  # (file, strength, "insert"|"move", "head"|"tail")

    head_files: list[str] = []
    to_remove: set[str] = set()
    for f, strength in anchors:
        if strength < 2.0 or len(head_files) >= 2 or f in head_files:
            continue
        if f in out:
            idx = out.index(f)
            if idx >= 10:
                head_files.append(f)
                to_remove.add(f)
                promotions.append((f, strength, "move", "head"))
        else:
            head_files.append(f)
            promotions.append((f, strength, "insert", "head"))
    if head_files:
        remaining = [f for f in out if f not in to_remove]
        out = remaining[:7] + head_files + remaining[7:]

    tail_files: list[str] = []
    for f, strength in anchors:
        if strength >= 2.0 or len(tail_files) >= 2 or f in tail_files or f in head_files:
            continue
        if f not in out:
            tail_files.append(f)
            promotions.append((f, strength, "insert", "tail"))
        # else: present anywhere in `out` (top-10 or not) -- leave untouched.
    for f in tail_files:
        pos = min(12, len(out))
        out = out[:pos] + [f] + out[pos:]

    if not head_files and not tail_files:
        return out, []
    return out, promotions


_TESTBRIDGE_EXTS = (".py", ".go", ".rs", ".js", ".ts")


def _apply_testbridge_promotions(
    out: list[str], corpus: Corpus, bm: dict[str, float], edges: dict[str, set[str]]
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Test-file lexical bridge channel, TAIL TIER ONLY: 14/52 measured
    File@10 failures (6/24 at @all) had a stray gold file directly imported
    by a top-5 BM25-matching test file -- diag_channels.py Signal B. Rank
    testlike code files (path matches _TESTLIKE_RE, extension restricted to
    _TESTBRIDGE_EXTS to dodge vendor-y matches) by the RAW (impl-prior-
    downweighted) body bm25 scores already computed for this query; the
    uniform 0.3 downweight doesn't change their relative order. Take the
    top-3 nonzero-scoring test files.

    Candidates are impl files (impl_prior == 1.0) reachable from those test
    files via the import graph (edges already include test<->impl edges).
    bridge_strength for a candidate is its best linking test file's score,
    normalized so the #1 test file's score = 1.0.

    A prior version (v6) also had a "head" tier: candidates linked to the #1
    test file (bridge_strength == 1.0) promoted at position 8, cap 1. A
    300-instance ablation measured that tier losing 7 and gaining only 2 at
    @10 -- it tended to promote low-specificity utility files (e.g.
    __init__.py, test fixtures/helpers) that the top test imports alongside
    its actual subject, since those utility files are also imported by every
    OTHER test in the repo and so look just as "linked to the #1 test" as
    the genuine subject file. The head tier is removed entirely; only the
    tail tier remains, ranked by a specificity score that penalizes exactly
    that failure mode:

        candidate_score = bridge_strength / log(2 + n_test_importers)

    where n_test_importers is the number of testlike files (this corpus's
    full _TESTLIKE_RE/_TESTBRIDGE_EXTS set, not just the top-3) that are
    import-graph neighbors of the candidate -- a file imported by many tests
    is common utility (deprioritized); a file imported by only the linking
    test is maximally specific to this issue.

    Tail tier: up to 3 candidates absent from `out` are inserted at position
    14. Positions 0-13 are never touched (a candidate already present
    anywhere in `out`, including below position 10, is left alone, not
    moved)."""
    testlike = [
        f for f in corpus.files
        if _TESTLIKE_RE.search(f) and Path(f).suffix in _TESTBRIDGE_EXTS
    ]
    testlike_set = set(testlike)
    ranked_tests = sorted(
        ((f, bm.get(f, 0.0)) for f in testlike), key=lambda kv: (-kv[1], kv[0])
    )
    top_tests = [(f, s) for f, s in ranked_tests if s > 0][:3]
    if not top_tests:
        return out, []
    top_score = top_tests[0][1]
    if top_score <= 0:
        return out, []

    # candidate -> (bridge_strength, linking_test)
    candidates: dict[str, tuple[float, str]] = {}
    for test, tscore in top_tests:
        for nbr in edges.get(test, ()):
            if impl_prior(nbr) != 1.0:
                continue
            strength = tscore / top_score
            cur = candidates.get(nbr)
            if cur is None or strength > cur[0]:
                candidates[nbr] = (strength, test)

    def specificity(f: str) -> float:
        strength = candidates[f][0]
        n_test_importers = len(edges.get(f, set()) & testlike_set)
        return strength / math.log(2 + n_test_importers)

    tail_pool = sorted(candidates, key=lambda f: (-specificity(f), f))

    records: list[tuple[str, str, str]] = []  # (file, tier, linking_test)
    tail_files: list[str] = []
    for f in tail_pool:
        if len(tail_files) >= 3:
            break
        if f in out:
            continue
        tail_files.append(f)
        records.append((f, "tail", candidates[f][1]))
    for f in tail_files:
        pos = min(14, len(out))
        out = out[:pos] + [f] + out[pos:]

    return out, records


# ---------------------------------------------------------------- docs-bridge channel

_DOCS_EXTENSIONS = (".rst", ".txt", ".md")
_DOCS_EXCLUDE_RE = re.compile(r"(^|/)(tests?|testing|__tests__)(/|$)", re.I)
_MAX_DOCS_FILE_BYTES = 500_000
_MAX_DOCS_FILES = 4000

_DOTTED_PATH_RE = re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){2,}\b")
_SPHINX_DIRECTIVE_RE = re.compile(
    r"(?:automodule|currentmodule|module|autoclass|autofunction)::\s*([\w\.]+)"
)


def _resolve_py_dotted(dotted: str, pyidx: dict[str, str]) -> str | None:
    """Same exact-then-shortening-prefix lookup build_import_graph()'s
    add_module() uses for `import x.y.z` resolution, factored out so the
    docs-bridge channel can resolve a dotted reference extracted from prose
    (not from an actual import statement) the identical way."""
    parts = [p for p in dotted.split(".") if p]
    for i in range(len(parts), 0, -1):
        hit = pyidx.get(".".join(parts[:i]))
        if hit:
            return hit
    return None


def _apply_docsbridge_promotions(
    out: list[str], corpus: Corpus, terms: list[str]
) -> tuple[list[str], list[tuple[str, str, int]]]:
    """Docs lexical bridge channel, TAIL TIER ONLY (position 16, cap 2): a
    corrected docs diagnostic measured 10/52 File@10 failures (3/24 @all)
    had a stray gold file referenced -- by dotted code path or a Sphinx
    automodule/autoclass/autofunction/currentmodule/module directive -- on a
    top-3 BM25-matching *.rst/*.txt/*.md doc page. Requires the Corpus to
    have been built with build_docs=True (see Corpus.docs_bm25); returns
    `out` unchanged if it wasn't, or if no doc page scores nonzero for this
    query.

    Candidates are resolved from the RAW page text (dotted identifier paths
    of depth >=3, and Sphinx directives) via the same exact-then-prefix
    python module-index lookup build_import_graph() uses for import
    resolution (_resolve_py_dotted) -- never through the lexical BM25 index,
    since a dotted path like `pkg.mod.Class` should resolve structurally,
    not by term overlap. Only resolutions that land on an implementation
    file (impl_prior == 1.0) count as candidates.

    Ranking key: (number of the top-3 pages referencing the candidate,
    then the best-referencing page's rank) -- a file referenced by all
    three top pages outranks one referenced by only the single best page.
    Up to 2 candidates absent from `out` are inserted at position 16; a
    candidate already present anywhere in `out` is left untouched.
    Positions 0-15 are never touched."""
    if not corpus.docs_tf:
        return out, []
    doc_scores = corpus.docs_bm25(terms)
    if not doc_scores:
        return out, []
    ranked_pages = sorted(doc_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_pages = [(f, s) for f, s in ranked_pages if s > 0][:3]
    if not top_pages:
        return out, []

    pyidx = _py_module_index(corpus.files)

    # candidate -> (n_pages_referencing, best_page_rank); best_page_rank 0 ==
    # the #1-scoring page, so lower is better.
    candidates: dict[str, tuple[int, int]] = {}
    for rank, (page, _score) in enumerate(top_pages):
        text = corpus.docs_text[page]
        refs: set[str] = {m.group(0) for m in _DOTTED_PATH_RE.finditer(text)}
        refs.update(m.group(1) for m in _SPHINX_DIRECTIVE_RE.finditer(text))
        resolved: set[str] = set()
        for ref in refs:
            hit = _resolve_py_dotted(ref, pyidx)
            if hit and impl_prior(hit) == 1.0:
                resolved.add(hit)
        for f in resolved:
            n, best_rank = candidates.get(f, (0, rank))
            candidates[f] = (n + 1, min(best_rank, rank))

    tail_pool = sorted(candidates, key=lambda f: (-candidates[f][0], candidates[f][1], f))

    records: list[tuple[str, str, int]] = []  # (file, tier, n_pages_referencing)
    tail_files: list[str] = []
    for f in tail_pool:
        if len(tail_files) >= 2:
            break
        if f in out:
            continue
        tail_files.append(f)
        records.append((f, "tail", candidates[f][0]))
    for f in tail_files:
        pos = min(16, len(out))
        out = out[:pos] + [f] + out[pos:]

    return out, records


def select_files(
    corpus: Corpus,
    terms: list[str],
    use_ppr: bool,
    k_lex: int = 10,
    k_graph: int = 8,
    seed_count: int = 8,
    floor_ratio: float = 0.05,
    cochange: dict[str, dict[str, int]] | None = None,
    cochange_strong: int = 5,
    anchors: list[tuple[str, float]] | None = None,
    use_testbridge: bool = False,
    use_docsbridge: bool = False,
) -> tuple[list[str], dict[str, float], Explain]:
    """Return candidate files: top BM25F picks, optionally UNIONed with the top
    graph-diffusion additions. The union is monotone: diffusion can only add
    files, never displace a lexical pick, so recall(bm25_ppr) >= recall(bm25).

    `cochange`, if given (file -> {other_file: co-commit count}), adds each
    source's co-change partners as additional frontier-expansion pool
    candidates exactly like import-edge / same-dir neighbors. Partners with
    count >= cochange_strong are additionally treated as "import neighbors"
    for the per-source Guarantee-1 step (a file that reliably changes
    alongside a source is, evidentially, as strong a link as an import).

    When the Corpus carries a commit-message field (history on), the msg
    field is NOT fused into lex_picks/sources/pool/additions at all -- an RRF
    fusion of body and msg rankings was tried and measured to destroy head
    precision (recall@1 fell .463->.337 in a 300-instance ablation) for a
    +5-instance gain at @all. The msg channel is therefore monotone: it may
    only APPEND extra candidates after the body-only additions are finalized
    (see the msg_bm25 top-up below), never influence lex_picks or displace/
    reorder anything the body ranking already chose.

    use_testbridge and use_docsbridge are both TAIL-ONLY channels (see
    _apply_testbridge_promotions / _apply_docsbridge_promotions): they insert
    at position >=14, never reorder or displace anything above it, and are
    independent of each other (each only ever inserts into slots the other
    left absent).

    Returns (out, scores, explain) -- explain is an Explain instance (see
    above) replacing lanes2.py's LAST_EXPLAIN module global.
    """
    bm = corpus.bm25(terms)
    if not bm:
        return [], {}, Explain()
    bm_n = _normalize(bm)
    ranked = sorted(bm_n.items(), key=lambda kv: -kv[1])
    best = ranked[0][1]
    lex_picks = [f for i, (f, s) in enumerate(ranked[:k_lex]) if i < 3 or s >= floor_ratio * best]
    # scores stays body-scale (bm_n): fusion decides WHICH files are picked,
    # not the magnitude fed to downstream packing -- keeps the body-score
    # scale that pack_regions' gain/cap math is calibrated against intact.
    scores = dict(bm_n)
    if not use_ppr:
        lex_out, promotions = _apply_anchor_promotions(lex_picks, anchors)
        tb_records: list[tuple[str, str, str]] = []
        if use_testbridge:
            edges = build_import_graph(corpus)
            lex_out, tb_records = _apply_testbridge_promotions(lex_out, corpus, bm, edges)
        db_records: list[tuple[str, str, int]] = []
        if use_docsbridge:
            lex_out, db_records = _apply_docsbridge_promotions(lex_out, corpus, terms)
        explain = Explain(
            lex_picks=lex_picks, anchor_promotions=promotions,
            testbridge=tb_records, docsbridge=db_records,
        )
        return lex_out, scores, explain

    # --- structural expansion: cluster hypothesis + pseudo-relevance feedback.
    # Every observed recall failure is a 1-hop neighbor (same package or direct
    # import) of a top-ranked seed whose own lexical evidence is ~0. So:
    #   1. take the top impl-file seeds,
    #   2. expand the query with each seed's distinctive tf-idf terms (RM3-lite),
    #   3. rank only the seeds' 1-hop neighborhood with the expanded query.
    edges = build_import_graph(corpus)
    same_dir: dict[str, list[str]] = defaultdict(list)
    for rel in corpus.files:
        same_dir[str(Path(rel).parent)].append(rel)

    # Expansion frontier: the 1-hop neighborhood (import edges + same package)
    # of the WHOLE retrieved set, not of a few chosen seeds. Every observed
    # recall failure was a structural neighbor of some retrieved file; picking
    # 3 "seeds" just reintroduced a magic number whose misses became the new
    # failures. Each neighbor is weighted by its strongest linking pick.
    sources = lex_picks[:6]

    # RM3-lite feedback terms from the top implementation picks.
    qset = set(terms)
    fb_terms: set[str] = set()
    for s in [f for f in sources if impl_prior(f) == 1.0][:3]:
        weighted = [
            (t, tf * math.log(1 + corpus.n_docs / (1 + corpus.df.get(t, 1))))
            for t, tf in corpus.tf[s].items() if t not in qset
        ]
        weighted.sort(key=lambda kv: -kv[1])
        fb_terms.update(t for t, _ in weighted[:20])

    bm_fb = corpus.bm25(sorted(fb_terms)) if fb_terms else {}
    fb_n = _normalize(bm_fb)

    pool: dict[str, float] = {}   # candidate -> strength of best linking pick
    owner: dict[str, str] = {}
    import_nbrs: dict[str, list[str]] = {}  # source -> its import-edge (or strong-cochange) candidates
    cochange_origin: set[str] = set()  # candidates reachable via a co-change edge
    fileset = set(corpus.files)
    for s in sources:
        w = bm_n.get(s, 0.0)
        imp: list[str] = []
        co_partners = cochange.get(s, {}) if cochange else {}
        neighbors = list(edges.get(s, ())) + same_dir.get(str(Path(s).parent), [])
        neighbors += [c for c in co_partners if c in fileset and c not in neighbors]
        for c in neighbors:
            if c in lex_picks or c == s or impl_prior(c) < 1.0:
                continue
            if c in edges.get(s, ()):
                imp.append(c)
            elif co_partners.get(c, 0) >= cochange_strong:
                imp.append(c)
            if c in co_partners:
                cochange_origin.add(c)
            if w > pool.get(c, 0.0):
                pool[c] = w
                owner[c] = s
        import_nbrs[s] = imp

    def add_score(c: str) -> float:
        # evidence dominates; the link strength is a soft factor so strong
        # evidence through a weaker pick still beats noise near the top pick.
        return (0.15 + bm_n.get(c, 0.0) + 0.8 * fb_n.get(c, 0.0)) * (0.5 + 0.5 * pool[c])

    # Cutoff: guarantee each source's single best neighbor (no package can be
    # starved), then fill to 16 by global evidence score.
    ranked_pool = sorted(pool, key=add_score, reverse=True)
    additions: list[str] = []
    if ranked_pool:
        pmax = add_score(ranked_pool[0])
        eligible = [c for c in ranked_pool if add_score(c) >= 0.15 * pmax]
        eligible_set = set(eligible)
        # Guarantee 0: eligible neighbors whose PATH mentions a query term.
        # Path matches proved precision-strong all through this experiment
        # (wsgi->handlers, routergroup->router, _validate_call->validators);
        # a structural neighbor named after the question is essentially never
        # noise. Cap by evidence to bound dilution.
        # Only discriminative terms count: a term in most paths (the package
        # name, "src", ...) carries no information, exactly as in idf.
        n = max(len(corpus.files), 1)
        qpath = {
            t for t in terms if len(t) > 3
            and sum(1 for f in corpus.files if t in corpus.ptoks[f]) / n < 0.10
        }
        path_hits = [c for c in eligible if qpath & corpus.ptoks.get(c, set())]
        for c in sorted(path_hits, key=add_score, reverse=True)[:6]:
            if c not in additions:
                additions.append(c)
        # Guarantee 1: each source's best DIRECT-IMPORT neighbor (definitional
        # dependency — the code the source actually calls).
        for s in sources:
            imp = [c for c in import_nbrs.get(s, []) if c in eligible_set]
            if imp:
                top = max(imp, key=add_score)
                if top not in additions:
                    additions.append(top)
        # Guarantee 2: each source's best neighbor overall (package coverage).
        groups: dict[str, list[str]] = defaultdict(list)
        for c in eligible:
            groups[owner[c]].append(c)  # already in score order
        for s in sources:
            grp = groups.get(s, [])
            if grp and grp[0] not in additions:
                additions.append(grp[0])
        for c in eligible:
            if len(additions) >= 16:
                break
            if c not in additions:
                additions.append(c)

    # History top-up: monotone-append only. The msg field never influences
    # lex_picks/sources/pool/additions above -- it can only tack a few extra
    # candidates onto the END of additions, past the 16 cap (up to 19 total),
    # so it can never displace or reorder anything the body ranking chose.
    msg_additions: list[str] = []
    if corpus.msg_tf:
        msg_scores = corpus.msg_bm25(terms)
        if msg_scores:
            msg_max = max(msg_scores.values())
            if msg_max > 0:
                already = set(lex_picks) | set(additions)
                msg_ranked = sorted(msg_scores.items(), key=lambda kv: -kv[1])
                for f, s in msg_ranked:
                    if len(msg_additions) >= 3:
                        break
                    if f in already or s < 0.35 * msg_max or impl_prior(f) != 1.0:
                        continue
                    msg_additions.append(f)
        additions.extend(msg_additions)

    out = lex_picks + additions
    for f in additions:
        scores[f] = max(scores.get(f, 0.0), 0.3 + 0.5 * fb_n.get(f, 0.0))
    out, anchor_promotions = _apply_anchor_promotions(out, anchors)
    tb_records: list[tuple[str, str, str]] = []
    if use_testbridge:
        out, tb_records = _apply_testbridge_promotions(out, corpus, bm, edges)
    db_records: list[tuple[str, str, int]] = []
    if use_docsbridge:
        out, db_records = _apply_docsbridge_promotions(out, corpus, terms)

    explain = Explain(
        sources=sources,
        lex_picks=lex_picks,
        pool=[(c, round(add_score(c), 4), round(pool[c], 2)) for c in ranked_pool],
        additions=additions,
        cochange_additions=[c for c in additions if c in cochange_origin],
        msg_additions=msg_additions,
        anchor_promotions=anchor_promotions,
        testbridge=tb_records,
        docsbridge=db_records,
    )
    return out, scores, explain


# ---------------------------------------------------------------- region packing

_PY_BLOCK_RE = re.compile(r"^(async def |def |class |@)", re.M)


def _python_blocks(text: str) -> list[tuple[int, int]]:
    """Top-level block spans (1-indexed, inclusive) split at column-0 def/class/decorator."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if _PY_BLOCK_RE.match(ln)]
    if not starts:
        return [(1, len(lines))]
    spans = [(1, starts[0])] if starts[0] > 0 else []
    for j, s in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(lines)
        spans.append((s + 1, end))
    return [(a, b) for a, b in spans if b >= a]


def _window_blocks(text: str, hit_lines: list[int], radius: int = 30) -> list[tuple[int, int]]:
    """Language-agnostic fallback: merged +-radius windows around hit lines."""
    n = len(text.splitlines())
    if not hit_lines:
        return [(1, min(n, 2 * radius))]
    spans: list[tuple[int, int]] = []
    for h in sorted(hit_lines):
        a, b = max(1, h - radius), min(n, h + radius)
        if spans and a <= spans[-1][1] + 5:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def _hit_lines(text: str, terms: set[str]) -> list[int]:
    hits = []
    for i, ln in enumerate(text.splitlines(), 1):
        low = ln.lower()
        if any(t in low for t in terms):
            hits.append(i)
    return hits


def pack_regions(
    corpus: Corpus,
    files: list[str],
    terms: list[str],
    scores: dict[str, float],
    budget_tokens: int,
    count_tokens,
) -> tuple[dict[str, list[tuple[int, int]]], str]:
    """Greedy weighted-coverage packing of regions under budget.

    Guarantees every selected file contributes at least its header + best region
    (so the bundle genuinely represents each file), then spends remaining budget
    on regions with the highest marginal (term-coverage x file-score) gain per token.
    Returns ({file: [spans]}, bundle_text).
    """
    tset = set(terms)
    candidates: list[dict] = []
    for rel in files:
        text = corpus.text[rel]
        lines = text.splitlines()
        hits = _hit_lines(text, tset)
        spans = _python_blocks(text) if rel.endswith(".py") else _window_blocks(text, hits)
        hitset = set(hits)
        for a, b in spans:
            seg = "\n".join(lines[a - 1: b])
            seg_terms = tset & set(tokenize(seg))
            n_hits = len(hitset & set(range(a, b + 1)))
            if not seg_terms and n_hits == 0 and a > 1:
                continue
            tok = count_tokens(seg)
            if tok == 0:
                continue
            gain = (len(seg_terms) + 0.5 * n_hits) * (0.3 + scores.get(rel, 0.0))
            candidates.append(
                {"file": rel, "span": (a, b), "tok": tok, "terms": seg_terms, "gain": gain, "text": seg}
            )

    chosen: dict[str, list[dict]] = defaultdict(list)
    spent = 0
    covered: set[str] = set()

    # pass 1: best region per file (recall guarantee for the selected set).
    # Every file gets representation. Allowances are evidence-proportional:
    # a floor of 120 tokens each, with half the budget's remainder distributed
    # by file score, so top-evidence files get deep regions instead of every
    # file getting an equally thin slice.
    n_files = max(len(files), 1)
    floor_tok = 120
    spare = max(0, budget_tokens // 2 - floor_tok * n_files)
    total_score = sum(scores.get(f, 0.0) for f in files) or 1.0
    caps = {
        f: floor_tok + int(spare * scores.get(f, 0.0) / total_score)
        for f in files
    }
    for rel in files:
        cands = [c for c in candidates if c["file"] == rel]
        if not cands:
            continue
        best = max(cands, key=lambda c: c["gain"] / max(c["tok"], 1))
        per_file_cap = caps[rel]
        if best["tok"] > per_file_cap:
            a, b = best["span"]
            seg_lines = corpus.text[rel].splitlines()[a - 1: b]
            keep = max(4, int(len(seg_lines) * per_file_cap / best["tok"]))
            seg = "\n".join(seg_lines[:keep])
            tok = count_tokens(seg)
            if tok > 2 * per_file_cap:
                # Line-count-proportional trim assumes ~uniform tokens/line;
                # it silently fails for pathological few-line segments (e.g. a
                # single minified/vendor line) where slicing seg_lines barely
                # shrinks the token count. Hard-truncate by characters
                # (~4 chars/token) as a backstop so no single file's forced
                # region can blow the pack budget.
                seg = seg[: per_file_cap * 4]
                tok = count_tokens(seg)
            best = {**best, "span": (a, a + keep - 1), "text": seg, "tok": tok}
        chosen[rel].append(best)
        spent += best["tok"]
        covered |= best["terms"]

    # pass 2: greedy marginal coverage
    remaining = [c for c in candidates if c not in [x for v in chosen.values() for x in v]]
    while remaining and spent < budget_tokens:
        def marginal(c: dict) -> float:
            new_terms = len(c["terms"] - covered)
            return (new_terms + 0.25 * len(c["terms"]) + 0.1) * (0.3 + scores.get(c["file"], 0.0)) / max(c["tok"], 1)
        remaining.sort(key=marginal, reverse=True)
        c = remaining.pop(0)
        if spent + c["tok"] > budget_tokens:
            if c["tok"] > 200:
                continue
            break
        chosen[c["file"]].append(c)
        spent += c["tok"]
        covered |= c["terms"]

    parts = []
    spans_out: dict[str, list[tuple[int, int]]] = {}
    for rel in files:
        if rel not in chosen:
            continue
        segs = sorted(chosen[rel], key=lambda c: c["span"][0])
        spans_out[rel] = [c["span"] for c in segs]
        body = "\n...\n".join(c["text"] for c in segs)
        parts.append(f"### {rel}\n{body}")
    return spans_out, "\n\n".join(parts)
