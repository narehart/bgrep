"""Hypothesis retrieval lanes for the archex headtohead task set.

Lanes (each: (task, repo_path) -> dict in the artifact-result shape):
  bm25          - Okapi BM25 over files with identifier-subtoken tokenization; return top-K full files.
  bm25_ppr      - BM25 seeds + personalized PageRank diffusion over the import/same-dir graph; top-K full files.
  bm25_ppr_pack - bm25_ppr candidate set, but return symbol/region-packed bundle under the task token budget
                  (greedy weighted-coverage packing, facility-location style).

No lane ever reads task.expected_files / expected_regions / expected_symbols.

This is a fork of lanes.py that adds three OPTIONAL git-history-derived
signals, all default OFF so lanes2 with flags off matches lanes.py's
*recall* exactly (candidate selection is identical):
  (a) commit-message field on Corpus (history_msgs), appended as a monotone
      top-up of up to 3 extra candidates in select_files() -- it never
      influences lex_picks/sources/pool/additions (an earlier RRF fusion of
      body and msg rankings measured to destroy head precision was removed),
      and it is never blended additively into bm25() (see msg_bm25())
  (b) comment/docstring field on Corpus (use_comments)
  (c) co-change edges in select_files() frontier expansion (cochange),
      including test-file bridge edges between production files that
      co-change with the same test (see history.py)
  (f) neighborhood-first retrieval for large (>3000-file) repos
      (select_files(use_neighborhood=True)): masks the body bm25 dict to a
      seed-anchored, structurally-expanded region before lex_picks/sources/
      pool/additions are chosen, so a monorepo's global lexical dilution
      can't bury the right subgraph (see the "neighborhood-first retrieval"
      section above select_files() for the full design).
See history.py for how (a)/(c) are mined.

Two additional fixes are UNCONDITIONAL (independent of any flag, since both
are corpus/packing hygiene rather than a retrieval signal):
  (d) vendor/minified/generated files (_VENDOR_RE, or any file with a line
      over _MAX_LINE_CHARS chars) are excluded from the Corpus entirely --
      this can change *which files exist in the corpus* and therefore token
      counts (a vendor file that previously occupied budget is now never a
      candidate), though it does not affect recall on non-vendor gold files.
  (e) pack_regions' per-file trim has a hard character-truncation backstop
      for segments that line-count-proportional trimming fails to shrink
      (e.g. a single pathologically long line).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

CODE_EXTENSIONS = (".py", ".ts", ".js", ".go", ".rs", ".java", ".kt", ".cs", ".swift", ".tsx", ".jsx")
# Extended profile for the multilingual baseline (lab experiment code only --
# NOT used by default; see Corpus's `extensions` param and
# swebench_driver2.py's --extensions flag). Adds PHP, Ruby (plus its
# rake/gemspec/erb siblings), the C-family (C/C++/Objective-C headers), and
# the .mjs/.cjs JS module-extension variants (needed so files reachable only
# via extensionless-import resolution, see _JS_RESOLVE_SUFFIXES, actually
# exist in the corpus to be added-to), deduped against CODE_EXTENSIONS.
# CODE_EXTENSIONS itself is untouched -- the default (non-extended) profile,
# including the Python Lite frozen-gate path, is byte-identical to before.
EXTENDED_EXTENSIONS = tuple(dict.fromkeys(
    CODE_EXTENSIONS + (".php", ".rb", ".rake", ".gemspec", ".erb",
                        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
                        ".java", ".kt", ".cs", ".swift", ".mjs", ".cjs")
))
MAX_FILE_BYTES = 2_000_000
LAST_EXPLAIN: dict = {}

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
# a document-level property, not a signal that should be gate-able). The
# dist/*.{mjs,cjs} clause is gated to extensions only EXTENDED_EXTENSIONS
# carries (CODE_EXTENSIONS has no .mjs/.cjs, so this arm can never match on
# the default/Python path -- measured regression fix: indexing .cjs/.mjs
# without it let compiled single-file bundles like dist/node/axios.cjs
# outrank real source, the same failure mode bundle.js was already excluded
# for, just under the extensions this task newly added to the corpus.
_VENDOR_RE = re.compile(
    r"(vendor|vendored|third_party|node_modules|\.min\.(js|css)$|bundle\.(js|mjs|cjs)$"
    r"|(^|/)dist/.*\.(mjs|cjs)$)",
    re.I,
)
_MAX_LINE_CHARS = 3000


def impl_prior(rel: str) -> float:
    """Document prior: implementation files are a priori more relevant to
    'how does X work' retrieval than tests/benches/examples/docs.

    .gemspec/.rake files get an intermediate 0.5 prior (not the 0.3 test
    prior): they are metadata/automation, not implementation, so v8's
    extension of EXTENDED_EXTENSIONS to include them let short,
    project-name-dense gemspecs/rake tasks outrank gold code at @1 (5
    Ruby-instance regression). But they are occasionally the fix site
    itself (e.g. a dependency-version bump lives in the .gemspec), so a
    full 0.3 test-file downweight would over-correct; 0.5 balances the two.
    .erb templates are left at the full 1.0 prior (they can be the actual
    view-layer implementation). This path is unreachable under the default
    CODE_EXTENSIONS profile -- .gemspec/.rake aren't in CODE_EXTENSIONS, so
    the Python/Lite frozen-gate path is untouched."""
    if rel.endswith((".gemspec", ".rake")):
        return 0.5
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
# JS/TS: function/class decls (incl. default/async/generator exports), arrow
# functions assigned to a const, `const X = (` (HOC-wrapped components etc.),
# interface/type aliases. Each alternative has exactly one capturing group,
# so the generic multi-group harvesting loop in Corpus.__init__ below (built
# for the Python/Go/Rust pattern of "one match, one group") works unchanged.
_JS_DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class)\s+(\w+)"
    r"|^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:\([^)]*\)|\w+)\s*=>"
    r"|^\s*(?:export\s+)?const\s+(\w+)\s*=\s*\("
    r"|^\s*(?:export\s+)?interface\s+(\w+)"
    r"|^\s*(?:export\s+)?type\s+(\w+)\s*=",
    re.M,
)
# Ruby: def name / def self.name / class Name(::Nested)* / module Name(::Nested)*.
# Scoped class/module names (containing "::") are further split into their
# individual segments right after extraction below, since issue text can
# name either the qualified or bare form and extract_symbol_anchors() only
# matches contiguous identifier runs (no "::").
_RB_DEF_RE = re.compile(
    r"^\s*def\s+(?:self\.)?([A-Za-z_]\w*[?!=]?)"
    r"|^\s*(?:class|module)\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)",
    re.M,
)

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


class Corpus:
    def __init__(
        self,
        repo_path: Path,
        history_msgs: dict[str, str] | None = None,
        use_comments: bool = False,
        build_docs: bool = False,
        extensions: tuple = CODE_EXTENSIONS,
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
            if not p.is_file() or p.suffix not in extensions:
                continue
            rel = str(p.relative_to(repo_path))
            if rel.startswith(".git/") or "/.git/" in rel:
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
                if rel.endswith(".py"):
                    def_re = _PY_DEF_RE
                elif rel.endswith(".go"):
                    def_re = _GO_DEF_RE
                elif rel.endswith(".rs"):
                    def_re = _RS_DEF_RE
                elif rel.endswith((".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs")):
                    def_re = _JS_DEF_RE
                elif rel.endswith(".rb"):
                    def_re = _RB_DEF_RE
                else:
                    def_re = None
                if def_re is not None:
                    syms: set[str] = set()
                    for m in def_re.finditer(text):
                        for g in m.groups():
                            if g:
                                syms.add(g)
                    if rel.endswith(".rb"):
                        # split scoped class/module names (Foo::Bar) into
                        # their individual segments -- see _RB_DEF_RE comment.
                        for s in list(syms):
                            if "::" in s:
                                syms.update(p for p in s.split("::") if p)
                    for sym in syms:
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
                if rel.startswith(".git/") or "/.git/" in rel:
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
# `export * from './x'` / `export * as ns from './x'` -- re-export barrels.
# Matched separately from _JS_IMPORT_RE (which also matches these lines, so
# the barrel<->target edge already exists) so build_import_graph can follow
# ONE further hop: an importer of the barrel also gets linked to whatever the
# barrel re-exports, since the barrel's own text may never become a ranking
# "source" (see module docstring / _js_reexport_map).
_JS_REEXPORT_STAR_RE = re.compile(
    r"""export\s*\*\s*(?:as\s+\w+\s+)?from\s+['"]([^'"]+)['"]"""
)
# Extensionless-import / directory-import resolution order: plain extensions
# first (most specific), then directory-index files. Applied both to normal
# relative imports and to tsconfig path-alias targets.
_JS_RESOLVE_SUFFIXES = (
    "", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    "/index.js", "/index.ts", "/index.jsx", "/index.tsx", "/index.mjs", "/index.cjs",
)
_RS_MOD_RE = re.compile(r"^\s*(?:pub\s+)?mod\s+(\w+)\s*;", re.M)
_RS_USE_RE = re.compile(r"^\s*(?:pub\s+)?use\s+(?:crate|super|self)::([\w:]+)", re.M)
_GO_IMPORT_RE = re.compile(r'"([\w\./\-]+)"')
_RB_REQUIRE_RELATIVE_RE = re.compile(r"""require_relative\s+['"]([^'"]+)['"]""")
_RB_REQUIRE_RE = re.compile(r"""^\s*require\s+['"]([^'"]+)['"]""", re.M)
# Only SCOPED constant references (containing "::") are resolved to files --
# a bare capitalized word (String, Error, Base, ...) matches far too many
# stdlib/gem/vendor names to be safe evidence; a qualified reference like
# `Foo::BarBaz` is specific enough to be worth the lookup.
_RB_CONST_REF_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)+\b")


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


# ---------------------------------------------------------------- Ruby require/autoload indexes

def _rb_require_index(files: list[str]) -> dict[str, str]:
    """`require "x"` resolution: gem-style convention, the whole subtree under
    ANY lib/ directory in the repo is one flat require-path namespace (so
    lib/my_gem/thing.rb is required as "my_gem/thing"). Best-effort -- no
    gemspec/$LOAD_PATH parsing, just the lib/ convention every packaged gem
    follows. Multiple lib/ roots (monorepo of gems) are all indexed; first
    (sorted-path) writer wins on key collision."""
    idx: dict[str, str] = {}
    for rel in files:
        if not rel.endswith(".rb"):
            continue
        parts = rel.split("/")
        for i, p in enumerate(parts):
            if p == "lib":
                tail = parts[i + 1:]
                if tail:
                    key = "/".join(tail)
                    if key.endswith(".rb"):
                        key = key[:-3]
                    if key:
                        idx.setdefault(key, rel)
                break
    return idx


def _rb_snake(name: str) -> str:
    """ActiveSupport-style underscore: CamelCase -> snake_case (simplified,
    no acronym special-casing -- good enough for a lookup key)."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _rb_autoload_key(rel: str) -> str | None:
    """Rails/gem autoload convention: file path -> expected snake_case
    constant path. Rooted at the nearest lib/ (gem convention: the whole
    lib/ subtree is one namespace, so lib/my_gem/thing.rb -> MyGem::Thing)
    or app/<component>/ (Rails convention: each app/* subdirectory --
    app/models, app/controllers, ... -- is its OWN autoload root, so the
    component name itself is not part of the constant path, e.g.
    app/models/foo/bar_baz.rb -> Foo::BarBaz, not Models::Foo::BarBaz)."""
    if not rel.endswith(".rb"):
        return None
    parts = rel.split("/")
    tail: list[str] | None = None
    for i, p in enumerate(parts):
        if p == "lib":
            tail = parts[i + 1:]
            break
        if p == "app" and i + 1 < len(parts):
            tail = parts[i + 2:]
            break
    if not tail:
        return None
    tail = list(tail)
    if tail[-1].endswith(".rb"):
        tail[-1] = tail[-1][:-3]
    if not tail[-1]:
        return None
    return "/".join(tail)


def _rb_const_index(files: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """snake_case(namespace/path) -> file (full, namespace-qualified), plus a
    leaf-only (last path segment) fallback for references that omit the
    namespace prefix -- e.g. `Bar` used inside `module Foo` to mean
    `Foo::Bar`, which a regex-only reference scan can't disambiguate from
    context. The leaf index only keeps segments unique across the whole repo
    (ambiguous leaves are dropped) to bound false-positive edges."""
    full: dict[str, str] = {}
    leaf_counts: Counter[str] = Counter()
    leaf_file: dict[str, str] = {}
    for rel in files:
        key = _rb_autoload_key(rel)
        if not key:
            continue
        full.setdefault(key, rel)
        leafname = key.rsplit("/", 1)[-1]
        leaf_counts[leafname] += 1
        leaf_file.setdefault(leafname, rel)
    leaf = {k: v for k, v in leaf_file.items() if leaf_counts[k] == 1}
    return full, leaf


def _rb_snake_const(name: str) -> str:
    return "/".join(_rb_snake(seg) for seg in name.split("::") if seg)


# ---------------------------------------------------------------- JS/TS re-export + tsconfig paths

def _js_reexport_map(files: list[str], text_of: dict[str, str], fileset: set[str]) -> dict[str, set[str]]:
    """barrel file -> its `export * from` resolved targets (see
    _JS_REEXPORT_STAR_RE). Consumed by build_import_graph to add a second-hop
    edge from a barrel's IMPORTERS straight to what the barrel re-exports,
    since the barrel file itself may never become a ranking source."""
    out: dict[str, set[str]] = defaultdict(set)
    for rel in files:
        if not rel.endswith((".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs")):
            continue
        base = Path(rel).parent
        for m in _JS_REEXPORT_STAR_RE.finditer(text_of[rel]):
            spec = m.group(1)
            if not spec.startswith("."):
                continue
            cand = os.path.normpath(str(base / spec))
            for suffix in _JS_RESOLVE_SUFFIXES:
                if cand + suffix in fileset:
                    out[rel].add(cand + suffix)
                    break
    return out


def _load_tsconfig_paths(repo_path: Path) -> dict[str, list[str]]:
    """Best-effort parse of compilerOptions.paths from a root tsconfig.json
    (or tsconfig.base.json). Returns {prefix: [target_prefix, ...]} with the
    trailing '/*' stripped from both pattern and targets (the overwhelmingly
    common case, e.g. "@/*": ["src/*"]); baseUrl is applied to targets when
    present. Not a full TS resolver (no monorepo-subpackage tsconfig
    discovery, no `extends` chain, comments/trailing-commas stripped with a
    regex rather than a real JSONC parser) -- malformed or absent config is
    silently ignored, matching this module's best-effort-everywhere policy."""
    for name in ("tsconfig.json", "tsconfig.base.json"):
        p = repo_path / name
        if not p.exists():
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
            cleaned = re.sub(r"(?<!:)//.*", "", raw)
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
            data = json.loads(cleaned)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        opts = data.get("compilerOptions") or {}
        paths = opts.get("paths") or {}
        base_url = opts.get("baseUrl") or "."
        out: dict[str, list[str]] = {}
        for pattern, targets in paths.items():
            if not isinstance(targets, list):
                continue
            key = pattern[:-2] if pattern.endswith("/*") else pattern
            vals: list[str] = []
            for t in targets:
                if not isinstance(t, str):
                    continue
                t = t[:-2] if t.endswith("/*") else t
                vals.append(os.path.normpath(str(Path(base_url) / t)) if base_url != "." else t)
            if vals:
                out[key] = vals
        if out:
            return out
    return {}


def _resolve_js_alias(spec: str, alias_map: dict[str, list[str]], fileset: set[str]) -> str | None:
    """Resolve a non-relative import spec against tsconfig path aliases."""
    for prefix, targets in alias_map.items():
        if spec != prefix and not spec.startswith(prefix + "/"):
            continue
        rest = spec[len(prefix):].lstrip("/")
        for target_base in targets:
            cand = os.path.normpath(f"{target_base}/{rest}" if rest else target_base)
            for suffix in _JS_RESOLVE_SUFFIXES:
                if cand + suffix in fileset:
                    return cand + suffix
    return None


def build_import_graph(corpus: Corpus) -> dict[str, set[str]]:
    """Undirected import edges between repo files, plus same-directory edges added
    separately by the diffusion step. Best-effort per language; unresolved imports ignored."""
    edges: dict[str, set[str]] = defaultdict(set)
    pyidx = _py_module_index(corpus.files)
    fileset = set(corpus.files)
    # Ruby-only indexes (cheap to build unconditionally: a no-op single pass
    # over corpus.files when there are no .rb files at all).
    rb_require_idx = _rb_require_index(corpus.files)
    rb_const_idx, rb_const_leaf_idx = _rb_const_index(corpus.files)
    # JS/TS-only: re-export barrels (see _js_reexport_map) and tsconfig path
    # aliases (see _load_tsconfig_paths), both best-effort/no-op when absent.
    js_reexport = _js_reexport_map(corpus.files, corpus.text, fileset)
    js_alias_map = _load_tsconfig_paths(corpus.repo_path)

    def add(a: str, b: str) -> None:
        if a != b and b in fileset:
            edges[a].add(b)
            edges[b].add(a)

    def resolve_py_module(mod: str, rel: str) -> str:
        """Resolve a possibly-relative module spec to an absolute dotted path."""
        if not mod.startswith("."):
            return mod
        level = len(mod) - len(mod.lstrip("."))
        rest = mod.lstrip(".")
        pkg_parts = list(Path(rel).parent.parts)
        # 'from .' = current package; each extra dot goes up one package
        pkg_parts = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
        return ".".join([*pkg_parts, *(rest.split(".") if rest else [])])

    def add_module(rel: str, mod: str) -> None:
        # exact, then progressively shorter prefixes (import x.y.z -> x/y.py etc.)
        parts = [p for p in mod.split(".") if p]
        for i in range(len(parts), 0, -1):
            hit = pyidx.get(".".join(parts[:i]))
            if hit:
                add(rel, hit)
                return

    for rel in corpus.files:
        text = corpus.text[rel]
        if rel.endswith(".py"):
            for m in _PY_FROM_RE.finditer(text):
                mod = resolve_py_module(m.group(1), rel)
                add_module(rel, mod)
                # `from X import y` where y is itself a submodule
                for name in m.group(2).strip("()").replace("\n", " ").split(","):
                    name = name.strip().split(" as ")[0].strip("*# \t")
                    if name and "." not in name:
                        sub = pyidx.get(f"{mod}.{name}")
                        if sub:
                            add(rel, sub)
            for m in _PY_PLAIN_IMPORT_RE.finditer(text):
                for spec in m.group(1).split(","):
                    mod = spec.strip().split(" as ")[0].strip()
                    if mod:
                        add_module(rel, mod)
        elif rel.endswith((".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs")):
            base = Path(rel).parent
            for m in _JS_IMPORT_RE.finditer(text):
                spec = next(g for g in m.groups() if g)
                if not spec.startswith("."):
                    # non-relative: try tsconfig path aliases (best-effort;
                    # no-op if the repo has no tsconfig.json paths section).
                    hit = _resolve_js_alias(spec, js_alias_map, fileset) if js_alias_map else None
                    if hit:
                        add(rel, hit)
                    continue
                cand = os.path.normpath(str(base / spec))
                target = None
                for suffix in _JS_RESOLVE_SUFFIXES:
                    if cand + suffix in fileset:
                        target = cand + suffix
                        break
                if target:
                    add(rel, target)
                    # re-export chain, 1 hop: if `target` is itself a barrel
                    # (`export * from './y'`), link straight to what it
                    # re-exports too -- `target` may never become a ranking
                    # source on its own, so without this the chain dead-ends.
                    for further in js_reexport.get(target, ()):
                        add(rel, further)
        elif rel.endswith(".rs"):
            base = Path(rel).parent
            for m in _RS_MOD_RE.finditer(text):
                name = m.group(1)
                for cand in (str(base / f"{name}.rs"), str(base / name / "mod.rs")):
                    if cand in fileset:
                        add(rel, cand)
            for m in _RS_USE_RE.finditer(text):
                head = m.group(1).split("::")[0]
                for cand in (str(base / f"{head}.rs"), str(base / head / "mod.rs"),
                             f"src/{head}.rs", f"src/{head}/mod.rs"):
                    if cand in fileset:
                        add(rel, cand)
        elif rel.endswith(".go"):
            # Go: same-package (same dir) linkage dominates; imports resolved by dir suffix.
            for m in _GO_IMPORT_RE.finditer(text):
                pkg = m.group(1)
                tail = pkg.rsplit("/", 1)[-1]
                for other in corpus.files:
                    if other.endswith(".go") and Path(other).parent.name == tail:
                        add(rel, other)
        elif rel.endswith(".rb"):
            base = Path(rel).parent
            for m in _RB_REQUIRE_RELATIVE_RE.finditer(text):
                cand = os.path.normpath(str(base / m.group(1)))
                for suffix in ("", ".rb"):
                    if cand + suffix in fileset:
                        add(rel, cand + suffix)
                        break
            for m in _RB_REQUIRE_RE.finditer(text):
                hit = rb_require_idx.get(m.group(1))
                if hit:
                    add(rel, hit)
            for m in _RB_CONST_REF_RE.finditer(text):
                key = _rb_snake_const(m.group(0))
                hit = rb_const_idx.get(key)
                if not hit:
                    hit = rb_const_leaf_idx.get(key.rsplit("/", 1)[-1])
                if hit:
                    add(rel, hit)
    return edges


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

def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mx = max(scores.values())
    return {k: v / mx for k, v in scores.items()} if mx > 0 else scores


def _apply_anchor_promotions(
    out: list[str], anchors: list[tuple[str, float]] | None, include_tail: bool = True
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
    and tail promotions never touch anything above position 10.

    include_tail=False (lanes3 use_global_order path only) skips the tail
    tier entirely -- head tier still applies unchanged. The tail candidates
    are instead scored and merged into select_files' unified below-lex_picks
    sort by _anchor_tail_candidates, so this flag prevents doing tail
    promotion twice."""
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
    if include_tail:
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


def _anchor_tail_candidates(
    out: list[str], anchors: list[tuple[str, float]] | None
) -> dict[str, float]:
    """lanes3 use_global_order companion to _apply_anchor_promotions'
    tail tier (strength < 2.0, cap 2): same eligibility (`f not in out`,
    i.e. `out` should already reflect head-tier promotion so head files are
    excluded here exactly as the legacy tail loop excludes them via
    `f in head_files`) but returns a {file: synthesized_score} dict instead
    of splicing at a fixed position, so select_files can fold these
    candidates into its single below-lex_picks sort.

    Score mapping onto the add_score scale (evidence floor 0.15, pool-link
    term capped at +0.5 in add_score's own (0.5 + 0.5*pool[c]) factor):
    tail strength is 1.0 or 1.5 (see extract_symbol_anchors), so
    0.15 + 0.5 * (strength / 2.0) lands in [0.275, 0.5125] -- comparable to a
    pool candidate with modest body/feedback evidence and mid-strength
    linking, matching this channel's historical position (spliced at 12,
    below the typical general-fill zone but still ahead of the 14/16
    testbridge/docsbridge tail)."""
    if not anchors:
        return {}
    cands: dict[str, float] = {}
    for f, strength in anchors:
        if strength >= 2.0 or len(cands) >= 2 or f in cands:
            continue
        if f not in out:
            cands[f] = 0.15 + 0.5 * (strength / 2.0)
    return cands


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


def _testbridge_tail_candidates(
    out: list[str], corpus: Corpus, bm: dict[str, float], edges: dict[str, set[str]]
) -> tuple[dict[str, float], list[tuple[str, str, str]]]:
    """lanes3 use_global_order companion to _apply_testbridge_promotions:
    identical candidate discovery/eligibility/cap (top-3 nonzero test files,
    specificity = bridge_strength / log(2 + n_test_importers), cap 3, must
    be absent from `out`) but returns {file: synthesized_score} instead of
    splicing at position 14.

    Score mapping: specificity has no fixed scale (it depends on
    log(2+n_test_importers) for this corpus), so it's max-normalized against
    the best candidate for THIS query -- the same max-normalize convention
    _normalize() uses for bm25 scores elsewhere in this module -- then mapped
    onto the same [0.15, 0.65] band _anchor_tail_candidates uses:
    0.15 + 0.5 * (specificity(f) / specificity(best))."""
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
        return {}, []
    top_score = top_tests[0][1]
    if top_score <= 0:
        return {}, []

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
    tail_pool = [f for f in tail_pool if f not in out][:3]
    if not tail_pool:
        return {}, []
    spec_max = specificity(tail_pool[0])

    scores: dict[str, float] = {}
    records: list[tuple[str, str, str]] = []
    for f in tail_pool:
        norm = specificity(f) / spec_max if spec_max > 0 else 0.0
        scores[f] = 0.15 + 0.5 * norm
        records.append((f, "tail", candidates[f][1]))
    return scores, records


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


def _docsbridge_tail_candidates(
    out: list[str], corpus: Corpus, terms: list[str]
) -> tuple[dict[str, float], list[tuple[str, str, int]]]:
    """lanes3 use_global_order companion to _apply_docsbridge_promotions:
    identical candidate discovery/eligibility/cap (top-3 doc pages, ranked by
    (n_pages_referencing, best_page_rank), cap 2, must be absent from `out`)
    but returns {file: synthesized_score} instead of splicing at position 16.

    Score mapping: composite(f) = n_pages_referencing - best_page_rank/4
    (n_pages dominates, in [1,3]; best_page_rank in {0,1,2} only breaks ties
    within the same n_pages, /4 keeps it from ever flipping an n_pages
    difference) is max-normalized against the best candidate for this query,
    then mapped onto the same [0.15, 0.65] band the other two tail channels
    use: 0.15 + 0.5 * (composite(f) / composite(best))."""
    if not corpus.docs_tf:
        return {}, []
    doc_scores = corpus.docs_bm25(terms)
    if not doc_scores:
        return {}, []
    ranked_pages = sorted(doc_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_pages = [(f, s) for f, s in ranked_pages if s > 0][:3]
    if not top_pages:
        return {}, []

    pyidx = _py_module_index(corpus.files)

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
    tail_pool = [f for f in tail_pool if f not in out][:2]
    if not tail_pool:
        return {}, []

    def composite(f: str) -> float:
        n, best_rank = candidates[f]
        return n - best_rank / 4.0

    comp_max = composite(tail_pool[0])
    scores: dict[str, float] = {}
    records: list[tuple[str, str, int]] = []
    for f in tail_pool:
        norm = composite(f) / comp_max if comp_max > 0 else 0.0
        scores[f] = 0.15 + 0.5 * norm
        records.append((f, "tail", candidates[f][0]))
    return scores, records


# ---------------------------------------------------------------- neighborhood-first retrieval (large repos)

# In large (>3000-file) monorepos, global BM25 dilutes: the right subgraph
# never seeds because the corpus-wide idf/length normalization spreads mass
# too thin across thousands of unrelated files (measured on the JS/TS
# multilingual slice -- babel's 13-16k-file monorepo). LARGER (arXiv:2605.
# 16352) and GraphCoder show the fix: anchor with a few high-precision
# lexical hits, expand structurally over the import/same-dir graph, then run
# the EXISTING full scoring pipeline only within the reached neighborhood.
# Gated OFF by default and only ever engages above _NEIGHBORHOOD_THRESHOLD
# files -- small repos (all measured SWE-bench Lite repos included, max
# ~2.2k files) keep the current pipeline byte-identical.
_NEIGHBORHOOD_THRESHOLD = 3000
# A=10 (not the originally-proposed 20): a 6-instance, 5-value A/R/hops sweep
# on the babel/babel subset of the JS/TS slice (the only >3000-file repo in
# that slice, so the only one where use_neighborhood ever actually engages --
# see swebench_driver2.py runs) measured A in {5,6,8,10,12,15} all beating
# A=20/A=40 by +1 recall (3/5 vs 2/5 gold-present), with no A value in that
# range beating any other -- fewer, sharper rare-term seeds keep the region
# tightly on-topic; A=20's extra seeds pull in enough tangential structural
# neighbors to dilute the in-region BM25 ranking back toward the same
# top-10 cutoff failure the global ranking already had. R and hops were
# NOT the lever: measured region sizes for this corpus topped out at ~330
# files, so R=800/hops=2 never bind (R=150 was tested and DID regress one
# instance by truncating a genuine 1-hop neighbor out of the region -- so
# the cap is real safety margin, not free to shrink without a larger/more
# diverse test corpus to validate against).
_NEIGHBORHOOD_ANCHOR_A = 10
_NEIGHBORHOOD_REGION_CAP = 800
_NEIGHBORHOOD_HOPS = 2
_NEIGHBORHOOD_MIN_REGION = 30
_NEIGHBORHOOD_RARE_DF_RATIO = 0.02


def _neighborhood_seeds(
    corpus: Corpus, terms: list[str], anchors: list[tuple[str, float]] | None
) -> tuple[list[str], dict[str, float]]:
    """Seed set for neighborhood-first retrieval: top-A files by the sum of
    idf of matched RARE query terms (df/n_docs < 2% -- a cheap, high-
    precision lexical signal), UNION all definition-symbol anchor files
    (the caller's already-computed `anchors`, see extract_symbol_anchors).
    These are seeds for structural expansion, never results themselves.

    seed_score gives a single priority key spanning both seed sources so
    region expansion can be ordered deterministically: anchor files (strong
    per-symbol identity evidence) always outrank rare-term seeds; rare-term
    seeds break ties by their idf-sum."""
    rare_terms = [
        t for t in dict.fromkeys(terms)
        if corpus.df.get(t, 0) and corpus.df[t] / corpus.n_docs < _NEIGHBORHOOD_RARE_DF_RATIO
    ]
    rare_score: dict[str, float] = defaultdict(float)
    for t in rare_terms:
        df = corpus.df[t]
        idf = math.log(1.0 + (corpus.n_docs - df + 0.5) / (df + 0.5))
        for rel in corpus.files:
            if t in corpus.tf[rel]:
                rare_score[rel] += idf
    top_rare = [f for f, _ in sorted(rare_score.items(), key=lambda kv: -kv[1])[:_NEIGHBORHOOD_ANCHOR_A]]
    anchor_files = [f for f, _ in (anchors or [])]
    seeds = list(dict.fromkeys(anchor_files + top_rare))
    seed_score: dict[str, float] = {}
    for f, strength in (anchors or []):
        seed_score[f] = max(seed_score.get(f, 0.0), 1000.0 + strength)
    for f, s in rare_score.items():
        if f not in seed_score:
            seed_score[f] = s
    return seeds, seed_score


def _expand_region(
    seeds: list[str],
    seed_score: dict[str, float],
    edges: dict[str, set[str]],
    same_dir: dict[str, list[str]],
    cap: int = _NEIGHBORHOOD_REGION_CAP,
    hops: int = _NEIGHBORHOOD_HOPS,
) -> set[str]:
    """BFS from `seeds` over import-graph + same-directory edges, capped at
    `cap` total files. Expansion stops by hop first (hop 2 never starts if
    the cap is already hit after hop 1), then within a hop by descending
    seed-score priority -- so a capped region always keeps the highest-
    confidence seeds' neighborhoods intact and drops the weakest seeds'
    expansions first. same_dir neighbors of a seed already reach that
    seed's directory siblings on hop 1, so "the seeds' directories'
    siblings" needs no special-casing beyond that."""
    region: set[str] = set(seeds)
    frontier = sorted(seeds, key=lambda f: -seed_score.get(f, 0.0))
    for _ in range(hops):
        if len(region) >= cap:
            break
        next_frontier: list[str] = []
        for node in frontier:
            if len(region) >= cap:
                break
            nbrs = set(edges.get(node, ())) | set(same_dir.get(str(Path(node).parent), []))
            nbrs.discard(node)
            for nb in sorted(nbrs):
                if nb in region:
                    continue
                if len(region) >= cap:
                    break
                region.add(nb)
                next_frontier.append(nb)
        frontier = next_frontier
    return region


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
    use_neighborhood: bool = False,
    use_global_order: bool = False,
) -> tuple[list[str], dict[str, float]]:
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

    use_neighborhood (default OFF) only ever engages when
    corpus.n_docs > _NEIGHBORHOOD_THRESHOLD (see the neighborhood-first
    retrieval section above): a seed set (rare-term idf anchors + symbol
    anchors) is expanded 2 hops over the import/same-dir graph into a capped
    region, and the body bm25 dict is masked to that region BEFORE
    lex_picks/sources/pool/additions are chosen -- every downstream channel
    (anchor promotions, testbridge, docsbridge, msg top-up) then operates
    exactly as it already does, just over a candidate set that excludes
    files outside the region. If the region comes out under
    _NEIGHBORHOOD_MIN_REGION files (or the mask empties bm entirely), this
    falls back to the unmasked global pipeline for this query and records
    the fallback in LAST_EXPLAIN['neighborhood'].

    use_global_order (default OFF; lanes3 experiment -- fixes additions-
    assembly-ORDER starvation) changes how the segment BELOW lex_picks is
    ordered, not which files are eligible to be in it. With it off, this
    function is byte-identical to lanes2: additions are assembled in PHASE
    order (Guarantee0 path-hits, then Guarantee1 one-import-neighbor/source,
    then Guarantee2 one-same-dir-neighbor/source, then general score-order
    fill, then anchor head/tail + testbridge tail + docsbridge tail spliced
    in at fixed positions 7/12/14/16) -- so a rank-4-7-by-add_score candidate
    that happens to lose its phase's single-slot race to a lower-scoring
    candidate from an earlier phase gets pushed out to rank ~21-28 even
    though it's better evidence than plenty of what's ahead of it. Only
    lex_picks (the monotone head) is a phase-ordered concept here; everything
    after it is not, or shouldn't be.

    With it on: the phases still decide ELIGIBILITY -- who MUST appear
    (Guarantee0/1/2 exactly as today, see the Go-package relaxation note
    below, plus anchor-tail/testbridge-tail/docsbridge-tail candidates) --
    but a single comparable score then decides ORDER. Pool members keep
    add_score(); anchor-tail/testbridge/docsbridge-tail candidates get a
    synthesized score onto the same scale (0.15 floor + up to 0.5 of a
    normalized channel-specific strength -- see _anchor_tail_candidates /
    _testbridge_tail_candidates / _docsbridge_tail_candidates, documented at
    each site). The whole below-lex_picks segment (guaranteed candidates +
    eligible pool fill + anchor/bridge tail candidates) is assembled as one
    set and sorted once by that score. Anchor HEAD-tier (strength >= 2.0)
    still runs exactly as before -- same code, same rank-8 splice -- as the
    very last step, on the fully assembled list (so a head anchor already
    present after the unified sort is only ever MOVED up from below rank 10,
    never displaced from the top 10 lex_picks already earned). The msg
    (commit-message) top-up channel is untouched either way: it remains a
    strict monotone append past everything else, per the existing measured
    invariant (RRF-fusing it into ranking cost recall@1 .463->.337 in the
    300-instance ablation cited above) -- it is explicitly NOT part of the
    "below lex_picks" segment this flag reorders.

    Go-package relaxation: Guarantee2 (today: each source's single best
    same-dir neighbor) starves sources whose actual directory is large (e.g.
    a big Go package) -- siblings ranked 2nd-5th by add_score for that source
    can be perfectly good evidence but never get a guaranteed slot under the
    1-per-source cap. When a source's same-dir sibling count is >= 8, up to 4
    additional same-dir siblings (globally capped across ALL sources -- not
    4 per source -- to keep the total list length growth bounded, see below)
    are guaranteed IF their add_score clears `fill_cutoff`: the add_score a
    candidate would need to make the natural top-16-by-score cut on its own
    merits, independent of which phase/source produced it (i.e. this only
    rescues candidates that were already "good enough", not merely present
    in a big directory).

    Total list length: the base fill cap (16) and the msg/anchor-tail/
    testbridge/docsbridge per-channel caps are unchanged; the Go-package
    relaxation is the only source of net growth, and it is globally capped
    at +4 candidates -- so select_files' output grows by at most 4 files
    versus the flag-off pipeline.

    Scoped to the use_ppr branch only -- use_ppr=False has no pool/additions
    to reorder (anchors/testbridge/docsbridge apply their legacy behavior
    unconditionally in that branch regardless of this flag).
    """
    global LAST_EXPLAIN
    bm = corpus.bm25(terms)
    if not bm:
        return [], {}

    edges: dict[str, set[str]] | None = None
    neighborhood_explain: dict = {}
    region: set[str] | None = None
    if use_neighborhood and corpus.n_docs > _NEIGHBORHOOD_THRESHOLD:
        same_dir_all: dict[str, list[str]] = defaultdict(list)
        for rel in corpus.files:
            same_dir_all[str(Path(rel).parent)].append(rel)
        edges = build_import_graph(corpus)
        seeds, seed_score = _neighborhood_seeds(corpus, terms, anchors)
        region = _expand_region(seeds, seed_score, edges, same_dir_all)
        region_size = len(region)
        fallback = region_size < _NEIGHBORHOOD_MIN_REGION
        masked_bm = {f: s for f, s in bm.items() if f in region} if not fallback else {}
        if fallback or not masked_bm:
            fallback = True
            region = None
        else:
            bm = masked_bm
        neighborhood_explain = {
            "active": True, "seed_count": len(seeds),
            "region_size": region_size, "fallback": fallback,
        }

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
            edges = edges if edges is not None else build_import_graph(corpus)
            lex_out, tb_records = _apply_testbridge_promotions(lex_out, corpus, bm, edges)
        db_records: list[tuple[str, str, int]] = []
        if use_docsbridge:
            lex_out, db_records = _apply_docsbridge_promotions(lex_out, corpus, terms)
        LAST_EXPLAIN = {
            "lex_picks": lex_picks, "anchor_promotions": promotions,
            "testbridge": tb_records, "docsbridge": db_records,
        }
        if neighborhood_explain:
            LAST_EXPLAIN["neighborhood"] = neighborhood_explain
        return lex_out, scores

    # --- structural expansion: cluster hypothesis + pseudo-relevance feedback.
    # Every observed recall failure is a 1-hop neighbor (same package or direct
    # import) of a top-ranked seed whose own lexical evidence is ~0. So:
    #   1. take the top impl-file seeds,
    #   2. expand the query with each seed's distinctive tf-idf terms (RM3-lite),
    #   3. rank only the seeds' 1-hop neighborhood with the expanded query.
    edges = edges if edges is not None else build_import_graph(corpus)
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
        if region is not None:
            # neighborhood mode: structural-expansion candidates are also
            # restricted to the region (only lex_picks/sources and the
            # body bm dict are masked above; this keeps the pool consistent
            # with them rather than reaching back out to the full corpus).
            neighbors = [c for c in neighbors if c in region]
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
    relax_extra_used = 0  # lanes3 use_global_order: Go-package relaxation, capped at 4 total
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

        if not use_global_order:
            # --- legacy phase-ordered assembly: byte-identical to lanes2 ---
            for c in sorted(path_hits, key=add_score, reverse=True)[:6]:
                if c not in additions:
                    additions.append(c)
            # Guarantee 1: each source's best DIRECT-IMPORT neighbor
            # (definitional dependency — the code the source actually calls).
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
        else:
            # --- lanes3 use_global_order assembly: the SAME Guarantee0/1/2
            # eligibility checks decide who MUST appear, but membership is
            # collected into a set first and the final order is decided by
            # one sort over add_score() at the end -- no phase gets to plant
            # a lower-scoring candidate ahead of a higher-scoring one from a
            # later phase just by running first. See select_files' docstring.
            guaranteed: list[str] = []
            guaranteed_set: set[str] = set()

            def _guarantee(c: str) -> None:
                if c not in guaranteed_set:
                    guaranteed_set.add(c)
                    guaranteed.append(c)

            # Guarantee 0: path hits (cap 6, same as legacy).
            for c in sorted(path_hits, key=add_score, reverse=True)[:6]:
                _guarantee(c)
            # Guarantee 1: each source's best direct-import neighbor.
            for s in sources:
                imp = [c for c in import_nbrs.get(s, []) if c in eligible_set]
                if imp:
                    _guarantee(max(imp, key=add_score))
            # Guarantee 2: each source's best same-dir/general neighbor, PLUS
            # the Go-package relaxation -- up to 4 extra same-dir siblings
            # (globally capped across all sources, not 4 per source) for
            # sources whose actual directory is large (>=8 files), gated on
            # `fill_cutoff`: the add_score a candidate needs to have made the
            # natural top-16-by-score cut on its own merits, independent of
            # which phase/source produced it.
            groups: dict[str, list[str]] = defaultdict(list)
            for c in eligible:
                groups[owner[c]].append(c)  # already in score order
            fill_cutoff = add_score(eligible[min(15, len(eligible) - 1)]) if eligible else 0.0
            for s in sources:
                grp = groups.get(s, [])
                if not grp:
                    continue
                _guarantee(grp[0])
                if relax_extra_used >= 4:
                    continue
                dir_pool_size = len(same_dir.get(str(Path(s).parent), []))
                if dir_pool_size < 8:
                    continue
                for c in grp[1:]:
                    if relax_extra_used >= 4:
                        break
                    if add_score(c) < fill_cutoff:
                        break  # grp is score-sorted -- no later c clears it either
                    if c in guaranteed_set:
                        continue
                    _guarantee(c)
                    relax_extra_used += 1
            # Eligible pool fill: everything else, up to the base cap (16)
            # plus whatever the relaxation above actually used (<=4) -- the
            # relaxation's guaranteed extras don't shrink the general fill's
            # own budget.
            fill_cap = 16 + relax_extra_used
            for c in eligible:
                if len(guaranteed) >= fill_cap:
                    break
                _guarantee(c)
            additions = sorted(guaranteed, key=lambda c: (-add_score(c), c))

    # History top-up: monotone-append only. The msg field never influences
    # lex_picks/sources/pool/additions above -- it can only tack a few extra
    # candidates onto the END of additions, past the 16 cap (up to 19 total),
    # so it can never displace or reorder anything the body ranking chose.
    # (lanes3 use_global_order: msg stays a strict append past the ENTIRE
    # below-lex_picks segment, including anchor/testbridge/docsbridge tail --
    # it is deliberately excluded from the unified score sort, see
    # select_files' docstring.)
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
        if not use_global_order:
            additions.extend(msg_additions)

    for f in additions:
        scores[f] = max(scores.get(f, 0.0), 0.3 + 0.5 * fb_n.get(f, 0.0))

    if not use_global_order:
        out = lex_picks + additions
        out, anchor_promotions = _apply_anchor_promotions(out, anchors)
        tb_records: list[tuple[str, str, str]] = []
        if use_testbridge:
            out, tb_records = _apply_testbridge_promotions(out, corpus, bm, edges)
        db_records: list[tuple[str, str, int]] = []
        if use_docsbridge:
            out, db_records = _apply_docsbridge_promotions(out, corpus, terms)
    else:
        # lanes3 use_global_order: fold anchor-tail/testbridge-tail/
        # docsbridge-tail candidates into the SAME comparable-score sort as
        # `additions` (pool members) -- ONE sort over the whole below-
        # lex_picks segment, not additions-sorted-then-bridges-appended --
        # then splice anchor head-tier last (its code/rank-8 behavior is
        # unchanged; only its input list differs).
        pre_bridge = lex_picks + additions
        anchor_tail = _anchor_tail_candidates(pre_bridge, anchors)
        tb_records: list[tuple[str, str, str]] = []
        tb_tail: dict[str, float] = {}
        if use_testbridge:
            tb_tail, tb_records = _testbridge_tail_candidates(pre_bridge, corpus, bm, edges)
        db_records: list[tuple[str, str, int]] = []
        db_tail: dict[str, float] = {}
        if use_docsbridge:
            db_tail, db_records = _docsbridge_tail_candidates(pre_bridge, corpus, terms)

        # Bridge candidates are guaranteed disjoint from `additions` (each
        # helper only returns files absent from pre_bridge), so this merge
        # never overwrites a real add_score with a synthesized one.
        bridge_scores: dict[str, float] = dict(anchor_tail)
        for f, s in tb_tail.items():
            bridge_scores[f] = max(bridge_scores.get(f, 0.0), s)
        for f, s in db_tail.items():
            bridge_scores[f] = max(bridge_scores.get(f, 0.0), s)

        below_scores: dict[str, float] = {c: add_score(c) for c in additions}
        below_scores.update(bridge_scores)
        additions = sorted(below_scores, key=lambda f: (-below_scores[f], f))
        # scores (pack_regions' packing-weight dict) already carries the
        # correct 0.3+0.5*fb_n value for pool members from the loop above --
        # only the newly-introduced bridge candidates need a scores entry.
        for f, s in bridge_scores.items():
            scores[f] = max(scores.get(f, 0.0), s)

        additions.extend(msg_additions)
        out = lex_picks + additions
        out, anchor_promotions = _apply_anchor_promotions(out, anchors, include_tail=False)

    LAST_EXPLAIN = {
        "seeds": sources,
        "lex_picks": lex_picks,
        "pool": [(c, round(add_score(c), 4), round(pool[c], 2)) for c in ranked_pool],
        "additions": additions,
        "cochange_additions": [c for c in additions if c in cochange_origin],
        "msg_additions": msg_additions,
        "anchor_promotions": anchor_promotions,
        "testbridge": tb_records,
        "docsbridge": db_records,
        "global_order": use_global_order,
        "relax_extra_used": relax_extra_used,
    }
    if neighborhood_explain:
        LAST_EXPLAIN["neighborhood"] = neighborhood_explain
    return out, scores


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


def _window_blocks(
    text: str, hit_lines: list[int], radius: int = 30, pref_starts: list[int] | None = None
) -> list[tuple[int, int]]:
    """Language-agnostic fallback: merged +-radius windows around hit lines.

    pref_starts (optional, e.g. Ruby def/class/module line numbers -- true
    def/class...end nesting is hard to parse with regex, so this is a cheap
    proxy rather than real block extraction): each window's start snaps back
    to the nearest preferred-start line at or before `h - radius` .. `h`
    instead of the raw `h - radius` cut, so the packed region begins at a
    method/class boundary instead of mid-body; the window's end likewise
    snaps to just before the next preferred-start line if that falls before
    `h + radius`. Default (pref_starts absent/None) is byte-identical to the
    original radius-only behavior -- every other language's callsite is
    unaffected."""
    n = len(text.splitlines())
    if not hit_lines:
        return [(1, min(n, 2 * radius))]
    starts_sorted = sorted(pref_starts) if pref_starts else None
    spans: list[tuple[int, int]] = []
    for h in sorted(hit_lines):
        a, b = max(1, h - radius), min(n, h + radius)
        if starts_sorted:
            back = [s for s in starts_sorted if a <= s <= h]
            if back:
                a = back[-1]
            fwd = [s for s in starts_sorted if h < s <= b]
            if fwd:
                b = fwd[0] - 1
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


_RB_BLOCK_START_RE = re.compile(r"^[ \t]*(?:def|class|module)\s+\S", re.M)


def _rb_block_starts(text: str) -> list[int]:
    """Line numbers of Ruby def/class/module starts at any indentation
    (Ruby methods nest under class/module, not column-0 like Python) --
    used as _window_blocks' pref_starts for region-packing preference."""
    return [text.count("\n", 0, m.start()) + 1 for m in _RB_BLOCK_START_RE.finditer(text)]


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
        if rel.endswith(".py"):
            spans = _python_blocks(text)
        elif rel.endswith(".rb"):
            spans = _window_blocks(text, hits, pref_starts=_rb_block_starts(text))
        else:
            spans = _window_blocks(text, hits)
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
