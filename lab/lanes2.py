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

import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

CODE_EXTENSIONS = (".py", ".ts", ".js", ".go", ".rs", ".java", ".kt", ".cs", ".swift", ".tsx", ".jsx")
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
        for p in sorted(repo_path.rglob("*")):
            if not p.is_file() or p.suffix not in CODE_EXTENSIONS:
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


def build_import_graph(corpus: Corpus) -> dict[str, set[str]]:
    """Undirected import edges between repo files, plus same-directory edges added
    separately by the diffusion step. Best-effort per language; unresolved imports ignored."""
    edges: dict[str, set[str]] = defaultdict(set)
    pyidx = _py_module_index(corpus.files)
    fileset = set(corpus.files)

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
        elif rel.endswith((".js", ".ts", ".jsx", ".tsx")):
            base = Path(rel).parent
            for m in _JS_IMPORT_RE.finditer(text):
                spec = next(g for g in m.groups() if g)
                if not spec.startswith("."):
                    continue
                cand = os.path.normpath(str(base / spec))
                for suffix in ("", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"):
                    if cand + suffix in fileset:
                        add(rel, cand + suffix)
                        break
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
    """
    bm = corpus.bm25(terms)
    if not bm:
        return [], {}
    bm_n = _normalize(bm)
    ranked = sorted(bm_n.items(), key=lambda kv: -kv[1])
    best = ranked[0][1]
    lex_picks = [f for i, (f, s) in enumerate(ranked[:k_lex]) if i < 3 or s >= floor_ratio * best]
    # scores stays body-scale (bm_n): fusion decides WHICH files are picked,
    # not the magnitude fed to downstream packing -- keeps the body-score
    # scale that pack_regions' gain/cap math is calibrated against intact.
    scores = dict(bm_n)
    if not use_ppr:
        return lex_picks, scores

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

    global LAST_EXPLAIN
    LAST_EXPLAIN = {
        "seeds": sources,
        "lex_picks": lex_picks,
        "pool": [(c, round(add_score(c), 4), round(pool[c], 2)) for c in ranked_pool],
        "additions": additions,
        "cochange_additions": [c for c in additions if c in cochange_origin],
        "msg_additions": msg_additions,
    }
    out = lex_picks + additions
    for f in additions:
        scores[f] = max(scores.get(f, 0.0), 0.3 + 0.5 * fb_n.get(f, 0.0))
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
