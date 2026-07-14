//! roust retrieval core -- a Rust port of `lab/lanes2.py` (frozen v7
//! pipeline). Must stay retrieval-logic-identical to lanes2.py; see
//! PARITY_NOTES.md for every place Python runtime semantics (dict/set
//! ordering, `Path` comparison, `str.splitlines()`, hash randomization)
//! required a deliberate, documented translation choice rather than a
//! literal one.

use crate::pyutil::{normpath_join, path_join_simple, path_sort_key, py_lower, py_parent, py_parent_name, py_splitlines};
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

pub const CODE_EXTENSIONS: &[&str] = &[
    ".py", ".ts", ".js", ".go", ".rs", ".java", ".kt", ".cs", ".swift", ".tsx", ".jsx",
];
pub const MAX_FILE_BYTES: u64 = 2_000_000;

pub fn is_code_file(rel: &str) -> bool {
    CODE_EXTENSIONS.iter().any(|ext| rel.ends_with(ext))
}

fn has_code_suffix(rel: &str) -> bool {
    // Matches Python's `p.suffix in CODE_EXTENSIONS` (last dotted component
    // only, NOT `endswith`) used by the Corpus file walk.
    match rel.rfind('.') {
        Some(idx) => CODE_EXTENSIONS.contains(&&rel[idx..]),
        None => false,
    }
}

pub(crate) fn suffix_of(rel: &str) -> &str {
    // Python `Path(rel).suffix`: the last dotted component of the *final
    // path component* (a leading-dot-only filename like ".gitignore" has no
    // suffix). Good enough here since it's only ever applied to plain
    // filenames without directory separators mixed in oddly.
    let name = match rel.rfind('/') {
        Some(idx) => &rel[idx + 1..],
        None => rel,
    };
    match name.rfind('.') {
        Some(idx) if idx > 0 => &name[idx..],
        _ => "",
    }
}

// ---------------------------------------------------------------- tokenization

static IDENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").unwrap());

static STOP: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "and", "for", "with", "that", "this", "from", "import", "return", "self", "def",
        "class", "not", "none", "true", "false", "let", "const", "var", "function", "func",
        "type", "struct", "impl", "use", "pub", "new", "int", "str", "string", "bool", "void",
        "null", "nil", "err", "error",
    ]
    .into_iter()
    .collect()
});

/// Conservative Porter-style suffix stripping. Length gates count Unicode
/// characters (Python `len(t)`), not bytes; all matched suffixes are ASCII
/// so byte-slicing off the matched suffix is safe once the char-count gate
/// passes. See lanes2.py's `stem()` docstring for the design rationale.
pub fn stem(t: &str) -> String {
    let clen = t.chars().count();
    let mut t = t.to_string();
    let mut clen = clen;

    if t.ends_with("ies") && clen > 4 {
        t = format!("{}i", &t[..t.len() - 3]);
        clen = clen - 3 + 1;
    } else if t.ends_with("sses") {
        t = t[..t.len() - 2].to_string();
        clen -= 2;
    } else if t.ends_with('s') && !t.ends_with("ss") && clen > 3 {
        t = t[..t.len() - 1].to_string();
        clen -= 1;
    }

    if t.ends_with("ing") && clen > 5 {
        t = t[..t.len() - 3].to_string();
        clen -= 3;
    } else if t.ends_with("ed") && clen > 4 {
        t = t[..t.len() - 2].to_string();
        clen -= 2;
    }

    if t.ends_with("er") && clen > 5 {
        t = t[..t.len() - 2].to_string();
        clen -= 2;
    } else if t.ends_with("or") && clen > 6 {
        t = t[..t.len() - 2].to_string();
        clen -= 2;
    }

    if t.ends_with('y') && clen > 4 {
        t = format!("{}i", &t[..t.len() - 1]);
    } else if t.ends_with('e') && clen > 4 {
        t = t[..t.len() - 1].to_string();
    }

    t
}

/// Hand-rolled equivalent of `_CAMEL_RE = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"`
/// (regex lookahead isn't supported by the `regex` crate). Operates on
/// Unicode scalar values but only ever recognizes ASCII upper/lower/digit
/// characters, exactly like the explicit `[A-Z]`/`[a-z]`/`[0-9]` classes in
/// the original -- any other character is simply skipped (matches
/// `re.finditer` never producing a match there).
fn camel_matches(chunk: &str) -> Vec<String> {
    let chars: Vec<char> = chunk.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_ascii_uppercase() {
            let mut run_end = i + 1;
            while run_end < n && chars[run_end].is_ascii_uppercase() {
                run_end += 1;
            }
            let run_len = run_end - i;

            let mut matched_len: Option<usize> = None;
            if run_len >= 2 {
                let mut k = run_len - 1;
                loop {
                    let look_pos = i + k + 1;
                    if look_pos < n && chars[look_pos].is_ascii_lowercase() {
                        matched_len = Some(k);
                        break;
                    }
                    if k == 1 {
                        break;
                    }
                    k -= 1;
                }
            }
            if let Some(k) = matched_len {
                out.push(chars[i..i + k].iter().collect());
                i += k;
                continue;
            }
            if i + 1 < n && chars[i + 1].is_ascii_lowercase() {
                let mut end = i + 1;
                while end < n && chars[end].is_ascii_lowercase() {
                    end += 1;
                }
                out.push(chars[i..end].iter().collect());
                i = end;
                continue;
            }
            out.push(chars[i..run_end].iter().collect());
            i = run_end;
        } else if c.is_ascii_lowercase() {
            let mut end = i + 1;
            while end < n && chars[end].is_ascii_lowercase() {
                end += 1;
            }
            out.push(chars[i..end].iter().collect());
            i = end;
        } else if c.is_ascii_digit() {
            let mut end = i + 1;
            while end < n && chars[end].is_ascii_digit() {
                end += 1;
            }
            out.push(chars[i..end].iter().collect());
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// Split an identifier into lowercase subtokens (snake_case + camelCase).
pub fn subtokens(word: &str) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    for chunk in word.split('_') {
        if chunk.is_empty() {
            continue;
        }
        for m in camel_matches(chunk) {
            parts.push(py_lower(&m));
        }
    }
    parts
        .into_iter()
        .filter(|p| p.chars().count() > 2 && !STOP.contains(p.as_str()))
        .map(|p| stem(&p))
        .collect()
}

pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for m in IDENT_RE.find_iter(text) {
        let w = m.as_str();
        let low = py_lower(w);
        if low.chars().count() > 2 && !STOP.contains(low.as_str()) {
            out.push(stem(&low));
        }
        let subs = subtokens(w);
        if subs.len() > 1 || (!subs.is_empty() && subs[0] != stem(&low)) {
            out.extend(subs);
        }
    }
    out
}

/// Query = question tokens + task keywords, subtoken-expanded, deduped.
pub fn query_terms(question: &str, keywords: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut terms: Vec<String> = Vec::new();
    let mut candidates: Vec<String> = tokenize(question);
    for k in keywords {
        let lowk = py_lower(k);
        candidates.push(stem(&lowk));
        candidates.extend(subtokens(k));
    }
    for t in candidates {
        if !seen.contains(&t) && t.chars().count() > 2 && !STOP.contains(t.as_str()) {
            seen.insert(t.clone());
            terms.push(t);
        }
    }
    terms
}

/// How many of `terms` exist ANYWHERE in `corpus`'s vocabulary (body,
/// comment/NL, docs-page, commit-message, or path-token fields) -- i.e. the
/// full set of fields `Corpus::bm25`/`docs_bm25`/`msg_bm25` ever look a term
/// up in, not just the primary body `df`. Returns `(matched, total)`, `total`
/// always equal to `terms.len()`. Purely diagnostic (feeds `--json`'s
/// `matched_query_terms`/`total_query_terms` stats and the zero-match exit-1
/// gate in `main.rs`) -- never consulted by `select_files`, so it cannot
/// itself change ranking.
pub fn query_term_coverage(corpus: &Corpus, terms: &[String]) -> (usize, usize) {
    let mut path_vocab: HashSet<&str> = HashSet::new();
    for toks in corpus.ptoks.values() {
        for t in toks {
            path_vocab.insert(t.as_str());
        }
    }
    let matched = terms
        .iter()
        .filter(|t| {
            corpus.df.contains_key(t.as_str())
                || corpus.com_df.contains_key(t.as_str())
                || corpus.docs_df.contains_key(t.as_str())
                || corpus.msg_df.contains_key(t.as_str())
                || path_vocab.contains(t.as_str())
        })
        .count();
    (matched, terms.len())
}

/// Calibrated low-confidence gate (issue #25). `top_score` is the raw,
/// pre-normalization top pooled BM25F score (`Explain::top_score`);
/// `matched_terms`/`total_terms` come from `query_term_coverage`. Trips when
/// EITHER the strongest candidate's raw score is below
/// `LOW_CONFIDENCE_TOP_SCORE`, OR fewer than
/// `LOW_CONFIDENCE_MATCH_FRACTION` of the query's terms exist anywhere in the
/// corpus vocabulary -- either one, on its own, is evidence the match is
/// coincidental rather than substantive.
///
/// Calibrated empirically (issue #25), not guessed:
///   - Real population: all 300 SWE-bench Lite (query, repo) pairs.
///     top_score min/p5/p25/median/p75/max = 12.04/37.44/63.07/89.98/139.18/361.99.
///     matched-fraction min/p5/p25/median/p75/max = 0.460/0.865/0.938/0.965/1.0/1.0.
///   - Gibberish population: 30 queries (15 random-ASCII, 10 shuffled-identifier
///     soup, 5 plausible-but-wrong feature descriptions) x 3 repos (django,
///     this repo, matplotlib) = 90 runs; 70/90 were literal zero-match (caught
///     by the exit-1 gate below, not this flag). Of the remaining 20
///     (nonzero-match) runs: top_score min/p5/p25/median/p75/max =
///     1.26/1.44/2.66/8.48/11.94/24.75; matched-fraction min/max = 0.20/0.875.
///
/// The two top_score distributions OVERLAP (gibberish max 24.75 > real min
/// 12.04) -- there is no threshold with full separation. Per the hard
/// constraint (zero false trips on the 300 real queries), thresholds are set
/// just under the observed real-query minima: 12.0 (vs. real min 12.04) and
/// 0.45 (vs. real min match-fraction 0.460). At these thresholds: 0/300 real
/// queries trip; 16/20 (80%) of the nonzero-match gibberish runs trip (all
/// with top_score < 12.0) -- combined with the 70 caught by exit-1, 86/90
/// (95.6%) of the full gibberish population is flagged one way or the other.
/// The 4 gibberish queries that slip through untripped are the deliberately
/// hardest case: plausible-but-wrong feature descriptions (e.g. "OAuth2
/// device code flow refresh token rotation", "GraphQL subscription resolver
/// batching") that happen to share enough real vocabulary with the target
/// repo (auth/token/schema-adjacent terms) to score above both thresholds --
/// an accepted, reported trade-off per the calibration spec, not a bug.
pub const LOW_CONFIDENCE_TOP_SCORE: f64 = 12.0;
pub const LOW_CONFIDENCE_MATCH_FRACTION: f64 = 0.45;

pub fn is_low_confidence(top_score: f64, matched_terms: usize, total_terms: usize) -> bool {
    if total_terms == 0 {
        return true;
    }
    let match_fraction = matched_terms as f64 / total_terms as f64;
    top_score < LOW_CONFIDENCE_TOP_SCORE || match_fraction < LOW_CONFIDENCE_MATCH_FRACTION
}

// ---------------------------------------------------------------- corpus + BM25

pub static TESTLIKE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(^|/)(tests?|testing|spec|specs|benches|benchmarks?|examples?|fixtures?|mocks?|docs?|__tests__|e2e|docs_src|tutorials?|samples?|demos?|playground|scripts?|integration|t)(/|$)|(^|/)(test_|conftest)|_test\.(py|go|rs|ts|js)$|\.test\.|\.spec\.",
    )
    .unwrap()
});

static VENDOR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(vendor|vendored|third_party|node_modules|\.min\.(js|css)$|bundle\.js$)").unwrap());

const MAX_LINE_CHARS: usize = 3000;

pub fn impl_prior(rel: &str) -> f64 {
    if TESTLIKE_RE.is_match(rel) {
        0.3
    } else {
        1.0
    }
}

static PATH_SPLIT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[/\\.\-]").unwrap());

pub fn path_tokens(rel: &str) -> HashSet<String> {
    let mut toks = HashSet::new();
    for part in PATH_SPLIT_RE.split(rel) {
        let low = py_lower(part);
        if low.chars().count() > 2 && !STOP.contains(low.as_str()) {
            toks.insert(stem(&low));
        }
        for s in subtokens(part) {
            toks.insert(s);
        }
    }
    toks
}

// ---------------------------------------------------------------- NL/comment extraction

static PY_DEF_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*(?:class|def)\s+(\w+)").unwrap());
static GO_DEF_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^func\s+(?:\([^)]*\)\s*)?(\w+)").unwrap());
static RS_DEF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*(?:pub\s+)?fn\s+(\w+)|^\s*(?:pub\s+)?struct\s+(\w+)").unwrap());
static JS_DEF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*(?:export\s+)?(?:function|class)\s+(\w+)").unwrap());

static PY_DOCSTRING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?s)"""(.*?)"""|'''(.*?)'''"#).unwrap());
static PY_COMMENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)#(.*)$").unwrap());
static C_BLOCK_COMMENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)/\*(.*?)\*/").unwrap());
static C_LINE_COMMENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)//(.*)$").unwrap());

pub fn extract_comments(rel: &str, text: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if rel.ends_with(".py") {
        for cap in PY_DOCSTRING_RE.captures_iter(text) {
            let g = cap.get(1).or_else(|| cap.get(2)).map(|m| m.as_str()).unwrap_or("");
            parts.push(g.to_string());
        }
        for cap in PY_COMMENT_RE.captures_iter(text) {
            parts.push(cap.get(1).map(|m| m.as_str()).unwrap_or("").to_string());
        }
    } else {
        for cap in C_BLOCK_COMMENT_RE.captures_iter(text) {
            parts.push(cap.get(1).map(|m| m.as_str()).unwrap_or("").to_string());
        }
        for cap in C_LINE_COMMENT_RE.captures_iter(text) {
            parts.push(cap.get(1).map(|m| m.as_str()).unwrap_or("").to_string());
        }
    }
    parts.join("\n")
}

// ---------------------------------------------------------------- docs field constants

pub const DOCS_EXTENSIONS: &[&str] = &[".rst", ".txt", ".md"];
static DOCS_EXCLUDE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(^|/)(tests?|testing|__tests__)(/|$)").unwrap());
pub(crate) const MAX_DOCS_FILE_BYTES: u64 = 500_000;
const MAX_DOCS_FILES: usize = 4000;

// ---------------------------------------------------------------- filesystem walk

/// Enumerate candidate files via `git ls-files` (tracked + untracked-but-
/// not-ignored), inheriting .gitignore/.git/info/exclude/global excludes --
/// exactly ripgrep's file-discovery semantics. Returns `None` if
/// `repo_path` isn't inside a git work tree, or if the git invocation fails
/// for ANY reason -- callers must fall back to a raw filesystem walk rather
/// than hard-fail indexing.
pub(crate) fn git_ls_files(repo_path: &Path) -> Option<Vec<String>> {
    let check = std::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(repo_path)
        .output()
        .ok()?;
    if !check.status.success() || String::from_utf8_lossy(&check.stdout).trim() != "true" {
        return None;
    }
    let output = std::process::Command::new("git")
        .args(["ls-files", "-z", "--cached", "--others", "--exclude-standard"])
        .current_dir(repo_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(
        output
            .stdout
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect(),
    )
}

/// Order paths the way `sorted(repo_path.rglob("*"))` does: full component-
/// wise (parts-tuple) comparison, computed once over ALL files then used to
/// sort; directories are never yielded (Python's loop `continue`s on
/// `not p.is_file()` immediately, so their traversal order is irrelevant to
/// the surviving files' relative order).
///
/// Prefers `git_ls_files` when `repo_path` is inside a git work tree (see
/// above), falling back to a raw filesystem walk otherwise -- or if the git
/// invocation fails for any reason. Falling back to the raw walk means
/// entries here are not guaranteed to be plain files in the git-list case
/// (e.g. a submodule gitlink); callers must still verify via `is_file`
/// metadata before reading.
///
/// `pub(crate)` so `cache.rs` can enumerate the SAME candidate set for its
/// manifest / history current-files scans (rather than re-walking with its
/// own logic) -- the manifest must cover exactly the files `Corpus::build`
/// indexes, or add/remove change detection would desync from what actually
/// gets (re)indexed.
pub(crate) fn walk_all_files(repo_path: &Path) -> Vec<String> {
    if let Some(mut rels) = git_ls_files(repo_path) {
        rels.sort_by(|a, b| path_sort_key(a).cmp(&path_sort_key(b)));
        return rels;
    }
    let mut out: Vec<String> = Vec::new();
    fn recurse(dir: &Path, base: &Path, out: &mut Vec<String>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        let mut items: Vec<std::fs::DirEntry> = entries.flatten().collect();
        items.sort_by_key(|e| e.file_name());
        for entry in items {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            if file_type.is_dir() {
                recurse(&path, base, out);
            } else {
                let is_file = if file_type.is_symlink() {
                    std::fs::metadata(&path).map(|m| m.is_file()).unwrap_or(false)
                } else {
                    file_type.is_file()
                };
                if is_file {
                    if let Ok(rel) = path.strip_prefix(base) {
                        if let Some(relstr) = rel.to_str() {
                            out.push(relstr.replace('\\', "/"));
                        }
                    }
                }
            }
        }
    }
    recurse(repo_path, repo_path, &mut out);
    out.sort_by(|a, b| path_sort_key(a).cmp(&path_sort_key(b)));
    out
}

fn read_text_lossy(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn counter_from_tokens(tokens: &[String]) -> IndexMap<String, u32> {
    let mut m = IndexMap::new();
    for t in tokens {
        *m.entry(t.clone()).or_insert(0) += 1;
    }
    m
}

// ---------------------------------------------------------------- Corpus

#[derive(Serialize, Deserialize)]
pub struct Corpus {
    pub repo_path: PathBuf,
    pub files: Vec<String>,
    pub text: HashMap<String, String>,
    pub ptoks: HashMap<String, HashSet<String>>,
    pub tf: HashMap<String, IndexMap<String, u32>>,
    pub doclen: HashMap<String, u32>,
    pub df: HashMap<String, u32>,
    pub use_comments: bool,
    pub com_tf: HashMap<String, IndexMap<String, u32>>,
    pub com_df: HashMap<String, u32>,
    pub def_index: HashMap<String, Vec<String>>,
    pub n_docs: usize,
    pub avg_len: f64,
    pub n_com_docs: usize,

    pub msg_tf: HashMap<String, IndexMap<String, u32>>,
    pub msg_df: HashMap<String, u32>,
    pub msg_doclen: HashMap<String, u32>,
    pub n_msg_docs: usize,
    pub msg_avg_len: f64,

    pub docs_files: Vec<String>,
    pub docs_text: HashMap<String, String>,
    pub docs_tf: HashMap<String, IndexMap<String, u32>>,
    pub docs_df: HashMap<String, u32>,
    pub docs_len: HashMap<String, u32>,
    pub n_docs_files: usize,
    pub docs_avg_len: f64,
}

impl Corpus {
    pub fn build(
        repo_path: &Path,
        history_msgs: Option<&IndexMap<String, String>>,
        use_comments: bool,
        build_docs: bool,
    ) -> Corpus {
        let mut files = Vec::new();
        let mut text: HashMap<String, String> = HashMap::new();
        let mut ptoks: HashMap<String, HashSet<String>> = HashMap::new();
        let mut tf: HashMap<String, IndexMap<String, u32>> = HashMap::new();
        let mut doclen: HashMap<String, u32> = HashMap::new();
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut com_tf: HashMap<String, IndexMap<String, u32>> = HashMap::new();
        let mut com_df: HashMap<String, u32> = HashMap::new();
        let mut def_index: HashMap<String, Vec<String>> = HashMap::new();

        let all_files = walk_all_files(repo_path);
        for rel in &all_files {
            if rel.starts_with(".git/") || rel.contains("/.git/") {
                continue;
            }
            if !has_code_suffix(rel) {
                continue;
            }
            if VENDOR_RE.is_match(rel) {
                continue;
            }
            let full = repo_path.join(rel);
            let meta = match std::fs::metadata(&full) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if !meta.is_file() {
                continue;
            }
            if meta.len() > MAX_FILE_BYTES {
                continue;
            }
            let txt = match read_text_lossy(&full) {
                Some(t) => t,
                None => continue,
            };
            let text_lines = py_splitlines(&txt);
            if let Some(maxlen) = text_lines.iter().map(|l| l.chars().count()).max() {
                if maxlen > MAX_LINE_CHARS {
                    continue;
                }
            }
            let toks = tokenize(&txt);
            if toks.is_empty() {
                continue;
            }
            files.push(rel.clone());
            ptoks.insert(rel.clone(), path_tokens(rel));
            let counts = counter_from_tokens(&toks);
            for term in counts.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            doclen.insert(rel.clone(), toks.len() as u32);
            tf.insert(rel.clone(), counts);

            if use_comments {
                let com_text = extract_comments(rel, &txt);
                let com_toks = tokenize(&com_text);
                if !com_toks.is_empty() {
                    let ctf = counter_from_tokens(&com_toks);
                    for term in ctf.keys() {
                        *com_df.entry(term.clone()).or_insert(0) += 1;
                    }
                    com_tf.insert(rel.clone(), ctf);
                }
            }

            if impl_prior(rel) == 1.0 {
                let def_re: Option<&LazyLock<Regex>> = if rel.ends_with(".py") {
                    Some(&PY_DEF_RE)
                } else if rel.ends_with(".go") {
                    Some(&GO_DEF_RE)
                } else if rel.ends_with(".rs") {
                    Some(&RS_DEF_RE)
                } else if rel.ends_with(".js") || rel.ends_with(".ts") || rel.ends_with(".jsx") || rel.ends_with(".tsx") {
                    Some(&JS_DEF_RE)
                } else {
                    None
                };
                if let Some(re) = def_re {
                    let mut syms: HashSet<String> = HashSet::new();
                    for cap in re.captures_iter(&txt) {
                        for gi in 1..cap.len() {
                            if let Some(g) = cap.get(gi) {
                                syms.insert(g.as_str().to_string());
                            }
                        }
                    }
                    for sym in syms {
                        def_index.entry(sym).or_default().push(rel.clone());
                    }
                }
            }

            text.insert(rel.clone(), txt);
        }

        let n_docs = files.len();
        let avg_len = if n_docs > 0 {
            doclen.values().map(|&v| v as f64).sum::<f64>() / n_docs as f64
        } else {
            1.0
        };
        let n_com_docs = com_tf.len();

        // commit-message field
        let mut msg_tf: HashMap<String, IndexMap<String, u32>> = HashMap::new();
        let mut msg_df: HashMap<String, u32> = HashMap::new();
        let mut msg_doclen: HashMap<String, u32> = HashMap::new();
        if let Some(hm) = history_msgs {
            if !hm.is_empty() {
                for rel in &files {
                    let msg = match hm.get(rel) {
                        Some(m) if !m.is_empty() => m,
                        _ => continue,
                    };
                    let mtoks = tokenize(msg);
                    if mtoks.is_empty() {
                        continue;
                    }
                    let mtf = counter_from_tokens(&mtoks);
                    for term in mtf.keys() {
                        *msg_df.entry(term.clone()).or_insert(0) += 1;
                    }
                    msg_doclen.insert(rel.clone(), mtoks.len() as u32);
                    msg_tf.insert(rel.clone(), mtf);
                }
            }
        }
        let n_msg_docs = msg_tf.len();
        let msg_avg_len = if n_msg_docs > 0 {
            msg_doclen.values().map(|&v| v as f64).sum::<f64>() / n_msg_docs as f64
        } else {
            1.0
        };

        // docs field
        let mut docs_files: Vec<String> = Vec::new();
        let mut docs_text: HashMap<String, String> = HashMap::new();
        let mut docs_tf: HashMap<String, IndexMap<String, u32>> = HashMap::new();
        let mut docs_df: HashMap<String, u32> = HashMap::new();
        let mut docs_len: HashMap<String, u32> = HashMap::new();
        if build_docs {
            let mut doc_paths: Vec<String> = Vec::new();
            for rel in &all_files {
                if rel.starts_with(".git/") || rel.contains("/.git/") {
                    continue;
                }
                let suf = suffix_of(rel);
                if !DOCS_EXTENSIONS.contains(&suf) {
                    continue;
                }
                if DOCS_EXCLUDE_RE.is_match(rel) {
                    continue;
                }
                doc_paths.push(rel.clone());
            }
            for rel in doc_paths.into_iter().take(MAX_DOCS_FILES) {
                let full = repo_path.join(&rel);
                let meta = match std::fs::metadata(&full) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if !meta.is_file() {
                    continue;
                }
                if meta.len() > MAX_DOCS_FILE_BYTES {
                    continue;
                }
                let txt = match read_text_lossy(&full) {
                    Some(t) => t,
                    None => continue,
                };
                let dtoks = tokenize(&txt);
                if dtoks.is_empty() {
                    continue;
                }
                let dcounts = counter_from_tokens(&dtoks);
                for term in dcounts.keys() {
                    *docs_df.entry(term.clone()).or_insert(0) += 1;
                }
                docs_len.insert(rel.clone(), dtoks.len() as u32);
                docs_tf.insert(rel.clone(), dcounts);
                docs_text.insert(rel.clone(), txt);
                docs_files.push(rel);
            }
        }
        let n_docs_files = docs_files.len();
        let docs_avg_len = if n_docs_files > 0 {
            docs_len.values().map(|&v| v as f64).sum::<f64>() / n_docs_files as f64
        } else {
            1.0
        };

        Corpus {
            repo_path: repo_path.to_path_buf(),
            files,
            text,
            ptoks,
            tf,
            doclen,
            df,
            use_comments,
            com_tf,
            com_df,
            def_index,
            n_docs,
            avg_len,
            n_com_docs,
            msg_tf,
            msg_df,
            msg_doclen,
            n_msg_docs,
            msg_avg_len,
            docs_files,
            docs_text,
            docs_tf,
            docs_df,
            docs_len,
            n_docs_files,
            docs_avg_len,
        }
    }

    // -------------------------------------------------------- incremental update
    //
    // Ports of `roust.core.Corpus.update_files` / `update_docs_files` (see
    // `cache.py`'s module docstring for the design). Both exist for
    // `crate::cache`'s incremental-update path (the common agent edit-loop
    // case: a file's CONTENT changed but its relpath set did not). Each
    // re-derives a modified file's contribution to the corpus from scratch
    // (subtract old, add new) using the identical per-file logic `build`
    // uses, so a successfully patched Corpus is observationally identical to
    // a fresh build over the same on-disk content. Neither method touches
    // `ptoks` (unchanged by a content-only edit) or the `msg_*` fields
    // (commit history is keyed on git HEAD, which incremental updates
    // require to be unchanged -- see `cache.rs`).
    //
    // Both are all-or-nothing: every file is pre-checked against `build`'s
    // own inclusion criteria BEFORE any mutation happens, so a `false`
    // return leaves the Corpus completely unmodified and the caller is free
    // to discard it and fall back to a full rebuild.

    /// Which definition-symbol regex (if any) applies to `rel`, by
    /// extension -- factored out of `build`'s per-file if/elif chain so
    /// `update_files` can reuse the identical mapping when
    /// subtracting/re-adding a modified file's `def_index` contributions.
    fn def_re_for(rel: &str) -> Option<&'static Regex> {
        if rel.ends_with(".py") {
            Some(&PY_DEF_RE)
        } else if rel.ends_with(".go") {
            Some(&GO_DEF_RE)
        } else if rel.ends_with(".rs") {
            Some(&RS_DEF_RE)
        } else if rel.ends_with(".js") || rel.ends_with(".ts") || rel.ends_with(".jsx") || rel.ends_with(".tsx") {
            Some(&JS_DEF_RE)
        } else {
            None
        }
    }

    fn def_syms(def_re: &Regex, text: &str) -> HashSet<String> {
        let mut syms = HashSet::new();
        for cap in def_re.captures_iter(text) {
            for gi in 1..cap.len() {
                if let Some(g) = cap.get(gi) {
                    syms.insert(g.as_str().to_string());
                }
            }
        }
        syms
    }

    /// Patch this Corpus in place for `rels` -- files already present in
    /// `self.files` whose on-disk content has changed. Re-reads each file
    /// directly from `self.repo_path` and applies exactly `build`'s per-file
    /// inclusion criteria (`MAX_FILE_BYTES`, `MAX_LINE_CHARS`, non-empty
    /// tokenization); if any file's new content fails a criterion (or can no
    /// longer be read), this is shaped like an add/remove and this method
    /// makes NO changes and returns `false` -- callers must fall back to a
    /// full rebuild. Returns `true`, having refreshed
    /// df/tf/doclen/text/def_index (and com_tf/com_df if use_comments) plus
    /// avg_len/n_com_docs, on full success.
    pub fn update_files(&mut self, rels: &[String]) -> bool {
        let mut new_text: HashMap<String, String> = HashMap::new();
        let mut new_toks: HashMap<String, Vec<String>> = HashMap::new();
        for rel in rels {
            let p = self.repo_path.join(rel);
            let meta = match std::fs::metadata(&p) {
                Ok(m) => m,
                Err(_) => return false,
            };
            if meta.len() > MAX_FILE_BYTES {
                return false;
            }
            let text = match read_text_lossy(&p) {
                Some(t) => t,
                None => return false,
            };
            let lines = py_splitlines(&text);
            if let Some(maxlen) = lines.iter().map(|l| l.chars().count()).max() {
                if maxlen > MAX_LINE_CHARS {
                    return false;
                }
            }
            let toks = tokenize(&text);
            if toks.is_empty() {
                return false;
            }
            new_text.insert(rel.clone(), text);
            new_toks.insert(rel.clone(), toks);
        }

        for rel in rels {
            // --- subtract old contributions (self.text[rel] is still old here)
            if let Some(old_tf) = self.tf.get(rel) {
                for term in old_tf.keys() {
                    if let Some(c) = self.df.get_mut(term) {
                        *c -= 1;
                        if *c == 0 {
                            self.df.remove(term);
                        }
                    }
                }
            }
            self.tf.remove(rel);
            self.doclen.remove(rel);
            if self.use_comments {
                if let Some(old_ctf) = self.com_tf.remove(rel) {
                    for term in old_ctf.keys() {
                        if let Some(c) = self.com_df.get_mut(term) {
                            *c -= 1;
                            if *c == 0 {
                                self.com_df.remove(term);
                            }
                        }
                    }
                }
            }
            let def_re = if impl_prior(rel) == 1.0 { Self::def_re_for(rel) } else { None };
            if let Some(re) = def_re {
                for sym in Self::def_syms(re, &self.text[rel]) {
                    if let Some(lst) = self.def_index.get_mut(&sym) {
                        lst.retain(|f| f != rel);
                    }
                }
            }

            // --- add new contributions
            let toks = &new_toks[rel];
            let counts = counter_from_tokens(toks);
            for term in counts.keys() {
                *self.df.entry(term.clone()).or_insert(0) += 1;
            }
            self.doclen.insert(rel.clone(), toks.len() as u32);
            self.tf.insert(rel.clone(), counts);
            self.text.insert(rel.clone(), new_text[rel].clone());
            if self.use_comments {
                let com_text = extract_comments(rel, &new_text[rel]);
                let com_toks = tokenize(&com_text);
                if !com_toks.is_empty() {
                    let ctf = counter_from_tokens(&com_toks);
                    for term in ctf.keys() {
                        *self.com_df.entry(term.clone()).or_insert(0) += 1;
                    }
                    self.com_tf.insert(rel.clone(), ctf);
                }
            }
            if let Some(re) = def_re {
                for sym in Self::def_syms(re, &new_text[rel]) {
                    self.def_index.entry(sym).or_default().push(rel.clone());
                }
            }
        }

        self.n_com_docs = self.com_tf.len();
        self.avg_len = if self.n_docs > 0 {
            self.doclen.values().map(|&v| v as f64).sum::<f64>() / self.n_docs as f64
        } else {
            1.0
        };
        true
    }

    /// Analogous to `update_files` but for the docs field (`*.rst`/`*.txt`/
    /// `*.md` pages collected when this Corpus was built with
    /// `build_docs=true`). Every rel must already be a member of
    /// `self.docs_files`. Returns `false` (no changes) if any file's new
    /// content would flip its `build` inclusion verdict (now oversized, or
    /// now tokenizes to nothing) or can no longer be read -- callers must
    /// fall back to a full rebuild.
    pub fn update_docs_files(&mut self, rels: &[String]) -> bool {
        let mut new_text: HashMap<String, String> = HashMap::new();
        let mut new_toks: HashMap<String, Vec<String>> = HashMap::new();
        for rel in rels {
            let p = self.repo_path.join(rel);
            let meta = match std::fs::metadata(&p) {
                Ok(m) => m,
                Err(_) => return false,
            };
            if meta.len() > MAX_DOCS_FILE_BYTES {
                return false;
            }
            let text = match read_text_lossy(&p) {
                Some(t) => t,
                None => return false,
            };
            let toks = tokenize(&text);
            if toks.is_empty() {
                return false;
            }
            new_text.insert(rel.clone(), text);
            new_toks.insert(rel.clone(), toks);
        }

        for rel in rels {
            if let Some(old_tf) = self.docs_tf.get(rel) {
                for term in old_tf.keys() {
                    if let Some(c) = self.docs_df.get_mut(term) {
                        *c -= 1;
                        if *c == 0 {
                            self.docs_df.remove(term);
                        }
                    }
                }
            }
            let toks = &new_toks[rel];
            let counts = counter_from_tokens(toks);
            for term in counts.keys() {
                *self.docs_df.entry(term.clone()).or_insert(0) += 1;
            }
            self.docs_len.insert(rel.clone(), toks.len() as u32);
            self.docs_tf.insert(rel.clone(), counts);
            self.docs_text.insert(rel.clone(), new_text[rel].clone());
        }

        self.docs_avg_len = if self.n_docs_files > 0 {
            self.docs_len.values().map(|&v| v as f64).sum::<f64>() / self.n_docs_files as f64
        } else {
            1.0
        };
        true
    }

    /// BM25F-style: body field (Okapi) + path field (binary match, weighted),
    /// multiplied by the implementation-file document prior, plus an
    /// optional comment/NL field term. See lanes2.py's `Corpus.bm25`
    /// docstring. The returned map's *insertion order* is load-bearing: it
    /// is later fed through `_normalize` and a stable `sorted()` whose
    /// tie-break is this order, so it must match Python's dict-insertion
    /// order exactly (see PARITY_NOTES.md).
    pub fn bm25(&self, terms: &[String]) -> IndexMap<String, f64> {
        self.bm25_params(terms, 1.2, 0.75, 2.5, true, 0.5)
    }

    pub fn bm25_params(
        &self,
        terms: &[String],
        k1: f64,
        b: f64,
        path_weight: f64,
        use_prior: bool,
        comment_weight: f64,
    ) -> IndexMap<String, f64> {
        let mut scores: IndexMap<String, f64> = IndexMap::new();
        for term in terms {
            if let Some(&dfv) = self.df.get(term) {
                let idf = (1.0 + (self.n_docs as f64 - dfv as f64 + 0.5) / (dfv as f64 + 0.5)).ln();
                for rel in &self.files {
                    if let Some(tfv) = self.tf.get(rel).and_then(|m| m.get(term)) {
                        let tfv = *tfv as f64;
                        let doclen = *self.doclen.get(rel).unwrap_or(&0) as f64;
                        let denom = tfv + k1 * (1.0 - b + b * doclen / self.avg_len);
                        *scores.entry(rel.clone()).or_insert(0.0) += idf * (tfv * (k1 + 1.0) / denom);
                    }
                }
                for rel in &self.files {
                    if self.ptoks.get(rel).map(|s| s.contains(term)).unwrap_or(false) {
                        *scores.entry(rel.clone()).or_insert(0.0) += path_weight * idf;
                    }
                }
            }
            if self.use_comments && !self.com_tf.is_empty() && self.n_com_docs > 0 {
                if let Some(&cdf) = self.com_df.get(term) {
                    let idf_com = (1.0 + (self.n_com_docs as f64 - cdf as f64 + 0.5) / (cdf as f64 + 0.5)).ln();
                    // Drive iteration via corpus.files (not com_tf.items()
                    // directly) -- com_tf's insertion order is a subset of
                    // corpus.files order anyway (built in the same
                    // per-file loop), so this reproduces Python's direct
                    // `self.com_tf.items()` iteration order exactly.
                    for rel in &self.files {
                        if let Some(ctf_counter) = self.com_tf.get(rel) {
                            if let Some(&ctf) = ctf_counter.get(term) {
                                let ctf = ctf as f64;
                                *scores.entry(rel.clone()).or_insert(0.0) +=
                                    comment_weight * idf_com * (ctf * (k1 + 1.0) / (ctf + k1));
                            }
                        }
                    }
                }
            }
        }
        if use_prior {
            scores.into_iter().map(|(rel, s)| (rel.clone(), s * impl_prior(&rel))).collect()
        } else {
            scores
        }
    }

    /// Standalone Okapi BM25 over the commit-message field only.
    pub fn msg_bm25(&self, terms: &[String]) -> IndexMap<String, f64> {
        self.msg_bm25_params(terms, 1.2, 0.5, true)
    }

    pub fn msg_bm25_params(&self, terms: &[String], k1: f64, b: f64, use_prior: bool) -> IndexMap<String, f64> {
        if self.msg_tf.is_empty() || self.n_msg_docs == 0 {
            return IndexMap::new();
        }
        let mut scores: IndexMap<String, f64> = IndexMap::new();
        for term in terms {
            let mdf = match self.msg_df.get(term) {
                Some(&v) => v,
                None => continue,
            };
            let idf = (1.0 + (self.n_msg_docs as f64 - mdf as f64 + 0.5) / (mdf as f64 + 0.5)).ln();
            for rel in &self.files {
                if let Some(mtf_counter) = self.msg_tf.get(rel) {
                    if let Some(&mtf) = mtf_counter.get(term) {
                        let mtf = mtf as f64;
                        let doclen = *self.msg_doclen.get(rel).unwrap_or(&0) as f64;
                        let denom = mtf + k1 * (1.0 - b + b * doclen / self.msg_avg_len);
                        *scores.entry(rel.clone()).or_insert(0.0) += idf * (mtf * (k1 + 1.0) / denom);
                    }
                }
            }
        }
        if use_prior {
            scores.into_iter().map(|(rel, s)| (rel.clone(), s * impl_prior(&rel))).collect()
        } else {
            scores
        }
    }

    /// Standard Okapi BM25 over the docs field.
    pub fn docs_bm25(&self, terms: &[String]) -> IndexMap<String, f64> {
        if self.docs_tf.is_empty() || self.n_docs_files == 0 {
            return IndexMap::new();
        }
        let (k1, b) = (1.2, 0.75);
        let mut scores: IndexMap<String, f64> = IndexMap::new();
        for term in terms {
            let ddf = match self.docs_df.get(term) {
                Some(&v) => v,
                None => continue,
            };
            let idf = (1.0 + (self.n_docs_files as f64 - ddf as f64 + 0.5) / (ddf as f64 + 0.5)).ln();
            for rel in &self.docs_files {
                if let Some(dtf_counter) = self.docs_tf.get(rel) {
                    if let Some(&dtf) = dtf_counter.get(term) {
                        let dtf = dtf as f64;
                        let doclen = *self.docs_len.get(rel).unwrap_or(&0) as f64;
                        let denom = dtf + k1 * (1.0 - b + b * doclen / self.docs_avg_len);
                        *scores.entry(rel.clone()).or_insert(0.0) += idf * (dtf * (k1 + 1.0) / denom);
                    }
                }
            }
        }
        scores
    }
}

// ---------------------------------------------------------------- definition-symbol anchors

static ANCHOR_IDENT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]{3,}").unwrap());
static CODE_SPAN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```.*?```|`[^`\n]+`").unwrap());

/// Definition-symbol anchor channel. See lanes2.py's `extract_symbol_anchors`
/// docstring. Fully deterministic (no raw-`set`-iteration hazard): `order`
/// is regex-match order over the question text, and `corpus.def_index[s]`
/// preserves corpus.files traversal order (each file contributes at most
/// one append per symbol, regardless of the per-file symbol dedup set's
/// internal order).
pub fn extract_symbol_anchors(question: &str, corpus: &Corpus) -> Vec<(String, f64)> {
    let code_spans: Vec<(usize, usize)> = CODE_SPAN_RE.find_iter(question).map(|m| (m.start(), m.end())).collect();
    let in_code = |pos: usize| code_spans.iter().any(|&(a, b)| a <= pos && pos < b);

    let mut occurrences: IndexMap<String, Vec<usize>> = IndexMap::new();
    for m in ANCHOR_IDENT_RE.find_iter(question) {
        occurrences.entry(m.as_str().to_string()).or_default().push(m.start());
    }
    let order: Vec<String> = occurrences.keys().cloned().collect();

    let mut best: IndexMap<String, f64> = IndexMap::new();
    let mut def_counts: HashMap<String, usize> = HashMap::new();
    for s in &order {
        // Python: `if s.lower() in _STOP: continue`
        if STOP.contains(py_lower(s).as_str()) {
            continue;
        }
        let files = match corpus.def_index.get(s) {
            Some(f) if f.len() <= 3 => f,
            _ => continue,
        };
        let occ = occurrences.get(s).unwrap();
        let strength_base = if occ.iter().any(|&p| in_code(p)) { 2.0 } else { 1.0 };
        let strength = if *s != py_lower(s) || s.contains('_') {
            strength_base + 0.5
        } else {
            strength_base
        };
        for f in files {
            let cur = best.get(f).copied().unwrap_or(-1.0);
            if strength > cur {
                best.insert(f.clone(), strength);
                def_counts.insert(f.clone(), files.len());
            }
        }
    }
    let mut result: Vec<(String, f64)> = best.into_iter().collect();
    // sorted(key=(strength, -def_count), reverse=True); stable sort, ties
    // broken by `best`'s insertion order (preserved since `result` is built
    // by iterating `best` in insertion order).
    result.sort_by(|a, b| {
        let ka = (a.1, -(*def_counts.get(&a.0).unwrap() as i64));
        let kb = (b.1, -(*def_counts.get(&b.0).unwrap() as i64));
        // f64::total_cmp rather than partial_cmp().unwrap(): never panics
        // regardless of NaN/inf, and agrees with partial_cmp on every
        // finite, non-NaN input (all `strength` values here are one of
        // 1.0/1.5/2.0/2.5, so this is a pure hardening, not a ranking
        // change). See `pack_regions`' pass-2 `marginal`-based sort, which
        // gets the same total_cmp hardening for the same reason: a caller
        // can pass an already-NaN score into either function (e.g. via a
        // drifted idf input upstream), and partial_cmp().unwrap() panics
        // the instant that NaN is compared against anything.
        kb.0.total_cmp(&ka.0).then_with(|| kb.1.cmp(&ka.1))
    });
    result
}

/// Best-effort recovery of WHICH rarity-gated definition symbol(s) (see
/// `extract_symbol_anchors` above) anchored each of `files` into the ranked
/// file list, for `pack_regions`' channel-aware packing: an anchor-selected
/// file's packed regions should include the anchored symbol's own
/// definition block, not just whatever region wins on generic term
/// density.
///
/// Deliberately a separate, independent pass over the question text rather
/// than a refactor of `extract_symbol_anchors` -- that function's return
/// value is parity-pinned. This helper may return MULTIPLE symbols per
/// file (in question-regex-match order, which `pack_regions` consumes
/// first-match-wins) and is never consumed by `select_files`, so nothing
/// here can affect the ranked file list. Applies the identical rarity gate
/// (<=3 defining files) so it only ever surfaces symbols that
/// `extract_symbol_anchors` itself would have considered.
pub fn anchor_def_symbols(
    question: &str,
    corpus: &Corpus,
    files: &HashSet<String>,
) -> IndexMap<String, Vec<String>> {
    let mut out: IndexMap<String, Vec<String>> = IndexMap::new();
    if files.is_empty() {
        return out;
    }
    for m in ANCHOR_IDENT_RE.find_iter(question) {
        let s = m.as_str();
        if STOP.contains(py_lower(s).as_str()) {
            continue;
        }
        let def_files = match corpus.def_index.get(s) {
            Some(f) if !f.is_empty() && f.len() <= 3 => f,
            _ => continue,
        };
        for f in def_files {
            if files.contains(f) {
                let entry = out.entry(f.clone()).or_default();
                if !entry.iter().any(|x| x == s) {
                    entry.push(s.to_string());
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------- import graph

static PY_FROM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*from\s+([\w.]+)\s+import\s+(\([^)]*\)|[^\n]+)").unwrap());
static PY_PLAIN_IMPORT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*import\s+([\w., ]+)").unwrap());
static JS_IMPORT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:from\s+['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)|import\s*\(\s*['"]([^'"]+)['"])"#).unwrap()
});
static RS_MOD_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*(?:pub\s+)?mod\s+(\w+)\s*;").unwrap());
static RS_USE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*(?:pub\s+)?use\s+(?:crate|super|self)::([\w:]+)").unwrap());
static GO_IMPORT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""([\w./\-]+)""#).unwrap());

fn py_module_index(files: &[String]) -> HashMap<String, String> {
    let mut idx = HashMap::new();
    for rel in files {
        if !rel.ends_with(".py") {
            continue;
        }
        let mod_name = rel[..rel.len() - 3].replace('/', ".");
        idx.insert(mod_name.clone(), rel.clone());
        if let Some(stripped) = mod_name.strip_suffix(".__init__") {
            idx.insert(stripped.to_string(), rel.clone());
        }
    }
    idx
}

/// Undirected import graph. Adjacency sets use `BTreeSet` (sorted
/// iteration) as the deliberate, documented deterministic stand-in for
/// Python's raw `set[str]` (see PARITY_NOTES.md item 2: `edges[s]`'s
/// iteration order is the one genuinely hash-randomization-exposed spot in
/// the whole pipeline).
pub type EdgeMap = HashMap<String, BTreeSet<String>>;

/// The set of files that `rel`'s OWN text authors an import edge to
/// (pre-symmetrization) -- i.e. the per-file body of `build_import_graph`'s
/// main loop, factored out so `crate::cache`'s incremental-update path
/// (`update_import_graph_for_files`) can recompute a single changed file's
/// authored edges without re-parsing the whole corpus. Must stay exactly in
/// sync with `build_import_graph`'s loop body -- best-effort per language,
/// unresolved imports ignored.
fn file_import_targets(
    rel: &str,
    text: &str,
    pyidx: &HashMap<String, String>,
    fileset: &HashSet<&String>,
) -> HashSet<String> {
    let mut targets: HashSet<String> = HashSet::new();

    let resolve_py_module = |module: &str| -> String {
        if !module.starts_with('.') {
            return module.to_string();
        }
        let level = module.len() - module.trim_start_matches('.').len();
        let rest = module.trim_start_matches('.');
        let parent = py_parent(rel);
        let mut pkg_parts: Vec<&str> = if parent == "." { Vec::new() } else { parent.split('/').collect() };
        if level > 1 {
            let keep = pkg_parts.len().saturating_sub(level - 1);
            pkg_parts.truncate(keep);
        }
        let mut all: Vec<&str> = pkg_parts;
        if !rest.is_empty() {
            all.extend(rest.split('.'));
        }
        all.join(".")
    };

    let add_module = |targets: &mut HashSet<String>, module: &str| {
        let parts: Vec<&str> = module.split('.').filter(|p| !p.is_empty()).collect();
        for i in (1..=parts.len()).rev() {
            let key = parts[..i].join(".");
            if let Some(hit) = pyidx.get(&key) {
                if hit != rel {
                    targets.insert(hit.clone());
                }
                return;
            }
        }
    };

    if rel.ends_with(".py") {
        for cap in PY_FROM_RE.captures_iter(text) {
            let module_spec = cap.get(1).unwrap().as_str();
            let module = resolve_py_module(module_spec);
            add_module(&mut targets, &module);
            let names_blob = cap.get(2).unwrap().as_str();
            let trimmed = names_blob.trim_matches(|c| c == '(' || c == ')');
            for name_raw in trimmed.replace('\n', " ").split(',') {
                let name = name_raw.trim();
                let name = name.split(" as ").next().unwrap_or(name).trim();
                let name = name.trim_matches(|c: char| c == '*' || c == '#' || c == ' ' || c == '\t');
                if !name.is_empty() && !name.contains('.') {
                    let sub_key = format!("{module}.{name}");
                    if let Some(sub) = pyidx.get(&sub_key) {
                        if sub != rel {
                            targets.insert(sub.clone());
                        }
                    }
                }
            }
        }
        for cap in PY_PLAIN_IMPORT_RE.captures_iter(text) {
            let spec_list = cap.get(1).unwrap().as_str();
            for spec in spec_list.split(',') {
                let module = spec.trim().split(" as ").next().unwrap_or("").trim();
                if !module.is_empty() {
                    add_module(&mut targets, module);
                }
            }
        }
    } else if rel.ends_with(".js") || rel.ends_with(".ts") || rel.ends_with(".jsx") || rel.ends_with(".tsx") {
        let base = py_parent(rel);
        for cap in JS_IMPORT_RE.captures_iter(text) {
            let spec = (1..=3).find_map(|i| cap.get(i)).map(|m| m.as_str()).unwrap_or("");
            if !spec.starts_with('.') {
                continue;
            }
            let cand = normpath_join(base, spec);
            let suffixes = ["", ".js", ".ts", ".jsx", ".tsx", "/index.js", "/index.ts"];
            for suffix in suffixes {
                let candidate = format!("{cand}{suffix}");
                if fileset.contains(&candidate) {
                    if candidate != rel {
                        targets.insert(candidate);
                    }
                    break;
                }
            }
        }
    } else if rel.ends_with(".rs") {
        let base = py_parent(rel);
        for cap in RS_MOD_RE.captures_iter(text) {
            let name = cap.get(1).unwrap().as_str();
            for cand in [path_join_simple(base, &format!("{name}.rs")), path_join_simple(base, &format!("{name}/mod.rs"))] {
                if fileset.contains(&cand) && cand != rel {
                    targets.insert(cand);
                }
            }
        }
        for cap in RS_USE_RE.captures_iter(text) {
            let head = cap.get(1).unwrap().as_str().split("::").next().unwrap_or("");
            for cand in [
                path_join_simple(base, &format!("{head}.rs")),
                path_join_simple(base, &format!("{head}/mod.rs")),
                format!("src/{head}.rs"),
                format!("src/{head}/mod.rs"),
            ] {
                if fileset.contains(&cand) && cand != rel {
                    targets.insert(cand);
                }
            }
        }
    } else if rel.ends_with(".go") {
        for cap in GO_IMPORT_RE.captures_iter(text) {
            let pkg = cap.get(1).unwrap().as_str();
            let tail = pkg.rsplit('/').next().unwrap_or(pkg);
            for other in fileset {
                if other.as_str() != rel && other.ends_with(".go") && py_parent_name(other) == tail {
                    targets.insert((*other).clone());
                }
            }
        }
    }
    targets
}

pub fn build_import_graph(corpus: &Corpus) -> EdgeMap {
    let mut edges: EdgeMap = HashMap::new();
    let pyidx = py_module_index(&corpus.files);
    let fileset: HashSet<&String> = corpus.files.iter().collect();

    for rel in &corpus.files {
        let text = &corpus.text[rel];
        let targets = file_import_targets(rel, text, &pyidx, &fileset);
        for t in targets {
            edges.entry(rel.clone()).or_default().insert(t.clone());
            edges.entry(t).or_default().insert(rel.clone());
        }
    }
    edges
}

/// Incrementally patch `edges` (mutated in place) for a batch of files whose
/// content changed but whose relpath set is unchanged -- see
/// `crate::cache`'s incremental-update path. `old_text` is `{rel: text}`
/// holding each changed file's PRE-edit text; `corpus.text[rel]` must
/// already hold the POST-edit text for every rel in `old_text` by the time
/// this is called (see `Corpus::update_files`, which must run first).
///
/// Recomputes each changed file's own authored-edge set
/// (`file_import_targets`) from both its old and new text. An edge
/// `(rel, t)` is removed only if `rel` was its SOLE author -- `t`'s own
/// current text, re-checked on demand, doesn't independently author it
/// back -- so an edge created by an UNCHANGED file Y importing a changed
/// file `rel` is left untouched even though `rel`'s content changed, since
/// Y's text (the sole source of that edge) didn't.
pub fn update_import_graph_for_files(corpus: &Corpus, edges: &mut EdgeMap, old_text: &HashMap<String, String>) {
    let fileset: HashSet<&String> = corpus.files.iter().collect();
    let pyidx = py_module_index(&corpus.files);

    let authored = |rel: &str, text: &str| -> HashSet<String> { file_import_targets(rel, text, &pyidx, &fileset) };

    let mut new_authored: HashMap<String, HashSet<String>> = HashMap::new();
    let mut old_authored: HashMap<String, HashSet<String>> = HashMap::new();
    for rel in old_text.keys() {
        if let Some(text) = corpus.text.get(rel) {
            new_authored.insert(rel.clone(), authored(rel, text));
        }
    }
    for (rel, text) in old_text {
        old_authored.insert(rel.clone(), authored(rel, text));
    }

    let other_authors = |t: &str, rel: &str| -> bool {
        if old_text.contains_key(t) {
            new_authored.get(t).map(|s| s.contains(rel)).unwrap_or(false)
        } else {
            match corpus.text.get(t) {
                Some(text) => authored(t, text).contains(rel),
                None => false,
            }
        }
    };

    let mut touched: HashSet<String> = old_text.keys().cloned().collect();
    for rel in old_text.keys() {
        let empty = HashSet::new();
        let old_set = old_authored.get(rel).unwrap_or(&empty);
        let new_set = new_authored.get(rel).unwrap_or(&empty);
        let removed: Vec<String> = old_set.difference(new_set).cloned().collect();
        let added: Vec<String> = new_set.difference(old_set).cloned().collect();
        touched.extend(removed.iter().cloned());
        touched.extend(added.iter().cloned());
        for t in &removed {
            if !other_authors(t, rel) {
                if let Some(s) = edges.get_mut(rel) {
                    s.remove(t);
                }
                if let Some(s) = edges.get_mut(t) {
                    s.remove(rel);
                }
            }
        }
        for t in &added {
            edges.entry(rel.clone()).or_default().insert(t.clone());
            edges.entry(t.clone()).or_default().insert(rel.clone());
        }
    }

    for k in &touched {
        if edges.get(k).map(|s| s.is_empty()).unwrap_or(false) {
            edges.remove(k);
        }
    }
}

/// Random walk with restart -- present for structural parity with
/// lanes2.py's `personalized_pagerank`, but note it is DEAD CODE relative
/// to the actual CLI/driver wiring: `select_files(use_ppr=True)` never
/// calls it (its own "structural expansion" block reimplements a different
/// additive scheme directly). Kept for completeness only, never invoked.
#[allow(dead_code)]
pub fn personalized_pagerank(
    seeds: &IndexMap<String, f64>,
    edges: &EdgeMap,
    same_dir: &HashMap<String, Vec<String>>,
    alpha: f64,
    iters: usize,
    same_dir_weight: f64,
) -> IndexMap<String, f64> {
    let total: f64 = seeds.values().sum();
    if total <= 0.0 {
        return IndexMap::new();
    }
    let restart: IndexMap<String, f64> = seeds.iter().map(|(k, v)| (k.clone(), v / total)).collect();
    let mut rank: IndexMap<String, f64> = restart.clone();
    for _ in 0..iters {
        let mut nxt: IndexMap<String, f64> = IndexMap::new();
        for (node, &mass) in &rank {
            if mass <= 1e-12 {
                continue;
            }
            let empty_set = BTreeSet::new();
            let nbrs = edges.get(node).unwrap_or(&empty_set);
            let empty_vec = Vec::new();
            let dir_nbrs = same_dir.get(py_parent(node)).unwrap_or(&empty_vec);
            let mut weights: Vec<(String, f64)> = nbrs.iter().map(|n| (n.clone(), 1.0)).collect();
            for n in dir_nbrs {
                if n != node && !nbrs.contains(n) {
                    weights.push((n.clone(), same_dir_weight));
                }
            }
            let wsum: f64 = weights.iter().map(|(_, w)| w).sum();
            if wsum <= 0.0 {
                *nxt.entry(node.clone()).or_insert(0.0) += (1.0 - alpha) * mass;
            } else {
                for (n, w) in weights {
                    *nxt.entry(n).or_insert(0.0) += (1.0 - alpha) * mass * (w / wsum);
                }
            }
        }
        for (k, v) in &restart {
            *nxt.entry(k.clone()).or_insert(0.0) += alpha * v;
        }
        rank = nxt;
    }
    rank
}

// ---------------------------------------------------------------- selection

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct Explain {
    pub lex_picks: Vec<String>,
    pub sources: Vec<String>,
    pub pool: Vec<(String, f64, f64)>,
    pub additions: Vec<String>,
    pub cochange_additions: Vec<String>,
    pub msg_additions: Vec<String>,
    pub anchor_promotions: Vec<(String, f64, String, String)>,
    pub testbridge: Vec<(String, String, String)>,
    pub docsbridge: Vec<(String, String, i64)>,
    /// Raw (pre-normalization) top pooled BM25F candidate score for this
    /// query -- i.e. `max(corpus.bm25(terms).values())`, computed before
    /// `normalize()` divides every score down to a [0,1] range and before
    /// `pack_regions` does any budget-driven packing. Unlike the
    /// normalized/packed scores exposed elsewhere, this is comparable
    /// query-to-query and corpus-to-corpus, which is exactly what makes it
    /// usable as a low-confidence calibration signal (see `main.rs`'s
    /// `low_confidence` gate): a genuinely weak/coincidental match still
    /// normalizes its best candidate to 1.0, but its raw top score stays
    /// small. Zero (the `Default` value) in the true no-match case, where
    /// `select_files` returns `Explain::default()` before this field would
    /// otherwise be set.
    pub top_score: f64,
}

fn normalize(scores: &IndexMap<String, f64>) -> IndexMap<String, f64> {
    if scores.is_empty() {
        return IndexMap::new();
    }
    let mx = scores.values().cloned().fold(f64::MIN, f64::max);
    if mx > 0.0 {
        scores.iter().map(|(k, v)| (k.clone(), v / mx)).collect()
    } else {
        scores.clone()
    }
}

/// Python `round()` (banker's rounding / round-half-to-even), used only in
/// Explain diagnostics.
fn py_round(x: f64, ndigits: i32) -> f64 {
    let factor = 10f64.powi(ndigits);
    let scaled = x * factor;
    let floor = scaled.floor();
    let diff = scaled - floor;
    let rounded = if (diff - 0.5).abs() < 1e-9 {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        scaled.round()
    };
    rounded / factor
}

fn apply_anchor_promotions(
    out: Vec<String>,
    anchors: Option<&[(String, f64)]>,
) -> (Vec<String>, Vec<(String, f64, String, String)>) {
    let anchors = match anchors {
        Some(a) if !a.is_empty() => a,
        _ => return (out, Vec::new()),
    };
    let mut promotions: Vec<(String, f64, String, String)> = Vec::new();
    let mut out = out;

    let mut head_files: Vec<String> = Vec::new();
    let mut to_remove: HashSet<String> = HashSet::new();
    for (f, strength) in anchors {
        if *strength < 2.0 || head_files.len() >= 2 || head_files.contains(f) {
            continue;
        }
        if let Some(idx) = out.iter().position(|x| x == f) {
            if idx >= 10 {
                head_files.push(f.clone());
                to_remove.insert(f.clone());
                promotions.push((f.clone(), *strength, "move".into(), "head".into()));
            }
        } else {
            head_files.push(f.clone());
            promotions.push((f.clone(), *strength, "insert".into(), "head".into()));
        }
    }
    if !head_files.is_empty() {
        let remaining: Vec<String> = out.into_iter().filter(|f| !to_remove.contains(f)).collect();
        let split = remaining.len().min(7);
        let mut new_out = remaining[..split].to_vec();
        new_out.extend(head_files.iter().cloned());
        new_out.extend(remaining[split..].to_vec());
        out = new_out;
    }

    let mut tail_files: Vec<String> = Vec::new();
    for (f, strength) in anchors {
        if *strength >= 2.0 || tail_files.len() >= 2 || tail_files.contains(f) || head_files.contains(f) {
            continue;
        }
        if !out.contains(f) {
            tail_files.push(f.clone());
            promotions.push((f.clone(), *strength, "insert".into(), "tail".into()));
        }
    }
    for f in &tail_files {
        let pos = out.len().min(12);
        out.insert(pos, f.clone());
    }

    if head_files.is_empty() && tail_files.is_empty() {
        return (out, Vec::new());
    }
    (out, promotions)
}

const TESTBRIDGE_EXTS: &[&str] = &[".py", ".go", ".rs", ".js", ".ts"];

fn apply_testbridge_promotions(
    out: Vec<String>,
    corpus: &Corpus,
    bm: &IndexMap<String, f64>,
    edges: &EdgeMap,
) -> (Vec<String>, Vec<(String, String, String)>) {
    let testlike: Vec<String> = corpus
        .files
        .iter()
        .filter(|f| TESTLIKE_RE.is_match(f) && TESTBRIDGE_EXTS.contains(&suffix_of(f)))
        .cloned()
        .collect();
    let testlike_set: HashSet<String> = testlike.iter().cloned().collect();

    let mut ranked_tests: Vec<(String, f64)> = testlike.iter().map(|f| (f.clone(), bm.get(f).copied().unwrap_or(0.0))).collect();
    ranked_tests.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top_tests: Vec<(String, f64)> = ranked_tests.into_iter().filter(|(_, s)| *s > 0.0).take(3).collect();
    if top_tests.is_empty() {
        return (out, Vec::new());
    }
    let top_score = top_tests[0].1;
    if top_score <= 0.0 {
        return (out, Vec::new());
    }

    let mut candidates: IndexMap<String, (f64, String)> = IndexMap::new();
    let empty = BTreeSet::new();
    for (test, tscore) in &top_tests {
        for nbr in edges.get(test).unwrap_or(&empty) {
            if impl_prior(nbr) != 1.0 {
                continue;
            }
            let strength = tscore / top_score;
            let replace = match candidates.get(nbr) {
                None => true,
                Some((cur, _)) => strength > *cur,
            };
            if replace {
                candidates.insert(nbr.clone(), (strength, test.clone()));
            }
        }
    }

    let specificity = |f: &str| -> f64 {
        let strength = candidates.get(f).unwrap().0;
        let n_test_importers = edges
            .get(f)
            .map(|s| s.iter().filter(|x| testlike_set.contains(x.as_str())).count())
            .unwrap_or(0);
        strength / (2.0 + n_test_importers as f64).ln()
    };

    let mut tail_pool: Vec<String> = candidates.keys().cloned().collect();
    tail_pool.sort_by(|a, b| specificity(b).total_cmp(&specificity(a)).then_with(|| a.cmp(b)));

    let mut records: Vec<(String, String, String)> = Vec::new();
    let mut tail_files: Vec<String> = Vec::new();
    let mut out = out;
    for f in &tail_pool {
        if tail_files.len() >= 3 {
            break;
        }
        if out.contains(f) {
            continue;
        }
        tail_files.push(f.clone());
        records.push((f.clone(), "tail".into(), candidates.get(f).unwrap().1.clone()));
    }
    for f in &tail_files {
        let pos = out.len().min(14);
        out.insert(pos, f.clone());
    }

    (out, records)
}

static DOTTED_PATH_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){2,}\b").unwrap());
static SPHINX_DIRECTIVE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:automodule|currentmodule|module|autoclass|autofunction)::\s*([\w.]+)").unwrap());

fn resolve_py_dotted(dotted: &str, pyidx: &HashMap<String, String>) -> Option<String> {
    let parts: Vec<&str> = dotted.split('.').filter(|p| !p.is_empty()).collect();
    for i in (1..=parts.len()).rev() {
        let key = parts[..i].join(".");
        if let Some(hit) = pyidx.get(&key) {
            return Some(hit.clone());
        }
    }
    None
}

fn apply_docsbridge_promotions(
    out: Vec<String>,
    corpus: &Corpus,
    terms: &[String],
) -> (Vec<String>, Vec<(String, String, i64)>) {
    if corpus.docs_tf.is_empty() {
        return (out, Vec::new());
    }
    let doc_scores = corpus.docs_bm25(terms);
    if doc_scores.is_empty() {
        return (out, Vec::new());
    }
    let mut ranked_pages: Vec<(String, f64)> = doc_scores.into_iter().collect();
    ranked_pages.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let top_pages: Vec<(String, f64)> = ranked_pages.into_iter().filter(|(_, s)| *s > 0.0).take(3).collect();
    if top_pages.is_empty() {
        return (out, Vec::new());
    }

    let pyidx = py_module_index(&corpus.files);
    let mut candidates: IndexMap<String, (i64, i64)> = IndexMap::new(); // (n_pages, best_rank)
    for (rank, (page, _score)) in top_pages.iter().enumerate() {
        let text = &corpus.docs_text[page];
        let mut refs: HashSet<String> = HashSet::new();
        for m in DOTTED_PATH_RE.find_iter(text) {
            refs.insert(m.as_str().to_string());
        }
        for cap in SPHINX_DIRECTIVE_RE.captures_iter(text) {
            refs.insert(cap.get(1).unwrap().as_str().to_string());
        }
        let mut resolved: HashSet<String> = HashSet::new();
        for r in &refs {
            if let Some(hit) = resolve_py_dotted(r, &pyidx) {
                if impl_prior(&hit) == 1.0 {
                    resolved.insert(hit);
                }
            }
        }
        for f in resolved {
            let entry = candidates.entry(f).or_insert((0, rank as i64));
            entry.0 += 1;
            entry.1 = entry.1.min(rank as i64);
        }
    }

    let mut tail_pool: Vec<String> = candidates.keys().cloned().collect();
    tail_pool.sort_by(|a, b| {
        let ca = candidates.get(a).unwrap();
        let cb = candidates.get(b).unwrap();
        cb.0.cmp(&ca.0).then(ca.1.cmp(&cb.1)).then(a.cmp(b))
    });

    let mut records: Vec<(String, String, i64)> = Vec::new();
    let mut tail_files: Vec<String> = Vec::new();
    let mut out = out;
    for f in &tail_pool {
        if tail_files.len() >= 2 {
            break;
        }
        if out.contains(f) {
            continue;
        }
        tail_files.push(f.clone());
        records.push((f.clone(), "tail".into(), candidates.get(f).unwrap().0));
    }
    for f in &tail_files {
        let pos = out.len().min(16);
        out.insert(pos, f.clone());
    }

    (out, records)
}

pub struct SelectParams<'a> {
    pub k_lex: usize,
    pub floor_ratio: f64,
    pub cochange: Option<&'a IndexMap<String, IndexMap<String, i64>>>,
    pub cochange_strong: i64,
    pub anchors: Option<&'a [(String, f64)]>,
    pub use_testbridge: bool,
    pub use_docsbridge: bool,
}

impl<'a> Default for SelectParams<'a> {
    fn default() -> Self {
        SelectParams {
            k_lex: 10,
            floor_ratio: 0.05,
            cochange: None,
            cochange_strong: 5,
            anchors: None,
            use_testbridge: false,
            use_docsbridge: false,
        }
    }
}

/// Return candidate files: top BM25F picks, UNIONed with structural-
/// expansion additions when `use_ppr` is true. See lanes2.py's
/// `select_files` docstring for the full design rationale -- this is a
/// direct translation; the one deliberate behavioral choice (not a
/// deviation, a documented tie-break for an underlying Python
/// nondeterminism) is noted at the `edges.get(s)` call below. See
/// PARITY_NOTES.md item 2.
pub fn select_files(
    corpus: &Corpus,
    terms: &[String],
    use_ppr: bool,
    params: &SelectParams,
) -> (Vec<String>, IndexMap<String, f64>, Explain) {
    let bm = corpus.bm25(terms);
    if bm.is_empty() {
        return (Vec::new(), IndexMap::new(), Explain::default());
    }
    let top_score = bm.values().cloned().fold(0.0_f64, f64::max);
    let bm_n = normalize(&bm);
    let mut ranked: Vec<(String, f64)> = bm_n.iter().map(|(k, v)| (k.clone(), *v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let best = ranked[0].1;
    let lex_picks: Vec<String> = ranked
        .iter()
        .take(params.k_lex)
        .enumerate()
        .filter(|(i, (_, s))| *i < 3 || *s >= params.floor_ratio * best)
        .map(|(_, (f, _))| f.clone())
        .collect();

    let mut scores: IndexMap<String, f64> = bm_n.clone();

    if !use_ppr {
        let (lex_out, promotions) = apply_anchor_promotions(lex_picks.clone(), params.anchors);
        let mut lex_out = lex_out;
        let mut tb_records = Vec::new();
        if params.use_testbridge {
            let edges = build_import_graph(corpus);
            let (o, r) = apply_testbridge_promotions(lex_out, corpus, &bm, &edges);
            lex_out = o;
            tb_records = r;
        }
        let mut db_records = Vec::new();
        if params.use_docsbridge {
            let (o, r) = apply_docsbridge_promotions(lex_out, corpus, terms);
            lex_out = o;
            db_records = r;
        }
        let explain = Explain {
            lex_picks,
            anchor_promotions: promotions,
            testbridge: tb_records,
            docsbridge: db_records,
            top_score,
            ..Default::default()
        };
        return (lex_out, scores, explain);
    }

    // --- structural expansion
    let edges = build_import_graph(corpus);
    let mut same_dir: HashMap<String, Vec<String>> = HashMap::new();
    for rel in &corpus.files {
        same_dir.entry(py_parent(rel).to_string()).or_default().push(rel.clone());
    }

    let sources: Vec<String> = lex_picks.iter().take(6).cloned().collect();

    let qset: HashSet<String> = terms.iter().cloned().collect();
    let mut fb_terms: HashSet<String> = HashSet::new();
    let impl_sources: Vec<&String> = sources.iter().filter(|f| impl_prior(f) == 1.0).take(3).collect();
    for s in impl_sources {
        let tf_map = corpus.tf.get(s);
        if let Some(tf_map) = tf_map {
            let mut weighted: Vec<(String, f64)> = tf_map
                .iter()
                .filter(|(t, _)| !qset.contains(t.as_str()))
                .map(|(t, &tfv)| {
                    let dfv = *corpus.df.get(t).unwrap_or(&1) as f64;
                    let w = tfv as f64 * (1.0 + corpus.n_docs as f64 / (1.0 + dfv)).ln();
                    (t.clone(), w)
                })
                .collect();
            weighted.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (t, _) in weighted.into_iter().take(20) {
                fb_terms.insert(t);
            }
        }
    }
    let mut fb_sorted: Vec<String> = fb_terms.into_iter().collect();
    fb_sorted.sort();
    let bm_fb = if !fb_sorted.is_empty() { corpus.bm25(&fb_sorted) } else { IndexMap::new() };
    let fb_n = normalize(&bm_fb);

    let mut pool: IndexMap<String, f64> = IndexMap::new();
    let mut owner: HashMap<String, String> = HashMap::new();
    let mut import_nbrs: HashMap<String, Vec<String>> = HashMap::new();
    let mut cochange_origin: HashSet<String> = HashSet::new();
    let fileset: HashSet<&String> = corpus.files.iter().collect();
    let lex_picks_set: HashSet<&String> = lex_picks.iter().collect();

    for s in &sources {
        let w = bm_n.get(s).copied().unwrap_or(0.0);
        let mut imp: Vec<String> = Vec::new();
        let co_partners: Option<&IndexMap<String, i64>> = params.cochange.and_then(|c| c.get(s));

        let mut neighbors: Vec<String> = Vec::new();
        // NOTE (PARITY_NOTES.md item 2): `edges.get(s)` is a Python `set`
        // in the reference; CPython's default hash randomization makes its
        // iteration order (and thus tie-breaks fed by it, downstream) not
        // reproducibly deterministic even between two runs of the *Python*
        // reference. We use sorted (alphabetical) iteration here as the
        // canonical, deterministic choice -- `BTreeSet` already iterates
        // that way.
        if let Some(adj) = edges.get(s) {
            neighbors.extend(adj.iter().cloned());
        }
        if let Some(sd) = same_dir.get(py_parent(s)) {
            neighbors.extend(sd.iter().cloned());
        }
        if let Some(cop) = co_partners {
            for c in cop.keys() {
                if fileset.contains(c) && !neighbors.contains(c) {
                    neighbors.push(c.clone());
                }
            }
        }

        for c in &neighbors {
            if lex_picks_set.contains(c) || c == s || impl_prior(c) < 1.0 {
                continue;
            }
            let is_import_edge = edges.get(s).map(|adj| adj.contains(c)).unwrap_or(false);
            if is_import_edge {
                imp.push(c.clone());
            } else if let Some(cop) = co_partners {
                if cop.get(c).copied().unwrap_or(0) >= params.cochange_strong {
                    imp.push(c.clone());
                }
            }
            if let Some(cop) = co_partners {
                if cop.contains_key(c) {
                    cochange_origin.insert(c.clone());
                }
            }
            if w > pool.get(c).copied().unwrap_or(0.0) {
                pool.insert(c.clone(), w);
                owner.insert(c.clone(), s.clone());
            }
        }
        import_nbrs.insert(s.clone(), imp);
    }

    let add_score = |c: &str, pool: &IndexMap<String, f64>| -> f64 {
        (0.15 + bm_n.get(c).copied().unwrap_or(0.0) + 0.8 * fb_n.get(c).copied().unwrap_or(0.0))
            * (0.5 + 0.5 * pool.get(c).copied().unwrap_or(0.0))
    };

    let mut ranked_pool: Vec<String> = pool.keys().cloned().collect();
    ranked_pool.sort_by(|a, b| add_score(b, &pool).total_cmp(&add_score(a, &pool)));

    let mut additions: Vec<String> = Vec::new();
    if !ranked_pool.is_empty() {
        let pmax = add_score(&ranked_pool[0], &pool);
        let eligible: Vec<String> = ranked_pool.iter().filter(|c| add_score(c, &pool) >= 0.15 * pmax).cloned().collect();
        let eligible_set: HashSet<&String> = eligible.iter().collect();

        let n = corpus.files.len().max(1) as f64;
        let qpath: HashSet<String> = terms
            .iter()
            .filter(|t| {
                if t.chars().count() <= 3 {
                    return false;
                }
                let cnt = corpus.files.iter().filter(|f| corpus.ptoks.get(*f).map(|s| s.contains(t.as_str())).unwrap_or(false)).count();
                (cnt as f64) / n < 0.10
            })
            .cloned()
            .collect();

        let path_hits: Vec<String> = eligible
            .iter()
            .filter(|c| {
                corpus
                    .ptoks
                    .get(*c)
                    .map(|pt| pt.iter().any(|t| qpath.contains(t)))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        let mut path_hits_sorted = path_hits.clone();
        path_hits_sorted.sort_by(|a, b| add_score(b, &pool).total_cmp(&add_score(a, &pool)));
        for c in path_hits_sorted.into_iter().take(6) {
            if !additions.contains(&c) {
                additions.push(c);
            }
        }

        // Guarantee 1: each source's best direct-import neighbor.
        for s in &sources {
            let imp: Vec<String> = import_nbrs
                .get(s)
                .map(|v| v.iter().filter(|c| eligible_set.contains(c)).cloned().collect())
                .unwrap_or_default();
            if !imp.is_empty() {
                let mut best_c = imp[0].clone();
                let mut best_score = add_score(&best_c, &pool);
                for c in &imp[1..] {
                    let sc = add_score(c, &pool);
                    if sc > best_score {
                        best_score = sc;
                        best_c = c.clone();
                    }
                }
                if !additions.contains(&best_c) {
                    additions.push(best_c);
                }
            }
        }

        // Guarantee 2: each source's best neighbor overall.
        let mut groups: IndexMap<String, Vec<String>> = IndexMap::new();
        for c in &eligible {
            groups.entry(owner.get(c).cloned().unwrap_or_default()).or_default().push(c.clone());
        }
        for s in &sources {
            if let Some(grp) = groups.get(s) {
                if let Some(first) = grp.first() {
                    if !additions.contains(first) {
                        additions.push(first.clone());
                    }
                }
            }
        }
        for c in &eligible {
            if additions.len() >= 16 {
                break;
            }
            if !additions.contains(c) {
                additions.push(c.clone());
            }
        }
    }

    // History top-up
    let mut msg_additions: Vec<String> = Vec::new();
    if !corpus.msg_tf.is_empty() {
        let msg_scores = corpus.msg_bm25(terms);
        if !msg_scores.is_empty() {
            let msg_max = msg_scores.values().cloned().fold(f64::MIN, f64::max);
            if msg_max > 0.0 {
                let already: HashSet<String> = lex_picks.iter().chain(additions.iter()).cloned().collect();
                let mut msg_ranked: Vec<(String, f64)> = msg_scores.into_iter().collect();
                msg_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (f, s) in msg_ranked {
                    if msg_additions.len() >= 3 {
                        break;
                    }
                    if already.contains(&f) || s < 0.35 * msg_max || impl_prior(&f) != 1.0 {
                        continue;
                    }
                    msg_additions.push(f);
                }
            }
        }
        additions.extend(msg_additions.iter().cloned());
    }

    let mut out: Vec<String> = lex_picks.clone();
    out.extend(additions.iter().cloned());
    for f in &additions {
        let v = 0.3 + 0.5 * fb_n.get(f).copied().unwrap_or(0.0);
        let cur = scores.get(f).copied().unwrap_or(0.0);
        scores.insert(f.clone(), cur.max(v));
    }

    let (out2, anchor_promotions) = apply_anchor_promotions(out, params.anchors);
    let mut out = out2;
    let mut tb_records = Vec::new();
    if params.use_testbridge {
        let (o, r) = apply_testbridge_promotions(out, corpus, &bm, &edges);
        out = o;
        tb_records = r;
    }
    let mut db_records = Vec::new();
    if params.use_docsbridge {
        let (o, r) = apply_docsbridge_promotions(out, corpus, terms);
        out = o;
        db_records = r;
    }

    let cochange_additions: Vec<String> = additions.iter().filter(|c| cochange_origin.contains(*c)).cloned().collect();
    let pool_explain: Vec<(String, f64, f64)> = ranked_pool
        .iter()
        .map(|c| (c.clone(), py_round(add_score(c, &pool), 4), py_round(pool.get(c).copied().unwrap_or(0.0), 2)))
        .collect();

    let explain = Explain {
        sources,
        lex_picks,
        pool: pool_explain,
        additions,
        cochange_additions,
        msg_additions,
        anchor_promotions,
        testbridge: tb_records,
        docsbridge: db_records,
        top_score,
    };

    (out, scores, explain)
}

// ---------------------------------------------------------------- region packing

// v2: matched per-line via `.captures()` (not `.find_iter()` over the whole
// text), so no `(?m)` flag is needed -- mirrors lanes2.py's
// `_PY_BLOCK_RE.match(ln)` per-line loop.
static PY_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^([ \t]*)(async def |def |class |@)").unwrap());

/// Signature-plus-body block spans (1-indexed, inclusive), split at EVERY
/// def/class/decorator header regardless of indentation -- not just
/// column-0, as a prior version did. Column-0-only splitting made an entire
/// class (every one of its methods) a single multi-hundred-line block, so
/// `pack_regions`' per-file token cap trimmed from that block's START and
/// any gold hunk past the class's first method was silently dropped.
///
/// Blocks now nest: a class's own span still covers its header through its
/// last method (so "class X: <giant body>" remains a valid whole-class
/// candidate for `pack_regions`' greedy packing, same as before), but each
/// direct child header (a method, or a nested function) ALSO gets its own,
/// tighter span from its header line to the next sibling header at the
/// same-or-lower indentation. This hands `pack_regions` candidates at
/// multiple granularities, so a hit deep inside a large class can be
/// represented by its own small method-level block instead of only ever
/// appearing as an early fragment of the whole class.
fn python_blocks(text: &str) -> Vec<(usize, usize)> {
    let lines = py_splitlines(text);
    let n = lines.len();
    // (0-indexed line, indent width)
    let mut headers: Vec<(usize, usize)> = Vec::new();
    for (i, ln) in lines.iter().enumerate() {
        if let Some(caps) = PY_BLOCK_RE.captures(ln) {
            let indent = caps.get(1).unwrap().as_str().chars().count();
            headers.push((i, indent));
        }
    }
    if headers.is_empty() {
        return vec![(1, n)];
    }

    let mut spans: Vec<(usize, usize)> = Vec::new();
    if headers[0].0 > 0 {
        spans.push((1, headers[0].0)); // leading preamble (imports, module docstring)
    }

    for (idx, &(i, indent)) in headers.iter().enumerate() {
        if lines[i].trim_start().starts_with('@') {
            continue; // standalone decorator: folded into the following def/class's span below
        }
        let mut start = i;
        let mut k = i as isize - 1;
        while k >= 0 && lines[k as usize].trim_start().starts_with('@') {
            start = k as usize;
            k -= 1;
        }
        let mut end = n;
        for &(j, ind2) in &headers[idx + 1..] {
            if ind2 <= indent {
                end = j;
                break;
            }
        }
        spans.push((start + 1, end));
    }
    spans.into_iter().filter(|&(a, b)| b >= a).collect()
}

static PY_DEF_LINE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[ \t]*(?:async def|def|class)\s+(\w+)").unwrap());

/// 1-indexed line number of each class/def header's FIRST occurrence in
/// `text` (Python only, any indentation), keyed by symbol name -- used to
/// seat an anchor-channel symbol's own signature+body block among
/// `python_blocks`' spans (a span with that exact start line is that
/// symbol's def block). First occurrence wins: a symbol name can recur
/// (overload-by-decorator, reassignment) but `pack_regions` only needs A
/// definition site to anchor the forced block at, not every one.
fn py_def_line_numbers(text: &str) -> HashMap<String, usize> {
    let mut out: HashMap<String, usize> = HashMap::new();
    for (i, ln) in py_splitlines(text).iter().enumerate() {
        if let Some(caps) = PY_DEF_LINE_RE.captures(ln) {
            let name = caps.get(1).unwrap().as_str().to_string();
            out.entry(name).or_insert(i + 1);
        }
    }
    out
}

fn window_blocks(text: &str, hit_lines: &[usize], radius: usize) -> Vec<(usize, usize)> {
    let n = py_splitlines(text).len();
    if hit_lines.is_empty() {
        return vec![(1, n.min(2 * radius))];
    }
    let mut sorted_hits = hit_lines.to_vec();
    sorted_hits.sort();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for &h in &sorted_hits {
        let a = if h > radius { h - radius } else { 1 };
        let b = (h + radius).min(n);
        if let Some(last) = spans.last_mut() {
            if a <= last.1 + 5 {
                last.1 = b;
                continue;
            }
        }
        spans.push((a, b));
    }
    spans
}

fn hit_lines(text: &str, terms: &HashSet<String>) -> Vec<usize> {
    let mut hits = Vec::new();
    for (i, ln) in py_splitlines(text).iter().enumerate() {
        let low = py_lower(ln);
        if terms.iter().any(|t| low.contains(t.as_str())) {
            hits.push(i + 1);
        }
    }
    hits
}

struct Candidate {
    file: String,
    span: (usize, usize),
    tok: usize,
    terms: HashSet<String>,
    gain: f64,
    text: String,
    name_score: f64,
}

// ---------------------------------------------------------------- region-level symbol-name anchoring
//
// Dogfood bug (query "how is the token budget enforced when packing regions
// into the bundle" against this repo's own core.rs/core.py): the query
// names `pack_regions` almost verbatim, yet pack_regions' region-scoring
// only rewarded query-TERM DENSITY in a region's body -- a region whose name
// IS the query (pack_regions) had no scoring edge over a same-file region
// that merely happens to mention "token" a lot in its body (e.g.
// subtokens()). Ported from lab/lanes2.py's pack_regions region-name-
// anchoring fix, integrated with THIS module's idf-weighted gain +
// anchor_symbols channel-aware forcing (neither of which lanes2 has).

/// Which definition-symbol regex (if any) applies to `rel`, by extension --
/// the same per-language mapping `Corpus::build` uses inline, factored out
/// so region candidates can be matched back to the symbol that defines them.
fn def_re_for(rel: &str) -> Option<&'static Regex> {
    if rel.ends_with(".py") {
        Some(&PY_DEF_RE)
    } else if rel.ends_with(".go") {
        Some(&GO_DEF_RE)
    } else if rel.ends_with(".rs") {
        Some(&RS_DEF_RE)
    } else if rel.ends_with(".js") || rel.ends_with(".ts") || rel.ends_with(".jsx") || rel.ends_with(".tsx") {
        Some(&JS_DEF_RE)
    } else {
        None
    }
}

/// (line_number, symbol_name) for every def/class/fn header `def_re`
/// matches in `text`, sorted ascending by line -- mirrors lanes2.py's
/// `_file_def_lines`. Line number is derived from the matched GROUP's start
/// byte offset, not the overall match's start: these regexes' leading
/// `\s*`/`^\s*` is greedy and, under multi-line mode, `^` matches at the
/// start of every line, so the overall match can begin several blank lines
/// above the actual `def`/`class` keyword. The captured identifier itself
/// is always on the same physical line as its keyword, so anchoring off
/// the group's own start is correct regardless of how much leading
/// whitespace was swallowed.
fn file_def_lines(text: &str, def_re: Option<&'static Regex>) -> Vec<(usize, String)> {
    let Some(re) = def_re else { return Vec::new() };
    let mut out: Vec<(usize, String)> = Vec::new();
    for cap in re.captures_iter(text) {
        for gi in 1..cap.len() {
            if let Some(g) = cap.get(gi) {
                let line = text[..g.start()].matches('\n').count() + 1;
                out.push((line, g.as_str().to_string()));
                break;
            }
        }
    }
    out.sort();
    out
}

/// The defining symbol whose header line falls inside span [a, b] -- i.e.
/// the region's own defining symbol, if any. Picks the EARLIEST matching
/// header (the region's "primary" symbol) when more than one falls in
/// range. Mirrors lanes2.py's `_region_symbol`.
fn region_symbol(def_lines: &[(usize, String)], a: usize, b: usize) -> Option<&str> {
    for (line, sym) in def_lines {
        if *line > b {
            break;
        }
        if *line >= a {
            return Some(sym.as_str());
        }
    }
    None
}

/// Region name-anchoring score: raw count of query-term subtokens the
/// defining symbol's own name contains, i.e.
/// |subtokens(symbol) intersect query_terms| -- NOT normalized by the
/// symbol's subtoken count (normalizing would score a single-subtoken
/// symbol like `subtokens` (1/1 = 1.0) as high as a two-subtoken symbol
/// like `pack_regions` matching BOTH query terms (2/2 = 1.0) -- exactly the
/// collision this fix exists to break). A symbol whose EVERY subtoken is a
/// query term gets a further +1 bonus. Mirrors lanes2.py's `_name_score`.
fn name_score(sym: Option<&str>, tset: &HashSet<String>) -> f64 {
    let Some(s) = sym else { return 0.0 };
    let sym_subs: HashSet<String> = subtokens(s).into_iter().collect();
    if sym_subs.is_empty() {
        return 0.0;
    }
    let overlap_count = sym_subs.intersection(tset).count();
    let mut score = overlap_count as f64;
    if overlap_count > 0 && overlap_count == sym_subs.len() {
        score += 1.0;
    }
    score
}

/// Greedy weighted-coverage packing of regions under budget. See
/// lanes2.py's `pack_regions` docstring.
///
/// A region's term coverage is weighted by each matched term's corpus idf
/// rather than counted flatly, so a region hitting a couple of rare,
/// highly-specific identifiers outranks one hitting many generic terms
/// ("error", "value", ...) that happen to be query terms too.
///
/// `anchor_symbols` (see `anchor_def_symbols`), if given, maps an
/// anchor-channel file to the definition-symbol(s) that anchored it in.
/// Channel-aware packing: the anchored symbol's OWN signature+body block
/// (found among `python_blocks`' nested spans by matching def-line number,
/// see `py_def_line_numbers`) is force-included as that file's pass-1
/// region, at a deeper cap than the generic score-proportional one -- the
/// query named this exact symbol, so its definition is the region least
/// likely to be noise, independent of how it happens to score on term
/// density.
///
/// `w_name` (0.0 from the production caller, i.e. DISABLED: a sweep of
/// {0.0, 0.5, 1.0} on the exact harness -- parity/region_eval2.py +
/// lab/agentless_metric.py, full300_v9 vs full300_v8/wname05, issue #4 --
/// measured that any positive weight regresses the shipped engine, LINE
/// all-or-nothing 35.7% -> 29.3% and line-fraction 0.4564 -> 0.3989, while
/// 0.5 and 1.0 score identically, i.e. the weight saturates; the original
/// validation was against the diverged lab pipeline, issue #8; parameter
/// and code paths kept for re-tuning) additionally
/// rewards a region whose OWN defining symbol name matches query terms (see
/// `name_score`), independent of the anchor channel above: `anchor_symbols`
/// only fires for files the anchor channel itself promoted, whereas this
/// term applies to every file's regions, so a region like `pack_regions`
/// can win purely on its name matching query terms "pack"/"region" even
/// when no anchor promotion happened. Applied to the SELECTION METRIC
/// (gain/tok, post-division) rather than folded into `gain` pre-division: a
/// flat additive term inside `gain` would be diluted away for large
/// regions by the same `/tok` division that makes them expensive, so a big
/// true-match region (e.g. ~1800-token `pack_regions`) could never
/// out-rank a small merely-dense one under gain/tok alone -- adding the
/// name bonus AFTER the division keeps it undiluted by region size, since
/// a symbol-name match is identity evidence independent of how long the
/// matched definition happens to be (measured via this repo's own dogfood
/// case, see lab/dogfood_pack_regions.py).
pub fn pack_regions(
    corpus: &Corpus,
    files: &[String],
    terms: &[String],
    scores: &IndexMap<String, f64>,
    budget_tokens: i64,
    count_tokens: &dyn Fn(&str) -> usize,
    anchor_symbols: Option<&IndexMap<String, Vec<String>>>,
    w_name: f64,
) -> (IndexMap<String, Vec<(usize, usize)>>, String) {
    let tset: HashSet<String> = terms.iter().cloned().collect();
    let idf: HashMap<String, f64> = tset
        .iter()
        .map(|t| {
            let df = corpus.df.get(t).copied().unwrap_or(0) as f64;
            let v = (1.0 + (corpus.n_docs as f64 - df + 0.5) / (df + 0.5)).ln();
            (t.clone(), v)
        })
        .collect();
    let weight = |seg_terms: &HashSet<String>| -> f64 { seg_terms.iter().map(|t| idf.get(t).copied().unwrap_or(0.0)).sum() };

    let mut candidates: Vec<Candidate> = Vec::new();

    for rel in files {
        let text = &corpus.text[rel];
        let lines = py_splitlines(text);
        let hits = hit_lines(text, &tset);
        let spans = if rel.ends_with(".py") { python_blocks(text) } else { window_blocks(text, &hits, 30) };
        let hitset: HashSet<usize> = hits.into_iter().collect();
        let def_lines: Vec<(usize, String)> =
            if w_name != 0.0 { file_def_lines(text, def_re_for(rel)) } else { Vec::new() };
        for (a, b) in spans {
            if a == 0 || b < a || a > lines.len() {
                // guard against degenerate spans; Python's 1-indexed slicing
                // lines[a-1:b] silently no-ops out of range.
            }
            let seg_lines: Vec<&str> = if a >= 1 && a <= lines.len() + 1 {
                let start = a.saturating_sub(1).min(lines.len());
                let end = b.min(lines.len());
                if start < end {
                    lines[start..end].to_vec()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            let seg = seg_lines.join("\n");
            let seg_tokens: HashSet<String> = tokenize(&seg).into_iter().collect();
            let seg_terms: HashSet<String> = tset.intersection(&seg_tokens).cloned().collect();
            let n_hits = (a..=b).filter(|l| hitset.contains(l)).count();
            if seg_terms.is_empty() && n_hits == 0 && a > 1 {
                continue;
            }
            let tok = count_tokens(&seg);
            if tok == 0 {
                continue;
            }
            let gain = (weight(&seg_terms) + 0.5 * n_hits as f64) * (0.3 + scores.get(rel).copied().unwrap_or(0.0));
            let ns = if w_name != 0.0 { name_score(region_symbol(&def_lines, a, b), &tset) } else { 0.0 };
            candidates.push(Candidate { file: rel.clone(), span: (a, b), tok, terms: seg_terms, gain, text: seg, name_score: ns });
        }
    }

    // Channel-aware forced region: for each anchor-selected file, seat its
    // anchored symbol's own def block (a candidate whose span starts exactly
    // at that symbol's def line, per `py_def_line_numbers`) as the pass-1
    // pick -- bypassing the generic gain/tok ranking entirely, since the
    // query named this exact symbol.
    let mut forced: HashMap<String, usize> = HashMap::new(); // file -> candidate index
    if let Some(anchor_map) = anchor_symbols {
        for (rel, syms) in anchor_map {
            if !rel.ends_with(".py") || !files.iter().any(|f| f == rel) {
                continue;
            }
            let def_lines = py_def_line_numbers(&corpus.text[rel]);
            let mut cand_by_start: HashMap<usize, usize> = HashMap::new();
            for (i, c) in candidates.iter().enumerate() {
                if &c.file == rel {
                    cand_by_start.insert(c.span.0, i);
                }
            }
            for sym in syms {
                if let Some(&ln) = def_lines.get(sym) {
                    if let Some(&idx) = cand_by_start.get(&ln) {
                        forced.insert(rel.clone(), idx);
                        break;
                    }
                }
            }
        }
    }

    let mut all_segments: Vec<Candidate> = Vec::new();
    let mut chosen_map: IndexMap<String, Vec<usize>> = IndexMap::new(); // file -> indices into all_segments
    let mut spent: i64 = 0;
    let mut covered: HashSet<String> = HashSet::new();

    let n_files = files.len().max(1) as i64;
    let floor_tok: i64 = 120;
    let spare = (budget_tokens / 2 - floor_tok * n_files).max(0);
    let total_score: f64 = {
        let s: f64 = files.iter().map(|f| scores.get(f).copied().unwrap_or(0.0)).sum();
        if s > 0.0 {
            s
        } else {
            1.0
        }
    };
    let caps: HashMap<String, i64> = files
        .iter()
        .map(|f| {
            let sc = scores.get(f).copied().unwrap_or(0.0);
            // Guard against a non-finite `scores` entry (NaN/inf, e.g. from
            // an upstream normalization/idf bug): `ratio` can otherwise be
            // NaN or +inf, and `(x as i64)` on +inf saturates to i64::MAX,
            // which then overflows the `floor_tok +` add below (checked
            // arithmetic panics in debug; silently wraps in release either
            // way it's wrong). A non-finite ratio contributes no bonus
            // depth instead; finite-input behavior is unchanged.
            let ratio = sc / total_score;
            let bonus_tok = if ratio.is_finite() { ((spare as f64) * ratio) as i64 } else { 0 };
            (f.clone(), floor_tok + bonus_tok)
        })
        .collect();
    // Anchor-forced files get a deeper cap on top of their score-proportional
    // one (up to budget/10) -- the definition block a query explicitly named
    // is worth more depth than generic density would otherwise buy it.
    let anchor_cap: i64 = (budget_tokens / 10).max(floor_tok);

    for rel in files {
        let idxs: Vec<usize> = candidates.iter().enumerate().filter(|(_, c)| &c.file == rel).map(|(i, _)| i).collect();
        if idxs.is_empty() {
            continue;
        }
        let forced_idx = forced.get(rel).copied();
        // Python's `max(cands, key=...)` returns the FIRST maximal element
        // on ties; Rust's `Iterator::max_by` returns the LAST. Replicate
        // Python's tie behavior with an explicit "strictly greater only
        // replaces" fold.
        let best_idx = if let Some(fi) = forced_idx {
            fi
        } else {
            let mut best_idx = idxs[0];
            let mut best_ratio = candidates[best_idx].gain / (candidates[best_idx].tok.max(1) as f64)
                + w_name * candidates[best_idx].name_score;
            for &i in &idxs[1..] {
                let ratio = candidates[i].gain / (candidates[i].tok.max(1) as f64) + w_name * candidates[i].name_score;
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_idx = i;
                }
            }
            best_idx
        };
        let mut best_span = candidates[best_idx].span;
        let mut best_text = candidates[best_idx].text.clone();
        let mut best_tok = candidates[best_idx].tok;
        let best_terms = candidates[best_idx].terms.clone();
        let mut per_file_cap = *caps.get(rel).unwrap_or(&floor_tok);
        if forced_idx.is_some() {
            per_file_cap = per_file_cap.max((best_tok as i64).min(anchor_cap));
        }

        if best_tok as i64 > per_file_cap {
            let (a, b) = best_span;
            let full_lines = py_splitlines(&corpus.text[rel]);
            let start = a.saturating_sub(1).min(full_lines.len());
            let end = b.min(full_lines.len());
            let seg_lines: Vec<&str> = if start < end { full_lines[start..end].to_vec() } else { Vec::new() };
            // NOTE: `keep` (used for the reported SPAN) is deliberately NOT
            // clamped to seg_lines.len() here, matching a quirk of the
            // Python original: `keep = max(4, int(len(seg_lines) *
            // per_file_cap / best["tok"]))` can exceed len(seg_lines) for a
            // very short/dense segment, but `seg_lines[:keep]` silently
            // no-ops at the slice boundary while the reported span
            // `(a, a + keep - 1)` still uses the uncapped value. Only the
            // TEXT slice is bounds-limited.
            let keep = (4usize).max(((seg_lines.len() as f64) * (per_file_cap as f64) / (best_tok as f64)) as usize);
            let slice_end = keep.min(seg_lines.len());
            let mut seg = seg_lines[..slice_end].join("\n");
            let mut tok = count_tokens(&seg);
            if tok as i64 > 2 * per_file_cap {
                let char_cap = (per_file_cap * 4) as usize;
                let truncated: String = seg.chars().take(char_cap).collect();
                seg = truncated;
                tok = count_tokens(&seg);
            }
            best_span = (a, a + keep - 1);
            best_text = seg;
            best_tok = tok;
        }

        let best_name_score = candidates[best_idx].name_score;
        let cand = Candidate {
            file: rel.clone(), span: best_span, tok: best_tok, terms: best_terms, gain: 0.0, text: best_text,
            name_score: best_name_score,
        };
        covered.extend(cand.terms.iter().cloned());
        spent += cand.tok as i64;
        let seg_idx = all_segments.len();
        all_segments.push(cand);
        chosen_map.entry(rel.clone()).or_default().push(seg_idx);
    }

    // pass 2: greedy marginal coverage over the ORIGINAL candidates minus
    // whichever became the pass-1 pick per file (Python compares dicts by
    // identity/equality against the `chosen` accumulator; here we track by
    // (file, span) identity, which uniquely determines a Python candidate
    // dict too since span+file is the natural key for this list).
    let pass1_keys: HashSet<(String, (usize, usize))> = chosen_map
        .iter()
        .flat_map(|(f, idxs)| idxs.iter().map(|&i| (f.clone(), all_segments[i].span)))
        .collect();
    let mut remaining: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            // A pass-1 pick may have been re-spanned (trimmed); only the
            // untrimmed original candidate remains eligible for pass 2 if
            // it wasn't the one chosen (same file+span as an ORIGINAL
            // candidate is excluded only if it's literally the chosen one).
            !pass1_keys.contains(&(c.file.clone(), c.span))
        })
        .map(|(i, _)| i)
        .collect();

    while !remaining.is_empty() && spent < budget_tokens {
        let marginal = |i: usize| -> f64 {
            let c = &candidates[i];
            let diff: HashSet<String> = c.terms.difference(&covered).cloned().collect();
            let new_weight = weight(&diff);
            let base = (new_weight + 0.25 * weight(&c.terms) + 0.1) * (0.3 + scores.get(&c.file).copied().unwrap_or(0.0))
                / (c.tok.max(1) as f64);
            // same undiluted-by-size name bonus as pass 1's selection metric
            // (see pack_regions' doc comment) -- otherwise a name-anchored
            // region too large to win pass 1 could never win pass 2's
            // marginal race either.
            base + w_name * c.name_score
        };
        // total_cmp, not partial_cmp().unwrap(): `marginal` folds in
        // `scores.get(&c.file)`, an externally-supplied score that isn't
        // guaranteed NaN-free (a caller-side idf/normalization bug upstream
        // could hand pack_regions a NaN here), and partial_cmp().unwrap()
        // panics ("called `Option::unwrap()` on a `None` value") the
        // instant a NaN is compared against anything. total_cmp gives a
        // deterministic, non-panicking total order and agrees with
        // partial_cmp on every finite, non-NaN input, so this is a pure
        // hardening for real inputs, not a ranking change.
        remaining.sort_by(|&a, &b| marginal(b).total_cmp(&marginal(a)));
        let i = remaining.remove(0);
        let tok = candidates[i].tok as i64;
        if spent + tok > budget_tokens {
            if candidates[i].tok > 200 {
                continue;
            }
            break;
        }
        spent += tok;
        covered.extend(candidates[i].terms.iter().cloned());
        let file = candidates[i].file.clone();
        let seg_idx = all_segments.len();
        all_segments.push(Candidate {
            file: candidates[i].file.clone(),
            span: candidates[i].span,
            tok: candidates[i].tok,
            terms: candidates[i].terms.clone(),
            gain: candidates[i].gain,
            text: candidates[i].text.clone(),
            name_score: candidates[i].name_score,
        });
        chosen_map.entry(file).or_default().push(seg_idx);
    }

    let mut parts: Vec<String> = Vec::new();
    let mut spans_out: IndexMap<String, Vec<(usize, usize)>> = IndexMap::new();
    for rel in files {
        let idxs = match chosen_map.get(rel) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        let mut segs: Vec<&Candidate> = idxs.iter().map(|&i| &all_segments[i]).collect();
        segs.sort_by_key(|c| c.span.0);
        spans_out.insert(rel.clone(), segs.iter().map(|c| c.span).collect());
        let body = segs.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n...\n");
        parts.push(format!("### {rel}\n{body}"));
    }
    (spans_out, parts.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 20+ fixtures generated from the Python reference:
    ///   PYTHONHASHSEED=0 uv run --project <archex venv> python3 -c
    ///     "import sys; sys.path.insert(0, '<repo>/lab'); import lanes2 as L;
    ///      print(L.stem('validators'))"  # etc.
    #[test]
    fn stem_matches_python_reference() {
        let cases: &[(&str, &str)] = &[
            ("validators", "validat"),
            ("validate", "validat"),
            ("dependencies", "dependenci"),
            ("dependency", "dependenci"),
            ("routing", "rout"),
            ("route", "rout"),
            ("router", "rout"),
            ("classes", "class"),
            ("glasses", "glass"),
            ("passes", "pass"),
            ("runs", "run"),
            ("running", "runn"),
            ("tested", "test"),
            ("tester", "test"),
            ("collector", "collect"),
            ("monitor", "monit"),
            ("factory", "factori"),
            ("category", "categori"),
            ("type", "type"),
            ("name", "name"),
            ("service", "servic"),
            ("using", "using"),
            ("handled", "handl"),
        ];
        for (input, expected) in cases {
            assert_eq!(stem(input), *expected, "stem({input:?})");
        }
    }

    /// Subtoken splitter fixtures (camelCase/snake_case boundary cases),
    /// generated from `lanes2.subtokens()` via the same archex venv.
    #[test]
    fn subtokens_matches_python_reference() {
        let cases: &[(&str, &[&str])] = &[
            ("HTTPResponse", &["http", "respons"]),
            ("parseHTTPResponse2", &["pars", "http", "respons"]),
            ("snake_case_name", &["snak", "case", "name"]),
            ("XMLHttpRequest", &["xml", "http", "request"]),
            ("A", &[]),
            ("ABc", &[]),
            ("alreadylower", &["alreadylow"]),
        ];
        for (input, expected) in cases {
            assert_eq!(subtokens(input), *expected, "subtokens({input:?})");
        }
    }

    /// tokenize() on 5 code snippets (Python/Go/JS/Rust/import-statement
    /// styles), generated from `lanes2.tokenize()` via the archex venv.
    #[test]
    fn tokenize_matches_python_reference() {
        let snippet0 = "def parseHTTPResponse2(self, data):\n    validators = self.get_validators()\n    return validators";
        assert_eq!(
            tokenize(snippet0),
            vec!["parsehttpresponse2", "pars", "http", "respons", "data", "validat", "get_validat", "get", "validat", "validat"]
        );

        let snippet1 = "class ConnectionPool:\n    def __init__(self):\n        self.routing_table = {}\n    def get_router(self):\n        pass";
        assert_eq!(
            tokenize(snippet1),
            vec!["connectionpool", "connection", "pool", "__init__", "init", "routing_tabl", "rout", "tabl", "get_rout", "get", "rout", "pass"]
        );

        let snippet2 = "function XMLHttpRequestHandler(req, res) {\n    const dependency_graph = buildGraph();\n    return dependency_graph;\n}";
        assert_eq!(
            tokenize(snippet2),
            vec![
                "xmlhttprequesthandl", "xml", "http", "request", "handl", "req", "res", "dependency_graph",
                "dependenci", "graph", "buildgraph", "build", "graph", "dependency_graph", "dependenci", "graph"
            ]
        );

        let snippet3 = "// snake_case_name test\nstruct HTTPClient {\n    keep_alive: bool,\n}";
        assert_eq!(
            tokenize(snippet3),
            vec!["snake_case_nam", "snak", "case", "name", "test", "httpclient", "http", "client", "keep_aliv", "keep", "aliv"]
        );

        let snippet4 = "import os\nfrom collections import defaultdict\n# a comment about testing\n";
        assert_eq!(tokenize(snippet4), vec!["collection", "defaultdict", "comment", "about", "test"]);
    }

    #[test]
    fn tokenize_already_lower() {
        assert_eq!(tokenize("already lower"), vec!["alreadi", "lower"]);
    }

    /// impl_prior() path classification, generated from
    /// `lanes2.impl_prior()` via the archex venv.
    #[test]
    fn impl_prior_matches_python_reference() {
        let cases: &[(&str, f64)] = &[
            ("src/main.py", 1.0),
            ("tests/test_main.py", 0.3),
            ("test_foo.py", 0.3),
            ("foo_test.py", 0.3),
            ("docs/index.md", 0.3),
            ("examples/demo.py", 0.3),
            ("lib/router.rs", 1.0),
            ("benches/bench_x.rs", 0.3),
            ("a/b/conftest.py", 0.3),
            ("spec/foo.spec.js", 0.3),
            ("foo.test.js", 0.3),
            ("vendor/x.js", 1.0),
            ("node_modules/y.js", 1.0),
            ("src/t/z.py", 0.3),
        ];
        for (path, expected) in cases {
            assert_eq!(impl_prior(path), *expected, "impl_prior({path:?})");
        }
    }

    /// Region-name-anchoring fix: `_file_def_lines`/`_region_symbol` fixtures
    /// generated from `lanes2._file_def_lines`/`_region_symbol` (also ported
    /// verbatim into src/roust/core.py) via the archex venv.
    #[test]
    fn file_def_lines_and_region_symbol_match_python_reference() {
        let text = "import os\n\n\ndef alpha(x):\n    return x\n\n\nclass Beta:\n    def gamma(self):\n        pass\n";
        let def_lines = file_def_lines(text, def_re_for("mod.py"));
        assert_eq!(
            def_lines,
            vec![(4, "alpha".to_string()), (8, "Beta".to_string()), (9, "gamma".to_string())]
        );
        assert_eq!(region_symbol(&def_lines, 1, 5), Some("alpha"));
        assert_eq!(region_symbol(&def_lines, 8, 10), Some("Beta"));
        assert_eq!(region_symbol(&def_lines, 9, 10), Some("gamma"));
        assert_eq!(region_symbol(&def_lines, 20, 30), None);
    }

    /// `_name_score` fixtures generated from `lanes2._name_score` (also
    /// ported verbatim into src/roust/core.py) via the archex venv.
    #[test]
    fn name_score_matches_python_reference() {
        let full_match: HashSet<String> = ["pack".to_string(), "region".to_string()].into_iter().collect();
        assert_eq!(name_score(Some("pack_regions"), &full_match), 3.0);

        let no_overlap: HashSet<String> =
            ["token".to_string(), "pack".to_string(), "budget".to_string()].into_iter().collect();
        assert_eq!(name_score(Some("subtokens"), &no_overlap), 0.0);

        let partial: HashSet<String> = ["pack".to_string()].into_iter().collect();
        assert_eq!(name_score(Some("pack_regions"), &partial), 1.0);

        assert_eq!(name_score(None, &full_match), 0.0);
        assert_eq!(name_score(Some("__init__"), &full_match), 0.0);
    }

    /// Dogfood regression (see lab/dogfood_pack_regions.py): a region whose
    /// DEFINING SYMBOL matches query terms must win pass-1 selection over a
    /// same-file region that only has denser body term matches for a
    /// generic term, once `w_name` is on -- and must NOT win when `w_name`
    /// is 0.0 (byte-identical to pre-fix ranking), so this also pins the
    /// pre-fix bug reproduction. budget_tokens=1 isolates pass-1's pick
    /// (pass 2 never runs since `spent >= budget_tokens` immediately after
    /// pass 1, which spends unconditionally regardless of budget).
    #[test]
    fn pack_regions_name_score_promotes_symbol_name_match() {
        let tmp = std::env::temp_dir().join(format!("roust_namescore_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        // subtokens: SHORT region, dense in query terms (token/budget/enforc)
        // -- wins on gain/tok pre-fix. pack_regions: LONG region (40 filler
        // lines) whose only query-term evidence is its own def line
        // ("pack"/"region" from the identifier) -- loses on density despite
        // being the actual query target, exactly the real dogfood bug's
        // shape (a big true-match region diluted by the /tok division).
        let filler: String = (0..40).map(|i| format!("    x{i} = {i}\n")).collect();
        let src = format!(
            "def subtokens(word):\n    \"\"\"token budget enforced.\"\"\"\n    return word.split('_')\n\n\ndef pack_regions(cap):\n{filler}    return cap\n"
        );
        std::fs::write(tmp.join("core.py"), src).unwrap();

        let corpus = Corpus::build(&tmp, None, false, false);
        let terms = query_terms("how is the token budget enforced when packing regions", &[]);
        let scores: IndexMap<String, f64> = [("core.py".to_string(), 1.0)].into_iter().collect();
        let files = vec!["core.py".to_string()];
        let count_tokens = |s: &str| -> usize { s.split_whitespace().count() };

        let def_lines = file_def_lines(&corpus.text["core.py"], def_re_for("core.py"));
        let sym_of = |spans: &IndexMap<String, Vec<(usize, usize)>>| -> Option<String> {
            spans["core.py"].first().and_then(|&(a, b)| region_symbol(&def_lines, a, b)).map(|s| s.to_string())
        };

        let (spans_off, _) = pack_regions(&corpus, &files, &terms, &scores, 1, &count_tokens, None, 0.0);
        assert_eq!(sym_of(&spans_off), Some("subtokens".to_string()), "pre-fix (w_name=0.0) reproduces the bug: body term-density picks the wrong region");

        let (spans_on, _) = pack_regions(&corpus, &files, &terms, &scores, 1, &count_tokens, None, 1.0);
        assert_eq!(sym_of(&spans_on), Some("pack_regions".to_string()), "w_name=1.0 must select pack_regions via name-score anchoring");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// Regression for issue #14: pack_regions' pass-2 greedy loop sorts
    /// `remaining` candidates by a `marginal` score that folds in a
    /// caller-supplied `scores` entry per file. If that score is NaN (e.g.
    /// upstream normalization/idf drift), the pre-fix
    /// `partial_cmp().unwrap()` comparator panics the instant the NaN
    /// candidate is compared against anything. Assert pack_regions instead
    /// completes without panicking and produces a deterministic ordering
    /// (same spans across repeated calls with the same NaN input).
    #[test]
    fn pack_regions_survives_nan_and_inf_scores() {
        let tmp = std::env::temp_dir().join(format!("roust_nanscore_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let filler: String = (0..20).map(|i| format!("    x{i} = {i}\n")).collect();
        std::fs::write(
            &tmp.join("a.py"),
            format!("def alpha_token():\n{filler}    return 1\n\n\ndef beta_budget():\n{filler}    return 2\n"),
        )
        .unwrap();
        std::fs::write(
            &tmp.join("b.py"),
            format!("def gamma_token():\n{filler}    return 3\n\n\ndef delta_budget():\n{filler}    return 4\n"),
        )
        .unwrap();

        let corpus = Corpus::build(&tmp, None, false, false);
        let terms = query_terms("token budget", &[]);
        let files = vec!["a.py".to_string(), "b.py".to_string()];
        let count_tokens = |s: &str| -> usize { s.split_whitespace().count() };

        // NaN and +inf scores, directly on the pack_regions contract (a
        // caller-supplied score map), not manufactured via any specific
        // upstream idf/normalization path.
        let scores: IndexMap<String, f64> =
            [("a.py".to_string(), f64::NAN), ("b.py".to_string(), f64::INFINITY)].into_iter().collect();

        // Large budget so pass 2's greedy loop actually runs over multiple
        // remaining candidates (pass 1 alone would only ever touch one span
        // per file).
        let (spans1, _) = pack_regions(&corpus, &files, &terms, &scores, 100_000, &count_tokens, None, 0.0);
        let (spans2, _) = pack_regions(&corpus, &files, &terms, &scores, 100_000, &count_tokens, None, 0.0);
        assert_eq!(spans1, spans2, "pack_regions must be deterministic given identical (NaN/inf-bearing) inputs");
        assert!(!spans1.is_empty(), "pack_regions should still select regions despite NaN/inf scores");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn camel_matches_edge_cases() {
        assert_eq!(camel_matches("A"), vec!["A"]);
        assert_eq!(camel_matches("ABc"), vec!["A", "Bc"]);
        assert_eq!(camel_matches("HTTPResponse"), vec!["HTTP", "Response"]);
        assert_eq!(camel_matches("XMLHttpRequest"), vec!["XML", "Http", "Request"]);
        assert_eq!(camel_matches("alreadylower"), vec!["alreadylower"]);
    }

    #[test]
    fn end_to_end_synthetic_repo_smoke() {
        // 3-file synthetic repo: a router module importing a validators
        // module, plus an unrelated test file, exercising Corpus building,
        // BM25, the import graph, and select_files' structural-expansion
        // path end to end.
        let tmp = std::env::temp_dir().join(format!("roust_smoke_{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("pkg")).unwrap();
        std::fs::create_dir_all(tmp.join("tests")).unwrap();
        std::fs::write(
            tmp.join("pkg/router.py"),
            "from .validators import validate_request\n\n\ndef route_request(req):\n    \"\"\"Route an incoming request after validation.\"\"\"\n    validate_request(req)\n    return handle(req)\n\n\ndef handle(req):\n    return req\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("pkg/validators.py"),
            "def validate_request(req):\n    \"\"\"Validate an incoming request payload.\"\"\"\n    if not req:\n        raise ValueError('bad request')\n    return True\n",
        )
        .unwrap();
        std::fs::write(
            tmp.join("tests/test_router.py"),
            "from pkg.router import route_request\n\n\ndef test_route_request():\n    assert route_request({'a': 1})\n",
        )
        .unwrap();

        let corpus = Corpus::build(&tmp, None, false, false);
        assert_eq!(corpus.n_docs, 3);
        assert!(corpus.files.contains(&"pkg/router.py".to_string()));

        let terms = query_terms("how does the router validate an incoming request", &[]);
        assert!(terms.contains(&"rout".to_string()));
        assert!(terms.contains(&"validat".to_string()));

        let params = SelectParams::default();
        let (files, _scores, _explain) = select_files(&corpus, &terms, true, &params);
        assert!(files.contains(&"pkg/router.py".to_string()));
        assert!(files.contains(&"pkg/validators.py".to_string()));

        let count_tokens = |s: &str| -> usize { s.split_whitespace().count() };
        let (spans, bundle) = pack_regions(&corpus, &files, &terms, &_scores, 4096, &count_tokens, None, 1.0);
        assert!(!bundle.is_empty());
        assert!(spans.contains_key("pkg/router.py"));

        std::fs::remove_dir_all(&tmp).ok();
    }

    // ---------------------------------------------------------------- issue #25:
    // low-confidence signal + query-term-coverage helpers

    #[test]
    fn query_term_coverage_counts_partial_match() {
        let tmp = std::env::temp_dir().join(format!("roust_qtc_partial_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(&tmp.join("widget.py"), "def widget_handler():\n    return 1\n").unwrap();
        let corpus = Corpus::build(&tmp, None, false, false);

        let terms = query_terms("widget handler zzznonexistentxyzzy", &[]);
        let (matched, total) = query_term_coverage(&corpus, &terms);
        assert_eq!(total, terms.len());
        assert!(matched >= 1 && matched < total, "expected a partial match, got {matched}/{total}");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn query_term_coverage_zero_when_nothing_in_vocabulary() {
        let tmp = std::env::temp_dir().join(format!("roust_qtc_zero_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(&tmp.join("widget.py"), "def widget_handler():\n    return 1\n").unwrap();
        let corpus = Corpus::build(&tmp, None, false, false);

        let terms = query_terms("zzznonexistentxyzzy qqxwibblewonk", &[]);
        let (matched, total) = query_term_coverage(&corpus, &terms);
        assert_eq!(matched, 0);
        assert_eq!(total, terms.len());
        assert!(total > 0);

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn is_low_confidence_threshold_logic() {
        // Strong raw score + full term coverage: confident.
        assert!(!is_low_confidence(LOW_CONFIDENCE_TOP_SCORE + 1.0, 5, 5));
        // Weak raw score even with full coverage: low-confidence.
        assert!(is_low_confidence(1.0, 5, 5));
        // Strong raw score but most terms unmatched: low-confidence.
        assert!(is_low_confidence(1000.0, 1, 10));
        // Literal zero-match case: low-confidence (superseded by the exit-1
        // gate in main.rs, but the predicate itself must not claim confidence).
        assert!(is_low_confidence(0.0, 0, 5));
        // No terms at all: low-confidence (vacuous, defensive case).
        assert!(is_low_confidence(5.0, 0, 0));
        // Right at the boundary: not low-confidence (strict `<`, not `<=`).
        assert!(!is_low_confidence(LOW_CONFIDENCE_TOP_SCORE, 5, 5));
    }
}
