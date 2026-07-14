//! Git-history mining for the semantic retrieval layer -- a faithful port of
//! `lab/history.py`. Single `git log` pass over the checked-out repo's recent
//! non-merge commit history, producing `msgs` (per-file commit message
//! field), `cochange` (per-file co-committed partner counts, plus derived
//! test-bridge edges), and `meta` (per-file commit/author summary, not
//! consumed by any scoring path -- kept for parity with the Python return
//! shape only).
//!
//! ORDERING NOTE: unlike select_files() in core.rs (which has exactly one
//! raw-`set`-iteration nondeterminism -- see PARITY_NOTES.md), history.py's
//! own logic never iterates a raw Python `set` in an order-sensitive way:
//! every dict here is insertion-ordered (deterministic, driven by git log's
//! own -- deterministic -- commit order), and the one spot combinations()
//! runs over a set (`combinations(sorted(set(code_files)), 2)`) is
//! explicitly pre-sorted first. So `IndexMap`, used throughout below to
//! mirror Python dict/Counter insertion order, gives byte-for-byte parity
//! with no caveats.

use crate::core::{is_code_file, TESTLIKE_RE};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::Command;

const SENTINEL: &str = "__C__";
const MAX_MSG_CHARS: usize = 40_000;
const MAX_MSGS_PER_FILE: usize = 25;
const BULK_COMMIT_FILE_LIMIT: usize = 20;
const MIN_COCHANGE_COUNT: i64 = 3;
const MAX_COCHANGE_PARTNERS: usize = 10;
const MAX_BRIDGE_CANDIDATES: usize = 50;
const MAX_AUTHORS_PER_FILE: usize = 5;

// ---------------------------------------------------------------- E6: per-line change-recency (Linespots-style)
//
// See `lab/research/wave2/fine-grained-fault-localization.md` (#2, Linespots)
// and issue #4. Deliberately NOT part of `mine_history`'s bulk pass above:
// that pass is one `git log --name-only` walk over up to 5000 commits for
// the WHOLE repo, and extending it to also carry hunk-level diffs
// (`git log -p`) for every commit/file would multiply index-time cost by
// the size of every patch in the mined window -- for a query that only ever
// looks at a handful of already-selected files. Computing it here instead,
// lazily, per file, only for files `pack_regions` is actually scoring,
// keeps index time byte-identical to before this feature (see
// `--recency-weight`'s CLI doc and cache.rs: no on-disk cache field needed
// either, since nothing here is persisted).
//
// LEAK SAFETY: both git invocations below (`git log ... -- <path>` and
// `git blame`) walk only commits reachable from the checked-out HEAD --
// exactly like `mine_history` above. The SWE-bench-style eval harness
// checks a repo out AT `base_commit` before invoking roust, so HEAD *is*
// base_commit at query time: no commit authored after `base_commit` can
// ever be reachable, so this can never see the fix that hasn't happened
// yet from the retrieval engine's point of view.

/// Cap on how many of a file's own historical commits establish the
/// recency-decay window (newest=1.0 .. oldest-in-window=0.0), independent of
/// `mine_history`'s repo-wide 5000-commit cap -- mirrors Linespots' own
/// ~500-commit decay horizon (wave2 doc, #2).
pub const RECENCY_MAX_COMMITS: usize = 500;
const RECENCY_STEEPNESS: f64 = 12.0;

/// Linespots' per-touch decay term: `1/(1+exp(-12*a+12))`, `a` = commit age
/// normalized to `[0, 1]` over the mined window, newest commit = 1.
fn recency_decay(a: f64) -> f64 {
    1.0 / (1.0 + (-RECENCY_STEEPNESS * a + RECENCY_STEEPNESS).exp())
}

/// Newest-first list of commit SHAs reachable from HEAD that touched `rel`,
/// capped at `RECENCY_MAX_COMMITS`. Establishes the recency RANK used to
/// normalize `recency_decay`'s `a` -- rank 0 (newest) -> a=1, rank
/// `len-1` (oldest in window) -> a=0.
fn file_commit_order(repo_path: &Path, rel: &str) -> Vec<String> {
    let output = Command::new("git")
        .args(["log", "-n", &RECENCY_MAX_COMMITS.to_string(), "--format=%H", "--", rel])
        .current_dir(repo_path)
        .output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

fn is_hex40(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// `(final_line_number, commit_sha)` for every line in `rel`'s CURRENT
/// (checked-out HEAD) content, via one `git blame --line-porcelain` pass.
/// Line numbers here are HEAD's own -- the exact coordinate space region
/// spans use -- which is why this is blame-based rather than replaying
/// each historical commit's own hunk ranges directly: those shift under
/// every later insertion/deletion and would silently misalign with current
/// spans. `--line-porcelain` repeats the full commit header for every
/// line (rather than once per contiguous same-commit run), which is more
/// output but trivial to parse unambiguously: a header line is the only
/// line shape starting with a bare 40-hex-char SHA followed by two
/// integers; content lines are always tab-prefixed.
fn blame_line_commits(repo_path: &Path, rel: &str) -> Vec<(usize, String)> {
    let output = Command::new("git")
        .args(["blame", "--line-porcelain", "--", rel])
        .current_dir(repo_path)
        .output();
    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut out = Vec::new();
    for line in stdout.lines() {
        if line.starts_with('\t') {
            continue;
        }
        let mut parts = line.split(' ');
        let sha = match parts.next() {
            Some(s) if is_hex40(s) => s,
            _ => continue,
        };
        if parts.next().is_none() {
            continue; // orig-line-number field
        }
        let final_line: usize = match parts.next().and_then(|s| s.parse().ok()) {
            Some(n) => n,
            None => continue,
        };
        out.push((final_line, sha.to_string()));
    }
    out
}

/// Per-line change-recency score for `rel`'s CURRENT content, normalized
/// per-file to `[0, 1]` (the file's most-recently-touched line scores
/// 1.0). A line absent from the returned map has no score (0, implicitly)
/// -- either the file/repo has no usable history, or blame attributed that
/// line to a commit outside the mined window entirely (treated as maximally
/// old: `a=0`, `recency_decay(0)` is already negligible so this collapses
/// to the same "no signal" outcome either way).
///
/// Simplified Linespots (wave2 doc, #2): the paper's per-line score is a
/// SUM of `recency_decay` terms, one per historical commit that ever
/// touched the line. Reconstructing that full multi-touch sum against
/// CURRENT line numbers would require replaying every historical hunk
/// forward through every later commit's insertions/deletions -- i.e.
/// reimplementing git blame's own line-tracking algorithm from scratch.
/// Instead this scores only each current line's SINGLE most recent
/// touching commit (from `git blame`, which already solves the line-
/// tracking problem for us). This is a low-cost approximation of the sum,
/// not an accuracy giveaway: steepness 12 makes any earlier term more than
/// ~15% of the window back from the line's latest touch contribute < 0.1
/// (a=0.85 -> ~0.15, a=0.70 -> ~0.02, a=0.5 -> ~0.002), so for the
/// overwhelming common case (a line's earlier touches, if any, sit more
/// than a handful of commits behind its latest one) the single latest term
/// already dominates the sum this is standing in for. We have no fix-
/// commit labels either (see issue #4 / the wave2 doc's leak-warning
/// discussion), so -- per the doc's own fallback -- this is raw churn
/// recency, not fix recency: a weaker but still net-positive proxy per
/// D'Ambros et al. (cited in the Linespots paper's related work).
pub fn file_line_recency(repo_path: &Path, rel: &str) -> HashMap<usize, f64> {
    let order = file_commit_order(repo_path, rel);
    if order.is_empty() {
        return HashMap::new();
    }
    let rank_of: HashMap<&str, usize> = order.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let denom = order.len().saturating_sub(1).max(1) as f64;

    let blame = blame_line_commits(repo_path, rel);
    let mut raw: HashMap<usize, f64> = HashMap::new();
    let mut max_raw = 0.0_f64;
    for (line, sha) in &blame {
        let a = match rank_of.get(sha.as_str()) {
            Some(&rank) => 1.0 - (rank as f64 / denom),
            None => 0.0, // touching commit fell outside the mined window
        };
        let score = recency_decay(a);
        if score > max_raw {
            max_raw = score;
        }
        raw.insert(*line, score);
    }
    if max_raw <= 0.0 {
        return HashMap::new();
    }
    for v in raw.values_mut() {
        *v /= max_raw;
    }
    raw
}

/// Insertion-ordered multiset, mirroring `collections.Counter`'s dict
/// semantics (first-seen order preserved for ties in `most_common()`).
pub type OrderedCounter = IndexMap<String, i64>;

fn counter_incr(c: &mut OrderedCounter, k: &str, by: i64) {
    *c.entry(k.to_string()).or_insert(0) += by;
}

/// `Counter.most_common()`: stable sort by count descending, ties broken by
/// the counter's insertion order (Rust's `sort_by` is stable, and the
/// intermediate Vec is built by iterating the IndexMap in insertion order,
/// so this matches exactly).
fn most_common(c: &OrderedCounter) -> Vec<(String, i64)> {
    let mut v: Vec<(String, i64)> = c.iter().map(|(k, n)| (k.clone(), *n)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileMeta {
    pub n_commits: i64,
    pub last_ts: i64,
    pub authors: IndexMap<String, i64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HistoryData {
    pub msgs: IndexMap<String, String>,
    pub cochange: IndexMap<String, IndexMap<String, i64>>,
    pub meta: IndexMap<String, FileMeta>,
}

fn looks_like_path(ln: &str) -> bool {
    let s = ln.trim();
    if s.is_empty() || s.contains(' ') || s.contains('\t') {
        return false;
    }
    s.contains('/') || (s.contains('.') && !s.starts_with('.'))
}

/// Split a list of lines into blank-line-delimited blocks (blanks dropped,
/// empty leading/trailing blocks dropped).
fn split_blocks<'a>(lines: &[&'a str]) -> Vec<Vec<&'a str>> {
    let mut blocks = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for &ln in lines {
        if ln.trim().is_empty() {
            if !cur.is_empty() {
                blocks.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(ln);
        }
    }
    if !cur.is_empty() {
        blocks.push(cur);
    }
    blocks
}

/// `rest` = every raw line after the header line, up to (excluding) the next
/// commit's sentinel line. Returns (message, files).
fn parse_commit(subject: &str, rest: &[&str]) -> (String, Vec<String>) {
    let blocks = split_blocks(rest);
    if blocks.is_empty() {
        return (subject.to_string(), Vec::new());
    }
    let last = blocks.last().unwrap();
    let (files, body_blocks): (Vec<String>, &[Vec<&str>]) = if last.iter().all(|ln| looks_like_path(ln)) {
        (
            last.iter().map(|ln| ln.trim().to_string()).collect(),
            &blocks[..blocks.len() - 1],
        )
    } else {
        (Vec::new(), &blocks[..])
    };
    let body = body_blocks
        .iter()
        .map(|b| b.join("\n"))
        .collect::<Vec<_>>()
        .join("\n\n");
    let message = if body.is_empty() {
        subject.to_string()
    } else {
        format!("{subject}\n{body}")
    };
    (message, files)
}

/// Derive production<->production "bridge" edges via a shared test-like
/// co-change partner (see history.py's `_bridge_cochange` docstring for the
/// full rationale).
fn bridge_cochange(cochange_counts: &IndexMap<String, OrderedCounter>) -> IndexMap<String, OrderedCounter> {
    let mut bridges: IndexMap<String, OrderedCounter> = IndexMap::new();
    for (t, partners) in cochange_counts {
        if !TESTLIKE_RE.is_match(t) {
            continue;
        }
        let mut qualifying: Vec<(String, i64)> = partners
            .iter()
            .filter(|(f, c)| **c >= MIN_COCHANGE_COUNT && !TESTLIKE_RE.is_match(f))
            .map(|(f, c)| (f.clone(), *c))
            .collect();
        qualifying.sort_by(|a, b| b.1.cmp(&a.1));
        qualifying.truncate(MAX_BRIDGE_CANDIDATES);

        for i in 0..qualifying.len() {
            for j in (i + 1)..qualifying.len() {
                let (a, ca) = &qualifying[i];
                let (b, cb) = &qualifying[j];
                let bridge = ca.min(cb) / 2;
                if bridge < 2 {
                    continue;
                }
                let cur_ab = bridges.entry(a.clone()).or_default().get(b).copied().unwrap_or(0);
                if bridge > cur_ab {
                    bridges.entry(a.clone()).or_default().insert(b.clone(), bridge);
                }
                let cur_ba = bridges.entry(b.clone()).or_default().get(a).copied().unwrap_or(0);
                if bridge > cur_ba {
                    bridges.entry(b.clone()).or_default().insert(a.clone(), bridge);
                }
            }
        }
    }
    bridges
}

/// Mine the last `max_commits` non-merge commits reachable from HEAD.
pub fn mine_history(
    repo_path: &Path,
    max_commits: usize,
    current_files: Option<&HashSet<String>>,
) -> HistoryData {
    if !repo_path.exists() {
        return HistoryData::default();
    }

    let pretty = format!("--pretty=format:{SENTINEL}%at%x00%an%x00%s%n%b");
    let output = Command::new("git")
        .args([
            "log",
            "--no-merges",
            "-n",
            &max_commits.to_string(),
            &pretty,
            "--name-only",
        ])
        .current_dir(repo_path)
        .output();
    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return HistoryData::default(),
    };
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if stdout.is_empty() {
        return HistoryData::default();
    }

    let lines: Vec<&str> = crate::pyutil::py_splitlines(&stdout);
    let headers: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, ln)| ln.starts_with(SENTINEL))
        .map(|(i, _)| i)
        .collect();

    let mut msgs: IndexMap<String, Vec<String>> = IndexMap::new();
    let mut cochange_counts: IndexMap<String, OrderedCounter> = IndexMap::new();
    let mut n_commits: OrderedCounter = IndexMap::new();
    let mut last_ts: IndexMap<String, i64> = IndexMap::new();
    let mut authors: IndexMap<String, OrderedCounter> = IndexMap::new();

    for (idx, &start) in headers.iter().enumerate() {
        let end = headers.get(idx + 1).copied().unwrap_or(lines.len());
        let header = &lines[start][SENTINEL.len()..];
        let mut parts = header.splitn(3, '\u{0}');
        let ts_str = parts.next().unwrap_or("");
        let author = parts.next().unwrap_or("");
        let subject = parts.next().unwrap_or("");
        let ts: i64 = ts_str.parse().unwrap_or(0);

        let (_msg, mut files) = parse_commit(subject, &lines[start + 1..end]);
        if let Some(fs) = current_files {
            files.retain(|f| fs.contains(f));
        }
        let n_files_total = files.len(); // pre-code-filter count, matches Python's `len(files)`
        let code_files: Vec<String> = files.into_iter().filter(|f| is_code_file(f)).collect();
        if code_files.is_empty() {
            continue;
        }
        for f in &code_files {
            counter_incr(&mut n_commits, f, 1);
            last_ts.entry(f.clone()).or_insert(ts);
            counter_incr(authors.entry(f.clone()).or_default(), author, 1);
            let list = msgs.entry(f.clone()).or_default();
            if list.len() < MAX_MSGS_PER_FILE {
                list.push(_msg.clone());
            }
        }
        if n_files_total <= BULK_COMMIT_FILE_LIMIT {
            let mut uniq: Vec<&String> = code_files.iter().collect::<HashSet<_>>().into_iter().collect();
            uniq.sort();
            if uniq.len() >= 2 {
                for i in 0..uniq.len() {
                    for j in (i + 1)..uniq.len() {
                        let a = uniq[i].clone();
                        let b = uniq[j].clone();
                        counter_incr(cochange_counts.entry(a.clone()).or_default(), &b, 1);
                        counter_incr(cochange_counts.entry(b).or_default(), &a, 1);
                    }
                }
            }
        }
    }

    let mut out_msgs: IndexMap<String, String> = IndexMap::new();
    for (f, parts) in &msgs {
        let text = parts.join("\n");
        let truncated: String = text.chars().take(MAX_MSG_CHARS).collect();
        out_msgs.insert(f.clone(), truncated);
    }

    let mut out_cochange: IndexMap<String, IndexMap<String, i64>> = IndexMap::new();
    for (f, counter) in &cochange_counts {
        let top: Vec<(String, i64)> = most_common(counter)
            .into_iter()
            .filter(|(_, n)| *n >= MIN_COCHANGE_COUNT)
            .take(MAX_COCHANGE_PARTNERS)
            .collect();
        if !top.is_empty() {
            let m: IndexMap<String, i64> = top.into_iter().collect();
            out_cochange.insert(f.clone(), m);
        }
    }

    let bridges = bridge_cochange(&cochange_counts);
    for (f, bcounter) in &bridges {
        let mut top_bridge_v = most_common(bcounter);
        top_bridge_v.truncate(MAX_COCHANGE_PARTNERS);
        if top_bridge_v.is_empty() {
            continue;
        }
        let top_bridge: IndexMap<String, i64> = top_bridge_v.into_iter().collect();
        let mut merged: IndexMap<String, i64> = out_cochange.get(f).cloned().unwrap_or_default();
        for (o, c) in &top_bridge {
            let cur = merged.get(o).copied().unwrap_or(0);
            merged.insert(o.clone(), cur.max(*c));
        }
        let mut merged_v: Vec<(String, i64)> = merged.into_iter().collect();
        merged_v.sort_by(|a, b| b.1.cmp(&a.1));
        merged_v.truncate(MAX_COCHANGE_PARTNERS);
        out_cochange.insert(f.clone(), merged_v.into_iter().collect());
    }

    let mut out_meta: IndexMap<String, FileMeta> = IndexMap::new();
    for (f, &n) in &n_commits {
        let mut auth_v = most_common(authors.get(f).unwrap());
        auth_v.truncate(MAX_AUTHORS_PER_FILE);
        out_meta.insert(
            f.clone(),
            FileMeta {
                n_commits: n,
                last_ts: *last_ts.get(f).unwrap_or(&0),
                authors: auth_v.into_iter().collect(),
            },
        );
    }

    HistoryData {
        msgs: out_msgs,
        cochange: out_cochange,
        meta: out_meta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn looks_like_path_cases() {
        assert!(looks_like_path("a/b.py"));
        assert!(looks_like_path("foo.py"));
        assert!(!looks_like_path(".hidden"));
        assert!(!looks_like_path("has space"));
        assert!(!looks_like_path(""));
        assert!(!looks_like_path("plainword"));
    }

    #[test]
    fn parse_commit_splits_file_list() {
        let rest = vec!["body line 1", "", "a/b.py", "c/d.py"];
        let (msg, files) = parse_commit("subject", &rest);
        assert_eq!(msg, "subject\nbody line 1");
        assert_eq!(files, vec!["a/b.py", "c/d.py"]);
    }

    #[test]
    fn parse_commit_no_file_list_block() {
        // last block doesn't parse as bare paths -> treated as body
        let rest = vec!["not a path list here"];
        let (msg, files) = parse_commit("subject", &rest);
        assert_eq!(msg, "subject\nnot a path list here");
        assert!(files.is_empty());
    }

    // -------------------------------------------------- E6: file_line_recency

    fn git(repo: &Path, args: &[&str]) {
        let status = Command::new("git").arg("-C").arg(repo).args(args).status().expect("failed to run git");
        assert!(status.success(), "git {args:?} failed in {repo:?}");
    }

    fn commit_all(repo: &Path, msg: &str) {
        git(repo, &["-c", "user.email=test@test.invalid", "-c", "user.name=test", "add", "-A"]);
        git(
            repo,
            &["-c", "user.email=test@test.invalid", "-c", "user.name=test", "commit", "-q", "-m", msg],
        );
    }

    /// A 3-commit fixture repo with one file, `f.py`, whose 3 lines are each
    /// last touched by a DIFFERENT commit, oldest to newest: line 1 (never
    /// touched again after the initial commit), line 2 (touched by the 2nd
    /// commit), line 3 (touched by the 3rd/newest commit). Mirrors
    /// `tests/incremental.rs`'s `make_git_repo` convention (per-invocation
    /// `-c user.email=/-c user.name=`, no reliance on global git config).
    fn make_recency_repo(tag: &str) -> PathBuf {
        let repo = std::env::temp_dir().join(format!("roust_recency_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&repo);
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q"]);
        std::fs::write(repo.join("f.py"), "a = 1\nb = 1\nc = 1\n").unwrap();
        commit_all(&repo, "test: create f.py"); // line 1, 2, 3 all last-touched here initially
        std::fs::write(repo.join("f.py"), "a = 1\nb = 2\nc = 1\n").unwrap();
        commit_all(&repo, "test: touch line 2"); // line 2 now last-touched here
        std::fs::write(repo.join("f.py"), "a = 1\nb = 2\nc = 3\n").unwrap();
        commit_all(&repo, "test: touch line 3"); // line 3 now last-touched here
        repo
    }

    /// Hand-computed expected decay values for `make_recency_repo`: 3
    /// commits total, newest-first rank c2=0, c1=1, c0=2, so `denom =
    /// max(3-1,1) = 2` and `a = 1 - rank/denom`. Line 1 (last touched by
    /// c0, a=0.0) -> raw `recency_decay(0.0) = 1/(1+e^12)`; line 2 (c1,
    /// a=0.5) -> `1/(1+e^6)`; line 3 (c2, a=1.0) -> `1/(1+e^0) = 0.5`
    /// (the maximum any single term can reach, since `a` is capped to
    /// `[0, 1]`). Normalizing per-file by the max raw value (line 3's 0.5)
    /// gives line3 = 1.0 exactly, and lines 1/2 as the ratios below.
    #[test]
    fn file_line_recency_matches_hand_computed_decay() {
        let repo = make_recency_repo("hand");
        let map = file_line_recency(&repo, "f.py");

        let raw_line1 = 1.0 / (1.0 + (12.0_f64).exp()); // a=0.0
        let raw_line2 = 1.0 / (1.0 + (6.0_f64).exp()); // a=0.5
        let raw_line3 = 0.5; // a=1.0 -> 1/(1+e^0)
        let expected1 = raw_line1 / raw_line3;
        let expected2 = raw_line2 / raw_line3;
        let expected3 = 1.0;

        assert!((map[&1] - expected1).abs() < 1e-9, "line1: {} vs {expected1}", map[&1]);
        assert!((map[&2] - expected2).abs() < 1e-9, "line2: {} vs {expected2}", map[&2]);
        assert!((map[&3] - expected3).abs() < 1e-9, "line3: {} vs {expected3}", map[&3]);
        // Monotonic recency ordering: newer touch -> strictly higher score.
        assert!(map[&1] < map[&2]);
        assert!(map[&2] < map[&3]);

        std::fs::remove_dir_all(&repo).ok();
    }

    /// Determinism: two independent calls over the same repo state produce
    /// byte-identical (not just numerically-close) maps.
    #[test]
    fn file_line_recency_deterministic() {
        let repo = make_recency_repo("determinism");
        let map1 = file_line_recency(&repo, "f.py");
        let map2 = file_line_recency(&repo, "f.py");
        assert_eq!(map1.len(), map2.len());
        for (k, v) in &map1 {
            assert_eq!(*v, map2[k], "line {k} diverged across calls");
        }
        std::fs::remove_dir_all(&repo).ok();
    }

    /// No git repo / unknown file -> empty map (default-path safe fallback),
    /// never a panic.
    #[test]
    fn file_line_recency_missing_repo_is_empty() {
        let missing = std::env::temp_dir().join(format!("roust_recency_missing_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&missing);
        std::fs::create_dir_all(&missing).unwrap();
        let map = file_line_recency(&missing, "nope.py");
        assert!(map.is_empty());
        std::fs::remove_dir_all(&missing).ok();
    }

    /// `recency_decay`'s own closed-form shape: strictly increasing in `a`
    /// over `[0, 1]`, with the documented max-at-`a=1` value of exactly 0.5
    /// (not 1.0) -- callers must normalize per-file rather than assume raw
    /// scores already sit in `[0, 1]`.
    #[test]
    fn recency_decay_shape() {
        assert!((recency_decay(1.0) - 0.5).abs() < 1e-12);
        assert!(recency_decay(0.0) < 1e-5);
        assert!(recency_decay(0.0) < recency_decay(0.5));
        assert!(recency_decay(0.5) < recency_decay(1.0));
    }
}
