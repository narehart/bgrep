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
use std::collections::HashSet;
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

#[derive(Debug, Clone, Default)]
pub struct FileMeta {
    pub n_commits: i64,
    pub last_ts: i64,
    pub authors: IndexMap<String, i64>,
}

#[derive(Debug, Clone, Default)]
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
}
