//! On-disk index cache for roust, stored under `<repo>/.roust/` -- a Rust
//! port of `roust.cache` (Python commit 16e7c71). See that module's
//! docstring for the full design rationale; the summary:
//!
//! Corpus construction (a full file walk + read + tokenize pass over every
//! candidate file in the repo) is the expensive part of a run; this module
//! caches that work (plus the import graph and, optionally, mined git
//! history) keyed on the repo's git HEAD sha (`"nogit"` if not a git repo)
//! plus the `with_history`/`with_docs` flags a query was run with.
//!
//! Rather than keying on a hash of every file's stat (which forces a full
//! rebuild on ANY change, however small), the cache stores a manifest --
//! `{relpath: (mtime_ns, size)}` for every candidate-extension file -- and on
//! load, diffs a fresh stat-only walk against it to classify the working
//! tree since the cache was written:
//!
//!   - unchanged: no work at all, the cached Corpus is returned as-is.
//!   - modified only (the common agent edit-loop case: existing files'
//!     content changed, nothing added/removed): the cached Corpus is
//!     PATCHED in place (`Corpus::update_files`/`update_docs_files`, then
//!     `core::update_import_graph_for_files`) instead of rebuilt.
//!   - HEAD moved, or any file added/removed: a conservative FULL REBUILD.
//!
//! ## Serialization format
//!
//! Serialized with `serde_json` rather than a binary format (e.g. bincode):
//! `serde_json` is already a direct dependency of this crate (used for
//! `--json`/`--explain` output), so caching adds no new dependency or
//! version-compatibility surface, and every type on the cache payload
//! (`Corpus`, `EdgeMap`, `HistoryData`) already derives plain
//! `serde::{Serialize, Deserialize}` with no custom (de)serialization code
//! needed. The payload is written to a fixed filename regardless of format,
//! so this choice is an internal implementation detail a future pass is
//! free to swap for a binary format (bincode, etc.) purely for size/speed
//! without changing this module's public API.
//!
//! ## Cache-file isolation from the Python implementation
//!
//! The Python cache (`roust.cache`) writes `<repo>/.roust/index.pkl`
//! (a pickle). This module writes `<repo>/.roust/rust-index.bin` -- a
//! DIFFERENT filename in the same directory, deliberately, so the two
//! independent implementations never attempt to read each other's cache
//! file (a pickle is not valid JSON and vice versa; even if it were, the
//! two Corpus shapes are not identical types). Running both the Python
//! `roust` console script and this crate's `roust` binary against the same
//! repo is therefore always safe -- each maintains its own cache entry.

use crate::core::{self, build_import_graph, update_import_graph_for_files, Corpus, EdgeMap};
use crate::history::{mine_history, HistoryData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const CACHE_VERSION: i64 = 3;
pub const CACHE_DIRNAME: &str = ".roust";
const INDEX_FILENAME: &str = "rust-index.bin";

/// `{relpath: (mtime_ns, size)}` for every candidate-extension file, as of
/// the snapshot a cached Corpus was built from.
pub type Manifest = HashMap<String, (i64, u64)>;

#[derive(Serialize, Deserialize)]
struct CachePayload {
    version: i64,
    key: String,
    corpus: Corpus,
    edges: EdgeMap,
    history: Option<HistoryData>,
    manifest: Manifest,
}

/// Serialize-only mirror of `CachePayload` holding references instead of
/// owned data, so `save()` never has to clone the (potentially large)
/// `Corpus`/`EdgeMap`/`HistoryData` the caller still needs to return.
#[derive(Serialize)]
struct CachePayloadRef<'a> {
    version: i64,
    key: &'a str,
    corpus: &'a Corpus,
    edges: &'a EdgeMap,
    history: &'a Option<HistoryData>,
    manifest: &'a Manifest,
}

fn git_head_sha(repo_path: &Path) -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_path)
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let sha = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if sha.is_empty() {
                "nogit".to_string()
            } else {
                sha
            }
        }
        _ => "nogit".to_string(),
    }
}

/// Stat-only pass (no file reads) over every candidate-extension file,
/// returning `{relpath: (mtime_ns, size)}`. Uses `core::walk_all_files` --
/// the SAME git-ls-files-first enumeration (falling back to a raw
/// filesystem walk outside a git work tree) that `Corpus::build` itself
/// walks -- so the manifest's file set matches exactly what the Corpus
/// indexes. This is what makes a .gitignore'd file appearing on disk (e.g.
/// inside .venv/) invisible to change detection: git ls-files never lists
/// it, so it was never a candidate in the first place, and its creation
/// can't flip the verdict away from "unchanged". Deliberately cheaper and
/// coarser than `Corpus::build`'s own per-file filtering (no vendor-regex /
/// oversize / long-line filtering) -- a spuriously-"changed" entry just
/// costs an extra reindex of that one file (or, for add/remove, a full
/// rebuild), which is always safe; it can never cause a stale hit.
fn scan_manifest(repo_path: &Path, with_docs: bool) -> Manifest {
    let mut exts: HashSet<&str> = core::CODE_EXTENSIONS.iter().copied().collect();
    if with_docs {
        exts.extend(core::DOCS_EXTENSIONS.iter().copied());
    }
    let mut manifest = Manifest::new();
    for rel in core::walk_all_files(repo_path) {
        if rel.starts_with(".git/") || rel.contains("/.git/") {
            continue;
        }
        if rel.starts_with(&format!("{CACHE_DIRNAME}/")) || rel.contains(&format!("/{CACHE_DIRNAME}/")) {
            continue;
        }
        if !exts.contains(core::suffix_of(&rel)) {
            continue;
        }
        let full = repo_path.join(&rel);
        let meta = match std::fs::metadata(&full) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_file() {
            continue;
        }
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        manifest.insert(rel, (mtime_ns, meta.len()));
    }
    manifest
}

enum Verdict {
    Unchanged,
    Modified,
    Full,
}

/// Stat-walk the repo's current candidate files and diff against `manifest`
/// (the snapshot the cached Corpus was built from). See `Verdict`.
fn classify_changes(repo_path: &Path, with_docs: bool, manifest: &Manifest) -> (Verdict, Manifest, Vec<String>) {
    let current = scan_manifest(repo_path, with_docs);
    let old_keys: HashSet<&String> = manifest.keys().collect();
    let new_keys: HashSet<&String> = current.keys().collect();
    if !old_keys.is_subset(&new_keys) || !new_keys.is_subset(&old_keys) {
        return (Verdict::Full, current, Vec::new());
    }
    let modified: Vec<String> = current
        .iter()
        .filter(|(k, v)| manifest.get(*k) != Some(v))
        .map(|(k, _)| k.clone())
        .collect();
    if modified.is_empty() {
        return (Verdict::Unchanged, current, Vec::new());
    }
    (Verdict::Modified, current, modified)
}

fn cache_key(repo_path: &Path, with_history: bool, with_docs: bool) -> String {
    let sha = git_head_sha(repo_path);
    format!("{sha}:h{}:d{}", with_history as i32, with_docs as i32)
}

fn cache_path(repo_path: &Path) -> PathBuf {
    repo_path.join(CACHE_DIRNAME).join(INDEX_FILENAME)
}

fn load(repo_path: &Path, key: &str) -> Option<CachePayload> {
    let path = cache_path(repo_path);
    if !path.exists() {
        return None;
    }
    let file = std::fs::File::open(&path).ok()?;
    let reader = std::io::BufReader::new(file);
    let payload: CachePayload = serde_json::from_reader(reader).ok()?;
    if payload.version != CACHE_VERSION || payload.key != key {
        return None;
    }
    Some(payload)
}

/// Cache directory not writable (read-only checkout, permissions, disk
/// full, ...): degrade to "no cache" rather than fail the query, mirroring
/// `roust.cache._save`'s `except OSError: pass`.
fn save(repo_path: &Path, key: &str, corpus: &Corpus, edges: &EdgeMap, history: &Option<HistoryData>, manifest: &Manifest) {
    let cache_dir = repo_path.join(CACHE_DIRNAME);
    if std::fs::create_dir_all(&cache_dir).is_err() {
        return;
    }
    let payload = CachePayloadRef { version: CACHE_VERSION, key, corpus, edges, history, manifest };
    let final_path = cache_path(repo_path);
    let tmp_path = cache_dir.join(format!("{INDEX_FILENAME}.tmp"));
    let write_result: std::io::Result<()> = (|| {
        let file = std::fs::File::create(&tmp_path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, &payload).map_err(std::io::Error::other)?;
        Ok(())
    })();
    match write_result {
        Ok(()) => {
            let _ = std::fs::rename(&tmp_path, &final_path);
        }
        Err(_) => {
            let _ = std::fs::remove_file(&tmp_path);
        }
    }
}

/// Cheap mirror of `Corpus::build`'s file-collection filter (extension,
/// `.git`/`.roust` exclusion) without reading/tokenizing file contents --
/// matches `roust.cache._build_fresh`'s `current_files` comprehension. Uses
/// `core::walk_all_files` (git-ls-files-first, same as `Corpus::build` and
/// `scan_manifest`) so a .gitignore'd file is never fed to history mining as
/// a "current" file either.
fn collect_current_code_files(repo_path: &Path) -> HashSet<String> {
    let mut files = HashSet::new();
    for rel in core::walk_all_files(repo_path) {
        if rel.starts_with(".git/") || rel.contains("/.git/") {
            continue;
        }
        if rel.starts_with(&format!("{CACHE_DIRNAME}/")) || rel.contains(&format!("/{CACHE_DIRNAME}/")) {
            continue;
        }
        if !core::is_code_file(&rel) {
            continue;
        }
        let full = repo_path.join(&rel);
        match std::fs::metadata(&full) {
            Ok(m) if m.is_file() => {}
            _ => continue,
        }
        files.insert(rel);
    }
    files
}

/// Attempt to patch `corpus`/`edges` in place for `modified` (files whose
/// content changed but whose relpath set didn't). All-or-nothing: if either
/// field's update is declined, returns `false` immediately. A partial patch
/// (code side succeeded, docs side declined) can leave `corpus` mutated but
/// inconsistent with a fresh build -- callers MUST discard `corpus`/`edges`
/// entirely (never save or return them) whenever this returns `false`,
/// falling back to a full rebuild instead.
fn try_incremental_update(corpus: &mut Corpus, edges: &mut EdgeMap, modified: &[String]) -> bool {
    let code_rels: Vec<String> = modified.iter().filter(|r| core::CODE_EXTENSIONS.contains(&core::suffix_of(r))).cloned().collect();
    let docs_rels: Vec<String> = modified.iter().filter(|r| core::DOCS_EXTENSIONS.contains(&core::suffix_of(r))).cloned().collect();

    // Defensive: `scan_manifest` doesn't apply Corpus::build's own
    // vendor/size/oversized-line filters, so a "modified" rel can name a
    // file that was never actually indexed (e.g. under a vendor/ path).
    // roust.cache's Python equivalent has no such guard and would raise a
    // KeyError in this situation; declining incrementally (forcing the
    // always-safe full-rebuild fallback) is strictly more robust and never
    // changes the observable result, so this is a deliberate hardening, not
    // a parity deviation.
    if code_rels.iter().any(|r| !corpus.text.contains_key(r)) {
        return false;
    }
    if docs_rels.iter().any(|r| !corpus.docs_text.contains_key(r)) {
        return false;
    }

    let old_text: HashMap<String, String> = code_rels.iter().map(|r| (r.clone(), corpus.text[r].clone())).collect();
    if !code_rels.is_empty() && !corpus.update_files(&code_rels) {
        return false;
    }
    if !docs_rels.is_empty() && !corpus.update_docs_files(&docs_rels) {
        return false;
    }
    if !code_rels.is_empty() {
        update_import_graph_for_files(corpus, edges, &old_text);
    }
    true
}

fn build_fresh(repo_path: &Path, with_history: bool, with_docs: bool) -> (Corpus, EdgeMap, Option<HistoryData>) {
    let history = if with_history {
        let current_files = collect_current_code_files(repo_path);
        Some(mine_history(repo_path, 5000, Some(&current_files)))
    } else {
        None
    };
    let history_msgs = history.as_ref().map(|h| &h.msgs);
    let corpus = Corpus::build(repo_path, history_msgs, false, with_docs);
    let edges = build_import_graph(&corpus);
    (corpus, edges, history)
}

/// Same as `load_or_build`, plus a 5th `update_kind` element -- one of
/// `"unchanged"` (cache hit, no work), `"incremental"` (an existing cached
/// Corpus was patched in place for modified files, no full rebuild) or
/// `"full"` (a fresh Corpus was built from scratch) -- exposed for
/// tests/tooling that need to assert which path was actually taken.
/// `load_or_build` itself collapses this to the boolean `cache_hit`.
pub fn load_or_build_ex(
    repo_path: &Path,
    with_history: bool,
    with_docs: bool,
    use_cache: bool,
    force_reindex: bool,
) -> (Corpus, EdgeMap, Option<HistoryData>, bool, &'static str) {
    let key = cache_key(repo_path, with_history, with_docs);

    if use_cache && !force_reindex {
        if let Some(payload) = load(repo_path, &key) {
            let CachePayload { mut corpus, mut edges, history, manifest, .. } = payload;
            let (verdict, new_manifest, modified) = classify_changes(repo_path, with_docs, &manifest);
            match verdict {
                Verdict::Unchanged => return (corpus, edges, history, true, "unchanged"),
                Verdict::Modified => {
                    if try_incremental_update(&mut corpus, &mut edges, &modified) {
                        save(repo_path, &key, &corpus, &edges, &history, &new_manifest);
                        return (corpus, edges, history, true, "incremental");
                    }
                    // Shaped like an add/remove after all (declined patch):
                    // `corpus`/`edges` above may be partially mutated and
                    // must not be reused -- fall through to a full rebuild.
                }
                Verdict::Full => {}
            }
        }
    }

    let (corpus, edges, history) = build_fresh(repo_path, with_history, with_docs);
    let manifest = scan_manifest(repo_path, with_docs);
    if use_cache {
        save(repo_path, &key, &corpus, &edges, &history, &manifest);
    }
    (corpus, edges, history, false, "full")
}

/// Load a cached `(Corpus, import-graph edges, history)` triple for
/// `repo_path`, else build it fresh and (unless `use_cache` is false) save
/// it. Returns `(corpus, edges, history_or_none, cache_hit)`. `cache_hit`
/// is true whenever a fresh Corpus build was avoided -- i.e. both the
/// "unchanged" and "incremental" cases from `load_or_build_ex` collapse to
/// `true` here; only "full" is `false`.
pub fn load_or_build(
    repo_path: &Path,
    with_history: bool,
    with_docs: bool,
    use_cache: bool,
    force_reindex: bool,
) -> (Corpus, EdgeMap, Option<HistoryData>, bool) {
    let (corpus, edges, history, cache_hit, _update_kind) = load_or_build_ex(repo_path, with_history, with_docs, use_cache, force_reindex);
    (corpus, edges, history, cache_hit)
}
