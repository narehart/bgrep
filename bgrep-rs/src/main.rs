//! bgrep-rs command-line interface -- a Rust port of `src/bgrep/cli.py`.
//!
//!     bgrep-rs [--json] [--files-only] [--budget N=8192] [--k N]
//!              [--no-history] [--no-docs] [--no-anchors] [--no-testbridge]
//!              [--explain] QUERY PATH
//!
//! Runs the frozen-v7 retrieval pipeline (bgrep_rs::core, bgrep_rs::history)
//! against a repo and prints a token-budgeted, region-packed bundle of the
//! files most relevant to QUERY.
//!
//! Default output (stdout) is the packed bundle text. A one-line stats
//! summary always goes to stderr, never stdout, so stdout stays
//! pipeable/parseable.
//!
//! Exit codes: 0 = results found, 1 = no results, 2 = usage error.

use bgrep_rs::core::{self, extract_symbol_anchors, pack_regions, query_terms, select_files, Corpus, SelectParams};
use bgrep_rs::history::mine_history;
use clap::Parser;
use indexmap::IndexMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "bgrep-rs",
    about = "Recall-first code retrieval for LLM coding agents -- grep-level recall at ~1% of the tokens."
)]
struct Args {
    /// natural-language query or issue text (pass it raw -- identifiers,
    /// error strings, and other anchors are signal, don't pre-clean it)
    query: String,

    /// repo path to search (default: .)
    #[arg(default_value = ".")]
    path: String,

    /// token budget for the packed bundle
    #[arg(long, default_value_t = 8192)]
    budget: i64,

    /// cap the number of returned files, 0 = no cap
    #[arg(long, default_value_t = 0)]
    k: i64,

    /// print ranked file paths only, one per line, instead of the packed bundle
    #[arg(long)]
    files_only: bool,

    /// machine-readable JSON output instead of the packed bundle
    #[arg(long)]
    json: bool,

    /// disable the git-history commit-message field + co-change frontier expansion
    #[arg(long)]
    no_history: bool,

    /// disable the docs-bridge signal (*.rst/*.txt/*.md page indexing + bridging)
    #[arg(long)]
    no_docs: bool,

    /// disable the definition-symbol anchor channel
    #[arg(long)]
    no_anchors: bool,

    /// disable the test-file lexical bridge channel
    #[arg(long)]
    no_testbridge: bool,

    /// dump the Explain diagnostic record as JSON to stderr
    #[arg(long)]
    explain: bool,
}

fn main() {
    let args = Args::parse();

    if args.budget <= 0 {
        eprintln!("bgrep-rs: error: --budget must be positive");
        std::process::exit(2);
    }
    if args.k < 0 {
        eprintln!("bgrep-rs: error: --k must be >= 0");
        std::process::exit(2);
    }

    let repo_path = PathBuf::from(&args.path);
    if !repo_path.is_dir() {
        eprintln!("bgrep-rs: error: not a directory: {}", args.path);
        std::process::exit(2);
    }

    let with_history = !args.no_history;
    let with_docs = !args.no_docs;
    let use_anchors = !args.no_anchors;
    let use_testbridge = !args.no_testbridge;

    let t0 = Instant::now();

    let history = if with_history {
        let current_files = list_current_files(&repo_path);
        Some(mine_history(&repo_path, 5000, Some(&current_files)))
    } else {
        None
    };
    let history_msgs: Option<&IndexMap<String, String>> = history.as_ref().map(|h| &h.msgs);

    let corpus = Corpus::build(&repo_path, history_msgs, false, with_docs);
    let index_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let terms = query_terms(&args.query, &[]);
    let anchors = if use_anchors { Some(extract_symbol_anchors(&args.query, &corpus)) } else { None };
    let cochange = if with_history {
        history.as_ref().map(|h| &h.cochange)
    } else {
        None
    };

    let params = SelectParams {
        cochange,
        anchors: anchors.as_deref(),
        use_testbridge,
        use_docsbridge: with_docs,
        ..Default::default()
    };
    let (mut files, scores, explain) = select_files(&corpus, &terms, true, &params);
    if args.k > 0 {
        files.truncate(args.k as usize);
    }

    let encoder = tiktoken_rs::cl100k_base_singleton();
    let count_tokens = |text: &str| -> usize { encoder.lock().encode_ordinary(text).len() };

    let (spans, bundle) = pack_regions(&corpus, &files, &terms, &scores, args.budget, &count_tokens);
    let query_ms = t1.elapsed().as_secs_f64() * 1000.0;

    if args.explain {
        eprintln!("{}", serde_json::to_string_pretty(&explain).unwrap());
    }

    let packed_files: Vec<String> = files.iter().filter(|f| spans.contains_key(f.as_str())).cloned().collect();
    let bundle_tokens = if !bundle.is_empty() { count_tokens(&bundle) } else { 0 };

    if !packed_files.is_empty() {
        if args.json {
            let files_json: Vec<serde_json::Value> = packed_files
                .iter()
                .enumerate()
                .map(|(i, f)| serde_json::json!({"path": f, "score_rank": i}))
                .collect();
            let regions_json: serde_json::Map<String, serde_json::Value> = packed_files
                .iter()
                .map(|f| {
                    let rs = &spans[f.as_str()];
                    let v: Vec<Vec<usize>> = rs.iter().map(|(a, b)| vec![*a, *b]).collect();
                    (f.clone(), serde_json::json!(v))
                })
                .collect();
            let payload = serde_json::json!({
                "query": args.query,
                "files": files_json,
                "regions": regions_json,
                "bundle": bundle,
                "stats": {
                    "files_indexed": corpus.n_docs,
                    "index_ms": index_ms.round() as i64,
                    "query_ms": query_ms.round() as i64,
                    "bundle_tokens": bundle_tokens,
                    "cache": "miss",
                },
            });
            println!("{}", serde_json::to_string(&payload).unwrap());
        } else if args.files_only {
            for f in &packed_files {
                println!("{f}");
            }
        } else {
            println!("{bundle}");
        }
    }

    eprintln!(
        "bgrep-rs: {} files, {} tokens (indexed {} files, index {}ms, query {}ms, cache miss)",
        packed_files.len(),
        bundle_tokens,
        corpus.n_docs,
        index_ms.round() as i64,
        query_ms.round() as i64,
    );

    if packed_files.is_empty() {
        std::process::exit(1);
    }
}

/// Cheap mirror of Corpus's file-collection filter (extension, .git, size
/// cap) without reading/tokenizing file contents -- matches
/// swebench_driver2.py's `_list_current_files`.
fn list_current_files(repo_path: &Path) -> HashSet<String> {
    let mut files = HashSet::new();
    fn recurse(dir: &Path, base: &Path, out: &mut HashSet<String>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
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
                if !is_file {
                    continue;
                }
                let rel = match path.strip_prefix(base) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let relstr = match rel.to_str() {
                    Some(s) => s.replace('\\', "/"),
                    None => continue,
                };
                if relstr.starts_with(".git/") || relstr.contains("/.git/") {
                    continue;
                }
                if !core::CODE_EXTENSIONS.iter().any(|ext| relstr.ends_with(ext)) {
                    continue;
                }
                if let Ok(meta) = std::fs::metadata(&path) {
                    if meta.len() > core::MAX_FILE_BYTES {
                        continue;
                    }
                }
                out.insert(relstr);
            }
        }
    }
    recurse(repo_path, repo_path, &mut files);
    files
}
