//! roust command-line interface -- a Rust port of `src/roust/cli.py`.
//!
//!     roust [--json] [--files-only] [--budget N=8192] [--k N]
//!              [--no-cache] [--reindex]
//!              [--no-history] [--no-docs] [--no-anchors] [--no-testbridge]
//!              [--explain] QUERY PATH
//!
//! Runs the frozen-v7 retrieval pipeline (roust::core, roust::cache,
//! roust::history) against a repo and prints a token-budgeted,
//! region-packed bundle of the files most relevant to QUERY.
//!
//! Default output (stdout) is the packed bundle text. A one-line stats
//! summary always goes to stderr, never stdout, so stdout stays
//! pipeable/parseable.
//!
//! Exit codes: 0 = results found (including low-confidence matches, flagged
//! via `low_confidence` in `--json` stats and a stderr warning), 1 = no
//! query term matched anything in the indexed corpus vocabulary, 2 = usage
//! error.

use roust::cache;
use roust::core::{
    anchor_def_symbols, extract_symbol_anchors, is_low_confidence, pack_regions, query_term_coverage, query_terms,
    select_files, SelectParams,
};
use clap::Parser;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

/// Engine provenance embedded at compile time by build.rs -- short git SHA
/// and a dirty flag scoped to `roust-rs/` paths (see build.rs). Powers
/// `roust --version` (e.g. `roust 0.2.0 (abc1234, clean)`) and the
/// `engine_sha`/`engine_dirty` fields in `--json` stats, so a stale `uv run
/// roust` wheel (doesn't rebuild on `roust-rs/src` changes) is always
/// identifiable rather than silently measured as current.
const ROUST_VERSION: &str =
    concat!(env!("CARGO_PKG_VERSION"), " (", env!("ROUST_GIT_SHA"), ", ", env!("ROUST_GIT_DIRTY"), ")");

#[derive(Parser, Debug)]
#[command(
    name = "roust",
    version = ROUST_VERSION,
    about = "Recall-first code retrieval for coding agents - one ranked, token-budgeted bundle per query, no model or API key required"
)]
struct Args {
    /// natural-language query or issue text (pass it raw -- identifiers,
    /// error strings, and other anchors are signal, don't pre-clean it)
    query: String,

    /// repo path to search (default: .)
    #[arg(default_value = ".")]
    path: String,

    /// token budget for the packed bundle (default 8192, sized for LLM
    /// context; humans reading by hand may prefer 2048)
    #[arg(long, default_value_t = 8192)]
    budget: i64,

    /// cap the number of returned files, 0 = no cap (humans reading by hand
    /// may prefer --k 8 for a tighter, scannable list)
    #[arg(long, default_value_t = 0)]
    k: i64,

    /// print ranked file paths only, one per line, instead of the packed bundle
    #[arg(long)]
    files_only: bool,

    /// machine-readable JSON output instead of the packed bundle
    #[arg(long)]
    json: bool,

    /// do not read or write the on-disk index cache (<repo>/.roust/)
    #[arg(long)]
    no_cache: bool,

    /// force a fresh index build even if a matching cache entry exists
    /// (still writes the cache afterward unless combined with --no-cache)
    #[arg(long)]
    reindex: bool,

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
        eprintln!("roust: error: --budget must be positive");
        std::process::exit(2);
    }
    if args.k < 0 {
        eprintln!("roust: error: --k must be >= 0");
        std::process::exit(2);
    }

    let repo_path = PathBuf::from(&args.path);
    if !repo_path.is_dir() {
        eprintln!("roust: error: not a directory: {}", args.path);
        std::process::exit(2);
    }

    let with_history = !args.no_history;
    let with_docs = !args.no_docs;
    let use_anchors = !args.no_anchors;
    let use_testbridge = !args.no_testbridge;

    let t0 = Instant::now();

    let (corpus, _edges, history, cache_hit) =
        cache::load_or_build(&repo_path, with_history, with_docs, !args.no_cache, args.reindex);
    let index_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let terms = query_terms(&args.query, &[]);
    let (matched_terms, total_terms) = query_term_coverage(&corpus, &terms);
    let zero_match = matched_terms == 0;
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

    let anchor_files: HashSet<String> = explain.anchor_promotions.iter().map(|(f, ..)| f.clone()).collect();
    let anchor_symbols = if anchor_files.is_empty() {
        indexmap::IndexMap::new()
    } else {
        anchor_def_symbols(&args.query, &corpus, &anchor_files)
    };
    let (spans, bundle) =
        pack_regions(&corpus, &files, &terms, &scores, args.budget, &count_tokens, Some(&anchor_symbols), 0.0);
    let query_ms = t1.elapsed().as_secs_f64() * 1000.0;

    if args.explain {
        eprintln!("{}", serde_json::to_string_pretty(&explain).unwrap());
    }

    let packed_files: Vec<String> = files.iter().filter(|f| spans.contains_key(f.as_str())).cloned().collect();
    let bundle_tokens = if !bundle.is_empty() { count_tokens(&bundle) } else { 0 };
    let cache_state = if cache_hit { "hit" } else { "miss" };
    let low_confidence = is_low_confidence(explain.top_score, matched_terms, total_terms);

    let mut stats = serde_json::json!({
        "files_indexed": corpus.n_docs,
        "index_ms": index_ms.round() as i64,
        "query_ms": query_ms.round() as i64,
        "bundle_tokens": bundle_tokens,
        "cache": cache_state,
        "top_score": explain.top_score,
        "matched_query_terms": matched_terms,
        "total_query_terms": total_terms,
        "engine_sha": env!("ROUST_GIT_SHA"),
        "engine_dirty": env!("ROUST_GIT_DIRTY") == "dirty",
    });
    if low_confidence {
        stats["low_confidence"] = serde_json::json!(true);
    }

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
                "stats": stats,
            });
            println!("{}", serde_json::to_string(&payload).unwrap());
        } else if args.files_only {
            for f in &packed_files {
                println!("{f}");
            }
        } else {
            println!("{bundle}");
        }
    } else if zero_match && args.json {
        // Literal zero-match case (issue #25): emit valid, parseable JSON
        // with an empty result set rather than nothing, so callers scripting
        // against --json never have to special-case "no stdout at all".
        let payload = serde_json::json!({
            "query": args.query,
            "files": [],
            "regions": {},
            "bundle": "",
            "stats": stats,
        });
        println!("{}", serde_json::to_string(&payload).unwrap());
    }

    // zero_match gets its own dedicated stderr message instead of the
    // generic low-confidence suffix -- it's the strictly stronger "nothing
    // in the corpus vocabulary matched at all" signal, not merely a weak
    // match.
    let confidence_note = if !zero_match && low_confidence { " [low-confidence match]" } else { "" };
    eprintln!(
        "roust: {} files, {} tokens (indexed {} files, index {}ms, query {}ms, cache {}){}",
        packed_files.len(),
        bundle_tokens,
        corpus.n_docs,
        index_ms.round() as i64,
        query_ms.round() as i64,
        cache_state,
        confidence_note,
    );
    if zero_match {
        eprintln!("roust: no query term matched anything in the indexed corpus vocabulary -- no results");
    }

    if zero_match {
        std::process::exit(1);
    }
}
