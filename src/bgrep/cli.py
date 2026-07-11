"""bgrep command-line interface.

    bgrep QUERY [PATH]

Runs the frozen-v7 retrieval pipeline (bgrep.core, bgrep.cache) against a
repo and prints a token-budgeted, region-packed bundle of the files most
relevant to QUERY -- grep-level recall at a fraction of grep's tokens, no
LLM/embeddings/network involved.

Default output (stdout) is the packed bundle text. A one-line stats summary
always goes to stderr, never stdout, so stdout stays pipeable/parseable.

Exit codes: 0 = results found, 1 = no results, 2 = usage error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from bgrep import cache as cache_mod
from bgrep.core import extract_symbol_anchors, get_token_counter, pack_regions, query_terms, select_files


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bgrep",
        description=(
            "Recall-first code retrieval for LLM coding agents -- "
            "grep-level recall at ~1% of the tokens."
        ),
    )
    ap.add_argument(
        "query",
        help="natural-language query or issue text (pass it raw -- identifiers, "
             "error strings, and other anchors are signal, don't pre-clean it)",
    )
    ap.add_argument("path", nargs="?", default=".", help="repo path to search (default: .)")
    ap.add_argument("--budget", type=int, default=8192,
                     help="token budget for the packed bundle (default: 8192)")
    ap.add_argument("--k", type=int, default=0,
                     help="cap the number of returned files, 0 = no cap (default: 0)")
    ap.add_argument("--files-only", action="store_true",
                     help="print ranked file paths only, one per line, instead of the packed bundle")
    ap.add_argument("--json", action="store_true",
                     help="machine-readable JSON output instead of the packed bundle")
    ap.add_argument("--no-cache", action="store_true",
                     help="do not read or write the on-disk index cache (<repo>/.bgrep/)")
    ap.add_argument("--reindex", action="store_true",
                     help="force a fresh index build even if a matching cache entry exists "
                          "(still writes the cache afterward unless combined with --no-cache)")
    ap.add_argument("--no-history", action="store_true",
                     help="disable the git-history commit-message field + co-change frontier expansion")
    ap.add_argument("--no-docs", action="store_true",
                     help="disable the docs-bridge signal (*.rst/*.txt/*.md page indexing + bridging)")
    ap.add_argument("--no-anchors", action="store_true",
                     help="disable the definition-symbol anchor channel")
    ap.add_argument("--no-testbridge", action="store_true",
                     help="disable the test-file lexical bridge channel")
    ap.add_argument("--explain", action="store_true",
                     help="dump the Explain diagnostic record (bgrep.core.Explain) as JSON to stderr")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)  # argparse exits(2) on usage errors, exits(0) on --help

    if args.budget <= 0:
        ap.error("--budget must be positive")
    if args.k < 0:
        ap.error("--k must be >= 0")

    repo_path = Path(args.path).resolve()
    if not repo_path.is_dir():
        print(f"bgrep: error: not a directory: {args.path}", file=sys.stderr)
        return 2

    with_history = not args.no_history
    with_docs = not args.no_docs
    use_anchors = not args.no_anchors
    use_testbridge = not args.no_testbridge

    t0 = time.perf_counter()
    corpus, _edges, history, cache_hit = cache_mod.load_or_build(
        repo_path,
        with_history=with_history,
        with_docs=with_docs,
        use_cache=not args.no_cache,
        force_reindex=args.reindex,
    )
    index_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    terms = query_terms(args.query, [])
    anchors = extract_symbol_anchors(args.query, corpus) if use_anchors else None
    cochange = history[1] if (with_history and history) else None
    files, scores, explain = select_files(
        corpus, terms, use_ppr=True, cochange=cochange, anchors=anchors,
        use_testbridge=use_testbridge, use_docsbridge=with_docs,
    )
    if args.k:
        files = files[: args.k]
    count_tokens = get_token_counter()
    spans, bundle = pack_regions(corpus, files, terms, scores, args.budget, count_tokens)
    query_ms = (time.perf_counter() - t1) * 1000

    if args.explain:
        print(json.dumps(dataclasses.asdict(explain), indent=2), file=sys.stderr)

    packed_files = [f for f in files if f in spans]
    bundle_tokens = count_tokens(bundle) if bundle else 0
    cache_state = "hit" if cache_hit else "miss"

    if packed_files:
        if args.json:
            payload = {
                "query": args.query,
                "files": [{"path": f, "score_rank": i} for i, f in enumerate(packed_files)],
                "regions": {f: [list(span) for span in spans[f]] for f in packed_files},
                "bundle": bundle,
                "stats": {
                    "files_indexed": corpus.n_docs,
                    "index_ms": round(index_ms),
                    "query_ms": round(query_ms),
                    "bundle_tokens": bundle_tokens,
                    "cache": cache_state,
                },
            }
            print(json.dumps(payload))
        elif args.files_only:
            for f in packed_files:
                print(f)
        else:
            print(bundle)

    stats_line = (
        f"bgrep: {len(packed_files)} files, {bundle_tokens} tokens "
        f"(indexed {corpus.n_docs} files, index {index_ms:.0f}ms, query {query_ms:.0f}ms, "
        f"cache {cache_state})"
    )
    print(stats_line, file=sys.stderr)

    return 0 if packed_files else 1


if __name__ == "__main__":
    sys.exit(main())
