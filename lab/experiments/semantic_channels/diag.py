"""Headroom diagnostic for two history channels on bgrep's residual @10 misses.

SIGNAL A - hunk index (locus): does the concatenated historical diff-hunk text
of a missing gold file rank it above the current top-10 (by a local BM25 over
{gold + top10}) when the current-file-content text does not?

SIGNAL B - translation table (IBM Model-1-lite): does expanding the query with
commit-message -> path-subtoken "translations" mined per-repo move a missing
gold file into the top-3 of a local content-BM25 re-rank of {gold + top10}
when the unexpanded query does not?

Read-only: every git call takes an explicit <rev> and touches no working tree
(`git show <rev>:<path>`, `git log <rev> -- <path>`), so this is safe to run
concurrently with another agent doing checkouts in the same repos.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

LAB_DIR = Path("/Users/nicholasarehart/programming-projects/bgrep/lab")
sys.path.insert(0, str(LAB_DIR))
from lanes2 import tokenize, path_tokens, stem, subtokens  # noqa: E402

RESULTS = LAB_DIR / "results_swebench" / "abl_bridges_v7.jsonl"
SCRATCH = Path("/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/3ab12e71-fab2-4a81-bb2b-84700d211ef2/scratchpad/bgrep_lab")
PARQUET = SCRATCH / "swebench_lite.parquet"
REPOS_DIR = SCRATCH / "swebench_repos"
OUT_DIR = SCRATCH / "hunk_diag"

HUNK_N_COMMITS = 40
MIN3 = 3

# ---------------------------------------------------------------- git helpers

def _run(args: list[str], cwd: Path, timeout: int = 60) -> str:
    r = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        return ""
    return r.stdout


def git_show(repo: Path, rev: str, path: str) -> str:
    return _run(["git", "show", f"{rev}:{path}"], cwd=repo)


_DIFF_HEADER_PREFIXES = ("+++", "---", "diff --git", "index ", "@@", "new file",
                          "deleted file", "old mode", "new mode", "similarity index",
                          "rename from", "rename to", "copy from", "copy to")


def git_hunk_doc(repo: Path, rev: str, path: str, n: int = HUNK_N_COMMITS) -> str:
    """Concatenate added+removed hunk lines (diff syntax stripped) from the
    last `n` commits (reachable from `rev`) that touched `path`, following
    renames. No working-tree state needed: `rev` is passed explicitly."""
    out = _run(
        ["git", "log", rev, "-n", str(n), "-p", "--follow", "--", path],
        cwd=repo, timeout=90,
    )
    if not out:
        return ""
    lines = []
    for ln in out.splitlines():
        if ln.startswith(_DIFF_HEADER_PREFIXES):
            continue
        if ln.startswith("+") or ln.startswith("-"):
            lines.append(ln[1:])
    return "\n".join(lines)


# ---------------------------------------------------------------- local BM25

def local_bm25(query: list[str], doc_toks: dict[str, list[str]]) -> dict[str, float]:
    """BM25 over a small ad-hoc document set (idf computed WITHIN this set --
    only relative ranking among {gold + top10} matters here, not cross-corpus
    comparability)."""
    docs = list(doc_toks.keys())
    n = len(docs)
    if n == 0:
        return {}
    tf = {d: Counter(t) for d, t in doc_toks.items()}
    doclen = {d: len(t) for d, t in doc_toks.items()}
    avg_len = (sum(doclen.values()) / n) if n else 1.0
    df: Counter[str] = Counter()
    qset = set(query)
    for d in docs:
        for term in qset:
            if tf[d][term] > 0:
                df[term] += 1
    k1, b = 1.2, 0.75
    scores = {d: 0.0 for d in docs}
    for term in qset:
        dfreq = df[term]
        if dfreq == 0:
            continue
        idf = math.log((n - dfreq + 0.5) / (dfreq + 0.5) + 1.0)
        for d in docs:
            f = tf[d][term]
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * doclen[d] / avg_len)
            scores[d] += idf * (f * (k1 + 1) / denom)
    return scores


def rank_of(target: str, scores: dict[str, float]) -> int:
    """1-indexed rank of `target` by descending score (ties broken by doc
    order i.e. insertion order of `scores`, stable)."""
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    for i, (d, _s) in enumerate(ordered, start=1):
        if d == target:
            return i
    return len(scores) + 1


# ---------------------------------------------------------------- load inputs

def load_failing():
    import pandas as pd
    recs = [json.loads(l) for l in open(RESULTS)]
    meta = pd.read_parquet(PARQUET).set_index("instance_id")
    fails = []
    for r in recs:
        top10 = r["returned_files"][:10]
        missing = [g for g in r["gold_files"] if g not in top10]
        if not missing:
            continue
        iid = r["instance_id"]
        if iid not in meta.index:
            continue
        row = meta.loc[iid]
        fails.append({
            "instance_id": iid,
            "repo": r["repo"],
            "top10": top10,
            "missing_gold": missing,
            "problem_statement": row["problem_statement"],
            "base_commit": row["base_commit"],
        })
    return fails


def repo_dir(repo: str) -> Path:
    return REPOS_DIR / repo.replace("/", "__")


# ---------------------------------------------------------------- Signal A

def signal_a(fails: list[dict]) -> tuple[list[dict], dict, float]:
    t0 = time.time()
    rows = []
    cache: dict[str, dict] = {}
    for f in fails:
        repo = repo_dir(f["repo"])
        rev = f["base_commit"]
        query = tokenize(f["problem_statement"])
        gold = f["missing_gold"][0]
        candidates = list(dict.fromkeys(f["top10"] + [gold]))  # dedup, keep order

        content_toks: dict[str, list[str]] = {}
        hunk_toks: dict[str, list[str]] = {}
        for c in candidates:
            content = git_show(repo, rev, c)
            content_toks[c] = tokenize(content) if content else []
            hdoc = git_hunk_doc(repo, rev, c)
            hunk_toks[c] = tokenize(hdoc) if hdoc else []

        content_scores = local_bm25(query, content_toks)
        hunk_scores = local_bm25(query, hunk_toks)
        rank_content = rank_of(gold, content_scores)
        rank_hunk = rank_of(gold, hunk_scores)
        headroom = rank_hunk <= MIN3 and rank_content > MIN3

        rows.append({
            "instance_id": f["instance_id"],
            "repo": f["repo"],
            "gold": gold,
            "n_candidates": len(candidates),
            "rank_content": rank_content,
            "rank_hunk": rank_hunk,
            "content_score": round(content_scores.get(gold, 0.0), 3),
            "hunk_score": round(hunk_scores.get(gold, 0.0), 3),
            "hunk_doc_empty": len(hunk_toks[gold]) == 0,
            "headroom": headroom,
        })
        cache[f["instance_id"]] = {
            "content_toks": content_toks,
            "query": query,
            "rank_content": rank_content,
        }
    return rows, cache, time.time() - t0


# ---------------------------------------------------------------- Signal B

def mine_translation_table(repo: Path) -> tuple[dict[str, Counter], Counter, int]:
    """Returns (t_table[msg_token] -> Counter(code_token -> raw co-occur
    count), path_token_df (over current tree at HEAD), n_files_at_head).
    msg tokens = tokenize(commit subject); code tokens = union of
    path_tokens() over every changed file path in that commit. Counted once
    per (commit, msg_token, code_token) pair (both sides de-duped to sets
    first) so a commit with a long subject or many files can't dominate via
    repetition alone."""
    out = _run(
        ["git", "log", "HEAD", "-n", "3000", "--no-merges",
         "--pretty=format:__C__%s", "--name-only"],
        cwd=repo, timeout=120,
    )
    co: dict[str, Counter] = defaultdict(Counter)
    if out:
        lines = out.splitlines()
        headers = [i for i, ln in enumerate(lines) if ln.startswith("__C__")]
        for idx, start in enumerate(headers):
            end = headers[idx + 1] if idx + 1 < len(headers) else len(lines)
            subject = lines[start][len("__C__"):]
            files = [ln.strip() for ln in lines[start + 1:end] if ln.strip()]
            if not files or len(files) > 20:
                continue
            msg_toks = set(tokenize(subject))
            if not msg_toks:
                continue
            code_toks: set[str] = set()
            for fp in files:
                code_toks |= path_tokens(fp)
            if not code_toks:
                continue
            for mt in msg_toks:
                for ct in code_toks:
                    co[mt][ct] += 1

    # path-token df over the current tree (cheap corpus-presence gate --
    # "appears-in-repo-with-df<10%", no gold peeking).
    ls = _run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repo, timeout=60)
    all_paths = [p for p in ls.splitlines() if p.strip()]
    df: Counter[str] = Counter()
    for p in all_paths:
        for t in path_tokens(p):
            df[t] += 1
    n_files = len(all_paths) or 1
    return co, df, n_files


def translate_query(query: list[str], t_table: dict[str, Counter],
                     df: Counter, n_files: int, top_k: int = 15) -> list[str]:
    agg: Counter[str] = Counter()
    for qt in set(query):
        partners = t_table.get(qt)
        if not partners:
            continue
        total = sum(partners.values())
        if total == 0:
            continue
        for ct, c in partners.items():
            agg[ct] += c / total  # t(code | msg) summed over query tokens
    gated = [ct for ct, _s in agg.most_common()
             if (df.get(ct, 0) / n_files) < 0.10]
    return gated[:top_k]


def signal_b(fails: list[dict], a_cache: dict) -> tuple[list[dict], dict, float]:
    t0 = time.time()
    repo_counts = Counter(f["repo"] for f in fails)
    top_repos = [r for r, _ in repo_counts.most_common(4)]

    rows = []
    examples: dict[str, list] = {}
    for repo_name in top_repos:
        repo = repo_dir(repo_name)
        t_table, df, n_files = mine_translation_table(repo)

        # sanity examples: a few interesting/common msg tokens
        sample_tokens = [t for t in ["migrat", "serializ", "deprecat", "regress",
                                      "queryset", "widget", "backend", "signal"]
                          if t in t_table]
        if not sample_tokens:
            sample_tokens = [t for t, _ in Counter(
                {k: sum(v.values()) for k, v in t_table.items()}
            ).most_common(6)]
        ex = []
        for t in sample_tokens[:6]:
            partners = t_table[t]
            total = sum(partners.values())
            top = [
                {"code_token": ct, "t_prob": round(c / total, 3),
                 "path_df_frac": round(df.get(ct, 0) / n_files, 3),
                 "kept_by_gate": (df.get(ct, 0) / n_files) < 0.10}
                for ct, c in partners.most_common(8)
            ]
            ex.append({"msg_token": t, "top_translations": top})
        examples[repo_name] = ex

        for f in fails:
            if f["repo"] != repo_name:
                continue
            key = f["instance_id"]
            cached = a_cache[key]
            gold = f["missing_gold"][0]
            content_toks = cached["content_toks"]
            query = cached["query"]
            rank_content_base = cached["rank_content"]

            expansion = translate_query(query, t_table, df, n_files)
            expanded_query = list(dict.fromkeys(query + expansion))
            expanded_scores = local_bm25(expanded_query, content_toks)
            rank_expanded = rank_of(gold, expanded_scores)
            headroom = rank_expanded <= MIN3 and rank_content_base > MIN3

            rows.append({
                "instance_id": f["instance_id"],
                "repo": repo_name,
                "gold": gold,
                "rank_content_base": rank_content_base,
                "rank_content_expanded": rank_expanded,
                "n_expansion_terms": len(expansion),
                "expansion_terms": expansion,
                "headroom": headroom,
            })
    return rows, examples, time.time() - t0


# ---------------------------------------------------------------- main

def main():
    fails = load_failing()
    print(f"failing @10 instances (gold missing from returned_files[:10]): {len(fails)}", file=sys.stderr)

    a_rows, a_cache, a_wall = signal_a(fails)

    b_rows, b_examples, b_wall = signal_b(fails, a_cache)

    a_headroom = sum(1 for r in a_rows if r["headroom"])
    b_headroom = sum(1 for r in b_rows if r["headroom"])

    report = {
        "n_failing_instances": len(fails),
        "signal_a": {
            "headroom_count": a_headroom,
            "headroom_pct_of_failing": round(100 * a_headroom / len(fails), 1),
            "wall_s": round(a_wall, 1),
            "rows": a_rows,
        },
        "signal_b": {
            "repos_evaluated": [r for r in Counter(f["repo"] for f in fails).most_common(4)],
            "n_instances_covered": len(b_rows),
            "headroom_count": b_headroom,
            "headroom_pct_of_covered": round(100 * b_headroom / len(b_rows), 1) if b_rows else 0.0,
            "wall_s": round(b_wall, 1),
            "translation_examples": b_examples,
            "rows": b_rows,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "n_failing": len(fails),
        "signal_a_headroom": a_headroom,
        "signal_a_wall_s": round(a_wall, 1),
        "signal_b_covered": len(b_rows),
        "signal_b_headroom": b_headroom,
        "signal_b_wall_s": round(b_wall, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
