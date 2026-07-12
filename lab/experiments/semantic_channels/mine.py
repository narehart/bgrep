"""Tracker k-NN headroom diagnostic.

For bgrep's residual SWE-bench Lite misses (File@10 and File@all failures on
abl_bridges_v7.jsonl), ask: do temporally-prior RESOLVED issues exist whose
text is lexically similar to the failing instance's problem_statement, and
whose linked commit touched the MISSING gold file before base_commit's date?

Read-only against the repo checkouts (only `git show -s --format=%cI`, no
checkout/clone) and against GitHub via `gh api` (read-only endpoints). All
outputs written under this tracker_diag/ dir.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

LAB = Path("/Users/nicholasarehart/programming-projects/bgrep/lab")
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/"
    "3ab12e71-fab2-4a81-bb2b-84700d211ef2/scratchpad/bgrep_lab"
)
RESULTS = LAB / "results_swebench" / "abl_bridges_v7.jsonl"
PARQUET = SCRATCH / "swebench_lite.parquet"
REPOS = SCRATCH / "swebench_repos"
OUT_DIR = SCRATCH / "tracker_diag"

sys.path.insert(0, str(LAB))
from lanes2 import tokenize  # noqa: E402

N_10ONLY_CAP = 15
COMMITS_PER_FILE = 10       # most-recent-before-base_date commits to inspect per missing file
REFS_PER_FILE_CAP = 8       # dedup'd issue/PR refs to resolve per missing file
CONTROL_N = 10
SLEEP_BETWEEN_CALLS = 0.05

_REF_RE = re.compile(r"#(\d+)")


# --------------------------------------------------------------------------- gh helpers

def gh_api(path: str, silent_404: bool = True):
    """Run `gh api <path>`, return parsed JSON or None on error (404 etc.)."""
    for attempt in range(3):
        proc = subprocess.run(
            ["gh", "api", path], capture_output=True, text=True
        )
        if proc.returncode == 0:
            time.sleep(SLEEP_BETWEEN_CALLS)
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                return None
        stderr = proc.stderr
        if "404" in stderr:
            if not silent_404:
                print(f"    404: {path}", file=sys.stderr)
            return None
        if "403" in stderr or "rate limit" in stderr.lower():
            wait = 5 * (attempt + 1)
            print(f"    rate-limited on {path}, sleeping {wait}s (attempt {attempt+1})", file=sys.stderr)
            time.sleep(wait)
            continue
        # other error - print once, give up
        print(f"    gh api error on {path}: {stderr.strip()[:200]}", file=sys.stderr)
        return None
    return None


def commits_touching(owner: str, repo: str, path: str, until_iso: str):
    q = f"repos/{owner}/{repo}/commits?path={path}&until={until_iso}&per_page=30"
    data = gh_api(q)
    if not data or not isinstance(data, list):
        return []
    return data


def issue_info(owner: str, repo: str, num: int):
    data = gh_api(f"repos/{owner}/{repo}/issues/{num}")
    if not data:
        return None
    return {
        "number": num,
        "title": data.get("title") or "",
        "body": data.get("body") or "",
        "state": data.get("state"),
        "closed_at": data.get("closed_at"),
        "html_url": data.get("html_url"),
        "pull_request": "pull_request" in data,
    }


def commit_pulls(owner: str, repo: str, sha: str):
    """PR(s) associated with a commit via GitHub's commit->PR index (more
    reliable than regex-parsing the commit message subject, which only
    catches the GitHub-generated '(#NNN)' squash-merge suffix)."""
    data = gh_api(f"repos/{owner}/{repo}/commits/{sha}/pulls")
    if not data or not isinstance(data, list):
        return []
    return [p.get("number") for p in data if p.get("number")]


# --------------------------------------------------------------------------- local git

def base_commit_date(owner: str, repo: str, sha: str) -> str | None:
    repo_dir = REPOS / f"{owner}__{repo}"
    if not (repo_dir / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "show", "-s", "--format=%cI", sha],
        cwd=str(repo_dir), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


# --------------------------------------------------------------------------- similarity

def parse_iso(s: str) -> datetime:
    """Parse an ISO8601 timestamp (either GitHub's UTC 'Z' form or git's
    local-offset '%cI' form) into a timezone-aware datetime for correct
    absolute-time comparison -- NEVER compare these strings lexicographically
    (a 'Z' string and a '+02:00' string sort wrong relative to each other
    even on the same calendar day)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tf_vector(text: str) -> Counter:
    try:
        toks = tokenize(text)
    except Exception:
        toks = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(toks)


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------- core mining

def mine_missing_file(owner: str, repo: str, missing_path: str, base_date: str,
                       query_text: str, log: list) -> dict:
    """Returns dict: {missing_path, n_commits, refs_checked, candidates: [...], best}"""
    commits = commits_touching(owner, repo, missing_path, base_date)[:COMMITS_PER_FILE]
    query_tf = tf_vector(query_text)
    base_date_dt = parse_iso(base_date)

    refs_seen: dict[int, None] = {}
    for c in commits:
        sha = c.get("sha")
        msg = c.get("commit", {}).get("message", "") or ""
        # (1) GitHub commit->PR association index -- catches the common case
        # of a normal (non-squash-suffixed) commit whose PR references the
        # original issue in its body.
        if sha:
            for pr_num in commit_pulls(owner, repo, sha):
                refs_seen.setdefault(pr_num, None)
        # (2) regex over the raw commit message -- catches squash-merge
        # '(#NNN)' suffixes and inline 'Fixes #NNN' trailers.
        for m in _REF_RE.finditer(msg):
            refs_seen.setdefault(int(m.group(1)), None)
        if len(refs_seen) >= REFS_PER_FILE_CAP:
            break

    ref_list = list(refs_seen.keys())[:REFS_PER_FILE_CAP]
    candidates = []
    seen_refs_from_pr_bodies: dict[int, None] = {}
    for num in ref_list:
        info = issue_info(owner, repo, num)
        if info is None:
            continue
        # a PR often names the *original* issue in its own body ("Fixes
        # #NNN") -- that original issue is usually closer in language to a
        # fresh problem_statement than the PR's own (terse) title, so pull
        # those refs in too (bounded by the same REFS_PER_FILE_CAP).
        if info["pull_request"] and len(refs_seen) < REFS_PER_FILE_CAP:
            for m in _REF_RE.finditer(info["title"] + "\n" + info["body"]):
                n2 = int(m.group(1))
                if n2 not in refs_seen:
                    refs_seen[n2] = None
                    seen_refs_from_pr_bodies[n2] = None
        if info["state"] != "closed" or not info["closed_at"]:
            continue
        if parse_iso(info["closed_at"]) >= base_date_dt:
            continue  # not temporally prior
        text = info["title"] + "\n" + info["body"]
        sim = cosine(query_tf, tf_vector(text))
        candidates.append({
            "ref": num,
            "sim": round(sim, 4),
            "closed_at": info["closed_at"],
            "title": info["title"][:120],
            "url": info["html_url"],
            "is_pr": info["pull_request"],
        })

    # resolve any additional refs pulled from PR bodies (already added to
    # refs_seen above, but issue_info wasn't called yet for the newly-added
    # ones if they came after the initial ref_list snapshot)
    for num in list(seen_refs_from_pr_bodies)[: max(0, REFS_PER_FILE_CAP - len(ref_list))]:
        if any(c["ref"] == num for c in candidates):
            continue
        info = issue_info(owner, repo, num)
        if info is None or info["state"] != "closed" or not info["closed_at"]:
            continue
        if parse_iso(info["closed_at"]) >= base_date_dt:
            continue
        text = info["title"] + "\n" + info["body"]
        sim = cosine(query_tf, tf_vector(text))
        candidates.append({
            "ref": num,
            "sim": round(sim, 4),
            "closed_at": info["closed_at"],
            "title": info["title"][:120],
            "url": info["html_url"],
            "is_pr": info["pull_request"],
        })

    candidates.sort(key=lambda c: -c["sim"])
    best = candidates[0] if candidates else None
    result = {
        "missing_path": missing_path,
        "n_commits_found": len(commits),
        "refs_checked": list(refs_seen.keys()),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best": best,
    }
    log.append(f"    {missing_path}: {len(commits)} commits, {len(refs_seen)} refs, "
               f"{len(candidates)} closed-prior candidates, best_sim="
               f"{best['sim'] if best else 'n/a'}")
    return result


# --------------------------------------------------------------------------- main

def load_results():
    rows = [json.loads(l) for l in open(RESULTS)]
    return rows


def at_k(rec, k):
    gold = set(rec["gold_files"])
    returned = rec["returned_files"][:k] if k is not None else rec["returned_files"]
    return gold.issubset(set(returned))


def main():
    import pandas as pd

    rows = load_results()
    by_id = {r["instance_id"]: r for r in rows}
    df = pd.read_parquet(PARQUET)
    meta = {row.instance_id: row for row in df.itertuples()}

    failall = [r for r in rows if not at_k(r, None)]
    fail10 = [r for r in rows if not at_k(r, 10)]
    fail10_only = sorted(
        [r for r in fail10 if at_k(r, None)],
        key=lambda r: r["instance_id"],
    )[:N_10ONLY_CAP]

    print(f"total instances: {len(rows)}")
    print(f"fail@10: {len(fail10)}  fail@all: {len(failall)}  "
          f"fail10_only selected (cap {N_10ONLY_CAP}): {len(fail10_only)}")

    targets = []  # (instance_id, missing_files, tier)
    for r in sorted(failall, key=lambda r: r["instance_id"]):
        targets.append((r["instance_id"], r["missing"], "@all"))
    for r in fail10_only:
        gold = set(r["gold_files"])
        top10 = set(r["returned_files"][:10])
        missing10 = sorted(gold - top10)
        targets.append((r["instance_id"], missing10, "@10-only"))

    per_instance_records = []
    all_best_sims = []
    log = []

    for inst_id, missing_files, tier in targets:
        m = meta.get(inst_id)
        if m is None:
            log.append(f"[{inst_id}] NOT FOUND in parquet, skip")
            continue
        owner, repo = m.repo.split("/")
        base_date = base_commit_date(owner, repo, m.base_commit)
        if base_date is None:
            log.append(f"[{inst_id}] no local repo checkout / bad sha, skip")
            continue
        print(f"[{tier}] {inst_id}  base_date={base_date}  missing={missing_files}")
        log.append(f"[{tier}] {inst_id}  base_date={base_date}  missing={missing_files}")

        file_results = []
        for g in missing_files:
            fr = mine_missing_file(owner, repo, g, base_date, m.problem_statement, log)
            file_results.append(fr)
            if fr["best"]:
                all_best_sims.append(fr["best"]["sim"])
            else:
                all_best_sims.append(0.0)

        per_instance_records.append({
            "instance_id": inst_id,
            "tier": tier,
            "repo": m.repo,
            "base_date": base_date,
            "missing_files": missing_files,
            "file_results": file_results,
        })

    # ---------------------------------------------------------------- control
    passing = [r for r in rows if r["all_present"]]
    passing_sorted = sorted(passing, key=lambda r: r["instance_id"])
    control_pairs = []
    for r in passing_sorted:
        for g in r["gold_files"]:
            control_pairs.append((r["instance_id"], g))
        if len(control_pairs) >= CONTROL_N:
            break
    control_pairs = control_pairs[:CONTROL_N]

    control_records = []
    control_best_sims = []
    print(f"\n-- control: {len(control_pairs)} non-missing gold files from passing instances --")
    log.append(f"\n-- control: {len(control_pairs)} non-missing gold files from passing instances --")
    for inst_id, g in control_pairs:
        r = by_id[inst_id]
        m = meta.get(inst_id)
        if m is None:
            continue
        owner, repo = m.repo.split("/")
        base_date = base_commit_date(owner, repo, m.base_commit)
        if base_date is None:
            log.append(f"[control] {inst_id} no local repo, skip")
            continue
        print(f"[control] {inst_id}  file={g}")
        log.append(f"[control] {inst_id}  file={g}  base_date={base_date}")
        fr = mine_missing_file(owner, repo, g, base_date, m.problem_statement, log)
        control_records.append({
            "instance_id": inst_id,
            "repo": m.repo,
            "base_date": base_date,
            "file": g,
            "file_result": fr,
        })
        control_best_sims.append(fr["best"]["sim"] if fr["best"] else 0.0)

    # ---------------------------------------------------------------- write outputs
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "target_results.json", "w") as f:
        json.dump(per_instance_records, f, indent=2)
    with open(OUT_DIR / "control_results.json", "w") as f:
        json.dump(control_records, f, indent=2)
    with open(OUT_DIR / "run_log.txt", "w") as f:
        f.write("\n".join(log))

    # ---------------------------------------------------------------- distribution / threshold
    def summarize(sims, label):
        sims_sorted = sorted(sims)
        n = len(sims_sorted)
        if n == 0:
            print(f"{label}: no data")
            return
        import statistics
        print(f"{label}: n={n}  min={sims_sorted[0]:.3f}  "
              f"p25={sims_sorted[n//4]:.3f}  median={statistics.median(sims_sorted):.3f}  "
              f"p75={sims_sorted[3*n//4]:.3f}  max={sims_sorted[-1]:.3f}  "
              f"mean={statistics.mean(sims_sorted):.3f}")
        print(f"  full sorted: {[round(s,3) for s in sims_sorted]}")

    print("\n=== best-sim distribution (target: @all + @10-only missing files) ===")
    summarize(all_best_sims, "target")
    print("\n=== best-sim distribution (control: non-missing gold files) ===")
    summarize(control_best_sims, "control")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "target_best_sims": all_best_sims,
            "control_best_sims": control_best_sims,
            "n_targets": len(all_best_sims),
            "n_controls": len(control_best_sims),
        }, f, indent=2)

    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
