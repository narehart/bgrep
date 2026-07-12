"""Mine a Scala file-localization benchmark from real GitHub issues.

For each target repo: search merged PRs that are linked to an issue
(`is:pr is:merged linked:issue`), recover the issue each PR closes by
parsing "Fixes/Closes/Resolves #N" (or full issue-URL) references out of
the PR body, then keep the pair if:

  - the issue body has >= 200 chars of real text
  - the PR is merged and has a resolvable base commit sha
  - the PR touches 1-5 non-test .scala files (test-path and doc/comment-only
    changes excluded from gold_files)

Output instance shape matches SWE-bench:
  {instance_id, repo, base_commit, problem_statement, gold_files}

Usage:  uv run python mine_scala.py [--out scala_loc.jsonl] [--cap-per-repo 10]
                                     [--target-total 55] [--pages-per-repo 6]

Requires an authenticated `gh` CLI (checked at startup).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

REPOS = [
    "akka/akka-core",             # akka/akka renamed -> akka/akka-core
    "typelevel/cats",
    "typelevel/cats-effect",
    "playframework/playframework",
    "scala/scala3",                # formerly lampepfl/dotty
    "zio/zio",
    "apache/pekko",
    "circe/circe",
    "com-lihaoyi/mill",
    "sbt/sbt",
]

TEST_PATH_RE = re.compile(
    r"(^|/)(test|tests|it|src/test)(/|$)|Test\.scala$|Tests\.scala$|"
    r"Spec\.scala$|Specs\.scala$|TestSuite\.scala$",
    re.IGNORECASE,
)

# "Fixes #123", "Closes: #123", "resolved https://github.com/org/repo/issues/123"
CLOSES_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s+"
    r"(?:#(?P<num1>\d+)|https?://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/issues/(?P<num2>\d+))",
    re.IGNORECASE,
)

SEARCH_SLEEP = 2.5   # search API: 30 req/min secondary limit
CORE_SLEEP = 0.35    # core REST API: generous, but be polite (secondary abuse limits)
MAX_RETRIES = 5


def gh_api(args: list[str], paginate: bool = False) -> object:
    """Run `gh api ...` and return parsed JSON, retrying on rate limits."""
    cmd = ["gh", "api"] + (["--paginate"] if paginate else []) + args
    for attempt in range(MAX_RETRIES):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out = proc.stdout.strip()
            if not out:
                return [] if paginate else {}
            if paginate:
                # gh merges array-returning endpoints into one JSON array when
                # possible; fall back to concatenated-documents parsing just
                # in case a given gh version prints one array per page.
                try:
                    return json.loads(out)
                except json.JSONDecodeError:
                    decoder = json.JSONDecoder()
                    merged: list = []
                    idx = 0
                    while idx < len(out):
                        while idx < len(out) and out[idx] in " \n\r\t":
                            idx += 1
                        if idx >= len(out):
                            break
                        obj, end = decoder.raw_decode(out, idx)
                        merged.extend(obj if isinstance(obj, list) else [obj])
                        idx = end
                    return merged
            return json.loads(out)
        stderr = proc.stderr
        if "rate limit" in stderr.lower() or "secondary rate limit" in stderr.lower():
            wait = 10 * (attempt + 1)
            print(f"  rate limited, sleeping {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        if "404" in stderr or "Not Found" in stderr:
            return None
        # transient / unknown error: brief backoff then retry
        print(f"  gh api error (attempt {attempt+1}): {stderr.strip()[:200]}", file=sys.stderr)
        time.sleep(3)
    return None


def search_merged_prs(repo: str, page: int) -> list[dict]:
    res = gh_api([
        "search/issues", "--method", "GET",
        "-f", f"q=repo:{repo} is:pr is:merged linked:issue",
        "-f", f"per_page=100",
        "-f", f"page={page}",
    ])
    if not res:
        return []
    return res.get("items", [])


def find_closed_issue_number(repo: str, pr_body: str) -> int | None:
    if not pr_body:
        return None
    for m in CLOSES_RE.finditer(pr_body):
        if m.group("num1"):
            return int(m.group("num1"))
        owner, rname, num2 = m.group("owner"), m.group("repo"), m.group("num2")
        if owner and rname and num2 and f"{owner}/{rname}".lower() == repo.lower():
            return int(num2)
    return None


def is_comment_or_blank_only(patch: str) -> bool:
    """Heuristic: True if a unified diff patch has no real code changed lines
    (only blank lines, //, /*, *, or scaladoc changed)."""
    if not patch:
        return True
    changed = [
        line[1:].strip()
        for line in patch.splitlines()
        if line[:1] in ("+", "-") and not line.startswith(("+++", "---"))
    ]
    code_lines = [
        ln for ln in changed
        if ln and not ln.startswith(("//", "/*", "*", "*/"))
    ]
    return len(code_lines) == 0


def gold_files_from_pr(repo: str, pr_number: int) -> list[str] | None:
    files = gh_api(["repos/%s/pulls/%d/files" % (repo, pr_number)], paginate=True)
    if files is None:
        return None
    if isinstance(files, dict):
        files = [files]
    gold = []
    for f in files:
        path = f.get("filename", "")
        if not path.endswith(".scala"):
            continue
        if TEST_PATH_RE.search(path):
            continue
        if f.get("status") == "removed":
            continue
        if is_comment_or_blank_only(f.get("patch", "")):
            continue
        gold.append(path)
    return sorted(set(gold))


def mine_repo(repo: str, cap: int, pages: int, stats: dict) -> list[dict]:
    instances: list[dict] = []
    seen_prs: set[int] = set()
    r_stats = stats.setdefault(repo, {
        "scanned_prs": 0, "no_issue_link": 0, "issue_fetch_fail": 0,
        "issue_body_short": 0, "pr_fetch_fail": 0, "not_merged_or_no_sha": 0,
        "file_count_out_of_range": 0, "accepted": 0,
    })

    for page in range(1, pages + 1):
        if len(instances) >= cap:
            break
        items = search_merged_prs(repo, page)
        time.sleep(SEARCH_SLEEP)
        if not items:
            break
        for item in items:
            if len(instances) >= cap:
                break
            pr_number = item["number"]
            if pr_number in seen_prs:
                continue
            seen_prs.add(pr_number)
            r_stats["scanned_prs"] += 1

            issue_num = find_closed_issue_number(repo, item.get("body") or "")
            if issue_num is None:
                r_stats["no_issue_link"] += 1
                continue

            issue = gh_api([f"repos/{repo}/issues/{issue_num}"])
            time.sleep(CORE_SLEEP)
            if not issue or "pull_request" in issue:
                r_stats["issue_fetch_fail"] += 1
                continue
            body = (issue.get("body") or "").strip()
            if len(body) < 200:
                r_stats["issue_body_short"] += 1
                continue

            pr = gh_api([f"repos/{repo}/pulls/{pr_number}"])
            time.sleep(CORE_SLEEP)
            if not pr:
                r_stats["pr_fetch_fail"] += 1
                continue
            if not pr.get("merged") or not pr.get("base", {}).get("sha"):
                r_stats["not_merged_or_no_sha"] += 1
                continue
            base_sha = pr["base"]["sha"]

            gold = gold_files_from_pr(repo, pr_number)
            time.sleep(CORE_SLEEP)
            if gold is None or not (1 <= len(gold) <= 5):
                r_stats["file_count_out_of_range"] += 1
                continue

            title = issue.get("title") or ""
            problem_statement = f"{title}\n\n{body}"
            instance = {
                "instance_id": f"{repo.replace('/', '__')}-{pr_number}",
                "repo": repo,
                "base_commit": base_sha,
                "problem_statement": problem_statement,
                "gold_files": gold,
            }
            instances.append(instance)
            r_stats["accepted"] += 1
            print(f"  [{repo}] accepted PR #{pr_number} (issue #{issue_num}): "
                  f"{len(gold)} gold file(s)", file=sys.stderr)

    return instances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "scala_loc.jsonl"))
    ap.add_argument("--cap-per-repo", type=int, default=10)
    ap.add_argument("--target-total", type=int, default=55)
    ap.add_argument("--pages-per-repo", type=int, default=6)
    ap.add_argument("--repos", nargs="*", default=REPOS)
    args = ap.parse_args()

    auth = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if auth.returncode != 0:
        print("gh CLI is not authenticated. Run `gh auth login` first.", file=sys.stderr)
        sys.exit(1)

    all_instances: list[dict] = []
    stats: dict = {}
    for repo in args.repos:
        if len(all_instances) >= args.target_total:
            print(f"target of {args.target_total} reached, stopping repo scan", file=sys.stderr)
            break
        print(f"mining {repo}...", file=sys.stderr)
        try:
            instances = mine_repo(repo, args.cap_per_repo, args.pages_per_repo, stats)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR mining {repo}: {e}", file=sys.stderr)
            instances = []
        all_instances.extend(instances)
        print(f"  {repo}: {len(instances)} instances", file=sys.stderr)

    with open(args.out, "w") as f:
        for inst in all_instances:
            f.write(json.dumps(inst) + "\n")

    print(f"\nwrote {len(all_instances)} instances to {args.out}", file=sys.stderr)
    print(json.dumps(stats, indent=2), file=sys.stderr)
    with open(HERE / "mine_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
