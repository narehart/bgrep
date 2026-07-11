"""Git-history mining for the semantic retrieval layer.

Single `git log` pass over the checked-out repo's recent non-merge commit
history. Produces three signals consumed by core.Corpus / core.select_files:

  msgs      - per-file concatenated commit subjects+bodies (a lightweight
              "commit message field" that often names symbols/behavior a
              patch touches even when the file's own source text doesn't),
              capped to each file's MAX_MSGS_PER_FILE most recent commits.
  cochange  - per-file top co-committed partner files with counts (files
              that historically change together are structurally related
              even absent an import edge), PLUS derived "bridge" edges
              between two production files that both co-change with the
              same test-like file. Production files mostly co-change with
              their OWN test file (which select_files' impl_prior excludes
              from the candidate pool), so direct production<->production
              co-change edges are sparse; a shared test file is evidence
              two production files are related even though they never
              appear together in a commit.
  meta      - per-file {"n_commits", "last_ts", "authors"} summary (not yet
              consumed by any scoring path; mined alongside msgs/cochange
              since it's a free byproduct of the same git-log pass).

No network access, no LLM: this is pure `git log` + string processing.

Parsing note: `git log --pretty=format:__C__%at%x00%an%x00%s%n%b --name-only`
emits, per commit: a sentinel-prefixed header line (author unix timestamp,
author name, subject -- NUL-separated), then the raw body (which may itself
contain blank lines between paragraphs), then a git-inserted blank-line
separator, then the changed-file list (one path per line), then another
blank-line separator (or EOF) before the next commit. Blank lines alone can't
distinguish a body paragraph break from the pre-file-list separator, but the
file list is always the LAST blank-line-delimited block before the next
sentinel/EOF, and (unlike prose) every line in it is a bare path with no
spaces -- so we use "last block, if it parses as bare paths" as the file-list
detector and everything before it as the message body. This resolves the
ambiguity without a second git-log pass.
"""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from bgrep.core import CODE_EXTENSIONS, _TESTLIKE_RE

_SENTINEL = "__C__"
MAX_MSG_CHARS = 40_000
MAX_MSGS_PER_FILE = 25
BULK_COMMIT_FILE_LIMIT = 20
MIN_COCHANGE_COUNT = 3
MAX_COCHANGE_PARTNERS = 10
MAX_BRIDGE_CANDIDATES = 50  # per test file, perf safety valve (see _bridge_cochange)
MAX_AUTHORS_PER_FILE = 5


def _is_code_file(rel: str) -> bool:
    return rel.endswith(CODE_EXTENSIONS)


def _looks_like_path(ln: str) -> bool:
    s = ln.strip()
    if not s or " " in s or "\t" in s:
        return False
    return "/" in s or ("." in s and not s.startswith("."))


def _split_blocks(lines: list[str]) -> list[list[str]]:
    """Split a list of lines into blank-line-delimited blocks (blanks dropped,
    empty leading/trailing blocks dropped)."""
    blocks: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)
    return blocks


def _parse_commit(subject: str, rest: list[str]) -> tuple[str, list[str]]:
    """rest = every raw line after the header line, up to (excluding) the
    next commit's sentinel line. Returns (message, files)."""
    blocks = _split_blocks(rest)
    if not blocks:
        return subject, []
    last = blocks[-1]
    if all(_looks_like_path(ln) for ln in last):
        files = [ln.strip() for ln in last]
        body_blocks = blocks[:-1]
    else:
        files = []
        body_blocks = blocks
    body = "\n\n".join("\n".join(b) for b in body_blocks)
    message = subject if not body else f"{subject}\n{body}"
    return message, files


def _bridge_cochange(
    cochange_counts: dict[str, Counter[str]],
) -> dict[str, Counter[str]]:
    """Derive production<->production "bridge" edges via a shared test-like
    co-change partner: if impl files a and b both co-change with the same
    test-like file t (each with count >= MIN_COCHANGE_COUNT), add an
    effective a<->b edge with count = min(count_a, count_b) // 2 (kept only
    if >= 2). `cochange_counts` is the raw, untrimmed global co-commit
    counter (already scoped to commits touching <= BULK_COMMIT_FILE_LIMIT
    files, from the caller). Each test file's qualifying partner list is
    capped to MAX_BRIDGE_CANDIDATES (highest-count first) before generating
    pairs, purely to bound combinatorial blowup for hub test files with
    hundreds of historical partners -- since bridge strength is
    min(count_a, count_b), dropping only the lowest-count long-tail
    partners can't remove any pair that would otherwise survive the
    MIN_COCHANGE_COUNT/`>= 2` thresholds ahead of a kept one.
    """
    bridges: dict[str, Counter[str]] = defaultdict(Counter)
    for t, partners in cochange_counts.items():
        if not _TESTLIKE_RE.search(t):
            continue
        qualifying = sorted(
            (
                (f, c) for f, c in partners.items()
                if c >= MIN_COCHANGE_COUNT and not _TESTLIKE_RE.search(f)
            ),
            key=lambda kv: -kv[1],
        )[:MAX_BRIDGE_CANDIDATES]
        for (a, ca), (b, cb) in combinations(qualifying, 2):
            bridge = min(ca, cb) // 2
            if bridge < 2:
                continue
            if bridge > bridges[a][b]:
                bridges[a][b] = bridge
            if bridge > bridges[b][a]:
                bridges[b][a] = bridge
    return bridges


def mine_history(
    repo_path: Path,
    max_commits: int = 5000,
    current_files: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, dict]]:
    """Mine the last `max_commits` non-merge commits reachable from HEAD.

    Because this runs on the already-checked-out repo (`git log` starts from
    HEAD), commits are naturally bounded to those <= the current checkout's
    commit -- no future leakage relative to whatever base_commit the caller
    checked out beforehand.

    `current_files`, if given, restricts all three return values to files
    that still exist in the current checkout (renamed/deleted paths from
    history are dropped since they can't be retrieval candidates).

    Returns (msgs, cochange, meta):
      msgs[file]     -> str, newest-first concatenated commit messages,
                         capped to MAX_MSGS_PER_FILE commits and MAX_MSG_CHARS.
      cochange[file] -> {other_file: count}, top MAX_COCHANGE_PARTNERS,
                         merging direct co-commit edges and test-bridge edges.
      meta[file]     -> {"n_commits": int, "last_ts": int (unix),
                          "authors": {name: count} top MAX_AUTHORS_PER_FILE}
    """
    if not repo_path.exists():
        return {}, {}, {}

    r = subprocess.run(
        [
            "git", "log", "--no-merges", "-n", str(max_commits),
            f"--pretty=format:{_SENTINEL}%at%x00%an%x00%s%n%b",
            "--name-only",
        ],
        cwd=repo_path, capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0 or not r.stdout:
        return {}, {}, {}

    lines = r.stdout.splitlines()
    # Locate sentinel-prefixed header lines. A commit subject/body could in
    # principle contain a line that itself starts with the sentinel; guard by
    # only treating a line as a new commit header when we're not already
    # inside an unterminated file-list block (best-effort -- see note above).
    headers: list[int] = [i for i, ln in enumerate(lines) if ln.startswith(_SENTINEL)]

    msgs: dict[str, list[str]] = defaultdict(list)
    cochange_counts: dict[str, Counter[str]] = defaultdict(Counter)
    n_commits: Counter[str] = Counter()
    last_ts: dict[str, int] = {}
    authors: dict[str, Counter[str]] = defaultdict(Counter)
    fileset = current_files

    for idx, start in enumerate(headers):
        end = headers[idx + 1] if idx + 1 < len(headers) else len(lines)
        header = lines[start][len(_SENTINEL):]
        ts_str, _, header_rest = header.partition("\x00")
        author, _, subject = header_rest.partition("\x00")
        try:
            ts = int(ts_str)
        except ValueError:
            ts = 0
        _msg, files = _parse_commit(subject, lines[start + 1: end])
        if fileset is not None:
            files = [f for f in files if f in fileset]
        code_files = [f for f in files if _is_code_file(f)]
        if not code_files:
            continue
        for f in code_files:
            n_commits[f] += 1
            if f not in last_ts:  # git log is newest-first: first seen = most recent
                last_ts[f] = ts
            authors[f][author] += 1
            if len(msgs[f]) < MAX_MSGS_PER_FILE:
                msgs[f].append(_msg)
        if len(files) <= BULK_COMMIT_FILE_LIMIT and len(code_files) >= 2:
            for a, b in combinations(sorted(set(code_files)), 2):
                cochange_counts[a][b] += 1
                cochange_counts[b][a] += 1

    out_msgs: dict[str, str] = {}
    for f, parts in msgs.items():
        # commits are newest-first (git log default order), so this keeps the
        # most recent messages first when truncated to MAX_MSG_CHARS.
        text = "\n".join(parts)
        out_msgs[f] = text[:MAX_MSG_CHARS]

    out_cochange: dict[str, dict[str, int]] = {}
    for f, counter in cochange_counts.items():
        top = [(o, n) for o, n in counter.most_common() if n >= MIN_COCHANGE_COUNT][:MAX_COCHANGE_PARTNERS]
        if top:
            out_cochange[f] = dict(top)

    bridges = _bridge_cochange(cochange_counts)
    for f, bcounter in bridges.items():
        top_bridge = dict(sorted(bcounter.items(), key=lambda kv: -kv[1])[:MAX_COCHANGE_PARTNERS])
        if not top_bridge:
            continue
        merged = dict(out_cochange.get(f, {}))
        for o, c in top_bridge.items():
            merged[o] = max(merged.get(o, 0), c)
        out_cochange[f] = dict(sorted(merged.items(), key=lambda kv: -kv[1])[:MAX_COCHANGE_PARTNERS])

    out_meta: dict[str, dict] = {
        f: {
            "n_commits": n_commits[f],
            "last_ts": last_ts[f],
            "authors": dict(authors[f].most_common(MAX_AUTHORS_PER_FILE)),
        }
        for f in n_commits
    }

    return out_msgs, out_cochange, out_meta
