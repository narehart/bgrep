"""Shared utilities for the tokenbench harness: SWE-bench Lite instance
loading, repo clone/checkout, tiktoken-based token counting, and lenient
`FILES:` line parsing/scoring.

Deliberately self-contained (no dependency on lab/lanes2.py, lab/swebench_
driver2.py, or the archex package) so the harness in this directory has a
minimal, auditable surface: everything that affects the headline token-usage
claim lives in lab/tokenbench/.
"""

from __future__ import annotations

import re
import subprocess
import urllib.request
from pathlib import Path

import tiktoken

TOKENBENCH_DIR = Path(__file__).resolve().parent
LAB_DIR = TOKENBENCH_DIR.parent
REPO_ROOT = LAB_DIR.parent  # .../bgrep

# Repo clones are shared with the other lab/swebench_driver*.py scripts so we
# don't reclone repos that already exist on disk.
#
# ---------------------------------------------------------------------------
# SHARED-CLONE HAZARD (issue #41): checkout() below MUTATES these clones
# (git checkout -f + git clean -fdq). lab/swebench_repos/ has historically
# been ONE physical set of clones shared across every worktree of this repo
# -- two concurrent evals (or an eval plus anything else reading those
# clones) in different worktrees race each other's checkouts and silently
# corrupt results. Discipline: when anything else might touch the clones,
# run from a worktree whose lab/swebench_repos/ is a PRIVATE copy (`cp -R`
# of the main repo's clones), never the main repo's shared directory.
# ---------------------------------------------------------------------------
REPO_CACHE = LAB_DIR / "swebench_repos"

PARQUET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/"
    "resolve/main/data/test-00000-of-00001.parquet"
)
PARQUET_CACHE = LAB_DIR / "swebench_lite.parquet"

_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/", re.M)
_ENC = tiktoken.get_encoding("cl100k_base")

# Current published Sonnet 4.5 pricing (standard tier, <=200K context):
# https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-07).
# Single source of truth: agent.py (per-turn cost ceiling, FIX 2), run_bench.py
# (between-pair budget cap) and summarize.py (post-hoc cost reporting) all
# import this so the three can never drift out of sync with each other.
PRICE_INPUT_PER_MTOK = 3.0
PRICE_OUTPUT_PER_MTOK = 15.0


def row_cost(api_input_tokens: int, api_output_tokens: int) -> float:
    """$ cost estimate from Anthropic-reported usage tokens at the pricing
    above."""
    return ((api_input_tokens or 0) / 1e6 * PRICE_INPUT_PER_MTOK
            + (api_output_tokens or 0) / 1e6 * PRICE_OUTPUT_PER_MTOK)


def count_tokens(text: str) -> int:
    """tiktoken cl100k_base token count. Used uniformly across all three
    conditions and all message roles (system/user/assistant/tool) so the
    headline token metric is apples-to-apples regardless of Anthropic's own
    (undisclosed) tokenizer -- see README.md 'Why tiktoken' note."""
    if not text:
        return 0
    return len(_ENC.encode(text, disallowed_special=()))


def load_instances(stride: int = 10, parquet_cache: Path | None = None) -> list[dict]:
    """Load SWE-bench Lite and take every `stride`-th instance in dataset
    order (stride=10 on 300 instances -> the 30-instance pilot set)."""
    import pandas as pd

    cache = parquet_cache if parquet_cache is not None else PARQUET_CACHE
    if not cache.exists():
        print(f"downloading SWE-bench Lite parquet from {PARQUET_URL}...", flush=True)
        urllib.request.urlretrieve(PARQUET_URL, cache)
    df = pd.read_parquet(cache)
    out = []
    for _, row in df.iterrows():
        gold = sorted(set(_DIFF_FILE_RE.findall(row["patch"])))
        out.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
            "gold_files": gold,
        })
    if stride and stride > 1:
        out = out[::stride]
    return out


def repo_clone(slug: str) -> Path:
    dest = REPO_CACHE / slug.replace("/", "__")
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {slug} (full history)...", flush=True)
    r = subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{slug}.git", str(dest)],
        capture_output=True, text=True, timeout=3600,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"clone {slug} failed: {r.stderr.strip()[:300]}")
    return dest


def checkout(repo: Path, sha: str) -> None:
    r = subprocess.run(["git", "checkout", "-f", "-q", sha], cwd=repo,
                        capture_output=True, text=True, timeout=300,
                        encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"checkout {sha} failed: {r.stderr.strip()[:300]}")
    subprocess.run(["git", "clean", "-fdq"], cwd=repo, capture_output=True,
                    text=True, timeout=300, encoding="utf-8", errors="replace")


# Tolerant of markdown emphasis around the label (`**FILES:**`, `- FILES:`,
# `# FILES:`) since we do not want a bare formatting choice by the model to
# register as a turns-exhausted failure.
_FILES_LINE_RE = re.compile(r"^[\s*#>-]*FILES\s*:\**\s*(.+?)\**\s*$", re.M | re.I)


def parse_files_line(text: str) -> list[str] | None:
    """Lenient parse of the agent's final `FILES: a.py, b.py` line. Returns
    None if no such line is present anywhere in the assistant's final-turn
    text. Strips backticks/quotes/leading `./` and trailing punctuation;
    splits on commas (falling back to whitespace if the model used no
    commas)."""
    if not text:
        return None
    matches = _FILES_LINE_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1]  # last FILES: line wins if the model restates it
    parts = raw.split(",") if "," in raw else raw.split()
    files: list[str] = []
    for p in parts:
        p = p.strip().strip("`'\"")
        p = re.sub(r"[.,;]+$", "", p).strip()
        if p.startswith("./"):
            p = p[2:]
        if p and p.lower() not in ("none", "n/a"):
            files.append(p)
    return files


def files_match(returned: list[str] | None, gold: list[str]) -> bool:
    """SUCCESS iff every gold file is present among the returned files
    (extra, non-gold files in the answer do not count against success --
    see spec: 'contains every gold file')."""
    if returned is None:
        return False
    rset = set(returned)
    return all(g in rset for g in gold)
