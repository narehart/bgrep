"""The two tools every condition (grep / roust / RAG) gets, unmodified.

run_command: a *whitelisted* subset of read-only shell commands (rg, grep,
find, ls, head, sed-range-read). Executed with shell=False -- argv passed
directly to subprocess, so shell metacharacters (pipes, redirects, command
substitution) are inert rather than dangerous, but we still reject strings
that look like an attempt to chain commands so the agent gets an honest
error instead of a silently-broken call.

read_file: an explicit line-range file reader. Capped at MAX_READ_LINES per
call (a hard guardrail, not just a prompt suggestion) so "read the whole
file" can't be gamed as a one-call workaround to the grep condition's
targeted-search framing -- this affects ONLY how many lines a single call
can return, not how many calls the agent may make.

Both tools are confined to the repo root: no absolute paths, no `..`
traversal.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

ALLOWED_COMMANDS = {"rg", "grep", "find", "ls", "head", "sed"}
FORBIDDEN_CHARS = set(";|&$`<>\n")
FORBIDDEN_FIND_FLAGS = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprintf", "-fls"}
FORBIDDEN_RG_FLAGS_PREFIXES = ("--pre", "--hostname-bin")  # rg preprocessor = arbitrary exec

MAX_READ_LINES = 400          # per read_file call
MAX_OUTPUT_CHARS = 8000       # per tool result, truncated beyond this
CMD_TIMEOUT_S = 20

_SED_RANGE_RE = re.compile(r"^\d+(,\d+)?p$")


TOOLS = [
    {
        "name": "run_command",
        "description": (
            "Run ONE read-only shell command using rg (ripgrep), grep, find, ls, head, "
            "or sed for a line-range read (e.g. `sed -n '120,160p' path/to/file.py`). "
            "Pass exactly one command with its arguments as plain text, like you would "
            "type it in a terminal -- but there is no real shell behind this: pipes (|), "
            "redirects (>, <), command chaining (;, &&), command substitution ($(), "
            "backticks), and sed -i / in-place edits are not supported and will be "
            "rejected. Use this for searching (rg/grep), exploring directory structure "
            "(find/ls), and peeking at line ranges (head/sed)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "e.g. `rg -n \"def parse_headers\" -t py` or `find . -name \"*.py\" -path \"*auth*\"`",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            f"Read a specific line range from a specific file (1-indexed, inclusive). "
            f"Prefer this over reading a whole large file top-to-bottom -- read a narrow "
            f"window around a grep hit instead. Capped at {MAX_READ_LINES} lines per call; "
            f"issue multiple calls with different ranges if you need more."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "repo-relative file path"},
                "start_line": {"type": "integer", "description": "1-indexed, inclusive"},
                "end_line": {"type": "integer", "description": "1-indexed, inclusive"},
            },
            "required": ["path", "start_line", "end_line"],
        },
    },
]


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n... [truncated, {len(text) - MAX_OUTPUT_CHARS} more chars -- narrow your query]"
    )


def _resolve_safe(repo_path: Path, rel: str) -> Path | None:
    """Resolve a repo-relative path, rejecting absolute paths and any
    attempt to escape repo_path via `..`. Returns None if unsafe."""
    if rel.startswith("/") or rel.startswith("~"):
        return None
    candidate = (repo_path / rel).resolve()
    try:
        candidate.relative_to(repo_path.resolve())
    except ValueError:
        return None
    return candidate


def _validate_command(command: str) -> tuple[list[str] | None, str | None]:
    bad = FORBIDDEN_CHARS & set(command)
    if bad:
        return None, (
            f"run_command: rejected -- contains disallowed character(s) {sorted(bad)}. "
            "No pipes/redirects/chaining/substitution; pass one plain command."
        )
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return None, f"run_command: could not parse command ({exc})"
    if not argv:
        return None, "run_command: empty command"
    prog = Path(argv[0]).name
    if prog not in ALLOWED_COMMANDS:
        return None, (
            f"run_command: '{prog}' is not allowed. Allowed commands: "
            f"{', '.join(sorted(ALLOWED_COMMANDS))}."
        )
    if prog == "find":
        if any(a in FORBIDDEN_FIND_FLAGS for a in argv[1:]):
            return None, "run_command: find -exec/-delete/-ok/-fprintf-style flags are not allowed."
    if prog == "rg":
        if any(a.startswith(FORBIDDEN_RG_FLAGS_PREFIXES) for a in argv[1:]):
            return None, "run_command: rg --pre (arbitrary preprocessor exec) is not allowed."
    if prog == "sed":
        if "-i" in argv[1:] or any(a.startswith("-i") for a in argv[1:]):
            return None, "run_command: sed -i (in-place edit) is not allowed; this tool is read-only."
        if "-n" not in argv[1:]:
            return None, "run_command: sed is restricted to range-read form: sed -n '<start>,<end>p' <file>."
        # find the script argument: the first non-flag token after -n that
        # isn't itself a flag/value we already consumed.
        script_ok = False
        for a in argv[1:]:
            if a in ("-n",) or a.startswith("-"):
                continue
            if _SED_RANGE_RE.match(a):
                script_ok = True
            break  # first non-flag token is the script; only that one counts
        if not script_ok:
            return None, "run_command: sed script must be a plain range-print, e.g. '10,50p' or '5p'."
    return argv, None


def run_command(command: str, repo_path: Path) -> str:
    argv, err = _validate_command(command)
    if err:
        return err
    # Reject any argument that looks like an absolute path or `..` escape
    # outside the repo (best-effort: flags starting with '-' are left alone).
    for a in argv[1:]:
        if a.startswith("-"):
            continue
        if a.startswith("/") or ".." in Path(a).parts:
            return f"run_command: path argument '{a}' escapes the repo root; use repo-relative paths only."
    try:
        r = subprocess.run(
            argv, cwd=repo_path, shell=False, capture_output=True, text=True,
            timeout=CMD_TIMEOUT_S, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"run_command: timed out after {CMD_TIMEOUT_S}s -- narrow your query."
    except FileNotFoundError:
        return f"run_command: '{argv[0]}' not found on this system."
    out = r.stdout
    # For rg/grep, returncode 1 means "no matches" -- not an error, don't
    # surface stderr for it. For the other commands a nonzero code is a
    # real error (bad path, bad flag, ...) and stderr is the useful part.
    no_match_ok = argv[0] in ("rg", "grep") and r.returncode == 1
    if r.returncode != 0 and not no_match_ok and r.stderr:
        out = (out + "\n" + r.stderr).strip()
    if not out.strip():
        out = "(no output)"
    return _truncate(out)


def read_file(path: str, start_line: int, end_line: int, repo_path: Path) -> str:
    if start_line < 1 or end_line < start_line:
        return "read_file: invalid range -- start_line must be >=1 and <= end_line."
    target = _resolve_safe(repo_path, path)
    if target is None:
        return f"read_file: path '{path}' escapes the repo root or is invalid; use a repo-relative path."
    if not target.exists() or not target.is_file():
        return f"read_file: no such file '{path}'."
    capped_end = min(end_line, start_line + MAX_READ_LINES - 1)
    try:
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"read_file: could not read '{path}' ({exc})"
    n = len(lines)
    if start_line > n:
        return f"read_file: '{path}' only has {n} lines."
    window = lines[start_line - 1 : min(capped_end, n)]
    body = "\n".join(f"{start_line + i:>6}\t{ln}" for i, ln in enumerate(window))
    note = ""
    if capped_end < end_line and capped_end < n:
        # The MAX_READ_LINES cap -- not EOF -- is what cut this window short.
        note = f"\n... [capped at {MAX_READ_LINES} lines/call -- issue another read_file call for more]"
    return _truncate(f"{path}:{start_line}-{min(capped_end, n)} (of {n} lines)\n{body}{note}")


def execute_tool(name: str, tool_input: dict, repo_path: Path) -> str:
    if name == "run_command":
        return run_command(tool_input.get("command", ""), repo_path)
    if name == "read_file":
        return read_file(
            tool_input.get("path", ""),
            int(tool_input.get("start_line", 1)),
            int(tool_input.get("end_line", 1)),
            repo_path,
        )
    return f"unknown tool '{name}'"
