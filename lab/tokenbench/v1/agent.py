"""The Anthropic tool-use agent loop shared by all three conditions.

Conditions differ ONLY in the contents of the first user message (A: bare
issue text; B: issue text + roust bundle; C: issue text + RAG top-12
chunks) and the label used for logging. System prompt, tools, model, max
turns, and temperature are identical -- that identity is the fairness
property this whole benchmark rests on.

Every request/response is logged to a JSONL transcript for auditability.
Token totals are computed two ways and both are recorded:
  * tiktoken cl100k_base over the actual message text sent/received each
    turn (the headline metric the spec asks for -- tokenizer-neutral,
    reproducible without an API key, comparable 1:1 across conditions).
  * the real Anthropic-reported usage.input_tokens/output_tokens (used for
    the $ cost estimate, since that's what billing actually uses).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from common import count_tokens, files_match, parse_files_line
from sandbox_tools import TOOLS, execute_tool

MODEL = "claude-sonnet-4-5-20250929"
MAX_TURNS = 15
MAX_TOKENS_PER_TURN = 4096
TEMPERATURE = 0

SYSTEM_PROMPT = """You are an expert code-localization agent. Given a GitHub issue/problem \
statement and read-only access to a cloned repository, your job is to identify EVERY file \
that must be edited to fix the issue -- not files that are merely related or worth glancing \
at, but the specific files a correct patch would touch.

You have two tools:
- run_command: run a single read-only command using rg (ripgrep), grep, find, ls, head, or \
sed (for a line-range read, e.g. `sed -n '120,160p' path/to/file.py`). Use it to search for \
symbols, error strings, class/function names, and to explore directory structure.
- read_file: read a specific line range of a specific file (path, start_line, end_line). \
Prefer this -- and prefer *narrow* ranges around a hit -- over reading a large file top to \
bottom; it costs you tokens and turns to dump whole files, so don't.

Work efficiently: start from the strongest signal in the issue text (error messages, \
tracebacks, class/function/identifier names, file paths already mentioned) and grep for it, \
then follow the trail (imports, call sites, the test file(s) that exercise the behavior). \
Avoid unfocused exploration like `find . -name "*.py"` with no filter, and avoid reading \
large files in full -- both burn turns and tokens without adding precision. You have a \
limited number of turns, so once the evidence converges, stop investigating and answer.

When you are confident you have found every file that needs to change, respond with NO \
further tool calls, and end your message with a line of exactly this form:

FILES: path/to/file1.py, path/to/file2.py

List repo-relative paths (as they would appear in `git diff`), comma-separated, on that one \
line. Only list files that need to be *edited* to fix the issue; do not include files you \
merely inspected along the way."""


def build_first_user_message(condition: str, problem_statement: str, extra_context: str | None) -> str:
    base = f"GitHub issue / problem statement:\n\n{problem_statement}"
    if condition == "grep" or not extra_context:
        return base
    if condition == "roust":
        preamble = (
            "Below is a head-start localization bundle produced by an automated tool "
            "(roust) for this issue: its best guess at the relevant files and code regions, "
            "token-budgeted. Treat it as a HINT, not ground truth -- it can miss files or "
            "include irrelevant ones. Use your tools to verify and expand on it before "
            "finalizing your FILES: answer."
        )
    elif condition == "rag":
        preamble = (
            "Below are the top-12 code chunks retrieved by semantic (embedding) search "
            "against this issue text. Treat them as a HINT, not ground truth -- semantic "
            "similarity is not the same as 'this file needs to change', and relevant files "
            "can be missing from this list entirely. Use your tools to verify and expand on "
            "it before finalizing your FILES: answer."
        )
    else:
        preamble = ""
    return f"{base}\n\n---\n{preamble}\n\n{extra_context}"


def _block_to_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": getattr(block, "type", "unknown"), "repr": repr(block)}


def _serialize_text(role_content: Any) -> str:
    """Flatten a message's content (str, or list of dict/SDK blocks) into
    plain text for tiktoken counting."""
    if isinstance(role_content, str):
        return role_content
    parts = []
    for block in role_content:
        d = _block_to_dict(block)
        t = d.get("type")
        if t == "text":
            parts.append(d.get("text", ""))
        elif t == "tool_use":
            parts.append(f"{d.get('name','')}({json.dumps(d.get('input', {}))})")
        elif t == "tool_result":
            c = d.get("content", "")
            parts.append(c if isinstance(c, str) else json.dumps(c))
        else:
            parts.append(json.dumps(d, default=str))
    return "\n".join(parts)


def run_agent(
    client: Any,
    instance: dict,
    condition: str,
    repo_path: Path,
    extra_context: str | None,
    log_fh,
    max_turns: int = MAX_TURNS,
    model: str = MODEL,
) -> dict:
    """Runs the tool-use loop to completion (FILES: line, turn cap, or
    error) and returns a result dict ready to be written to results.jsonl."""
    t0 = time.perf_counter()
    first_msg = build_first_user_message(condition, instance["problem_statement"], extra_context)
    messages: list[dict] = [{"role": "user", "content": first_msg}]

    tiktoken_input_total = 0
    tiktoken_output_total = 0
    api_input_total = 0
    api_output_total = 0
    tool_call_count = 0
    turns_used = 0
    final_text = ""
    error: str | None = None
    hit_turn_cap = False

    try:
        for turn in range(1, max_turns + 1):
            turns_used = turn
            request_text = SYSTEM_PROMPT + "\n\n" + "\n\n".join(
                f"[{m['role']}]\n{_serialize_text(m['content'])}" for m in messages
            )
            tiktoken_input_total += count_tokens(request_text)

            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS_PER_TURN,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            usage = getattr(response, "usage", None)
            api_in = getattr(usage, "input_tokens", 0) if usage else 0
            api_out = getattr(usage, "output_tokens", 0) if usage else 0
            api_input_total += api_in
            api_output_total += api_out

            response_text = _serialize_text(response.content)
            tiktoken_output_total += count_tokens(response_text)

            log_fh.write(json.dumps({
                "instance_id": instance["instance_id"], "condition": condition, "turn": turn,
                "request_messages": [
                    {"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                     else [_block_to_dict(b) for b in m["content"]]}
                    for m in messages
                ],
                "response": {
                    "stop_reason": response.stop_reason,
                    "content": [_block_to_dict(b) for b in response.content],
                    "usage": {"input_tokens": api_in, "output_tokens": api_out},
                },
            }) + "\n")
            log_fh.flush()

            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [b for b in response.content if _block_to_dict(b).get("type") == "tool_use"]
            text_blocks = [_block_to_dict(b).get("text", "") for b in response.content
                            if _block_to_dict(b).get("type") == "text"]
            if text_blocks:
                final_text = "\n".join(text_blocks)

            if not tool_use_blocks:
                # No tool calls this turn -- the agent believes it's done.
                break

            tool_results = []
            for b in tool_use_blocks:
                d = _block_to_dict(b)
                tool_call_count += 1
                result_text = execute_tool(d.get("name", ""), d.get("input", {}) or {}, repo_path)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": d.get("id"), "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            hit_turn_cap = True
    except Exception as exc:  # noqa: BLE001 -- want every failure mode captured, not crash the sweep
        error = f"{type(exc).__name__}: {exc}"[:500]

    wall_clock_s = time.perf_counter() - t0
    returned_files = parse_files_line(final_text)
    gold_files = instance["gold_files"]
    # Per spec: exhausting the turn cap is a failure even if a FILES: line
    # happened to appear in the same (over-budget) final turn as a tool call.
    success = False if (error or hit_turn_cap) else files_match(returned_files, gold_files)

    return {
        "instance_id": instance["instance_id"],
        "condition": condition,
        "repo": instance["repo"],
        "gold_files": gold_files,
        "returned_files": returned_files,
        "success": success,
        "error": error,
        "hit_turn_cap": hit_turn_cap,
        "turns_used": turns_used,
        "tool_calls": tool_call_count,
        "wall_clock_s": round(wall_clock_s, 2),
        "tiktoken_input_tokens": tiktoken_input_total,
        "tiktoken_output_tokens": tiktoken_output_total,
        "tiktoken_total_tokens": tiktoken_input_total + tiktoken_output_total,
        "api_input_tokens": api_input_total,
        "api_output_tokens": api_output_total,
        "api_total_tokens": api_input_total + api_output_total,
        "final_text": final_text,
    }
