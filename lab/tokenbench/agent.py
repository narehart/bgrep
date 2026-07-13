"""The Anthropic tool-use agent loop shared by all four v2 arms.

v1 -> v2 change (see README.md '## Why v2'): retrieval is no longer
INJECTED into the first user message. All four arms get the SAME first
user message (bare issue text) and differ ONLY in which tools are on their
toolbelt:

    A grep        run_command, read_file
    B roust       roust, read_file                  (no grep -- roust is
                                                       THE search tool)
    C roust_grep  roust, run_command, read_file      (agent's choice)
    D rag_grep    rag_search, run_command, read_file (agent's choice)

System prompt, model, max_turns, and temperature are identical across arms
-- that identity is the fairness property this whole benchmark rests on.
The only prompt text that varies by arm is the one-line "your tools are:"
paragraph (which tools exist necessarily differs) and the immediately
following one-sentence opening-move hint; everything else (the stop-early
instruction, the FILES: format, the guardrails) is shared verbatim.

Every request/response is logged to a JSONL transcript for auditability.
Token totals are computed two ways and both are recorded:
  * tiktoken cl100k_base over the actual message text sent/received each
    turn (the headline metric -- tokenizer-neutral, reproducible without an
    API key, comparable 1:1 across arms).
  * the real Anthropic-reported usage.input_tokens/output_tokens (used for
    the $ cost estimate, since that's what billing actually uses).

Additionally (v2, for the fairness audit): every roust/rag_search tool
result's tiktoken length is recorded individually in
`retrieval_tool_tokens`, so summarize.py can report the actual mean tokens
returned per retrieval-tool call, per arm -- auditing whether the ~8192
roust-budget / rag k=24 match actually landed on real repos/queries rather
than trusting the pre-run calibration in rag_tool.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from common import count_tokens, files_match, parse_files_line
from rag_tool import RAG_SEARCH_TOOL, rag_search
from roust_tool import ROUST_TOOL, roust_search
from sandbox_tools import READ_FILE_TOOL, RUN_COMMAND_TOOL, read_file, run_command

MODEL = "claude-sonnet-4-5-20250929"
MAX_TURNS = 30
MAX_TOKENS_PER_TURN = 4096
TEMPERATURE = 0
ROUST_BUDGET = 8192

ARMS: dict[str, list[str]] = {
    "grep": ["run_command", "read_file"],
    "roust": ["roust", "read_file"],
    "roust_grep": ["roust", "run_command", "read_file"],
    "rag_grep": ["rag_search", "run_command", "read_file"],
}

_TOOL_SCHEMAS = {
    "run_command": RUN_COMMAND_TOOL,
    "read_file": READ_FILE_TOOL,
    "roust": ROUST_TOOL,
    "rag_search": RAG_SEARCH_TOOL,
}

_TOOL_LINES = {
    "run_command": "- run_command: run a single read-only search/exploration command using rg (ripgrep), "
                   "grep, find, ls, head, or sed (for a line-range read, e.g. `sed -n '120,160p' path/to/file.py`).",
    "read_file": "- read_file: read a specific line range of a specific file (path, start_line, end_line). "
                 "Prefer narrow ranges around a hit over reading whole files top to bottom -- it costs you "
                 "tokens and turns to dump whole files, so don't.",
    "roust": "- roust: a recall-first code-retrieval tool built for agents -- call it with a query (the issue "
             "text, an error string, a symbol name, or a refined query) and it returns a token-budgeted bundle "
             "of the files/regions it judges most relevant, ranked, with source packed in.",
    "rag_search": "- rag_search: semantic (embedding) search -- call it with a query and it returns the top "
                  "matching code chunks, ranked by cosine similarity, with file path/line range and source text. "
                  "Semantic similarity is not the same as 'this file needs to change'.",
}

_OPENING_HINT = {
    "grep": "Lead with a targeted rg/grep search for the strongest signal in the issue text (error messages, "
            "tracebacks, class/function/identifier names), then follow the trail (imports, call sites, the "
            "test file(s) that exercise the behavior).",
    "roust": "Call roust with the issue text (or a refined query) as your first move -- it is your only search "
             "tool -- then use read_file to verify/expand on specific hits.",
    "roust_grep": "Call roust with the issue text (or a refined query) as your first move, then use "
                  "run_command/read_file to verify hits or search for anything roust's bundle didn't cover.",
    "rag_grep": "Call rag_search with the issue text (or a refined query) as your first move, then use "
                "run_command/read_file to verify hits or search for anything the retrieved chunks didn't cover.",
}

SYSTEM_PROMPT_TAIL = """
Avoid unfocused exploration (e.g. listing every file with no filter) and avoid reading large files in full -- \
both burn turns and tokens without adding precision.

Answer with FILES: as soon as you are confident. Additional exploration costs tokens and does not improve \
your score. Do not verify beyond what you need -- once the evidence converges on the right file(s), stop \
investigating and answer immediately; do not spend further turns double-checking a conclusion you already \
trust.

When you are confident you have found every file that needs to change, respond with NO further tool calls, \
and end your message with a line of exactly this form:

FILES: path/to/file1.py, path/to/file2.py

List repo-relative paths (as they would appear in `git diff`), comma-separated, on that one line. Only list \
files that need to be *edited* to fix the issue; do not include files you merely inspected along the way."""


def build_system_prompt(arm: str) -> str:
    tool_names = ARMS[arm]
    tools_block = "\n".join(_TOOL_LINES[t] for t in tool_names)
    head = (
        "You are an expert code-localization agent. Given a GitHub issue/problem statement and read-only "
        "access to a cloned repository, your job is to identify EVERY file that must be edited to fix the "
        "issue -- not files that are merely related or worth glancing at, but the specific files a correct "
        "patch would touch.\n\nYou have the following tools:\n" + tools_block + "\n\n" + _OPENING_HINT[arm]
    )
    return head + SYSTEM_PROMPT_TAIL


def build_first_user_message(problem_statement: str) -> str:
    # Identical across all four arms -- no injected retrieval context. This
    # is the v1->v2 fix: retrieval is a tool call the agent chooses to make,
    # never pre-billed into every turn's resent history.
    return f"GitHub issue / problem statement:\n\n{problem_statement}"


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


_RETRIEVAL_TOOLS = {"roust", "rag_search"}


def execute_tool(name: str, tool_input: dict, repo_path: Path, instance: dict,
                  roust_budget: int, rag_k: int) -> tuple[str, dict]:
    """Dispatches a tool call. Returns (result_text, call_meta) -- call_meta
    always carries {"tool": name, "tokens": <tiktoken len of result_text>,
    "retrieval": bool}; roust/rag_search calls additionally carry the
    tool's own meta (files/stats). Used to build the per-arm, per-tool
    'mean tokens returned per call' fairness audit in summarize.py."""
    if name == "run_command":
        text = run_command(tool_input.get("command", ""), repo_path)
        return text, {"tool": name, "tokens": count_tokens(text), "retrieval": False}
    if name == "read_file":
        text = read_file(
            tool_input.get("path", ""), int(tool_input.get("start_line", 1)),
            int(tool_input.get("end_line", 1)), repo_path,
        )
        return text, {"tool": name, "tokens": count_tokens(text), "retrieval": False}
    if name == "roust":
        text, meta = roust_search(tool_input.get("query", ""), repo_path, budget=roust_budget)
        return text, {"tool": name, "tokens": count_tokens(text), "retrieval": True, **meta}
    if name == "rag_search":
        text, meta = rag_search(tool_input.get("query", ""), repo_path, instance["repo"],
                                 instance["base_commit"], k=rag_k)
        return text, {"tool": name, "tokens": count_tokens(text), "retrieval": True, **meta}
    text = f"unknown tool '{name}'"
    return text, {"tool": name, "tokens": count_tokens(text), "retrieval": False}


def run_agent(
    client: Any,
    instance: dict,
    arm: str,
    repo_path: Path,
    log_fh,
    max_turns: int = MAX_TURNS,
    model: str = MODEL,
    roust_budget: int = ROUST_BUDGET,
    rag_k: int | None = None,
) -> dict:
    """Runs the tool-use loop to completion (FILES: line, turn cap, or
    error) and returns a result dict ready to be written to results.jsonl."""
    from rag_tool import TOP_K as _DEFAULT_RAG_K

    if rag_k is None:
        rag_k = _DEFAULT_RAG_K

    t0 = time.perf_counter()
    tool_names = ARMS[arm]
    tools_schema = [_TOOL_SCHEMAS[t] for t in tool_names]
    system_prompt = build_system_prompt(arm)
    first_msg = build_first_user_message(instance["problem_statement"])
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
    tool_call_log: list[dict] = []  # one entry per tool call: {"tool", "tokens", "retrieval", ...}

    try:
        for turn in range(1, max_turns + 1):
            turns_used = turn
            request_text = system_prompt + "\n\n" + "\n\n".join(
                f"[{m['role']}]\n{_serialize_text(m['content'])}" for m in messages
            )
            tiktoken_input_total += count_tokens(request_text)

            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS_PER_TURN,
                temperature=TEMPERATURE,
                system=system_prompt,
                tools=tools_schema,
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
                "instance_id": instance["instance_id"], "arm": arm, "turn": turn,
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
                result_text, call_meta = execute_tool(
                    d.get("name", ""), d.get("input", {}) or {}, repo_path, instance,
                    roust_budget, rag_k,
                )
                tool_call_log.append(call_meta)
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
        "arm": arm,
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
        "tool_call_log": tool_call_log,
    }
