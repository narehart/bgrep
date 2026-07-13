#!/usr/bin/env python3
"""Prints a human-readable excerpt of one (instance, arm) transcript --
used to pull the 4 example-behavior excerpts (one per arm) for the report.
Not part of the scored pipeline.

Usage:
    lab/tokenbench/.venv/bin/python3 lab/tokenbench/excerpt.py \\
        lab/tokenbench/transcripts/<instance_id>__<arm>.jsonl [--max-turns N] [--max-chars N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        t = b.get("type")
        if t == "text":
            parts.append(f"[text] {b.get('text','')}")
        elif t == "tool_use":
            parts.append(f"[tool_use] {b.get('name')}({json.dumps(b.get('input', {}))})")
        elif t == "tool_result":
            c = b.get("content", "")
            parts.append(f"[tool_result] {c if isinstance(c, str) else json.dumps(c)}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-chars", type=int, default=600)
    args = ap.parse_args()

    lines = Path(args.path).read_text().splitlines()
    print(f"=== {args.path} ({len(lines)} turns logged) ===\n")
    for line in lines[: args.max_turns]:
        row = json.loads(line)
        turn = row["turn"]
        resp = row["response"]
        text = _text_of(resp["content"])
        if len(text) > args.max_chars:
            text = text[: args.max_chars] + f"... [+{len(text) - args.max_chars} chars]"
        usage = resp.get("usage", {})
        print(f"--- turn {turn} (stop={resp.get('stop_reason')}, "
              f"api_in={usage.get('input_tokens')}, api_out={usage.get('output_tokens')}) ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
