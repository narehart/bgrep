"""Scripted fake Anthropic client for validating the harness's wiring
(tool-call dispatch, message threading, JSONL logging, token counting,
FILES: parsing/scoring, resume-safety, cost tracking) WITHOUT a real API
key or any network spend. Not used for the real pilot -- see run_bench.py
--mock.

Mimics just enough of the `anthropic` SDK response shape (`.content` list
of blocks with `.model_dump()`, `.stop_reason`, `.usage.input_tokens/
output_tokens`) for agent.py to treat it identically to the real client.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass
class Block:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None

    def model_dump(self) -> dict:
        d = {"type": self.type}
        if self.type == "text":
            d["text"] = self.text
        elif self.type == "tool_use":
            d.update({"id": self.id, "name": self.name, "input": self.input})
        return d


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class Response:
    content: list
    stop_reason: str
    usage: Usage


class _Messages:
    def __init__(self):
        self._counter = itertools.count(1)

    def create(self, model, max_tokens, temperature, system, tools, messages, **_):
        # Turn 1: probe with a whitelisted grep. Turn 2: a narrow read_file.
        # Turn 3+: finalize with a FILES: line naming a plausible-looking
        # (but not gold-verified) file, purely to exercise parsing/scoring.
        n_prior_assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
        call_id = f"toolu_mock_{next(self._counter)}"
        approx_in = sum(len(str(m.get("content", ""))) for m in messages) // 4

        if n_prior_assistant_turns == 0:
            content = [Block(type="tool_use", id=call_id, name="run_command",
                              input={"command": "ls"})]
            stop_reason = "tool_use"
        elif n_prior_assistant_turns == 1:
            content = [Block(type="tool_use", id=call_id, name="read_file",
                              input={"path": "setup.py", "start_line": 1, "end_line": 20})]
            stop_reason = "tool_use"
        else:
            content = [Block(type="text", text="Based on my investigation.\n\nFILES: setup.py")]
            stop_reason = "end_turn"

        out_len = sum(len(str(c.text or "")) + len(str(c.input or "")) for c in content) // 4 + 10
        return Response(content=content, stop_reason=stop_reason,
                         usage=Usage(input_tokens=approx_in, output_tokens=out_len))


class MockClient:
    def __init__(self):
        self.messages = _Messages()
