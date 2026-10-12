"""Extract tool calls from Qwen/Hermes-style generations.

Qwen emits each call as::

    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>

``parse_strict`` accepts only that shape -- it is what a serving stack (vLLM's
hermes parser, Ollama, ...) would accept, so it is the number we report.

``parse_lenient`` pulls out any JSON object with a ``name`` key that sits
between tool-call tags, even if the tags themselves are malformed (the old
fine-tunes often opened with ``</tool_call>``). The gap between the two
scores is "right call, broken wrapper".
"""
from __future__ import annotations

import json
import re

_STRICT = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TAG = re.compile(r"</?tool_call>")


def _as_call(obj) -> dict | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):  # some models double-encode arguments
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"], "arguments": args}


def parse_strict(text: str) -> list[dict] | None:
    """Calls if every ``<tool_call>`` block is well-formed JSON, else ``None``."""
    if text.count("<tool_call>") != text.count("</tool_call>"):
        return None
    blocks = _STRICT.findall(text)
    if len(blocks) != text.count("<tool_call>"):
        return None
    calls = []
    for b in blocks:
        try:
            call = _as_call(json.loads(b))
        except json.JSONDecodeError:
            return None
        if call is None:
            return None
        calls.append(call)
    return calls


def parse_lenient(text: str) -> list[dict]:
    calls = []
    decoder = json.JSONDecoder()
    for chunk in _TAG.split(text):
        chunk = chunk.strip()
        if not chunk.startswith("{"):
            continue
        try:
            obj, _ = decoder.raw_decode(chunk)
        except json.JSONDecodeError:
            continue
        if (call := _as_call(obj)) is not None:
            calls.append(call)
    return calls
