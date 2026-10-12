"""Score predicted tool calls against expected ones.

A prediction is correct when the multiset of calls matches exactly: same
function names, same argument keys, same values (``6 == 6.0``). Call order is
ignored because parallel calls have no inherent order; on this test set that
changes nothing versus the old order-sensitive check, but it is the right
contract.

Each failure gets exactly one category, checked in this order:

``no_call``       the model answered in prose instead of calling a tool
``format``        tool-call tags or JSON were malformed (strict parse failed)
``wrong_calls``   different function names or a different number of calls
``wrong_args``    right functions, wrong arguments
"""
from __future__ import annotations

import json
from collections import Counter


def _norm(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _key(call: dict) -> str:
    return json.dumps(_norm({"name": call["name"], "arguments": call["arguments"]}), sort_keys=True)


def calls_equal(pred: list[dict], expected: list[dict]) -> bool:
    return Counter(map(_key, pred)) == Counter(map(_key, expected))


def categorize(text: str, strict: list[dict] | None, expected: list[dict]) -> str:
    if strict is not None and calls_equal(strict, expected):
        return "correct"
    if "tool_call" not in text and "{" not in text:
        return "no_call"
    if strict is None:
        return "format"
    if Counter(c["name"] for c in strict) != Counter(c["name"] for c in expected):
        return "wrong_calls"
    return "wrong_args"


def summarize(records: list[dict]) -> dict:
    n = len(records)
    clean = [r for r in records if not r["noisy"]]
    cats = Counter(r["category"] for r in records)
    pct = lambda k, rs: round(100 * sum(r[k] for r in rs) / len(rs), 2) if rs else float("nan")
    return {
        "n": n,
        "exact_match": pct("match", records),
        "exact_match_lenient": pct("match_lenient", records),
        "exact_match_clean": pct("match", clean),
        "n_clean": len(clean),
        **{f"err_{k}": cats.get(k, 0) for k in ("no_call", "format", "wrong_calls", "wrong_args")},
    }
