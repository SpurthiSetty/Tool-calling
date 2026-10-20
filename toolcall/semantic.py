"""Semantic match: count a call as correct when it means the same thing.

Exact match is strict about how a value is written. Semantic match accepts a
prediction when, argument by argument, it is equivalent to the label under
three rules and nothing else:

1. Arithmetic strings are evaluated. ``"7 / 2"``, ``"10 ** 2"``, ``"10^2"``
   and ``"math.sqrt(2)"`` become numbers. A whitelist AST walker does this:
   only numbers, + - * / // % **, ``pi``/``e`` and a few ``math`` functions.
   Nothing is ever ``eval``-ed. Python literals in strings (``"[1, [2]]"``)
   are read with ``ast.literal_eval``.
2. Numbers compare with a small tolerance (``math.isclose``, rel 1e-6).
3. A parameter left out on one side equals passing the schema's documented
   ``default`` on the other (``significance_level`` omitted == ``0.05``).

Different values stay different even when they would give the same result:
meeting times ``[12, 13]`` and ``[1200, 1300]`` are not equivalent here.
"""
from __future__ import annotations

import ast
import math
import operator

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_CONSTS = {"pi": math.pi, "e": math.e}
_FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs}
_MISSING = object()


def _arith(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _arith(node.left), _arith(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_arith(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "math":
        if node.attr in _CONSTS:
            return _CONSTS[node.attr]
    if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "math" \
            else f.id if isinstance(f, ast.Name) else None
        if name in _FUNCS:
            return _FUNCS[name](_arith(node.args[0]))
    raise ValueError("not arithmetic")


def resolve(value):
    """A string that encodes a number or a Python literal becomes that value."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return _arith(ast.parse(text.replace("^", "**"), mode="eval").body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
        pass
    if text[:1] in "[{(":
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):  # e.g. "{[1, 2]}"
            pass
    return value


def values_equivalent(a, b) -> bool:
    a, b = resolve(a), resolve(b)
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if num(a) and num(b):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equivalent(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def call_equivalent(pred: dict, expected: dict, tools: dict[str, dict]) -> bool:
    if pred["name"] != expected["name"]:
        return False
    params = tools.get(expected["name"], {}).get("parameters", {})
    for k in set(pred["arguments"]) | set(expected["arguments"]):
        a = pred["arguments"].get(k, _MISSING)
        b = expected["arguments"].get(k, _MISSING)
        default = params.get(k, {}).get("default", _MISSING) if isinstance(params.get(k), dict) else _MISSING
        a = default if a is _MISSING else a
        b = default if b is _MISSING else b
        if a is _MISSING or b is _MISSING or not values_equivalent(a, b):
            return False
    return True


def calls_equivalent(pred: list[dict], expected: list[dict], tool_list: list[dict]) -> bool:
    """Is there a one-to-one pairing of predicted and expected calls where every pair is equivalent?"""
    if len(pred) != len(expected):
        return False
    tools = {t["name"]: t for t in tool_list}
    ok = [[call_equivalent(p, e, tools) for e in expected] for p in pred]
    owner = [-1] * len(expected)

    def assign(i: int, seen: set) -> bool:  # augmenting-path bipartite matching
        for j, good in enumerate(ok[i]):
            if good and j not in seen:
                seen.add(j)
                if owner[j] == -1 or assign(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return all(assign(i, set()) for i in range(len(pred)))
