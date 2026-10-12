"""Turn the raw CSVs into chat-format JSONL that both MLX and TRL can train on.

Raw rows look like ``tools, answers, query, id`` where ``tools`` and
``answers`` are JSON strings (an xLAM-style subset: 10 functions, one tool
offered per query, 1-9 calls per answer).

What this fixes compared to the old notebooks:

* Every example is written as its own JSON line with ``json.dumps``. The old
  ``train_dataset_toolcalling.jsonl`` went through ``datasets.Dataset``, which
  unified every tool's parameter struct into one schema and injected
  ``"ip": null, "intervals": null, ...`` into *every* tool and *every* call.
* Test queries that also appear in train are dropped from train (4 leaked).
* A validation split is carved out of train so that hyper-parameters are
  chosen on validation, never on the test set.
* Rows whose answers violate the declared parameter types (e.g. a ``float``
  radius given as the string ``"7 / 2"``) are flagged ``noisy``. They are
  kept by default and only filtered when an experiment asks for it.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SYSTEM_PROMPT = "You are a helpful assistant that uses the provided tools to answer."

_SCALARS = {"int": (int,), "float": (int, float), "str": (str,)}


def load_raw(split: str) -> list[dict]:
    with (RAW / f"{split}_dataset.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["tools"] = json.loads(r["tools"])
        r["answers"] = json.loads(r["answers"])
        r["id"] = int(r["id"])
    return rows


def _type_ok(value, declared: str) -> bool:
    base = declared.split(",")[0].strip()  # "float, optional" -> "float"
    if base in _SCALARS:
        return isinstance(value, _SCALARS[base]) and not isinstance(value, bool)
    if base.startswith("List"):
        return isinstance(value, list)
    return True


def is_noisy(row: dict) -> bool:
    """True if any expected argument has the wrong type for its declared schema."""
    tools = {t["name"]: t for t in row["tools"]}
    for call in row["answers"]:
        params = tools[call["name"]]["parameters"]
        for k, v in call["arguments"].items():
            if k not in params or not _type_ok(v, params[k]["type"]):
                return True
    return False


def normalize_labels(row: dict) -> dict:
    """Make one convention win where the labels disagree with each other.

    ``min_meeting_rooms``: 36 train answers give times as decimal hours
    (``[9, 10.5]``) and 6 as HHMM integers (``[900, 1030]``). A fine-tune picks
    up the minority form and applies it everywhere, so rewrite HHMM to hours.
    """
    calls = []
    for c in row["answers"]:
        args = c["arguments"]
        iv = args.get("intervals") if c["name"] == "min_meeting_rooms" else None
        if iv and all(isinstance(t, (int, float)) for p in iv for t in p) and max(max(p) for p in iv) >= 100:
            hours = lambda t: t // 100 + round((t % 100) / 60, 4)
            args = {**args, "intervals": [[hours(t) for t in p] for p in iv]}
        calls.append({**c, "arguments": args})
    return {**row, "answers": calls}


_FLATTEN_TEMPLATES = [
    "Flatten {lst}.",
    "Can you flatten this list for me: {lst}?",
    "What does {lst} look like once it's flattened?",
    "I have a nested list {lst}. Please flatten it into a single list.",
    "Turn {lst} into a flat list.",
    "Flatten the following nested structure: {lst}",
    "Could you un-nest {lst} so there's only one level?",
    "Please flatten {lst} and give me the result.",
]
_WORDS = ["apple", "x", "foo", "bar", "red", "cat", "q", "hello", "z9", "beta", "kiwi", "sun"]


def _random_nested(rng: random.Random, depth: int = 0) -> list:
    out = []
    for _ in range(rng.randint(1, 4)):
        roll = rng.random()
        if depth < 3 and roll < 0.35:
            out.append(_random_nested(rng, depth + 1))
        elif roll < 0.6:
            out.append(rng.choice(_WORDS))
        elif roll < 0.85:
            out.append(rng.randint(-5, 99))
        else:
            out.append(rng.choice([True, False, None, 2.5]))
    return out


def synth_flatten(tool: dict, n: int, seed: int) -> list[dict]:
    """Queries that quote Python literals (``['a', True, None]``) paired with JSON answers.

    The train set has only 3 such queries, and fine-tunes copy the Python
    syntax straight into the tool call, which is not valid JSON.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        lst = _random_nested(rng)
        if not any(isinstance(x, list) for x in lst):
            lst.append(_random_nested(rng, 2))
        query = rng.choice(_FLATTEN_TEMPLATES).format(lst=repr(lst))
        rows.append({"id": -1 - i, "query": query, "tools": [tool],
                     "answers": [{"name": "flatten_list", "arguments": {"nested_list": lst}}]})
    return rows


def to_chat(row: dict, system_prompt: str | None = SYSTEM_PROMPT) -> dict:
    """One example in OpenAI-style chat format, which Qwen's template renders natively."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": row["query"]})
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in row["answers"]
            ],
        }
    )
    tools = [{"type": "function", "function": t} for t in row["tools"]]
    return {"id": row["id"], "messages": messages, "tools": tools}


def split_train_valid(rows: list[dict], valid_frac: float, seed: int) -> tuple[list, list]:
    """Stratify by tool name so every function shows up in validation."""
    by_tool = defaultdict(list)
    for r in rows:
        by_tool[r["tools"][0]["name"]].append(r)
    rng = random.Random(seed)
    train, valid = [], []
    for name in sorted(by_tool):
        group = by_tool[name]
        rng.shuffle(group)
        n_valid = max(1, round(len(group) * valid_frac))
        valid += group[:n_valid]
        train += group[n_valid:]
    rng.shuffle(train)
    return train, valid


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(out_dir: Path, drop_noisy: bool, valid_frac: float, seed: int, system_prompt: str | None,
          normalize: bool = False, synth_flatten_n: int = 0) -> dict:
    train_raw, test_raw = load_raw("train"), load_raw("test")
    test_queries = {r["query"] for r in test_raw}
    leaked = [r for r in train_raw if r["query"] in test_queries]
    train_raw = [r for r in train_raw if r["query"] not in test_queries]

    # Split before filtering so every experiment is scored on the same validation rows.
    train, valid = split_train_valid(train_raw, valid_frac, seed)
    noisy = [r for r in train if is_noisy(r)]
    if drop_noisy:
        train = [r for r in train if not is_noisy(r)]
    if normalize:
        train = [normalize_labels(r) for r in train]
    if synth_flatten_n:
        tool = next(r["tools"][0] for r in train if r["tools"][0]["name"] == "flatten_list")
        train += synth_flatten(tool, synth_flatten_n, seed)
        random.Random(seed).shuffle(train)

    # MLX reads train.jsonl / valid.jsonl from the directory; extra keys are ignored.
    write_jsonl(out_dir / "train.jsonl", [to_chat(r, system_prompt) for r in train])
    write_jsonl(out_dir / "valid.jsonl", [to_chat(r, system_prompt) for r in valid])
    stats = {
        "train": len(train),
        "valid": len(valid),
        "dropped_leaked": len(leaked),
        "noisy_in_train": len(noisy),
        "dropped_noisy": len(noisy) if drop_noisy else 0,
        "synthetic": synth_flatten_n,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def eval_rows(split: str, seed: int = 0, valid_frac: float = 0.1) -> list[dict]:
    """Rows to evaluate on: ``test`` is the held-out CSV, ``valid`` is the carve-out of train."""
    if split == "test":
        rows = load_raw("test")
    else:
        test_queries = {r["query"] for r in load_raw("test")}
        train_raw = [r for r in load_raw("train") if r["query"] not in test_queries]
        rows = split_train_valid(train_raw, valid_frac, seed)[1]
    for r in rows:
        r["noisy"] = is_noisy(r)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "default")
    p.add_argument("--drop-noisy", action="store_true")
    p.add_argument("--normalize-labels", action="store_true")
    p.add_argument("--no-system-prompt", action="store_true")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    stats = build(a.out, a.drop_noisy, a.valid_frac, a.seed, None if a.no_system_prompt else SYSTEM_PROMPT,
                  a.normalize_labels)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
