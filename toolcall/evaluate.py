"""Generate tool calls for an eval split and score them.

    python -m toolcall.evaluate --model Qwen/Qwen3-4B-Instruct-2507 --run base-4b
    python -m toolcall.evaluate --model ... --adapter runs/e2/adapters --run e2

Writes ``results/<run>/<split>_predictions.csv`` and appends one row per
split to ``results/leaderboard.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from .data import ROOT, SYSTEM_PROMPT, eval_rows, to_chat
from .metrics import calls_equal, categorize, summarize
from .parsing import parse_lenient, parse_strict
from .semantic import calls_equivalent

LEADERBOARD = ROOT / "results" / "leaderboard.csv"


def prompts_for(tokenizer, rows: list[dict], system_prompt: str | None) -> list[list[int]]:
    prompts = []
    for r in rows:
        chat = to_chat(r, system_prompt)
        prompts.append(
            tokenizer.apply_chat_template(
                chat["messages"][:-1], tools=chat["tools"], add_generation_prompt=True, return_dict=False,
                enable_thinking=False,  # hybrid Qwen3 models; ignored by the -Instruct-2507 template
            )
        )
    return prompts


def score(rows: list[dict], texts: list[str]) -> list[dict]:
    records = []
    for r, text in zip(rows, texts):
        strict = parse_strict(text)
        lenient = parse_lenient(text)
        records.append(
            {
                "id": r["id"],
                "query": r["query"],
                "predicted_text": text,
                "predicted_calls": json.dumps(strict if strict is not None else lenient),
                "expected_calls": json.dumps(r["answers"]),
                "match": strict is not None and calls_equal(strict, r["answers"]),
                "match_lenient": calls_equal(lenient, r["answers"]),
                "match_semantic": strict is not None and calls_equivalent(strict, r["answers"], r["tools"]),
                "category": categorize(text, strict, r["answers"]),
                "noisy": r["noisy"],
            }
        )
    return records


def evaluate(model_path: str, adapter: str | None, run: str, splits: list[str],
             system_prompt: str | None = SYSTEM_PROMPT, batch_size: int = 32, max_tokens: int = 512) -> dict:
    from mlx_lm import batch_generate, load

    model, tokenizer = load(model_path, adapter_path=adapter)
    out_dir = ROOT / "results" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in splits:
        rows = eval_rows(split)
        prompts = prompts_for(tokenizer, rows, system_prompt)
        t0 = time.time()
        texts = []
        for i in range(0, len(prompts), batch_size):
            texts += batch_generate(model, tokenizer, prompts[i : i + batch_size], max_tokens=max_tokens).texts
        records = score(rows, texts)
        with (out_dir / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        s = {"run": run, "split": split, "model": model_path, "adapter": adapter or "",
             **summarize(records), "gen_seconds": round(time.time() - t0, 1)}
        _append_leaderboard(s)
        print(json.dumps(s))
        summaries[split] = s
    return summaries


def _append_leaderboard(row: dict) -> None:
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if LEADERBOARD.exists():
        with LEADERBOARD.open(encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if (r["run"], r["split"]) != (row["run"], row["split"])]
    with LEADERBOARD.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(existing + [row])


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a model (optionally with a LoRA adapter) on tool calling.")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--run", required=True)
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--no-system-prompt", action="store_true")
    a = p.parse_args()
    evaluate(a.model, a.adapter, a.run, a.splits, None if a.no_system_prompt else SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
