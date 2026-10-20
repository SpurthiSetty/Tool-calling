"""Fine-tune on the full xLAM-60k set with Nebius Token Factory.

    python -m toolcall.nebius build  --model Qwen/Qwen3-1.7B [--limit N]
    python -m toolcall.nebius launch --model Qwen/Qwen3-1.7B --lr 5e-5 --epochs 1
    python -m toolcall.nebius status
    python -m toolcall.nebius download

Token Factory's conversational format expects every conversation to end in a
natural-language reply, but ours end in a tool call. So examples are uploaded
pretokenized: the same chat-template rendering MLX trains on, with the prompt
labelled -100 so loss falls only on the assistant turn.

The API key is read from ``~/.config/nebius/api_key`` and never printed.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from .data import ROOT, SYSTEM_PROMPT, eval_rows, is_noisy, load_raw, normalize_labels, to_chat, xlam_rows

API = "https://api.tokenfactory.nebius.com/v1/"
KEY_FILE = Path.home() / ".config" / "nebius" / "api_key"
OUT = ROOT / "runs" / "nebius"
STATE = OUT / "state.json"


def client():
    from openai import OpenAI

    return OpenAI(base_url=API, api_key=KEY_FILE.read_text().strip())


def pretokenize(tokenizer, row: dict, max_len: int) -> dict | None:
    chat = to_chat(row, SYSTEM_PROMPT)
    ids = tokenizer.apply_chat_template(chat["messages"], tools=chat["tools"], return_dict=False)
    prompt = tokenizer.apply_chat_template(chat["messages"][:-1], tools=chat["tools"],
                                           add_generation_prompt=True, return_dict=False)
    if len(ids) > max_len or ids[: len(prompt)] != prompt:
        return None
    labels = [-100] * len(prompt) + ids[len(prompt):]
    return {"token_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def build(model: str, max_len: int, limit: int | None, seed: int) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    rows = xlam_rows({r["query"] for r in load_raw("test")} | {r["query"] for r in eval_rows("valid")})
    n_all = len(rows)
    rows = [normalize_labels(r) for r in rows if not is_noisy(r)]
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    train, too_long, n_tokens = [], 0, 0
    for r in rows:
        ex = pretokenize(tok, r, max_len)
        if ex is None:
            too_long += 1
            continue
        train.append(ex)
        n_tokens += len(ex["token_ids"])
    valid = [ex for r in eval_rows("valid") if (ex := pretokenize(tok, r, max_len))]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("valid", valid)):
        with (OUT / f"{name}.jsonl").open("w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
    stats = {"model": model, "xlam_rows_after_holdout": n_all, "dropped_noisy": n_all - len(rows) if not limit else None,
             "dropped_too_long": too_long, "train": len(train), "valid": len(valid),
             "train_tokens": n_tokens, "max_len": max_len}
    (OUT / "build_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(**kv) -> None:
    STATE.write_text(json.dumps({**_state(), **kv}, indent=2))


def launch(model: str, lr: float, epochs: int, rank: int, alpha: int, context: int, batch: int, seed: int) -> None:
    c = client()
    train = c.files.create(file=open(OUT / "train.jsonl", "rb"), purpose="fine-tune")
    valid = c.files.create(file=open(OUT / "valid.jsonl", "rb"), purpose="fine-tune")
    job = c.fine_tuning.jobs.create(
        model=model, training_file=train.id, validation_file=valid.id, suffix="xlam-toolcall", seed=seed,
        hyperparameters={"lora": True, "lora_r": rank, "lora_alpha": alpha, "lora_dropout": 0.05,
                         "learning_rate": lr, "n_epochs": epochs, "batch_size": batch,
                         "context_length": context, "packing": True, "warmup_ratio": 0.03},
    )
    _save(job_id=job.id, model=model, train_file=train.id, valid_file=valid.id)
    print("launched", job.id, job.status)


def status() -> None:
    c = client()
    job = c.fine_tuning.jobs.retrieve(_state()["job_id"])
    print(job.status, getattr(job, "error", None) or "")
    for e in reversed(list(c.fine_tuning.jobs.list_events(job.id, limit=10))):
        print(" ", time.strftime("%H:%M:%S", time.localtime(e.created_at)), e.message)


def download() -> Path:
    c = client()
    job_id = _state()["job_id"]
    ckpts = list(c.fine_tuning.jobs.checkpoints.list(job_id))
    latest = max(ckpts, key=lambda ck: ck.step_number)
    dest = OUT / "adapter"
    dest.mkdir(parents=True, exist_ok=True)
    for fid in latest.result_files:
        meta = c.files.retrieve(fid)
        path = dest / meta.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(c.files.content(fid).read())
        print("saved", path)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="Qwen/Qwen3-1.7B")
    b.add_argument("--max-len", type=int, default=2048)
    b.add_argument("--limit", type=int)
    b.add_argument("--seed", type=int, default=0)
    l = sub.add_parser("launch")
    l.add_argument("--model", default="Qwen/Qwen3-1.7B")
    l.add_argument("--lr", type=float, default=5e-5)
    l.add_argument("--epochs", type=int, default=1)
    l.add_argument("--rank", type=int, default=8)
    l.add_argument("--alpha", type=int, default=32)  # alpha / r = 4, as in the MLX runs
    l.add_argument("--context", type=int, default=2048)
    l.add_argument("--batch", type=int, default=8)
    l.add_argument("--seed", type=int, default=0)
    sub.add_parser("status")
    sub.add_parser("download")
    a = p.parse_args()
    if a.cmd == "build":
        print(json.dumps(build(a.model, a.max_len, a.limit, a.seed), indent=2))
    elif a.cmd == "launch":
        launch(a.model, a.lr, a.epochs, a.rank, a.alpha, a.context, a.batch, a.seed)
    elif a.cmd == "status":
        status()
    else:
        download()


if __name__ == "__main__":
    main()
