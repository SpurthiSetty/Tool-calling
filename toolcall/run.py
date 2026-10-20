"""Run one experiment end to end: build data, LoRA-train with MLX, evaluate.

    python -m toolcall.run experiments/e2_mask_prompt.yaml

An experiment file has four sections::

    name: e2_mask_prompt
    model: Qwen/Qwen3-4B-Instruct-2507
    data:  {drop_noisy: false, system_prompt: true}
    train: {epochs: 3, mask_prompt: true, ...}   # anything else goes to mlx_lm lora

``train: null`` evaluates the model as-is (a baseline).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

from .data import ROOT, SYSTEM_PROMPT, build
from .evaluate import evaluate

TRAIN_DEFAULTS = {
    "fine_tune_type": "lora",
    "optimizer": "adamw",
    "num_layers": -1,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "mask_prompt": True,
    "max_seq_length": 2048,
    "grad_checkpoint": True,
    "steps_per_report": 10,
    "seed": 0,
    "lora_parameters": {
        "rank": 8,
        "scale": 4.0,  # = lora_alpha / r in PEFT terms; the notebooks used 32 / 8
        "dropout": 0.05,
        "keys": [f"self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        + [f"mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")],
    },
}


def run(path: Path) -> dict:
    exp = yaml.safe_load(path.read_text())
    name, model = exp["name"], exp["model"]
    data_cfg = exp.get("data") or {}
    system_prompt = SYSTEM_PROMPT if data_cfg.get("system_prompt", True) else None
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = None
    if exp.get("train") is not None:
        data_dir = run_dir / "data"
        stats = build(data_dir, data_cfg.get("drop_noisy", False), 0.1, 0, system_prompt,
                      data_cfg.get("normalize_labels", False), data_cfg.get("synth_flatten", 0),
                      data_cfg.get("source", "repo"), data_cfg.get("limit"))
        train = {**TRAIN_DEFAULTS, **exp["train"]}
        train["lora_parameters"] = {**TRAIN_DEFAULTS["lora_parameters"], **exp["train"].get("lora_parameters", {})}
        epochs = train.pop("epochs", 3)
        train.setdefault("iters", math.ceil(epochs * stats["train"] / train["batch_size"]))
        train.setdefault("steps_per_eval", max(10, train["iters"] // 6))
        train.setdefault("save_every", train["iters"])
        adapter = str(run_dir / "adapters")
        cfg = {**train, "model": model, "train": True, "data": str(data_dir), "adapter_path": adapter, "val_batches": -1}
        (run_dir / "lora_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        with (run_dir / "train.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-m", "mlx_lm", "lora", "-c", str(run_dir / "lora_config.yaml")],
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )

    summaries = evaluate(model, adapter, name, exp.get("splits", ["valid", "test"]), system_prompt)
    (run_dir / "summary.json").write_text(json.dumps({"experiment": exp, "results": summaries}, indent=2))
    return summaries


def main() -> None:
    for p in sys.argv[1:]:
        run(Path(p))


if __name__ == "__main__":
    main()
