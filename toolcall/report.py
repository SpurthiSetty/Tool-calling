"""Print ``results/leaderboard.csv`` as a markdown table.

Runs named ``<name>_seed<k>`` are folded into ``<name>`` and shown as
mean ± sample standard deviation.

    python -m toolcall.report
"""
from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict

from .evaluate import LEADERBOARD

COLS = [("test", "exact_match"), ("test", "semantic_match"), ("test", "exact_match_clean"), ("valid", "exact_match"),
        ("test", "err_format")]


def main() -> None:
    with LEADERBOARD.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    models = {}
    for r in rows:
        name = re.sub(r"_seed\d+$", "", r["run"])
        models[name] = r["model"].split("/")[-1]
        for split, col in COLS:
            if r["split"] == split:
                groups[name][(split, col)].append(float(r[col]))

    def cell(vals: list[float]) -> str:
        if not vals:
            return ""
        mean = statistics.mean(vals)
        if len(vals) == 1:
            return f"{mean:g}"
        return f"{mean:.1f} ± {statistics.stdev(vals):.1f} (n={len(vals)})"

    print("| run | model | test EM | test semantic | test EM (clean) | valid EM | test format errors |")
    print("|---|---|---|---|---|---|---|")
    for name, vals in groups.items():
        print(f"| {name} | {models[name]} | " + " | ".join(cell(vals[c]) for c in COLS) + " |")


if __name__ == "__main__":
    main()
