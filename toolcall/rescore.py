"""Re-score saved predictions with the current metrics, without regenerating.

    python -m toolcall.rescore

Reads every ``results/<run>/<split>_predictions.csv``, scores the saved model
outputs again, rewrites the file, and refreshes that run's leaderboard row.
"""
from __future__ import annotations

import csv

from .data import ROOT, eval_rows
from .evaluate import LEADERBOARD
from .metrics import summarize
from .evaluate import score


def main() -> None:
    with LEADERBOARD.open(encoding="utf-8") as f:
        board = list(csv.DictReader(f))
    rows_by_split = {s: {r["id"]: r for r in eval_rows(s)} for s in ("valid", "test")}
    for entry in board:
        path = ROOT / "results" / entry["run"] / f"{entry['split']}_predictions.csv"
        with path.open(encoding="utf-8") as f:
            saved = list(csv.DictReader(f))
        rows = [rows_by_split[entry["split"]][int(p["id"])] for p in saved]
        records = score(rows, [p["predicted_text"] for p in saved])
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
        entry.update(summarize(records))
    fields = list(dict.fromkeys(k for e in board for k in e))
    with LEADERBOARD.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(board)
    print(f"re-scored {len(board)} run/split pairs")


if __name__ == "__main__":
    main()
