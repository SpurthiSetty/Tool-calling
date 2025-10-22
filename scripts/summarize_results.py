"""Summarize result CSVs in the repository `results/` folder.

For each CSV file found in `results/` (excluding the summary file itself), this script computes:
- total_rows: number of rows (excluding header)
- total_matches: sum of values in the `match` column; treats values like 1, '1', 'true', 'True', 'yes', 'y' as matches
- percent_correct: total_matches / total_rows * 100 (or NaN if total_rows == 0)

Output is written to `results/summary_results.csv` with columns:
filename,total_rows,total_matches,percent_correct

Usage: python scripts/summarize_results.py
"""
import csv
import os
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SUMMARY_FILE = RESULTS_DIR / "summary_results.csv"

# values considered a match
MATCH_TRUE = {"1", "true", "True", "TRUE", "yes", "Yes", "y", "Y", "t", "T"}


def is_match_value(val):
    if val is None:
        return False
    s = str(val).strip()
    if s == "":
        return False
    # try numeric
    try:
        n = float(s)
        return n != 0.0
    except Exception:
        pass
    return s in MATCH_TRUE


def summarize_file(path: Path):
    total = 0
    matches = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # if no header, fall back to counting lines and try to access by index
        if reader.fieldnames is None:
            for _ in reader:
                total += 1
            return total, matches

        # find a 'match' column name case-insensitive
        match_col = None
        for name in reader.fieldnames:
            if name is None:
                continue
            if name.strip().lower() == "match":
                match_col = name
                break
        for row in reader:
            total += 1
            if match_col is not None:
                if is_match_value(row.get(match_col)):
                    matches += 1
            else:
                # if no match column, try to find any column named like 'predicted' or last column
                # fallback: look for a column named 'match' ignoring case in other ways
                # as last resort, treat no matches
                pass
    return total, matches


def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return

    rows = []
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        if p.name == SUMMARY_FILE.name:
            continue
        total, matches = summarize_file(p)
        pct = (matches / total * 100) if total > 0 else float("nan")
        # round to 4 decimals
        pct = round(pct, 4) if not math.isnan(pct) else pct
        rows.append({
            "filename": p.name,
            "total_rows": total,
            "total_matches": matches,
            "percent_correct": pct,
        })

    # write summary
    with SUMMARY_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "total_rows", "total_matches", "percent_correct"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
