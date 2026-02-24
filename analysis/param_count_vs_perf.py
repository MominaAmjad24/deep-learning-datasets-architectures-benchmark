# analysis/param_count_vs_perf.py
from __future__ import annotations
import json
from pathlib import Path
import csv


def main():
    base = Path("runs")
    rows = []

    for result_path in base.glob("*/*result.json"):
        r = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": r["dataset"],
            "architecture": r["architecture"],
            "device": r["device"],
            "param_count": r["param_count"],
            "test_accuracy": r["test_accuracy"],
            "test_f1": r["test_f1"] if r["test_f1"] is not None else "",
            "training_time_seconds": r["training_time_seconds"],
        })

    if not rows:
        raise FileNotFoundError("No runs/*/result.json found. Run run_experiment.py first.")

    out = base / "param_count_vs_perf.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

