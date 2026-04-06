from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import flatten_dict, save_json, setup_logger


SUMMARY_FIELDS = [
    "metrics.accuracy",
    "metrics.total_runtime_seconds",
    "metrics.samples_per_second",
    "metrics.tokens_per_second",
    "metrics.gpu_peak_memory_allocated_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics.")
    parser.add_argument("--input-dir", type=str, default="outputs")
    parser.add_argument("--output-dir", type=str, default="outputs/summary")
    return parser.parse_args()


def collect_metrics(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in root.rglob("eval_metrics.json"):
        experiment_dir = metrics_path.parent
        with metrics_path.open("r", encoding="utf-8") as handle:
            record = flatten_dict(json.load(handle))
        record["experiment_dir"] = str(experiment_dir)
        rows.append(record)
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = ["experiment_dir"] + SUMMARY_FIELDS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    args = parse_args()
    logger = setup_logger("summarizing")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_metrics(input_dir)
    save_json({"experiments": rows}, output_dir / "summary.json")
    write_csv(rows, output_dir / "summary.csv")
    logger.info("Collected %s experiment result files", len(rows))
    logger.info("Saved summary outputs to %s", output_dir)


if __name__ == "__main__":
    main()
