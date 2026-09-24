"""Compare stored official and fixed-path scores on shared evaluable windows."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from src.analysis.paired_work_bootstrap import METRICS, REPORTED


def mean_absolute_difference(
    official: list[dict[str, float]], implementation: list[dict[str, float]]
) -> dict[str, float]:
    if not official or len(official) != len(implementation):
        raise ValueError("Expected nonempty, equally sized paired score lists")
    differences = {metric: [] for metric in REPORTED}
    for old, new in zip(official, implementation, strict=True):
        pair = []
        for raw in (old, new):
            scores = {metric: float(raw[metric]) for metric in METRICS}
            if any(not math.isfinite(v) or not 0 <= v <= 1 + 1e-9 for v in scores.values()):
                raise ValueError("Expected finite component scores between zero and one")
            # Composite differences must be taken after averaging each window's components.
            scores["MV2H4"] = sum(scores[m] for m in METRICS if m != "Meter") / 4
            scores["MV2H5"] = sum(scores[m] for m in METRICS) / 5
            pair.append(scores)
        for metric in REPORTED:
            differences[metric].append(abs(pair[0][metric] - pair[1][metric]))
    return {metric: math.fsum(values) / len(values) for metric, values in differences.items()}


def load_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["system"], row["chunk_id"]
            if key in rows:
                raise ValueError(f"Duplicate system/window pair in {path}: {key}")
            rows[key] = row
    return rows


def score_ranges(rows: list[dict[str, float]]) -> dict:
    ranges = {
        metric: {"min": min(r[metric] for r in rows), "max": max(r[metric] for r in rows)}
        for metric in REPORTED
    }
    return {
        "by_metric": ranges,
        "all_metrics": {
            "min": min(r["min"] for r in ranges.values()),
            "max": max(r["max"] for r in ranges.values()),
        },
        "composites": {
            "min": min(ranges[m]["min"] for m in ("MV2H4", "MV2H5")),
            "max": max(ranges[m]["max"] for m in ("MV2H4", "MV2H5")),
        },
    }


def analyze_dataset(directory: Path) -> dict:
    task_path, result_path = directory / "tasks.jsonl", directory / "results.jsonl"
    tasks, results = load_rows(task_path), load_rows(result_path)
    if not tasks or tasks.keys() != results.keys():
        raise ValueError(f"Task/result inventories are empty or differ: {directory}")
    systems = sorted({system for system, _ in tasks})
    inventories = {s: {cid for system, cid in tasks if system == s} for s in systems}
    expected = inventories[systems[0]]
    if any(ids != expected for ids in inventories.values()):
        raise ValueError(f"Systems do not share the same input window inventory: {directory}")
    valid = {}
    for key, row in results.items():
        status = row["fixed_status"]
        if status == "success":
            if row["fixed"] is None:
                raise ValueError(f"Successful result has no scores: {key}")
            valid[key] = row["fixed"]
        elif status == "empty_score":
            # Existing empty-score results are scored zero; conversion failures have no score.
            valid[key] = dict.fromkeys(METRICS, 0.0)
    common = set.intersection(*[{cid for system, cid in valid if system == s} for s in systems])
    ids = sorted(common)
    rows = {}
    for system in systems:
        rows[system] = {
            "input_status_counts": dict(Counter(results[system, cid]["fixed_status"] for cid in expected)),
            "paired_windows": len(ids),
            "zero_windows_retained": sum(results[system, cid]["fixed_status"] == "empty_score" for cid in ids),
            "mean_absolute_difference": mean_absolute_difference(
                [tasks[system, cid]["official"] for cid in ids],
                [valid[system, cid] for cid in ids],
            ),
        }
    return {
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (task_path, result_path)
        },
        "candidate_windows": len(expected),
        "common_windows": len(common),
        "excluded_windows": sorted(expected - common),
        "systems": rows,
        "ranges": score_ranges([r["mean_absolute_difference"] for r in rows.values()]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=DIRECTORY",
                        help="Directory containing paired tasks.jsonl and results.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="New output JSON file")
    parser.add_argument("--row", action="append", default=[], metavar="ROW=SYSTEM,SYSTEM,...",
                        help="A table row whose systems are training seeds of one model; "
                        "its difference is the mean over them, and the summary averages rows")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    datasets = {}
    for spec in args.dataset:
        name, separator, directory = spec.partition("=")
        if not separator or not name or not directory or name in datasets:
            parser.error(f"Expected unique NAME=DIRECTORY: {spec}")
        datasets[name] = analyze_dataset(Path(directory))
    table_rows = {}
    for spec in args.row:
        name, separator, systems = spec.partition("=")
        if not separator or not name or not systems or name in table_rows:
            parser.error(f"Expected unique ROW=SYSTEM,...: {spec}")
        table_rows[name] = systems.split(",")
    for name, dataset in datasets.items():
        if not table_rows:
            break
        missing = [s for systems in table_rows.values() for s in systems if s not in dataset["systems"]]
        if missing:
            parser.error(f"{name}: rows name systems it has no scores for: {missing}")
        per_row = {
            row: {m: sum(dataset["systems"][s]["mean_absolute_difference"][m] for s in systems) / len(systems)
                  for m in REPORTED}
            for row, systems in table_rows.items()
        }
        dataset["rows"] = per_row
        dataset["mean_over_rows"] = {
            m: sum(r[m] for r in per_row.values()) / len(per_row) for m in REPORTED
        }
    rows = [r["mean_absolute_difference"] for d in datasets.values() for r in d["systems"].values()]
    payload = {
        "statistic": "mean of per-window absolute score differences",
        "window_policy": "all-system evaluable intersection within each dataset; scored zeros retained",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "datasets": datasets,
        "ranges": score_ranges(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for name, dataset in datasets.items():
        print(f"{name}: {dataset['common_windows']} / {dataset['candidate_windows']} windows")
        for system, row in dataset["systems"].items():
            values = " ".join(f"{m}={row['mean_absolute_difference'][m]:.6f}" for m in REPORTED)
            print(f"  {system}: {values}")
        if "rows" in dataset:
            for row, values in dataset["rows"].items():
                print(f"  row {row}: " + " ".join(f"{m}={values[m]:.6f}" for m in REPORTED))
            summary = dataset["mean_over_rows"]
            largest = max(summary, key=summary.get)
            print(f"  mean over rows: MV2H5={summary['MV2H5']:.4f}, largest {largest}={summary[largest]:.4f}")
    print(json.dumps(payload["ranges"], indent=2))


if __name__ == "__main__":
    main()
