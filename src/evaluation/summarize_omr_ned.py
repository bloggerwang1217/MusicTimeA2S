#!/usr/bin/env python3
"""Summarize successful OMR-NED results and count failures separately."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.omr_ned import OMRNEDResult, aggregate_omr_ned_results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(
    pairs: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(row["task_id"]): row for row in results}
    metrics: list[OMRNEDResult] = []
    values: list[float] = []
    statuses: Counter[str] = Counter()
    failed: list[str] = []
    rows_out: list[dict[str, Any]] = []
    for pair in pairs:
        task_id = str(pair.get("task_id") or pair.get("artifact_stem") or pair.get("recording_id"))
        piece = str(pair.get("piece_id") or task_id)
        row = by_id.get(task_id)
        status = str(row.get("status", "missing")) if row else "missing"
        statuses[status] += 1
        if row and status == "success":
            metric = OMRNEDResult(
                edit_distance=int(row["OMR-ED"]),
                pred_symbols=int(row["pred_symbols"]),
                gt_symbols=int(row["gt_symbols"]),
                omr_ned=float(row["OMR-NED"]),
            )
        else:
            failed.append(task_id)
            rows_out.append({"task_id": task_id, "piece_id": piece, "status": status, "scored": False})
            continue
        metrics.append(metric)
        values.append(metric.omr_ned)
        rows_out.append({
            "task_id": task_id, "piece_id": piece, "status": status, "scored": True,
            **metric.to_dict(),
        })
    if not values:
        raise RuntimeError("no scored pairs")
    summary = {
        "failure_policy": "successful_only",
        "n_total": len(pairs),
        "n_success": statuses.get("success", 0),
        "n_fail": len(failed),
        "n_scored": len(values),
        "failed_task_ids": failed,
        "status_counts": dict(sorted(statuses.items())),
        "mean_OMR-NED_percent": 100.0 * float(np.mean(values)),
        "aggregate": aggregate_omr_ned_results(metrics),
        "rows": rows_out,
    }
    summary["corpus_OMR-NED_percent"] = 100.0 * float(summary["aggregate"]["corpus_OMR-NED"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(
        _read_jsonl(args.pairs), _read_jsonl(args.results),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
