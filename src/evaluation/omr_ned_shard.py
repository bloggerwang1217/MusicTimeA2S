#!/usr/bin/env python3
"""Shard and merge system-independent OMR-NED score-pair evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .omr_ned import (
    OMRNEDResult,
    ScorePairTask,
    aggregate_omr_ned_results,
    evaluate_score_pair,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JSONL inventory is empty: {path}")
    return rows


def _task_id(row: dict[str, Any]) -> str:
    value = row.get("task_id") or row.get("artifact_stem") or row.get("recording_id")
    if not value:
        raise ValueError("Every pair row needs task_id, artifact_stem, or recording_id")
    return str(value)


def _path_field(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return str(value)
    raise ValueError(f"Pair {_task_id(row)!r} lacks every path field {names}")


def _validate_inventory(rows: list[dict[str, Any]]) -> list[str]:
    ids = [_task_id(row) for row in rows]
    duplicates = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate pair task IDs: {duplicates[:10]}")
    for row in rows:
        _path_field(row, "prediction_xml", "pred_path", "pred_score")
        _path_field(row, "reference_xml", "gt_path", "gt_score")
    return ids


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_shard(
    pairs_path: Path,
    output_dir: Path,
    start: int,
    end: int,
    musicdiff_root: Path,
) -> Path:
    pairs = _read_jsonl(pairs_path)
    _validate_inventory(pairs)
    if start < 0 or end <= start or end > len(pairs):
        raise ValueError(f"Invalid shard [{start}, {end}) for {len(pairs)} pairs")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    for source in pairs[start:end]:
        task = ScorePairTask(
            task_id=_task_id(source),
            pred_score=_path_field(source, "prediction_xml", "pred_path", "pred_score"),
            gt_score=_path_field(source, "reference_xml", "gt_path", "gt_score"),
            musicdiff_root=str(musicdiff_root),
        )
        result = evaluate_score_pair(task).to_dict()
        for name in ("artifact_stem", "recording_id", "piece_id"):
            if source.get(name) is not None:
                result[name] = source[name]
        output_rows.append(result)

    output_path = output_dir / f"shard_{start:06d}_{end:06d}.jsonl"
    _atomic_write_jsonl(output_path, output_rows)
    return output_path


def merge_shards(pairs_path: Path, output_dir: Path) -> tuple[Path, Path]:
    pairs = _read_jsonl(pairs_path)
    expected_ids = _validate_inventory(pairs)
    expected = set(expected_ids)

    by_id: dict[str, dict[str, Any]] = {}
    for shard_path in sorted(output_dir.glob("shard_*.jsonl")):
        for row in _read_jsonl(shard_path):
            task_id = str(row.get("task_id", ""))
            if not task_id:
                raise ValueError(f"Shard row lacks task_id: {shard_path}")
            if task_id in by_id:
                raise ValueError(f"Duplicate shard result: {task_id}")
            by_id[task_id] = row

    missing = sorted(expected - set(by_id))
    unexpected = sorted(set(by_id) - expected)
    if missing or unexpected:
        raise ValueError(
            "OMR-NED shard inventory mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    ordered = [by_id[task_id] for task_id in expected_ids]
    results_path = output_dir / "results.jsonl"
    _atomic_write_jsonl(results_path, ordered)

    statuses = Counter(str(row.get("status", "")) for row in ordered)
    failed = [row for row in ordered if row.get("status") != "success"]
    summary: dict[str, Any] = {
        "n_total": len(ordered),
        "n_success": statuses.get("success", 0),
        "status_counts": dict(sorted(statuses.items())),
    }
    if not failed:
        metrics = [
            OMRNEDResult(
                edit_distance=int(row["OMR-ED"]),
                pred_symbols=int(row["pred_symbols"]),
                gt_symbols=int(row["gt_symbols"]),
                omr_ned=float(row["OMR-NED"]),
            )
            for row in ordered
        ]
        summary.update(aggregate_omr_ned_results(metrics))
        summary["mean_OMR-NED_percent"] = 100.0 * float(summary["mean_OMR-NED"])
        summary["corpus_OMR-NED_percent"] = 100.0 * float(summary["corpus_OMR-NED"])

    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(summary_path)
    if failed:
        examples = [f"{row['task_id']}: {row.get('error_message', '')}" for row in failed[:10]]
        raise RuntimeError(
            f"OMR-NED evaluation failed closed for {len(failed)} pairs: {examples}"
        )
    return results_path, summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--musicdiff-root", type=Path, default=Path("external/efficient-musicdiff"))
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--merge", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.merge:
        if args.start is not None or args.end is not None:
            raise SystemExit("--merge cannot be combined with --start/--end")
        results, summary = merge_shards(args.pairs, args.output_dir)
        print(f"Results: {results}")
        print(f"Summary: {summary}")
        return
    if args.start is None or args.end is None:
        raise SystemExit("Shard mode requires --start and --end")
    print(evaluate_shard(
        args.pairs,
        args.output_dir,
        args.start,
        args.end,
        args.musicdiff_root,
    ))


if __name__ == "__main__":
    main()
