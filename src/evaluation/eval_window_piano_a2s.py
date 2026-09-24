#!/usr/bin/env python3
"""Render score predictions and evaluate them against frozen GT MIDI."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.evaluation.asap import sha256_file
from src.evaluation.mv2h import (
    EvaluationResult,
    MidiPairTask,
    MV2HResult,
    evaluate_midi_pair,
    print_evaluation_summary,
    summarize_evaluations,
)

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "task_id", "chunk_index", "piece_id", "performance",
    "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony",
    "MV2H", "MV2H_custom", "MV2H4", "MV2H5", "pred_path", "gt_path",
]


def _parse_chunk_id(chunk_id: str) -> Dict[str, Any]:
    piece_id, separator, perf_chunk = chunk_id.rpartition("#")
    if not separator:
        return {"piece_id": chunk_id, "performance": "", "chunk_index": 0}
    performance, dot, position = perf_chunk.rpartition(".")
    if not dot:
        return {"piece_id": piece_id, "performance": perf_chunk, "chunk_index": 0}
    try:
        chunk_index = int(position)
    except ValueError:
        chunk_index = 0
    return {
        "piece_id": piece_id,
        "performance": performance,
        "chunk_index": chunk_index,
    }


def _load_grounding(path: Path) -> List[dict]:
    entries: List[dict] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"Missing chunk_id at {path}:{line_number}")
            if chunk_id in seen:
                raise ValueError(f"Duplicate grounding chunk_id: {chunk_id}")
            seen.add(chunk_id)
            entries.append(entry)
    return entries


def _grounding_shard(
    entries: List[dict], start: Optional[int], end: Optional[int]
) -> List[dict]:
    shard = entries[slice(start, end)]
    if not shard:
        raise ValueError(f"Empty grounding shard [{start}, {end})")
    return shard


def _validate_native_references(entries: List[dict], root: Path) -> None:
    for entry in entries:
        if entry.get("bar_count_authority") != "frozen_mapping":
            continue
        chunk_id = entry["chunk_id"]
        if entry.get("reference_status") != "ready":
            raise RuntimeError(f"Native reference is not ready: {chunk_id}")
        relative = entry.get("reference_midi")
        expected_hash = entry.get("reference_midi_sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"Native reference provenance is incomplete: {chunk_id}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Native reference MIDI is missing: {path}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"Native reference MIDI hash changed: {chunk_id}")


def _pred_midi_path(output_dir: Path, chunk_id: str) -> Path:
    return output_dir / "pred_midi" / f"{chunk_id.replace('/', '_')}_pred.mid"


def _prediction_manifest_path(
    output_dir: Path,
    start: Optional[int],
    end: Optional[int],
) -> Path:
    if start is None and end is None:
        return output_dir / "prediction_manifest.jsonl"
    start_value = start or 0
    end_value = end if end is not None else start_value
    return output_dir / (
        f"prediction_manifest_{start_value:06d}_{end_value:06d}.jsonl"
    )


def write_prediction_manifest(
    grounding: List[dict],
    pred_dir: Path,
    output_dir: Path,
    output: Path,
) -> None:
    rows = []
    for entry in grounding:
        chunk_id = entry["chunk_id"]
        kern_path = pred_dir / f"{chunk_id}.krn"
        midi_path = _pred_midi_path(output_dir, chunk_id)
        if midi_path.is_file():
            status = "ready"
            error_message = ""
        elif not kern_path.is_file():
            status = "missing_pred_kern"
            error_message = "Prediction kern does not exist"
        else:
            status = "pred_midi_conversion_failed"
            error_message = "Prediction kern could not be rendered to MIDI"
        rows.append({
            "chunk_id": chunk_id,
            "status": status,
            "error_message": error_message,
            "prediction_midi": (
                midi_path.relative_to(output_dir).as_posix()
                if midi_path.is_file() else None
            ),
        })
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ))
    temporary.replace(output)


def merge_prediction_manifests(
    output_dir: Path,
    grounding: List[dict],
) -> Path:
    rows: Dict[str, dict] = {}
    for path in sorted(output_dir.glob("prediction_manifest_[0-9]*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = row["chunk_id"]
                if chunk_id in rows:
                    raise ValueError(
                        f"Duplicate chunk_id across prediction manifests: {chunk_id}"
                    )
                rows[chunk_id] = row
    expected = [entry["chunk_id"] for entry in grounding]
    missing = [chunk_id for chunk_id in expected if chunk_id not in rows]
    extra = sorted(set(rows) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Prediction manifest coverage mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    output = output_dir / "prediction_manifest.jsonl"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(rows[chunk_id], ensure_ascii=False, sort_keys=True) + "\n"
        for chunk_id in expected
    ))
    temporary.replace(output)
    return output


def _render_kern_to_midi(kern_text: str, output_midi: Path) -> bool:
    """Run our system's own kern→MusicXML→MIDI chain."""
    import music21

    from src.score.generate_score import kern_to_musicxml
    from src.score.kern_postprocess import kern_pre_midi_clean

    cleaned = kern_pre_midi_clean(kern_text, n_spines=2)
    tmp_krn = tmp_xml = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".krn", delete=False, mode="w"
        ) as handle:
            tmp_krn = handle.name
            handle.write(cleaned)
        with tempfile.NamedTemporaryFile(
            suffix=".musicxml", delete=False
        ) as handle:
            tmp_xml = handle.name
        kern_to_musicxml(tmp_krn, tmp_xml)
        score = music21.converter.parse(tmp_xml)
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(output_midi))
        return output_midi.exists()
    except Exception as error:
        logger.debug("kern→MIDI failed: %s", error)
        output_midi.unlink(missing_ok=True)
        return False
    finally:
        for temporary in (tmp_krn, tmp_xml):
            if temporary:
                Path(temporary).unlink(missing_ok=True)


def _render_worker(args: tuple[str, str, str]) -> tuple[str, str]:
    chunk_id, kern_path_raw, midi_path_raw = args
    kern_path = Path(kern_path_raw)
    midi_path = Path(midi_path_raw)
    if midi_path.exists():
        return chunk_id, "ready"
    if not kern_path.exists():
        return chunk_id, "missing_pred_kern"
    if _render_kern_to_midi(kern_path.read_text(), midi_path):
        return chunk_id, "ready"
    return chunk_id, "pred_midi_conversion_failed"


def render_predictions(
    pred_dir: Path,
    grounding: List[dict],
    output_dir: Path,
    workers: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    missing_gt = sum(
        entry.get("gt_midi_status") != "ready" for entry in grounding
    )
    if missing_gt:
        counts["gt_midi_missing"] = missing_gt
    tasks = [
        (
            entry["chunk_id"],
            str(pred_dir / f"{entry['chunk_id']}.krn"),
            str(_pred_midi_path(output_dir, entry["chunk_id"])),
        )
        for entry in grounding
        if entry.get("gt_midi_status") == "ready"
    ]
    n_workers = min(max(workers, 1), max(len(tasks), 1))
    if n_workers == 1:
        outcomes = map(_render_worker, tasks)
        for _, status in tqdm(
            outcomes, total=len(tasks), desc="render pred MIDI", unit="chunk"
        ):
            counts[status] = counts.get(status, 0) + 1
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_render_worker, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(tasks),
                desc="render pred MIDI", unit="chunk",
            ):
                _, status = future.result()
                counts[status] = counts.get(status, 0) + 1
    logger.info("Render status: %s", dict(sorted(counts.items())))
    return counts


def _missing_result(
    entry: dict,
    pred_dir: Path,
    output_dir: Path,
    gt_root: Path,
) -> Optional[EvaluationResult]:
    chunk_id = entry["chunk_id"]
    pred_kern = pred_dir / f"{chunk_id}.krn"
    pred_midi = _pred_midi_path(output_dir, chunk_id)
    gt_relative = entry.get("reference_midi") or entry.get("gt_midi")
    gt_midi = gt_root / gt_relative if gt_relative else None
    reference_status = entry.get("reference_status", entry.get("gt_midi_status"))
    if reference_status != "ready" or gt_midi is None:
        if entry.get("bar_count_authority") == "frozen_mapping":
            raise RuntimeError(f"Native reference is not ready: {chunk_id}")
        return EvaluationResult(
            task_id=chunk_id,
            pred_path=str(pred_midi),
            gt_path="",
            status="gt_midi_missing",
            error_message="Grounding has no GT MIDI",
        )
    if not gt_midi.exists():
        if entry.get("bar_count_authority") == "frozen_mapping":
            raise FileNotFoundError(
                f"Native reference MIDI is missing for {chunk_id}: {gt_midi}"
            )
        return EvaluationResult(
            task_id=chunk_id,
            pred_path=str(pred_midi),
            gt_path=str(gt_midi),
            status="gt_midi_missing",
            error_message="Grounded GT MIDI path does not exist",
        )
    if not pred_kern.exists():
        return EvaluationResult(
            task_id=chunk_id,
            pred_path=str(pred_midi),
            gt_path=str(gt_midi),
            status="missing_pred_kern",
            error_message="Grounded kern slice of our system does not exist",
        )
    if not pred_midi.exists():
        return EvaluationResult(
            task_id=chunk_id,
            pred_path=str(pred_midi),
            gt_path=str(gt_midi),
            status="pred_midi_conversion_failed",
            error_message="Prediction MIDI of our system was not rendered",
        )
    return None


def evaluate_predictions(
    pred_dir: Path,
    grounding: List[dict],
    gt_root: Path,
    output_dir: Path,
    mv2h_bin: str,
    timeout: int,
    workers: int,
) -> List[EvaluationResult]:
    immediate: Dict[str, EvaluationResult] = {}
    tasks: List[MidiPairTask] = []
    slice_path = pred_dir / '_slice_status.jsonl'
    slices = {
        row['chunk_id']: row for row in (
            json.loads(line) for line in slice_path.read_text().splitlines() if line.strip()
        )
    } if slice_path.is_file() else {}
    for entry in grounding:
        sliced = slices.get(entry['chunk_id'])
        if sliced and sliced['status'] != 'ready':
            immediate[entry['chunk_id']] = EvaluationResult(
                task_id=entry['chunk_id'],
                pred_path=str(_pred_midi_path(output_dir, entry['chunk_id'])),
                gt_path=str(gt_root / (entry.get('reference_midi') or entry['gt_midi'])),
                status=sliced['status'], error_message=sliced.get('error', ''),
            )
            continue
        missing = _missing_result(entry, pred_dir, output_dir, gt_root)
        if missing is not None:
            immediate[entry["chunk_id"]] = missing
            continue
        chunk_id = entry["chunk_id"]
        pred_midi = _pred_midi_path(output_dir, chunk_id)
        # A prediction MIDI cut at an arbitrary performance time carries its
        # pick-up length beside it; a plain rendered window has none.
        anacrusis = None
        sidecar = pred_midi.with_suffix(".json")
        if sidecar.is_file():
            anacrusis = json.loads(sidecar.read_text()).get("anacrusis_subbeats")
        tasks.append(MidiPairTask(
            task_id=chunk_id,
            pred_midi=str(pred_midi),
            gt_midi=str(gt_root / (
                entry.get("reference_midi") or entry["gt_midi"]
            )),
            mv2h_bin=mv2h_bin,
            timeout=timeout,
            pred_anacrusis=anacrusis,
        ))

    evaluated: Dict[str, EvaluationResult] = {}
    n_workers = min(max(workers, 1), max(len(tasks), 1))
    if n_workers == 1:
        for task in tqdm(tasks, desc="MV2H", unit="chunk"):
            evaluated[task.task_id] = evaluate_midi_pair(task)
    elif tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(evaluate_midi_pair, task): task for task in tasks}
            for future in tqdm(
                as_completed(futures), total=len(tasks), desc="MV2H", unit="chunk"
            ):
                task = futures[future]
                try:
                    evaluated[task.task_id] = future.result()
                except Exception as error:
                    evaluated[task.task_id] = EvaluationResult(
                        task_id=task.task_id,
                        pred_path=task.pred_midi,
                        gt_path=task.gt_midi,
                        status="executor_error",
                        error_message=str(error),
                    )
    return [
        immediate.get(entry["chunk_id"]) or evaluated[entry["chunk_id"]]
        for entry in grounding
    ]


def _result_row(result: EvaluationResult) -> Dict[str, Any]:
    row = result.to_dict()
    if result.metrics is not None:
        row["MV2H4"] = result.metrics.mv2h_custom
        row["MV2H5"] = (
            result.metrics.multi_pitch + result.metrics.voice
            + result.metrics.meter + result.metrics.value
            + result.metrics.harmony
        ) / 5
    else:
        row["MV2H4"] = 0.0
        row["MV2H5"] = 0.0
    row.update(_parse_chunk_id(result.task_id))
    return row


def write_results(results: List[EvaluationResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(_result_row(result))
    temporary.replace(output)


def _row_result(row: dict) -> EvaluationResult:
    metrics = None
    # A zero score is still MV2H's result; its shard row carries the metrics.
    if row.get("status") in ("success", "zero_score"):
        metrics = MV2HResult.from_dict({
            key: float(row[key])
            for key in (
                "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H"
            )
        })
    return EvaluationResult(
        task_id=row["task_id"],
        pred_path=row.get("pred_path", ""),
        gt_path=row.get("gt_path", ""),
        status=row.get("status", "error"),
        error_message=row.get("error_message", ""),
        metrics=metrics,
    )


def merge_shards(
    output_dir: Path,
    grounding: List[dict],
    output_csv: Path,
) -> List[EvaluationResult]:
    rows: Dict[str, dict] = {}
    shards = sorted(output_dir.glob("eval_asap_shard_*.csv"))
    if not shards:
        raise FileNotFoundError(f"No shard CSV files in {output_dir}")
    for shard in shards:
        with shard.open() as handle:
            for row in csv.DictReader(handle):
                task_id = row["task_id"]
                if task_id in rows:
                    raise ValueError(f"Duplicate task_id across shards: {task_id}")
                rows[task_id] = row
    expected = [entry["chunk_id"] for entry in grounding]
    missing = [chunk_id for chunk_id in expected if chunk_id not in rows]
    extra = sorted(set(rows) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Shard coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )
    results = [_row_result(rows[chunk_id]) for chunk_id in expected]
    write_results(results, output_csv)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render kern predictions and evaluate explicit GT-MIDI grounding"
    )
    parser.add_argument(
        "--phase", choices=("render", "eval", "all", "merge"), default="all"
    )
    parser.add_argument("--pred-dir", type=Path)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument(
        "--piano-a2s-results-dir", type=Path,
        help="Legacy external-reference alias for --reference-root",
    )
    parser.add_argument("--mv2h-bin", default="external/MV2H/bin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--chunk-start", type=int)
    parser.add_argument("--chunk-end", type=int)
    parser.add_argument("--workers", "-j", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--chunk-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    grounding = _load_grounding(args.grounding)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slice_summary = args.pred_dir / '_slice_summary.json' if args.pred_dir else None
    protocol = json.loads(slice_summary.read_text()) if slice_summary and slice_summary.is_file() else {}
    if args.phase in ('eval', 'all', 'merge'):
        metadata = {
            'pairing': protocol.get('pairing', 'pre_segmented'),
            'timeout': args.chunk_timeout,
            'grounding': str(args.grounding.resolve()),
            'reference_root': str(args.reference_root.resolve()) if args.reference_root else None,
            'pred_dir': str(args.pred_dir.resolve()) if args.pred_dir else None,
        }
        temporary = args.output_dir / f'evaluation_inputs.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(metadata, sort_keys=True) + '\n')
        temporary.replace(args.output_dir / 'evaluation_inputs.json')
    output_csv = args.output_csv or args.output_dir / "eval_asap.csv"

    if args.phase == "merge":
        prediction_manifest = merge_prediction_manifests(
            args.output_dir, grounding
        )
        results = merge_shards(args.output_dir, grounding, output_csv)
        print_evaluation_summary(summarize_evaluations(results, len(grounding)))
        print(f"Prediction manifest: {prediction_manifest}")
        print(f"Results CSV: {output_csv}")
        return

    if args.pred_dir is None or not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {args.pred_dir}")
    gt_root = args.reference_root or args.piano_a2s_results_dir
    if gt_root is None or not gt_root.exists():
        raise FileNotFoundError(
            f"Reference root does not exist: {gt_root}"
        )
    if args.phase in ("eval", "all") and not Path(args.mv2h_bin).exists():
        raise FileNotFoundError(f"MV2H binary directory does not exist: {args.mv2h_bin}")

    shard = _grounding_shard(grounding, args.chunk_start, args.chunk_end)
    _validate_native_references(shard, gt_root)
    print(f"Phase: {args.phase}")
    print(f"Grounding rows: {len(shard)}")
    print(f"Prediction kern: {args.pred_dir}")
    print(f"Reference GT: {gt_root}")
    print(f"Output: {args.output_dir}")

    if args.phase in ("render", "all"):
        render_predictions(args.pred_dir, shard, args.output_dir, args.workers)
        manifest_path = _prediction_manifest_path(
            args.output_dir, args.chunk_start, args.chunk_end
        )
        write_prediction_manifest(
            shard, args.pred_dir, args.output_dir, manifest_path
        )
        print(f"Prediction manifest: {manifest_path}")
    if args.phase in ("eval", "all"):
        results = evaluate_predictions(
            args.pred_dir,
            shard,
            gt_root,
            args.output_dir,
            args.mv2h_bin,
            args.chunk_timeout,
            args.workers,
        )
        if args.output_csv is None and (
            args.chunk_start is not None or args.chunk_end is not None
        ):
            start = args.chunk_start or 0
            end = args.chunk_end if args.chunk_end is not None else start + len(shard)
            output_csv = args.output_dir / f"eval_asap_shard_{start:06d}_{end:06d}.csv"
        write_results(results, output_csv)
        print_evaluation_summary(summarize_evaluations(results, len(shard)))
        print(f"Results CSV: {output_csv}")


if __name__ == "__main__":
    main()
