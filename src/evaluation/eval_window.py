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
from src.evaluation.prepare_preseg import validate_windows
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
        if entry["reference_status"] != "ready":
            continue
        relative = entry.get("reference_midi")
        expected_hash = entry.get("reference_midi_sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"Native reference provenance is incomplete: {entry['chunk_id']}")
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Native reference artifact mismatch: {path}")


def prediction_sources(grounding: List[dict], pred_dir: Path) -> tuple[dict[str, dict], dict]:
    return {
        row["chunk_id"]: {"kind": "kern", "path": str(pred_dir / f"{row['chunk_id']}.krn"),
                          "status": "ready"}
        for row in grounding
    }, {"input_format": "kern", "requested_windows": len(grounding)}


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


def write_prediction_manifest(grounding: List[dict], sources: dict[str, dict],
                              statuses: dict[str, str], output_dir: Path, output: Path) -> None:
    rows = []
    for entry in grounding:
        chunk_id = entry["chunk_id"]
        midi = _pred_midi_path(output_dir, chunk_id)
        source = sources[chunk_id]
        status = statuses[chunk_id]
        rows.append({
            "chunk_id": chunk_id, "status": status,
            "error_message": source.get("error_message", "") if status != "ready" else "",
            "source_chunk_id": source.get("source_chunk_id", chunk_id),
            "source_path": source.get("path"),
            "prediction_midi": midi.relative_to(output_dir).as_posix() if status == "ready" else None,
            "prediction_midi_sha256": sha256_file(midi) if status == "ready" else None,
        })
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
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


def _render_worker(args: tuple[str, str, str, str]) -> tuple[str, str]:
    chunk_id, source_raw, kind, midi_raw = args
    source, midi = Path(source_raw), Path(midi_raw)
    midi.unlink(missing_ok=True)
    if not source.is_file():
        return chunk_id, "missing_pred_kern"
    assert kind == "kern"
    ready = _render_kern_to_midi(source.read_text(), midi)
    return chunk_id, "ready" if ready else "pred_midi_conversion_failed"


def render_predictions(sources: dict[str, dict], grounding: List[dict],
                       output_dir: Path, workers: int) -> dict[str, str]:
    tasks, statuses = [], {}
    for entry in grounding:
        chunk_id = entry["chunk_id"]
        source = sources[chunk_id]
        if entry["reference_status"] != "ready":
            statuses[chunk_id] = entry["reference_status"]
        elif source["status"] != "ready":
            statuses[chunk_id] = source["status"]
        else:
            tasks.append((chunk_id, source["path"], source["kind"],
                          str(_pred_midi_path(output_dir, chunk_id))))
    if workers == 1:
        statuses.update(_render_worker(task) for task in tqdm(tasks, desc="render prediction MIDI"))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            statuses.update(executor.map(_render_worker, tasks))
    return statuses


def evaluate_predictions(predictions: dict[str, dict], grounding: List[dict],
                         gt_root: Path, output_dir: Path, mv2h_bin: str,
                         timeout: int, workers: int) -> List[EvaluationResult]:
    immediate: Dict[str, EvaluationResult] = {}
    tasks: List[MidiPairTask] = []
    for entry in grounding:
        chunk_id = entry["chunk_id"]
        prediction = predictions[chunk_id]
        gt_midi = gt_root / entry["reference_midi"] if entry.get("reference_midi") else None
        status = (entry["reference_status"] if entry["reference_status"] != "ready"
                  else prediction["status"])
        if status != "ready":
            immediate[chunk_id] = EvaluationResult(
                task_id=chunk_id, pred_path=str(_pred_midi_path(output_dir, chunk_id)),
                gt_path=str(gt_midi or ""), status=status,
                error_message=prediction.get("error_message", ""),
            )
            continue
        midi = output_dir / prediction["prediction_midi"]
        if not midi.is_file() or sha256_file(midi) != prediction["prediction_midi_sha256"]:
            raise RuntimeError(f"Rendered prediction MIDI changed: {midi}")
        tasks.append(MidiPairTask(task_id=chunk_id, pred_midi=str(midi), gt_midi=str(gt_midi),
                                  mv2h_bin=mv2h_bin, timeout=timeout))

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("render", "eval", "all", "merge"), default="all")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Kern excerpts named by evaluation chunk ID")
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--gt-score-midi-root", type=Path, required=True)
    parser.add_argument("--mv2h-bin", default="external/MV2H/bin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--chunk-start", type=int)
    parser.add_argument("--chunk-end", type=int)
    parser.add_argument("--workers", "-j", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--chunk-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    grounding = _load_grounding(args.grounding)
    validate_windows(grounding)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pairing": "recording_and_annotated_audio_interval", "n_bars": 5, "stride": 5,
        "timeout": args.chunk_timeout, "single_path": False,
        "grounding": str(args.grounding.resolve()), "grounding_sha256": sha256_file(args.grounding),
        "gt_score_midi_root": str(args.gt_score_midi_root.resolve()),
        "pred_dir": str(args.pred_dir.resolve()),
    }
    inputs = args.output_dir / "evaluation_inputs.json"
    if inputs.exists() and json.loads(inputs.read_text()) != metadata:
        raise ValueError(f"Evaluation inputs changed: use a new output directory: {args.output_dir}")
    temporary = inputs.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(metadata, sort_keys=True) + "\n")
    temporary.replace(inputs)
    output_csv = args.output_csv or args.output_dir / "eval_asap.csv"
    if args.phase == "merge":
        merge_prediction_manifests(args.output_dir, grounding)
        results = merge_shards(args.output_dir, grounding, output_csv)
        print_evaluation_summary(summarize_evaluations(results, len(grounding)))
        return
    sources, pairing = prediction_sources(grounding, args.pred_dir)
    temporary = args.output_dir / f"pairing.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(pairing, sort_keys=True) + "\n")
    temporary.replace(args.output_dir / "pairing.json")
    shard = _grounding_shard(grounding, args.chunk_start, args.chunk_end)
    _validate_native_references(shard, args.gt_score_midi_root)
    manifest_path = _prediction_manifest_path(args.output_dir, args.chunk_start, args.chunk_end)
    if args.phase in ("render", "all"):
        statuses = render_predictions(sources, shard, args.output_dir, args.workers)
        write_prediction_manifest(shard, sources, statuses, args.output_dir, manifest_path)
    if args.phase in ("eval", "all"):
        predictions = {row["chunk_id"]: row for row in _load_grounding(manifest_path)}
        if set(predictions) != {row["chunk_id"] for row in shard}:
            raise ValueError("Rendered prediction manifest differs from the requested shard")
        results = evaluate_predictions(predictions, shard, args.gt_score_midi_root, args.output_dir,
                                       args.mv2h_bin, args.chunk_timeout, args.workers)
        if args.output_csv is None and (args.chunk_start is not None or args.chunk_end is not None):
            start = args.chunk_start or 0
            end = args.chunk_end if args.chunk_end is not None else start + len(shard)
            output_csv = args.output_dir / f"eval_asap_shard_{start:06d}_{end:06d}.csv"
        write_results(results, output_csv)
        print_evaluation_summary(summarize_evaluations(results, len(shard)))
    print(f"Results CSV: {output_csv}")


if __name__ == "__main__":
    main()
