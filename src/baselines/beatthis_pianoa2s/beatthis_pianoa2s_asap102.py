#!/usr/bin/env python3
"""Run Piano-A2S on tracked or annotated downbeats and assemble whole scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.evaluation.asap import load_asap102_inventory, sha256_file
from src.datasets.syn.syn_manifest import select_test_renders
from src.evaluation.omr_ned import (
    OMRNEDEvaluator,
    aggregate_omr_ned_results,
    read_folder_results,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ))
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _read_jsonl(path: Path, key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing {key} at {path}:{line_number}")
            if value in seen:
                raise ValueError(f"Duplicate {key} in {path}: {value}")
            seen.add(value)
            rows.append(row)
    return rows


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _syn_records(
    manifest_path: Path,
    syn_root: Path,
    *,
    select_renders: bool = True,
) -> list[dict[str, Any]]:
    """Turn Syn manifest rows into the recording records the stages expect.

    The manifest lists mel and kern paths; the cascade needs the rendered wav
    and the canonical MusicXML the audio was rendered from. MuseSyn scores are
    exported without the corpus prefix, and a few keep their ``.mscz`` origin
    in the file name. Every movement is its own piece for clustering.
    """
    xml_dir = syn_root / "xml"
    records: list[dict[str, Any]] = []
    entries = json.loads(manifest_path.read_text())
    # A rendered set outside the Syn corpus names its recordings after the
    # performance, not the timbre, so the test-render filter finds nothing.
    if select_renders:
        entries = select_test_renders(entries)
    for entry in entries:
        rendering_id = entry["id"]
        kern_stem = Path(entry["kern_gt_path"]).stem
        candidates = [
            xml_dir / f"{stem}{suffix}"
            for stem in (kern_stem, kern_stem.removeprefix("musesyn_"))
            for suffix in (".xml", ".mscz.xml")
        ]
        source_xml = next((path for path in candidates if path.is_file()), None)
        if source_xml is None:
            raise FileNotFoundError(f"No canonical MusicXML for {kern_stem}")
        # Downstream scorers join on the ASAP-102 spelling piece#performance.
        records.append({
            "recording_id": f"{kern_stem}#{rendering_id}",
            "artifact_stem": rendering_id,
            "piece_id": kern_stem,
            "performance_id": rendering_id,
            "source_xml": source_xml,
            "performance_audio": syn_root / "audio" / f"{rendering_id}.wav",
        })
    records.sort(key=lambda value: value["recording_id"])
    return records


def _inventory(
    metadata: Path | None,
    asap_root: Path | None,
    *,
    syn_manifest: Path | None = None,
    syn_root: Path | None = None,
    select_renders: bool = True,
    recording_list: Path | None = None,
) -> list[dict[str, Any]]:
    # A Syn manifest swaps the recording source; the record layout and every
    # stage after this point are shared with the ASAP-102 inventory.
    if syn_manifest is not None:
        records = _syn_records(
            syn_manifest, syn_root, select_renders=select_renders)
    else:
        records = load_asap102_inventory(metadata, asap_root)
    if recording_list is not None:
        selected = set(recording_list.read_text().splitlines()) - {"", "name"}
        matched = set()
        kept = []
        for record in records:
            aliases = {record["recording_id"], record["artifact_stem"]}
            if aliases & selected:
                kept.append(record)
                matched.update(aliases & selected)
        if not selected or matched != selected:
            raise ValueError(f"Unmatched recording selection: {sorted(selected - matched)}")
        records = kept
    rows: list[dict[str, Any]] = []
    for record in records:
        audio = Path(record["performance_audio"])
        if not audio.is_file():
            raise FileNotFoundError(
                f"ASAP-102 audio is missing for {record['recording_id']}: {audio}"
            )
        rows.append({
            "recording_id": record["recording_id"],
            "artifact_stem": record["artifact_stem"],
            "piece_id": record["piece_id"],
            "performance_id": record["performance_id"],
            "performance_audio": str(audio.resolve()),
            "performance_audio_sha256": sha256_file(audio),
            "source_xml": str(Path(record["source_xml"]).resolve()),
            "source_xml_sha256": sha256_file(Path(record["source_xml"])),
        })
    return rows


def _oracle_downbeats(
    records: list[dict[str, Any]], syn_manifest: Path | None,
) -> list[dict[str, Any]]:
    import soundfile as sf

    entries = {}
    manifest_sha256 = None
    if syn_manifest is not None:
        manifest_sha256 = sha256_file(syn_manifest)
        source_entries = json.loads(syn_manifest.read_text())
        entries = {entry["id"]: entry for entry in source_entries}
        if len(entries) != len(source_entries):
            raise ValueError("Duplicate Syn annotation identities")
    rows = []
    for record in records:
        audio = Path(record["performance_audio"])
        info = sf.info(audio)
        duration = info.frames / info.samplerate
        if syn_manifest is not None:
            entry = entries[record["artifact_stem"]]
            downbeats = [float(beat["sec"]) for beat in entry["audio_beats"]
                         if beat["is_downbeat"]]
            source = syn_manifest
        else:
            source = audio.with_name(f"{record['performance_id']}_annotations.txt")
            downbeats = []
            for line in source.read_text().splitlines():
                fields = line.split()
                if len(fields) >= 3 and fields[2].split(",")[0] == "db":
                    downbeats.append(float(fields[0]))
        if (len(downbeats) < 2
                or any(not math.isfinite(t) or not 0 <= t <= duration for t in downbeats)
                or any(b <= a for a, b in zip(downbeats, downbeats[1:]))):
            raise ValueError(f"Invalid annotated downbeats: {record['recording_id']}")
        # Preserve the event list: audio endpoints are not annotated downbeats.
        rows.append({
            **record, "status": "ready", "error_message": "",
            "downbeat_source": "oracle", "downbeats": downbeats,
            "annotation_path": str(source.resolve()),
            "annotation_sha256": manifest_sha256 or sha256_file(source),
            "duration_sec": duration,
            "uncovered_head_sec": downbeats[0],
            "uncovered_tail_sec": duration - downbeats[-1],
        })
    return rows


def _oracle_reference(
    records: list[dict[str, Any]], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapping = _read_jsonl(Path(config["reference_mapping"]), "artifact_stem")
    manifest = json.loads(Path(config["evaluation_manifest"]).read_text())
    stems = {record["artifact_stem"] for record in records}
    mapping = [row for row in mapping if row["artifact_stem"] in stems]
    manifest = [row for row in manifest if row["id"] in stems]
    if ({row["artifact_stem"] for row in mapping} != stems
            or {row["id"] for row in manifest} != stems
            or len(mapping) != len(stems) or len(manifest) != len(stems)):
        raise ValueError("Oracle reference inventory differs from selected recordings")
    by_stem = {row["artifact_stem"]: row for row in mapping}
    for row in manifest:
        if row["piece_id"] != by_stem[row["id"]]["piece_id"]:
            raise ValueError(f"Inconsistent reference work identity: {row['id']}")
    return mapping, manifest


def _assign_beatthis_checkpoints(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    policy = str(config.get("beatthis_checkpoint_policy", "fixed"))
    if policy == "fixed":
        checkpoint = str(config.get("beatthis_checkpoint", "final0"))
        assigned = [{
            **record,
            "beatthis_checkpoint": checkpoint,
            "beatthis_checkpoint_mapping": "fixed",
        } for record in records]
        return assigned, {f"fixed:{checkpoint}": len(assigned)}
    if policy != "holdout_fold":
        raise ValueError(f"Unknown Beat This! checkpoint policy: {policy}")

    split_raw = config.get("beatthis_fold_split")
    if not split_raw:
        raise ValueError("beatthis_fold_split is required for holdout_fold")
    split_path = Path(split_raw).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Beat This! fold split does not exist: {split_path}")
    fold_by_name: dict[str, int] = {}
    with split_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                name, fold_raw = line.rstrip().rsplit("\t", 1)
                fold = int(fold_raw)
            except ValueError as error:
                raise ValueError(
                    f"Invalid Beat This! fold row at {split_path}:{line_number}"
                ) from error
            if name in fold_by_name:
                raise ValueError(f"Duplicate Beat This! fold entry: {name}")
            fold_by_name[name] = fold

    absent_checkpoint = str(config.get("beatthis_absent_checkpoint", "fold0"))
    assigned: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        piece_id = record["piece_id"]
        performance_id = record["performance_id"]
        candidate = piece_id.replace("#", "_") + "_" + performance_id
        if candidate in fold_by_name:
            checkpoint = f"fold{fold_by_name[candidate]}"
            mapping = "holdout_direct"
        else:
            composer = piece_id.split("#")[0]
            candidates = [
                name for name in fold_by_name
                if name.endswith("_" + performance_id)
                and name.startswith(composer + "_")
            ]
            if len(candidates) == 1:
                checkpoint = f"fold{fold_by_name[candidates[0]]}"
                mapping = "holdout_renamed"
            elif not candidates:
                checkpoint = absent_checkpoint
                mapping = "absent_from_training"
            else:
                raise ValueError(
                    f"Ambiguous Beat This! fold mapping for "
                    f"{record['recording_id']}: {candidates}"
                )
        assigned.append({
            **record,
            "beatthis_checkpoint": checkpoint,
            "beatthis_checkpoint_mapping": mapping,
        })
        counts[f"{mapping}:{checkpoint}"] += 1
    return assigned, dict(sorted(counts.items()))


def _clean_child_environment() -> dict[str, str]:
    env = os.environ.copy()
    virtual_env = env.pop("VIRTUAL_ENV", None)
    if virtual_env:
        virtual_env_bin = os.path.normpath(str(Path(virtual_env) / "bin"))
        env["PATH"] = os.pathsep.join(
            entry
            for entry in env.get("PATH", "").split(os.pathsep)
            if os.path.normpath(entry) != virtual_env_bin
        )
    return env


def _validate_inventory_rows(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    expected_ids = [row["recording_id"] for row in expected]
    actual_ids = [row["recording_id"] for row in actual]
    if actual_ids != expected_ids:
        raise RuntimeError(f"{label} does not match the ordered ASAP-102 inventory")


def _failure_window(
    record: dict[str, Any],
    *,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        **record,
        "chunk_id": f"{record['artifact_stem']}.failure",
        "window_index": -1,
        "start_bar": 0,
        "start_sec": 0.0,
        "end_sec": 0.0,
        "duration_sec": 0.0,
        "keep_start_measure": 0,
        "keep_measure_count": 0,
        "status": status,
        "error_message": error_message,
    }


def _window_starts(n_bars: int) -> list[tuple[int, int, int]]:
    """Return (start, retained-offset, retained-count) for complete coverage."""
    if n_bars < 5:
        return []
    starts = list(range(0, n_bars - 4, 5))
    windows = [(start, 0, 5) for start in starts]
    covered = starts[-1] + 5
    if covered < n_bars:
        tail_start = n_bars - 5
        overlap = covered - tail_start
        windows.append((tail_start, overlap, 5 - overlap))
    return windows


def _build_windows(
    records: list[dict[str, Any]],
    beat_rows: list[dict[str, Any]],
    *,
    min_duration: float,
    max_duration: float,
) -> list[dict[str, Any]]:
    beats_by_id = {row["recording_id"]: row for row in beat_rows}
    rows: list[dict[str, Any]] = []
    for record in records:
        beat_row = beats_by_id[record["recording_id"]]
        if beat_row["status"] != "ready":
            rows.append(_failure_window(
                record,
                status=str(beat_row["status"]),
                error_message=str(beat_row.get("error_message", "")),
            ))
            continue
        downbeats = [float(value) for value in beat_row.get("downbeats", [])]
        if (
            len(downbeats) < 6
            or any(not math.isfinite(value) for value in downbeats)
            or any(right <= left for left, right in zip(downbeats, downbeats[1:]))
        ):
            rows.append(_failure_window(
                record,
                status="beatthis_invalid_downbeats",
                error_message="At least six strictly increasing downbeats are required",
            ))
            continue
        windows = _window_starts(len(downbeats) - 1)
        if not windows:
            rows.append(_failure_window(
                record,
                status="beatthis_insufficient_downbeats",
                error_message="Beat This! produced fewer than five complete bars",
            ))
            continue
        for window_index, (start, keep_start, keep_count) in enumerate(windows):
            start_sec = downbeats[start]
            end_sec = downbeats[start + 5]
            duration = end_sec - start_sec
            status = "ready"
            error_message = ""
            if duration < min_duration or duration > max_duration:
                status = "piano_a2s_duration_out_of_range"
                error_message = (
                    f"Five predicted bars last {duration:.6f}s; "
                    f"allowed range is [{min_duration}, {max_duration}]s"
                )
            rows.append({
                **record,
                "chunk_id": f"{record['artifact_stem']}.w{window_index:04d}",
                "window_index": window_index,
                "start_bar": start,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration,
                "keep_start_measure": keep_start,
                "keep_measure_count": keep_count,
                "status": status,
                "error_message": error_message,
            })
    return rows


def _evaluate_stage(
    records: list[dict[str, Any]],
    output_root: Path,
    *,
    prediction_manifest: Path,
    metric_dirname: str,
    failure_policy: str,
    musicdiff_root: Path,
) -> dict[str, Any]:
    predictions = _read_jsonl(prediction_manifest, "recording_id")
    prediction_by_id = {row["recording_id"]: row for row in predictions}
    expected = [record["recording_id"] for record in records]
    if set(prediction_by_id) != set(expected):
        raise ValueError("Prediction manifest does not match ASAP-102 inventory")

    pair_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="beatthis-pianoa2s-omr-") as temporary:
        pair_root = Path(temporary)
        gt_root = pair_root / "gt"
        pred_root = pair_root / "pred"
        gt_root.mkdir()
        pred_root.mkdir()
        for record in records:
            recording_id = record["recording_id"]
            prediction = prediction_by_id[recording_id]
            status = prediction["status"]
            prediction_xml = prediction.get("prediction_xml")
            gt_path = Path(record["source_xml"])
            if (
                not gt_path.is_file()
                or sha256_file(gt_path) != record["source_xml_sha256"]
            ):
                raise RuntimeError(f"Source XML hash mismatch: {recording_id}")
            if status == "ready" and isinstance(prediction_xml, str):
                pred_path = Path(prediction_xml)
                if not pred_path.is_file():
                    raise FileNotFoundError(f"Prediction XML is missing: {pred_path}")
                filename = f"{record['artifact_stem']}.musicxml"
                (pred_root / filename).symlink_to(pred_path.resolve())
                (gt_root / filename).symlink_to(gt_path.resolve())
            status_counts[status] += 1
            pair_rows.append({
                "recording_id": recording_id,
                "artifact_stem": record["artifact_stem"],
                "piece_id": record["piece_id"],
                "status": status,
                "prediction_xml": prediction_xml,
                "reference_xml": str(gt_path.resolve()),
                "reference_xml_sha256": record["source_xml_sha256"],
            })

        ready = status_counts.get("ready", 0)
        if ready == 0:
            raise RuntimeError("No predictions are evaluable")
        metric_root = output_root / metric_dirname
        evaluator = OMRNEDEvaluator(musicdiff_root)
        corpus_score, output_csv = evaluator.evaluate_folders(
            gt_root, pred_root, metric_root
        )

    folder_results = read_folder_results(output_csv)
    metrics_by_stem = {
        Path(row.pred_path).stem: row.metrics for row in folder_results
    }
    if len(metrics_by_stem) != ready:
        raise RuntimeError("musicdiff result count differs from ready predictions")
    values: list[float] = []
    for row in pair_rows:
        if row["status"] != "ready":
            continue
        metrics = metrics_by_stem[row["artifact_stem"]]
        row.update(metrics.to_dict())
        values.append(metrics.omr_ned)

    _write_jsonl(output_root / metric_dirname / "pairs.jsonl", pair_rows)
    summary = {
        "failure_policy": failure_policy,
        "n_total": len(records),
        "n_evaluable": len(values),
        "n_fail": len(records) - len(values),
        "status_counts": dict(sorted(status_counts.items())),
        "mean_OMR-NED_percent": 100.0 * float(np.mean(values)),
        "corpus_OMR-NED_percent": 100.0 * corpus_score,
        "aggregate": aggregate_omr_ned_results(list(metrics_by_stem.values())),
        "musicdiff_root": str(musicdiff_root.resolve()),
        "musicdiff_commit": _git_head(musicdiff_root),
        "musicdiff_output_csv": str(output_csv.resolve()),
    }
    if failure_policy == "blank_fill":
        summary.update({
            "n_blank_filled_recordings": sum(
                int(row.get("n_blank_filled_windows", 0)) > 0
                for row in predictions
            ),
            "n_blank_filled_windows": sum(
                int(row.get("n_blank_filled_windows", 0))
                for row in predictions
            ),
            "n_blank_filled_measures": sum(
                int(row.get("n_blank_filled_measures", 0))
                for row in predictions
            ),
        })
    _write_json(output_root / metric_dirname / "summary.json", summary)
    return summary


def _required_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if not value:
        raise ValueError(f"{label} must be configured")
    path = Path(value).resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} does not exist: {path}")
    return path


MACHINE_PATH_KEYS = (
    "asap_root", "beatthis_fold_split",
    "piano_a2s_repo", "piano_a2s_hparams", "piano_a2s_checkpoint",
)


BASELINES_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "baselines.yaml"


def load_baseline_config(config_file: Path, run: str) -> dict[str, Any]:
    """One run of the baselines file: its entry under ``runs`` on top of the
    shared keys (machine-local locations and the settings every run shares)."""
    document = yaml.safe_load(config_file.read_text()) or {}
    runs = document.get("runs") or {}
    if run not in runs:
        raise ValueError(f"{config_file}: no run named {run!r}; runs: {', '.join(sorted(runs))}")
    shared = {key: value for key, value in document.items() if key != "runs"}
    merged = {**shared, **(runs[run] or {})}
    for key in MACHINE_PATH_KEYS:
        if str(merged.get(key) or "").startswith("/path/to/"):
            raise ValueError(f"{config_file}: {key} is not filled in")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASELINES_CONFIG)
    parser.add_argument("--run", required=True, help="entry under `runs` in the config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decode-only", action="store_true")
    args = parser.parse_args()
    config = load_baseline_config(args.config, args.run)
    downbeat_source = config.get("downbeat_source", "beatthis")
    if downbeat_source not in {"beatthis", "oracle"}:
        raise ValueError(f"Unknown downbeat source: {downbeat_source}")

    project_root = Path.cwd().resolve()
    syn_manifest = None
    syn_root = None
    if config.get("syn_manifest"):
        syn_manifest = _required_path(config.get("syn_manifest"), "Syn manifest")
        syn_root = _required_path(config.get("syn_root"), "Syn root", directory=True)
        metadata = None
        asap_root = None
    else:
        metadata = _required_path(config.get("asap102_metadata"), "ASAP-102 metadata")
        asap_root = _required_path(config.get("asap_root"), "ASAP root", directory=True)
    output_root = Path(config["asap102_output_root"]).resolve()
    musicdiff_root = _required_path(
        config.get("musicdiff_root"), "musicdiff root", directory=True
    )
    piano_repo = _required_path(
        config.get("piano_a2s_repo"), "Piano-A2S repository", directory=True
    )
    piano_hparams = _required_path(
        config.get("piano_a2s_hparams"), "Piano-A2S hparams"
    )
    piano_checkpoint = _required_path(
        config.get("piano_a2s_checkpoint"),
        "Piano-A2S checkpoint",
        directory=True,
    )
    records = _inventory(
        metadata, asap_root, syn_manifest=syn_manifest, syn_root=syn_root,
        select_renders=bool(config.get("syn_select_test_renders", True)),
        recording_list=Path(config["recording_list"]) if config.get("recording_list") else None,
    )
    if downbeat_source == "oracle":
        if len(records) != int(config["expected_recordings"]):
            raise ValueError("Oracle inventory size differs from configured evaluation set")
        beatthis_checkpoint_summary = {}
        beat_rows = _oracle_downbeats(records, syn_manifest)
        reference_mapping, evaluation_manifest = _oracle_reference(records, config)
    else:
        records, beatthis_checkpoint_summary = _assign_beatthis_checkpoints(records, config)

    inventory_path = output_root / "inventory.jsonl"
    beat_output = output_root / "beatthis" / "predictions"
    beat_manifest = output_root / downbeat_source / "manifest.jsonl"
    chunks_input = output_root / "chunks" / "input-manifest.jsonl"
    chunks_output = output_root / "chunks" / "prediction-manifest.jsonl"
    prediction_manifest = output_root / "prediction-manifest.jsonl"
    blank_fill_manifest = output_root / "prediction-manifest-blank-fill.jsonl"
    beat_runner = Path(__file__).with_name("beatthis_runner.py").resolve()
    piano_runner = Path(__file__).with_name("piano_a2s_runner.py").resolve()
    beat_command = [
        "conda", "run", "-n", str(config.get("beatthis_conda_env", "beatthis")),
        "--no-capture-output", "python", str(beat_runner),
        "--inventory", str(inventory_path.resolve()),
        "--output-dir", str(beat_output.resolve()),
        "--manifest", str(beat_manifest.resolve()),
        "--checkpoint", str(config.get("beatthis_checkpoint", "final0")),
        "--device", str(config.get("beatthis_device", "cpu")),
    ]
    piano_command = [
        "conda", "run", "-n", str(config.get("piano_a2s_conda_env", "a2s2024")),
        "--no-capture-output", "python", str(piano_runner),
        "--repository", str(piano_repo),
        "--hparams", str(piano_hparams),
        "--checkpoint-dir", str(piano_checkpoint),
        "--inventory", str(inventory_path.resolve()),
        "--chunks", str(chunks_input.resolve()),
        "--output-root", str((output_root / "chunks").resolve()),
        "--chunk-manifest", str(chunks_output.resolve()),
        "--prediction-manifest", str(prediction_manifest.resolve()),
        "--blank-fill-prediction-manifest", str(blank_fill_manifest.resolve()),
        "--device", str(config.get("piano_a2s_device", "cuda")),
    ]

    preview = {
        "system": f"{downbeat_source}_pianoa2s",
        "downbeat_source": downbeat_source,
        "gpu_id": (os.environ.get("GPU", str(config.get("gpu_id", 0)))
                   if downbeat_source == "oracle" else str(config.get("gpu_id", 0))),
        "recordings": len(records),
        "output_root": str(output_root),
        "reference_policy": "ASAP-102 inventory source_xml with SHA256 verification",
        "beatthis_checkpoint_policy": config.get(
            "beatthis_checkpoint_policy", "fixed"
        ),
        "beatthis_checkpoint_summary": beatthis_checkpoint_summary,
        "beatthis_command": beat_command if downbeat_source == "beatthis" else None,
        "piano_a2s_command": piano_command,
    }
    if downbeat_source == "oracle":
        chunk_rows = _build_windows(
            records, beat_rows,
            min_duration=float(config.get("piano_a2s_min_duration", 4.0)),
            max_duration=float(config.get("piano_a2s_max_duration", 12.0)),
        )
        preview.update({
            "reference_policy": "unchanged selected rows of the configured reference mapping",
            "works": len({row["piece_id"] for row in reference_mapping}),
            "window_status_counts": dict(Counter(row["status"] for row in chunk_rows)),
            "reference_mapping": config["reference_mapping"],
            "reference_mapping_sha256": sha256_file(Path(config["reference_mapping"])),
            "evaluation_manifest": config["evaluation_manifest"],
            "evaluation_manifest_sha256": sha256_file(Path(config["evaluation_manifest"])),
            "endpoint_policy": "annotated downbeats only; no inferred endpoints",
        })
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(inventory_path, records)
    child_env = _clean_child_environment()
    child_env["CUDA_VISIBLE_DEVICES"] = preview["gpu_id"]

    if downbeat_source == "oracle":
        _write_json(output_root / "run_inputs.json", preview)
        _write_jsonl(beat_manifest, beat_rows)
        _write_jsonl(output_root / "reference" / "mapping.jsonl", reference_mapping)
        _write_json(output_root / "reference" / "manifest.json", evaluation_manifest)
        (output_root / "reference" / "recordings.txt").write_text(
            "name\n" + "\n".join(row["artifact_stem"] for row in records) + "\n"
        )
    else:
        completed = subprocess.run(beat_command, env=child_env, cwd=project_root)
        if completed.returncode != 0:
            raise RuntimeError(f"Beat This! process exited {completed.returncode}")
    beat_rows = _read_jsonl(beat_manifest, "recording_id")
    _validate_inventory_rows(records, beat_rows, label="Downbeat manifest")

    chunk_rows = _build_windows(
        records,
        beat_rows,
        min_duration=float(config.get("piano_a2s_min_duration", 4.0)),
        max_duration=float(config.get("piano_a2s_max_duration", 12.0)),
    )
    _write_jsonl(chunks_input, chunk_rows)
    completed = subprocess.run(piano_command, env=child_env, cwd=piano_repo)
    if completed.returncode != 0:
        raise RuntimeError(f"Piano-A2S process exited {completed.returncode}")
    prediction_rows = _read_jsonl(prediction_manifest, "recording_id")
    _validate_inventory_rows(records, prediction_rows, label="Piano-A2S manifest")
    blank_fill_rows = _read_jsonl(blank_fill_manifest, "recording_id")
    _validate_inventory_rows(
        records,
        blank_fill_rows,
        label="Piano-A2S blank-fill manifest",
    )

    if args.decode_only:
        print(json.dumps({"prediction_manifest": str(blank_fill_manifest),
                          "status_counts": dict(Counter(row["status"] for row in blank_fill_rows))}))
        return

    recording_failure_summary = _evaluate_stage(
        records,
        output_root,
        prediction_manifest=prediction_manifest,
        metric_dirname="omr_ned_recording_failure",
        failure_policy="recording_failure",
        musicdiff_root=musicdiff_root,
    )
    blank_fill_summary = _evaluate_stage(
        records,
        output_root,
        prediction_manifest=blank_fill_manifest,
        metric_dirname="omr_ned_blank_fill",
        failure_policy="blank_fill",
        musicdiff_root=musicdiff_root,
    )
    print(json.dumps({
        "recording_failure": recording_failure_summary,
        "blank_fill": blank_fill_summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
