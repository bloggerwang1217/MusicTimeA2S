#!/usr/bin/env python3
"""Run the external Piano-A2S checkpoint on predicted five-bar windows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ))
    temporary.replace(path)


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _fingerprint(
    row: dict[str, Any],
    *,
    checkpoint_sha256: str,
    hparams_sha256: str,
) -> str:
    fields = {
        key: row.get(key)
        for key in (
            "chunk_id", "performance_audio_sha256", "start_sec", "end_sec",
            "keep_start_measure", "keep_measure_count", "status",
        )
    }
    fields.update({
        "checkpoint_sha256": checkpoint_sha256,
        "hparams_sha256": hparams_sha256,
    })
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _decode_prediction(outputs: tuple[Any, Any, Any, Any], labels: Any, time_sigs: list[str]) -> list[list[Any]]:
    import torch

    eos = labels.labels_map["<eos>"]

    def unpad(sequence: Any) -> list[int]:
        values = sequence.detach().cpu().tolist()
        try:
            end = values.index(eos)
        except ValueError:
            end = len(values)
        return [int(value) for value in values[:end]]

    time_out, key_out, upper_out, lower_out = outputs
    time_ids = time_out[0].argmax(dim=-1).detach().cpu().tolist()
    key_ids = key_out[0].argmax(dim=-1).detach().cpu().tolist()
    upper_ids = upper_out[0].argmax(dim=-1)
    lower_ids = lower_out[0].argmax(dim=-1)
    prediction: list[list[Any]] = []
    for index in range(5):
        prediction.append([
            int(key_ids[index]) - 6,
            time_sigs[int(time_ids[index])],
            unpad(lower_ids[index]),
            unpad(upper_ids[index]),
        ])
    return prediction


def _write_native_xml(prediction: list[list[Any]], output: Path, converter: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".piano-a2s-xml-", dir=output.parent
    ) as temporary:
        previous = Path.cwd()
        try:
            os.chdir(temporary)
            score = converter(prediction)
            temporary_xml = Path(temporary) / "prediction.musicxml"
            score.write("musicxml", fp=str(temporary_xml))
            temporary_xml.replace(output)
        finally:
            os.chdir(previous)


def _assemble_recording(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    blank_fill: bool = False,
) -> None:
    import music21

    ordered = sorted(rows, key=lambda row: int(row["window_index"]))
    scores = {
        row["chunk_id"]: music21.converter.parse(row["prediction_xml"])
        for row in ordered
        if row["status"] == "ready"
    }
    if not scores:
        raise ValueError("At least one ready Piano-A2S window is required")
    if any(len(score.parts) != 2 for score in scores.values()):
        raise ValueError("Piano-A2S output must contain exactly two parts")

    assembled = copy.deepcopy(next(iter(scores.values())))
    assembled_parts = list(assembled.parts)
    for part in assembled_parts:
        for measure in list(part.getElementsByClass(music21.stream.Measure)):
            part.remove(measure)

    measure_number = 1
    for row in ordered:
        start = int(row["keep_start_measure"])
        count = int(row["keep_measure_count"])
        score = scores.get(row["chunk_id"])
        if score is None:
            if not blank_fill:
                raise ValueError(f"{row['chunk_id']}: window is not ready")
            for part in assembled_parts:
                for offset in range(count):
                    measure = music21.stream.Measure(number=measure_number + offset)
                    part.append(measure)
        else:
            for part_index, source_part in enumerate(score.parts):
                measures = list(source_part.getElementsByClass(music21.stream.Measure))
                if len(measures) < start + count:
                    raise ValueError(
                        f"{row['chunk_id']}: expected five output measures, "
                        f"found {len(measures)}"
                    )
                selected = measures[start:start + count]
                for offset, measure in enumerate(selected):
                    copied = copy.deepcopy(measure)
                    copied.number = measure_number + offset
                    assembled_parts[part_index].append(copied)
        measure_number += count

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".piano-a2s-full-", dir=output.parent
    ) as temporary:
        temporary_xml = Path(temporary) / "prediction.musicxml"
        assembled.write("musicxml", fp=str(temporary_xml))
        temporary_xml.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--hparams", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--blank-fill-prediction-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repository = args.repository.resolve()
    hparams_path = args.hparams.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    sys.path.insert(0, str(repository))
    os.chdir(repository)

    import numpy as np
    import torch
    import torchaudio
    from data_processing.humdrum import LabelsMultiple, get_xml_from_target
    from finetune import ASR
    from utilities import get_VQT, load, save

    inventory = _read_jsonl(args.inventory)
    chunks = _read_jsonl(args.chunks)
    checkpoint_sha256 = _tree_sha256(checkpoint_dir)
    hparams_sha256 = _sha256(hparams_path)
    repository_commit = _git_head(repository)

    wav_root = args.output_root / "wav"
    vqt_root = args.output_root / "vqt"
    json_root = args.output_root / "prediction_json"
    xml_root = args.output_root / "prediction_xml"
    full_root = args.output_root.parent / "full_xml"
    blank_fill_root = args.output_root.parent / "full_xml_blank_fill"
    for path in (
        wav_root, vqt_root, json_root, xml_root, full_root, blank_fill_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    rows_by_id: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for row in chunks:
        fingerprint = _fingerprint(
            row,
            checkpoint_sha256=checkpoint_sha256,
            hparams_sha256=hparams_sha256,
        )
        result_path = json_root / f"{row['chunk_id']}.json"
        xml_path = xml_root / f"{row['chunk_id']}.musicxml"
        if row["status"] != "ready":
            result = {
                **row,
                "input_fingerprint": fingerprint,
                "prediction_json": None,
                "prediction_xml": None,
                "error_message": row.get("error_message", ""),
            }
            rows_by_id[row["chunk_id"]] = result
            continue
        if result_path.is_file() and xml_path.is_file():
            existing = json.loads(result_path.read_text())
            if (
                existing.get("status") == "ready"
                and existing.get("input_fingerprint") == fingerprint
                and existing.get("prediction_xml_sha256") == _sha256(xml_path)
            ):
                rows_by_id[row["chunk_id"]] = existing
                continue
        pending.append({**row, "input_fingerprint": fingerprint})

    brain = None
    model = None
    hparams = None
    labels = None
    time_signatures = None
    if pending:
        hparams = load(str(hparams_path))
        hparams["checkpointer"].checkpoints_dir = checkpoint_dir
        hparams["save_folder"] = str(checkpoint_dir)
        brain = ASR(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts={"device": args.device},
            checkpointer=hparams["checkpointer"],
        )
        brain.checkpointer.recover_if_possible(device=torch.device(args.device))
        model = brain.modules.transcription
        model.eval()
        labels = LabelsMultiple(extended=True)
        time_signatures = load("data_processing/metadata/time_signature_list.json")

    audio_cache: dict[str, tuple[Any, int]] = {}
    for row in pending:
        result_path = json_root / f"{row['chunk_id']}.json"
        xml_path = xml_root / f"{row['chunk_id']}.musicxml"
        result = {
            **row,
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_sha256": checkpoint_sha256,
            "hparams": str(hparams_path),
            "hparams_sha256": hparams_sha256,
            "piano_a2s_repository": str(repository),
            "piano_a2s_commit": repository_commit,
        }
        try:
            audio_path = str(row["performance_audio"])
            if audio_path not in audio_cache:
                audio, sample_rate = torchaudio.load(audio_path)
                audio = audio.mean(dim=0)
                peak = audio.abs().max()
                if float(peak) > 0:
                    audio = audio / peak
                audio_cache[audio_path] = (audio, sample_rate)
            audio, sample_rate = audio_cache[audio_path]
            start = int(float(row["start_sec"]) * sample_rate)
            end = int(float(row["end_sec"]) * sample_rate)
            chunk_audio = audio[start:end]
            if chunk_audio.numel() == 0:
                raise ValueError("predicted downbeat window contains no audio")

            wav_path = wav_root / f"{row['chunk_id']}.wav"
            vqt_path = vqt_root / f"{row['chunk_id']}.npy"
            torchaudio.save(str(wav_path), chunk_audio.unsqueeze(0), sample_rate)
            assert hparams is not None
            vqt = get_VQT(str(wav_path), hparams["VQT_params"])
            save(vqt, str(vqt_path))

            max_frames = int(hparams["max_frame_num"])
            padded = torch.zeros((max_frames, vqt.shape[-1]), dtype=torch.float32)
            frames = min(max_frames, vqt.shape[0])
            padded[:frames] = torch.from_numpy(np.asarray(vqt[:frames])).float()
            spectrogram = padded.unsqueeze(0).unsqueeze(0).to(args.device)
            assert model is not None and labels is not None and time_signatures is not None
            with torch.no_grad():
                outputs = model(
                    spectrogram=spectrogram,
                    inference=True,
                    ground_truth=None,
                    teacher_forcing_ratio=0.0,
                    device=args.device,
                )
            prediction = _decode_prediction(outputs, labels, time_signatures)
            _write_native_xml(prediction, xml_path, get_xml_from_target)
            result.update({
                "status": "ready",
                "error_message": "",
                "prediction": prediction,
                "prediction_json": str(result_path.resolve()),
                "prediction_xml": str(xml_path.resolve()),
                "prediction_xml_sha256": _sha256(xml_path),
                "wav": str(wav_path.resolve()),
                "wav_sha256": _sha256(wav_path),
                "vqt": str(vqt_path.resolve()),
                "vqt_sha256": _sha256(vqt_path),
            })
        except Exception as error:
            result.update({
                "status": "piano_a2s_failed",
                "error_message": f"{type(error).__name__}: {error}",
                "prediction_json": str(result_path.resolve()),
                "prediction_xml": None,
            })
        _write_json(result_path, result)
        rows_by_id[row["chunk_id"]] = result

    ordered_chunk_rows = [rows_by_id[row["chunk_id"]] for row in chunks]
    _write_jsonl(args.chunk_manifest, ordered_chunk_rows)

    chunks_by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered_chunk_rows:
        chunks_by_recording[row["recording_id"]].append(row)

    recording_rows: list[dict[str, Any]] = []
    blank_fill_rows: list[dict[str, Any]] = []
    for record in inventory:
        recording_id = record["recording_id"]
        recording_chunks = chunks_by_recording.get(recording_id, [])
        full_xml = full_root / f"{record['artifact_stem']}.musicxml"
        status = "ready"
        error_message = ""
        if not recording_chunks:
            status = "piano_a2s_no_windows"
            error_message = "Beat This! produced fewer than five complete bars"
        else:
            failed = [row for row in recording_chunks if row["status"] != "ready"]
            if failed:
                status = str(failed[0]["status"])
                error_message = str(failed[0].get("error_message", ""))
            else:
                try:
                    _assemble_recording(recording_chunks, full_xml)
                except Exception as error:
                    status = "piano_a2s_assembly_failed"
                    error_message = f"{type(error).__name__}: {error}"

        recording_rows.append({
            "recording_id": recording_id,
            "artifact_stem": record["artifact_stem"],
            "piece_id": record["piece_id"],
            "performance_id": record["performance_id"],
            "status": status,
            "error_message": error_message,
            "prediction_xml": str(full_xml.resolve()) if status == "ready" else None,
            "prediction_xml_sha256": _sha256(full_xml) if status == "ready" else None,
            "piano_a2s_repository": str(repository),
            "piano_a2s_commit": repository_commit,
            "piano_a2s_checkpoint": str(checkpoint_dir),
            "piano_a2s_checkpoint_sha256": checkpoint_sha256,
            "piano_a2s_hparams_sha256": hparams_sha256,
        })
        blank_fill_xml = blank_fill_root / f"{record['artifact_stem']}.musicxml"
        blank_status = "ready"
        blank_error = ""
        blank_windows = [
            row["chunk_id"] for row in recording_chunks if row["status"] != "ready"
        ]
        blank_measures = sum(
            int(row["keep_measure_count"])
            for row in recording_chunks
            if row["status"] != "ready"
        )
        if not recording_chunks:
            blank_status = "piano_a2s_no_windows"
            blank_error = "Beat This! produced fewer than five complete bars"
        elif not any(row["status"] == "ready" for row in recording_chunks):
            blank_status = "piano_a2s_blank_fill_no_ready_windows"
            blank_error = "No ready Piano-A2S window exists to supply the score skeleton"
        else:
            try:
                _assemble_recording(
                    recording_chunks,
                    blank_fill_xml,
                    blank_fill=True,
                )
            except Exception as error:
                blank_status = "piano_a2s_blank_fill_assembly_failed"
                blank_error = f"{type(error).__name__}: {error}"
        blank_fill_rows.append({
            "recording_id": recording_id,
            "artifact_stem": record["artifact_stem"],
            "piece_id": record["piece_id"],
            "performance_id": record["performance_id"],
            "status": blank_status,
            "error_message": blank_error,
            "prediction_xml": (
                str(blank_fill_xml.resolve()) if blank_status == "ready" else None
            ),
            "prediction_xml_sha256": (
                _sha256(blank_fill_xml) if blank_status == "ready" else None
            ),
            "blank_filled_windows": blank_windows,
            "n_blank_filled_windows": len(blank_windows),
            "n_blank_filled_measures": blank_measures,
            "piano_a2s_repository": str(repository),
            "piano_a2s_commit": repository_commit,
            "piano_a2s_checkpoint": str(checkpoint_dir),
            "piano_a2s_checkpoint_sha256": checkpoint_sha256,
            "piano_a2s_hparams_sha256": hparams_sha256,
        })
    _write_jsonl(args.prediction_manifest, recording_rows)
    _write_jsonl(args.blank_fill_prediction_manifest, blank_fill_rows)
    ready_recordings = sum(row["status"] == "ready" for row in recording_rows)
    blank_ready = sum(row["status"] == "ready" for row in blank_fill_rows)
    print(
        f"Piano-A2S: recording-failure={ready_recordings}/{len(recording_rows)} "
        f"blank-fill={blank_ready}/{len(blank_fill_rows)} ready"
    )


if __name__ == "__main__":
    main()
