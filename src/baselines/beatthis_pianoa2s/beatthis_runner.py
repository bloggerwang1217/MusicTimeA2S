#!/usr/bin/env python3
"""Run one Beat This! checkpoint over an ASAP-102 recording manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def _package_version() -> str:
    for distribution in ("beat-this", "beat_this"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", default="final0")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from beat_this.inference import File2Beats

    records = _read_jsonl(args.inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package_version = _package_version()

    pending_by_checkpoint: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        checkpoint = str(record.get("beatthis_checkpoint", args.checkpoint))
        checkpoint_path = Path(checkpoint)
        checkpoint_sha256 = (
            _sha256(checkpoint_path) if checkpoint_path.is_file() else None
        )
        output = args.output_dir / f"{record['artifact_stem']}.json"
        if output.is_file():
            existing = json.loads(output.read_text())
            if (
                existing.get("status") == "ready"
                and existing.get("performance_audio_sha256")
                == record["performance_audio_sha256"]
                and existing.get("checkpoint") == checkpoint
                and existing.get("checkpoint_sha256") == checkpoint_sha256
            ):
                rows_by_id[record["recording_id"]] = existing
                continue
        pending_by_checkpoint.setdefault(checkpoint, []).append((record, output))

    for checkpoint in sorted(pending_by_checkpoint):
        checkpoint_path = Path(checkpoint)
        checkpoint_sha256 = (
            _sha256(checkpoint_path) if checkpoint_path.is_file() else None
        )
        model = File2Beats(
            checkpoint_path=checkpoint,
            device=args.device,
            dbn=False,
        )
        for record, output in pending_by_checkpoint[checkpoint]:
            row = {
                "recording_id": record["recording_id"],
                "artifact_stem": record["artifact_stem"],
                "piece_id": record["piece_id"],
                "performance_id": record["performance_id"],
                "performance_audio": record["performance_audio"],
                "performance_audio_sha256": record["performance_audio_sha256"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_mapping": record.get("beatthis_checkpoint_mapping"),
                "beat_this_version": package_version,
                "dbn": False,
                "device": args.device,
            }
            try:
                beats, downbeats = model(record["performance_audio"])
                row.update({
                    "status": "ready",
                    "error_message": "",
                    "beats": [float(value) for value in beats],
                    "downbeats": [float(value) for value in downbeats],
                })
            except Exception as error:
                row.update({
                    "status": "beatthis_failed",
                    "error_message": f"{type(error).__name__}: {error}",
                    "beats": [],
                    "downbeats": [],
                })
            _write_json(output, row)
            rows_by_id[record["recording_id"]] = row
        del model

    ordered = [rows_by_id[record["recording_id"]] for record in records]
    _write_jsonl(args.manifest, ordered)
    ready = sum(row["status"] == "ready" for row in ordered)
    print(f"Beat This!: {ready}/{len(ordered)} recordings ready")


if __name__ == "__main__":
    main()
