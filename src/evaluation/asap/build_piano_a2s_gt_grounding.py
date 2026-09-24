#!/usr/bin/env python3
"""Build and compare reproducible Piano-A2S GT-MIDI grounding manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _repository_provenance(repo: Path) -> dict[str, Any]:
    status = _git(repo, "status", "--short")
    provenance: dict[str, Any] = {
        "path": str(repo.resolve()),
        "head": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "status": status.splitlines() if status else [],
    }
    for name in ("verovio", "humextra"):
        nested = repo / name
        if nested.is_dir():
            provenance[f"{name}_head"] = _git(nested, "rev-parse", "HEAD")
    return provenance


def _load_our_recordings(manifest_path: Path) -> dict[str, dict[str, Any]]:
    with manifest_path.open() as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError("Our manifest must contain a JSON list")

    recordings: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = f"{entry['piece_id']}#{entry['performance_id']}"
        if key in recordings:
            raise ValueError(f"Duplicate recording ID in our manifest: {key}")
        recordings[key] = entry
    return recordings


def _index_gt_midis(results_dir: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    pattern = "tasks/*/midi/target/*_target.mid"
    for path in sorted(results_dir.glob(pattern)):
        chunk_id = path.name.removesuffix("_target.mid")
        previous = indexed.get(chunk_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate GT MIDI for {chunk_id}: {previous} and {path}"
            )
        indexed[chunk_id] = path
    return indexed


def _parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    try:
        recording_id, position = chunk_id.rsplit(".", 1)
        position_index = int(position)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid Piano-A2S chunk ID: {chunk_id}") from error
    if position_index < 0:
        raise ValueError(f"Negative Piano-A2S position index: {chunk_id}")
    return recording_id, position_index


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _sequence_sha256(entries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            (
                f"{entry['recording_id']}\t{entry['position_index']}\t"
                f"{entry['chunk_id']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_dir = args.target_dir.resolve()
    results_dir = args.results_dir.resolve()
    recordings = _load_our_recordings(args.our_manifest)
    gt_midis = _index_gt_midis(results_dir)
    upbeat_recordings: set[str] = set()
    if args.upbeat_recordings is not None:
        upbeat_recordings = {
            line.strip()
            for line in args.upbeat_recordings.read_text().splitlines()
            if line.strip()
        }
        unknown = sorted(upbeat_recordings - set(recordings))
        if unknown:
            raise ValueError(
                f"{len(unknown)} upbeat recordings are not in our manifest; "
                f"first: {unknown[:3]}"
            )

    entries: list[dict[str, Any]] = []
    by_recording: dict[str, list[int]] = defaultdict(list)
    unmatched: list[str] = []
    out_of_range: list[str] = []
    missing_gt_excluded = 0

    for target_path in target_dir.glob("*.pkl"):
        chunk_id = target_path.stem
        recording_id, position_index = _parse_chunk_id(chunk_id)
        recording = recordings.get(recording_id)
        if recording is None:
            unmatched.append(chunk_id)
            continue

        # The chunk builder skips a pickup when it enumerates downbeats, so on
        # a pickup recording chunk k holds measures k+1..k+5 of the manifest.
        start_bar_index = position_index + (recording_id in upbeat_recordings)
        measure_count = len(recording.get("audio_measures", []))
        if start_bar_index + 5 > measure_count:
            out_of_range.append(chunk_id)
            continue

        gt_midi = gt_midis.get(chunk_id)
        if args.require_gt_midi and gt_midi is None:
            missing_gt_excluded += 1
            continue
        piece_id, performance_id = recording_id.rsplit("#", 1)
        entries.append(
            {
                "chunk_id": chunk_id,
                "recording_id": recording_id,
                "piece_id": piece_id,
                "performance_id": performance_id,
                "position_index": position_index,
                "start_bar_index": start_bar_index,
                "end_bar_index_exclusive": start_bar_index + 5,
                "target_pkl": _relative(target_path, target_dir),
                "target_pkl_sha256": _sha256(target_path),
                "gt_midi_status": "ready" if gt_midi is not None else "missing",
                "gt_midi": (
                    _relative(gt_midi, results_dir) if gt_midi is not None else None
                ),
                "gt_midi_sha256": _sha256(gt_midi) if gt_midi is not None else None,
            }
        )
        by_recording[recording_id].append(position_index)

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} Piano-A2S chunks do not match our manifest; "
            f"first: {unmatched[:3]}"
        )
    if out_of_range:
        raise ValueError(
            f"{len(out_of_range)} Piano-A2S chunks exceed the measure sequence of our manifest; "
            f"first: {out_of_range[:3]}"
        )

    entries.sort(key=lambda entry: (entry["recording_id"], entry["position_index"]))
    ready = sum(entry["gt_midi_status"] == "ready" for entry in entries)
    summary = {
        "target_chunks": len(entries),
        "recordings": len(by_recording),
        "pieces": len({entry["piece_id"] for entry in entries}),
        "gt_midi_ready": ready,
        "gt_midi_missing": len(entries) - ready,
        "gt_midi_missing_excluded": missing_gt_excluded,
        "position_sequence_sha256": _sequence_sha256(entries),
        "inputs": {
            "our_manifest": str(args.our_manifest.resolve()),
            "target_dir": str(target_dir),
            "results_dir": str(results_dir),
            **(
                {
                    "upbeat_recordings": str(args.upbeat_recordings.resolve()),
                    "upbeat_recordings_sha256": _sha256(args.upbeat_recordings),
                }
                if args.upbeat_recordings is not None
                else {}
            ),
        },
        "piano_a2s_repository": _repository_provenance(
            args.piano_a2s_repo.resolve()
        ),
    }
    return entries, summary


def _write_manifest(
    entries: list[dict[str, Any]], summary: dict[str, Any], output: Path
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(_json_line(entry) for entry in entries)
    output.write_text(payload)
    summary["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return summary_path


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            chunk_id = entry["chunk_id"]
            if chunk_id in entries:
                raise ValueError(f"Duplicate {chunk_id} in {path}:{line_number}")
            entries[chunk_id] = entry
    return entries


def compare_manifests(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = _load_jsonl(reference_path)
    candidate = _load_jsonl(candidate_path)
    reference_ids = set(reference)
    candidate_ids = set(candidate)
    fields = (
        "recording_id",
        "position_index",
        "start_bar_index",
        "end_bar_index_exclusive",
        "target_pkl_sha256",
        "gt_midi_status",
        "gt_midi_sha256",
    )

    mismatches: list[dict[str, Any]] = []
    for chunk_id in sorted(reference_ids & candidate_ids):
        different = [
            field
            for field in fields
            if reference[chunk_id].get(field) != candidate[chunk_id].get(field)
        ]
        if different:
            mismatches.append({"chunk_id": chunk_id, "fields": different})

    return {
        "reference": str(reference_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "reference_chunks": len(reference),
        "candidate_chunks": len(candidate),
        "missing_chunks": sorted(reference_ids - candidate_ids),
        "extra_chunks": sorted(candidate_ids - reference_ids),
        "content_mismatches": mismatches,
        "identical": (
            reference_ids == candidate_ids and not mismatches
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Piano-A2S target/GT-MIDI grounding manifest"
    )
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--our-manifest", type=Path, required=True)
    parser.add_argument("--piano-a2s-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-gt-midi",
        action="store_true",
        help="Exclude reference-undefined chunks before freezing the manifest",
    )
    parser.add_argument(
        "--upbeat-recordings",
        type=Path,
        help="Recording ids (one per line) whose chunk index is offset by a "
        "pickup measure; without it every chunk starts at its index",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Compare the newly built manifest with a prior build",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="Write the comparison report as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path, label in (
        (args.target_dir, "target-dir"),
        (args.results_dir, "results-dir"),
        (args.our_manifest, "our-manifest"),
        (args.piano_a2s_repo, "piano-a2s-repo"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    entries, summary = build_manifest(args)
    summary_path = _write_manifest(entries, summary, args.output)
    print(
        f"Built {args.output}: {summary['target_chunks']} chunks, "
        f"{summary['recordings']} recordings, "
        f"GT MIDI {summary['gt_midi_ready']} ready / "
        f"{summary['gt_midi_missing']} missing"
    )
    print(f"Summary: {summary_path}")

    if args.reference is None:
        return 0

    comparison = compare_manifests(args.reference, args.output)
    comparison_output = args.comparison_output or args.output.with_suffix(
        ".comparison.json"
    )
    comparison_output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(f"Comparison: {comparison_output}")
    print("IDENTICAL" if comparison["identical"] else "DIFFERENT")
    return 0 if comparison["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
