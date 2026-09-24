"""Select non-overlapping five-bar evaluation windows from native references."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from src.evaluation.asap import sha256_file


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def time_key(recording_id: str, start: float, end: float) -> tuple[str, float, float]:
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"Invalid audio interval for {recording_id}: [{start}, {end})")
    # Both dataset manifest builders store annotated boundaries at four decimals.
    return recording_id, round(start, 4), round(end, 4)


def first_full_measure(item: dict) -> int:
    """Index of the first annotated measure that is a whole bar.

    A pickup carries fewer beats than the bar after it. Starting the grid on it
    would make the first window four notated bars plus a fragment, and would
    shift every later window of that recording by one bar against any grid that
    counts from the first downbeat.
    """
    bars = item["audio_measures"]
    if len(bars) < 2:
        return 0

    def beats_in(bar: dict) -> int:
        # By time, not by the beat's measure label: an anacrusis that carries no
        # beat of its own still occupies the first entry of audio_measures.
        return sum(1 for beat in item["audio_beats"]
                   if bar["start_sec"] <= beat["sec"] < bar["end_sec"])

    return 1 if beats_in(bars[0]) < beats_in(bars[1]) else 0


def validate_windows(rows: list[dict]) -> None:
    seen, intervals = set(), set()
    by_recording: dict[str, list[tuple[float, float]]] = {}
    bars_by_recording: dict[str, list[int]] = {}
    if not rows:
        raise ValueError("Empty evaluation window inventory")
    for row in rows:
        start, end = row["start_bar_index"], row["end_bar_index_exclusive"]
        if start < 0 or end != start + 5:
            raise ValueError(f"Expected five bars at stride five: {row['chunk_id']}")
        bars_by_recording.setdefault(row["recording_id"], []).append(start)
        key = time_key(row["recording_id"], row["start_sec"], row["end_sec"])
        if row["chunk_id"] in seen or key in intervals:
            raise ValueError(f"Duplicate evaluation window: {row['chunk_id']}")
        seen.add(row["chunk_id"])
        intervals.add(key)
        by_recording.setdefault(key[0], []).append(key[1:])
    for recording, spans in by_recording.items():
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError(f"Overlapping evaluation audio windows: {recording}")
    for recording, starts in bars_by_recording.items():
        starts.sort()
        if starts[0] not in (0, 1) or any(b - a != 5 for a, b in zip(starts, starts[1:])):
            raise ValueError(f"Windows are not five bars apart: {recording}")


def prepare(manifest: Path, gt_score_midi_root: Path, output: Path,
            recording_list: Path | None) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    entries = json.loads(manifest.read_text())
    if recording_list is not None:
        names = recording_list.read_text().splitlines()
        if not names or names[0] != "name":
            raise ValueError("Recording list must begin with a name header")
        keep = set(names[1:]) - {""}
        missing = keep - {item["id"] for item in entries}
        if missing:
            raise ValueError(f"Unknown recordings: {sorted(missing)}")
        entries = [item for item in entries if item["id"] in keep]
    native = {}
    source_grounding = gt_score_midi_root / "grounding.jsonl"
    for row in read_rows(source_grounding):
        key = row["recording_id"], row["start_bar_index"], row["end_bar_index_exclusive"]
        if key in native:
            raise ValueError(f"Duplicate native reference range: {key}")
        native[key] = row

    windows, selected, without_windows, after_pickup = [], [], [], []
    for item in entries:
        piece = item.get("piece_id") or Path(item["kern_gt_path"]).stem
        perf = item.get("performance_id") or item["id"]
        recording = f"{piece}#{perf}"
        bars = item["audio_measures"]
        selected.append({**item, "piece_id": piece, "performance_id": perf})
        offset = first_full_measure(item)
        if offset:
            after_pickup.append(recording)
        if len(bars) - offset < 5:
            without_windows.append(recording)
        for start in range(offset, len(bars) - 4, 5):
            end = start + 5
            reference = native.get((recording, start, end), {})
            row = {
                **reference,
                "chunk_id": f"{recording}.{start}",
                "recording_id": recording, "piece_id": piece, "performance_id": perf,
                "position_index": start, "start_bar_index": start,
                "end_bar_index_exclusive": end,
                "start_sec": bars[start]["start_sec"], "end_sec": bars[end - 1]["end_sec"],
                "reference_status": reference.get("reference_status", "reference_window_missing"),
                "reference_midi": reference.get("reference_midi"),
                "reference_midi_sha256": reference.get("reference_midi_sha256"),
            }
            # The evaluation identity is the actual start, not a producer's chunk counter.
            for key in ("gt_midi", "gt_midi_status", "gt_midi_sha256"):
                row.pop(key, None)
            if row["reference_status"] == "ready":
                path = gt_score_midi_root / row["reference_midi"]
                if not path.is_file() or sha256_file(path) != row["reference_midi_sha256"]:
                    raise ValueError(f"Native reference artifact mismatch: {path}")
            windows.append(row)
    windows.sort(key=lambda row: (row["recording_id"], row["start_bar_index"]))
    validate_windows(windows)
    recordings_with_windows = {row["recording_id"] for row in windows}
    selected = [item for item in selected
                if f"{item['piece_id']}#{item['performance_id']}" in recordings_with_windows]
    summary = {
        "n_bars": 5, "stride": 5, "windows": len(windows),
        "recordings": len(entries), "works": len({row["piece_id"] for row in windows}),
        "recordings_without_complete_windows": without_windows,
        "recordings_starting_after_a_pickup": sorted(after_pickup),
        "reference_status": dict(Counter(row["reference_status"] for row in windows)),
        "unavailable_reference_windows": [row["chunk_id"] for row in windows
                                          if row["reference_status"] != "ready"],
        "manifest": str(manifest.resolve()), "manifest_sha256": sha256_file(manifest),
        "gt_score_midi_root": str(gt_score_midi_root.resolve()),
        "source_grounding_sha256": sha256_file(source_grounding),
        "recording_list": str(recording_list.resolve()) if recording_list else None,
        "recording_list_sha256": sha256_file(recording_list) if recording_list else None,
    }
    output.mkdir(parents=True)
    (output / "grounding.jsonl").write_text("".join(json.dumps(row) + "\n" for row in windows))
    (output / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-score-midi-root", type=Path, required=True)
    parser.add_argument("--recording-list", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.gt_score_midi_root, args.output_dir,
                             args.recording_list), indent=2))


if __name__ == "__main__":
    main()
