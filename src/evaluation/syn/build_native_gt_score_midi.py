#!/usr/bin/env python3
"""Cut the ground-truth score MIDI of every Syn evaluation window.

One MIDI per five-bar window, sliced out of the rendering's own ground-truth
kern, so MV2H has something to score each window's prediction against.

Writes the same artifact set the ASAP-102 side provides, so the grounded slicer,
the MV2H scorers and the OMR-NED tools run on Syn unchanged:

- ``grounding.jsonl``   one row per five-bar evaluation window, on the grid the
  evaluation uses: stride five from the first whole bar, so a pickup is never
  part of a window
- ``mapping.jsonl``     one row per rendering with its complete bar count
- ``window_midi/``      the ground-truth score MIDI of every window, cut from
  ``kern_gt/``
- ``manifest.json``     the split manifest with ``piece_id`` / ``performance_id``
- ``recordings.txt``    name-header list of every rendering id

Recording ids follow the ASAP-102 spelling ``piece#performance``; here the piece
is the kern stem (one movement or one piece) and the performance is the
rendering id.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from src.evaluation.asap import sha256_file
from src.evaluation.prepare_preseg import first_full_measure
from src.evaluation.syn import SynDataset
from src.datasets.syn.syn_manifest import select_test_renders

N_BARS = 5


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ))
    temporary.replace(path)


def canonical_xml(xml_dir: Path, kern_stem: str) -> Path:
    # MuseSyn scores are exported without the corpus prefix, and a few keep
    # their .mscz origin in the file name.
    for stem in (kern_stem, kern_stem.removeprefix("musesyn_")):
        for suffix in (".xml", ".mscz.xml"):
            path = xml_dir / f"{stem}{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No canonical MusicXML for {kern_stem} in {xml_dir}")


_WORKER_DATASET: SynDataset | None = None


def _init_worker(manifest_dir: str) -> None:
    global _WORKER_DATASET
    root = Path(manifest_dir)
    _WORKER_DATASET = SynDataset(
        manifest_path=str(root / "test_manifest.json"),
        metadata_path=str(root / "augmentation_metadata.json"),
        kern_gt_dir=str(root / "kern_gt"),
        manifest_dir=str(root),
    )


def _render_window(job: tuple[str, str, int, int, str]) -> tuple[str, str, str]:
    chunk_id, kern_gt_path, line_start, line_end, output = job
    temporary = _WORKER_DATASET.get_gt_kern_slice_midi(kern_gt_path, line_start, line_end)
    if temporary is None:
        return chunk_id, "reference_conversion_failed", ""
    shutil.move(temporary, output)
    return chunk_id, "ready", sha256_file(Path(output))


def build(manifest_dir: Path, out_dir: Path, workers: int, limit: int | None = None,
          stride: int = N_BARS) -> dict[str, Any]:
    started = time.time()
    manifest_path = manifest_dir / "test_manifest.json"
    dataset = SynDataset(
        manifest_path=str(manifest_path),
        metadata_path=str(manifest_dir / "augmentation_metadata.json"),
        kern_gt_dir=str(manifest_dir / "kern_gt"),
        manifest_dir=str(manifest_dir),
    )
    entries = select_test_renders(json.loads(manifest_path.read_text()))
    dataset.manifest = {e["id"]: e for e in entries}
    if limit is not None:
        entries = entries[:limit]
        dataset.manifest = {e["id"]: e for e in entries}
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = out_dir / "window_midi"
    midi_dir.mkdir(exist_ok=True)

    sidecar: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    recording_of: dict[str, str] = {}
    for entry in entries:
        rendering_id = entry["id"]
        kern_stem = Path(entry["kern_gt_path"]).stem
        recording_id = f"{kern_stem}#{rendering_id}"
        recording_of[rendering_id] = recording_id
        sidecar.append({**entry, "piece_id": kern_stem, "performance_id": rendering_id})
        n_measures = len(entry["audio_measures"])
        source_xml = canonical_xml(manifest_dir / "xml", kern_stem)
        mapping.append({
            "recording_id": recording_id,
            "artifact_stem": rendering_id,
            "piece_id": kern_stem,
            "performance_id": rendering_id,
            "source_measure_count": n_measures,
            "source_measure_groups": [[index] for index in range(n_measures)],
            "source_xml": str(source_xml.resolve()),
            "source_xml_sha256": sha256_file(source_xml),
            "reference_kern": str((manifest_dir / entry["kern_gt_path"]).resolve()),
        })

    grounding: list[dict[str, Any]] = []
    jobs: list[tuple[str, str, int, int, str]] = []
    # The windows an anacrusis piece is evaluated on start one bar in, so its
    # reference has to be cut there too.
    first_bar = {entry["id"]: first_full_measure(entry) for entry in entries}
    for chunk in dataset.iter_5bar_windows(n_bars=N_BARS, stride=stride, first_bar=first_bar):
        recording_id = recording_of[chunk.perf_id]
        chunk_id = f"{recording_id}.{chunk.chunk_index}"
        relative = f"window_midi/{chunk.perf_id}.{chunk.chunk_index}.mid"
        grounding.append({
            "chunk_id": chunk_id,
            "recording_id": recording_id,
            "piece_id": recording_id.rsplit("#", 1)[0],
            "performance_id": chunk.perf_id,
            "position_index": chunk.chunk_index,
            "start_bar_index": first_bar[chunk.perf_id] + chunk.chunk_index * stride,
            "end_bar_index_exclusive": first_bar[chunk.perf_id] + chunk.chunk_index * stride + N_BARS,
            "reference_midi": relative,
            "gt_midi": relative,
            "reference_status": "pending",
            "gt_midi_status": "pending",
            "reference_midi_sha256": None,
        })
        jobs.append((
            chunk_id, chunk.kern_gt_path, chunk.kern_line_start, chunk.kern_line_end,
            str(out_dir / relative),
        ))

    by_chunk = {row["chunk_id"]: row for row in grounding}
    print(f"Rendering {len(jobs)} reference windows with {workers} workers", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(manifest_dir),)) as pool:
        for done, (chunk_id, status, digest) in enumerate(pool.map(_render_window, jobs, chunksize=16), 1):
            row = by_chunk[chunk_id]
            row["reference_status"] = status
            row["gt_midi_status"] = status
            if status == "ready":
                row["reference_midi_sha256"] = digest
                # The frozen mapping is the bar-count authority for proportional pairing.
                row["bar_count_authority"] = "frozen_mapping"
            if done % 500 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}]", file=sys.stderr)

    _write_jsonl(out_dir / "grounding.jsonl", grounding)
    _write_jsonl(out_dir / "mapping.jsonl", mapping)
    (out_dir / "manifest.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    (out_dir / "recordings.txt").write_text("name\n" + "".join(f"{e['id']}\n" for e in entries))
    counts = Counter(row["reference_status"] for row in grounding)
    summary = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "recordings": len(entries),
        "pieces": len({row["piece_id"] for row in mapping}),
        "windows": len(grounding),
        "window_status": dict(sorted(counts.items())),
        "n_bars": N_BARS,
        "stride": stride,
        "recordings_starting_after_a_pickup": sorted(k for k, v in first_bar.items() if v),
        "elapsed_sec": round(time.time() - started, 1),
    }
    (out_dir / "grounding-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/experiments/syn"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Only the first N manifest renderings (smoke)")
    parser.add_argument("--stride", type=int, default=N_BARS, help="Window stride in bars")
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.out_dir}")
    print(json.dumps(build(args.manifest_dir, args.out_dir, args.workers, args.limit,
                           args.stride), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
