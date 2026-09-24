#!/usr/bin/env python3
"""Present the Syn test renderings in the folder layout the Piano-A2S ASAP
chunk builder reads, so its own pipeline cuts, filters and converts the
five-bar windows.

Output under --out:
    asap-dataset/<stem>/xml_score.musicxml     symlink to the canonical XML
    asap-dataset/<stem>/<rendering>.wav        symlink to the rendered audio
    asap-dataset/<stem>/<rendering>_annotations.txt
    test_pieces.txt        piece list (`name` header), one row per stem
    test_perfs.tsv         stem, rendering, absolute wav path
    upbeat_recordings.txt  `<stem>#<rendering>` ids whose score opens with a pickup
    layout_summary.json    counts, input hashes and per-recording accounting

The annotation rows follow the ASAP beat format (`time<TAB>time<TAB>label`):
a downbeat is `db,<meter>,<sharps>` on the manifest's measure start, other
beats are `b`. A pickup measure is written as a leading `b` row so the
builder counts it the way it counts an ASAP pickup; its chunk index then
starts one measure after the manifest's measure index, which
upbeat_recordings.txt records for the grounding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import music21 as m21

from src.datasets.syn.syn_manifest import select_test_renders


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_measure_labels(xml_path: str) -> list[tuple[str, int]]:
    """(meter, sharps) in force at each measure of the first part."""
    score = m21.converter.parse(xml_path)
    labels: list[tuple[str, int]] = []
    meter, sharps = None, None
    for measure in score.parts[0].getElementsByClass("Measure"):
        for signature in measure.getElementsByClass("TimeSignature"):
            meter = signature.ratioString
        for key in measure.getElementsByClass("KeySignature"):
            sharps = int(key.sharps)
        if meter is None:
            raise ValueError(f"{xml_path}: measure {measure.number} has no meter")
        # A score without a key signature is in C: zero sharps.
        labels.append((meter, 0 if sharps is None else sharps))
    return labels


def annotation_rows(item: dict, labels: list[tuple[str, int]]) -> tuple[list[str], bool]:
    measures = item["audio_measures"]
    if len(measures) != len(labels):
        raise ValueError(
            f"{item['id']}: {len(measures)} audio measures, {len(labels)} score measures"
        )
    pickup = measures[0].get("measure") == 0
    events: list[tuple[float, int, str]] = []
    for index, (measure, (meter, sharps)) in enumerate(zip(measures, labels)):
        time = float(measure["start_sec"])
        # Every row restates meter and key; the reader keeps the last value
        # it saw, so a full label on every downbeat is equivalent to ASAP's
        # change-only labels.
        kind = "b" if index == 0 and pickup else "db"
        events.append((time, 0, f"{kind},{meter},{sharps}"))
    first_downbeat = next(t for t, _, text in events if text.startswith("db"))
    downbeat_times = {round(t, 6) for t, _, text in events if text.startswith("db")}
    for beat in item.get("audio_beats", []):
        time = float(beat["sec"])
        if beat.get("is_downbeat") or time < first_downbeat:
            continue
        if round(time, 6) in downbeat_times:
            continue
        events.append((time, 1, "b"))
    events.sort(key=lambda event: (event[0], event[1]))
    return [f"{time}\t{time}\t{text}" for time, _, text in events], pickup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Syn test manifest carrying piece_id and performance_id")
    parser.add_argument("--mapping", type=Path, required=True,
                        help="mapping.jsonl naming each rendering's canonical XML")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.out.resolve() / "asap-dataset"
    if root.exists():
        raise SystemExit(f"refusing to overwrite {root}")

    items = select_test_renders(json.loads(args.manifest.read_text()))
    mapping = {}
    for line in args.mapping.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            mapping[row["performance_id"]] = row
    missing = [item["id"] for item in items if item["id"] not in mapping]
    if missing:
        raise SystemExit(f"{len(missing)} renderings absent from mapping: {missing[:3]}")

    xml_by_stem: dict[str, str] = {}
    for item in items:
        xml = mapping[item["id"]]["source_xml"]
        if xml_by_stem.setdefault(item["piece_id"], xml) != xml:
            raise SystemExit(f"{item['piece_id']}: renderings disagree on the source XML")
    stems = sorted(xml_by_stem)
    with ProcessPoolExecutor(args.workers) as pool:
        labels_by_stem = dict(zip(stems, pool.map(score_measure_labels,
                                                  [xml_by_stem[s] for s in stems])))

    root.mkdir(parents=True)
    tsv_rows, upbeats, recordings = [], [], []
    for item in sorted(items, key=lambda entry: (entry["piece_id"], entry["id"])):
        stem, rendering = item["piece_id"], item["id"]
        if item["performance_id"] != rendering:
            raise SystemExit(f"{rendering}: performance_id differs from the manifest id")
        folder = root / stem
        folder.mkdir(exist_ok=True)
        score_link = folder / "xml_score.musicxml"
        if not score_link.exists():
            os.symlink(Path(xml_by_stem[stem]).resolve(), score_link)
        audio = (args.audio_dir / f"{rendering}.wav").resolve()
        if not audio.is_file():
            raise SystemExit(f"missing audio {audio}")
        os.symlink(audio, folder / f"{rendering}.wav")
        rows, pickup = annotation_rows(item, labels_by_stem[stem])
        (folder / f"{rendering}_annotations.txt").write_text("\n".join(rows) + "\n")
        tsv_rows.append(f"{stem}\t{rendering}\t{folder / (rendering + '.wav')}")
        if pickup:
            upbeats.append(f"{stem}#{rendering}")
        recordings.append({
            "recording_id": f"{stem}#{rendering}",
            "measures": len(item["audio_measures"]),
            "pickup": pickup,
            "meters": sorted({meter for meter, _ in labels_by_stem[stem]}),
        })

    out = args.out.resolve()
    (out / "manifest.json").write_text(json.dumps(items, indent=2) + "\n")
    (out / "test_pieces.txt").write_text("name\n" + "".join(f"{s}\n" for s in stems))
    (out / "test_perfs.tsv").write_text("".join(row + "\n" for row in tsv_rows))
    (out / "upbeat_recordings.txt").write_text("".join(row + "\n" for row in upbeats))
    summary = {
        "recordings": len(tsv_rows),
        "pieces": len(stems),
        "upbeat_recordings": len(upbeats),
        "inputs": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "mapping": str(args.mapping.resolve()),
            "mapping_sha256": _sha256(args.mapping),
            "audio_dir": str(args.audio_dir.resolve()),
        },
        "per_recording": recordings,
    }
    (out / "layout_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("recordings", "pieces",
                                                     "upbeat_recordings")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
