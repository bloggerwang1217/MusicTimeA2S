#!/usr/bin/env python3
"""Score complete scores against complete references, one score per recording.

The counterpart of eval_window.py, which scores five-measure excerpts. Three
differences from it are deliberate, not oversights:

* No per-unit time budget. A whole score is scored on one alignment, so there is
  no timeout status; eval_window.py caps each window instead.
* The inventory is a reference mapping plus one MusicXML pair list per system,
  not a grounding manifest of rendered excerpts.
* An initial incomplete measure is restored by rewriting the metrical metadata
  after conversion; eval_window.py passes a sub-beat count to the converter.

Scoring is `mv2h.Main -s`: one minimum-cost alignment, ties preferring diagonal,
then a reference deletion, then a prediction insertion. Every score event stays
available to the MV2H component scorers; no window selects the material.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import mido

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.asap.eval_asap_native_gt import _write_reference_xml_and_midi  # noqa: E402
from src.evaluation.mv2h import MV2HEvaluator  # noqa: E402

logger = logging.getLogger(__name__)

METRICS = ["Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H"]
CSV_FIELDS = ["task_id", "arm", "recording_id", "piece_id", "status", "error_message",
              *METRICS, "MV2H4"]
QUARTER_TEMPO = 500000  # 120 BPM, so score time and MV2H's millisecond grid line up


# ---------------------------------------------------------------------------
# metrical metadata
# ---------------------------------------------------------------------------

def initial_pickup(xml: Path, midi: Path) -> Fraction:
    """Length of an initial incomplete measure, in converter sub beats."""
    signatures = []
    for track in mido.MidiFile(midi).tracks:
        time_position = 0
        for message in track:
            time_position += message.time
            if message.type == "time_signature" and time_position == 0:
                signatures.append((message.numerator, message.denominator))
    if not signatures:
        return Fraction(0)
    numerator, denominator = signatures[-1]
    assert len(set(signatures)) == 1
    lengths = []
    for part in ET.parse(xml).getroot().findall("part"):
        measure = part.find("measure")
        if measure is None:
            continue
        divisions = Fraction(measure.findtext("attributes/divisions", "1"))
        position = last = end = Fraction(0)
        for element in measure:
            duration = Fraction(element.findtext("duration", "0")) / divisions
            if element.tag == "backup":
                position -= duration
            elif element.tag == "forward":
                position += duration
                end = max(end, position)
            elif element.tag == "note":
                if element.find("chord") is not None:
                    onset = last
                else:
                    onset = position
                    last = onset
                    position += duration
                end = max(end, onset + duration)
        lengths.append(end)
    length = max(lengths, default=0)
    nominal = Fraction(4 * numerator, denominator)
    subbeat = Fraction(2 if numerator <= 4 or numerator % 3 else 4, denominator)
    return length / subbeat if 0 < length < nominal else Fraction(0)


def apply_pickup(text: str, pickup: Fraction) -> str:
    """Refine the tatum grid so a pickup shorter than one sub beat survives."""
    if not pickup:
        return text
    factor = pickup.denominator
    lines = text.splitlines()
    out: List[str] = []
    tatums: List[int] = []
    for line in lines:
        if line.startswith("Tatum "):
            tatums.append(int(line.split()[1]))
        elif line.startswith("Hierarchy "):
            fields = line.split()
            fields[2] = str(int(fields[2]) * factor)
            anacrusis = (pickup * factor if len(fields) < 5 or fields[4] == "0"
                         else int(fields[3][2:]) * factor)
            assert int(anacrusis) == anacrusis
            fields[3] = "a=" + str(int(anacrusis))
            out.append(" ".join(fields))
        else:
            out.append(line)
    tatums = sorted(set(tatums))
    dense: List[int] = []
    for start, stop in zip(tatums, tatums[1:]):
        dense.extend(int(Fraction(start) + Fraction(stop - start) * i / factor)
                     for i in range(factor))
    dense.extend(tatums[-1:])
    out.extend("Tatum " + str(tatum) for tatum in dense)
    result = "\n".join(out) + "\n"
    untouched = [s for s in lines if not s.startswith(("Tatum ", "Hierarchy "))]
    assert untouched == [s for s in result.splitlines()
                         if not s.startswith(("Tatum ", "Hierarchy "))]
    return result


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def read_rows(path: Path) -> List[dict]:
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    # Concurrent shards write the same inventory and protocol files.
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def mv2h_commit(mv2h_bin: str) -> Optional[str]:
    """Identify the checkout the scoring classes were built from, if it is one."""
    run = subprocess.run(["git", "-C", str(Path(mv2h_bin).resolve().parent), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return run.stdout.strip() if run.returncode == 0 else None


def build_inventory(mapping: Path, arms: List[str]) -> List[dict]:
    """One row per recording per arm, plus the reference arm."""
    inventory = []
    reference_ids = []
    for row in read_rows(mapping):
        source = Path(row["source_xml"])
        reference_ids.append(row["recording_id"])
        inventory.append(dict(arm="reference", recording_id=row["recording_id"],
                              piece_id=row["piece_id"], source=str(source),
                              input_status="ready", source_sha256=sha(source),
                              manifest=str(mapping), manifest_sha256=sha(mapping)))
    expected = set(reference_ids)
    for spec in arms:
        name, pairs = spec.split("=", 1)
        seen = set()
        for row in read_rows(Path(pairs)):
            source = Path(row["prediction_xml"]) if row.get("prediction_xml") else None
            ready = row["status"] == "ready" and source is not None and source.is_file()
            seen.add(row["recording_id"])
            inventory.append(dict(
                arm=name, recording_id=row["recording_id"], piece_id=row["piece_id"],
                source=str(source) if source else None,
                input_status="ready" if ready else row["status"],
                source_sha256=sha(source) if ready else None,
                manifest=str(pairs), manifest_sha256=sha(Path(pairs))))
        if seen != expected:
            raise ValueError(f"{name}: recording set differs from the mapping "
                             f"(missing {len(expected - seen)}, extra {len(seen - expected)})")
    return inventory


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def prepare_one(args: tuple) -> dict:
    """Export one score to MIDI, convert it, and restore its first measure."""
    row, out, mv2h_bin = args
    start = time.monotonic()
    directory = out / row["arm"] / row["recording_id"].replace("#", "__")
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(row, directory=str(directory))

    cached = directory / "export.json"
    if cached.is_file():
        previous = json.loads(cached.read_text())
        if (previous.get("source_sha256") == row["source_sha256"]
                and previous.get("status") == "ready"):
            return previous

    if row["input_status"] != "ready":
        return dict(result, status=row["input_status"], seconds=0)
    try:
        source = Path(row["source"])
        assert sha(source) == row["source_sha256"]
        midi = directory / "score.mid"
        xml = directory / "score.musicxml"
        _write_reference_xml_and_midi(source.read_bytes(), xml, midi)
        rendered = mido.MidiFile(midi)
        for track in rendered.tracks:
            for message in track:
                if message.type == "set_tempo":
                    message.tempo = QUARTER_TEMPO
        rendered.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=QUARTER_TEMPO, time=0))
        rendered.save(midi)
        notes = sum(m.type == "note_on" and m.velocity > 0
                    for t in rendered.tracks for m in t)
        result["notes"] = notes
        if notes == 0:
            return dict(result, status="empty_prediction", seconds=time.monotonic() - start)
        run = subprocess.run(["java", "-cp", str(mv2h_bin),
                              "mv2h.tools.Converter", "-i", str(midi)],
                             capture_output=True, text=True)
        (directory / "converter.stderr.txt").write_text(run.stderr)
        if run.returncode:
            raise RuntimeError("MV2H Converter: " + run.stderr[-2000:])
        raw = directory / "score.mv2h.txt"
        raw.write_text(run.stdout)
        pickup = initial_pickup(xml, midi)
        if pickup:
            converted = directory / "score.pickup.mv2h.txt"
            converted.write_text(apply_pickup(raw.read_text(), pickup))
        else:
            converted = raw
        result.update(status="ready", converted=str(converted), midi_sha256=sha(midi),
                      pickup_subbeats=str(pickup), converted_sha256=sha(converted),
                      raw_converted_sha256=sha(raw))
    except Exception as error:
        result.update(status="conversion_failed", error=f"{type(error).__name__}: {error}")
    result["seconds"] = time.monotonic() - start
    write_json(directory / "export.json", result)
    return result


def score_one(args: tuple) -> dict:
    """Score one prepared prediction against its prepared reference."""
    prediction, reference, settings = args
    row = dict(prediction)
    start = time.monotonic()
    directory = Path(prediction["directory"])
    if reference["status"] != "ready":
        return dict(row, status="reference_failed",
                    error=reference.get("error", reference["status"]))
    if prediction["status"] != "ready":
        row["metrics"] = {k: 0.0 for k in METRICS + ["MV2H4"]}
        write_json(directory / "result.json", row)
        return row

    evaluator = MV2HEvaluator(settings["mv2h_bin"], timeout=settings["timeout"],
                              single_path=True)
    try:
        scored = evaluator.evaluate_converted(
            reference["converted"], prediction["converted"],
            java_heap=settings["java_heap"], log_dir=directory)
        if scored is None:
            raise RuntimeError((directory / "scorer.stderr.txt").read_text()[-2000:])
        metrics = {k: v for k, v in scored.to_dict().items() if k in METRICS}
        assert set(metrics) == set(METRICS)
        assert all(math.isfinite(v) and 0 <= v <= 1.000000001 for v in metrics.values())
        # sum() compensates; MV2HResult.mv2h_custom adds pairwise, and the two
        # disagree in the last place.
        metrics["MV2H4"] = sum(metrics[k] for k in
                               ["Multi-pitch", "Voice", "Value", "Harmony"]) / 4
        row.update(status="success", metrics=metrics)
    except Exception as error:
        row.update(status="scoring_failed", error=str(error))
    row["score_seconds"] = time.monotonic() - start
    write_json(directory / "result.json", row)
    return row


def result_row(row: dict) -> Dict[str, Any]:
    metrics = row.get("metrics", {})
    return dict(task_id=f"{row['arm']}/{row['recording_id']}", arm=row["arm"],
                recording_id=row["recording_id"], piece_id=row["piece_id"],
                status=row["status"], error_message=row.get("error", ""),
                **{k: metrics.get(k, "") for k in METRICS + ["MV2H4"]})


def write_results(rows: List[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["arm"], r["recording_id"])):
            writer.writerow(result_row(row))
    temporary.replace(output_csv)


def merge_shards(output_dir: Path, inventory: List[dict], output_csv: Path) -> List[dict]:
    rows: Dict[str, dict] = {}
    shards = sorted(output_dir.glob("results_shard_*.json"))
    if not shards:
        raise FileNotFoundError(f"No shard results in {output_dir}")
    for shard in shards:
        for row in json.loads(shard.read_text()):
            key = f"{row['arm']}/{row['recording_id']}"
            if key in rows:
                raise ValueError(f"Duplicate unit across shards: {key}")
            rows[key] = row
    expected = [f"{r['arm']}/{r['recording_id']}" for r in inventory if r["arm"] != "reference"]
    missing = [key for key in expected if key not in rows]
    extra = sorted(set(rows) - set(expected))
    if missing or extra:
        raise ValueError(f"Shard coverage mismatch: missing={len(missing)} extra={len(extra)}")
    merged = [rows[key] for key in expected]
    write_json(output_dir / "results.json", merged)
    write_results(merged, output_csv)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("render", "eval", "all", "merge"), default="all")
    parser.add_argument("--mapping", type=Path, required=True,
                        help="Reference mapping JSONL: recording_id, piece_id, source_xml")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PAIRS_JSONL")
    parser.add_argument("--mv2h-bin", default="external/MV2H/bin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--chunk-start", type=int)
    parser.add_argument("--chunk-end", type=int)
    parser.add_argument("--workers", "-j", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-recording budget in seconds; the protocol runs without one")
    parser.add_argument("--java-heap", default="4g")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or out / "eval_whole_score.csv"

    inventory = build_inventory(args.mapping, args.arm)
    write_json(out / "inventory.json", inventory)
    write_json(out / "protocol.json", dict(
        name="whole-score deterministic-path DTW-MV2H",
        scorer="mv2h.Main -s", canonical_equivalent=False,
        mv2h_bin=str(Path(args.mv2h_bin).resolve()),
        mv2h_commit=mv2h_commit(args.mv2h_bin),
        tempo_quarter_bpm=120, gap_penalty="mv2h.Main -p default",
        tie_priority=["diagonal", "delete_reference", "insert_prediction"],
        onset_tolerance_ms=0, value_tolerance_ms=20, grouping_tolerance_ms=20,
        segmentation="none", score_time_rescaling="none", timeout=args.timeout,
        unit="recording", recordings=len({r["recording_id"] for r in inventory}),
        prediction_failure="status preserved; metrics zero-filled and never counted as a score",
        reference_failure="unresolved; never silently excluded or scored as model failure",
        pickup="preserve actual first-measure length independently in each score",
        source_sha256={str(p): sha(p) for p in [
            Path(__file__), ROOT / "src/evaluation/mv2h.py",
            ROOT / "src/evaluation/asap/eval_asap_native_gt.py"]}))

    if args.phase == "merge":
        merge_shards(out, inventory, output_csv)
        return

    if args.phase in ("render", "all"):
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            prepared = list(pool.map(prepare_one, [(r, out, args.mv2h_bin) for r in inventory]))
        write_json(out / "exports.json", prepared)
    else:
        prepared = json.loads((out / "exports.json").read_text())
        current = {(r["arm"], r["recording_id"]): r for r in inventory}
        for row in prepared:
            if row["source_sha256"] != current[(row["arm"], row["recording_id"])]["source_sha256"]:
                raise ValueError(f"Input changed since render: {row['recording_id']}")
    if args.phase == "render":
        return

    reference = {r["recording_id"]: r for r in prepared if r["arm"] == "reference"}
    pending = [r for r in prepared if r["arm"] != "reference"]
    pending.sort(key=lambda r: (r["arm"], r["recording_id"]))
    sharded = args.chunk_start is not None or args.chunk_end is not None
    start, stop = args.chunk_start or 0, args.chunk_end if args.chunk_end is not None else len(pending)
    settings = dict(mv2h_bin=args.mv2h_bin, timeout=args.timeout, java_heap=args.java_heap)
    jobs = [(r, reference[r["recording_id"]], settings) for r in pending[start:stop]]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(score_one, jobs))

    if sharded:
        write_json(out / f"results_shard_{start:06d}_{stop:06d}.json", results)
        return
    write_json(out / "results.json", sorted(results, key=lambda r: (r["arm"], r["recording_id"])))
    write_results(results, output_csv)
    for arm in sorted({r["arm"] for r in results}):
        rows = [r for r in results if r["arm"] == arm]
        logger.info("%s %s", arm, {s: sum(r["status"] == s for r in rows)
                                   for s in sorted({r["status"] for r in rows})})


if __name__ == "__main__":
    main()
