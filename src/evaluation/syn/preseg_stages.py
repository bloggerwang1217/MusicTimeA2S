"""Completion checks and the decode manifest for the pre-segmented Syn stages.

    python -m src.evaluation.syn.preseg_stages pa2s-mv2h RESULTS_DIR N_TASKS
    python -m src.evaluation.syn.preseg_stages score EVAL_CSV GROUNDING
    python -m src.evaluation.syn.preseg_stages decode-manifest MANIFEST GROUNDING OUT
    python -m src.evaluation.syn.preseg_stages select-inputs TEST_MANIFEST PRESEG OUT KIND
    python -m src.evaluation.syn.preseg_stages check-manifest TEST_MANIFEST MANIFEST
    python -m src.evaluation.syn.preseg_stages check-chunks TEST_MANIFEST CHUNK_DIR
    python -m src.evaluation.syn.preseg_stages decode-inventory KERN_DIR GROUNDING

Each check prints a JSON record on success and exits non-zero otherwise, so a
stage is marked done only from the evidence its own tasks left behind.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path


def _test_manifest(path: str) -> dict[str, dict]:
    from src.datasets.syn.syn_manifest import select_test_renders

    items = json.loads(Path(path).read_text())
    selected = select_test_renders(items)
    if not items or len(selected) != len(items):
        raise ValueError(f"{path}: test manifest is empty or contains non-test soundfonts")
    indexed = {item["id"]: item for item in items}
    if len(indexed) != len(items):
        raise ValueError(f"{path}: duplicate test rendering IDs")
    return indexed


def _check_render(item: dict, official: dict) -> None:
    render_id = official["id"]
    if (item["performance_id"] != render_id
            or item["piece_id"] != Path(official["kern_gt_path"]).stem):
        raise ValueError(f"{render_id}: recording identity differs from the test manifest")
    for key in ("mel_path", "kern_gt_path", "n_frames", "audio_measures"):
        if item.get(key) != official.get(key):
            raise ValueError(f"{render_id}: {key} differs from the test manifest")


def check_manifest(test_manifest: str, manifest: str) -> None:
    official = _test_manifest(test_manifest)
    items = json.loads(Path(manifest).read_text())
    if len(items) != len(official) or {item["id"] for item in items} != set(official):
        raise ValueError(f"{manifest}: manifest does not cover exactly the current test renders")
    for item in items:
        normalized = {
            "performance_id": item["id"],
            "piece_id": Path(item["kern_gt_path"]).stem,
            **item,
        }
        _check_render(normalized, official[item["id"]])


def check_chunks(test_manifest: str, directory: str) -> None:
    official = _test_manifest(test_manifest)
    paths = list(Path(directory).iterdir())
    if not paths:
        raise ValueError(f"{directory}: no chunks")
    for path in paths:
        recording, _, position = path.stem.rpartition(".")
        piece, _, render = recording.partition("#")
        if (not position.isdigit() or render not in official
                or piece != Path(official[render]["kern_gt_path"]).stem):
            raise ValueError(f"{path}: chunk is outside the current test split")


def select_inputs(test_manifest: str, preseg: str, out: str, kind: str) -> None:
    """Select existing windows by the current split without changing their identity."""
    if kind not in ("decode", "score"):
        raise ValueError(f"Unknown grounding kind: {kind}")
    official = _test_manifest(test_manifest)
    source = Path(preseg)
    items = json.loads((source / "manifest_decode.json").read_text())
    selected = [item for item in items if item["id"] in official]
    indexed = {}
    for item in selected:
        _check_render(item, official[item["id"]])
        recording = f"{item['piece_id']}#{item['performance_id']}"
        if recording in indexed:
            raise ValueError(f"Duplicate manifest recording: {recording}")
        indexed[recording] = item

    groundings = {}
    source_counts = {}
    for name in (["decode", "score"] if kind == "score" else ["decode"]):
        rows = [json.loads(line) for line in (source / f"grounding_{name}.jsonl").read_text().splitlines()
                if line.strip()]
        source_counts[name] = len(rows)
        kept = []
        seen = set()
        for row in rows:
            recording = f"{row['piece_id']}#{row['performance_id']}"
            chunk = f"{recording}.{row['position_index']}"
            if row["recording_id"] != recording or row["chunk_id"] != chunk:
                raise ValueError(f"{row['chunk_id']}: inconsistent window identity")
            if row["performance_id"] not in official:
                continue
            if recording not in indexed or chunk in seen:
                raise ValueError(f"{row['chunk_id']}: duplicate or inconsistent window identity")
            start, end = row["start_bar_index"], row["end_bar_index_exclusive"]
            if start < 0 or end - start != 5 or end > len(indexed[recording]["audio_measures"]):
                raise ValueError(f"{chunk}: invalid five-bar range [{start}, {end})")
            seen.add(chunk)
            kept.append(row)
        if not kept:
            raise ValueError(f"{source}: no {name} windows in the current test split")
        groundings[name] = kept
    if kind == "score":
        decode = {row["chunk_id"]: row for row in groundings["decode"]}
        for row in groundings["score"]:
            other = decode.get(row["chunk_id"])
            if other is None or any(row[key] != other[key] for key in (
                "recording_id", "start_bar_index", "end_bar_index_exclusive", "target_pkl_sha256",
            )):
                raise ValueError(f"{row['chunk_id']}: scoring window differs from the decode window")

    recordings = {row["recording_id"] for row in groundings["decode"]}
    selected = [item for item in selected
                if f"{item['piece_id']}#{item['performance_id']}" in recordings]
    absent = sorted(set(official) - {item["id"] for item in selected})
    summary = {
        "test_manifest": str(Path(test_manifest).resolve()),
        "source": str(source.resolve()),
        "test_recordings": len(official),
        "source_recordings": len(items),
        "decode_recordings": len(selected),
        "test_recordings_without_decode_windows": len(absent),
        "source_windows": source_counts,
        "selected_windows": {name: len(rows) for name, rows in groundings.items()},
    }
    output = Path(out)
    payloads = {
        "manifest_decode.json": json.dumps(selected, indent=2) + "\n",
        "recordings.txt": "name\n" + "".join(render + "\n" for render in official),
        f"selection_{kind}.json": json.dumps({**summary, "recordings_without_decode_windows": absent}, indent=2) + "\n",
        **{f"grounding_{name}.jsonl": "".join(json.dumps(row) + "\n" for row in rows)
           for name, rows in groundings.items()},
    }
    # Workers share these inputs; a later submission must not change their row indices.
    for name, text in payloads.items():
        path = output / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f"{path}: selected inputs changed; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, text in payloads.items():
        path = output / name
        if path.exists():
            continue
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as handle:
            handle.write(text)
        os.replace(handle.name, path)
    print(json.dumps(summary))


def decode_inventory(kern_dir: str, grounding: str) -> None:
    windows = [json.loads(line)["chunk_id"] for line in Path(grounding).read_text().splitlines()
               if line.strip()]
    ready = sum((Path(kern_dir) / f"{chunk}.krn").is_file() for chunk in windows)
    print(json.dumps({"windows": len(windows), "kern": ready}))


def pa2s_mv2h(results: str, tasks: str) -> None:
    """Every Piano-A2S MV2H task wrote its summary, and together they cover every prediction."""
    results_dir, n_tasks = Path(results), int(tasks)
    predictions = len(list((results_dir / "test").glob("*.json")))
    missing = [t for t in range(n_tasks) if not (results_dir / f"slurm_task_{t}.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} MV2H tasks wrote no summary: {missing[:20]}")
    counts: Counter[str] = Counter()
    for t in range(n_tasks):
        counts.update(json.loads((results_dir / f"slurm_task_{t}.json").read_text())["status_counts"])
    if sum(counts.values()) != predictions:
        raise SystemExit(f"task summaries cover {sum(counts.values())} of {predictions} predictions")
    print(json.dumps({"predictions": predictions, "tasks": n_tasks, "status_counts": dict(sorted(counts.items())),
                      "score_files": len(list((results_dir / "mv2h").glob("*_mv2h.json")))}))


def score(eval_csv: str, grounding: str) -> None:
    """The merged CSV holds exactly the scoring windows, in grounding order."""
    rows = list(csv.DictReader(open(eval_csv)))
    windows = [json.loads(line)["chunk_id"] for line in open(grounding) if line.strip()]
    if [row["task_id"] for row in rows] != windows:
        raise SystemExit(f"merged CSV has {len(rows)} rows, grounding has {len(windows)} windows")
    print(json.dumps({"windows": len(rows), "status_counts": dict(sorted(Counter(r["status"] for r in rows).items()))}))


def decode_manifest(manifest: str, grounding: str, out: str) -> None:
    """The decoder falls back to its own stride-five scopes for a recording with
    no grounding rows, so it only receives recordings that have windows."""
    recordings = {json.loads(line)["recording_id"]
                  for line in Path(grounding).read_text().splitlines() if line.strip()}
    items = [item for item in json.loads(Path(manifest).read_text())
             if f"{item['piece_id']}#{item['performance_id']}" in recordings]
    if len(items) != len(recordings):
        raise SystemExit(f"{len(recordings)} grounded recordings, {len(items)} manifest matches")
    Path(out).write_text(json.dumps(items, indent=2) + "\n")
    print(json.dumps({"decode_recordings": len(items)}))


COMMANDS = {
    "pa2s-mv2h": pa2s_mv2h, "score": score, "decode-manifest": decode_manifest,
    "select-inputs": select_inputs, "check-manifest": check_manifest,
    "check-chunks": check_chunks, "decode-inventory": decode_inventory,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[sys.argv[1]](*sys.argv[2:])
