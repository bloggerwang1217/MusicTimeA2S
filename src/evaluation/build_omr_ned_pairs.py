#!/usr/bin/env python3
"""Build an OMR-NED pair manifest from whole-piece kern predictions.

Converts every ``<artifact_stem>.krn`` in a prediction directory to MusicXML
with the project's converter21 route and pairs it with the frozen ASAP-102
reference ``xml_score.musicxml``.  The resulting ``pairs.jsonl`` feeds
``omr_ned_shard.py`` / ``slurm_omr_ned.sh`` unchanged.

A recording whose kern is missing or fails to convert stays in the manifest
with its intended (absent) ``prediction_xml`` path, so the shard worker records
it as an error and the merge fails closed instead of silently shrinking the
denominator.

``--prediction-manifest`` pairs already-rendered MusicXML predictions (a
cascade's own output) with the ``--mapping`` references instead.  Both modes
also write ``pairs_ready.jsonl``, the rows musicdiff can score; the summary
still reads ``pairs.jsonl`` and scores every other recording as a failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.asap import load_asap102_inventory, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = REPO_ROOT / "src" / "datasets" / "asap" / "metadata_R.csv"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines.yaml"
PRE_CLEAN_MODES = ("none", "midi")
STDERR_TAIL_CHARS = 2000
KERN_PITCH = re.compile(r"[A-Ga-g]")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_pairs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(out_dir / "pairs.jsonl", rows)
    _write_jsonl(out_dir / "pairs_ready.jsonl", [
        row for row in rows if row["status"] == "ready" and row.get("prediction_xml")
    ])


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve_asap_root(cli_value: Path | None, config_path: Path) -> Path:
    if cli_value is not None:
        return cli_value
    if not config_path.is_file():
        raise FileNotFoundError(
            f"--asap-root not given and config is missing: {config_path}"
        )
    config = yaml.safe_load(config_path.read_text()) or {}
    raw = config.get("asap_root")
    if not raw:
        raise ValueError(f"Config lacks asap_root: {config_path}")
    if str(raw).startswith("/path/to/"):
        raise ValueError(f"{config_path}: asap_root is not filled in")
    return Path(raw)


def _locate_kern(pred_kern_dir: Path, record: dict[str, Any]) -> Path | None:
    # Whole-piece exports are named by artifact_stem; accept the raw
    # recording_id spelling as well so hand-made directories still resolve.
    for name in (record["artifact_stem"], record["recording_id"]):
        candidate = pred_kern_dir / f"{name}.krn"
        if candidate.is_file():
            return candidate
    return None


def _kern_has_notes(text: str) -> bool:
    return any(
        line and line[0] not in "*!=" and KERN_PITCH.search(line)
        for line in text.splitlines()
    )


def convert_one(kern_path: Path, xml_path: Path, pre_clean: str) -> None:
    """Convert one kern file in-process; run through ``--convert-one``."""
    import music21

    from src.score.generate_score import kern_to_musicxml

    source = kern_path
    if pre_clean == "midi":
        from src.score.kern_postprocess import kern_pre_midi_clean

        cleaned_dir = xml_path.parent.parent / "kern_cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        source = cleaned_dir / kern_path.name
        source.write_text(
            kern_pre_midi_clean(kern_path.read_text(), n_spines=2),
            encoding="utf-8",
        )
    kern_to_musicxml(source, xml_path)
    if not xml_path.is_file() or xml_path.stat().st_size == 0:
        raise RuntimeError(f"MusicXML was not written: {xml_path}")
    # converter21 answers a spine-topology error (e.g. field count not matching
    # a *^ split) with an empty score and no exception; scoring that would pass
    # off a broken prediction as a near-1.0 NED instead of a conversion failure.
    written = music21.converter.parse(str(xml_path), forceSource=True)
    if not any(True for _ in written.recurse().notes) and _kern_has_notes(
        source.read_text()
    ):
        raise RuntimeError(
            "converter21 produced an empty score from a kern file that has notes"
        )


def _convert_in_subprocess(
    kern_path: Path,
    xml_path: Path,
    pre_clean: str,
) -> tuple[str, str, str]:
    # A separate interpreter per file keeps a converter21 crash or corrupt
    # parse from taking the whole build down; the exit code becomes the status.
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.evaluation.build_omr_ned_pairs",
        "--convert-one",
        str(kern_path),
        str(xml_path),
        "--pre-clean",
        pre_clean,
    ]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 and xml_path.is_file() and xml_path.stat().st_size > 0:
        return "ready", "", ""
    xml_path.unlink(missing_ok=True)
    stderr_lines = [
        line for line in (completed.stderr or completed.stdout or "").splitlines()
        if line.strip()
    ]
    last_line = stderr_lines[-1].strip() if stderr_lines else ""
    tail = "\n".join(stderr_lines)[-STDERR_TAIL_CHARS:]
    return "conversion_failed", f"exit={completed.returncode}: {last_line}", tail


def _mapping_rows(mapping: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in mapping.read_text().splitlines() if line.strip()]


def _mapping_records(mapping: Path) -> list[dict[str, Any]]:
    # A frozen mapping already names each recording's reference score, so it
    # can stand in for the ASAP-102 inventory on other splits.
    rows = _mapping_rows(mapping)
    return sorted((
        {key: row[key] for key in ("recording_id", "artifact_stem", "piece_id", "performance_id", "source_xml")}
        for row in rows
    ), key=lambda row: row["recording_id"])


def build_pairs(
    *,
    pred_kern_dir: Path,
    out_dir: Path,
    metadata: Path | None,
    asap_root: Path | None,
    pre_clean: str,
    limit: int | None,
    workers: int,
    resume: bool,
    mapping: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    if mapping is not None:
        records = _mapping_records(mapping)
    else:
        records = load_asap102_inventory(metadata, asap_root)
    if limit is not None:
        records = records[:limit]
    xml_dir = out_dir / "musicxml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, Path, Path]] = []
    for index, record in enumerate(records):
        reference = Path(record["source_xml"]).resolve()
        xml_path = (xml_dir / f"{record['artifact_stem']}.musicxml").resolve()
        kern_path = _locate_kern(pred_kern_dir, record)
        row: dict[str, Any] = {
            "recording_id": record["recording_id"],
            "artifact_stem": record["artifact_stem"],
            "piece_id": record["piece_id"],
            "performance_id": record["performance_id"],
            "status": "pending",
            "error_message": "",
            "prediction_kern": str(kern_path.resolve()) if kern_path else None,
            "prediction_kern_sha256": sha256_file(kern_path) if kern_path else None,
            "prediction_xml": str(xml_path),
            "prediction_xml_sha256": None,
            "reference_xml": str(reference),
            "reference_xml_sha256": sha256_file(reference),
        }
        rows.append(row)
        if kern_path is None:
            row["status"] = "kern_missing"
            row["error_message"] = (
                f"no {record['artifact_stem']}.krn under {pred_kern_dir}"
            )
            continue
        if resume and xml_path.is_file() and xml_path.stat().st_size > 0:
            row["status"] = "ready"
            continue
        jobs.append((index, kern_path.resolve(), xml_path))

    def run(job: tuple[int, Path, Path]) -> tuple[int, str, str, str]:
        index, kern_path, xml_path = job
        status, message, tail = _convert_in_subprocess(kern_path, xml_path, pre_clean)
        return index, status, message, tail

    print(
        f"Converting {len(jobs)} kern files with {workers} workers "
        f"(pre_clean={pre_clean})",
        file=sys.stderr,
    )
    stderr_tails: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for done, (index, status, message, tail) in enumerate(pool.map(run, jobs), 1):
            rows[index]["status"] = status
            rows[index]["error_message"] = message
            if tail:
                stderr_tails[index] = tail
            print(
                f"[{done}/{len(jobs)}] {rows[index]['artifact_stem']}: {status}",
                file=sys.stderr,
            )

    for row in rows:
        if row["status"] == "ready":
            row["prediction_xml_sha256"] = sha256_file(Path(row["prediction_xml"]))

    status_counts = Counter(row["status"] for row in rows)
    failures = [
        {
            **{
                key: row[key]
                for key in (
                    "recording_id",
                    "artifact_stem",
                    "status",
                    "error_message",
                    "prediction_kern",
                )
            },
            "stderr_tail": stderr_tails.get(index, ""),
        }
        for index, row in enumerate(rows)
        if row["status"] != "ready"
    ]
    _write_pairs(out_dir, rows)

    import converter21
    import music21

    report = {
        "pred_kern_dir": str(pred_kern_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "metadata": str(metadata.resolve()) if metadata else None,
        "asap_root": str(asap_root.resolve()) if asap_root else None,
        "mapping": str(mapping.resolve()) if mapping else None,
        "pre_clean": pre_clean,
        "limit": limit,
        "resume": resume,
        "converter": {
            "route": "converter21 -> music21 -> MusicXML",
            "converter21": getattr(converter21, "__version__", None),
            "music21": music21.__version__,
        },
        "n_total": len(rows),
        "n_ready": status_counts.get("ready", 0),
        "n_failed": len(failures),
        "status_counts": dict(sorted(status_counts.items())),
        "failures": failures,
        "pairs": str((out_dir / "pairs.jsonl").resolve()),
        "elapsed_sec": round(time.time() - started, 1),
    }
    _write_json(out_dir / "build_report.json", report)
    return report


def pair_score_predictions(prediction_manifest: Path, mapping: Path, out_dir: Path) -> dict[str, Any]:
    """Pair each rendered MusicXML prediction with its mapped reference score."""
    mapped = _mapping_rows(mapping)
    reference = {row["artifact_stem"]: row for row in mapped}
    if len(reference) != len(mapped):
        raise ValueError("Reference mapping has duplicate artifact identities")
    rows: list[dict[str, Any]] = []
    for line in prediction_manifest.read_text().splitlines():
        if not line.strip():
            continue
        prediction = json.loads(line)
        # Render identity is shared even when a cascade uses different work IDs.
        ref = reference[prediction["artifact_stem"]]
        ready = prediction["status"] == "ready" and prediction.get("prediction_xml")
        rows.append({
            "recording_id": ref["recording_id"],
            "artifact_stem": prediction["artifact_stem"],
            "piece_id": ref["piece_id"],
            "performance_id": ref["performance_id"],
            "status": prediction["status"],
            "error_message": prediction.get("error_message", ""),
            "prediction_xml": prediction["prediction_xml"] if ready else None,
            "prediction_xml_sha256": prediction.get("prediction_xml_sha256") if ready else None,
            "reference_xml": ref["source_xml"],
            "reference_xml_sha256": ref["source_xml_sha256"],
        })
    if len({row["artifact_stem"] for row in rows}) != len(rows):
        raise ValueError("Prediction manifest has duplicate artifact identities")
    missing = set(reference) - {row["artifact_stem"] for row in rows}
    if missing:
        raise ValueError(f"prediction manifest lacks {len(missing)} reference recordings")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_pairs(out_dir, rows)
    return {"n_total": len(rows), "status_counts": dict(sorted(Counter(r["status"] for r in rows).items()))}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-kern-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--asap-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML whose asap_root is used when --asap-root is omitted",
    )
    parser.add_argument(
        "--pre-clean",
        choices=PRE_CLEAN_MODES,
        default="none",
        help="'midi' applies the grace-strip + tiefix cleanup the MV2H adapter uses",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        help="Frozen mapping.jsonl naming each recording's reference score; replaces the ASAP-102 inventory",
    )
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        help="JSONL of rendered MusicXML predictions to pair with --mapping; no kern conversion",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing non-empty MusicXML outputs instead of reconverting",
    )
    parser.add_argument(
        "--convert-one",
        nargs=2,
        metavar=("KERN", "MUSICXML"),
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.convert_one is not None:
        kern_path, xml_path = args.convert_one
        convert_one(kern_path, xml_path, args.pre_clean)
        return
    if args.prediction_manifest is not None:
        if args.mapping is None or args.out_dir is None:
            raise SystemExit("--prediction-manifest needs --mapping and --out-dir")
        print(json.dumps(pair_score_predictions(args.prediction_manifest, args.mapping, args.out_dir),
                         indent=2, sort_keys=True))
        return
    if args.pred_kern_dir is None or args.out_dir is None:
        raise SystemExit("--pred-kern-dir and --out-dir are required")
    if not args.pred_kern_dir.is_dir():
        raise SystemExit(f"Prediction kern directory does not exist: {args.pred_kern_dir}")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    report = build_pairs(
        pred_kern_dir=args.pred_kern_dir,
        out_dir=args.out_dir,
        metadata=None if args.mapping else args.metadata,
        asap_root=None if args.mapping else _resolve_asap_root(args.asap_root, args.config),
        pre_clean=args.pre_clean,
        limit=args.limit,
        workers=args.workers,
        resume=args.resume,
        mapping=args.mapping,
    )
    print(json.dumps(
        {key: report[key] for key in ("n_total", "n_ready", "n_failed", "status_counts", "pairs")},
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
