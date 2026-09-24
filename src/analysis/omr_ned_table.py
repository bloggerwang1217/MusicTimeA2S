#!/usr/bin/env python3
"""Whole-score OMR-NED table on a manifest and a recording subset.

Each system supplies a completed ``summary.json`` or an explicit summary path;
a baseline supplies musicdiff's ``output.csv``. Cells report corpus OMR-NED
(total edit distance over total symbols) plus/minus the half-width of its
work-clustered percentile 95% interval. Exact endpoints are retained in JSON.
Only successful results contribute; failures are counted separately.

Usage:
    poetry run python -m src.analysis.omr_ned_table \\
        --manifest data/experiments/asap102/test_manifest.json \\
        --subset src/datasets/asap/asap102_hft_clean_74_recordings.txt \\
        --subset-name "hFT-clean 74" \\
        --system Ours=data/experiments/asap102/omr_ned_selfseg_<tag> \\
        --system "w/o Fourier PE"=data/experiments/asap102/omr_ned_selfseg_<other tag> \\
        --published "Rubato=64.3 ± 3.9" \\
        --out data/experiments/asap102/omr_ned_table_<date>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.analysis.paired_work_bootstrap import compare_work_ratio_seeds
from src.evaluation.omr_ned_subset import load_list


def spec(value: str) -> tuple[str, list[str]]:
    """NAME=PATH[,PATH...]: one output per training seed of that system."""
    name, sep, paths = value.partition("=")
    if not sep or not name or not paths:
        raise argparse.ArgumentTypeError("expected NAME=PATH[,PATH...]")
    return name, paths.split(",")


def work_totals_of(rows: list[dict[str, Any]], works: list[str]) -> list[list[int]]:
    by_work = defaultdict(list)
    for row in rows:
        by_work[row["piece_id"]].append((row["OMR-ED"], row["total_symbols"]))
    return [[sum(ed for ed, _ in by_work[w]), sum(n for _, n in by_work[w])] for w in works]


def corpus_cell(seeds: list[list[dict[str, Any]]], works: list[str], args) -> dict[str, Any]:
    """One cell of the table; several row lists are the seeds of one system."""
    totals = [work_totals_of(rows, works) for rows in seeds]
    boot = compare_work_ratio_seeds(totals, totals, args.resamples, args.seed)
    lo, hi = [100 * v for v in boot["candidate_ci95"]]
    first = seeds[0]
    return {
        "n": len(first),
        "corpus": 100 * boot["candidate_mean"],
        "ci95": [lo, hi],
        "ci95_half_width": (hi - lo) / 2,
        "works": len({r["piece_id"] for r in first}),
        "inventory_works": len(works),
        "total_edit_distance": sum(r["OMR-ED"] for r in first),
        "total_symbols": sum(r["total_symbols"] for r in first),
        "recordings": sorted(r["task_id"] for r in first),
        "seeds": len(seeds),
        "bootstrap": boot,
    }


def summary_path(source: Path) -> Path:
    return source / "summary.json" if source.is_dir() else source


def our_cell(sources: list[Path], keep: set[str] | None,
             recording_to_work: dict[str, str], args) -> dict[str, Any]:
    per_seed, meta = [], None
    for source in sources:
        rows, successful, works, info = _read_summary(source, keep, recording_to_work)
        per_seed.append(successful)
        meta = meta or info
        meta["n_fail"] = max(meta["n_fail"], info["n_fail"])
    return {**corpus_cell(per_seed, works, args), **meta}


def _read_summary(source: Path, keep: set[str] | None, recording_to_work: dict[str, str]):
    summary = json.loads(summary_path(source).read_text())
    rows = summary["rows"]
    if len({r["task_id"] for r in rows}) != len(rows) or len(rows) != summary["n_total"]:
        raise ValueError(f"Incomplete or duplicate summary rows: {source}")
    expected = set(recording_to_work) if keep is None else keep
    rows = [r for r in rows if r["task_id"] in expected]
    if {r["task_id"] for r in rows} != expected:
        raise ValueError(f"Summary does not account for the selected inventory: {source}")
    for row in rows:
        if row["piece_id"] != recording_to_work[row["task_id"]]:
            raise ValueError(f"Inconsistent work identity: {row['task_id']}")
    successful = [r for r in rows if r["status"] == "success"]
    works = sorted({recording_to_work[r] for r in expected})
    return rows, successful, works, {
        "n_total": len(rows), "n_fail": len(rows) - len(successful),
        "failed_recordings": sorted(r["task_id"] for r in rows if r["status"] != "success")}


def baseline_cell(csv_path: Path, keep: set[str] | None, recording_to_work: dict[str, str], args) -> dict[str, Any]:
    rows = []
    expected = set(recording_to_work) if keep is None else keep
    for row in csv.DictReader(csv_path.open()):
        stem = Path(row[" gtpath"].strip()).name.replace(".musicxml", "")
        if not stem or not row[" OMR-ED (OMR Edit Distance)"].strip():
            continue
        if stem not in expected:
            continue
        try:
            ed = int(float(row[" OMR-ED (OMR Edit Distance)"]))
            total = int(float(row[" total numsyms (in both scores)"]))
        except ValueError:
            continue  # repeated header or total row inside the csv
        rows.append({"task_id": stem, "piece_id": recording_to_work[stem],
                     "OMR-ED": ed, "total_symbols": total})
    if len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate baseline recordings: {csv_path}")
    failed = sorted(expected - {r["task_id"] for r in rows})
    works = sorted({recording_to_work[r] for r in expected})
    return {**corpus_cell([rows], works, args), "n_total": len(expected),
            "n_fail": len(failed), "failed_recordings": failed}


def fmt(cell: dict[str, Any] | None) -> str:
    if cell is None or cell.get("corpus") is None:
        return "--"
    text = f"{cell['corpus']:.1f}"
    if "ci95_half_width" in cell:
        text += f" ± {cell['ci95_half_width']:.1f}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True, help="Recording id to piece_id mapping")
    parser.add_argument("--subset", type=Path, required=True, help="name-header recording list")
    parser.add_argument("--subset-name", default="subset")
    parser.add_argument("--system", type=spec, action="append", default=[])
    parser.add_argument("--baseline", type=spec, action="append", default=[])
    parser.add_argument("--published", type=spec, action="append", default=[],
                        help="NAME=text printed verbatim in the full column")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    keep = load_list(args.subset)
    manifest = json.loads(args.manifest.read_text())
    recording_to_work = {r["id"]: r["piece_id"] for r in manifest}
    if len(recording_to_work) != len(manifest) or not keep <= recording_to_work.keys():
        raise ValueError("Duplicate manifest ids or subset recordings absent from manifest")

    rows: list[dict[str, Any]] = []
    for name, text in args.published:
        rows.append({"system": name, "kind": "published",
                     "all": {"text": ",".join(text)}, "subset": None})
    for name, paths in args.baseline:
        if len(paths) != 1:
            raise ValueError(f"A baseline is one released system, not seeds: {name}")
        rows.append({"system": name, "kind": "baseline",
                     "all": baseline_cell(Path(paths[0]), None, recording_to_work, args),
                     "subset": baseline_cell(Path(paths[0]), keep, recording_to_work, args)})
    for name, paths in args.system:
        sources = [Path(path) for path in paths]
        rows.append({"system": name, "kind": "ours",
                     "all": our_cell(sources, None, recording_to_work, args),
                     "subset": our_cell(sources, keep, recording_to_work, args)})

    lines = [f"| System | Manifest inventory | {args.subset_name} |", "|---|---:|---:|"]
    for row in rows:
        if row["kind"] == "published":
            lines.append(f"| {row['system']} | {row['all']['text']} | -- |")
        else:
            lines.append(f"| {row['system']} | {fmt(row['all'])} | {fmt(row['subset'])} |")
    lines.append("")
    lines.append("Corpus OMR-NED (lower is better); ± half-width of the work-clustered percentile 95% CI "
                 f"({args.resamples} resamples, seed {args.seed}). The half-width summarizes interval width; "
                 "point ± half-width need not reproduce the exact endpoints retained in JSON. "
                 "Each system uses only its successful recordings; failures are reported separately. "
                 "Published uncertainties are quoted, not recomputed.")
    for row in rows:
        if row["kind"] != "published":
            a, b = row["all"], row["subset"]
            lines.append(f"- {row['system']}: n {a['n']} / {b['n']}, "
                         f"failures {a['n_fail']} / {b['n_fail']}, works {a['works']} / {b['works']}")
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = [args.manifest, args.subset, Path(__file__), Path(__file__).with_name("paired_work_bootstrap.py")]
    inputs += [summary_path(Path(path)) for _, paths in args.system or [] for path in paths]
    inputs += [Path(path) for _, paths in args.baseline or [] for path in paths]
    sources = [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
    (args.out / "omr_ned_table.md").write_text("\n".join(lines) + "\n")
    (args.out / "omr_ned_table.json").write_text(json.dumps(
        {"subset_list": str(args.subset), "subset_name": args.subset_name, "sources": sources,
         "statistic": "corpus OMR-NED", "failure_policy": "successful_only",
         "resamples": args.resamples, "seed": args.seed,
         "rows": rows}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
