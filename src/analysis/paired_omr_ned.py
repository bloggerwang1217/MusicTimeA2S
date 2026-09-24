"""Compare successful-output corpus OMR-NED with shared work bootstrap draws."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.analysis.omr_ned_table import summary_path
from src.analysis.paired_work_bootstrap import compare_work_ratio_seeds
from src.evaluation.omr_ned_subset import load_list


def parse_system(spec: str) -> tuple[str, list[Path]]:
    """NAME=PATH[,PATH...]: one output directory per training seed of a system."""
    name, _, paths = spec.partition("=")
    if not name or not paths:
        raise argparse.ArgumentTypeError("system spec must be NAME=PATH[,PATH...]")
    return name, [Path(path) for path in paths.split(",")]


def load_scores(source: Path, works: dict[str, str], raw_results: bool) -> tuple[dict, Path]:
    if raw_results:
        path = source / "omr_ned" / "results.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif source.is_dir() or source.suffix == ".json":
        path = summary_path(source)
        records = json.loads(path.read_text())["rows"]
    else:
        records = []
        path = source
        for row in csv.DictReader(path.open()):
            rid = Path(row[" gtpath"].strip()).name.replace(".musicxml", "")
            if rid not in works or not row[" OMR-ED (OMR Edit Distance)"].strip():
                continue
            try:
                edits = int(float(row[" OMR-ED (OMR Edit Distance)"]))
                symbols = int(float(row[" total numsyms (in both scores)"]))
            except ValueError:
                continue
            records.append({"task_id": rid, "piece_id": works[rid], "status": "success",
                            "OMR-ED": edits, "total_symbols": symbols})
    rows = {}
    for row in records:
        rid = row.get("task_id") or row["artifact_stem"]
        if rid in rows:
            raise ValueError(f"Duplicate recording: {source}/{rid}")
        rows[rid] = row
    return rows, path


def work_totals(rows: dict, keep: set[str], works: dict[str, str], names: list[str]) -> tuple[np.ndarray, dict]:
    totals = np.zeros((len(names), 2))
    index = {w: i for i, w in enumerate(names)}
    successful = []
    for rid in sorted(keep):
        row = rows.get(rid)
        if row is None or row["status"] != "success":
            continue
        if row["piece_id"] != works[rid]:
            raise ValueError(f"Conflicting work identity: {rid}")
        totals[index[works[rid]]] += (row["OMR-ED"], row["total_symbols"])
        successful.append(rid)
    return totals, {"n_total": len(keep), "n_success": len(successful),
                    "n_fail": len(keep) - len(successful),
                    "successful_recordings": successful,
                    "failed_recordings": sorted(keep - set(successful))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=parse_system, required=True)
    p.add_argument("--reference", type=parse_system, action="append", required=True)
    p.add_argument("--scores", choices=("results", "summary"), default="results")
    p.add_argument("--subset", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.scores == "summary" and not a.manifest:
        p.error("--scores summary requires --manifest to account for missing recordings")
    specs = [a.candidate, *a.reference]
    if len({name for name, _ in specs}) != len(specs):
        p.error("system names must be distinct")
    works = ({m["id"]: m["piece_id"] for m in json.loads(a.manifest.read_text())}
             if a.manifest else {})
    systems, paths = {}, []
    for name, sources in specs:
        systems[name] = []
        for source in sources:
            rows, path = load_scores(source, works, a.scores == "results")
            systems[name].append(rows)
            paths.append(path)
    if not works:
        for seeds in systems.values():
          for rows in seeds:
            for rid, row in rows.items():
                if rid in works and works[rid] != row["piece_id"]:
                    raise ValueError(f"Conflicting work identity: {rid}")
                works[rid] = row["piece_id"]
    keep = load_list(a.subset) if a.subset else set(works)
    if not keep or not keep <= works.keys():
        raise ValueError("Empty inventory or recording absent from manifest")
    names = sorted({works[rid] for rid in keep})
    totals, coverage = {}, {}
    for name, seeds in systems.items():
        pairs = [work_totals(rows, keep, works, names) for rows in seeds]
        totals[name] = [t for t, _ in pairs]
        per_seed = [c for _, c in pairs]
        coverage[name] = (per_seed[0] if len(per_seed) == 1
                          else {"seeds": per_seed, "n_total": per_seed[0]["n_total"],
                                "n_success": min(c["n_success"] for c in per_seed),
                                "n_fail": max(c["n_fail"] for c in per_seed)})
    cname = a.candidate[0]
    result = {
        "candidate": cname, "subset": str(a.subset) if a.subset else None,
        "resamples": a.n_boot, "seed": a.seed, "works": names,
        "statistic": "difference in corpus OMR-NED on each system's successful recordings",
        "failure_policy": "successful_only",
        "bootstrap": "shared work draws; separate edit-distance and symbol totals",
        "coverage": coverage,
        "seeds": {name: len(seeds) for name, seeds in totals.items()},
        "comparisons": {name: compare_work_ratio_seeds(totals[cname], totals[name],
                                                       a.n_boot, a.seed)
                        for name, _ in a.reference},
    }
    paths += [p for p in (a.manifest, a.subset, Path(__file__),
                         Path(__file__).with_name("paired_work_bootstrap.py")) if p]
    result["sources"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in paths]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "paired_omr_ned.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["Corpus OMR-NED differences in percentage points; shared work draws, successful only.", "",
             "| Candidate | Reference | Difference | 95% CI |", "|---|---|---:|---|"]
    for name, pair in result["comparisons"].items():
        lo, hi = pair["ci95"]
        lines.append(f"| {cname} | {name} | {100 * pair['diff']:.3f} | [{100 * lo:.3f}, {100 * hi:.3f}] |")
    (a.out / "paired_omr_ned.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
