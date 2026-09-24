"""Summarize complete-score MV2H with recording means and paired work bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.analysis.paired_work_bootstrap import (
    METRICS, REPORTED, compare_work_ratio_seeds,
)


def parse_system(spec: str) -> tuple[str, tuple[str, list[list[Path]]]]:
    """NAME[:ARM]=PATH[,PATH...]: one results file per training seed of a system.

    ARM is the ``arm`` label the rows carry inside the results file; it defaults
    to NAME. Naming it lets one results file serve several table rows (the
    self-segmented decode under each boundary condition writes its own file,
    each labelled ``ours``) and lets a foreign file's label be renamed.
    """
    name, _, paths = spec.partition("=")
    if not name or not paths:
        raise argparse.ArgumentTypeError("system spec must be NAME[:ARM]=PATH[,PATH...]")
    name, _, arm = name.partition(":")
    return name, (arm or name, [[Path(p)] for p in paths.split(",")])



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", type=Path)
    parser.add_argument("--system", action="append", type=parse_system,
                        help="NAME[:ARM]=PATH[,PATH...]; several paths are training seeds")
    parser.add_argument("--recording-list", type=Path, required=True)
    parser.add_argument("--candidate", default="ours")
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    keep = set(args.recording_list.read_text().splitlines()) - {"", "name"}
    names = [args.candidate, *args.reference]
    if args.system:
        sources = dict(args.system)
        if set(sources) != set(names):
            raise ValueError("--system must name exactly the candidate and its references")
    elif args.results:
        sources = {name: (name, [list(args.results)]) for name in names}
    else:
        raise ValueError("Pass --results or --system")
    all_paths = [path for _, seeds in sources.values() for group in seeds for path in group]

    matched = set()

    def load(arm: str, paths: list[Path]) -> dict:
        rows = {}
        for path in paths:
            for row in json.loads(path.read_text()):
                if row["arm"] != arm:
                    continue
                rid = row["recording_id"]
                aliases = {rid, rid.replace("#", "__"), rid.rsplit("#", 1)[-1]}
                if not aliases & keep:
                    continue
                matched.update(aliases & keep)
                if rid in rows:
                    raise ValueError(f"Duplicate result: {arm}/{rid}")
                rows[rid] = row
        return rows

    systems = {name: [load(arm, group) for group in seeds] for name, (arm, seeds) in sources.items()}
    if matched != keep:
        raise ValueError(f"Recording list has {len(keep - matched)} unmatched entries")
    inventory = set(systems[args.candidate][0])
    if not inventory or any(set(rows) != inventory
                            for seeds in systems.values() for rows in seeds):
        raise ValueError("Systems must account for the same complete recording inventory")
    scored = lambda r: r["status"] in ("success", "zero_score")
    ids = sorted(inventory)
    reference_rows = systems[args.candidate][0]
    works = sorted({reference_rows[rid]["piece_id"] for rid in ids})
    work_index = {work: i for i, work in enumerate(works)}
    totals, coverage = {}, {}
    for name, seeds in systems.items():
        totals[name], per_seed_coverage = [], []
        for rows in seeds:
            seed_totals = {m: np.zeros((len(works), 2)) for m in REPORTED}
            successful, failed = [], []
            for rid in ids:
                row = rows[rid]
                if row["piece_id"] != reference_rows[rid]["piece_id"]:
                    raise ValueError(f"Conflicting work identity for {rid}")
                if row["status"] == "reference_failed":
                    raise ValueError(f"Unresolved reference failure: {rid}")
                if not scored(row):
                    failed.append(rid)
                    continue
                successful.append(rid)
                scores = {m: float(row["metrics"][m]) for m in METRICS}
                if not all(np.isfinite(v) and 0 <= v <= 1.000000001 for v in scores.values()):
                    raise ValueError(f"Invalid MV2H scores: {name}/{rid}")
                scores["MV2H4"] = sum(scores[m] for m in METRICS if m != "Meter") / 4
                scores["MV2H5"] = sum(scores[m] for m in METRICS) / 5
                for m, value in scores.items():
                    seed_totals[m][work_index[row["piece_id"]]] += (value, 1)
            if not successful:
                raise ValueError(f"No successful recordings: {name}")
            totals[name].append(seed_totals)
            per_seed_coverage.append({
                "n_total": len(ids), "n_success": len(successful), "n_fail": len(failed),
                "successful_recording_ids": successful, "failed_recording_ids": failed,
                "successful_works": len({rows[r]["piece_id"] for r in successful}),
            })
        coverage[name] = (per_seed_coverage[0] if len(per_seed_coverage) == 1
                          else {"seeds": per_seed_coverage,
                                "n_total": len(ids),
                                "n_success": min(c["n_success"] for c in per_seed_coverage),
                                "n_fail": max(c["n_fail"] for c in per_seed_coverage)})
    result = {
        "protocol": {"unit": "recording", "statistic": "recording-macro mean",
                     "inventory_recordings": len(inventory), "recordings": len(ids),
                     "works": len(works), "failure_policy": "successful_only",
                     "comparison": "shared work draws; separate successful-recording denominators",
                     "n_boot": args.n_boot, "seed": args.seed,
                     "seeds": {name: len(seeds) for name, seeds in totals.items()},
                     "arms": {name: arm for name, (arm, _) in sources.items()}},
        "recording_ids": ids,
        "coverage": coverage,
        "means": {name: {m: float(np.mean([t[m][:, 0].sum() / t[m][:, 1].sum() for t in seeds]))
                         for m in REPORTED} for name, seeds in totals.items()},
        "comparisons": {
            name: {m: compare_work_ratio_seeds([t[m] for t in totals[args.candidate]],
                                               [t[m] for t in totals[name]],
                                               args.n_boot, args.seed)
                   for m in REPORTED}
            for name in args.reference},
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [*all_paths, args.recording_list, Path(__file__),
                             Path(__file__).with_name("paired_work_bootstrap.py")]},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "whole_score_table.json").write_text(json.dumps(result, indent=2) + "\n")
    text = [f"{len(ids)} recordings / {len(works)} works; successful only per system", "",
            "| System | n_success | n_fail | " + " | ".join(REPORTED) + " |",
            "|---|---:|---:|" + "---:|" * len(REPORTED)]
    for name, row in result["means"].items():
        text.append(f"| {name} | {coverage[name]['n_success']} | {coverage[name]['n_fail']} | "
                    + " | ".join(f"{row[m]:.3f}" for m in REPORTED) + " |")
    (args.out / "whole_score_table.md").write_text("\n".join(text) + "\n")
    print(json.dumps(result["protocol"]))


if __name__ == "__main__":
    main()
