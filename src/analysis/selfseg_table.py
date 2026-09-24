"""Assemble the whole-piece table block of one test set from the pairwise grid.

replicate_tables.sh runs whole_score_table and paired_omr_ned once per
candidate row against every other row and writes them under one directory:

    GRID/ws_<row>/whole_score_table.json    recording-mean MV2H per row, paired intervals
    GRID/omr_<row>/paired_omr_ned.json      corpus OMR-NED per row, paired intervals

This reads that grid and prints the block as the paper shows it: the means, the
transcribed share, and the two marks on our rows, ``*`` for a significant
improvement over the beat-tracker cascade and ``†`` for a significant
improvement over every other row. The Piano-A2S rows are the compared
system and carry no mark. Better means a higher MV2H component and a lower
OMR-NED; an interval that excludes zero is significant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Row keys in table order, with the row's caption. The cascade is the reference
# every other row is compared against and carries no mark of its own.
ROWS = (
    ("bt", "Piano-A2S with Beat This! downbeat segmentation"),
    ("pa2s_oracle", "Piano-A2S with annotated downbeats"),
    ("ours", "Ours, predicted boundaries and coordinate"),
    ("ann", "Ours, annotated boundaries"),
    ("annphase", "Ours, annotated boundaries and coordinate"),
)
CASCADE = "bt"
MARKED = ("ours", "ann", "annphase")
METRICS = ("Meter", "MV2H5", "OMR-NED")


def better(metric: str, pair: dict) -> bool:
    low, high = pair["ci95"]
    return high < 0 if metric == "OMR-NED" else low > 0


def load_grid(grid: Path) -> tuple[dict, dict, dict]:
    """Return means, coverage and, per (candidate, reference, metric), whether the
    candidate is significantly better."""
    means: dict[str, dict[str, float]] = {}
    coverage: dict[str, dict] = {}
    wins: dict[tuple[str, str, str], bool] = {}
    for row, _ in ROWS:
        if row == CASCADE:
            continue
        ws = json.loads((grid / f"ws_{row}" / "whole_score_table.json").read_text())
        omr = json.loads((grid / f"omr_{row}" / "paired_omr_ned.json").read_text())
        for name, values in ws["means"].items():
            means.setdefault(name, {}).update({m: values[m] for m in ("Meter", "MV2H5")})
            coverage.setdefault(name, ws["coverage"][name])
        for name, pair in omr["comparisons"].items():
            means.setdefault(row, {})["OMR-NED"] = 100 * pair["candidate_mean"]
            means.setdefault(name, {}).setdefault("OMR-NED", 100 * pair["reference_mean"])
        for name, pairs in ws["comparisons"].items():
            for metric in ("Meter", "MV2H5"):
                wins[(row, name, metric)] = better(metric, pairs[metric])
        for name, pair in omr["comparisons"].items():
            wins[(row, name, "OMR-NED")] = better("OMR-NED", pair)
    return means, coverage, wins


def mark(row: str, metric: str, wins: dict) -> str:
    if row not in MARKED:
        return ""
    others = [r for r, _ in ROWS if r != row]
    if all(wins[(row, other, metric)] for other in others):
        return "†"
    return "*" if wins[(row, CASCADE, metric)] else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--name", default="test set", help="caption of the block")
    parser.add_argument("--out", type=Path, required=True, help="output stem; writes .md and .json")
    args = parser.parse_args()
    means, coverage, wins = load_grid(args.grid)
    n_total = next(iter(coverage.values()))["n_total"]
    cells = {}
    lines = [f"{args.name}: {n_total} recordings; means over training seeds where a row has several",
             "", "| System | F_meter | MV2H5 | OMR-NED | Transcribed (%) |",
             "|---|---:|---:|---:|---:|"]
    for row, caption in ROWS:
        marks = {m: mark(row, m, wins) for m in METRICS}
        n_success = coverage[row]["n_success"]
        cells[row] = {"caption": caption, "marks": marks, "n_success": n_success, "n_total": n_total,
                      **{m: means[row][m] for m in METRICS}}
        lines.append(f"| {caption} | {means[row]['Meter']:.3f}{marks['Meter']} "
                     f"| {means[row]['MV2H5']:.3f}{marks['MV2H5']} "
                     f"| {means[row]['OMR-NED']:.1f}{marks['OMR-NED']} "
                     f"| {round(100 * n_success / n_total)} ({n_success}/{n_total}) |")
    lines += ["", "Marks on our rows: *: significant improvement over the cascade; †: significant "
                  "improvement over every other row (paired work bootstrap, 95% interval excludes zero)."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"name": args.name, "grid": str(args.grid), "rows": cells,
         "wins": {f"{c}>{r}:{m}": v for (c, r, m), v in wins.items()}}, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
