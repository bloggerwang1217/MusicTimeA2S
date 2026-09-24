#!/usr/bin/env python3
"""Corpus OMR-NED of a recording subset from per-recording results.

Inputs are either our ``results.jsonl`` (from ``omr_ned_shard.py``; pass
``--pairs`` too so ``summarize_omr_ned`` can count failed outputs on the
subset) or a musicdiff ``output.csv`` (baseline runs).  The subset is a
``name``-header list of recording stems, e.g.
``src/datasets/asap/asap102_hft_clean_74_recordings.txt``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_list(path: Path) -> set[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names or names[0] != "name":
        raise SystemExit(f"list must start with a 'name' header: {path}")
    return set(names[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--results", type=Path, help="our results.jsonl")
    parser.add_argument("--pairs", type=Path, help="our pairs.jsonl (filtered copy is written)")
    parser.add_argument("--csv", type=Path, help="musicdiff output.csv of a baseline")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    keep = load_list(args.recordings)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.csv:
        edit = symbols = 0
        present = 0
        for row in csv.DictReader(args.csv.open()):
            stem = Path(row[" gtpath"].strip()).name.replace(".musicxml", "")
            if stem not in keep or not row[" OMR-ED (OMR Edit Distance)"].strip():
                continue
            present += 1
            edit += int(float(row[" OMR-ED (OMR Edit Distance)"]))
            symbols += int(float(row[" total numsyms (in both scores)"]))
        summary = {
            "recordings_in_list": len(keep),
            "recordings_present": present,
            "corpus_OMR-NED_percent": 100.0 * edit / symbols if symbols else None,
            "note": "baseline csv: recordings absent from the csv are excluded, not scored 1",
        }
        (args.out_dir / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        return
    if not (args.results and args.pairs):
        raise SystemExit("give --results and --pairs, or --csv")
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    sub_rows = [r for r in rows if r["task_id"] in keep]
    sub_pairs = [p for p in pairs if p["artifact_stem"] in keep]
    (args.out_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sub_rows))
    (args.out_dir / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in sub_pairs))
    print(f"{len(sub_rows)} results, {len(sub_pairs)} pairs written to {args.out_dir}; now run "
          f"summarize_omr_ned on them")


if __name__ == "__main__":
    main()
