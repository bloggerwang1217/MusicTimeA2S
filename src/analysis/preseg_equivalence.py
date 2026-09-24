#!/usr/bin/env python3
"""Official multi-path MV2H versus one fixed alignment on the Table 1 windows.

For every window of the Table 1 intersection and every declared system, the
MIDI pair the official scoring used is converted exactly as that scoring
converted it (with that system's MV2H copy), then scored by `WholeScoreMV2H`
(one minimum-cost alignment, ties broken diagonal, up, left) compiled against
the same copy. `--rerun-official` also repeats `mv2h.Main -a` so its agreement
with the stored score shows the inputs match. `mv2h_agreement.py` reads the
output directory.

    prepare --out DIR --grounding G [--work-list L | --recording-list L [--manifest M]]
            --system NAME=SCORES ... --piano-a2s-results DIR
    run     --out DIR [--workers N] [--rerun-official]
    collect --out DIR

SCORES is a per-window evaluator CSV (an Ours arm; its pred_path and gt_path
name the MIDI pair) or a directory of Piano-A2S `<chunk_id>_mv2h.json` files
(its MIDI pair is found under --piano-a2s-results). A system with several
training seeds is declared once per seed under distinct names.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.analysis.paired_work_bootstrap import (
    METRICS, load_grounding, load_system, load_work_list,
)

JAVA_SOURCE = Path(__file__).resolve().with_name("WholeScoreMV2H.java")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classes_dir(out: Path, mv2h_bin: Path) -> Path:
    return out / "classes" / hashlib.sha256(str(mv2h_bin.resolve()).encode()).hexdigest()[:12]


def select_windows(args) -> list[dict]:
    rows = load_grounding(args.grounding)
    if args.work_list is not None:
        works = load_work_list(args.work_list)
        rows = [row for row in rows if row["piece_id"] in works]
    if args.recording_list is not None:
        keep = load_work_list(args.recording_list)
        ident = {}
        if args.manifest is not None:
            ident = {f"{m['piece_id']}#{m['performance_id']}": m["id"]
                     for m in json.loads(args.manifest.read_text())}
        rows = [row for row in rows if ident.get(row["recording_id"], row["recording_id"]) in keep]
    if not rows:
        raise SystemExit("the selection leaves no grounding windows")
    return rows


def prepare(args) -> None:
    rows = select_windows(args)
    ids = [row["chunk_id"] for row in rows]
    common = set(ids)
    for _, source in args.system:
        common &= load_system(source, ids)[2]
    windows = [row for row in rows if row["chunk_id"] in common]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "grounding_intersection.jsonl").open("w") as handle:
        for row in windows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    pa2s_midi = {}
    if args.piano_a2s_results is not None:
        for kind in ("pred", "target"):
            for path in (args.piano_a2s_results / "tasks").glob(f"*/midi/{kind}/*_{kind}.mid"):
                pa2s_midi[(kind, path.name.removesuffix(f"_{kind}.mid"))] = path
    tasks = []
    for system, source in args.system:
        if source.is_dir():
            if not pa2s_midi:
                raise SystemExit(f"{system} is a Piano-A2S score directory; pass --piano-a2s-results")
            for chunk_id in sorted(common):
                tasks.append({
                    "system": system, "chunk_id": chunk_id, "mv2h_bin": str(args.piano_a2s_mv2h_bin),
                    "gt_midi": str(pa2s_midi[("target", chunk_id)]),
                    "pred_midi": str(pa2s_midi[("pred", chunk_id)]), "anacrusis": None,
                    "official": {k: json.loads((source / f"{chunk_id}_mv2h.json").read_text())[k]
                                 for k in METRICS},
                    "official_status": "json",
                })
            continue
        with source.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["task_id"] not in common:
                    continue
                pred = Path(row["pred_path"])
                sidecar = pred.with_suffix(".json")
                anacrusis = (json.loads(sidecar.read_text()).get("anacrusis_subbeats")
                             if sidecar.is_file() else None)
                tasks.append({
                    "system": system, "chunk_id": row["task_id"], "mv2h_bin": str(args.mv2h_bin),
                    "gt_midi": row["gt_path"], "pred_midi": str(pred), "anacrusis": anacrusis,
                    "official": {k: float(row[k] or 0.0) for k in METRICS},
                    "official_status": row["status"],
                })
    counts = {s: sum(t["system"] == s for t in tasks) for s, _ in args.system}
    if set(counts.values()) != {len(common)}:
        raise SystemExit(f"task counts {counts} != {len(common)} windows")
    with (out / "tasks.jsonl").open("w") as handle:
        for task in tasks:
            handle.write(json.dumps(task, sort_keys=True) + "\n")
    print(json.dumps({"windows": len(common), "tasks": len(tasks),
                      "recordings": len({r["recording_id"] for r in windows}),
                      "works": len({r["piece_id"] for r in windows})}))


def parse_scores(text: str) -> dict[str, float]:
    found = dict(re.findall(r"^(Multi-pitch|Voice|Meter|Value|Harmony|MV2H): ([-0-9.Ee]+)$",
                            text, flags=re.M))
    if set(found) != {*METRICS, "MV2H"}:
        raise ValueError(f"unparsed MV2H output: {text[-300:]!r}")
    return {k: float(v) for k, v in found.items()}


def score(task: dict, out: Path, rerun_official: bool) -> dict:
    bin_dir = Path(task["mv2h_bin"])
    result = {"system": task["system"], "chunk_id": task["chunk_id"]}
    with tempfile.TemporaryDirectory(dir=os.environ.get("EQUIV_SCRATCH")) as work:
        gt_conv, pred_conv = Path(work, "gt.conv.txt"), Path(work, "pred.conv.txt")
        convert = ["java", "-cp", str(bin_dir), "mv2h.tools.Converter", "-i"]
        extra = [] if task["anacrusis"] is None else ["-a", str(int(task["anacrusis"]))]
        # A MIDI the converter rejects is this window's own result, not a reason
        # to lose the whole run; collect counts it and drops the window.
        for source, target, more in ((task["gt_midi"], gt_conv, []),
                                     (task["pred_midi"], pred_conv, extra)):
            with target.open("w") as handle:
                done = subprocess.run(convert + [source] + more, stdout=handle, stderr=subprocess.PIPE)
            if done.returncode != 0:
                result["fixed"], result["fixed_status"] = None, "convert_failed"
                result["fixed_error"] = done.stderr.decode("utf-8", "replace")[-300:]
                result["official_rerun"], result["tied_alignments"] = None, None
                result["official_rerun_skipped"] = True
                return result
        result["input_sha256"] = {
            "gt_midi": sha256(Path(task["gt_midi"])), "pred_midi": sha256(Path(task["pred_midi"])),
            "gt_conv": sha256(gt_conv), "pred_conv": sha256(pred_conv),
        }
        fixed = subprocess.run(
            ["java", "-cp", f"{bin_dir}:{classes_dir(out, bin_dir)}", "WholeScoreMV2H",
             str(gt_conv), str(pred_conv)], capture_output=True, text=True)
        if fixed.returncode == 0:
            result["fixed"] = parse_scores(fixed.stdout)
            result["fixed_status"] = "success"
        elif "Empty score" in fixed.stderr:
            result["fixed"], result["fixed_status"] = None, "empty_score"
        else:
            result["fixed"], result["fixed_status"] = None, "error"
            result["fixed_error"] = fixed.stderr[-500:]
        if not rerun_official:
            result["official_rerun"] = None
            result["official_rerun_skipped"] = True
            result["tied_alignments"] = None
            return result
        # The official run prints one progress line per tied alignment, so its
        # output goes to a file; the last progress line gives the tie count.
        log = Path(work, "official.out")
        with log.open("w") as handle:
            official = subprocess.run(
                ["java", "-cp", str(bin_dir), "mv2h.Main", "-g", str(gt_conv), "-t", str(pred_conv), "-a"],
                stdout=handle, stderr=subprocess.PIPE, text=True)
        text = log.read_text()
        totals = re.findall(r"Evaluating alignment \d+ / (\d+)", text[-4000:])
        result["tied_alignments"] = int(totals[-1]) if totals else None
        if official.returncode == 0:
            result["official_rerun"] = parse_scores(text)
        else:
            result["official_rerun"], result["official_error"] = None, official.stderr[-500:]
    return result


def run(args) -> None:
    out = args.out
    tasks = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    for bin_dir in {Path(t["mv2h_bin"]) for t in tasks}:
        target = classes_dir(out, bin_dir)
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["javac", "-cp", str(bin_dir), "-d", str(target), str(JAVA_SOURCE)], check=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.is_file():
        done = {(r["system"], r["chunk_id"]) for r in map(json.loads, results_path.read_text().splitlines())}
    todo = [t for t in tasks if (t["system"], t["chunk_id"]) not in done]
    print(f"{len(done)} done, {len(todo)} to score", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, results_path.open("a") as handle:
        futures = [pool.submit(score, t, out, args.rerun_official) for t in todo]
        for index, future in enumerate(as_completed(futures), 1):
            handle.write(json.dumps(future.result(), sort_keys=True) + "\n")
            handle.flush()
            if index % 500 == 0:
                print(f"{index}/{len(todo)}", flush=True)


def collect(args) -> None:
    out = args.out
    tasks = {(t["system"], t["chunk_id"]): t
             for t in map(json.loads, (out / "tasks.jsonl").read_text().splitlines())}
    results = {(r["system"], r["chunk_id"]): r
               for r in map(json.loads, (out / "results.jsonl").read_text().splitlines())}
    if set(results) != set(tasks):
        raise SystemExit(f"{len(results)} results for {len(tasks)} tasks")
    checks = {}
    for system in sorted({s for s, _ in tasks}):
        keys = sorted(k for k in tasks if k[0] == system)
        rerun_mismatch, above_official, statuses, rerun_skipped = [], [], {}, 0
        slug = re.sub(r"[^A-Za-z0-9]+", "_", system).strip("_").lower()
        with (out / f"fixed_{slug}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task_id", "status", *METRICS, "tied_alignments"])
            writer.writeheader()
            for key in keys:
                task, result = tasks[key], results[key]
                official = task["official"]
                rerun = result.get("official_rerun")
                if result.get("official_rerun_skipped"):
                    rerun_skipped += 1
                elif rerun is None or any(abs(rerun[m] - official[m]) > 1e-9 for m in METRICS):
                    rerun_mismatch.append(key[1])
                status = result["fixed_status"]
                # An empty score is MV2H's zero; the official scoring kept it the same way.
                fixed = result["fixed"] or ({m: 0.0 for m in METRICS} if status == "empty_score" else None)
                statuses[status] = statuses.get(status, 0) + 1
                if fixed is not None and sum(fixed[m] for m in METRICS) / 5 > sum(official[m] for m in METRICS) / 5 + 1e-12:
                    above_official.append(key[1])
                writer.writerow({
                    "task_id": key[1],
                    "status": {"success": "success", "empty_score": "zero_score"}.get(status, "error"),
                    **({m: fixed[m] for m in METRICS} if fixed else {m: "" for m in METRICS}),
                    "tied_alignments": result.get("tied_alignments"),
                })
        checks[system] = {
            "windows": len(keys), "fixed_status": statuses,
            "official_rerun_skipped": rerun_skipped,
            "official_rerun_mismatch": len(rerun_mismatch), "rerun_mismatch_first": rerun_mismatch[:10],
            "fixed_mv2h5_above_official": len(above_official), "above_first": above_official[:10],
        }
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


def parse_system(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("system spec must be NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "run", "collect"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grounding", type=Path)
    parser.add_argument("--work-list", type=Path)
    parser.add_argument("--recording-list", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--system", type=parse_system, action="append", default=[])
    parser.add_argument("--piano-a2s-results", type=Path)
    parser.add_argument("--mv2h-bin", type=Path, default=Path("external/MV2H/bin"))
    parser.add_argument("--piano-a2s-mv2h-bin", type=Path,
                        help="the MV2H copy Piano-A2S scored with (its own repository's MV2H/bin)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    parser.add_argument("--rerun-official", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.grounding is None or not args.system:
            parser.error("prepare needs --grounding and at least one --system")
        if any(p.is_dir() for _, p in args.system) and args.piano_a2s_mv2h_bin is None:
            parser.error("a Piano-A2S score directory needs --piano-a2s-mv2h-bin")
    {"prepare": prepare, "run": run, "collect": collect}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
