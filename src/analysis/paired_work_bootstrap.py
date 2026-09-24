#!/usr/bin/env python3
"""Paired, work-clustered bootstrap for the fixed five-bar-window MV2H gate.

Windows from the same work are not independent draws, so resampling windows
understates the variance of a system difference.  This script keeps the
protocol statistic (the pooled mean over every window in the fixed grounding,
failures scored zero) and bootstraps it by resampling *works* with
replacement.  Per-work win/loss counts accompany every interval because a
large mean difference driven by one work is not a decisive win.

A system trained with several seeds is declared as a comma-separated list of
score sources.  Its score is the mean over seeds of the pooled window mean, and
every bootstrap draw resamples its seeds with replacement, independently of the
work draw and of every other system, so the interval also carries training
randomness.  A single-source system is unchanged.

Usage:
    poetry run python -m src.analysis.paired_work_bootstrap \
        --grounding grounding.jsonl \
        --system A=/path/eval_asap.csv \
        --system C=/seed42/eval_asap.csv,/seed91/eval_asap.csv \
        --system Piano-A2S=/path/to/mv2h_json_dir \
        --candidate C --reference A --reference Piano-A2S \
        --out data/experiments/.../bootstrap
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("Multi-pitch", "Voice", "Meter", "Value", "Harmony")
REPORTED = (*METRICS, "MV2H4", "MV2H5")


def load_grounding(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    ids = [row["chunk_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("grounding contains duplicate chunk_id values")
    return rows


def metric_row(raw: dict | None) -> dict[str, float]:
    values = {
        name: float(raw.get(name) or 0.0) if raw is not None else 0.0
        for name in METRICS
    }
    values["MV2H4"] = sum(values[name] for name in METRICS if name != "Meter") / 4
    values["MV2H5"] = sum(values[name] for name in METRICS) / 5
    return values


def load_work_list(path: Path) -> set[str]:
    """Work ids from a one-column list with a `name` header (ASAP list format)."""
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names or names[0] != "name":
        raise RuntimeError(f"work list must start with a 'name' header: {path}")
    return set(names[1:])


def load_system(
    path: Path, ids: list[str]
) -> tuple[dict[str, dict[str, float]], dict, set[str]]:
    """Return per-window metric rows keyed by chunk_id, coverage counts and the
    set of windows the system actually evaluated.

    A CSV is a per-window evaluator CSV; a directory holds Piano-A2S style
    `<chunk_id>_mv2h.json` files, one per window its own evaluator scored.
    A window is evaluated when its JSON exists (directory) or its CSV status is
    `success` or `zero_score` (MV2H returned a score, even an all-zero one);
    under `--windows all` every other window scores zero, under `--windows
    intersection` it is dropped for every system.
    """
    coverage = {"present": 0, "absent": 0, "status": defaultdict(int)}
    rows: dict[str, dict[str, float]] = {}
    evaluated: set[str] = set()
    if path.is_dir():
        for chunk_id in ids:
            file = path / f"{chunk_id}_mv2h.json"
            raw = json.loads(file.read_text()) if file.is_file() else None
            coverage["present" if raw is not None else "absent"] += 1
            if raw is not None:
                evaluated.add(chunk_id)
            rows[chunk_id] = metric_row(raw)
        coverage["status"] = {}
        return rows, coverage, evaluated
    with path.open(newline="") as handle:
        raw_rows = {}
        for row in csv.DictReader(handle):
            if row["task_id"] in raw_rows:
                raise RuntimeError(f"duplicate evaluation row: {row['task_id']}")
            raw_rows[row["task_id"]] = row
    for chunk_id in ids:
        raw = raw_rows.get(chunk_id)
        if raw is None:
            coverage["absent"] += 1
        else:
            coverage["present"] += 1
            coverage["status"][raw.get("status", "")] += 1
            if raw.get("status", "") in ("success", "zero_score"):
                evaluated.add(chunk_id)
        rows[chunk_id] = metric_row(raw)
    coverage["status"] = dict(coverage["status"])
    return rows, coverage, evaluated


def compare_work_ratios(
    candidate: np.ndarray, reference: np.ndarray, n_boot: int, seed: int,
) -> dict:
    """Inputs contain numerator and denominator totals in the same work order."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (candidate.shape != reference.shape or candidate.ndim != 2
            or candidate.shape[1] != 2 or len(candidate) == 0 or n_boot < 2):
        raise ValueError("Expected matching nonempty work-by-two totals and at least two draws")
    for totals in (candidate, reference):
        if (not np.isfinite(totals).all() or np.any(totals < 0)
                or totals[:, 1].sum() <= 0
                or np.any((totals[:, 1] == 0) & (totals[:, 0] != 0))):
            raise ValueError("Invalid score totals or no successful observations")
    draws = np.random.default_rng(seed).integers(len(candidate), size=(n_boot, len(candidate)))
    candidate_draws = candidate[draws].sum(axis=1)
    reference_draws = reference[draws].sum(axis=1)
    if np.any(candidate_draws[:, 1] == 0) or np.any(reference_draws[:, 1] == 0):
        raise ValueError("A bootstrap draw has no successful observations; its score is undefined")
    candidate_scores = candidate_draws[:, 0] / candidate_draws[:, 1]
    reference_scores = reference_draws[:, 0] / reference_draws[:, 1]
    differences = candidate_scores - reference_scores
    candidate_mean = float(candidate[:, 0].sum() / candidate[:, 1].sum())
    reference_mean = float(reference[:, 0].sum() / reference[:, 1].sum())
    lo, hi = np.percentile(differences, [2.5, 97.5])
    return {
        "candidate_mean": candidate_mean, "reference_mean": reference_mean,
        "diff": candidate_mean - reference_mean,
        "ci95": [float(lo), float(hi)],
        "candidate_ci95": np.percentile(candidate_scores, [2.5, 97.5]).tolist(),
        "reference_ci95": np.percentile(reference_scores, [2.5, 97.5]).tolist(),
        "ci_lower_gt_zero": bool(lo > 0), "ci_upper_lt_zero": bool(hi < 0),
        "bootstrap_sd": float(differences.std(ddof=1)),
    }


def compare_work_ratio_seeds(
    candidate: list[np.ndarray], reference: list[np.ndarray], n_boot: int, seed: int,
) -> dict:
    """`compare_work_ratios` when a system carries several training seeds.

    Each system is a list of work-by-two totals, one per seed. The work draw is
    taken first and shared, so a system with one seed sees exactly the draw
    `compare_work_ratios` would have given it; seed draws are taken only for
    systems that have more than one, candidate before reference. A seed keeps
    its own denominator, since a seed that fails a recording does not score it.
    """
    if not candidate or not reference:
        raise ValueError("Each system needs at least one seed")
    shapes = {tuple(np.asarray(t).shape) for t in (*candidate, *reference)}
    if len(shapes) != 1:
        raise ValueError("Every seed must cover the same works")
    n_works = len(candidate[0])
    if n_works == 0 or n_boot < 2:
        raise ValueError("Expected nonempty works and at least two draws")
    rng = np.random.default_rng(seed)
    draws = rng.integers(n_works, size=(n_boot, n_works))
    scores = []
    for system in (candidate, reference):
        per_seed = []
        for totals in system:
            totals = np.asarray(totals, dtype=float)
            drawn = totals[draws].sum(axis=1)
            if np.any(drawn[:, 1] == 0):
                raise ValueError("A bootstrap draw has no successful observations")
            per_seed.append(drawn[:, 0] / drawn[:, 1])
        per_seed = np.stack(per_seed)
        if len(system) > 1:
            pick = rng.integers(0, len(system), size=(n_boot, len(system)))
            per_seed = np.take_along_axis(per_seed.T, pick, axis=1).T
        scores.append(per_seed.mean(axis=0))
    candidate_scores, reference_scores = scores
    differences = candidate_scores - reference_scores
    mean = lambda system: float(np.mean([np.asarray(t)[:, 0].sum() / np.asarray(t)[:, 1].sum()
                                         for t in system]))
    candidate_mean, reference_mean = mean(candidate), mean(reference)
    lo, hi = np.percentile(differences, [2.5, 97.5])
    return {
        "candidate_mean": candidate_mean, "reference_mean": reference_mean,
        "diff": candidate_mean - reference_mean,
        "ci95": [float(lo), float(hi)],
        "candidate_ci95": np.percentile(candidate_scores, [2.5, 97.5]).tolist(),
        "reference_ci95": np.percentile(reference_scores, [2.5, 97.5]).tolist(),
        "ci_lower_gt_zero": bool(lo > 0), "ci_upper_lt_zero": bool(hi < 0),
        "bootstrap_sd": float(differences.std(ddof=1)),
        "candidate_seeds": len(candidate), "reference_seeds": len(reference),
    }


def cluster_bootstrap(
    diff: np.ndarray,
    work_index: np.ndarray,
    n_works: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pooled-mean difference under work resampling with replacement."""
    work_sum = np.bincount(work_index, weights=diff, minlength=n_works)
    work_count = np.bincount(work_index, minlength=n_works).astype(float)
    draws = rng.integers(0, n_works, size=(n_boot, n_works))
    return work_sum[draws].sum(axis=1) / work_count[draws].sum(axis=1)


def seed_cluster_bootstrap(
    systems: list[np.ndarray],
    work_index: np.ndarray,
    n_works: int,
    n_boot: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Pooled means of several systems on one shared work draw, each system's
    seeds resampled with replacement on top.

    Each entry of `systems` is a (seeds × windows) array.  The work draw is
    taken first so a single-seed system sees exactly the draw
    `cluster_bootstrap` would have given it; seed draws are only taken for
    systems with more than one seed, in `systems` order.
    """
    work_count = np.bincount(work_index, minlength=n_works).astype(float)
    draws = rng.integers(0, n_works, size=(n_boot, n_works))
    count = work_count[draws].sum(axis=1)
    scores = []
    for seeds in systems:
        work_sum = np.stack(
            [np.bincount(work_index, weights=row, minlength=n_works) for row in seeds]
        )
        per_seed = work_sum[:, draws].sum(axis=2)  # seeds × n_boot
        if len(seeds) > 1:
            pick = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
            per_seed = np.take_along_axis(per_seed.T, pick, axis=1).T
        scores.append(per_seed.mean(axis=0) / count)
    return scores


def summarize_pair(
    cand: np.ndarray,
    ref: np.ndarray,
    work_index: np.ndarray,
    works: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Inputs contain window scores, optionally with a leading seed axis.

    Single-seed pairs use window differences to preserve numerical precision.
    """
    cand, ref = np.atleast_2d(cand), np.atleast_2d(ref)
    rng = np.random.default_rng(seed)
    if len(cand) == 1 and len(ref) == 1:
        cand, ref = cand[0], ref[0]
        diff = cand - ref
        boot = cluster_bootstrap(diff, work_index, len(works), n_boot, rng)
    else:
        cand_boot, ref_boot = seed_cluster_bootstrap(
            [cand, ref], work_index, len(works), n_boot, rng
        )
        boot = cand_boot - ref_boot
        cand, ref = cand.mean(axis=0), ref.mean(axis=0)
        diff = cand - ref
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per_work = {}
    wins = losses = ties = 0
    for i, work in enumerate(works):
        mask = work_index == i
        mean_diff = float(diff[mask].mean())
        per_work[work] = {
            "windows": int(mask.sum()),
            "candidate": float(cand[mask].mean()),
            "reference": float(ref[mask].mean()),
            "diff": mean_diff,
        }
        if abs(mean_diff) < 1e-12:
            ties += 1
        elif mean_diff > 0:
            wins += 1
        else:
            losses += 1
    return {
        "candidate_mean": float(cand.mean()),
        "reference_mean": float(ref.mean()),
        "diff": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "ci_lower_gt_zero": bool(lo > 0),
        "ci_upper_lt_zero": bool(hi < 0),
        "bootstrap_sd": float(boot.std(ddof=1)),
        "works_win": wins,
        "works_loss": losses,
        "works_tie": ties,
        "per_work": per_work,
    }


def parse_system(spec: str) -> tuple[str, list[Path]]:
    name, _, paths = spec.partition("=")
    if not name or not paths:
        raise argparse.ArgumentTypeError("system spec must be NAME=PATH[,PATH...]")
    return name, [Path(p) for p in paths.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--system", type=parse_system, action="append", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--work-list", type=Path, default=None,
        help="keep only grounding rows whose piece_id is in this `name`-header list",
    )
    parser.add_argument(
        "--recording-list", type=Path, default=None,
        help="keep only grounding rows whose recording is in this `name`-header list",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest whose `id` spells the recordings of --recording-list when the "
        "grounding names them piece_id#performance_id",
    )
    parser.add_argument(
        "--windows", choices=("all", "intersection"), default="all",
        help="all: every grounding window, unevaluated windows score zero; "
        "intersection: only windows every declared system evaluated",
    )
    args = parser.parse_args()

    grounding = load_grounding(args.grounding)
    if args.work_list is not None:
        keep = load_work_list(args.work_list)
        grounding = [row for row in grounding if row["piece_id"] in keep]
        if not grounding:
            raise SystemExit("work list leaves no grounding windows")
    if args.recording_list is not None:
        keep = load_work_list(args.recording_list)
        ident: dict[str, str] = {}
        if args.manifest is not None:
            ident = {
                f"{m['piece_id']}#{m['performance_id']}": m["id"]
                for m in json.loads(args.manifest.read_text())
            }
        grounding = [
            row for row in grounding
            if ident.get(row["recording_id"], row["recording_id"]) in keep
        ]
        if not grounding:
            raise SystemExit("recording list leaves no grounding windows")
    ids = [row["chunk_id"] for row in grounding]

    systems = dict(args.system)
    for name in (args.candidate, *args.reference):
        if name not in systems:
            raise SystemExit(f"unknown system {name!r}; declare it with --system")

    # A system is a list of seed sources; a one-source system keeps the
    # original coverage record, a multi-seed one records every source.
    loaded_rows = {}
    coverage = {}
    evaluated_by = {}
    for name, paths in systems.items():
        per_seed = [load_system(path, ids) for path in paths]
        loaded_rows[name] = [rows for rows, _, _ in per_seed]
        records = [{"path": str(path), **cov} for path, (_, cov, _) in zip(paths, per_seed)]
        coverage[name] = records[0] if len(records) == 1 else {"seeds": records}
        # Under intersection a window must be evaluated by every seed as well.
        evaluated_by[name] = set.intersection(*(ev for _, _, ev in per_seed))
    windows_before = len(ids)
    if args.windows == "intersection":
        common = set.intersection(*evaluated_by.values())
        grounding = [row for row in grounding if row["chunk_id"] in common]
        ids = [row["chunk_id"] for row in grounding]
        if not ids:
            raise SystemExit("no window is evaluated by every system")
    works = sorted({row["piece_id"] for row in grounding})
    work_pos = {work: i for i, work in enumerate(works)}
    work_index = np.array([work_pos[row["piece_id"]] for row in grounding])
    recordings = {row["recording_id"] for row in grounding}
    loaded = {
        name: {
            metric: np.array(
                [[rows[chunk_id][metric] for chunk_id in ids] for rows in seed_rows]
            )
            for metric in REPORTED
        }
        for name, seed_rows in loaded_rows.items()
    }
    n_seeds = {name: len(paths) for name, paths in systems.items()}
    multi_seed = any(n > 1 for n in n_seeds.values())

    means = {
        name: {metric: float(arr[metric].mean()) for metric in REPORTED}
        for name, arr in loaded.items()
    }
    # Each system's own interval under the same work resampling; a fresh
    # generator per cell keeps every cell independent of report order.
    system_ci = {}
    for name, arr in loaded.items():
        system_ci[name] = {}
        for metric in REPORTED:
            if len(arr[metric]) == 1:
                draws = cluster_bootstrap(
                    arr[metric][0], work_index, len(works), args.n_boot,
                    np.random.default_rng(args.seed),
                )
            else:
                (draws,) = seed_cluster_bootstrap(
                    [arr[metric]], work_index, len(works), args.n_boot,
                    np.random.default_rng(args.seed),
                )
            lo, hi = np.percentile(draws, [2.5, 97.5])
            system_ci[name][metric] = {
                "mean": means[name][metric], "ci95": [float(lo), float(hi)],
            }
    comparisons = {}
    for ref in args.reference:
        comparisons[ref] = {
            metric: summarize_pair(
                loaded[args.candidate][metric],
                loaded[ref][metric],
                work_index,
                works,
                args.n_boot,
                args.seed,
            )
            for metric in REPORTED
        }

    payload = {
        "protocol": {
            "grounding": str(args.grounding),
            "work_list": str(args.work_list) if args.work_list else None,
            "recording_list": str(args.recording_list) if args.recording_list else None,
            "manifest": str(args.manifest) if args.manifest else None,
            "windows_mode": args.windows,
            "windows_in_set": windows_before,
            "windows": len(ids),
            "works": len(works),
            "recordings": len(recordings),
            "failures": (
                "missing/non-success rows contribute zero" if args.windows == "all"
                else "windows not evaluated by every system are dropped for all"
            ),
            "statistic": (
                "pooled window mean; works resampled with replacement"
                + ("; per-system mean over training seeds, seeds resampled "
                   "with replacement independently per system" if multi_seed else "")
            ),
            **({"training_seeds": n_seeds} if multi_seed else {}),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "ci": "percentile 2.5/97.5",
        },
        "coverage": coverage,
        "means": means,
        "system_ci": system_ci,
        "candidate": args.candidate,
        "comparisons": comparisons,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paired_work_bootstrap.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    lines = []
    lines.append(
        f"windows: {args.windows}, {len(ids)} of {windows_before} in set; "
        f"works {len(works)}, recordings {len(recordings)}"
    )
    lines.append("")
    lines.append("| system | " + " | ".join(REPORTED) + " |")
    lines.append("|---|" + "---:|" * len(REPORTED))
    for name in systems:
        lines.append(
            f"| {name} | " + " | ".join(f"{means[name][m]:.4f}" for m in REPORTED) + " |"
        )
    lines.append("")
    lines.append("| system | " + " | ".join(f"{m} 95% CI" for m in REPORTED) + " |")
    lines.append("|---|" + "---:|" * len(REPORTED))
    for name in systems:
        lines.append(
            f"| {name} | " + " | ".join(
                f"[{system_ci[name][m]['ci95'][0]:.4f}, {system_ci[name][m]['ci95'][1]:.4f}]"
                for m in REPORTED
            ) + " |"
        )
    lines.append("")
    for ref, table in comparisons.items():
        lines.append(f"**{args.candidate} − {ref}** (paired, work-cluster bootstrap)")
        lines.append("")
        lines.append("| metric | diff | 95% CI | lower>0 | works W/L/T |")
        lines.append("|---|---:|---:|:---:|---:|")
        for metric in REPORTED:
            row = table[metric]
            lines.append(
                f"| {metric} | {row['diff']:+.4f} | "
                f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | "
                f"{'yes' if row['ci_lower_gt_zero'] else 'no'} | "
                f"{row['works_win']}/{row['works_loss']}/{row['works_tie']} |"
            )
        lines.append("")
    report = "\n".join(lines)
    (args.out / "paired_work_bootstrap.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
