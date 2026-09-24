#!/usr/bin/env python3
"""System-independent OMR-NED evaluation through musicdiff."""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MUSICDIFF_ROOT = (
    Path(__file__).resolve().parents[2] / "external" / "efficient-musicdiff"
)


@dataclass(frozen=True)
class OMRNEDResult:
    """Notation edit distance and its normalization for one score pair."""

    edit_distance: int
    pred_symbols: int
    gt_symbols: int
    omr_ned: float

    @property
    def total_symbols(self) -> int:
        return self.pred_symbols + self.gt_symbols

    def to_dict(self) -> dict[str, int | float]:
        return {
            "OMR-ED": self.edit_distance,
            "pred_symbols": self.pred_symbols,
            "gt_symbols": self.gt_symbols,
            "total_symbols": self.total_symbols,
            "OMR-NED": self.omr_ned,
        }


@dataclass(frozen=True)
class ScorePairTask:
    """One prediction/reference score pair."""

    task_id: str
    pred_score: str
    gt_score: str
    musicdiff_root: str = str(DEFAULT_MUSICDIFF_ROOT)


@dataclass
class EvaluationResult:
    """Status and optional metric for one score pair."""

    task_id: str
    pred_path: str
    gt_path: str
    status: str
    metrics: OMRNEDResult | None = None
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "pred_path": self.pred_path,
            "gt_path": self.gt_path,
            "status": self.status,
            "error_message": self.error_message,
        }
        if self.metrics is not None:
            result.update(self.metrics.to_dict())
        return result


@dataclass(frozen=True)
class FolderResult:
    """One row from musicdiff folder evaluation."""

    pred_path: str
    gt_path: str
    metrics: OMRNEDResult


def _load_musicdiff(root: Path):
    package = root / "musicdiff" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(
            f"musicdiff submodule is not available at {root}; "
            "run git submodule update --init external/efficient-musicdiff"
        )

    root_text = str(root.resolve())
    loaded = sys.modules.get("musicdiff")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if root.resolve() not in loaded_path.parents:
            raise RuntimeError(
                f"A different musicdiff package is already loaded: {loaded_path}"
            )
        return loaded

    sys.path.insert(0, root_text)
    try:
        return importlib.import_module("musicdiff")
    finally:
        if sys.path and sys.path[0] == root_text:
            sys.path.pop(0)


class OMRNEDEvaluator:
    """Evaluate score notation with the vendored musicdiff implementation."""

    def __init__(self, musicdiff_root: str | Path = DEFAULT_MUSICDIFF_ROOT):
        self.musicdiff_root = Path(musicdiff_root).resolve()
        self._musicdiff = _load_musicdiff(self.musicdiff_root)

    def evaluate(
        self,
        gt_score_path: str | Path,
        pred_score_path: str | Path,
    ) -> OMRNEDResult:
        gt_path = Path(gt_score_path).resolve()
        pred_path = Path(pred_score_path).resolve()
        if not gt_path.is_file():
            raise FileNotFoundError(f"Ground-truth score does not exist: {gt_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"Predicted score does not exist: {pred_path}")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cost = self._musicdiff.diff(
                str(pred_path),
                str(gt_path),
                visualize_diffs=False,
                print_omr_ned_output=True,
            )
        if cost is None:
            raise RuntimeError("musicdiff returned no edit distance")

        try:
            payload = json.loads(output.getvalue())
            result = OMRNEDResult(
                edit_distance=int(payload["OMR-ED"]),
                pred_symbols=int(payload["numSymbolsInPredicted"]),
                gt_symbols=int(payload["numSymbolsInGroundTruth"]),
                omr_ned=float(payload["OMR-NED"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Unexpected musicdiff output: {output.getvalue()!r}"
            ) from error
        if result.edit_distance != int(cost):
            raise RuntimeError(
                "musicdiff edit-distance outputs disagree: "
                f"return={cost} json={result.edit_distance}"
            )
        return result

    def evaluate_folders(
        self,
        gt_folder: str | Path,
        pred_folder: str | Path,
        output_folder: str | Path,
    ) -> tuple[float, Path]:
        gt_root = Path(gt_folder).resolve()
        pred_root = Path(pred_folder).resolve()
        output_root = Path(output_folder).resolve()
        for path, label in ((gt_root, "GT"), (pred_root, "prediction")):
            if not path.is_dir():
                raise NotADirectoryError(f"{label} folder does not exist: {path}")

        gt_names = {path.name for path in gt_root.iterdir() if path.is_file()}
        pred_names = {path.name for path in pred_root.iterdir() if path.is_file()}
        if gt_names != pred_names:
            missing = sorted(gt_names - pred_names)
            extra = sorted(pred_names - gt_names)
            raise ValueError(
                "Score-folder inventories differ: "
                f"missing_predictions={missing[:10]} extra_predictions={extra[:10]}"
            )
        if not gt_names:
            raise ValueError("Score folders are empty")

        output_root.mkdir(parents=True, exist_ok=True)
        overall, output_path = self._musicdiff.diff_ml_training(
            predicted_folder=str(pred_root),
            ground_truth_folder=str(gt_root),
            output_folder=str(output_root),
        )
        return float(overall), Path(output_path)


def evaluate_score_pair(task: ScorePairTask) -> EvaluationResult:
    """Evaluate one score pair while retaining an explicit status."""
    try:
        metrics = OMRNEDEvaluator(task.musicdiff_root).evaluate(
            task.gt_score,
            task.pred_score,
        )
        return EvaluationResult(
            task_id=task.task_id,
            pred_path=task.pred_score,
            gt_path=task.gt_score,
            status="success",
            metrics=metrics,
        )
    except Exception as error:
        logger.debug("OMR-NED evaluation failed: %s", error)
        return EvaluationResult(
            task_id=task.task_id,
            pred_path=task.pred_score,
            gt_path=task.gt_score,
            status="error",
            error_message=str(error),
        )


def aggregate_omr_ned_results(
    results: list[OMRNEDResult],
) -> dict[str, int | float]:
    """Report macro and symbol-weighted OMR-NED without choosing failure policy."""
    if not results:
        return {
            "n_samples": 0,
            "mean_OMR-NED": 0.0,
            "corpus_OMR-NED": 0.0,
            "total_OMR-ED": 0,
            "total_symbols": 0,
        }
    total_cost = sum(result.edit_distance for result in results)
    total_symbols = sum(result.total_symbols for result in results)
    return {
        "n_samples": len(results),
        "mean_OMR-NED": sum(result.omr_ned for result in results) / len(results),
        "corpus_OMR-NED": total_cost / total_symbols if total_symbols else 0.0,
        "total_OMR-ED": total_cost,
        "total_symbols": total_symbols,
    }


def read_folder_results(path: str | Path) -> list[FolderResult]:
    """Read per-score rows from musicdiff's folder-mode CSV."""
    csv_path = Path(path)
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"musicdiff output is empty: {csv_path}")
    header = [value.strip() for value in rows[0]]
    required = {
        "gtpath",
        "predpath",
        "gt numsyms",
        "pred numsyms",
        "OMR-ED (OMR Edit Distance)",
        "OMR-NED (OMR-ED / total numsyms)",
    }
    missing = required - set(header)
    if missing:
        raise ValueError(
            f"musicdiff output lacks required columns: {sorted(missing)}"
        )
    index = {name: header.index(name) for name in required}
    parsed: list[FolderResult] = []
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        values = [value.strip() for value in row]
        if not values[index["gtpath"]] or values[index["gtpath"]] == "gtpath":
            continue
        try:
            metrics = OMRNEDResult(
                edit_distance=int(values[index["OMR-ED (OMR Edit Distance)"]]),
                pred_symbols=int(values[index["pred numsyms"]]),
                gt_symbols=int(values[index["gt numsyms"]]),
                omr_ned=float(
                    values[index["OMR-NED (OMR-ED / total numsyms)"]]
                ),
            )
        except ValueError:
            continue
        parsed.append(FolderResult(
            pred_path=values[index["predpath"]],
            gt_path=values[index["gtpath"]],
            metrics=metrics,
        ))
    if not parsed:
        raise ValueError(f"musicdiff output has no per-score rows: {csv_path}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate score notation with OMR-NED",
    )
    parser.add_argument(
        "--musicdiff-root",
        type=Path,
        default=DEFAULT_MUSICDIFF_ROOT,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred", type=Path, help="One predicted score")
    mode.add_argument("--pred-folder", type=Path, help="Folder of predicted scores")
    parser.add_argument("--gt", type=Path, help="One ground-truth score")
    parser.add_argument("--gt-folder", type=Path, help="Folder of ground-truth scores")
    parser.add_argument("--output-folder", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    evaluator = OMRNEDEvaluator(args.musicdiff_root)
    if args.pred is not None:
        if args.gt is None or args.gt_folder is not None or args.output_folder is not None:
            raise SystemExit("single-pair mode requires --pred and --gt only")
        result = evaluator.evaluate(args.gt, args.pred)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return

    if args.gt_folder is None or args.output_folder is None or args.gt is not None:
        raise SystemExit(
            "folder mode requires --pred-folder, --gt-folder, and --output-folder"
        )
    overall, output_path = evaluator.evaluate_folders(
        args.gt_folder,
        args.pred_folder,
        args.output_folder,
    )
    folder_rows = read_folder_results(output_path)
    aggregate = aggregate_omr_ned_results(
        [row.metrics for row in folder_rows]
    )
    aggregate.update({
        "corpus_OMR-NED": overall,
        "output_csv": str(output_path),
    })
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
