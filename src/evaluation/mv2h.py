#!/usr/bin/env python3
"""
MV2H Evaluation Module

Standalone module for MV2H (Multi-pitch, Voice, Meter, Value, Harmony) evaluation.
Can be used on any system's output.

Reference:
    McLeod & Steedman (2018)
    "Evaluating Automatic Polyphonic Music Transcription"

Metrics:
    - Multi-pitch: Pitch accuracy
    - Voice: Voice separation accuracy
    - Meter: Metrical structure accuracy (excluded from MV2H_custom)
    - Value: Note value accuracy
    - Harmony: Harmonic structure accuracy
    - MV2H: Official average of all 5 metrics
    - MV2H_custom: Average of 4 metrics (excludes Meter, used by Zeng et al.)

Usage:
    from evaluation.mv2h import MV2HEvaluator, MV2HResult

    evaluator = MV2HEvaluator(mv2h_bin="external/MV2H/bin")
    result = evaluator.evaluate(gt_midi_path, pred_midi_path)

    print(result.mv2h_custom)  # 4-metric average (no Meter)
    print(result.to_dict())    # All 6 metrics
"""

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MV2HResult:
    """
    Container for MV2H evaluation results.

    Attributes:
        multi_pitch: Pitch accuracy (0-1)
        voice: Voice separation accuracy (0-1)
        meter: Metrical structure accuracy (0-1)
        value: Note value accuracy (0-1)
        harmony: Harmonic structure accuracy (0-1)
        mv2h: Official MV2H score (average of all 5)
        mv2h_custom: Custom score (average of 4, excludes Meter)
    """

    multi_pitch: float = 0.0
    voice: float = 0.0
    meter: float = 0.0
    value: float = 0.0
    harmony: float = 0.0
    mv2h: float = 0.0  # Official: (MP + V + M + Va + H) / 5

    @property
    def mv2h_custom(self) -> float:
        """
        Custom MV2H score excluding Meter (used by Zeng et al.).

        Formula: (Multi-pitch + Voice + Value + Harmony) / 4
        """
        return (self.multi_pitch + self.voice + self.value + self.harmony) / 4

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary with all metrics."""
        return {
            "Multi-pitch": self.multi_pitch,
            "Voice": self.voice,
            "Meter": self.meter,
            "Value": self.value,
            "Harmony": self.harmony,
            "MV2H": self.mv2h,
            "MV2H_custom": self.mv2h_custom,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MV2HResult":
        """Create from dictionary."""
        return cls(
            multi_pitch=d.get("Multi-pitch", 0.0),
            voice=d.get("Voice", 0.0),
            meter=d.get("Meter", 0.0),
            value=d.get("Value", 0.0),
            harmony=d.get("Harmony", 0.0),
            mv2h=d.get("MV2H", 0.0),
        )


@dataclass
class MidiPairTask:
    """One prediction/reference MIDI pair evaluated under one timeout."""

    task_id: str
    pred_midi: str
    gt_midi: str
    mv2h_bin: str
    timeout: Optional[int]
    # Sub-beats before the prediction's first full bar (MV2H Converter -a);
    # None keeps the converter's default of zero.
    pred_anacrusis: Optional[int] = None
    single_path: bool = False


@dataclass
class EvaluationResult:
    """Status and optional metrics for one MIDI pair."""

    task_id: str
    pred_path: str
    gt_path: str
    status: str
    metrics: Optional[MV2HResult] = None
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "task_id": self.task_id,
            "pred_path": self.pred_path,
            "gt_path": self.gt_path,
            "status": self.status,
            "error_message": self.error_message,
        }
        if self.metrics:
            result.update(self.metrics.to_dict())
        return result

    def __repr__(self) -> str:
        return (
            f"MV2HResult(MP={self.multi_pitch:.3f}, V={self.voice:.3f}, "
            f"M={self.meter:.3f}, Va={self.value:.3f}, H={self.harmony:.3f}, "
            f"MV2H={self.mv2h:.3f}, custom={self.mv2h_custom:.3f})"
        )


# =============================================================================
# MV2H EVALUATOR
# =============================================================================


class MV2HEvaluator:
    """
    MV2H evaluation wrapper for the Java MV2H tool.

    Uses DTW alignment (-a flag) for robust comparison between
    ground truth and prediction MIDI files.

    Args:
        mv2h_bin: Path to MV2H bin directory (containing compiled Java classes)
        timeout: Timeout in seconds for each evaluation (default: 300)
        use_alignment: Whether to use DTW alignment (default: True)
        single_path: Score one minimum-cost alignment (-s) rather than the best of
            all of them (-a); requires use_alignment

    Example:
        evaluator = MV2HEvaluator(mv2h_bin="external/MV2H/bin", timeout=120)
        result = evaluator.evaluate("gt.mid", "pred.mid")
        print(f"MV2H (custom): {result.mv2h_custom:.2%}")
    """

    def __init__(
        self,
        mv2h_bin: str,
        timeout: Optional[int] = 300,
        use_alignment: bool = True,
        single_path: bool = False,
    ):
        self.mv2h_bin = mv2h_bin
        self.timeout = timeout
        self.use_alignment = use_alignment
        self.single_path = single_path
        if single_path and not use_alignment:
            raise ValueError("single_path scoring is an alignment mode; use_alignment must be True")

        # Validate MV2H bin exists
        if not Path(mv2h_bin).exists():
            raise FileNotFoundError(
                f"MV2H bin not found: {mv2h_bin}\n"
                "Please compile MV2H: cd MV2H && make"
            )

    def evaluate_converted(
        self,
        gt_converted: str,
        pred_converted: str,
        java_heap: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ) -> Optional[MV2HResult]:
        """Score two files already in the MV2H text format.

        Callers that prepare their own conversion — whole scores, whose metrical
        metadata is rewritten after conversion — enter the pipeline here.
        `log_dir` keeps the scorer's own output next to the scored pair.
        """
        mv2h_cmd = ["java"]
        if java_heap:
            mv2h_cmd.append(f"-Xmx{java_heap}")
        mv2h_cmd += ["-cp", self.mv2h_bin, "mv2h.Main", "-g", gt_converted, "-t", pred_converted]
        if self.use_alignment:
            mv2h_cmd.append("-s" if self.single_path else "-a")

        result_mv2h = subprocess.run(
            mv2h_cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if log_dir is not None:
            (log_dir / "scorer.stdout.txt").write_text(result_mv2h.stdout)
            (log_dir / "scorer.stderr.txt").write_text(result_mv2h.stderr)

        if result_mv2h.returncode != 0:
            logger.debug(f"MV2H evaluation failed: {result_mv2h.stderr}")
            return None

        # Parse output (last 6 lines contain metrics)
        return self._parse_output(result_mv2h.stdout)

    def evaluate(
        self,
        gt_midi_path: str,
        pred_midi_path: str,
        pred_anacrusis: Optional[int] = None,
    ) -> Optional[MV2HResult]:
        """
        Run MV2H evaluation between ground truth and prediction MIDI files.

        Pipeline:
        1. Convert GT MIDI to MV2H format
        2. Convert Pred MIDI to MV2H format
        3. Run MV2H evaluation with DTW alignment

        Args:
            gt_midi_path: Path to ground truth MIDI file
            pred_midi_path: Path to prediction MIDI file

        Returns:
            MV2HResult object or None if evaluation failed
        """
        # Create unique temp files for parallel safety
        uid = uuid.uuid4().hex[:8]
        gt_conv = f"/tmp/mv2h_gt_{uid}.conv.txt"
        pred_conv = f"/tmp/mv2h_pred_{uid}.conv.txt"

        try:
            # Step 1: Convert GT MIDI to MV2H format
            result_gt = subprocess.run(
                ["java", "-cp", self.mv2h_bin, "mv2h.tools.Converter", "-i", gt_midi_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result_gt.returncode != 0:
                logger.debug(f"GT conversion failed: {result_gt.stderr}")
                return None

            with open(gt_conv, "w") as f:
                f.write(result_gt.stdout)

            # Step 2: Convert Pred MIDI to MV2H format
            pred_cmd = ["java", "-cp", self.mv2h_bin, "mv2h.tools.Converter", "-i", pred_midi_path]
            if pred_anacrusis is not None:
                pred_cmd += ["-a", str(int(pred_anacrusis))]
            result_pred = subprocess.run(
                pred_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result_pred.returncode != 0:
                logger.debug(f"Pred conversion failed: {result_pred.stderr}")
                return None

            with open(pred_conv, "w") as f:
                f.write(result_pred.stdout)

            # Step 3: Run MV2H evaluation
            return self.evaluate_converted(gt_conv, pred_conv)

        except subprocess.TimeoutExpired:
            logger.debug(f"MV2H timeout after {self.timeout}s")
            raise  # Let caller handle timeout separately from mv2h_failed

        except Exception as e:
            logger.debug(f"MV2H error: {e}")
            return None

        finally:
            # Cleanup temporary files
            for f in [gt_conv, pred_conv]:
                try:
                    os.remove(f)
                except OSError:
                    pass

    def _parse_output(self, output: str) -> Optional[MV2HResult]:
        """Parse MV2H output to extract metrics."""
        lines = output.strip().splitlines()
        if len(lines) < 6:
            logger.debug(f"Unexpected MV2H output: {output}")
            return None

        metrics = {}
        for line in lines[-6:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass

        if not metrics:
            return None

        return MV2HResult.from_dict(metrics)


def evaluate_midi_pair(task: MidiPairTask) -> EvaluationResult:
    """Evaluate one already-rendered MIDI pair without format conversion."""
    try:
        evaluator = MV2HEvaluator(task.mv2h_bin, timeout=task.timeout,
                                  single_path=task.single_path)
        metrics = evaluator.evaluate(
            task.gt_midi, task.pred_midi, pred_anacrusis=task.pred_anacrusis,
        )
        if metrics is None:
            return EvaluationResult(
                task_id=task.task_id,
                pred_path=task.pred_midi,
                gt_path=task.gt_midi,
                status="mv2h_failed",
                error_message="MV2H returned no result",
            )
        if metrics.mv2h == 0 and metrics.multi_pitch == 0:
            return EvaluationResult(
                task_id=task.task_id,
                pred_path=task.pred_midi,
                gt_path=task.gt_midi,
                status="zero_score",
                error_message="MV2H returned zero",
                metrics=metrics,
            )
        return EvaluationResult(
            task_id=task.task_id,
            pred_path=task.pred_midi,
            gt_path=task.gt_midi,
            status="success",
            metrics=metrics,
        )
    except subprocess.TimeoutExpired:
        return EvaluationResult(
            task_id=task.task_id,
            pred_path=task.pred_midi,
            gt_path=task.gt_midi,
            status="timeout",
            error_message=f"MV2H exceeded {task.timeout}s timeout",
        )
    except Exception as error:
        return EvaluationResult(
            task_id=task.task_id,
            pred_path=task.pred_midi,
            gt_path=task.gt_midi,
            status="error",
            error_message=str(error),
        )


# =============================================================================
# AGGREGATION UTILITIES
# =============================================================================


def aggregate_mv2h_results(results: List[MV2HResult]) -> Dict[str, float]:
    """
    Aggregate multiple MV2H results into summary statistics.

    Args:
        results: List of MV2HResult objects

    Returns:
        Dictionary with average metrics and counts
    """
    if not results:
        return {
            "n_samples": 0,
            "Multi-pitch": 0.0,
            "Voice": 0.0,
            "Meter": 0.0,
            "Value": 0.0,
            "Harmony": 0.0,
            "MV2H": 0.0,
            "MV2H_custom": 0.0,
        }

    n = len(results)
    return {
        "n_samples": n,
        "Multi-pitch": sum(r.multi_pitch for r in results) / n,
        "Voice": sum(r.voice for r in results) / n,
        "Meter": sum(r.meter for r in results) / n,
        "Value": sum(r.value for r in results) / n,
        "Harmony": sum(r.harmony for r in results) / n,
        "MV2H": sum(r.mv2h for r in results) / n,
        "MV2H_custom": sum(r.mv2h_custom for r in results) / n,
    }


def summarize_evaluations(
    results: List[EvaluationResult], total: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate successful-only and failures-as-zero scores."""
    total = len(results) if total is None else total
    if total < len(results):
        raise ValueError("Summary denominator cannot be smaller than its rows")
    successful = [
        result.metrics
        for result in results
        if result.status == "success" and result.metrics is not None
    ]
    successful_only = aggregate_mv2h_results(successful)
    failures_as_zero = {
        metric: successful_only.get(metric, 0.0) * len(successful) / total
        if total else 0.0
        for metric in (
            "Multi-pitch", "Voice", "Meter", "Value", "Harmony",
            "MV2H", "MV2H_custom",
        )
    }
    status_counts: Dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    return {
        "n_total": total,
        "n_rows": len(results),
        "n_successful": len(successful),
        "status_counts": status_counts,
        "successful_only": successful_only,
        "failures_as_zero": failures_as_zero,
    }


def print_evaluation_summary(summary: Dict[str, Any]) -> None:
    """Print coverage plus both MV2H aggregation protocols."""
    print("\n" + "=" * 70)
    print("MV2H Evaluation Summary")
    print("=" * 70)
    print(
        f"n_evaluated / n_total = "
        f"{summary['n_successful']} / {summary['n_total']}"
    )
    print("Status breakdown:")
    for status, count in sorted(summary["status_counts"].items()):
        print(f"  {status:28s}: {count}")
    for title, key in (
        ("Successful-only", "successful_only"),
        ("Failures-as-zero", "failures_as_zero"),
    ):
        print("\n" + title + ":")
        metrics = summary[key]
        for metric in (
            "Multi-pitch", "Voice", "Meter", "Value", "Harmony",
            "MV2H", "MV2H_custom",
        ):
            print(f"  {metric:15s}: {metrics.get(metric, 0.0) * 100:6.2f}%")
    print("=" * 70)


def print_mv2h_summary(summary: Dict[str, float], title: str = "MV2H Results") -> None:
    """Print formatted MV2H summary."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Samples: {summary.get('n_samples', 0)}")
    print("-" * 60)

    for metric in ["Multi-pitch", "Voice", "Meter", "Value", "Harmony"]:
        if metric in summary:
            print(f"  {metric:15s}: {summary[metric] * 100:6.2f}%")

    print("-" * 60)
    if "MV2H" in summary:
        print(f"  {'MV2H (official)':15s}: {summary['MV2H'] * 100:6.2f}%")
    if "MV2H_custom" in summary:
        print(f"  {'MV2H (custom)':15s}: {summary['MV2H_custom'] * 100:6.2f}%  <- excludes Meter")
    print("=" * 60)
