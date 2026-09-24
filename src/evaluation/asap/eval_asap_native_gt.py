#!/usr/bin/env python3
"""
ASAP reference construction and model-agnostic MV2H evaluation.

Native actions build or validate one frozen recording-to-score mapping, then
evaluate prediction manifests with one strict result row per reference window.
The older full/chunks modes remain available for their existing callers and
retain their MuseScore conversion behavior.

Usage:
    python -m src.evaluation.asap.eval_asap_native_gt --mode full \
        --pred_dir data/prediction_midi \
        --gt_dir /path/to/asap \
        --mv2h_bin external/MV2H/bin \
        --output results.csv \
        --workers 8

    # Retry failed tasks
    python -m src.evaluation.asap.eval_asap_native_gt --retry_file failed.txt \
        --mv2h_bin external/MV2H/bin \
        --output results_retry.csv \
        --timeout 300
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

from src.evaluation.asap import (
    Asap102MappingError,
    ASAPDataset,
    ChunkInfo,
    extract_chunks_batch,
    load_asap102_inventory,
    load_mapping_adjudications,
    performance_order_musicxml_bytes,
    resolve_asap102_mapping,
    sequence_sha256,
    sha256_file,
    valid_five_bar_positions,
)

# Import shared modules
from src.evaluation.mv2h import (
    EvaluationResult as EvalResult,
)
from src.evaluation.mv2h import (
    MidiPairTask as ChunkEvalTask,
)
from src.evaluation.mv2h import (
    MV2HEvaluator,
    MV2HResult,
    aggregate_mv2h_results,
    print_mv2h_summary,
)
from src.evaluation.mv2h import (
    evaluate_midi_pair as evaluate_chunk_task,
)
from src.evaluation.mv2h import (
    print_evaluation_summary as print_chunk_summary,
)
from src.evaluation.mv2h import (
    summarize_evaluations as compute_chunk_summary,
)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MuseScoreConfig:
    """MuseScore 4.6.5 configuration."""

    binary_path: str = "tools/mscore"
    timeout: int = 60
    force_overwrite: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    mv2h_bin: str = "external/MV2H/bin"
    mv2h_timeout: int = 120
    mv2h_chunk_timeout: int = 300
    mscore_config: MuseScoreConfig = None
    workers: int = os.cpu_count() or 4
    cache_dir: Optional[str] = None  # MusicXML cache dir; defaults to {output_dir}/full_musicxml

    def __post_init__(self):
        if self.mscore_config is None:
            self.mscore_config = MuseScoreConfig()


# =============================================================================
# YAML CONFIG LOADING
# =============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load evaluation configuration from YAML file.

    The YAML config allows centralizing all paths and parameters for easier
    management and reproducibility of experiments.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration values
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# MUSESCORE CONVERSION
# =============================================================================


def convert_midi_to_musicxml(
    midi_path: str,
    output_path: str,
    config: MuseScoreConfig,
) -> bool:
    """Convert MIDI to MusicXML using MuseScore 4.6.5."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [config.binary_path, midi_path, "-o", output_path]
    if config.force_overwrite:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=config.timeout
        )
        return result.returncode == 0 and Path(output_path).exists()

    except subprocess.TimeoutExpired:
        logger.debug(f"MuseScore timeout: {midi_path}")
        return False
    except FileNotFoundError:
        logger.error(f"MuseScore not found: {config.binary_path}")
        return False
    except Exception as e:
        logger.debug(f"MuseScore error: {e}")
        return False


def convert_musicxml_to_midi(
    musicxml_path: str,
    midi_path: str,
    config: MuseScoreConfig,
) -> bool:
    """Convert MusicXML to MIDI using MuseScore."""
    Path(midi_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [config.binary_path, musicxml_path, "-o", midi_path]
    if config.force_overwrite:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=config.timeout
        )
        return result.returncode == 0 and Path(midi_path).exists()
    except Exception as e:
        logger.debug(f"Conversion error: {e}")
        return False


# =============================================================================
# EVALUATION TASK
# =============================================================================


@dataclass
class EvalTask:
    """Single evaluation task."""

    task_id: str
    pred_midi_path: str
    gt_midi_path: str
    output_dir: str
    mscore_config: MuseScoreConfig
    mv2h_bin: str
    mv2h_timeout: int


def evaluate_single_task(task: EvalTask) -> EvalResult:
    """
    Evaluate single prediction.

    Pipeline:
    1. MIDI → MusicXML (MuseScore)
    2. MusicXML → MIDI (consistent format)
    3. MV2H evaluation
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Convert pred MIDI → MusicXML
            pred_xml = os.path.join(temp_dir, f"{task.task_id}.musicxml")
            if not convert_midi_to_musicxml(task.pred_midi_path, pred_xml, task.mscore_config):
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="musescore_failed",
                )

            # Step 2: MusicXML → MIDI (for MV2H)
            pred_midi = os.path.join(temp_dir, f"{task.task_id}_converted.mid")
            if not convert_musicxml_to_midi(pred_xml, pred_midi, task.mscore_config):
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="midi_conversion_failed",
                )

            # Save MusicXML if output_dir specified
            if task.output_dir:
                out_xml = os.path.join(task.output_dir, "musicxml", f"{task.task_id}.musicxml")
                Path(out_xml).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pred_xml, out_xml)

            # Step 3: MV2H evaluation
            evaluator = MV2HEvaluator(task.mv2h_bin, timeout=task.mv2h_timeout)
            metrics = evaluator.evaluate(task.gt_midi_path, pred_midi)

            if metrics is None:
                return EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="mv2h_failed",
                )

            return EvalResult(
                task_id=task.task_id,
                pred_path=task.pred_midi_path,
                gt_path=task.gt_midi_path,
                status="success",
                metrics=metrics,
            )

    except Exception as e:
        return EvalResult(
            task_id=task.task_id,
            pred_path=task.pred_midi_path,
            gt_path=task.gt_midi_path,
            status="error",
            error_message=str(e),
        )


# =============================================================================
# PARALLEL EVALUATION
# =============================================================================


def run_parallel_evaluation(
    tasks: List[EvalTask],
    workers: int,
) -> List[EvalResult]:
    """Run tasks in parallel."""
    results = []
    n_workers = min(workers, len(tasks))

    logger.info(f"Running {len(tasks)} tasks with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_single_task, t): t for t in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating"):
            try:
                results.append(future.result())
            except Exception as e:
                task = futures[future]
                results.append(EvalResult(
                    task_id=task.task_id,
                    pred_path=task.pred_midi_path,
                    gt_path=task.gt_midi_path,
                    status="executor_error",
                    error_message=str(e),
                ))

    return results


# =============================================================================
# FULL SONG EVALUATION
# =============================================================================


def run_full_song_evaluation(
    pred_dir: str,
    gt_dir: str,
    output_dir: str,
    config: EvalConfig,
) -> List[EvalResult]:
    """Run full song evaluation."""
    asap = ASAPDataset(gt_dir)
    pred_path = Path(pred_dir)

    # Find MIDI files
    midi_files = list(pred_path.rglob("*.mid")) + list(pred_path.rglob("*.midi"))
    logger.info(f"Found {len(midi_files)} MIDI files")

    # Build tasks
    tasks = []
    skipped = []

    for midi_file in midi_files:
        file_id = midi_file.stem
        gt_path = asap.find_ground_truth_midi(str(midi_file))

        if gt_path is None:
            skipped.append(file_id)
            continue

        tasks.append(EvalTask(
            task_id=file_id,
            pred_midi_path=str(midi_file),
            gt_midi_path=gt_path,
            output_dir=output_dir,
            mscore_config=config.mscore_config,
            mv2h_bin=config.mv2h_bin,
            mv2h_timeout=config.mv2h_timeout,
        ))

    if skipped:
        logger.warning(f"Skipped {len(skipped)} files without ground truth")

    # Run evaluation
    results = run_parallel_evaluation(tasks, config.workers)

    # Add skipped
    for file_id in skipped:
        results.append(EvalResult(
            task_id=file_id, pred_path="", gt_path="", status="no_ground_truth"
        ))

    return results


# =============================================================================
# CHUNK EVALUATION
# =============================================================================


def group_chunks_by_piece_performance(
    chunks: List[ChunkInfo],
) -> Dict[Tuple[str, str], List[ChunkInfo]]:
    """
    Group chunks by (piece_id, performance) for efficient batch processing.

    The chunk_id in Zeng format encodes both piece and performance:
        chunk_id = 'Bach#Prelude#bwv_875#Ahfat01M.10'
        -> piece_id = 'Bach#Prelude#bwv_875'
        -> performance = 'Ahfat01M'
        -> chunk_index = 10

    By grouping chunks, we can convert each prediction MIDI to MusicXML
    only once per (piece, performance) pair, then extract all chunks from it.

    Args:
        chunks: List of ChunkInfo objects from Zeng CSV

    Returns:
        Dictionary mapping (piece_id, performance) tuple to list of chunks
    """
    grouped: Dict[Tuple[str, str], List[ChunkInfo]] = {}

    for chunk in chunks:
        # Parse chunk_id: 'Bach#Prelude#bwv_875#Ahfat01M.10'
        # Split from right to separate piece_id from performance.chunk_index
        parts = chunk.chunk_id.rsplit("#", 1)

        if len(parts) != 2:
            logger.warning(f"Invalid chunk_id format: {chunk.chunk_id}")
            continue

        piece_id = parts[0]  # 'Bach#Prelude#bwv_875'
        perf_chunk = parts[1]  # 'Ahfat01M.10'

        # Extract performance name (remove chunk index)
        performance = perf_chunk.rsplit(".", 1)[0]  # 'Ahfat01M'

        key = (piece_id, performance)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(chunk)

    return grouped


def find_pred_midi(
    pred_dir: str,
    piece_id: str,
    performance: str,
) -> Optional[str]:
    """
    Find prediction MIDI file for given piece and performance.

    Supports two directory structures:
    1. Flat: pred_dir/performance.mid
    2. ASAP-style: pred_dir/Composer/Work/Piece/performance.mid

    The ASAP-style structure: one prediction MIDI per recording.

    Args:
        pred_dir: Root directory containing prediction MIDI files
        piece_id: Zeng-style piece identifier (e.g., 'Bach#Prelude#bwv_875')
        performance: Performance ID (e.g., 'Ahfat01M')

    Returns:
        Path to prediction MIDI file or None if not found
    """
    pred_path = Path(pred_dir)

    # Convert piece_id from '#' separator to path components
    # 'Bach#Prelude#bwv_875' -> ['Bach', 'Prelude', 'bwv_875']
    path_parts = piece_id.split("#")

    # Try ASAP-style structure first: pred_dir/Bach/Prelude/bwv_875/Ahfat01M.mid
    search_dir = pred_path / "/".join(path_parts)
    if search_dir.exists():
        # Look for files starting with performance ID
        for ext in [".mid", ".midi"]:
            midi_file = search_dir / f"{performance}{ext}"
            if midi_file.exists():
                return str(midi_file)

        # Also try glob pattern for variations (e.g., performance_001.mid)
        for midi_file in search_dir.glob(f"{performance}*"):
            if midi_file.suffix.lower() in [".mid", ".midi"]:
                return str(midi_file)

    # Try flat structure: pred_dir/Bach_Prelude_bwv_875_Ahfat01M.mid
    flat_name = "_".join(path_parts) + f"_{performance}"
    for ext in [".mid", ".midi"]:
        flat_path = pred_path / f"{flat_name}{ext}"
        if flat_path.exists():
            return str(flat_path)

    # Try recursive search as fallback
    for midi_file in pred_path.rglob(f"*{performance}*.mid"):
        return str(midi_file)

    logger.debug(f"No prediction found for {piece_id}#{performance}")
    return None


def convert_to_musicxml_cached(
    midi_path: str,
    cache_dir: str,
    config: MuseScoreConfig,
) -> Optional[str]:
    """
    Convert MIDI to MusicXML using MuseScore, with caching.

    Caching ensures each MIDI file is only converted once, even when
    extracting multiple chunks from the same prediction.

    Args:
        midi_path: Path to input MIDI file
        cache_dir: Directory to store converted MusicXML files
        config: MuseScore configuration

    Returns:
        Path to MusicXML file or None if conversion failed
    """
    cache_subdir = Path(cache_dir)
    cache_subdir.mkdir(parents=True, exist_ok=True)

    # Create cache path based on MIDI filename
    midi_stem = Path(midi_path).stem
    musicxml_path = cache_subdir / f"{midi_stem}.musicxml"

    # Return cached version if exists
    if musicxml_path.exists():
        logger.debug(f"Using cached MusicXML: {musicxml_path}")
        return str(musicxml_path)

    # Convert MIDI to MusicXML
    if convert_midi_to_musicxml(midi_path, str(musicxml_path), config):
        return str(musicxml_path)

    return None


def save_failed_chunks(
    results: List[EvalResult],
    output_dir: str,
) -> Tuple[int, int]:
    """
    Save failed chunks for analysis and potential retry.

    Separates failures into categories matching Zeng's evaluation:
    - timeouts.txt: Chunks that exceeded MV2H time limit
    - errors.txt: All other failures with details

    Args:
        results: List of EvalResult objects
        output_dir: Directory to save failure lists

    Returns:
        Tuple of (timeout_count, error_count)
    """
    timeouts = [r for r in results if r.status == "timeout"]
    errors = [r for r in results if r.status not in ["success", "timeout"]]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if timeouts:
        timeout_path = os.path.join(output_dir, "timeouts.txt")
        with open(timeout_path, "w") as f:
            f.write(f"# Total: {len(timeouts)} chunks timeout\n")
            f.write("# Format: chunk_id\n")
            for r in timeouts:
                f.write(f"{r.task_id}\n")
        logger.info(f"Timeout chunks saved to: {timeout_path}")

    if errors:
        error_path = os.path.join(output_dir, "errors.txt")
        with open(error_path, "w") as f:
            f.write(f"# Total: {len(errors)} chunks with errors\n")
            f.write("# Format: chunk_id\\tstatus\\terror_message\n")
            for r in errors:
                f.write(f"{r.task_id}\t{r.status}\t{r.error_message}\n")
        logger.info(f"Error chunks saved to: {error_path}")

    return len(timeouts), len(errors)


def run_chunk_evaluation(
    pred_dir: str,
    gt_dir: str,
    chunk_csv: str,
    output_dir: str,
    config: EvalConfig,
    output_csv: Optional[str] = None,
    musicxml_dir: Optional[str] = None,
) -> Tuple[List[EvalResult], Dict[str, Any]]:
    """
    Run 5-bar chunk evaluation with actual measure extraction.

    This function implements apple-to-apple comparison with Zeng's evaluation:
    1. Load chunk definitions from Zeng CSV (piece#performance.chunk_index format)
    2. Group chunks by (piece, performance) for efficient processing
    3. Convert each prediction MIDI to MusicXML (via MuseScore, once per piece)
    4. Extract 5-bar chunks from both pred MusicXML and GT MusicXML
    5. Run MV2H on extracted chunk MIDI files
    6. Report results using both Zeng's method and include-failures method

    Incremental saving:
    - Results are saved to CSV immediately after each chunk completes
    - Already completed chunks (status='success' in CSV) are skipped
    - This enables resuming evaluation after interruption

    Args:
        pred_dir: Directory containing prediction MIDI files
        gt_dir: ASAP dataset directory with ground truth
        chunk_csv: Path to Zeng chunk CSV file
        output_dir: Output directory for results and intermediate files
        config: Evaluation configuration
        output_csv: Path to results CSV (enables incremental saving and resume)

    Returns:
        Tuple of (results list, summary dict)
    """
    asap = ASAPDataset(gt_dir)
    chunks = asap.load_chunks(chunk_csv)
    grouped = group_chunks_by_piece_performance(chunks)

    total_chunks = len(chunks)
    logger.info(f"Processing {len(grouped)} (piece, performance) pairs, {total_chunks} total chunks")

    # Load already completed chunks if output_csv exists
    completed_chunks = set()
    if output_csv:
        completed_chunks = load_completed_chunks(output_csv)
        if completed_chunks:
            logger.info(f"Found {len(completed_chunks)} already completed chunks, will skip")
        # Initialize CSV with headers if needed
        init_csv_file(output_csv, CHUNK_CSV_FIELDS)

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chunk_midi_dir = os.path.join(output_dir, "chunk_midi")
    Path(chunk_midi_dir).mkdir(parents=True, exist_ok=True)

    # Cache dir for MusicXML (parse MIDI → MusicXML once per piece, reuse for all chunks)
    musicxml_cache_dir = config.cache_dir or os.path.join(output_dir, "full_musicxml")

    # Prepare all chunk evaluation tasks
    tasks: List[ChunkEvalTask] = []
    skipped_pieces: List[str] = []
    conversion_errors: List[str] = []
    skipped_completed: int = 0

    # Process each (piece, performance) group
    for (piece_id, performance), piece_chunks in tqdm(
        grouped.items(),
        desc="Extracting chunks",
        unit="piece",
    ):
        # 1. Find prediction MIDI
        pred_file = find_pred_midi(pred_dir, piece_id, performance)
        if pred_file is None:
            skipped_pieces.append(f"{piece_id}#{performance}")
            continue

        # 2. Find ground truth MusicXML
        gt_xml = asap.find_ground_truth_xml_by_piece_id(piece_id)
        if gt_xml is None:
            logger.debug(f"No GT MusicXML for: {piece_id}")
            skipped_pieces.append(f"{piece_id}#{performance}")
            continue

        # 3. Convert prediction MIDI to MusicXML (cached)
        pred_xml = convert_to_musicxml_cached(pred_file, musicxml_cache_dir, config.mscore_config)
        if pred_xml is None:
            conversion_errors.append(f"{piece_id}#{performance}")
            continue

        # 4. Filter out already completed chunks
        chunks_to_extract = []
        for chunk in piece_chunks:
            if chunk.chunk_id in completed_chunks:
                skipped_completed += 1
            else:
                chunks_to_extract.append(chunk)

        if not chunks_to_extract:
            continue

        # 5. Prepare batch extraction lists (parse each XML only once!)
        pred_chunks_list = [
            (
                chunk.start_measure,
                chunk.end_measure,
                os.path.join(chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_pred.mid"),
            )
            for chunk in chunks_to_extract
        ]
        gt_chunks_list = [
            (
                chunk.start_measure,
                chunk.end_measure,
                os.path.join(chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_gt.mid"),
            )
            for chunk in chunks_to_extract
        ]

        # 6. Batch extract (parse XML once, extract all chunks)
        pred_results = extract_chunks_batch(pred_xml, pred_chunks_list)
        gt_results = extract_chunks_batch(gt_xml, gt_chunks_list)

        # 7. Create evaluation tasks
        for chunk in chunks_to_extract:
            pred_chunk_path = os.path.join(
                chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_pred.mid"
            )
            gt_chunk_path = os.path.join(
                chunk_midi_dir, f"{chunk.chunk_id.replace('#', '_')}_gt.mid"
            )

            pred_chunk = pred_results.get(pred_chunk_path)
            gt_chunk = gt_results.get(gt_chunk_path)

            if pred_chunk and gt_chunk:
                tasks.append(ChunkEvalTask(
                    task_id=chunk.chunk_id,
                    pred_midi=pred_chunk,
                    gt_midi=gt_chunk,
                    mv2h_bin=config.mv2h_bin,
                    timeout=config.mv2h_chunk_timeout,
                ))
            else:
                logger.debug(f"Failed to extract chunk: {chunk.chunk_id}")

    if skipped_completed:
        logger.info(f"Skipped {skipped_completed} already completed chunks")
    if skipped_pieces:
        logger.warning(f"Skipped {len(skipped_pieces)} pieces (no prediction or GT)")
    if conversion_errors:
        logger.warning(f"Conversion failed for {len(conversion_errors)} pieces")

    logger.info(f"Prepared {len(tasks)} new chunk evaluation tasks")

    # Run parallel evaluation with incremental saving
    new_results: List[EvalResult] = []
    n_workers = min(config.workers, len(tasks)) if tasks else 1

    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(evaluate_chunk_task, t): t for t in tasks}

            for future in tqdm(
                as_completed(futures),
                total=len(tasks),
                desc="Evaluating chunks",
            ):
                try:
                    result = future.result()
                except Exception as e:
                    task = futures[future]
                    result = EvalResult(
                        task_id=task.task_id,
                        pred_path=task.pred_midi,
                        gt_path=task.gt_midi,
                        status="executor_error",
                        error_message=str(e),
                    )

                new_results.append(result)

                # Incremental save to CSV
                if output_csv:
                    append_result_to_csv(result, output_csv, CHUNK_CSV_FIELDS)

    # Save failed chunks to separate file for easy retry
    save_failed_chunks(new_results, output_dir)

    # Load all results (including previously completed) for summary
    all_results = new_results
    if output_csv and completed_chunks:
        # Reload all results from CSV for accurate summary
        all_results = load_results_from_csv(output_csv)
        logger.info(f"Loaded {len(all_results)} total results for summary")

    # Compute summary
    summary = compute_chunk_summary(all_results, total_chunks)

    return all_results, summary


# =============================================================================
# RETRY FUNCTIONALITY
# =============================================================================


def save_failed_tasks(results: List[EvalResult], output_path: str) -> int:
    """Save failed tasks for retry."""
    failed_statuses = ["mv2h_failed", "musescore_failed", "midi_conversion_failed",
                       "error", "executor_error", "zero_score"]
    failed = [r for r in results if r.status in failed_statuses]

    if not failed:
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Failed tasks for retry\n")
        f.write("# task_id\\tpred_path\\tgt_path\\tstatus\n")
        for r in failed:
            f.write(f"{r.task_id}\t{r.pred_path}\t{r.gt_path}\t{r.status}\n")

    logger.info(f"Saved {len(failed)} failed tasks to: {output_path}")
    return len(failed)


def load_retry_tasks(
    retry_path: str,
    output_dir: str,
    config: EvalConfig,
) -> List[EvalTask]:
    """Load tasks from retry file."""
    tasks = []

    with open(retry_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            task_id, pred_path, gt_path = parts[0], parts[1], parts[2]

            if not Path(pred_path).exists() or not Path(gt_path).exists():
                continue

            tasks.append(EvalTask(
                task_id=task_id,
                pred_midi_path=pred_path,
                gt_midi_path=gt_path,
                output_dir=output_dir,
                mscore_config=config.mscore_config,
                mv2h_bin=config.mv2h_bin,
                mv2h_timeout=config.mv2h_timeout,
            ))

    logger.info(f"Loaded {len(tasks)} retry tasks")
    return tasks


# =============================================================================
# RESULTS OUTPUT
# =============================================================================


# CSV field names for chunk evaluation results
# Includes chunk_index for position-based analysis (e.g., success rate by chunk position)
CHUNK_CSV_FIELDS = [
    "task_id", "chunk_index", "piece_id", "performance",
    "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom",
    "pred_path", "gt_path",
]

# CSV field names for full song evaluation
FULL_CSV_FIELDS = [
    "task_id", "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony", "MV2H", "MV2H_custom",
    "pred_path", "gt_path",
]


def parse_chunk_id(chunk_id: str) -> Dict[str, Any]:
    """
    Parse Zeng-style chunk_id into components for analysis.

    Args:
        chunk_id: Zeng chunk identifier (e.g., 'Bach#Prelude#bwv_875#Ahfat01M.10')

    Returns:
        Dictionary with piece_id, performance, chunk_index

    Example:
        parse_chunk_id('Bach#Prelude#bwv_875#Ahfat01M.10')
        -> {'piece_id': 'Bach#Prelude#bwv_875', 'performance': 'Ahfat01M', 'chunk_index': 10}
    """
    # chunk_id = "Bach#Prelude#bwv_875#Ahfat01M.10"
    parts = chunk_id.rsplit("#", 1)
    if len(parts) != 2:
        return {"piece_id": chunk_id, "performance": "", "chunk_index": 0}

    piece_id = parts[0]  # Bach#Prelude#bwv_875
    perf_chunk = parts[1]  # Ahfat01M.10

    perf_parts = perf_chunk.rsplit(".", 1)
    if len(perf_parts) == 2:
        performance = perf_parts[0]  # Ahfat01M
        try:
            chunk_index = int(perf_parts[1])  # 10
        except ValueError:
            chunk_index = 0
    else:
        performance = perf_chunk
        chunk_index = 0

    return {"piece_id": piece_id, "performance": performance, "chunk_index": chunk_index}


def load_completed_chunks(csv_path: str) -> set:
    """
    Load task_ids of already completed chunks from existing CSV.

    This enables resuming evaluation from where it left off.
    Only chunks with status='success' are considered completed.

    Args:
        csv_path: Path to existing results CSV

    Returns:
        Set of completed task_ids
    """
    completed = set()
    if not Path(csv_path).exists():
        return completed

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only skip if successfully evaluated (not failed)
                if row.get("status") == "success":
                    completed.add(row.get("task_id", ""))
    except Exception as e:
        logger.warning(f"Failed to read existing CSV: {e}")

    return completed


def init_csv_file(csv_path: str, fieldnames: List[str]) -> None:
    """
    Initialize CSV file with headers if it doesn't exist.

    Args:
        csv_path: Path to CSV file
        fieldnames: CSV column names
    """
    if Path(csv_path).exists():
        return

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()


def append_result_to_csv(result: EvalResult, csv_path: str, fieldnames: List[str]) -> None:
    """
    Append a single evaluation result to CSV file.

    This enables incremental saving so results are preserved even if
    the process is interrupted.

    Args:
        result: Evaluation result to append
        csv_path: Path to CSV file
        fieldnames: CSV column names
    """
    row = result.to_dict()

    # Add parsed chunk info for chunk evaluation
    if "chunk_index" in fieldnames:
        parsed = parse_chunk_id(result.task_id)
        row.update(parsed)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def load_results_from_csv(csv_path: str) -> List[EvalResult]:
    """
    Load all evaluation results from CSV file.

    This is used to reload results after resuming for accurate summary computation.

    Args:
        csv_path: Path to results CSV file

    Returns:
        List of EvalResult objects
    """
    results = []
    if not Path(csv_path).exists():
        return results

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Reconstruct MV2HResult if metrics are present
                metrics = None
                if row.get("Multi-pitch") and row.get("status") == "success":
                    try:
                        metrics = MV2HResult(
                            multi_pitch=float(row.get("Multi-pitch", 0)),
                            voice=float(row.get("Voice", 0)),
                            meter=float(row.get("Meter", 0)),
                            value=float(row.get("Value", 0)),
                            harmony=float(row.get("Harmony", 0)),
                            mv2h=float(row.get("MV2H", 0)),
                        )
                    except (ValueError, TypeError):
                        pass

                results.append(EvalResult(
                    task_id=row.get("task_id", ""),
                    pred_path=row.get("pred_path", ""),
                    gt_path=row.get("gt_path", ""),
                    status=row.get("status", "unknown"),
                    metrics=metrics,
                    error_message=row.get("error_message", ""),
                ))
    except Exception as e:
        logger.error(f"Failed to load results from CSV: {e}")

    return results


def save_results_csv(results: List[EvalResult], output_path: str, is_chunk_mode: bool = False) -> None:
    """
    Save all results to CSV (used for final output or non-incremental mode).

    Args:
        results: List of evaluation results
        output_path: Path to output CSV file
        is_chunk_mode: If True, use chunk CSV format with chunk_index
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = CHUNK_CSV_FIELDS if is_chunk_mode else FULL_CSV_FIELDS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            if is_chunk_mode:
                parsed = parse_chunk_id(r.task_id)
                row.update(parsed)
            writer.writerow(row)

    logger.info(f"Results saved to: {output_path}")


def compute_summary(results: List[EvalResult]) -> Dict[str, Any]:
    """Compute summary statistics."""
    successful = [r.metrics for r in results if r.status == "success" and r.metrics]

    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    if not successful:
        return {"n_total": len(results), "n_success": 0, "status_breakdown": status_counts}

    agg = aggregate_mv2h_results(successful)
    agg["n_total"] = len(results)
    agg["n_success"] = len(successful)
    agg["n_failed"] = len(results) - len(successful)
    agg["status_breakdown"] = status_counts
    return agg


# =============================================================================
# FORMAL ASAP-102 NATIVE REFERENCE AND EVALUATION
# =============================================================================


NATIVE_RESULT_FIELDS = [
    "chunk_id", "recording_id", "piece_id", "performance_id",
    "position_index", "status", "error_message",
    "Multi-pitch", "Voice", "Meter", "Value", "Harmony",
    "MV2H4", "MV2H5", "prediction_midi", "reference_midi",
]


def _load_jsonl(path: Path, id_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get(id_key)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"Missing {id_key} at {path}:{line_number}")
            if identifier in seen:
                raise ValueError(f"Duplicate {id_key}: {identifier}")
            seen.add(identifier)
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: List[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    ))
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _native_mapping_records(
    *,
    asap_root: Path,
    metadata_path: Path,
    annotations_path: Path,
    adjudications_path: Path,
    mapping_policy: str,
    expected_count: int | None = 102,
    require_audio: bool = True,
) -> list[dict[str, Any]]:
    import music21

    inventory = load_asap102_inventory(
        metadata_path, asap_root,
        expected_count=expected_count, require_audio=require_audio,
    )
    annotations = json.loads(annotations_path.read_text())
    adjudications = (
        load_mapping_adjudications(adjudications_path)
        if mapping_policy == "approved-fallback" else {}
    )
    annotation_hash = sha256_file(annotations_path)
    score_cache: dict[Path, Any] = {}
    records: list[dict[str, Any]] = []
    for recording in inventory:
        key = recording["annotation_key"]
        if key not in annotations:
            raise Asap102MappingError(f"Official annotation is missing: {key}")
        source_xml = recording["source_xml"]
        if source_xml not in score_cache:
            score_cache[source_xml] = music21.converter.parse(source_xml)
        resolution = resolve_asap102_mapping(
            recording,
            annotations[key],
            score_cache[source_xml],
            adjudications,
            mapping_policy=mapping_policy,
        )
        record = {
            "recording_id": recording["recording_id"],
            "artifact_stem": recording["artifact_stem"],
            "asap102_performance_id": recording["asap102_performance_id"],
            "piece_id": recording["piece_id"],
            "performance_id": recording["performance_id"],
            "annotation_key": key,
            "source_xml": str(source_xml.resolve()),
            "source_xml_sha256": sha256_file(source_xml),
            "official_annotation": str(annotations_path.resolve()),
            "official_annotation_sha256": annotation_hash,
            "performance_annotation": str(
                recording["performance_annotation"].resolve()
            ),
            "performance_annotation_sha256": sha256_file(
                recording["performance_annotation"]
            ),
            "performance_midi": str(recording["performance_midi"].resolve()),
            "performance_midi_sha256": sha256_file(
                recording["performance_midi"]
            ),
            **resolution,
        }
        records.append(record)
    records.sort(key=lambda value: value["recording_id"])
    return records


def _reference_input_fingerprint(
    records: List[dict[str, Any]],
    *,
    metadata_sha256: str,
    annotations_sha256: str,
    adjudications_sha256: str,
    music21_version: str,
) -> str:
    stable = []
    for record in records:
        stable.append({
            key: record[key]
            for key in (
                "recording_id", "source_xml_sha256",
                "official_annotation_sha256", "performance_annotation_sha256",
                "performance_midi_sha256", "mapping_provenance",
                "mapping_policy", "source_measure_groups", "undefined_positions",
            )
        })
    return sequence_sha256([{
        "metadata_sha256": metadata_sha256,
        "annotations_sha256": annotations_sha256,
        "adjudications_sha256": adjudications_sha256,
        "music21_version": music21_version,
    }, *stable])


def _validate_reference(
    output_root: Path,
    reference_input_sha256: str,
) -> bool:
    summary_path = output_root / "mapping-summary.json"
    mapping_path = output_root / "mapping.jsonl"
    grounding_path = output_root / "grounding.jsonl"
    grounding_summary_path = output_root / "grounding-summary.json"
    sensitivity_path = output_root / "grounding-asap101.jsonl"
    sensitivity_summary_path = output_root / "grounding-asap101-summary.json"
    if not all(path.is_file() for path in (
        summary_path, mapping_path, grounding_path, grounding_summary_path,
        sensitivity_path, sensitivity_summary_path,
    )):
        return False
    summary = json.loads(summary_path.read_text())
    if summary.get("reference_input_sha256") != reference_input_sha256:
        return False
    mapping_rows = _load_jsonl(mapping_path, "recording_id")
    if summary.get("mapping_inventory_sha256") != sequence_sha256(mapping_rows):
        return False
    for row in mapping_rows:
        relative = row.get("full_reference_xml")
        expected = row.get("full_reference_xml_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            return False
        path = output_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            return False
    grounding_rows = _load_jsonl(grounding_path, "chunk_id")
    grounding_summary = json.loads(grounding_summary_path.read_text())
    if grounding_summary.get("ordered_inventory_sha256") != sequence_sha256(
        grounding_rows
    ):
        return False
    sensitivity_rows = _load_jsonl(sensitivity_path, "chunk_id")
    sensitivity_summary = json.loads(sensitivity_summary_path.read_text())
    if sensitivity_summary.get("ordered_inventory_sha256") != sequence_sha256(
        sensitivity_rows
    ):
        return False
    if any(row.get("performance_id") == "Sekino05M" for row in sensitivity_rows):
        return False
    expected_sensitivity = [
        row for row in grounding_rows if row.get("performance_id") != "Sekino05M"
    ]
    if sensitivity_rows != expected_sensitivity:
        return False
    for row in grounding_rows:
        for path_key, hash_key in (
            ("reference_xml", "reference_xml_sha256"),
            ("reference_midi", "reference_midi_sha256"),
        ):
            relative = row.get(path_key)
            expected = row.get(hash_key)
            if not isinstance(relative, str) or not isinstance(expected, str):
                return False
            path = output_root / relative
            if not path.is_file() or sha256_file(path) != expected:
                return False
    return True


def write_score_midi(score, output: Path) -> dict[str, int]:
    from collections import Counter
    from music21 import midi

    midi_file = midi.translate.music21ObjectToMidiFile(
        score, addStartDelay=False, encoding='utf-8',
    )
    silent_notes = reordered_notes = 0
    for track in midi_file.tracks:
        absolute = []
        tick = 0
        for event in track.events:
            if event.isDeltaTime():
                tick += event.time
            else:
                absolute.append((tick, event))
        positions = {id(event): (index, tick) for index, (tick, event) in enumerate(absolute)}
        omitted: set[int] = set()
        delayed: set[int] = set()
        after_on = {}
        for index, (tick, event) in enumerate(absolute):
            if event.type != midi.ChannelVoiceMessages.NOTE_ON:
                continue
            partner = event.correspondingEvent
            if partner is None or id(partner) not in positions:
                raise ValueError('Exported note has no corresponding note-off event')
            end_index, end_tick = positions[id(partner)]
            if end_tick < tick:
                raise ValueError('Exported note ends before its onset')
            if event.velocity == 0:
                # A zero-velocity note-on is a release, not a silent attack.
                omitted.update((id(event), id(partner)))
                silent_notes += 1
            elif end_tick == tick and end_index < index:
                # Global off-before-on sorting reverses zero-duration pairs.
                delayed.add(id(partner))
                after_on[id(event)] = partner
                reordered_notes += 1

        ordered = []
        for tick, event in absolute:
            if id(event) in omitted or id(event) in delayed:
                continue
            ordered.append((tick, event))
            if id(event) in after_on:
                ordered.append((tick, after_on[id(event)]))

        active: Counter[tuple[int, int]] = Counter()
        for tick, event in ordered:
            if event.type not in (
                midi.ChannelVoiceMessages.NOTE_ON,
                midi.ChannelVoiceMessages.NOTE_OFF,
            ):
                continue
            key = (event.channel, event.pitch)
            if event.type == midi.ChannelVoiceMessages.NOTE_ON and event.velocity > 0:
                active[key] += 1
            elif active[key]:
                active[key] -= 1
            else:
                raise ValueError(f'Unpaired note-off at tick {tick}: {key}')
        if any(active.values()):
            raise ValueError('Exported MIDI contains notes without release events')

        if omitted or delayed:
            track.events = []
            previous_tick = 0
            for tick, event in ordered:
                delta = midi.DeltaTime(track)
                delta.time = tick - previous_tick
                track.events.extend((delta, event))
                previous_tick = tick

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi_file.open(output, 'wb')
    try:
        midi_file.write()
    finally:
        midi_file.close()
    return {'silent_notes': silent_notes, 'reordered_notes': reordered_notes}


def _write_reference_xml_and_midi(
    xml_bytes: bytes,
    xml_path: Path,
    midi_path: Path,
) -> None:
    import music21

    xml_path.parent.mkdir(parents=True, exist_ok=True)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_bytes(xml_bytes)
    score = music21.converter.parse(xml_path)
    for part in score.parts:
        for measure in part.getElementsByClass(music21.stream.Measure):
            if isinstance(measure.leftBarline, music21.repeat.RepeatMark):
                measure.leftBarline = None
            if isinstance(measure.rightBarline, music21.repeat.RepeatMark):
                measure.rightBarline = None
        for marker in list(
            part.recurse().getElementsByClass(music21.repeat.RepeatMark)
        ):
            if marker.activeSite is not None:
                marker.activeSite.remove(marker)
        for bracket in list(
            part.recurse().getElementsByClass(music21.spanner.RepeatBracket)
        ):
            if bracket.activeSite is not None:
                bracket.activeSite.remove(bracket)
    write_score_midi(score, midi_path)
    if not midi_path.is_file():
        raise RuntimeError(f"music21 did not write reference MIDI: {midi_path}")


def build_native_reference(
    *,
    asap_root: Path,
    metadata_path: Path,
    annotations_path: Path,
    adjudications_path: Path,
    output_root: Path,
    mapping_policy: str = "source-faithful",
    dry_run: bool = False,
    expected_count: int | None = 102,
    require_audio: bool = True,
) -> dict[str, Any]:
    import music21

    records = _native_mapping_records(
        asap_root=asap_root,
        metadata_path=metadata_path,
        annotations_path=annotations_path,
        adjudications_path=adjudications_path,
        mapping_policy=mapping_policy,
        expected_count=expected_count,
        require_audio=require_audio,
    )
    metadata_hash = sha256_file(metadata_path)
    annotations_hash = sha256_file(annotations_path)
    adjudications_hash = (
        sha256_file(adjudications_path)
        if mapping_policy == "approved-fallback" else "inactive"
    )
    input_hash = _reference_input_fingerprint(
        records,
        metadata_sha256=metadata_hash,
        annotations_sha256=annotations_hash,
        adjudications_sha256=adjudications_hash,
        music21_version=music21.__version__,
    )
    provenance_counts: dict[str, int] = {}
    for record in records:
        provenance = record["mapping_provenance"]
        provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
    planned_windows = sum(
        len(valid_five_bar_positions(
            record["source_measure_groups"], record["undefined_positions"]
        ))
        for record in records
    )
    preview = {
        "recordings": len(records),
        "pieces": len({record["piece_id"] for record in records}),
        "valid_five_bar_windows": planned_windows,
        "mapping_provenance": dict(sorted(provenance_counts.items())),
        "mapping_policy": mapping_policy,
        "reference_input_sha256": input_hash,
    }
    if dry_run:
        return preview

    if output_root.exists():
        raise FileExistsError(
            f"Native reference output already exists; choose a new output directory: {output_root}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(
        dir=output_root.parent,
        prefix=f".{output_root.name}.",
    ))
    try:
        grounding: list[dict[str, Any]] = []
        published_records: list[dict[str, Any]] = []
        for record in tqdm(records, desc="native reference", unit="recording"):
            groups = record["source_measure_groups"]
            stem = record["artifact_stem"]
            full_relative = Path("full_xml") / f"{stem}.musicxml"
            full_path = temporary_root / full_relative
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_bytes(Path(record["source_xml"]).read_bytes())
            music21.converter.parse(full_path)

            published_record = dict(record)
            published_record["full_reference_xml"] = full_relative.as_posix()
            published_record["full_reference_xml_sha256"] = sha256_file(full_path)
            published_records.append(published_record)

            for position in valid_five_bar_positions(
                groups, record["undefined_positions"]
            ):
                chunk_id = f"{record['recording_id']}.{position}"
                artifact = f"{stem}.{position}"
                xml_relative = Path("window_xml") / f"{artifact}.musicxml"
                midi_relative = Path("window_midi") / f"{artifact}.mid"
                _write_reference_xml_and_midi(
                    performance_order_musicxml_bytes(
                        Path(record["source_xml"]),
                        groups[position:position + 5],
                    ),
                    temporary_root / xml_relative,
                    temporary_root / midi_relative,
                )
                grounding.append({
                    "chunk_id": chunk_id,
                    "recording_id": record["recording_id"],
                    "piece_id": record["piece_id"],
                    "performance_id": record["performance_id"],
                    "position_index": position,
                    "start_bar_index": position,
                    "end_bar_index_exclusive": position + 5,
                    "bar_count_authority": "frozen_mapping",
                    "mapping_provenance": record["mapping_provenance"],
                    "source_measure_ordinals": groups[position:position + 5],
                    "reference_status": "ready",
                    "gt_midi_status": "ready",
                    "reference_xml": xml_relative.as_posix(),
                    "reference_xml_sha256": sha256_file(
                        temporary_root / xml_relative
                    ),
                    "reference_midi": midi_relative.as_posix(),
                    "reference_midi_sha256": sha256_file(
                        temporary_root / midi_relative
                    ),
                    "gt_midi": midi_relative.as_posix(),
                })

        grounding.sort(key=lambda row: (
            row["recording_id"], row["position_index"]
        ))
        _write_jsonl(temporary_root / "mapping.jsonl", published_records)
        mapping_summary = {
            **preview,
            "music21_version": music21.__version__,
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": metadata_hash,
            "annotations_path": str(annotations_path.resolve()),
            "annotations_sha256": annotations_hash,
            "adjudications_path": (
                str(adjudications_path.resolve())
                if mapping_policy == "approved-fallback" else None
            ),
            "adjudications_sha256": (
                adjudications_hash
                if mapping_policy == "approved-fallback" else None
            ),
            "mapping_inventory_sha256": sequence_sha256(published_records),
        }
        _write_json(temporary_root / "mapping-summary.json", mapping_summary)
        _write_jsonl(temporary_root / "grounding.jsonl", grounding)
        grounding_summary = {
            "recordings": len({row["recording_id"] for row in grounding}),
            "windows": len(grounding),
            "ordered_inventory_sha256": sequence_sha256(grounding),
            "reference_input_sha256": input_hash,
        }
        _write_json(temporary_root / "grounding-summary.json", grounding_summary)
        # The one-out sensitivity grounding names a recording of ASAP-102;
        # another inventory has no such counterpart.
        if expected_count == 102:
            sensitivity = [
                row for row in grounding if row["performance_id"] != "Sekino05M"
            ]
            if len({row["recording_id"] for row in sensitivity}) != 101:
                raise RuntimeError("ASAP-101 must exclude exactly Sekino05M")
            _write_jsonl(temporary_root / "grounding-asap101.jsonl", sensitivity)
            _write_json(
                temporary_root / "grounding-asap101-summary.json",
                {
                    "recordings": 101,
                    "windows": len(sensitivity),
                    "excluded_recording": next(
                        record["recording_id"]
                        for record in records
                        if record["performance_id"] == "Sekino05M"
                    ),
                    "ordered_inventory_sha256": sequence_sha256(sensitivity),
                    "reference_input_sha256": input_hash,
                },
            )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return {**preview, "status": "built", "output_root": str(output_root)}


def _load_native_grounding(
    grounding_path: Path,
    gt_score_midi_root: Path,
) -> list[dict[str, Any]]:
    rows = _load_jsonl(grounding_path, "chunk_id")
    for row in rows:
        if row.get("reference_status") != "ready":
            raise RuntimeError(f"Reference is not ready: {row['chunk_id']}")
        for path_key, hash_key in (
            ("reference_xml", "reference_xml_sha256"),
            ("reference_midi", "reference_midi_sha256"),
        ):
            relative = row.get(path_key)
            expected_hash = row.get(hash_key)
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise RuntimeError(
                    f"Incomplete reference provenance: {row['chunk_id']}"
                )
            path = gt_score_midi_root / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"Reference artifact is missing for {row['chunk_id']}: {path}"
                )
            if sha256_file(path) != expected_hash:
                raise RuntimeError(
                    f"Reference artifact hash changed: {row['chunk_id']}:{path_key}"
                )
    return rows


def _resolve_manifest_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def evaluate_native_manifest(
    *,
    grounding_path: Path,
    gt_score_midi_root: Path,
    prediction_manifest: Path,
    prediction_root: Path,
    mv2h_bin: str,
    timeout: int,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grounding = _load_native_grounding(grounding_path, gt_score_midi_root)
    predictions = {
        row["chunk_id"]: row
        for row in _load_jsonl(prediction_manifest, "chunk_id")
    }
    expected_ids = {row["chunk_id"] for row in grounding}
    extras = sorted(set(predictions) - expected_ids)
    if extras:
        raise ValueError(
            f"Prediction manifest has {len(extras)} unknown chunks; first={extras[:3]}"
        )

    immediate: dict[str, EvalResult] = {}
    tasks: list[ChunkEvalTask] = []
    for reference in grounding:
        chunk_id = reference["chunk_id"]
        prediction = predictions.get(chunk_id)
        reference_midi = gt_score_midi_root / reference["reference_midi"]
        if prediction is None:
            immediate[chunk_id] = EvalResult(
                task_id=chunk_id,
                pred_path="",
                gt_path=str(reference_midi),
                status="missing_system_output",
                error_message="Prediction manifest has no row for this reference window",
            )
            continue
        declared_status = prediction.get("status", "ready")
        prediction_midi = _resolve_manifest_path(
            prediction_root, prediction.get("prediction_midi")
        )
        if declared_status != "ready":
            immediate[chunk_id] = EvalResult(
                task_id=chunk_id,
                pred_path=str(prediction_midi or ""),
                gt_path=str(reference_midi),
                status=str(declared_status),
                error_message=str(prediction.get("error_message", "")),
            )
            continue
        if prediction_midi is None or not prediction_midi.is_file():
            immediate[chunk_id] = EvalResult(
                task_id=chunk_id,
                pred_path=str(prediction_midi or ""),
                gt_path=str(reference_midi),
                status="missing_system_output",
                error_message="Prediction MIDI does not exist",
            )
            continue
        tasks.append(ChunkEvalTask(
            task_id=chunk_id,
            pred_midi=str(prediction_midi),
            gt_midi=str(reference_midi),
            mv2h_bin=mv2h_bin,
            timeout=timeout,
        ))

    evaluated: dict[str, EvalResult] = {}
    n_workers = min(max(workers, 1), max(len(tasks), 1))
    if n_workers == 1:
        for task in tqdm(tasks, desc="native MV2H", unit="window"):
            evaluated[task.task_id] = evaluate_chunk_task(task)
    elif tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(evaluate_chunk_task, task): task for task in tasks
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="native MV2H", unit="window",
            ):
                task = futures[future]
                try:
                    evaluated[task.task_id] = future.result()
                except Exception as error:
                    evaluated[task.task_id] = EvalResult(
                        task_id=task.task_id,
                        pred_path=task.pred_midi,
                        gt_path=task.gt_midi,
                        status="executor_error",
                        error_message=str(error),
                    )

    result_rows: list[dict[str, Any]] = []
    for reference in grounding:
        result = immediate.get(reference["chunk_id"]) or evaluated[reference["chunk_id"]]
        metrics = result.metrics if result.status == "success" else None
        values = metrics.to_dict() if metrics is not None else {}
        metric_values = {
            key: float(values.get(key, 0.0))
            for key in ("Multi-pitch", "Voice", "Meter", "Value", "Harmony")
        }
        mv2h4 = sum(metric_values[key] for key in (
            "Multi-pitch", "Voice", "Value", "Harmony"
        )) / 4
        mv2h5 = sum(metric_values.values()) / 5
        result_rows.append({
            "chunk_id": reference["chunk_id"],
            "recording_id": reference["recording_id"],
            "piece_id": reference["piece_id"],
            "performance_id": reference["performance_id"],
            "position_index": reference["position_index"],
            "status": result.status,
            "error_message": result.error_message,
            **metric_values,
            "MV2H4": mv2h4,
            "MV2H5": mv2h5,
            "prediction_midi": result.pred_path,
            "reference_midi": result.gt_path,
        })

    total = len(result_rows)
    status_counts: dict[str, int] = {}
    for row in result_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    strict = {
        metric: sum(float(row[metric]) for row in result_rows) / total
        if total else 0.0
        for metric in (
            "Multi-pitch", "Voice", "Meter", "Value", "Harmony",
            "MV2H4", "MV2H5",
        )
    }
    successful = [row for row in result_rows if row["status"] == "success"]
    successful_only = {
        metric: sum(float(row[metric]) for row in successful) / len(successful)
        if successful else 0.0
        for metric in strict
    }
    summary = {
        "n_total": total,
        "n_successful": len(successful),
        "status_counts": dict(sorted(status_counts.items())),
        "formal_failures_as_zero": strict,
        "diagnostic_successful_only": successful_only,
        "grounding_sha256": sha256_file(grounding_path),
        "prediction_manifest_sha256": sha256_file(prediction_manifest),
    }
    return result_rows, summary


def write_native_results(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NATIVE_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_csv)
    _write_json(output_csv.with_suffix(".summary.json"), summary)


def _run_native_action(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[3]
    if not args.asap_root:
        raise SystemExit("--asap-root: the asap-dataset checkout is required for this action")
    asap_root = Path(args.asap_root)
    metadata = Path(
        args.asap102_metadata
        or project_root / "src/datasets/asap/metadata_R.csv"
    )
    annotations = Path(
        args.annotations or asap_root / "asap_annotations.json"
    )
    adjudications = Path(
        args.adjudications
        or Path(__file__).with_name("asap102_mapping_adjudications.json")
    )
    gt_score_midi_root = Path(
        args.gt_score_midi_root
        or project_root / "data/experiments/asap102/native_reference"
    )

    if args.native_action == "build-reference":
        summary = build_native_reference(
            asap_root=asap_root,
            metadata_path=metadata,
            annotations_path=annotations,
            adjudications_path=adjudications,
            output_root=gt_score_midi_root,
            mapping_policy=args.mapping_policy,
            dry_run=args.dry_run,
            expected_count=args.expected_recordings,
            require_audio=not args.allow_missing_audio,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.native_action == "validate-reference":
        grounding = Path(args.grounding or gt_score_midi_root / "grounding.jsonl")
        summary_path = gt_score_midi_root / "mapping-summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Mapping summary does not exist: {summary_path}"
            )
        mapping_summary = json.loads(summary_path.read_text())
        if mapping_summary.get("recordings") != args.expected_recordings:
            raise RuntimeError(
                f"Reference must contain {args.expected_recordings} "
                f"recordings: {summary_path}"
            )
        input_hash = mapping_summary.get("reference_input_sha256")
        if not isinstance(input_hash, str) or not _validate_reference(
            gt_score_midi_root, input_hash
        ):
            raise RuntimeError(
                f"Native reference inventory or artifact hashes do not match: "
                f"{gt_score_midi_root}"
            )
        rows = _load_jsonl(grounding, "chunk_id")
        summary = {
            "recordings": len({row["recording_id"] for row in rows}),
            "windows": len(rows),
            "grounding_sha256": sha256_file(grounding),
            "reference_input_sha256": input_hash,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.native_action == "evaluate":
        if args.prediction_manifest is None:
            raise ValueError("--prediction-manifest is required for native evaluation")
        if args.output is None:
            raise ValueError("--output is required for native evaluation")
        grounding = Path(args.grounding or gt_score_midi_root / "grounding.jsonl")
        prediction_manifest = Path(args.prediction_manifest)
        prediction_root = Path(
            args.prediction_root or prediction_manifest.parent
        )
        mv2h_bin = args.mv2h_bin or "external/MV2H/bin"
        if not Path(mv2h_bin).exists():
            raise FileNotFoundError(f"MV2H bin not found: {mv2h_bin}")
        rows, summary = evaluate_native_manifest(
            grounding_path=grounding,
            gt_score_midi_root=gt_score_midi_root,
            prediction_manifest=prediction_manifest,
            prediction_root=prediction_root,
            mv2h_bin=mv2h_bin,
            timeout=args.chunk_timeout or 300,
            workers=args.workers or os.cpu_count() or 4,
        )
        write_native_results(rows, summary, Path(args.output))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unknown native action: {args.native_action}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="ASAP MV2H Evaluation (MuseScore 4.6.5 + MV2H)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full song evaluation
  python -m src.evaluation.asap.eval_asap_native_gt --mode full \\
      --pred_dir data/prediction_midi --gt_dir /path/to/asap \\
      --mv2h_bin external/MV2H/bin --output results/full.csv

  # 5-bar chunk evaluation
  python -m src.evaluation.asap.eval_asap_native_gt --mode chunks \\
      --pred_dir data/prediction_midi --gt_dir /path/to/asap \\
      --chunk_csv /path/to/zeng_test_chunk_set.csv \\
      --mv2h_bin external/MV2H/bin --output results/chunks.csv

  # Using YAML config
  python -m src.evaluation.asap.eval_asap_native_gt --config configs/asap_evaluate.yaml
        """,
    )

    # Config file
    parser.add_argument(
        "--config",
        help="YAML config file (CLI args override config values)",
    )

    parser.add_argument(
        "--native-action",
        choices=("build-reference", "validate-reference", "evaluate"),
        help="Use the frozen ASAP-102 native reference protocol",
    )
    parser.add_argument("--asap-root")
    parser.add_argument("--asap102-metadata")
    parser.add_argument("--annotations")
    parser.add_argument("--adjudications")
    parser.add_argument(
        "--mapping-policy",
        choices=("source-faithful", "approved-fallback"),
        default="source-faithful",
    )
    parser.add_argument("--gt-score-midi-root")
    parser.add_argument("--expected-recordings", type=int, default=102)
    parser.add_argument(
        "--allow-missing-audio", action="store_true",
        help="Accept rows whose ASAP-102 metadata audio cell is empty (MIDI-only sets)",
    )
    parser.add_argument("--grounding")
    parser.add_argument("--prediction-manifest")
    parser.add_argument("--prediction-root")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and report the native inventory without writing artifacts",
    )

    # Mode
    parser.add_argument("--mode", choices=["full", "chunks"], default=None)
    parser.add_argument("--pred_dir", help="Prediction MIDI directory")
    parser.add_argument("--gt_dir", help="ASAP dataset directory")
    parser.add_argument("--chunk_csv", help="Chunk CSV (for chunks mode)")

    # Paths
    parser.add_argument("--mv2h_bin", help="MV2H bin directory")
    parser.add_argument("--mscore_bin", help="MuseScore binary")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--output_dir", help="Output directory for MusicXML and chunks")
    parser.add_argument(
        "--cache_dir",
        help="MusicXML cache dir (defaults to {output_dir}/full_musicxml). "
             "Set explicitly to share cache across runs or separate baselines.",
    )

    # Processing
    parser.add_argument("-j", "--workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="MV2H timeout (sec)")
    parser.add_argument("--chunk_timeout", type=int, default=None, help="Chunk MV2H timeout (sec)")
    parser.add_argument("--mscore_timeout", type=int, default=None)

    # Retry
    parser.add_argument("--retry_file", help="Retry file path")
    parser.add_argument("--save_failed", help="Save failed tasks path")

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.native_action is not None:
        _run_native_action(args)
        return

    # Load YAML config if provided
    file_config: Dict[str, Any] = {}
    if args.config:
        if not Path(args.config).exists():
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        file_config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")

    # Merge config: CLI args override file config
    def get_value(cli_val, config_key, default=None):
        if cli_val is not None:
            return cli_val
        return file_config.get(config_key, default)

    mode = get_value(args.mode, "mode", "full")
    pred_dir = get_value(args.pred_dir, "pred_dir")
    gt_dir = get_value(args.gt_dir, "gt_dir")
    chunk_csv = get_value(args.chunk_csv, "chunk_csv")
    mv2h_bin = get_value(args.mv2h_bin, "mv2h_bin", "external/MV2H/bin")
    mscore_bin = get_value(args.mscore_bin, "mscore_bin", "tools/mscore")
    output_path = get_value(args.output, "output_csv", "results/asap_eval.csv")
    output_dir = get_value(args.output_dir, "output_dir")
    cache_dir = get_value(args.cache_dir, "cache_dir")
    workers = get_value(args.workers, "workers", os.cpu_count() or 4)
    timeout = get_value(args.timeout, "timeout", 120)
    chunk_timeout = get_value(args.chunk_timeout, "chunk_timeout", 300)
    mscore_timeout = get_value(args.mscore_timeout, "mscore_timeout", 60)

    # Determine mode
    retry_mode = args.retry_file is not None

    # Validate
    if not retry_mode:
        if not pred_dir:
            parser.error("--pred_dir required (or use --retry_file or --config)")
        if not gt_dir:
            parser.error("--gt_dir required (or use --retry_file or --config)")
        if mode == "chunks" and not chunk_csv:
            parser.error("--chunk_csv required for chunks mode")

        if not Path(pred_dir).exists():
            print(f"Prediction directory not found: {pred_dir}")
            sys.exit(1)
        if not Path(gt_dir).exists():
            print(f"Ground truth directory not found: {gt_dir}")
            sys.exit(1)
    else:
        if not Path(args.retry_file).exists():
            print(f"Retry file not found: {args.retry_file}")
            sys.exit(1)

    if not Path(mv2h_bin).exists():
        print(f"MV2H bin not found: {mv2h_bin}")
        sys.exit(1)

    if not Path(mscore_bin).exists():
        print(f"MuseScore not found: {mscore_bin}")
        sys.exit(1)

    # Set default output_dir from output path
    if output_dir is None:
        output_dir = str(Path(output_path).parent)

    # Config
    config = EvalConfig(
        mv2h_bin=mv2h_bin,
        mv2h_timeout=timeout,
        mv2h_chunk_timeout=chunk_timeout,
        mscore_config=MuseScoreConfig(binary_path=mscore_bin, timeout=mscore_timeout),
        workers=workers,
        cache_dir=cache_dir,
    )

    # Print config
    print("=" * 70)
    print("ASAP MV2H Evaluation (MuseScore 4.6.5 + MV2H)")
    print("=" * 70)
    print(f"Mode:           {'RETRY' if retry_mode else mode}")
    print(f"Workers:        {workers}")
    if mode == "chunks":
        print(f"Chunk timeout:  {chunk_timeout}s (MV2H), {mscore_timeout}s (MuseScore)")
    else:
        print(f"Timeout:        {timeout}s (MV2H), {mscore_timeout}s (MuseScore)")
    print(f"Prediction dir: {pred_dir}")
    print(f"Ground truth:   {gt_dir}")
    print(f"Output:         {output_path}")
    if mode == "chunks":
        print(f"Chunk CSV:      {chunk_csv}")
    print("=" * 70)

    # Run evaluation
    if retry_mode:
        tasks = load_retry_tasks(args.retry_file, output_dir, config)
        results = run_parallel_evaluation(tasks, config.workers)
        summary = compute_summary(results)
        # Print standard summary for retry mode
        print_mv2h_summary(summary, "MV2H (retry)")

    elif mode == "full":
        results = run_full_song_evaluation(pred_dir, gt_dir, output_dir, config)
        summary = compute_summary(results)
        # Save failed for full mode
        failed_path = args.save_failed or os.path.join(output_dir, "failed.txt")
        n_failed = save_failed_tasks(results, failed_path)
        if n_failed > 0:
            print(f"\nFailed tasks saved to: {failed_path}")
            print(f"Retry: python -m src.evaluation.asap.eval_asap_native_gt --retry_file {failed_path} --timeout 300 ...")
        # Print standard summary
        print_mv2h_summary(summary, "MV2H (full)")

    else:  # chunks mode
        results, summary = run_chunk_evaluation(
            pred_dir, gt_dir, chunk_csv, output_dir, config,
            output_csv=output_path,  # Enable incremental saving
        )
        # Print chunk-specific summary (both methods)
        print_chunk_summary(summary)
        # Note: Results are saved incrementally during evaluation

    # For full mode, save results (chunk mode saves incrementally)
    if mode == "full":
        save_results_csv(results, output_path, is_chunk_mode=False)

    # Save summary JSON
    summary_path = Path(output_path).with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults CSV:  {output_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
