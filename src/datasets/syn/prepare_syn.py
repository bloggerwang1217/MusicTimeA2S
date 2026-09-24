"""
Syn dataset Preprocessing Pipeline
=======================================

Prepares training data for Syn dataset model:
- Phase 1: Score → Kern (MuseSyn XML + HumSyn kern)
- Phase 2: Canonical MusicXML → MIDI → Audio (with data augmentation)
- Phase 2.5: Audio → Mel Spectrogram
- Phase 3: Create training manifests (train/valid/test)

Data augmentation:
- Tempo scaling (four equal log2 strata inside one square-root-of-two of the
  native tempo, sampled once each and randomly assigned to render versions)
- Multi-soundfont synthesis (one of each training timbre per work, randomly
  assigned independently of tempo)
- Fixed -27 LUFS integrated loudness normalization without compression

Usage:
    poetry run python src/datasets/syn/prepare_syn.py --phase 1      # Score → Kern
    poetry run python src/datasets/syn/prepare_syn.py --phase 2      # Kern → Audio
    poetry run python src/datasets/syn/prepare_syn.py --phase 2.5    # Audio → Mel
    poetry run python src/datasets/syn/prepare_syn.py --phase 3      # Create manifests
    poetry run python src/datasets/syn/prepare_syn.py --phase 2 --workers 8  # Parallel
    poetry run python src/datasets/syn/prepare_syn.py                # Full pipeline (1,1.5,2,2.5,3)
"""

import hashlib
import json
import logging
import math
import os
import re

# Keep nested numerical libraries inside the CPU budget implied by --workers.
for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "4")

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import converter21
import music21 as m21
import pandas as pd
from tqdm import tqdm

from src.audio.mel import process_audio_batch
from src.audio.render_timing import (
    attach_measure_supervision_flags,
    build_tempo_map,
    epr_render_fingerprint,
    epr_seconds_at,
    extract_beat_offsets,
    extract_measure_offsets,
    extract_phase_grid_offsets,
    initial_qpm,
    inject_measure_markers,
    read_beat_times_from_midi,
    read_epr_render_fingerprint,
    read_grid_times_from_midi,
    read_measure_times_from_midi,
    scaled_score_tempo,
)
from src.a2s.piano.foundation import HFT_MEL
from src.a2s.piano.generate_augmentation_metadata import generate_metadata
from src.audio.render_epr import EPR_STYLES, EPRError, EPRTimeout, render_epr
from src.datasets.syn.syn_manifest import create_manifest, load_protocol_exclude
from src.score.tempo import parse_omd_tempo
from src.preprocessing.humsyn_processor import HumSynProcessor
from src.preprocessing.musesyn_processor import MuseSynProcessor
from src.utils import set_seed, SEED_DATA_AUGMENTATION

# Register converter21 for robust humdrum parsing
converter21.register()

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_HUMSYN_DIR = Path("data/datasets/HumSyn")
DEFAULT_MUSESYN_DIR = Path("data/datasets/MuseSyn")
DEFAULT_OUTPUT_DIR = Path("data/experiments/syn")
DEFAULT_METADATA_DIR = Path("src/datasets/syn")
DEFAULT_SOUNDFONT_DIR = Path("data/soundfonts/piano")
DEFAULT_AUG_CONFIG = Path("src/audio/augmentation.json")

# Render the score's own tempo; no performance model in the chain.
NO_EPR: Optional[str] = None


def load_augmentation_config(config_path: Path = DEFAULT_AUG_CONFIG) -> Dict[str, Any]:
    """Load augmentation settings from JSON config.

    Returns:
        Dictionary with keys:
        - tempo_enabled: bool
        - tempo_range: tuple (min_scale, max_scale)
        - tempo_sampling: stratified_log2
        - tempo_max_log2_deviation: half-width around the native tempo
        - tempo_num_strata: one stratum per train render
        - train_soundfonts: list
        - valid_soundfonts: list
        - test_soundfonts: list
        - num_versions: dict
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Tempo augmentation settings
    tempo_config = config.get('tempo', {})
    tempo_enabled = tempo_config.get('enabled', True)
    tempo_sampling = tempo_config.get('sampling', 'stratified_log2')
    if tempo_sampling != 'stratified_log2':
        raise ValueError(
            f"unsupported tempo sampling contract {tempo_sampling!r}"
        )
    tempo_max_log2_deviation = float(
        tempo_config.get('max_log2_deviation', 0.5)
    )
    if not math.isfinite(tempo_max_log2_deviation) or (
        tempo_max_log2_deviation <= 0
    ):
        raise ValueError("tempo.max_log2_deviation must be finite and positive")
    tempo_num_strata = int(tempo_config.get(
        'num_strata', config['num_versions']['train']
    ))
    if tempo_num_strata != int(config['num_versions']['train']):
        raise ValueError(
            "tempo.num_strata must equal num_versions.train so every work "
            "contributes one render from each log-tempo stratum"
        )
    tempo_range = (
        2.0 ** (-tempo_max_log2_deviation),
        2.0 ** tempo_max_log2_deviation,
    )
    epr = config['epr']
    styles = (
        [epr['default_style_by_prefix'][prefix] for prefix in epr['default_style_by_prefix']]
        + list(epr['style_pool']) + list(epr['test_styles'])
    )
    unknown = {s for s in styles if s is not None and s not in EPR_STYLES}
    if unknown:
        raise ValueError(f"styles the performance model does not know: {sorted(unknown)}")

    return {
        'epr': epr,
        'tempo_enabled': tempo_enabled,
        'tempo_range': tempo_range,
        'tempo_sampling': tempo_sampling,
        'tempo_max_log2_deviation': tempo_max_log2_deviation,
        'tempo_num_strata': tempo_num_strata,
        'train_soundfonts': config['soundfonts']['train'],
        'valid_soundfonts': config['soundfonts']['valid'],
        'test_soundfonts': config['soundfonts']['test'],
        'num_versions': config['num_versions'],
    }


def default_epr_style(stem: str, epr: Dict[str, Any]) -> Optional[str]:
    """Style of the piece's own version (NO_EPR = score-tempo render).

    Longest matching prefix wins, so the policy does not depend on the order
    the config happens to list its corpora in.
    """
    matches = [prefix for prefix in epr["default_style_by_prefix"]
               if stem.startswith(prefix)]
    if not matches:
        raise ValueError(f"no rendering policy for {stem!r}")
    return epr["default_style_by_prefix"][max(matches, key=len)]


def _factor_rng(stem: str, factor: str):
    """A deterministic RNG stream independent of every other render factor."""
    import numpy as np

    digest = hashlib.sha256(
        f"{SEED_DATA_AUGMENTATION}|{stem}|{factor}".encode("utf-8")
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _epr_style_plan(
    stem: str, split: str, n_versions: int, epr: Dict[str, Any],
) -> List[Optional[str]]:
    """One reproducible style per version, independent of other factors."""
    import random

    if split == "test":
        test_styles = epr["test_styles"]
        if n_versions > len(test_styles):
            raise ValueError(
                f"{n_versions} test versions beyond the fixed style assignment")
        return list(test_styles[:n_versions])
    if any(stem.startswith(prefix) for prefix in epr["never_epr_prefixes"]):
        return [NO_EPR] * n_versions
    # Preserve the established style draws while making their RNG local: the
    # prior implementation seeded Python's random module from this value,
    # whereas tempo used NumPy's independent stream.
    file_seed = int(hashlib.md5(stem.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(SEED_DATA_AUGMENTATION + file_seed)
    return [default_epr_style(stem, epr)] + [
        rng.choice(epr["style_pool"])
        for _ in range(1, n_versions)
    ]


def render_plan_for_work(
    stem: str,
    split: str,
    n_versions: int,
    soundfonts: List[str],
    aug_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Cross tempo, timbre and style without sharing their RNG streams."""
    import numpy as np

    styles = _epr_style_plan(stem, split, n_versions, aug_config['epr'])
    if aug_config['tempo_enabled'] and split == 'train':
        n_strata = aug_config['tempo_num_strata']
        width = 2.0 * aug_config['tempo_max_log2_deviation'] / n_strata
        draw_rng = _factor_rng(stem, "tempo-draw")
        log_scales = (
            -aug_config['tempo_max_log2_deviation']
            + width * (np.arange(n_strata) + draw_rng.random(n_strata))
        )
        assignment = _factor_rng(
            stem, "tempo-assignment"
        ).permutation(n_strata)
        version_log_scales = log_scales[assignment]
        strata = assignment.astype(int).tolist()
    else:
        version_log_scales = np.zeros(n_versions, dtype=np.float64)
        strata = [None] * n_versions

    if split == 'train':
        if len(soundfonts) != n_versions or len(set(soundfonts)) != n_versions:
            raise ValueError(
                "train soundfonts must contain one distinct entry per version"
            )
        font_assignment = _factor_rng(
            stem, "timbre-assignment"
        ).permutation(n_versions)
        version_fonts = [
            [soundfonts[int(font_assignment[version])]]
            for version in range(n_versions)
        ]
    elif split == 'valid':
        version_fonts = [[soundfonts[0]] for _ in range(n_versions)]
    else:
        version_fonts = [list(soundfonts) for _ in range(n_versions)]

    return [
        {
            "epr_style": styles[version],
            "tempo_log2": float(version_log_scales[version]),
            "tempo_scaling": float(2.0 ** version_log_scales[version]),
            "tempo_stratum": strata[version],
            "soundfonts": version_fonts[version],
        }
        for version in range(n_versions)
    ]


_TEMPO_OVERRIDES: Optional[Dict[str, List[Tuple[int, float, float]]]] = None
_TEMPO_OVERRIDES_PATH = Path(__file__).resolve().parent.parent.parent / "datasets" / "syn" / "tempo_overrides.csv"


def _load_tempo_overrides() -> Dict[str, List[Tuple[int, float, float]]]:
    """Curated tempo marks per file: (measure, referent quarterLength, bpm).

    A metronome mark is a bar, a note value and a number; carrying only the
    number forces a silent convention on the other two.  `referent` is a kern
    recip, so `4.` is a dotted quarter — 12/8 counted in dotted quarters and
    counted in quarters differ by a factor of 1.5.
    """
    global _TEMPO_OVERRIDES
    if _TEMPO_OVERRIDES is None:
        _TEMPO_OVERRIDES = {}
        if _TEMPO_OVERRIDES_PATH.exists():
            import csv
            from src.score.kern_utils import _DUR_TO_FRAC
            with open(_TEMPO_OVERRIDES_PATH) as f:
                for row in csv.DictReader(f):
                    recip = row["referent"]
                    if recip not in _DUR_TO_FRAC:
                        raise ValueError(
                            f"tempo_overrides: {row['file']} has unknown "
                            f"referent {recip!r}")
                    _TEMPO_OVERRIDES.setdefault(row["file"], []).append(
                        (int(row["measure"]),
                         float(_DUR_TO_FRAC[recip]),
                         float(row["bpm"])))
    return _TEMPO_OVERRIDES


_CUE_OVERRIDES: Optional[Dict[str, str]] = None
_CUE_OVERRIDES_PATH = Path(__file__).resolve().parent.parent.parent / "datasets" / "syn" / "cue_overrides.csv"


def _load_cue_overrides() -> Dict[str, str]:
    """Per-file *cue treatment: "strip" (orchestral/chamber reference cue,
    not played by the pianist) vs "keep" (cadenza, the default for any
    file absent from cue_overrides.csv — the pianist does play these).
    """
    global _CUE_OVERRIDES
    if _CUE_OVERRIDES is None:
        _CUE_OVERRIDES = {}
        if _CUE_OVERRIDES_PATH.exists():
            import csv
            with open(_CUE_OVERRIDES_PATH) as f:
                for row in csv.DictReader(f):
                    _CUE_OVERRIDES[row["file"]] = row["treatment"]
    return _CUE_OVERRIDES


def get_cue_treatment(stem: str) -> str:
    """Look up the *cue treatment for a given kern file stem."""
    return _load_cue_overrides().get(stem, "keep")


def _score_has_tempo(score: m21.stream.Score) -> bool:
    """Check if score has at least one MetronomeMark with a usable number."""
    for tm in score.flatten().getElementsByClass(m21.tempo.MetronomeMark):
        if tm.number is not None or tm.numberSounding is not None:
            return True
    return False


def ensure_tempo(score: m21.stream.Score, stem: str, kern_content: Optional[str] = None) -> None:
    """Give the score the tempo it should be rendered at.

    Resolution order:
    1. tempo_overrides.csv — an override, so it replaces whatever the file
       carries.  A curated row exists because the notated mark is missing or
       because the parser reads it wrong (an unrecognised note-name spelling
       silently becomes a quarter), and a row that loses to the value it was
       written to correct would be decoration.
    2. the score's own marks
    3. parse_omd_tempo() on kern !!!OMD records
    4. fallback 120 BPM
    """
    marks = list(_load_tempo_overrides().get(stem, []))

    if marks:
        for stream in [score] + list(score.recurse(streamsOnly=True)):
            for mark in list(stream.getElementsByClass(m21.tempo.MetronomeMark)):
                stream.remove(mark)
    elif _score_has_tempo(score):
        return

    if not marks and kern_content:
        for line in kern_content.splitlines():
            if line.startswith("!!!OMD"):
                val = line.split(":", 1)[1].strip() if ":" in line else ""
                resolved = parse_omd_tempo(val)
                if resolved is not None:
                    marks = [(1, 1.0, float(resolved))]
                    break

    if not marks:
        marks = [(1, 1.0, 120.0)]

    measures = list(score.parts[0].getElementsByClass(m21.stream.Measure))
    for measure_num, referent_ql, bpm in marks:
        # A mark printed at bar N sounds on every pass through it, so when
        # repeats are already expanded every measure of that number gets one.
        targets = [m for m in measures if m.number == measure_num]
        if not targets:
            raise ValueError(f"{stem}: no measure {measure_num} for tempo mark")
        for measure in targets:
            measure.insert(0.0, m21.tempo.MetronomeMark(
                number=bpm, referent=m21.duration.Duration(referent_ql)))
    logger.info(f"Set {len(marks)} tempo mark(s) for {stem}")


def extract_measure_times(score: m21.stream.Score) -> List[Dict[str, Any]]:
    """Extract measure timing from a music21 Score (single source of truth).

    Uses the Score's internal MetronomeMark objects and measure offsets.
    This matches exactly what score.write('midi') produces, so timing is
    guaranteed to align with the rendered audio.

    Args:
        score: music21 Score object (already parsed via converter21)

    Returns:
        List of measure info (in sequential order, may have duplicate numbers
        from repeat expansion):
        [
            {"measure": 1, "start_sec": 0.0, "end_sec": 1.25},
            {"measure": 2, "start_sec": 1.25, "end_sec": 2.50},
            ...
        ]
    """
    if not score.parts:
        return []

    part = score.parts[0]
    tempo_map = build_tempo_map(score)

    def offset_to_seconds(offset: float) -> float:
        """Convert quarter-note offset to seconds using tempo map."""
        seconds = 0.0
        prev_offset = 0.0
        prev_qpm = tempo_map[0][1]

        for t_offset, t_qpm in tempo_map:
            if t_offset >= offset:
                break
            seconds += (t_offset - prev_offset) * (60.0 / prev_qpm)
            prev_offset = t_offset
            prev_qpm = t_qpm

        seconds += (offset - prev_offset) * (60.0 / prev_qpm)
        return seconds

    # Extract ALL measures in sequential order (no dedup by number,
    # since repeat expansion can produce duplicate measure numbers)
    measures_info = []
    for measure in part.getElementsByClass(m21.stream.Measure):
        try:
            offset_start = float(measure.offset)
            offset_end = offset_start + float(measure.duration.quarterLength)

            measures_info.append({
                "measure": measure.number,
                "start_sec": round(offset_to_seconds(offset_start), 4),
                "end_sec": round(offset_to_seconds(offset_end), 4),
            })
        except Exception:
            continue

    return measures_info


def phase1_score_to_kern(
    humsyn_dir: Path = DEFAULT_HUMSYN_DIR,
    musesyn_dir: Path = DEFAULT_MUSESYN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    keep_dynam: bool = True,
    keep_grace: bool = False,
    keep_trill: bool = False,
    keep_non_trill_ornaments: bool = False,
    keep_arpeggio: bool = False,
    save_visual_info: bool = True,
    workers: int = 1,
) -> Dict[str, str]:
    """Phase 1: Convert scores to kern format.

    Args:
        humsyn_dir: Path to HumSyn directory
        musesyn_dir: Path to MuseSyn directory
        output_dir: Path to output directory
        metadata_dir: Path to metadata directory
        keep_dynam: Retain **dynam spines (default True — dynamics render
            into audio; Phase 1.5 strips them before tokenizing).
        keep_grace: Retain grace notes (default: stripped).
        keep_trill: Retain trill signifiers (default: stripped). VirtuosoNet
            reads trill-mark as an input feature, so this is a real
            VirtuosoNet ablation lever.
        keep_non_trill_ornaments: Retain mordent/turn/shake/schleifer
            signifiers (default: stripped). VirtuosoNet's parser never turns
            these into an input feature, so toggling this cannot change
            VirtuosoNet-rendered audio.
        keep_arpeggio: Retain arpeggio marks (default: stripped).
        save_visual_info: Whether to save visual info JSON files for Visual Aux Head
        workers: Number of parallel worker processes per corpus (default
            1 = sequential). HumSyn and MuseSyn are still processed one
            corpus after the other; only the per-file loop within each
            is parallelized.

    Returns:
        Dictionary of {filename: status} for all processed files
    """
    kern_output_dir = output_dir / "kern"
    kern_output_dir.mkdir(parents=True, exist_ok=True)

    # Visual info output directory (for Visual Auxiliary Head ground truth)
    visual_output_dir = output_dir / "visual" if save_visual_info else None
    if visual_output_dir:
        visual_output_dir.mkdir(parents=True, exist_ok=True)

    # MusicXML output directory (VirtuosoNet input) — produced alongside
    # kern/ from the same Clean+Expand+Cue, same keep_* flags.
    xml_output_dir = output_dir / "xml"
    xml_output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Process HumSyn
    logger.info("Processing HumSyn...")
    selected_chopin_path = metadata_dir / "selected_chopin.txt"
    repeat_map_dir = output_dir / "repeat_map"
    repeat_map_dir.mkdir(parents=True, exist_ok=True)

    humsyn_processor = HumSynProcessor(
        input_dir=humsyn_dir,
        output_dir=kern_output_dir,
        visual_dir=visual_output_dir,
        repeat_map_dir=repeat_map_dir,
        selected_chopin_path=selected_chopin_path,
        xml_dir=xml_output_dir,
        keep_dynam=keep_dynam,
        keep_grace=keep_grace,
        keep_trill=keep_trill,
        keep_non_trill_ornaments=keep_non_trill_ornaments,
        keep_arpeggio=keep_arpeggio,
    )
    humsyn_results = humsyn_processor.process_all(workers=workers)
    all_results.update(humsyn_results)

    # Process MuseSyn
    logger.info("Processing MuseSyn...")
    musesyn_processor = MuseSynProcessor(
        input_dir=musesyn_dir,
        output_dir=kern_output_dir,
        visual_dir=visual_output_dir,
        repeat_map_dir=repeat_map_dir,
        xml_dir=xml_output_dir,
        keep_dynam=keep_dynam,
        keep_grace=keep_grace,
        keep_trill=keep_trill,
        keep_non_trill_ornaments=keep_non_trill_ornaments,
        keep_arpeggio=keep_arpeggio,
    )
    musesyn_results = musesyn_processor.process_all(workers=workers)
    all_results.update(musesyn_results)

    return all_results


def _create_single_gt(args: Tuple) -> Tuple[str, str, str]:
    """Pool-dispatch wrapper: unpack the task tuple and resolve cue policy."""
    (kern_path, kern_gt_dir, inventory_dir, keep_grace, keep_trill,
     keep_non_trill_ornaments, keep_arpeggio) = args
    from src.score.standardize_kern import create_single_gt

    return create_single_gt(
        kern_path, kern_gt_dir, inventory_dir,
        keep_grace=keep_grace,
        keep_trill=keep_trill,
        keep_non_trill_ornaments=keep_non_trill_ornaments,
        keep_arpeggio=keep_arpeggio,
        strip_cue=(get_cue_treatment(kern_path.stem) == "strip"),
    )


def create_ground_truth_kern(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    keep_grace: bool = False,
    keep_trill: bool = False,
    keep_non_trill_ornaments: bool = False,
    keep_arpeggio: bool = False,
    workers: int = 1,
) -> Dict[str, str]:
    """Create ground truth kern files.

    Reads kern/ (repeat-expanded, Phase 1 output), drops non-kern spines
    (**dynam is audio-only), standardizes, and stores the deterministic
    canonical-writer text.  The worker independently reconstructs it again;
    success requires exact text equality.  Incomplete tuplets, metric timeline
    errors, tokenizer OOV, and writer contract errors are reported separately.

    Args:
        output_dir:  Base output directory (reads from kern/, writes to
                     kern_gt/)
        keep_grace / keep_trill / keep_non_trill_ornaments / keep_arpeggio:
            retain those signifiers in the ground truth (default: stripped).
            Only effective if Phase 1 kept them in kern/ too.
        workers: Number of parallel workers (1 = sequential)

    Returns:
        Dictionary of {filename: status}
    """
    kern_dir = output_dir / "kern"
    kern_gt_dir = output_dir / "kern_gt"
    inventory_dir = output_dir / "canonicalization-diffs"
    kern_gt_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)
    for stale in inventory_dir.glob("*.diff"):
        stale.unlink()

    results = {}
    diff_family_by_stem: Dict[str, str] = {}
    kern_files = sorted(kern_dir.glob("*.krn"))
    tasks = [
        (kern_path, kern_gt_dir, inventory_dir, keep_grace, keep_trill,
         keep_non_trill_ornaments, keep_arpeggio)
        for kern_path in kern_files
    ]

    logger.info(f"Creating ground truth kern files: {len(kern_files)} files")

    if workers <= 1:
        for task in tqdm(tasks, desc="Creating kern_gt"):
            stem, status, families = _create_single_gt(task)
            if status != "success":
                logger.error(f"Ground-truth status {stem}: {status}")
            results[stem] = status
            diff_family_by_stem[stem] = families
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_create_single_gt, t): t[0].stem
                       for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Creating kern_gt"):
                stem, status, families = future.result()
                if status != "success":
                    logger.error(f"Ground-truth status {stem}: {status}")
                results[stem] = status
                diff_family_by_stem[stem] = families

    # Summary
    success = sum(1 for v in results.values() if v == "success")
    oov = sum(1 for value in results.values() if value.startswith("oov:"))
    incomplete_tuplets = sum(
        1 for value in results.values()
        if value.startswith("incomplete_tuplet:")
    )
    metric_timeline_errors = sum(
        1 for value in results.values()
        if value.startswith("metric_timeline:")
    )
    writer_errors = sum(
        1 for value in results.values() if value.startswith("writer_error:")
    )
    logger.info(
        "Ground truth creation complete: %d canonical, %d incomplete "
        "tuplets, %d metric timeline errors, %d OOV, %d writer errors",
        success, incomplete_tuplets, metric_timeline_errors, oov,
        writer_errors,
    )

    status_path = output_dir / "canonicalization-status.tsv"
    status_tmp = status_path.with_name(f".{status_path.name}")

    def _tsv_cell(value: str) -> str:
        # A worker exception may contain source whitespace; keep the ledger
        # one-record-per-line so it remains mechanically auditable.
        return (value.replace("\t", "\\t")
                .replace("\r", "\\r")
                .replace("\n", "\\n"))

    status_tmp.write_text(
        "file\tstatus\tdetail\tdiff_families\n" + "".join(
            f"{_tsv_cell(stem)}\t{_tsv_cell(status.partition(':')[0])}\t"
            f"{_tsv_cell(status.partition(':')[2].strip())}\t"
            f"{_tsv_cell(diff_family_by_stem.get(stem, ''))}\n"
            for stem, status in sorted(results.items())
        ),
        encoding="utf-8",
    )
    os.replace(status_tmp, status_path)

    summary_path = output_dir / "canonicalization-summary.json"
    summary_tmp = summary_path.with_name(f".{summary_path.name}")
    family_counts = Counter(
        family
        for families in diff_family_by_stem.values()
        for family in families.split(',')
        if family
    )
    summary_tmp.write_text(json.dumps({
        "total": len(results),
        "success": success,
        "incomplete_tuplet": incomplete_tuplets,
        "metric_timeline": metric_timeline_errors,
        "oov": oov,
        "writer_error": writer_errors,
        "text_diff_files": len(list(inventory_dir.glob("*.diff"))),
        "diff_family_file_counts": dict(sorted(family_counts.items())),
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(summary_tmp, summary_path)

    return results


def _merge_contiguous_voice_ties_for_midi(score) -> int:
    """Merge complete, adjacent tie chains inside measure-local Voices."""
    merged = 0

    def tie_type(element) -> Optional[str]:
        notes = (element.notes
                 if isinstance(element, m21.chord.Chord) else [element])
        types = {
            note.tie.type if note.tie is not None else None
            for note in notes
        }
        return next(iter(types)) if len(types) == 1 else None

    def pitch_key(element) -> Tuple[int, ...]:
        return tuple(sorted(pitch.midi for pitch in element.pitches))

    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            for voice in measure.voices:
                elements = sorted(
                    voice.notesAndRests,
                    key=lambda element: Fraction(
                        voice.elementOffset(element)).limit_denominator(
                            10 ** 7),
                )
                index = 0
                while index < len(elements):
                    first = elements[index]
                    if (not isinstance(first, m21.note.NotRest)
                            or tie_type(first) != "start"):
                        index += 1
                        continue

                    chain = [first]
                    expected_offset = (
                        Fraction(voice.elementOffset(first)).limit_denominator(
                            10 ** 7)
                        + Fraction(first.duration.quarterLength).limit_denominator(
                            10 ** 7)
                    )
                    cursor = index + 1
                    complete = False
                    while cursor < len(elements):
                        following = elements[cursor]
                        following_offset = Fraction(
                            voice.elementOffset(following)).limit_denominator(
                                10 ** 7)
                        if (not isinstance(following, m21.note.NotRest)
                                or following_offset != expected_offset
                                or pitch_key(following) != pitch_key(first)):
                            break
                        following_tie = tie_type(following)
                        if following_tie not in {"continue", "stop"}:
                            break
                        chain.append(following)
                        expected_offset += Fraction(
                            following.duration.quarterLength).limit_denominator(
                                10 ** 7)
                        if following_tie == "stop":
                            complete = True
                            break
                        cursor += 1

                    if not complete:
                        index += 1
                        continue

                    first.duration.quarterLength = sum(
                        (element.duration.quarterLength for element in chain),
                        Fraction(0),
                    )
                    first.tie = None
                    if isinstance(first, m21.chord.Chord):
                        for note in first.notes:
                            note.tie = None
                    for element in chain[1:]:
                        voice.remove(element)
                    merged += len(chain) - 1
                    elements = [
                        element for position, element in enumerate(elements)
                        if not index < position <= cursor
                    ]
                    index += 1

    return merged


def _process_single_kern(args: Tuple) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """Worker function for parallel processing a single kern file.

    Args:
        args: Tuple of (kern_path, output_dir, soundfont_dir, split,
                        num_versions, available_soundfonts, aug_config)

    Returns:
        Tuple of:
        - results: Dictionary of {audio_name: status}
        - alignment: Dictionary of {audio_key: alignment_info}
          alignment_info contains: tempo_scaling, duration_sec, audio_measures
    """
    (
        kern_path,
        output_dir,
        soundfont_dir,
        split,
        num_versions,
        available_soundfonts,
        aug_config,
    ) = args

    # Import inside worker to avoid pickling issues
    import converter21
    import music21 as m21
    converter21.register()

    from midi2audio import FluidSynth
    from src.audio.reverb import draw_reverb_plan
    from src.audio.synthesis import audio_matches_contract, render_one_midi

    midi_dir = output_dir / "midi"
    audio_dir = output_dir / "audio"
    stem = kern_path.stem
    results = {}
    alignment_info = {}  # {audio_key: {tempo_scaling, duration_sec, audio_measures}}

    logger.debug(f"Processing: {stem}")

    # Set reproducible seed based on filename
    # NOTE: Use hashlib.md5 instead of hash() because Python's hash() is
    # randomized across interpreter sessions (PYTHONHASHSEED). This ensures
    # reproducible augmentation choices across multiple runs.
    file_seed = int(hashlib.md5(stem.encode()).hexdigest(), 16) % (2**32)
    set_seed(SEED_DATA_AUGMENTATION + file_seed)

    # Check if this is a MuseSyn file
    is_musesyn = stem.startswith("musesyn_")

    try:
        # Phase 1's canonical MusicXML is the rendering source for both
        # corpora.  It already contains cue/repeat/figure decisions and the
        # resolved tempo map, so Phase 2 must not rebuild either from kern or
        # tempo_overrides.csv.
        xml_stem = stem[8:] if is_musesyn else stem
        xml_path = output_dir / "xml" / f"{xml_stem}.xml"
        if not xml_path.exists():
            return {f"{stem}_v0": f"error: XML not found - {xml_path}"}, {}

        score = m21.converter.parse(str(xml_path))
        if not _score_has_tempo(score):
            return {
                f"{stem}_v0": f"error: canonical XML has no tempo - {xml_path}"
            }, {}
        if is_musesyn:
            _merge_contiguous_voice_ties_for_midi(score)

    except Exception as e:
        return {f"{stem}_v0": f"error: parse failed - {e}"}, {}

    # Ensure instruments are set
    for part in score.parts:
        if not part.getElementsByClass(m21.instrument.Instrument):
            part.insert(0, m21.instrument.Piano())

    # Extract measure boundary offsets (in quarter-notes) for MIDI marker
    # injection.  These are converted to ticks and embedded into the MIDI
    # file so that after tempo scaling we can read back precise measure
    # times in seconds directly from the MIDI's own tempo map.
    measure_offsets = extract_measure_offsets(score)
    beat_offsets = extract_beat_offsets(score)

    # One room per render slot, one slot of the family left dry. test is
    # the internal evaluation material and stays dry throughout.
    reverb_plan = (
        draw_reverb_plan(stem) if split in ("train", "valid") else None
    )
    render_plan = render_plan_for_work(
        stem, split, num_versions, available_soundfonts, aug_config
    )

    for version in range(num_versions):
        plan = render_plan[version]
        soundfonts = plan["soundfonts"]
        epr_style = plan["epr_style"]
        tempo_scaling = plan["tempo_scaling"]
        reverb = reverb_plan[version] if reverb_plan else None

        # One name shape for every split: the version index is what the style
        # assignment is keyed on, so it has to survive into the file name.
        midi_key = f"{stem}_v{version}"
        midi_path = midi_dir / f"{midi_key}.mid"
        audio_targets = [
            (soundfont, audio_dir / f"{midi_key}~{soundfont[:-4]}.wav")
            for soundfont in soundfonts
        ]
        audio_name = audio_targets[0][1].name

        try:
            qpm_primo = initial_qpm(score) * tempo_scaling
            render_fingerprint = (
                epr_render_fingerprint(
                    xml_path, epr_style, qpm_primo, interval_in_16th=1
                )
                if epr_style is not None else None
            )

            # Skip audio generation for renders that already satisfy the audio
            # contract, and keep the performance itself: the MIDI is the
            # performance, a wav is one rendering of it. Reuse hangs on the MIDI
            # alone, so re-rendering audio never resamples the performance
            # model. Timing comes from the MIDI's embedded markers, the single
            # source of truth.
            pending = [t for t in audio_targets
                       if not audio_matches_contract(t[1])]
            done = [t for t in audio_targets if audio_matches_contract(t[1])]
            reuse_existing_midi = midi_path.exists()
            published_render_fingerprint = render_fingerprint
            if reuse_existing_midi and epr_style is not None:
                # Existing performance MIDI is authoritative; its embedded
                # provenance must not be replaced by the current XML recipe.
                published_render_fingerprint = read_epr_render_fingerprint(
                    str(midi_path))
            if not reuse_existing_midi and done:
                # Audio and timing metadata are one artifact.  Rebuild all
                # timbres when their shared MIDI is absent or predates the
                # current exact EPR timing contract.
                pending = list(audio_targets)
                done = []
            if reuse_existing_midi:
                audio_measures = read_measure_times_from_midi(str(midi_path))
                audio_beats = read_beat_times_from_midi(str(midi_path))
                audio_grid = read_grid_times_from_midi(str(midi_path))
                if not audio_measures or not audio_beats or not audio_grid:
                    reuse_existing_midi = False
                    pending = list(audio_targets)
                    done = []
            if reuse_existing_midi:
                audio_measures = attach_measure_supervision_flags(
                    audio_measures, measure_offsets)
                duration_sec = audio_measures[-1]["end_sec"] if audio_measures else 0.0
                for _, existing_path in done:
                    results[existing_path.name] = "skipped (exists)"
                alignment_info[midi_key] = {
                    "tempo_scaling": tempo_scaling,
                    "tempo_log2": plan["tempo_log2"],
                    "tempo_stratum": plan["tempo_stratum"],
                    "duration_sec": round(duration_sec, 4),
                    "audio_measures": audio_measures,
                    "audio_beats": audio_beats,
                    "audio_grid": audio_grid,
                    "epr_style": epr_style,
                    "reverb": reverb,
                    "render_fingerprint": published_render_fingerprint,
                }
            if not pending:
                continue

            # Write MIDI.  Speed is decided here, before the file exists: the
            # performance model has to be told the tempo it is playing at, and
            # a score render answers the same request by carrying the scaled
            # mark into the write.
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            midi_written = False
            failure = "error: midi write failed"
            seconds_at = None
            phase_grid_offsets = extract_phase_grid_offsets(
                beat_offsets, [offset for offset, _ in build_tempo_map(score)])
            try:
                if reuse_existing_midi:
                    midi_written = True
                elif epr_style is None:
                    # Atomic publish; see inject_measure_markers.
                    midi_tmp = midi_path.with_name(f".{midi_path.name}")
                    with scaled_score_tempo(score, tempo_scaling):
                        score.write("midi", fp=str(midi_tmp))
                    os.replace(midi_tmp, midi_path)
                else:
                    render = render_epr(
                        xml_path,
                        epr_style,
                        midi_dir,
                        qpm_primo=qpm_primo,
                        interval_in_16th=1,
                    )
                    os.replace(render.midi_path, midi_path)
                    tempo_points = render.tempo_points
                    seconds_at = lambda qn: epr_seconds_at(tempo_points, qn)
                    phase_grid_offsets = extract_phase_grid_offsets(
                        beat_offsets,
                        [point.quarter_offset for point in tempo_points],
                    )
                if not reuse_existing_midi and midi_path.exists():
                    print(f"[MIDI OK] {midi_path.name}", flush=True)
                    midi_written = True
                elif not reuse_existing_midi:
                    print(f"[MIDI FAIL] {midi_path.name} - file not created", flush=True)
            except EPRTimeout as e:
                print(f"[EPR TIMEOUT] {midi_path.name} - {e}", flush=True)
                failure = "error: epr timeout"
            except EPRError as e:
                print(f"[EPR FAIL] {midi_path.name} - {e}", flush=True)
                failure = f"error: epr render failed - {type(e).__name__}"
            except Exception as e:
                print(f"[MIDI FAIL] {midi_path.name} - {e}", flush=True)
                failure = "error: midi write failed"

            if not midi_written:
                for _, target_path in audio_targets:
                    results[target_path.name] = failure
                continue

            # Inject measure-boundary markers into the MIDI file.  A score
            # render locates them by score offset; a performance keeps its
            # timing in the notes, so its markers are placed by the grid the
            # model played on.
            if measure_offsets and not reuse_existing_midi:
                inject_measure_markers(
                    str(midi_path), measure_offsets, beat_offsets,
                    seconds_at=seconds_at,
                    grid_offsets=phase_grid_offsets,
                    epr_fingerprint=render_fingerprint)

            # Read timing from the MIDI that FluidSynth will render.
            if not reuse_existing_midi:
                audio_measures = read_measure_times_from_midi(str(midi_path))
                audio_beats = read_beat_times_from_midi(str(midi_path))
                audio_grid = read_grid_times_from_midi(str(midi_path))
            if not audio_measures:
                for _, target_path in audio_targets:
                    results[target_path.name] = "error: midi has no measure markers"
                continue
            audio_measures = attach_measure_supervision_flags(
                audio_measures, measure_offsets)
            duration_sec = audio_measures[-1]["end_sec"]

            # Render MIDI -> Audio with loudness normalization.  One MIDI, one
            # timbre per output; test asks for every timbre of every version.
            for soundfont, audio_path in pending:
                sf_path = soundfont_dir / soundfont
                fs = FluidSynth(str(sf_path), sample_rate=44100)
                try:
                    render_one_midi(fs, str(midi_path), str(audio_path),
                                    reverb=reverb)

                    if audio_path.exists():
                        print(f"[AUDIO OK] {audio_path.name}", flush=True)
                        results[audio_path.name] = "success"
                        # Timing belongs to the MIDI; every timbre of it reads
                        # the same entry.
                        alignment_info[midi_key] = {
                            "tempo_scaling": tempo_scaling,
                            "tempo_log2": plan["tempo_log2"],
                            "tempo_stratum": plan["tempo_stratum"],
                            "duration_sec": round(duration_sec, 4),
                            "audio_measures": audio_measures,
                            "audio_beats": audio_beats,
                            "audio_grid": audio_grid,
                            "epr_style": epr_style,
                            "reverb": reverb,
                            "render_fingerprint": render_fingerprint,
                        }
                    else:
                        print(f"[AUDIO FAIL] {audio_path.name} - file not created", flush=True)
                        results[audio_path.name] = "error: audio not created"
                except Exception as e:
                    print(f"[AUDIO FAIL] {audio_path.name} - {e}", flush=True)
                    results[audio_path.name] = f"error: audio render failed - {e}"

        except Exception as e:
            results[audio_name] = f"error: {e}"

    return results, alignment_info


def phase2_kern_to_audio(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    soundfont_dir: Path = DEFAULT_SOUNDFONT_DIR,
    aug_config_path: Path = DEFAULT_AUG_CONFIG,
    workers: int = 1,
) -> Dict[str, str]:
    """Phase 2: Render canonical MusicXML to audio with augmentation.

    The kern directory supplies the corpus item list and split identity; all
    musical timing, meter, and tempo used for rendering and annotations come
    from the matching Phase-1 MusicXML.

    Args:
        output_dir: Path to output directory (contains kern/ and xml/)
        metadata_dir: Path to metadata directory
        soundfont_dir: Path to soundfont directory
        aug_config_path: Path to augmentation config JSON
        workers: Number of parallel workers (1 = sequential)

    Returns:
        Dictionary of {filename: status}
    """
    # Load augmentation config
    aug_config = load_augmentation_config(aug_config_path)
    tempo_enabled = aug_config['tempo_enabled']
    tempo_range = aug_config['tempo_range']
    train_soundfonts = aug_config['train_soundfonts']
    valid_soundfonts = aug_config['valid_soundfonts']
    test_soundfonts = aug_config['test_soundfonts']
    num_versions_config = aug_config['num_versions']

    if tempo_enabled:
        logger.info(f"Tempo augmentation ENABLED (range: {tempo_range[0]:.2f}-{tempo_range[1]:.2f})")
    else:
        logger.info("Tempo augmentation DISABLED")

    kern_dir = output_dir / "kern"
    midi_dir = output_dir / "midi"
    audio_dir = output_dir / "audio"

    midi_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load split files
    test_split = set()
    valid_split = set()

    test_split_path = metadata_dir / "test_split.txt"
    valid_split_path = metadata_dir / "valid_split.txt"

    # "!" marks comments: entries like "beethoven#sonata03-1" rule out "#".
    if test_split_path.exists():
        df = pd.read_csv(test_split_path, comment="!")
        test_split = set(df["name"].tolist())
    if valid_split_path.exists():
        df = pd.read_csv(valid_split_path, comment="!")
        valid_split = set(df["name"].tolist())

    excluded_stems = load_protocol_exclude(metadata_dir)
    if excluded_stems:
        logger.info(f"Protocol exclusion: {len(excluded_stems)} stems will not be rendered")

    # Check available soundfonts (train, valid, test separately)
    available_train_soundfonts = [sf for sf in train_soundfonts if (soundfont_dir / sf).exists()]
    available_valid_soundfonts = [sf for sf in valid_soundfonts if (soundfont_dir / sf).exists()]
    available_test_soundfonts = [sf for sf in test_soundfonts if (soundfont_dir / sf).exists()]

    if not available_train_soundfonts:
        logger.error(f"No train soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {train_soundfonts}")
        return {}

    if not available_valid_soundfonts:
        logger.error(f"No valid soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {valid_soundfonts}")
        return {}

    if not available_test_soundfonts:
        logger.error(f"No test soundfonts found in {soundfont_dir}")
        logger.info(f"Expected soundfonts: {test_soundfonts}")
        return {}

    logger.info(f"Available train soundfonts: {available_train_soundfonts}")
    logger.info(f"Available valid soundfonts: {available_valid_soundfonts}")
    logger.info(f"Available test soundfonts: {available_test_soundfonts}")

    # Prepare task list
    kern_files = sorted(kern_dir.glob("*.krn"))
    tasks = []

    for kern_path in kern_files:
        stem = kern_path.stem

        if stem in excluded_stems:
            continue

        # Determine split
        split = "train"
        for test_name in test_split:
            if _match_split_name(stem, test_name):
                split = "test"
                break
        if split == "train":
            for valid_name in valid_split:
                if _match_split_name(stem, valid_name):
                    split = "valid"
                    break

        n_versions = num_versions_config[split]
        # Each split uses its own soundfont list
        if split == "train":
            soundfonts_for_split = available_train_soundfonts
        elif split == "valid":
            soundfonts_for_split = available_valid_soundfonts
        else:  # test
            soundfonts_for_split = available_test_soundfonts

        tasks.append((
            kern_path,
            output_dir,
            soundfont_dir,
            split,
            n_versions,
            soundfonts_for_split,
            aug_config,
        ))

    logger.info(f"Processing {len(tasks)} kern files with {workers} workers...")

    all_results = {}

    # Generate augmentation metadata first (creates fresh metadata with kern_measures)
    # This replaces the old incremental load, ensuring metadata is always in sync with kern files
    logger.info("Generating augmentation metadata...")
    metadata = generate_metadata(
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        aug_config_path=aug_config_path,
    )
    metadata_path = output_dir / "augmentation_metadata.json"

    # Save initial metadata.  Atomic publish: this file is rewritten every few
    # files below, and a kill mid-dump must not leave a truncated JSON that
    # phase 2.5/3 then read.
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}")
    with open(metadata_tmp, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    os.replace(metadata_tmp, metadata_path)
    expected_renders = sum(
        len(entry.get("renders", [])) for entry in metadata.values())
    logger.info(
        "Generated metadata for %d MIDI versions and %d audio renders",
        len(metadata), expected_renders,
    )

    updated_count = 0
    save_interval = 10  # Save metadata every N kern files

    alignment_mismatches = []

    def update_metadata_incremental(alignment: dict) -> int:
        """Update metadata in memory with alignment info. Returns count of updates."""
        count = 0
        for audio_key, align_info in alignment.items():
            if audio_key in metadata:
                # Validate: Score measure count must match kern_gt measure count.
                # Score is the single source of truth for measure numbering/timing.
                kern_measures = metadata[audio_key].get("kern_measures", [])
                audio_measures = align_info["audio_measures"]
                if kern_measures and audio_measures and len(kern_measures) != len(audio_measures):
                    alignment_mismatches.append(
                        f"{audio_key}: kern_measures={len(kern_measures)}, "
                        f"audio_measures(Score)={len(audio_measures)}"
                    )

                planned_scale = float(metadata[audio_key]["tempo_scaling"])
                actual_scale = float(align_info["tempo_scaling"])
                if not math.isclose(
                    planned_scale, actual_scale, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"{audio_key}: planned tempo scale={planned_scale}, "
                        f"but Phase 2 used {actual_scale}"
                    )
                if metadata[audio_key].get("tempo_stratum") != align_info.get(
                    "tempo_stratum"
                ):
                    raise RuntimeError(
                        f"{audio_key}: planned tempo stratum="
                        f"{metadata[audio_key].get('tempo_stratum')}, but "
                        f"Phase 2 used {align_info.get('tempo_stratum')}"
                    )
                metadata[audio_key]["tempo_scaling"] = actual_scale
                metadata[audio_key]["tempo_log2"] = align_info["tempo_log2"]
                metadata[audio_key]["duration_sec"] = align_info["duration_sec"]
                metadata[audio_key]["audio_measures"] = align_info["audio_measures"]
                metadata[audio_key]["audio_beats"] = align_info.get("audio_beats", [])
                if align_info.get("audio_grid"):
                    metadata[audio_key]["audio_grid"] = align_info["audio_grid"]
                actual_style = align_info.get("epr_style")
                if metadata[audio_key].get("epr_style") != actual_style:
                    raise RuntimeError(
                        f"{audio_key}: planned epr_style="
                        f"{metadata[audio_key].get('epr_style')!r}, but Phase 2 "
                        f"used {actual_style!r}"
                    )
                metadata[audio_key]["reverb"] = align_info.get("reverb")
                if align_info.get("render_fingerprint"):
                    metadata[audio_key]["render_fingerprint"] = (
                        align_info["render_fingerprint"])
                count += 1
        return count

    def save_metadata():
        """Save metadata to disk (atomic publish; see the initial save)."""
        if metadata:
            tmp = metadata_path.with_name(f".{metadata_path.name}")
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            os.replace(tmp, metadata_path)

    if workers == 1:
        # Sequential processing with incremental metadata updates
        for i, task in enumerate(tqdm(tasks, desc="Processing kern files")):
            results, alignment = _process_single_kern(task)
            all_results.update(results)
            updated_count += update_metadata_incremental(alignment)

            # Save periodically
            if (i + 1) % save_interval == 0:
                save_metadata()
    else:
        # Parallel processing with incremental metadata updates (in main thread)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_kern, task): task[0] for task in tasks}

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing kern files")):
                kern_path = futures[future]
                try:
                    results, alignment = future.result()
                    all_results.update(results)
                    updated_count += update_metadata_incremental(alignment)

                    # Save periodically
                    if (i + 1) % save_interval == 0:
                        save_metadata()
                except Exception as e:
                    import traceback
                    logger.error(f"Error processing {kern_path}: {e}\n{traceback.format_exc()}")
                    all_results[kern_path.stem] = f"error: {e}"

    # Final save
    save_metadata()
    logger.info(f"Updated {updated_count} entries in {metadata_path}")

    # Report alignment validation
    if alignment_mismatches:
        logger.error(
            f"ALIGNMENT MISMATCH: {len(alignment_mismatches)} entries have "
            f"kern_measures vs audio_measures(Score) count disagreement:"
        )
        for msg in alignment_mismatches[:20]:
            logger.error(f"  {msg}")
        if len(alignment_mismatches) > 20:
            logger.error(f"  ... and {len(alignment_mismatches) - 20} more")
    else:
        logger.info("Alignment validation PASSED: all kern_measures/audio_measures counts match")

    # Summary
    generated = sum(1 for v in all_results.values() if v == "success")
    reused = sum(1 for v in all_results.values() if v.startswith("skipped"))
    errors = sum(1 for v in all_results.values() if v.startswith("error"))
    logger.info(
        "Phase 2 render work: %d generated this run, %d reused existing, "
        "%d errors",
        generated, reused, errors,
    )

    expected_midi = {f"{key}.mid" for key in metadata}
    expected_audio = {
        f"{render['audio_key']}.wav"
        for entry in metadata.values()
        for render in entry.get("renders", [])
    }
    present_midi = {path.name for path in midi_dir.glob("*.mid")}
    present_audio = {path.name for path in audio_dir.glob("*.wav")}
    missing_midi = expected_midi - present_midi
    missing_audio = expected_audio - present_audio
    unexpected_midi = present_midi - expected_midi
    unexpected_audio = present_audio - expected_audio
    logger.info(
        "Phase 2 artifact inventory: MIDI %d/%d, audio %d/%d",
        len(expected_midi - missing_midi), len(expected_midi),
        len(expected_audio - missing_audio), len(expected_audio),
    )
    if missing_midi or missing_audio:
        logger.error(
            "Phase 2 artifact inventory MISMATCH: missing MIDI=%d, audio=%d",
            len(missing_midi), len(missing_audio),
        )
    else:
        logger.info("Phase 2 artifact inventory PASSED: all expected outputs exist")
    if unexpected_midi or unexpected_audio:
        logger.warning(
            "Phase 2 output directory also contains stale/unexpected files: "
            "MIDI=%d, audio=%d",
            len(unexpected_midi), len(unexpected_audio),
        )

    logger.info(
        "Phase 2 complete: %d/%d audio renders ready "
        "(%d generated this run, %d reused existing, %d errors)",
        len(expected_audio - missing_audio), len(expected_audio),
        generated, reused, errors,
    )

    if alignment_mismatches:
        raise RuntimeError(
            f"Phase 2 produced {len(alignment_mismatches)} measure-alignment "
            "mismatch(es); manifests cannot safely pair audio and score bars"
        )

    return all_results


# ============================================================
# Phase 2.5: Audio → Mel Spectrogram
# ============================================================

def phase2_5_audio_to_mel(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    workers: int = 4,
) -> Dict[str, int]:
    """Phase 2.5: Convert audio files to mel spectrograms.

    Processes all audio files listed in augmentation_metadata.json and
    saves mel spectrograms as .npy files. Mel format = HFT_MEL (frozen).
    """
    metadata_path = output_dir / "augmentation_metadata.json"
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        logger.info("Run Phase 2 first to generate augmentation_metadata.json")
        return {"generated": 0, "skipped": 0, "missing": 0, "failed": 0}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    mel_dir = output_dir / "mel"
    mel_dir.mkdir(exist_ok=True)
    audio_dir = output_dir / "audio"

    # metadata is keyed by MIDI; the mel files follow its renders.
    tasks = [
        (render["audio_key"],
         audio_dir / f"{render['audio_key']}.wav",
         mel_dir / f"{render['audio_key']}.npy")
        for entry in metadata.values()
        for render in entry.get("renders", [])
    ]

    logger.info(f"Phase 2.5: Converting {len(tasks)} audio files to mel spectrograms")
    logger.info(f"Mel spec: {HFT_MEL}")

    stats = {"generated": 0, "skipped": 0, "missing": 0, "failed": 0}
    results = process_audio_batch(
        tasks, spec=HFT_MEL, workers=workers, description="Generating mel")
    for status, _ in results.values():
        stats[status] += 1

    # Summary
    logger.info(f"\nPhase 2.5 complete:")
    logger.info(f"  Generated: {stats['generated']}")
    logger.info(f"  Skipped:   {stats['skipped']}")
    logger.info(f"  Missing:   {stats['missing']}")
    logger.info(f"  Failed:    {stats['failed']}")
    logger.info(f"  Total mel: {len(list(mel_dir.glob('*.npy')))}")

    return stats


def _match_split_name(processed_name: str, split_name: str) -> bool:
    """Check if a processed filename matches a split file entry.

    Handles mappings like:
    - "beethoven_piano_sonatas_sonata01-1" ↔ "beethoven#sonata01-1"
    - "humdrum_chopin_first_editions_001-1a-HO" ↔ "chopin#001-1a-HO"
    - "joplin_entertainer" ↔ "joplin#entertainer"
    - "musesyn_SomeSong" ↔ "SomeSong"
    """
    # Direct match
    if processed_name == split_name:
        return True

    # Handle "repo#piece" format
    if "#" in split_name:
        prefix, piece = split_name.split("#", 1)

        # Map prefix to our naming convention
        prefix_map = {
            "beethoven": "beethoven_piano_sonatas",
            "haydn": "haydn_piano_sonatas",
            "mozart": "mozart_piano_sonatas",
            "chopin": "humdrum_chopin_first_editions",
            "joplin": "joplin",
            "scarlatti": "scarlatti_keyboard_sonatas",
        }

        if prefix in prefix_map:
            expected_prefix = prefix_map[prefix]
            expected_name = f"{expected_prefix}_{piece}"
            if processed_name == expected_name:
                return True

    # Handle MuseSyn (no prefix in split file)
    if processed_name.startswith("musesyn_"):
        musesyn_name = processed_name[8:]  # Remove "musesyn_" prefix
        if musesyn_name == split_name:
            return True

    return False


def main():
    """Main entry point for preprocessing pipeline."""
    import argparse
    import signal

    def terminate_process_group(signum, frame):
        # ProcessPool workers and renderer subprocesses inherit this group;
        # killing it avoids waiting on nested work after an interrupt.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("\n[INFO] Caught signal, terminating render processes...", flush=True)
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

    signal.signal(signal.SIGINT, terminate_process_group)
    signal.signal(signal.SIGTERM, terminate_process_group)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Syn dataset preprocessing pipeline")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "1-full", "1.5", "2", "2.5", "3", "all"],
        default="all",
        help="Phase to run: 1 (Score→Kern only), 1-full (Score→Kern + 1.5), "
             "1.5 (kern_gt only), 2 (canonical XML→Audio), 2.5 (Audio→Mel), "
             "3 (Create manifests), all (all phases; default).",
    )
    parser.add_argument(
        "--humsyn-dir",
        type=Path,
        default=DEFAULT_HUMSYN_DIR,
        help="HumSyn input directory",
    )
    parser.add_argument(
        "--musesyn-dir",
        type=Path,
        default=DEFAULT_MUSESYN_DIR,
        help="MuseSyn input directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Metadata directory",
    )
    parser.add_argument(
        "--soundfont-dir",
        type=Path,
        default=DEFAULT_SOUNDFONT_DIR,
        help="Soundfont directory",
    )
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=DEFAULT_AUG_CONFIG,
        help="Augmentation config JSON file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for Phase 1/1.5/2/2.5 (default: 1 = sequential)",
    )
    parser.add_argument(
        "--keep-grace",
        action="store_true",
        help="Phase 1: retain grace notes in kern/ (default: stripped)",
    )
    parser.add_argument(
        "--keep-trill",
        action="store_true",
        help="Phase 1: retain trill signifiers in kern/ (default: stripped)",
    )
    parser.add_argument(
        "--keep-non-trill-ornaments",
        action="store_true",
        help="Phase 1: retain mordent/turn/shake/schleifer signifiers in kern/ (default: stripped)",
    )
    parser.add_argument(
        "--keep-arpeggio",
        action="store_true",
        help="Phase 1: retain arpeggio marks in kern/ (default: stripped)",
    )
    parser.add_argument(
        "--gt-keep-grace",
        action="store_true",
        help="Phase 1.5: retain grace notes in kern_gt/ (default: stripped)",
    )
    parser.add_argument(
        "--gt-keep-trill",
        action="store_true",
        help="Phase 1.5: retain trill signifiers in kern_gt/ (default: stripped)",
    )
    parser.add_argument(
        "--gt-keep-non-trill-ornaments",
        action="store_true",
        help="Phase 1.5: retain mordent/turn/shake/schleifer signifiers in kern_gt/ (default: stripped)",
    )
    parser.add_argument(
        "--gt-keep-arpeggio",
        action="store_true",
        help="Phase 1.5: retain arpeggio marks in kern_gt/ (default: stripped)",
    )
    args = parser.parse_args()

    # Determine which phases to run
    phase = args.phase
    run_phase1 = phase in {"1", "1-full", "all"}
    run_phase1_5 = phase in {"1-full", "1.5", "all"}
    run_phase2 = phase in {"2", "all"}
    run_phase2_5 = phase == "2.5" or phase == "all"
    run_phase3 = phase == "3" or phase == "all"

    # Run requested phases
    if run_phase1:
        logger.info("=== Phase 1: Score → Kern ===")
        phase1_results = phase1_score_to_kern(
            humsyn_dir=args.humsyn_dir,
            musesyn_dir=args.musesyn_dir,
            output_dir=args.output_dir,
            metadata_dir=args.metadata_dir,
            keep_grace=args.keep_grace,
            keep_trill=args.keep_trill,
            keep_non_trill_ornaments=args.keep_non_trill_ornaments,
            keep_arpeggio=args.keep_arpeggio,
            workers=args.workers,
        )
        success = sum(1 for v in phase1_results.values() if v == "success")
        skipped = sum(1 for v in phase1_results.values() if v == "skipped")
        errors = sum(1 for v in phase1_results.values() if v.startswith("error"))
        logger.info(f"Phase 1 complete: {success} success, {skipped} skipped, {errors} errors (total {len(phase1_results)})")

    if run_phase1_5:
        # Create ground truth kern files
        logger.info("=== Phase 1.5: Creating Ground Truth Kern ===")
        gt_results = create_ground_truth_kern(
            output_dir=args.output_dir,
            keep_grace=args.gt_keep_grace,
            keep_trill=args.gt_keep_trill,
            keep_non_trill_ornaments=args.gt_keep_non_trill_ornaments,
            keep_arpeggio=args.gt_keep_arpeggio,
            workers=args.workers,
        )
        gt_success = sum(1 for v in gt_results.values() if v == "success")
        logger.info(f"Phase 1.5 complete: {gt_success}/{len(gt_results)} successful")

    if run_phase2:
        logger.info("=== Phase 2: Canonical MusicXML → Audio ===")
        phase2_kern_to_audio(
            output_dir=args.output_dir,
            metadata_dir=args.metadata_dir,
            soundfont_dir=args.soundfont_dir,
            aug_config_path=args.aug_config,
            workers=args.workers,
        )

    if run_phase2_5:
        logger.info("=== Phase 2.5: Audio → Mel ===")
        phase2_5_stats = phase2_5_audio_to_mel(
            output_dir=args.output_dir,
            workers=args.workers,
        )
        logger.info(f"Phase 2.5 complete: {phase2_5_stats['generated']} generated, "
                    f"{phase2_5_stats['skipped']} skipped")

    if run_phase3:
        logger.info("=== Phase 3: Create Manifests ===")
        phase3_counts = create_manifest(
            data_dir=args.output_dir,
            metadata_dir=args.metadata_dir,
        )
        total = sum(phase3_counts.values())
        logger.info(f"Phase 3 complete: {total} total samples in manifests")


if __name__ == "__main__":
    main()
