"""
Synthetic piano dataset (HumSyn / MuseSyn) evaluation module.

Mirrors src/evaluation/asap/ for the synthetic test set.

Loads manifest.json and augmentation_metadata.json to enumerate
5-bar evaluation windows aligned to audio measure boundaries, and
provides GT kern-slice → MIDI conversion for MV2H evaluation.

Usage:
    from evaluation.syn import SynDataset, SynChunkInfo

    syn = SynDataset(
        manifest_path='data/experiments/syn/test_manifest.json',
        metadata_path='data/experiments/syn/augmentation_metadata.json',
        kern_gt_dir='data/experiments/syn/kern_gt',
        manifest_dir='data/experiments/syn',
    )

    for chunk in syn.iter_5bar_windows():
        gt_midi = syn.get_gt_kern_slice_midi(
            chunk.kern_gt_path, chunk.kern_line_start, chunk.kern_line_end
        )
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import music21
from src.datasets.syn.metadata import index_augmentation_metadata_by_render
from src.score.generate_score import kern_to_musicxml
from src.score.kern_postprocess import (
    kern_pre_midi_clean,
    slice_kern_for_midi,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SynChunkInfo:
    """A single evaluation chunk (n-bar window) from the synthetic dataset.

    Attributes:
        chunk_id:         Unique ID of the form '{perf_id}.{chunk_index}'.
        perf_id:          Performance identifier (key in manifest / metadata).
        chunk_index:      Zero-based index of this window within the performance.
        start_measure:    First measure number in the window.
        end_measure:      Last measure number in the window (inclusive).
        start_sec:        Audio start time (seconds).
        end_sec:          Audio end time (seconds).
        mel_path:         Absolute path to the mel spectrogram (.pt file).
        kern_gt_path:     Absolute path to the GT kern file.
        kern_line_start:  First line index (0-based) of the window in the kern file.
        kern_line_end:    Last line index (0-based, inclusive) of the window.
    """

    chunk_id: str
    perf_id: str
    chunk_index: int
    start_measure: int
    end_measure: int
    start_sec: float
    end_sec: float
    mel_path: str
    kern_gt_path: str
    kern_line_start: int
    kern_line_end: int

    def __repr__(self) -> str:
        return (
            f"SynChunkInfo({self.chunk_id}, "
            f"m{self.start_measure}-{self.end_measure}, "
            f"{self.start_sec:.1f}s-{self.end_sec:.1f}s)"
        )


# =============================================================================
# DATASET CLASS
# =============================================================================


class SynDataset:
    """Handler for synthetic piano dataset evaluation.

    Reads manifest.json (list of {id, mel_path}) and
    augmentation_metadata.json (per-performance timing info) to enumerate
    sliding-window evaluation chunks.

    Args:
        manifest_path:  Path to test_manifest.json.
        metadata_path:  Path to augmentation_metadata.json.
        kern_gt_dir:    Directory containing GT kern files.
        manifest_dir:   Base directory used to resolve mel_path values in
                        the manifest (mel_path is relative to this dir).
    """

    def __init__(
        self,
        manifest_path: str,
        metadata_path: str,
        kern_gt_dir: str,
        manifest_dir: str,
    ) -> None:
        self.kern_gt_dir = Path(kern_gt_dir)
        self.manifest_dir = Path(manifest_dir)

        with open(manifest_path) as f:
            manifest_list = json.load(f)
        # Keyed by performance id for fast lookup.
        self.manifest: Dict[str, dict] = {item["id"]: item for item in manifest_list}

        with open(metadata_path) as f:
            self.metadata = index_augmentation_metadata_by_render(json.load(f))

        logger.info(
            f"SynDataset: {len(self.manifest)} performances, "
            f"{len(self.metadata)} metadata entries"
        )

    # ------------------------------------------------------------------
    # Window iteration
    # ------------------------------------------------------------------

    def iter_5bar_windows(
        self,
        n_bars: int = 5,
        stride: int = 1,
        first_bar: Optional[Dict[str, int]] = None,
    ) -> Iterator[SynChunkInfo]:
        """Yield SynChunkInfo for every n-bar sliding window in the dataset.

        Windows align exactly to audio measure boundaries provided in
        augmentation_metadata['audio_measures'].

        Args:
            n_bars:  Number of bars per chunk (default 5, Zeng-style).
            stride:  Sliding window stride in measures (default 1).
            first_bar: Per-performance index of the bar the windows start on
                (default 0 for every performance). A piece whose first notated
                bar is an anacrusis starts at 1, so no window is part pickup.
        """
        for perf_id, item in self.manifest.items():
            if perf_id not in self.metadata:
                logger.warning(f"No metadata for {perf_id}, skipping")
                continue

            meta = self.metadata[perf_id]
            audio_measures: List[dict] = meta.get("audio_measures", [])
            kern_measures: List[dict] = meta.get("kern_measures", [])
            kern_file: str = meta.get("kern_file", "")

            kern_gt_path = self.kern_gt_dir / kern_file
            if not kern_gt_path.exists():
                logger.warning(f"GT kern missing: {kern_gt_path}, skipping")
                continue

            mel_path = str(self.manifest_dir / item["mel_path"])
            n_measures = len(audio_measures)
            if n_measures < n_bars:
                continue

            # Pair audio and kern measures positionally. Per
            # extract_kern_measures' contract its "measure" numbers are local
            # indices, not authoritative labels: audio labels an anacrusis
            # bar 0 while kern numbers its rest-padded counterpart 1, so a
            # label join shifts every window of a pickup piece by one bar.
            if len(kern_measures) != n_measures:
                raise ValueError(
                    f"{perf_id}: audio_measures ({n_measures}) vs "
                    f"kern_measures ({len(kern_measures)}) length mismatch — "
                    f"metadata inconsistent, refusing to window this piece"
                )

            chunk_idx = 0
            start_bar = (first_bar or {}).get(perf_id, 0)
            for i in range(start_bar, n_measures - n_bars + 1, stride):
                window = audio_measures[i : i + n_bars]
                km_window = kern_measures[i : i + n_bars]

                yield SynChunkInfo(
                    chunk_id=f"{perf_id}.{chunk_idx}",
                    perf_id=perf_id,
                    chunk_index=chunk_idx,
                    start_measure=window[0]["measure"],
                    end_measure=window[-1]["measure"],
                    start_sec=window[0]["start_sec"],
                    end_sec=window[-1]["end_sec"],
                    mel_path=mel_path,
                    kern_gt_path=str(kern_gt_path),
                    kern_line_start=km_window[0]["line_start"],
                    kern_line_end=km_window[-1]["line_end"],
                )
                chunk_idx += 1

    # ------------------------------------------------------------------
    # Ground-truth extraction
    # ------------------------------------------------------------------

    def get_gt_kern_slice_midi(
        self,
        kern_gt_path: str,
        kern_line_start: int,
        kern_line_end: int,
    ) -> Optional[str]:
        """Extract a line-range from a GT kern file and write a temp MIDI.

        Makes the slice self-contained by carrying the active meter and key
        into its first bar before handing it to music21.

        Args:
            kern_gt_path:    Path to the GT kern file.
            kern_line_start: First line index to include (0-based, inclusive).
            kern_line_end:   Last line index to include (0-based, inclusive).

        Returns:
            Path to a temporary MIDI file, or None on failure.
        """
        tmp_krn = tmp_xml = tmp_mid = None
        try:
            with open(kern_gt_path) as f:
                lines = f.readlines()

            # kern_measures stores 1-indexed line numbers (sanitize_kern uses
            # enumerate(..., start=1)); convert to 0-indexed Python slice.
            # Walk back from line_start to include the leading `=N` barline
            # so music21 anchors the measure correctly.
            slice_start = kern_line_start - 1
            scan = slice_start - 1
            while scan >= 0:
                stripped = lines[scan].lstrip()
                if stripped.startswith('='):
                    slice_start = scan
                    break
                if stripped and not stripped.startswith(('*', '!')):
                    break
                scan -= 1
            kern_slice = slice_kern_for_midi(
                "".join(lines), slice_start, kern_line_end,
            )

            # Strip grace notes + tiefix-style cross-slice tie repair before
            # handing to music21 (see src/score/kern_postprocess.py).
            kern_slice = kern_pre_midi_clean(kern_slice)

            with tempfile.NamedTemporaryFile(suffix=".krn", delete=False, mode="w") as f:
                tmp_krn = f.name
                f.write(kern_slice)
            with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
                tmp_xml = f.name
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                tmp_mid = f.name

            kern_to_musicxml(tmp_krn, tmp_xml)
            score = music21.converter.parse(tmp_xml)
            score.write("midi", fp=tmp_mid)
            return tmp_mid

        except Exception as e:
            logger.debug(f"GT kern slice → MIDI failed ({kern_gt_path}): {e}")
            return None
        finally:
            for p in [tmp_krn, tmp_xml]:
                if p and os.path.exists(p):
                    os.remove(p)

    def get_pred_kern_slice_midi(
        self,
        pred_kern_path: str,
        measure_offset: int,
        n_bars: int = 5,
    ) -> Optional[str]:
        """Extract n_bars measures from a full-song pred kern and write a temp MIDI.

        Uses measure position (0-based offset) rather than printed measure numbers,
        so it is robust to different numbering conventions between pred and GT.

        Args:
            pred_kern_path:  Path to the full-song pred kern file.
            measure_offset:  0-based index of the first measure to extract.
            n_bars:          Number of consecutive measures to extract.

        Returns:
            Path to a temporary MIDI file, or None on failure.
        """
        tmp_krn = tmp_xml = tmp_mid = None
        try:
            with open(pred_kern_path) as f:
                lines = f.readlines()

            # Locate barline positions (lines starting with "=").
            bar_positions = [i for i, l in enumerate(lines) if l.startswith("=")]

            if measure_offset + n_bars > len(bar_positions):
                return None  # Not enough measures in pred kern.

            start_line = bar_positions[measure_offset]
            # Include lines up to (but not including) the opening barline of the
            # next window.  If this is the last window, take until end-of-file.
            if measure_offset + n_bars < len(bar_positions):
                end_line = bar_positions[measure_offset + n_bars]
            else:
                end_line = len(lines)

            kern_slice = slice_kern_for_midi(
                "".join(lines), start_line, end_line,
            )

            with tempfile.NamedTemporaryFile(suffix=".krn", delete=False, mode="w") as f:
                tmp_krn = f.name
                f.write(kern_slice)
            with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
                tmp_xml = f.name
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                tmp_mid = f.name

            kern_to_musicxml(tmp_krn, tmp_xml)
            score = music21.converter.parse(tmp_xml)
            score.write("midi", fp=tmp_mid)
            return tmp_mid

        except Exception as e:
            logger.debug(f"Pred kern slice → MIDI failed ({pred_kern_path}): {e}")
            return None
        finally:
            for p in [tmp_krn, tmp_xml]:
                if p and os.path.exists(p):
                    os.remove(p)
