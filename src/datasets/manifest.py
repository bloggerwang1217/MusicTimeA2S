"""Manifest compilation shared by dataset preparation pipelines."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.a2s.piano.foundation import HFT_MEL

logger = logging.getLogger(__name__)


def create_manifests_from_metadata(
    data_dir: Union[str, Path],
    metadata_filename: str = "augmentation_metadata.json",
    validate_files: bool = True,
    sample_validation_count: int = 5,
    excluded_stems: Optional[set[str]] = None,
    excluded_render_ids: Optional[set[str]] = None,
    splits: Sequence[str] = ("train", "valid", "test"),
) -> Dict[str, int]:
    """Compile split manifests from the shared render metadata schema."""
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    excluded_stems = excluded_stems or set()
    excluded_render_ids = excluded_render_ids or set()
    mel_dir = data_dir / "mel"
    kern_gt_dir = data_dir / "kern_gt"
    manifests: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in splits
    }
    missing_mel: List[str] = []
    missing_kern: set[str] = set()
    metadata_updated = 0
    excluded_entries = 0

    logger.info("Creating manifests from %d entries", len(metadata))
    for entry in metadata.values():
        if Path(entry["kern_file"]).stem in excluded_stems:
            excluded_entries += 1
            continue
        split = entry["split"]
        if split not in manifests:
            raise ValueError(f"Unknown manifest split: {split!r}")
        kern_gt_path = kern_gt_dir / entry["kern_file"]

        # Timing belongs to the MIDI-level entry; its timbre renders share it.
        for render in entry.get("renders", []):
            key = render["audio_key"]
            if key in excluded_render_ids:
                continue
            mel_path = mel_dir / f"{key}.npy"
            if validate_files:
                if not mel_path.exists():
                    missing_mel.append(key)
                    continue
                if not kern_gt_path.exists():
                    missing_kern.add(entry["kern_file"])
                    continue

            mel = np.load(str(mel_path), mmap_mode="r")
            n_frames = mel.shape[-1]
            duration_sec = entry.get("duration_sec")
            if duration_sec is None:
                duration_sec = n_frames / (
                    HFT_MEL.sample_rate / HFT_MEL.hop_length)
                logger.warning(
                    "%s: duration_sec missing, using mel-based fallback", key)

            if render.get("n_frames") != n_frames:
                render["n_frames"] = n_frames
                metadata_updated += 1

            manifest_entry = {
                "id": key,
                "mel_path": f"mel/{key}.npy",
                "kern_gt_path": f"kern_gt/{entry['kern_file']}",
                "duration_sec": round(duration_sec, 4),
                "n_frames": n_frames,
                "split": split,
            }
            for timing_key in ("audio_beats", "audio_measures", "audio_grid"):
                if entry.get(timing_key):
                    manifest_entry[timing_key] = entry[timing_key]
            manifests[split].append(manifest_entry)

    if metadata_updated:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Updated %d entries in %s", metadata_updated, metadata_path)

    if missing_mel:
        logger.error("Missing mel files: %d", len(missing_mel))
        for key in missing_mel:
            logger.error("  [MISSING_MEL] %s", key)
    if missing_kern:
        logger.error("Missing kern_gt files: %d", len(missing_kern))
        for kern in sorted(missing_kern):
            logger.error("  [MISSING_KERN] %s", kern)

    counts = {}
    for split, samples in manifests.items():
        manifest_path = data_dir / f"{split}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        counts[split] = len(samples)
        logger.info("  %s: %d samples -> %s", split, len(samples), manifest_path)

    if sample_validation_count > 0 and mel_dir.exists():
        _validate_n_frames(manifests, mel_dir, sample_validation_count)

    expected_counts = {
        split: sum(
            sum(render["audio_key"] not in excluded_render_ids
                for render in entry.get("renders", []))
            for entry in metadata.values()
            if entry["split"] == split
            and Path(entry["kern_file"]).stem not in excluded_stems
        )
        for split in manifests
    }
    if excluded_stems:
        logger.info(
            "Protocol exclusion: %d/%d metadata entries withheld (%d stems listed)",
            excluded_entries, len(metadata), len(excluded_stems),
        )
    mismatched = {
        split: (counts[split], expected_counts[split])
        for split in manifests
        if counts[split] != expected_counts[split]
    }
    if mismatched:
        logger.error("Manifest count MISMATCH: %s", mismatched)
    else:
        logger.info("Manifest count PASSED: generated entries match metadata")
    return counts


def _validate_n_frames(
    manifests: Dict[str, List[Dict[str, Any]]],
    mel_dir: Path,
    sample_count: int,
) -> None:
    """Check that sampled manifest lengths still match their mel headers."""
    all_entries = [entry for rows in manifests.values() for entry in rows]
    if not all_entries:
        logger.warning("No entries to validate")
        return

    mismatches = 0
    for entry in random.sample(
            all_entries, min(sample_count, len(all_entries))):
        mel_path = mel_dir / f"{entry['id']}.npy"
        if not mel_path.exists():
            continue
        actual = np.load(str(mel_path), mmap_mode="r").shape[-1]
        if abs(actual - entry["n_frames"]) > 1:
            mismatches += 1
    if mismatches:
        logger.warning("Found %d n_frames mismatches", mismatches)
