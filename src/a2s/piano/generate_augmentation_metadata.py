"""
Generate Augmentation Metadata
==============================

Generates MIDI-keyed metadata for Phase 2 audio synthesis, including:
1. Performance style and its audio renders
2. Augmentation settings
3. Kern measure boundaries for chunking

The alignment timing is added by Phase 2 after the MIDI exists.

Output: data/experiments/syn/augmentation_metadata.json
{
    "beethoven_piano_sonatas_sonata01-1_v0": {
        "kern_file": "beethoven_piano_sonatas_sonata01-1.krn",
        "epr_style": "Beethoven",
        "split": "train",
        "version": 0,
        "renders": [
            {"soundfont": "TimGM6mb.sf2",
             "audio_key": "beethoven_piano_sonatas_sonata01-1_v0~TimGM6mb"}
        ],
        "kern_measures": [
            {"measure": 1, "line_start": 25, "line_end": 29},
            {"measure": 2, "line_start": 30, "line_end": 38}
        ],
        "tempo_scaling": 0.923,
        "duration_sec": 195.3,
        "audio_measures": [],
        "audio_beats": []
    },
}

Usage:
    poetry run python -m src.a2s.piano.generate_augmentation_metadata
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.datasets.syn.syn_manifest import load_protocol_exclude

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/experiments/syn")
DEFAULT_METADATA_DIR = Path("src/datasets/syn")
DEFAULT_AUG_CONFIG = Path("src/audio/augmentation.json")


def load_augmentation_config(config_path: Path = DEFAULT_AUG_CONFIG) -> Dict[str, Any]:
    """Load augmentation settings from JSON config.

    One implementation, in prepare_syn; imported lazily because that module
    imports this one.
    """
    from src.datasets.syn.prepare_syn import load_augmentation_config as _load

    return _load(config_path)


def _match_split_name(processed_name: str, split_name: str) -> bool:
    """Check if a processed filename matches a split file entry."""
    if processed_name == split_name:
        return True

    if "#" in split_name:
        prefix, piece = split_name.split("#", 1)
        prefix_map = {
            "beethoven": "beethoven_piano_sonatas",
            "haydn": "haydn_piano_sonatas",
            "mozart": "mozart_piano_sonatas",
            "chopin": "humdrum_chopin_first_editions",
            "joplin": "joplin",
            "scarlatti": "scarlatti_keyboard_sonatas",
        }
        if prefix in prefix_map:
            expected_name = f"{prefix_map[prefix]}_{piece}"
            if processed_name == expected_name:
                return True

    if processed_name.startswith("musesyn_"):
        musesyn_name = processed_name[8:]
        if musesyn_name == split_name:
            return True

    return False


# Canonical implementation lives in score.sanitize_kern to avoid circular
# imports (src.a2s.piano pulls in model.py / transformers).
from src.score.sanitize_kern import extract_kern_measures  # noqa: E402


def generate_metadata(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    aug_config_path: Path = DEFAULT_AUG_CONFIG,
) -> Dict[str, Any]:
    """Generate augmentation metadata for all kern files.

    Reproduces the exact random choices made during Phase 2.
    """
    # Load augmentation config from JSON
    from src.datasets.syn.prepare_syn import render_plan_for_work

    aug_config = load_augmentation_config(aug_config_path)
    tempo_enabled = aug_config['tempo_enabled']
    tempo_range = aug_config['tempo_range']
    train_soundfonts = aug_config['train_soundfonts']
    valid_soundfonts = aug_config['valid_soundfonts']
    test_soundfonts = aug_config['test_soundfonts']
    num_versions = aug_config['num_versions']

    if tempo_enabled:
        logger.info(f"Tempo augmentation ENABLED (range: {tempo_range[0]:.2f}-{tempo_range[1]:.2f})")
    else:
        logger.info("Tempo augmentation DISABLED")

    # Read kern_measures from kern_gt/ (repeat-expanded) so measure counts
    # align with audio_measures (which are generated from expanded audio)
    kern_gt_dir = output_dir / "kern_gt"
    kern_dir = output_dir / "kern"

    # Load split files
    test_split = set()
    valid_split = set()

    test_split_path = metadata_dir / "test_split.txt"
    valid_split_path = metadata_dir / "valid_split.txt"

    # "!" is the comment marker because split names may contain "#".
    if test_split_path.exists():
        df = pd.read_csv(test_split_path, comment="!")
        test_split = set(df["name"].tolist())
    if valid_split_path.exists():
        df = pd.read_csv(valid_split_path, comment="!")
        valid_split = set(df["name"].tolist())

    all_kern_files = sorted(kern_dir.glob("*.krn"))
    excluded_stems = load_protocol_exclude(metadata_dir)
    kern_files = [
        path for path in all_kern_files
        if path.stem not in excluded_stems
    ]
    logger.info(
        "Metadata corpus: %d eligible stems, %d protocol-excluded",
        len(kern_files), len(all_kern_files) - len(kern_files),
    )
    metadata = {}

    # Cache for kern_measures (shared across versions of same kern file)
    kern_measures_cache = {}

    for kern_path in kern_files:
        stem = kern_path.stem

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

        # Extract kern_measures: line numbers from kern_gt (for ChunkedDataset slicing).
        # Timing (start_sec, end_sec) is NOT computed here — it comes from Phase 2
        # via extract_measure_times(Score), which is the single source of truth
        # matching the rendered MIDI/audio.
        if stem not in kern_measures_cache:
            kern_gt_path = kern_gt_dir / f"{stem}.krn"
            if kern_gt_path.exists():
                kern_measures_cache[stem] = extract_kern_measures(kern_gt_path)
            else:
                # Fallback to original kern if kern_gt not yet generated
                logger.warning(f"kern_gt not found for {stem}, using kern/")
                kern_measures_cache[stem] = extract_kern_measures(kern_path)
        kern_measures = kern_measures_cache[stem]

        n_versions = num_versions[split]

        # Determine soundfonts list for this split
        if split == "train":
            split_soundfonts = train_soundfonts
        elif split == "valid":
            split_soundfonts = valid_soundfonts
        else:  # test
            split_soundfonts = test_soundfonts

        render_plan = render_plan_for_work(
            stem, split, n_versions, split_soundfonts, aug_config
        )

        for version in range(n_versions):
            plan = render_plan[version]
            version_soundfonts = plan["soundfonts"]

            # Keyed by MIDI, not by audio file: timing, tempo and style belong
            # to the performance, and the timbres are renders of it.
            metadata[f"{stem}_v{version}"] = {
                "kern_file": f"{stem}.krn",
                "epr_style": plan["epr_style"],
                "split": split,
                "version": version,
                "tempo_augmented": tempo_enabled and split == "train",
                "tempo_sampling": (
                    aug_config["tempo_sampling"]
                    if (tempo_enabled and split == "train") else None
                ),
                "tempo_stratum": plan["tempo_stratum"],
                "tempo_log2": plan["tempo_log2"],
                "tempo_range": (
                    list(tempo_range)
                    if (tempo_enabled and split == "train") else None
                ),
                "kern_measures": kern_measures,
                "renders": [
                    {"soundfont": soundfont,
                     "audio_key": f"{stem}_v{version}~{soundfont[:-4]}"}
                    for soundfont in version_soundfonts
                ],
                # === To be filled by Phase 2 ===
                "tempo_scaling": plan["tempo_scaling"],
                "duration_sec": None,
                "audio_measures": None,
                "render_fingerprint": None,
            }

    return metadata


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate augmentation metadata")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (contains kern/)",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Metadata directory (contains split files)",
    )
    parser.add_argument(
        "--aug-config",
        type=Path,
        default=DEFAULT_AUG_CONFIG,
        help="Augmentation config JSON file",
    )

    args = parser.parse_args()

    logger.info("Generating augmentation metadata...")

    metadata = generate_metadata(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        aug_config_path=args.aug_config,
    )

    # Save metadata (atomic publish: a kill mid-dump must not leave a
    # truncated file for downstream readers)
    output_path = args.output_dir / "augmentation_metadata.json"
    output_tmp = output_path.with_name(f".{output_path.name}")
    with open(output_tmp, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    os.replace(output_tmp, output_path)

    # Summary
    splits = {}
    total_measures = 0
    unique_kern_files = set()

    for info in metadata.values():
        split = info["split"]
        splits[split] = splits.get(split, 0) + 1

        if info["kern_file"] not in unique_kern_files:
            unique_kern_files.add(info["kern_file"])
            total_measures += len(info["kern_measures"])

    logger.info(f"Generated metadata for {len(metadata)} MIDI entries")
    logger.info(f"  Unique kern files: {len(unique_kern_files)}")
    logger.info(f"  Total measures: {total_measures}")
    logger.info(f"  Splits: {splits}")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
