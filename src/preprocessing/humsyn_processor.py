"""
HumSyn Processor for MusicTime-A2S training data
================================================

Processes HumSyn kern files for training data preparation.

Supported repositories:
- beethoven-piano-sonatas
- haydn-piano-sonatas
- mozart-piano-sonatas
- joplin
- scarlatti-keyboard-sonatas
- humdrum-chopin-first-editions

**dynam spines are retained by default (dynamics render into audio);
Phase 1.5 strips them before tokenizing.  Grace notes, ornament
signifiers, and arpeggio marks are stripped by default (keep_* flags
retain them).
"""

import json
import re
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import converter21
import music21 as m21

from src.score.clean_kern import (
    clean_kern_sequence,
    extract_visual_from_sequence,
    strip_arpeggio_marks,
    strip_grace_designators,
    strip_spines,
    strip_non_trill_ornament_marks,
    strip_trill_marks,
)
from src.score.standardize_kern import strip_grace_notes
from src.score.expand_repeat import (
    expand_kern_repeats_with_mapping,
    renumber_score_measures,
)
from src.score.kern_errata import apply_kern_errata, errata_files
from src.score.kern_utils import infer_missing_meter
from src.score.sanitize_piano_score import preserve_opening_anacrusis

converter21.register()

logger = logging.getLogger(__name__)


class HumSynProcessor:
    """Process HumSyn kern files for MusicTime-A2S training.

    Handles:
    - Chopin filtering via selected_chopin.txt
    - Unified kern cleaning via clean_kern_sequence()
    """

    # Files to exclude from processing (always excluded)
    EXCLUDED: Set[str] = {"school.krn"}

    # HumSyn repository names
    REPOS = [
        "beethoven-piano-sonatas",
        "haydn-piano-sonatas",
        "mozart-piano-sonatas",
        "joplin",
        "scarlatti-keyboard-sonatas",
        "humdrum-chopin-first-editions",
    ]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        visual_dir: Optional[Path] = None,
        repeat_map_dir: Optional[Path] = None,
        selected_chopin_path: Optional[Path] = None,
        xml_dir: Optional[Path] = None,
        keep_dynam: bool = True,
        keep_grace: bool = False,
        keep_trill: bool = False,
        keep_non_trill_ornaments: bool = False,
        keep_arpeggio: bool = False,
    ):
        """Initialize HumSyn processor.

        Args:
            input_dir: Path to HumSyn directory (data/datasets/HumSyn)
            output_dir: Path to output directory for processed kern files
            visual_dir: Path to output directory for visual info JSON files.
                        If None, visual info is not saved.
            repeat_map_dir: Path to output directory for repeat map JSON files.
                        If None, repeat maps are not saved.
            selected_chopin_path: Path to selected_chopin.txt for filtering
            xml_dir: Path to output directory for MusicXML files (VirtuosoNet
                        input). If None, MusicXML is not written.
            keep_dynam: If True, retain **dynam spines (for VirtuosoNet rendering).
            keep_grace: If True, retain grace notes (default: stripped).
            keep_trill: If True, retain trill signifiers (default: stripped).
                        VirtuosoNet reads trill-mark as an input feature, so
                        this one is a real VirtuosoNet ablation lever.
            keep_non_trill_ornaments: If True, retain mordent/turn/shake/
                        schleifer signifiers (default: stripped). VirtuosoNet's
                        parser never turns these into an input feature, so
                        toggling this cannot change VirtuosoNet-rendered audio.
            keep_arpeggio: If True, retain arpeggio marks (default: stripped).
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir = Path(visual_dir) if visual_dir else None
        if self.visual_dir:
            self.visual_dir.mkdir(parents=True, exist_ok=True)
        self.repeat_map_dir = Path(repeat_map_dir) if repeat_map_dir else None
        if self.repeat_map_dir:
            self.repeat_map_dir.mkdir(parents=True, exist_ok=True)
        self.xml_dir = Path(xml_dir) if xml_dir else None
        if self.xml_dir:
            self.xml_dir.mkdir(parents=True, exist_ok=True)
        self.filter_chopin = True
        self.keep_dynam = keep_dynam
        self.keep_grace = keep_grace
        self.keep_trill = keep_trill
        self.keep_non_trill_ornaments = keep_non_trill_ornaments
        self.keep_arpeggio = keep_arpeggio

        # Load Chopin selection list (when filtering is enabled)
        self.selected_chopin: Optional[Set[str]] = None
        if self.filter_chopin and selected_chopin_path and Path(selected_chopin_path).exists():
            self.selected_chopin = self._load_selected_chopin(selected_chopin_path)
            logger.info(f"Loaded {len(self.selected_chopin)} selected Chopin pieces")

    def _load_selected_chopin(self, path: Path) -> Set[str]:
        """Load selected Chopin filenames from text file."""
        selected = set()
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove .krn extension if present
                    if line.endswith(".krn"):
                        line = line[:-4]
                    selected.add(line)
        return selected

    def _should_process_chopin(self, filename: str) -> bool:
        """Check if Chopin file should be processed based on selection."""
        if not self.filter_chopin or self.selected_chopin is None:
            return True

        # Remove .krn extension
        name = filename[:-4] if filename.endswith(".krn") else filename
        return name in self.selected_chopin

    def process_one(
        self, krn_path: Path, repo_name: str
    ) -> Optional[Tuple[str, List[List[Dict[str, Any]]], Dict]]:
        """Process a single kern file.

        Args:
            krn_path: Path to the kern file
            repo_name: Name of the HumSyn repository

        Returns:
            Tuple of (cleaned kern content, visual info, repeat_map),
            or None if file should be skipped.
            Visual info is extracted BEFORE cleaning to preserve stem/beam/position markers.
            repeat_map can be used to fold the expanded kern back to the original structure.
        """
        filename = krn_path.name

        # Check exclusion list (always applied)
        if filename in self.EXCLUDED:
            logger.info(f"Skipping excluded file: {filename}")
            return None

        # Check Chopin selection
        if repo_name == "humdrum-chopin-first-editions":
            if not self._should_process_chopin(filename):
                return None

        output_stem = f"{repo_name.replace('-', '_')}_{krn_path.stem}"

        # Read raw kern file
        with open(krn_path, "r", encoding="utf-8", errors="replace") as f:
            kern_raw = f.read()

        # Errata are stated against the source as published, so they are
        # applied before anything strips or rewrites it; kern/ and xml/
        # then inherit the correction from the same text.
        kern_raw = apply_kern_errata(kern_raw, output_stem)

        # Strip non-kern spines
        kern_raw = strip_spines(kern_raw, keep_dynam=self.keep_dynam)

        # Figure-level strips (default off): these signifiers change the
        # sounding figure, so kern/ and kern_gt/ exclude them unless the
        # matching keep flag is set.  Must run before visual extraction —
        # strip_grace_notes drops lines, which would desync visual info.
        if not self.keep_grace:
            kern_raw = strip_grace_notes(kern_raw)
            kern_raw = strip_grace_designators(kern_raw)
        if not self.keep_trill:
            kern_raw = strip_trill_marks(kern_raw)
        if not self.keep_non_trill_ornaments:
            kern_raw = strip_non_trill_ornament_marks(kern_raw)
        if not self.keep_arpeggio:
            kern_raw = strip_arpeggio_marks(kern_raw)

        # CRITICAL: Extract visual info BEFORE cleaning (preserves stem/beam/position)
        visual_info = extract_visual_from_sequence(kern_raw)

        # Apply unified kern cleaning (removes visual markers; cue is
        # resolved per-file below, after repeat expansion)
        kern_cleaned = clean_kern_sequence(kern_raw, warn_tuplet_ratio=False, strip_cue=False)

        # Expand repeats so kern/ output matches MIDI/audio playback order.
        # Expansion labels (*>[A,A,B,...], *>A, etc.) are consumed here;
        # the resulting kern has no repeat markers, just linear content.
        kern_expanded, repeat_map = expand_kern_repeats_with_mapping(kern_cleaned)

        # Resolve *cue passages here so kern/ is the single cue-resolved
        # source both Phase 1.5 (kern_gt) and Phase 2 (audio) consume.
        # Runs after expansion: cue spans are tracked on the same linear
        # form the downstream pipelines read.  strip_cue_passages keeps
        # line count intact (markers become null interpretations), so
        # repeat_map stays valid.
        # Lazy imports: prepare_syn imports this module at top level.
        from src.datasets.syn.prepare_syn import get_cue_treatment
        from src.score.sanitize_kern import sanitize_cue
        kern_expanded = sanitize_cue(kern_expanded, get_cue_treatment(output_stem))

        # A source with no meter at all leaves both the kern/ output and the
        # export below without one, and the beat grid is read off the meter.
        with_meter = infer_missing_meter(kern_expanded)
        if with_meter != kern_expanded:
            logger.warning(f"{output_stem}: no meter in source, declared one")
            kern_expanded = with_meter

        # HumSyn has no MusicXML source — this is the only point where a
        # music21 object exists at all, so it owns the canonical rendering
        # score.  Best-effort: a failure here must not block the independent
        # kern/ artifact, and Phase 2 will refuse the missing XML explicitly.
        if self.xml_dir:
            try:
                score = m21.converter.parse(kern_expanded, format="humdrum")
                opening_anacrusis = preserve_opening_anacrusis(score)
                renumber_score_measures(
                    score,
                    opening_anacrusis=opening_anacrusis,
                )
                # Lazy import avoids a module cycle: prepare_syn owns the
                # curated tempo policy and imports this processor.
                from src.datasets.syn.prepare_syn import ensure_tempo
                ensure_tempo(score, output_stem, kern_raw)
                xml_out = str(self.xml_dir / f"{output_stem}.xml")
                try:
                    score.write("musicxml", fp=xml_out)
                except m21.musicxml.xmlObjects.MusicXMLExportException:
                    # Pathological tuplet (e.g. a cross-barline cadenza
                    # fragment whose ratio isn't a power of 2) can make
                    # music21's tuplet-bracket consolidation compute an
                    # unexportable "normal type". makeNotation=False skips
                    # that consolidation — the file still exports, just
                    # without a few display-only hidden filler rests.
                    logger.warning(
                        f"{output_stem}: retrying MusicXML export with "
                        f"makeNotation=False (tuplet bracket limitation)")
                    score.write("musicxml", fp=xml_out, makeNotation=False)
            except Exception as e:
                logger.warning(f"{output_stem}: MusicXML export failed: {e}")

        return kern_expanded, visual_info, repeat_map

    def process_all(self, workers: int = 1) -> Dict[str, str]:
        """Process all HumSyn repositories.

        Args:
            workers: Number of parallel worker processes (default 1 =
                sequential). process_one is CPU-bound (kern parsing plus,
                when xml_dir is set, a full music21 parse + MusicXML
                write), so >1 speeds up wall-clock time on multi-core
                machines. Relies on the fork start method (Linux default)
                to inherit the already-registered converter21 state.

        Returns:
            Dictionary mapping {output_filename: status}
            Status is one of: "success", "skipped", "error: <message>"
        """
        results: Dict[str, str] = {}

        tasks: List[Tuple[Path, str, str]] = []
        for repo_name in self.REPOS:
            repo_path = self.input_dir / repo_name / "kern"

            if not repo_path.exists():
                logger.warning(f"Repository not found: {repo_path}")
                continue

            logger.info(f"Found {repo_name}")
            for krn_path in sorted(repo_path.glob("*.krn")):
                output_name = f"{repo_name.replace('-', '_')}_{krn_path.stem}.krn"
                tasks.append((krn_path, repo_name, output_name))

        # A correction that names a file the corpus no longer has would
        # otherwise go unnoticed: per-file application never runs for it.
        known = {name[:-len(".krn")] for _, _, name in tasks}
        missing = [stem for stem in errata_files() if stem not in known]
        if missing:
            raise ValueError(
                f"kern_errata names files absent from the HumSyn corpus: "
                f"{', '.join(missing)}")

        def write_result(output_name: str, result) -> None:
            if result is None:
                results[output_name] = "skipped"
                return

            kern_cleaned, visual_info, repeat_map = result

            # Write cleaned kern (repeat-expanded)
            output_path = self.output_dir / output_name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(kern_cleaned)

            # Write visual info if visual_dir is configured
            if self.visual_dir:
                visual_path = self.visual_dir / output_name.replace(".krn", ".json")
                with open(visual_path, "w", encoding="utf-8") as f:
                    json.dump(visual_info, f)

            # Write repeat map if repeat_map_dir is configured
            if self.repeat_map_dir:
                map_path = self.repeat_map_dir / output_name.replace(".krn", ".json")
                with open(map_path, "w", encoding="utf-8") as f:
                    json.dump(repeat_map, f, indent=2, ensure_ascii=False)

            results[output_name] = "success"

        if workers <= 1:
            for krn_path, repo_name, output_name in tasks:
                try:
                    write_result(output_name, self.process_one(krn_path, repo_name))
                except Exception as e:
                    logger.error(f"Error processing {krn_path}: {e}")
                    results[output_name] = f"error: {e}"
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.process_one, krn_path, repo_name): (krn_path, output_name)
                    for krn_path, repo_name, output_name in tasks
                }
                for future in as_completed(futures):
                    krn_path, output_name = futures[future]
                    try:
                        write_result(output_name, future.result())
                    except Exception as e:
                        logger.error(f"Error processing {krn_path}: {e}")
                        results[output_name] = f"error: {e}"

        # Summary
        success = sum(1 for v in results.values() if v == "success")
        skipped = sum(1 for v in results.values() if v == "skipped")
        errors = sum(1 for v in results.values() if v.startswith("error"))
        logger.info(f"HumSyn processing complete: {success} success, {skipped} skipped, {errors} errors")

        return results


def main():
    """CLI entry point for HumSyn processing."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process HumSyn kern files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/HumSyn"),
        help="Input HumSyn directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/syn/kern"),
        help="Output directory for processed kern files",
    )
    parser.add_argument(
        "--selected-chopin",
        type=Path,
        default=Path("src/datasets/syn/selected_chopin.txt"),
        help="Path to selected Chopin file list",
    )
    parser.add_argument(
        "--keep-grace",
        action="store_true",
        help="Retain grace notes (default: stripped)",
    )
    parser.add_argument(
        "--keep-trill",
        action="store_true",
        help="Retain trill signifiers (default: stripped)",
    )
    parser.add_argument(
        "--keep-non-trill-ornaments",
        action="store_true",
        help="Retain mordent/turn/shake/schleifer signifiers (default: stripped)",
    )
    parser.add_argument(
        "--keep-arpeggio",
        action="store_true",
        help="Retain arpeggio marks (default: stripped)",
    )

    args = parser.parse_args()

    processor = HumSynProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        selected_chopin_path=args.selected_chopin,
        keep_grace=args.keep_grace,
        keep_trill=args.keep_trill,
        keep_non_trill_ornaments=args.keep_non_trill_ornaments,
        keep_arpeggio=args.keep_arpeggio,
    )

    results = processor.process_all()

    # Print summary
    print(f"\nProcessed {len(results)} files:")
    for status in ["success", "skipped"]:
        count = sum(1 for v in results.values() if v == status)
        print(f"  {status}: {count}")
    error_count = sum(1 for v in results.values() if v.startswith("error"))
    if error_count:
        print(f"  errors: {error_count}")


if __name__ == "__main__":
    main()
