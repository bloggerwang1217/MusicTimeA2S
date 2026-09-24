"""
MuseSyn Processor
=====================================

Processes MuseSyn MusicXML files to Humdrum kern format for training data preparation.

Pipeline:
1. Parse MusicXML with music21
2. Sanitize score (fix cross-staff, hidden notes, etc.)
3. Extract repeat structure (barlines, DaCapo/Segno, volta brackets)
4. Expand repeats at music21 level (if present)
5. Build rich repeat_map (expanded → original measure mapping)
6. Export to Humdrum kern via converter21
7. Clean kern sequence (remove visual tokens)
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

# standardize_xml registers converter21 at import.
from src.score.standardize_kern import write_bar_issues
from src.score.standardize_xml import phase1_bar_issues, standardize_xml

logger = logging.getLogger(__name__)


class MuseSynProcessor:
    """Process MuseSyn MusicXML files to cleaned kern format.

    Uses converter21 for MusicXML → Humdrum conversion (better success rate than verovio).
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        visual_dir: Optional[Path] = None,
        repeat_map_dir: Optional[Path] = None,
        xml_dir: Optional[Path] = None,
        keep_dynam: bool = True,
        keep_grace: bool = False,
        keep_trill: bool = False,
        keep_non_trill_ornaments: bool = False,
        keep_arpeggio: bool = False,
    ):
        """Initialize MuseSyn processor.

        Args:
            input_dir: Path to MuseSyn directory (data/datasets/MuseSyn)
            output_dir: Path to output directory for processed kern files
            visual_dir: Path to output directory for visual info JSON files.
                        If None, visual info is not saved.
            repeat_map_dir: Path to output directory for repeat map JSON files.
                        If None, repeat maps are not saved.
            xml_dir: Path to output directory for MusicXML files (VirtuosoNet
                        input). If None, MusicXML is not written.
            keep_dynam: Retain **dynam spines (for VirtuosoNet rendering).
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
        self.keep_dynam = keep_dynam
        self.keep_grace = keep_grace
        self.keep_trill = keep_trill
        self.keep_non_trill_ornaments = keep_non_trill_ornaments
        self.keep_arpeggio = keep_arpeggio

    def process_one(
        self, xml_path: Path
    ) -> Optional[Tuple[str, List[List[Dict[str, Any]]], Dict, List[Dict]]]:
        """Process a single MusicXML file to kern.

        Args:
            xml_path: Path to the MusicXML file

        Returns:
            Tuple of (cleaned kern content, visual info, repeat_map,
            cleared bars),
            or None if processing failed.
            Visual info is extracted from converter21 output BEFORE cleaning.
        """
        def tempo_policy(score_for_xml) -> None:
            # Lazy import avoids a module cycle: prepare_syn owns the
            # curated tempo policy and imports this processor.
            from src.datasets.syn.prepare_syn import ensure_tempo
            ensure_tempo(score_for_xml, f"musesyn_{xml_path.stem}")

        try:
            return standardize_xml(
                xml_path,
                keep_dynam=self.keep_dynam,
                keep_grace=self.keep_grace,
                keep_trill=self.keep_trill,
                keep_non_trill_ornaments=self.keep_non_trill_ornaments,
                keep_arpeggio=self.keep_arpeggio,
                xml_out=(
                    self.xml_dir / f"{xml_path.stem}.xml"
                    if self.xml_dir else None),
                tempo_policy=tempo_policy if self.xml_dir else None,
            )
        except Exception as e:
            logger.error(f"Error processing {xml_path}: {e}")
            return None


    def process_all(self, workers: int = 1) -> Dict[str, str]:
        """Process all MusicXML files in MuseSyn directory.

        Args:
            workers: Number of parallel worker processes (default 1 =
                sequential). process_one is CPU-bound (music21 parsing,
                repeat expansion, and, when xml_dir is set, a deep copy +
                MusicXML write), so >1 speeds up wall-clock time on
                multi-core machines. Relies on the fork start method
                (Linux default) to inherit the already-registered
                converter21 state.

        Returns:
            Dictionary mapping {output_filename: status}
            Status is one of: "success", "error: <message>"
        """
        results: Dict[str, str] = {}

        # Find all MusicXML files
        xml_patterns = ["*.xml", "*.musicxml", "*.mxl"]
        xml_files = []
        for pattern in xml_patterns:
            xml_files.extend(self.input_dir.glob(f"**/{pattern}"))

        xml_files = sorted(set(xml_files))
        logger.info(f"Found {len(xml_files)} MusicXML files in MuseSyn")

        def write_result(output_name: str, xml_path: Path, result) -> None:
            if result is None:
                results[output_name] = "error: processing failed"
                return

            kern_cleaned, visual_info, repeat_map, cleared_bars = result

            # Write cleaned kern (repeat-expanded)
            output_path = self.output_dir / output_name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(kern_cleaned)
            write_bar_issues(output_path, phase1_bar_issues(cleared_bars))

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
            for xml_path in xml_files:
                output_name = f"musesyn_{xml_path.stem}.krn"
                try:
                    write_result(output_name, xml_path, self.process_one(xml_path))
                except Exception as e:
                    logger.error(f"Error processing {xml_path}: {e}")
                    results[output_name] = f"error: {e}"
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.process_one, xml_path): xml_path
                    for xml_path in xml_files
                }
                for future in as_completed(futures):
                    xml_path = futures[future]
                    output_name = f"musesyn_{xml_path.stem}.krn"
                    try:
                        write_result(output_name, xml_path, future.result())
                    except Exception as e:
                        logger.error(f"Error processing {xml_path}: {e}")
                        results[output_name] = f"error: {e}"

        # Summary
        success = sum(1 for v in results.values() if v == "success")
        errors = sum(1 for v in results.values() if v.startswith("error"))
        logger.info(f"MuseSyn processing complete: {success} success, {errors} errors")

        return results


def main():
    """CLI entry point for MuseSyn processing."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process MuseSyn MusicXML files to kern")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/datasets/MuseSyn"),
        help="Input MuseSyn directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/syn/kern"),
        help="Output directory for processed kern files",
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

    processor = MuseSynProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        keep_grace=args.keep_grace,
        keep_trill=args.keep_trill,
        keep_non_trill_ornaments=args.keep_non_trill_ornaments,
        keep_arpeggio=args.keep_arpeggio,
    )

    results = processor.process_all()

    # Print summary
    print(f"\nProcessed {len(results)} files:")
    success = sum(1 for v in results.values() if v == "success")
    errors = sum(1 for v in results.values() if v.startswith("error"))
    print(f"  success: {success}")
    print(f"  errors: {errors}")

    # Print error details
    if errors > 0:
        print("\nError details:")
        for name, status in results.items():
            if status.startswith("error"):
                print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
