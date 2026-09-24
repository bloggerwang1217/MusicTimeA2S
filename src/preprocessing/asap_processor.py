"""
ASAP Processor
=====================================

Processes ASAP xml_score.musicxml files to Humdrum kern format through the
shared standardize_xml chain, so ASAP scores get the identical Phase 1 as
the synthetic corpora.  No meter-rewrite allowlist:
ASAP audio is real performance recordings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# standardize_xml registers converter21 at import.
from src.score.standardize_kern import write_bar_issues
from src.score.standardize_xml import phase1_bar_issues, standardize_xml

logger = logging.getLogger(__name__)


class ASAPProcessor:
    """Process ASAP MusicXML scores to cleaned kern format."""

    def __init__(
        self,
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
        """Initialize ASAP processor.

        Args:
            output_dir: Output directory for processed kern files.
            visual_dir: Output directory for visual info JSON files.
                        If None, visual info is not saved.
            repeat_map_dir: Output directory for repeat map JSON files.
                        If None, repeat maps are not saved.
            xml_dir: Output directory for the canonical rendering MusicXML.
                        The augmentation renders read it, so the audio's bars
                        are the same bars the kern supervises.
            keep_dynam: Retain **dynam spines (the kern_gt stage strips
                        them regardless).
            keep_grace / keep_trill / keep_non_trill_ornaments /
            keep_arpeggio: retain those signifiers (default: stripped).
        """
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
        self, xml_path: Path, xml_out: Optional[Path] = None,
        measure_order: Optional[List[int]] = None,
        xml_override_stem: Optional[str] = None,
    ) -> Optional[Tuple[str, List[List[Dict[str, Any]]], Dict, List[Dict]]]:
        """Process a single MusicXML file to kern.

        Returns:
            Tuple of (cleaned kern content, visual info, repeat_map,
            cleared bars), or None if processing failed.
        """
        try:
            return standardize_xml(
                Path(xml_path),
                keep_dynam=self.keep_dynam,
                keep_grace=self.keep_grace,
                keep_trill=self.keep_trill,
                keep_non_trill_ornaments=self.keep_non_trill_ornaments,
                keep_arpeggio=self.keep_arpeggio,
                # This repertoire writes free passages inside a fermata
                # bar, which the cut cannot tell from the hold itself.
                fermata_hold_cut=False,
                xml_out=xml_out,
                measure_order=measure_order,
                xml_override_stem=xml_override_stem,
            )
        except Exception as e:
            logger.error(f"Error processing {xml_path}: {e}")
            return None

    def process_piece(
        self, xml_path: Path, stem: str,
        measure_order: Optional[List[int]] = None,
    ) -> Optional[Path]:
        """Process one score and write kern / visual / repeat_map.

        ASAP names every score ``xml_score.musicxml``, so the output stem
        comes from the caller (piece id), not the file name.

        Returns:
            Path to the written kern file, or None if processing failed.
        """
        xml_out = (self.xml_dir / f"{stem}.xml") if self.xml_dir else None
        result = self.process_one(
            Path(xml_path), xml_out=xml_out, measure_order=measure_order,
            xml_override_stem=stem)
        if result is None:
            return None
        kern_cleaned, visual_info, repeat_map, cleared_bars = result

        kern_path = self.output_dir / f"{stem}.krn"
        kern_path.write_text(kern_cleaned, encoding="utf-8")
        write_bar_issues(kern_path, phase1_bar_issues(cleared_bars))
        if self.visual_dir:
            (self.visual_dir / f"{stem}.json").write_text(
                json.dumps(visual_info), encoding="utf-8")
        if self.repeat_map_dir:
            (self.repeat_map_dir / f"{stem}.json").write_text(
                json.dumps(repeat_map, indent=2, ensure_ascii=False),
                encoding="utf-8")
        return kern_path
