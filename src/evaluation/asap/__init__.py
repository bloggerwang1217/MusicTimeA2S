#!/usr/bin/env python3
"""
ASAP Dataset Handling Module

Standalone module for ASAP (Aligned Scores and Performances) dataset operations.
Can be used on any system's output.

ASAP Dataset Structure:
    asap-dataset/
    ├── Bach/
    │   ├── Prelude/
    │   │   ├── bwv_875/
    │   │   │   ├── midi_score.mid      <- Ground truth MIDI
    │   │   │   ├── xml_score.musicxml  <- Ground truth MusicXML
    │   │   │   └── <performance_id>/   <- Performance recordings
    │   │   └── ...
    │   └── ...
    └── ...

Chunk Evaluation:
    For 5-bar chunk evaluation (Zeng et al.), this module supports:
    1. Loading chunk definitions from CSV
    2. Extracting measure ranges from MIDI/MusicXML
    3. Matching predictions to ground truth

Usage:
    from evaluation.asap import ASAPDataset, ChunkInfo

    asap = ASAPDataset("/path/to/asap-dataset")
    gt_path = asap.find_ground_truth("Bach_Prelude_bwv_875_performance1")

    # For chunk evaluation
    chunks = asap.load_chunks("/path/to/zeng_test_chunk_set.csv")
"""

import bisect
import copy
import csv
import hashlib
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Asap102MappingError(ValueError):
    """A recording cannot be assigned one frozen source-measure route."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_asap102_inventory(
    metadata_path: Path,
    asap_root: Path,
    *,
    expected_count: int | None = 102,
    require_audio: bool = True,
) -> list[dict[str, Any]]:
    """Load the exact recording-level ASAP-102 inventory.

    A performance that the ASAP-102 metadata carries as MIDI only has no audio cell, and its
    rendering is named by the caller; such a set is loaded with
    ``require_audio`` off and its own ``expected_count``.
    """
    recordings: list[dict[str, Any]] = []
    seen: set[str] = set()
    with metadata_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") != "ASAP" or row.get("split") != "test":
                continue
            external = row.get("performance_MIDI_external", "")
            prefix = "{ASAP}/"
            if not external.startswith(prefix):
                raise Asap102MappingError(f"Not an ASAP performance path: {external!r}")
            relative_midi = Path(external[len(prefix):])
            audio_external = row.get("performance_audio_external", "")
            performance_audio: Path | None = None
            if audio_external.startswith(prefix):
                performance_audio = asap_root / Path(audio_external[len(prefix):])
            elif require_audio:
                raise Asap102MappingError(
                    f"Not an ASAP performance audio path: {audio_external!r}"
                )
            annotation_key = relative_midi.as_posix()
            if annotation_key in seen:
                raise Asap102MappingError(
                    f"Duplicate ASAP-102 recording path: {annotation_key}"
                )
            seen.add(annotation_key)
            piece_path = relative_midi.parent
            piece_id = "#".join(piece_path.parts)
            performance_id = relative_midi.stem
            recording_id = f"{piece_id}#{performance_id}"
            artifact_stem = "__".join((*piece_path.parts, performance_id))
            recordings.append({
                "recording_id": recording_id,
                "artifact_stem": artifact_stem,
                "asap102_performance_id": row.get("performance_id", ""),
                "piece_id": piece_id,
                "performance_id": performance_id,
                "annotation_key": annotation_key,
                "source_xml": asap_root / piece_path / "xml_score.musicxml",
                "performance_audio": performance_audio,
                "performance_midi": asap_root / relative_midi,
                "performance_annotation": (
                    asap_root / piece_path / f"{performance_id}_annotations.txt"
                ),
                "metadata_aligned": row.get("aligned", ""),
            })
    recordings.sort(key=lambda value: value["recording_id"])
    if expected_count is not None and len(recordings) != expected_count:
        raise Asap102MappingError(
            f"ASAP-102 inventory must contain {expected_count} recordings, "
            f"found {len(recordings)}"
        )
    for recording in recordings:
        for key in ("source_xml", "performance_midi", "performance_annotation"):
            if not recording[key].is_file():
                raise FileNotFoundError(
                    f"ASAP-102 input is missing for {recording['recording_id']}: "
                    f"{recording[key]}"
                )
    return recordings


def parse_downbeats_score_map(value: Any) -> list[tuple[int, ...]] | None:
    """Parse ASAP map cells while retaining connected source measures."""
    if not isinstance(value, list):
        return None
    groups: list[tuple[int, ...]] = []
    for cell in value:
        if isinstance(cell, bool):
            raise Asap102MappingError(f"Invalid boolean score-map cell: {cell!r}")
        if isinstance(cell, int):
            group = (cell,)
        else:
            text = str(cell)
            if not re.fullmatch(r"\d+(?:-\d+)*", text):
                raise Asap102MappingError(f"Invalid score-map cell: {cell!r}")
            group = tuple(int(part) for part in text.split("-"))
        groups.append(group)
    return groups


def load_mapping_adjudications(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Mapping adjudication file does not exist: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Asap102MappingError("Mapping adjudications must be a JSON object")
    return value


def _source_has_pickup(score: Any) -> bool:
    from music21 import stream

    first_measures = [
        list(part.getElementsByClass(stream.Measure))[0]
        for part in score.parts
    ]
    if any(measure.paddingLeft > 0 for measure in first_measures):
        return True
    return not any(
        note.beat == 1
        for measure in first_measures
        for note in measure.recurse().notes
    )


def _notation_routes(score: Any, expected_downbeats: int) -> list[tuple[str, list[int]]]:
    from music21 import repeat, stream

    first_part = score.parts[0]
    measures = list(first_part.getElementsByClass(stream.Measure))
    written = list(range(len(measures)))
    routes: list[tuple[str, list[int]]] = [("written", written)]
    repeat_marks = list(first_part.recurse().getElementsByClass(repeat.RepeatMark))
    if repeat_marks:
        try:
            expanded = list(repeat.Expander(streamObj=first_part).measureMap())
        except Exception as error:
            raise Asap102MappingError(f"Repeat expansion failed: {error}") from error
        routes.append(("expanded", expanded))

    pickup = _source_has_pickup(score)
    candidates: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for provenance, route in routes:
        normalized = route[1:] if pickup and route and route[0] == 0 else route
        key = tuple(normalized)
        if len(normalized) == expected_downbeats and key not in seen:
            seen.add(key)
            candidates.append((provenance, normalized))
    return candidates


def _b_r_positions(
    annotation: dict[str, Any],
    *,
    pickup_inserted: bool,
    route_length: int,
) -> list[int]:
    downbeats = sorted(float(value) for value in annotation["performance_downbeats"])
    result: set[int] = set()
    for raw_time, marker in annotation.get("performance_beats_type", {}).items():
        if marker != "bR":
            continue
        index = bisect.bisect_right(downbeats, float(raw_time)) - 1
        position = 0 if index < 0 else index + int(pickup_inserted)
        if 0 <= position < route_length:
            result.add(position)
    return sorted(result)


def resolve_asap102_mapping(
    recording: dict[str, Any],
    annotation: dict[str, Any],
    score: Any,
    adjudications: dict[str, dict[str, Any]],
    *,
    mapping_policy: str = "source-faithful",
) -> dict[str, Any]:
    """Resolve one recording without consulting any system prediction."""
    if mapping_policy not in {"source-faithful", "approved-fallback"}:
        raise Asap102MappingError(f"Unknown mapping policy: {mapping_policy}")
    measures_by_part = [
        list(part.getElementsByClass("Measure")) for part in score.parts
    ]
    counts = {len(measures) for measures in measures_by_part}
    if len(counts) != 1:
        raise Asap102MappingError(
            f"Source parts disagree on measure count: {recording['recording_id']}"
        )
    source_measure_count = counts.pop()
    annotation_key = recording["annotation_key"]
    adjudication = adjudications.get(annotation_key)
    official = parse_downbeats_score_map(annotation.get("downbeats_score_map"))
    use_fallback = mapping_policy == "approved-fallback" and adjudication is not None

    if official is not None:
        groups = list(official)
        provenance = "official"
        if use_fallback:
            if adjudication.get("status") != "user_approved":
                raise Asap102MappingError(
                    f"Mapping fallback lacks user approval: {annotation_key}"
                )
            if adjudication.get("operation") != "drop_official_positions":
                raise Asap102MappingError(
                    f"Unsupported official-map adjudication: {annotation_key}"
                )
            drop = adjudication.get("drop_official_positions")
            if not isinstance(drop, list) or not all(isinstance(i, int) for i in drop):
                raise Asap102MappingError(
                    f"Invalid official-map drop list: {annotation_key}"
                )
            expected = adjudication.get("expected_source_groups", {})
            for index in drop:
                if not 0 <= index < len(groups):
                    raise Asap102MappingError(
                        f"Official-map drop is out of range: {annotation_key}:{index}"
                    )
                if str(index) in expected and list(groups[index]) != expected[str(index)]:
                    raise Asap102MappingError(
                        f"Official-map evidence changed: {annotation_key}:{index}"
                    )
            drop_set = set(drop)
            groups = [group for index, group in enumerate(groups) if index not in drop_set]
            provenance = "user_approved_official_adjustment"
    elif use_fallback:
        if adjudication.get("status") != "user_approved":
            raise Asap102MappingError(
                f"Mapping fallback lacks user approval: {annotation_key}"
            )
        route = adjudication.get("source_measure_ordinals")
        if not isinstance(route, list) or not all(isinstance(i, int) for i in route):
            raise Asap102MappingError(
                f"Adjudicated route must be an explicit ordinal list: {annotation_key}"
            )
        groups = [(ordinal,) for ordinal in route]
        provenance = "user_approved_fallback"
    elif annotation.get("score_and_performance_aligned") is True:
        expected = len(annotation.get("midi_score_downbeats", []))
        candidates = _notation_routes(score, expected)
        if len(candidates) != 1:
            labels = [label for label, _ in candidates]
            raise Asap102MappingError(
                f"Notation route is not unique for {annotation_key}: {labels}"
            )
        provenance, route = candidates[0]
        groups = [(ordinal,) for ordinal in route]
    else:
        groups = [(ordinal,) for ordinal in range(source_measure_count)]
        provenance = "source_written_unaligned"

    for group in groups:
        if not group:
            raise Asap102MappingError(f"Empty source-measure group: {annotation_key}")
        for ordinal in group:
            if not 0 <= ordinal < source_measure_count:
                raise Asap102MappingError(
                    f"Source ordinal {ordinal} is out of range for {annotation_key}"
                )

    pickup_inserted = (
        _source_has_pickup(score)
        and groups
        and 0 not in groups[0]
    )
    if pickup_inserted:
        groups.insert(0, (0,))

    undefined_positions = _b_r_positions(
        annotation,
        pickup_inserted=pickup_inserted,
        route_length=len(groups),
    )
    return {
        "mapping_provenance": provenance,
        "mapping_policy": mapping_policy,
        "source_measure_groups": [list(group) for group in groups],
        "source_measure_count": source_measure_count,
        "pickup_inserted": pickup_inserted,
        "undefined_positions": undefined_positions,
        "coarse_aligned": annotation.get("score_and_performance_aligned"),
        "performance_downbeats": len(annotation.get("performance_downbeats", [])),
        "score_downbeats": len(annotation.get("midi_score_downbeats", [])),
    }


def valid_five_bar_positions(
    source_measure_groups: Sequence[Sequence[int]],
    undefined_positions: Sequence[int],
) -> list[int]:
    undefined = set(undefined_positions)
    return [
        position
        for position in range(max(0, len(source_measure_groups) - 4))
        if not undefined.intersection(range(position, position + 5))
    ]


_ACTIVE_ATTRIBUTE_TAGS = {
    "divisions", "key", "time", "staves", "clef", "staff-details",
    "transpose", "measure-style",
}
_PLAYBACK_SOUND_ATTRIBUTES = {
    "dacapo", "dalsegno", "segno", "coda", "tocoda", "fine",
    "forward-repeat",
}
_PLAYBACK_WORDS = re.compile(
    r"\b(?:d\.?\s*c\.?|d\.?\s*s\.?|da\s+capo|dal\s+segno|"
    r"to\s+coda|fine)\b",
    re.IGNORECASE,
)


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child) == name]


def _attribute_key(element: ET.Element) -> tuple[str, str | None]:
    return _local_name(element), element.get("number")


def _measure_contexts(
    measures: Sequence[ET.Element],
) -> tuple[list[dict[tuple[str, str | None], ET.Element]], list[ET.Element | None]]:
    attributes: dict[tuple[str, str | None], ET.Element] = {}
    tempo_direction: ET.Element | None = None
    attribute_contexts: list[dict[tuple[str, str | None], ET.Element]] = []
    tempo_contexts: list[ET.Element | None] = []
    for measure in measures:
        attribute_contexts.append(copy.deepcopy(attributes))
        tempo_contexts.append(copy.deepcopy(tempo_direction))
        for child in measure:
            if _local_name(child) == "attributes":
                for value in child:
                    if _local_name(value) in _ACTIVE_ATTRIBUTE_TAGS:
                        attributes[_attribute_key(value)] = copy.deepcopy(value)
            elif _local_name(child) == "direction":
                has_tempo = any(
                    _local_name(value) == "metronome"
                    for value in child.iter()
                ) or any(
                    _local_name(value) == "sound" and value.get("tempo") is not None
                    for value in child.iter()
                )
                if has_tempo:
                    tempo_direction = copy.deepcopy(child)
    return attribute_contexts, tempo_contexts


def _inject_measure_context(
    measure: ET.Element,
    attributes: dict[tuple[str, str | None], ET.Element],
    tempo_direction: ET.Element | None,
) -> None:
    children = list(measure)
    rhythmic_indices = [
        index for index, child in enumerate(children)
        if _local_name(child) in {"note", "backup", "forward"}
    ]
    first_rhythmic = min(rhythmic_indices, default=len(children))
    leading_attributes = [
        child for index, child in enumerate(children)
        if _local_name(child) == "attributes" and index < first_rhythmic
    ]
    if leading_attributes:
        target = leading_attributes[0]
    else:
        target = ET.Element("attributes")
        insert_at = 0
        while (
            insert_at < len(children)
            and _local_name(children[insert_at]) == "print"
        ):
            insert_at += 1
        measure.insert(insert_at, target)
    present = {
        _attribute_key(value)
        for node in leading_attributes
        for value in node
        if _local_name(value) in _ACTIVE_ATTRIBUTE_TAGS
    }
    missing = [
        copy.deepcopy(value)
        for key, value in attributes.items()
        if key not in present
    ]
    for value in reversed(missing):
        target.insert(0, value)

    leading_directions = [
        child for index, child in enumerate(children)
        if _local_name(child) == "direction" and index < first_rhythmic
    ]
    has_tempo = any(
        _local_name(value) == "metronome"
        for direction in leading_directions
        for value in direction.iter()
    ) or any(
        _local_name(value) == "sound" and value.get("tempo") is not None
        for direction in leading_directions
        for value in direction.iter()
    )
    if tempo_direction is not None and not has_tempo:
        children = list(measure)
        insert_at = next(
            (
                index for index, child in enumerate(children)
                if _local_name(child) in {"note", "backup", "forward"}
            ),
            len(children),
        )
        measure.insert(insert_at, copy.deepcopy(tempo_direction))


def _remove_materialized_playback_controls(measure: ET.Element) -> None:
    for parent in measure.iter():
        for child in list(parent):
            if _local_name(child) in {"repeat", "ending"}:
                parent.remove(child)
    for direction in list(_direct_children(measure, "direction")):
        is_playback_control = False
        for value in direction.iter():
            name = _local_name(value)
            if name in {"segno", "coda"}:
                is_playback_control = True
            elif name == "sound" and _PLAYBACK_SOUND_ATTRIBUTES.intersection(
                value.attrib
            ):
                is_playback_control = True
            elif name == "words" and _PLAYBACK_WORDS.search(value.text or ""):
                is_playback_control = True
        if is_playback_control:
            measure.remove(direction)


def performance_order_musicxml_bytes(
    source_xml: Path,
    source_measure_groups: Sequence[Sequence[int]],
) -> bytes:
    """Assemble source measures by ordinal while retaining their notation."""
    tree = ET.parse(source_xml)
    root = tree.getroot()
    parts = _direct_children(root, "part")
    if not parts:
        raise Asap102MappingError(f"MusicXML has no parts: {source_xml}")
    for part in parts:
        source_measures = _direct_children(part, "measure")
        attribute_contexts, tempo_contexts = _measure_contexts(source_measures)
        for measure in source_measures:
            part.remove(measure)
        previous_ordinal: int | None = None
        for position, group in enumerate(source_measure_groups, start=1):
            for segment, ordinal in enumerate(group):
                if not 0 <= ordinal < len(source_measures):
                    raise Asap102MappingError(
                        f"Source ordinal {ordinal} is out of range in {source_xml}"
                    )
                measure = copy.deepcopy(source_measures[ordinal])
                _remove_materialized_playback_controls(measure)
                suffix = "" if len(group) == 1 else chr(ord("a") + segment)
                measure.set("number", f"{position}{suffix}")
                if previous_ordinal is None or ordinal != previous_ordinal + 1:
                    _inject_measure_context(
                        measure,
                        attribute_contexts[ordinal],
                        tempo_contexts[ordinal],
                    )
                part.append(measure)
                previous_ordinal = ordinal
    ET.indent(tree, space="  ")
    body = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    declaration, newline, remainder = body.partition(b"\n")
    doctype = (
        b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 '
        b'Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
    )
    return declaration + b"\n" + doctype + b"\n" + remainder + b"\n"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ChunkInfo:
    """
    Information about a 5-bar chunk for evaluation.

    Used for Zeng-style chunk evaluation where full songs are
    divided into 5-bar segments for fine-grained assessment.

    Attributes:
        chunk_id: Unique identifier for the chunk
        piece_id: Identifier of the parent piece
        start_measure: Starting measure number (1-indexed)
        end_measure: Ending measure number (inclusive)
        asap_path: Relative path within ASAP dataset
    """

    chunk_id: str
    piece_id: str
    start_measure: int
    end_measure: int
    asap_path: str = ""

    def __repr__(self) -> str:
        return f"ChunkInfo({self.chunk_id}, m{self.start_measure}-{self.end_measure})"


@dataclass
class PieceInfo:
    """
    Information about a piece in the ASAP dataset.

    Attributes:
        piece_id: Unique identifier (e.g., "Bach_Prelude_bwv_875")
        composer: Composer name
        work: Work name
        piece: Piece name (may be empty)
        midi_score_path: Path to ground truth MIDI
        xml_score_path: Path to ground truth MusicXML
        performances: List of performance IDs
    """

    piece_id: str
    composer: str
    work: str
    piece: str = ""
    midi_score_path: str = ""
    xml_score_path: str = ""
    performances: List[str] = field(default_factory=list)


# =============================================================================
# ASAP DATASET CLASS
# =============================================================================


class ASAPDataset:
    """
    ASAP dataset handler for ground truth operations.

    Provides utilities for:
    - Finding ground truth files for predictions
    - Loading chunk definitions for 5-bar evaluation
    - Iterating over pieces and performances

    Args:
        base_dir: Path to ASAP dataset root directory

    Example:
        asap = ASAPDataset("/data/asap-dataset")

        # Find ground truth for a prediction
        gt_path = asap.find_ground_truth_midi("Bach_Prelude_bwv_875_perf1")

        # Load chunks for evaluation
        chunks = asap.load_chunks("zeng_test_chunks.csv")
    """

    # Common ground truth filenames in ASAP
    GT_MIDI_NAMES = ["midi_score.mid", "midi_score.midi"]
    GT_XML_NAMES = ["xml_score.musicxml", "xml_score.xml"]

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

        if not self.base_dir.exists():
            raise FileNotFoundError(f"ASAP dataset not found: {base_dir}")

        logger.info(f"ASAP dataset initialized: {base_dir}")

    def find_ground_truth_midi(
        self,
        pred_identifier: str,
        pred_base_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find ground truth MIDI for a prediction file.

        Tries multiple naming conventions and path patterns to match.

        Args:
            pred_identifier: Prediction file stem or path
                Examples:
                - "Bach_Prelude_bwv_875_performance1"
                - "/path/to/pred/Bach_Prelude_bwv_875_performance1.mid"
            pred_base_dir: Base directory of predictions (for relative path extraction)

        Returns:
            Path to ground truth MIDI or None if not found
        """
        # Extract filename stem if full path provided
        if os.path.sep in pred_identifier or pred_identifier.endswith((".mid", ".midi")):
            pred_name = Path(pred_identifier).stem
        else:
            pred_name = pred_identifier

        # Try to parse naming convention: Composer_Work_Piece_PerformanceID
        # or: Composer_Work_PerformanceID
        parts = pred_name.split("_")

        if len(parts) >= 3:
            composer = parts[0]

            # Try different path structures
            for i in range(1, len(parts)):
                for j in range(i + 1, len(parts) + 1):
                    work = "_".join(parts[1:j])
                    piece = "_".join(parts[j:-1]) if j < len(parts) - 1 else ""

                    # Try: Composer/Work/Piece/midi_score.mid
                    if piece:
                        for gt_name in self.GT_MIDI_NAMES:
                            gt_path = self.base_dir / composer / work / piece / gt_name
                            if gt_path.exists():
                                return str(gt_path)

                    # Try: Composer/Work/midi_score.mid
                    for gt_name in self.GT_MIDI_NAMES:
                        gt_path = self.base_dir / composer / work / gt_name
                        if gt_path.exists():
                            return str(gt_path)

        logger.debug(f"No ground truth found for: {pred_identifier}")
        return None

    def find_ground_truth_xml(
        self,
        pred_identifier: str,
    ) -> Optional[str]:
        """
        Find ground truth MusicXML for a prediction file.

        Args:
            pred_identifier: Prediction file stem or path

        Returns:
            Path to ground truth MusicXML or None if not found
        """
        # First find the MIDI ground truth
        midi_path = self.find_ground_truth_midi(pred_identifier)
        if midi_path is None:
            return None

        # Look for XML in same directory
        midi_dir = Path(midi_path).parent
        for xml_name in self.GT_XML_NAMES:
            xml_path = midi_dir / xml_name
            if xml_path.exists():
                return str(xml_path)

        return None

    def find_ground_truth_midi_by_piece_id(self, piece_id: str) -> Optional[str]:
        """
        Find ground truth MIDI using Zeng-style piece_id.

        Converts '#' separator to path structure for ASAP dataset lookup.

        Args:
            piece_id: Zeng-style piece identifier (e.g., 'Bach#Prelude#bwv_875')

        Returns:
            Path to ground truth MIDI or None if not found

        Example:
            piece_id = 'Bach#Prelude#bwv_875'
            -> searches: asap-dataset/Bach/Prelude/bwv_875/midi_score.mid
        """
        # Convert '#' separator to path components
        # Bach#Prelude#bwv_875 -> ['Bach', 'Prelude', 'bwv_875']
        path_parts = piece_id.split("#")

        for gt_name in self.GT_MIDI_NAMES:
            gt_path = self.base_dir / "/".join(path_parts) / gt_name
            if gt_path.exists():
                return str(gt_path)

        logger.debug(f"No ground truth MIDI found for piece_id: {piece_id}")
        return None

    def find_ground_truth_xml_by_piece_id(self, piece_id: str) -> Optional[str]:
        """
        Find ground truth MusicXML using Zeng-style piece_id.

        Converts '#' separator to path structure for ASAP dataset lookup.

        Args:
            piece_id: Zeng-style piece identifier (e.g., 'Bach#Prelude#bwv_875')

        Returns:
            Path to ground truth MusicXML or None if not found

        Example:
            piece_id = 'Bach#Prelude#bwv_875'
            -> searches: asap-dataset/Bach/Prelude/bwv_875/xml_score.musicxml
        """
        # Convert '#' separator to path components
        path_parts = piece_id.split("#")

        for gt_name in self.GT_XML_NAMES:
            gt_path = self.base_dir / "/".join(path_parts) / gt_name
            if gt_path.exists():
                return str(gt_path)

        logger.debug(f"No ground truth XML found for piece_id: {piece_id}")
        return None

    def load_chunks(self, csv_path: str) -> List[ChunkInfo]:
        """
        Load chunk definitions from CSV file.

        Supports two CSV formats:
        1. Standard: chunk_id,piece_id,start_measure,end_measure,asap_path
        2. Zeng format: chunk_id,piece,performance,chunk_index,start_measure,end_measure

        For Zeng format, 'piece' column is used as piece_id.
        The '#' separator in piece_id (e.g., 'Bach#Prelude#bwv_875') is preserved.

        Args:
            csv_path: Path to chunk CSV file

        Returns:
            List of ChunkInfo objects
        """
        chunks = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both 'piece_id' (standard) and 'piece' (Zeng format)
                piece_id = row.get("piece_id") or row.get("piece", "")

                chunk = ChunkInfo(
                    chunk_id=row.get("chunk_id", ""),
                    piece_id=piece_id,
                    start_measure=int(row.get("start_measure", 0)),
                    end_measure=int(row.get("end_measure", 0)),
                    asap_path=row.get("asap_path", ""),
                )
                chunks.append(chunk)

        logger.info(f"Loaded {len(chunks)} chunks from {csv_path}")
        return chunks

    def group_chunks_by_piece(
        self,
        chunks: List[ChunkInfo],
    ) -> Dict[str, List[ChunkInfo]]:
        """
        Group chunks by piece ID.

        Args:
            chunks: List of ChunkInfo objects

        Returns:
            Dictionary mapping piece_id to list of chunks
        """
        grouped: Dict[str, List[ChunkInfo]] = {}
        for chunk in chunks:
            if chunk.piece_id not in grouped:
                grouped[chunk.piece_id] = []
            grouped[chunk.piece_id].append(chunk)
        return grouped

    def iter_pieces(self) -> List[PieceInfo]:
        """
        Iterate over all pieces in the dataset.

        Yields:
            PieceInfo objects for each piece found
        """
        pieces = []

        for composer_dir in self.base_dir.iterdir():
            if not composer_dir.is_dir():
                continue
            composer = composer_dir.name

            for work_dir in composer_dir.iterdir():
                if not work_dir.is_dir():
                    continue
                work = work_dir.name

                # Check if this directory has ground truth (is a piece)
                has_gt = any(
                    (work_dir / name).exists()
                    for name in self.GT_MIDI_NAMES + self.GT_XML_NAMES
                )

                if has_gt:
                    # This is a piece directory
                    piece_id = f"{composer}_{work}"
                    midi_path = ""
                    xml_path = ""

                    for name in self.GT_MIDI_NAMES:
                        if (work_dir / name).exists():
                            midi_path = str(work_dir / name)
                            break

                    for name in self.GT_XML_NAMES:
                        if (work_dir / name).exists():
                            xml_path = str(work_dir / name)
                            break

                    pieces.append(
                        PieceInfo(
                            piece_id=piece_id,
                            composer=composer,
                            work=work,
                            midi_score_path=midi_path,
                            xml_score_path=xml_path,
                        )
                    )
                else:
                    # Check subdirectories for pieces
                    for piece_dir in work_dir.iterdir():
                        if not piece_dir.is_dir():
                            continue

                        has_piece_gt = any(
                            (piece_dir / name).exists()
                            for name in self.GT_MIDI_NAMES + self.GT_XML_NAMES
                        )

                        if has_piece_gt:
                            piece = piece_dir.name
                            piece_id = f"{composer}_{work}_{piece}"
                            midi_path = ""
                            xml_path = ""

                            for name in self.GT_MIDI_NAMES:
                                if (piece_dir / name).exists():
                                    midi_path = str(piece_dir / name)
                                    break

                            for name in self.GT_XML_NAMES:
                                if (piece_dir / name).exists():
                                    xml_path = str(piece_dir / name)
                                    break

                            pieces.append(
                                PieceInfo(
                                    piece_id=piece_id,
                                    composer=composer,
                                    work=work,
                                    piece=piece,
                                    midi_score_path=midi_path,
                                    xml_score_path=xml_path,
                                )
                            )

        logger.info(f"Found {len(pieces)} pieces in ASAP dataset")
        return pieces


# =============================================================================
# CHUNK EXTRACTION UTILITIES
# =============================================================================


def extract_measures_midi(
    midi_path: str,
    start_measure: int,
    end_measure: int,
    output_path: str,
) -> bool:
    """
    Extract a range of measures from a MIDI file.

    Note: This requires music21 for measure-based extraction.

    Args:
        midi_path: Path to input MIDI file
        start_measure: Starting measure (1-indexed)
        end_measure: Ending measure (inclusive)
        output_path: Path to output MIDI file

    Returns:
        True if successful, False otherwise
    """
    try:
        from music21 import converter

        score = converter.parse(midi_path)
        extracted = score.measures(start_measure, end_measure)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        extracted.write("midi", fp=output_path)
        return True

    except ImportError:
        logger.error("music21 required for measure extraction: pip install music21")
        return False

    except Exception as e:
        logger.error(f"Measure extraction failed: {e}")
        return False


def extract_measures_musicxml(
    musicxml_path: str,
    start_measure: int,
    end_measure: int,
    output_path: str,
) -> bool:
    """
    Extract a range of measures from a MusicXML file.

    Args:
        musicxml_path: Path to input MusicXML file
        start_measure: Starting measure (1-indexed)
        end_measure: Ending measure (inclusive)
        output_path: Path to output MusicXML file

    Returns:
        True if successful, False otherwise
    """
    try:
        from music21 import converter

        score = converter.parse(musicxml_path)
        extracted = score.measures(start_measure, end_measure)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        extracted.write("musicxml", fp=output_path)
        return True

    except ImportError:
        logger.error("music21 required for measure extraction: pip install music21")
        return False

    except Exception as e:
        logger.error(f"Measure extraction failed: {e}")
        return False


def extract_measures_to_midi(
    musicxml_path: str,
    start_measure: int,
    end_measure: int,
    output_midi_path: str,
) -> Optional[str]:
    """
    Extract a range of measures from a MusicXML file and output as MIDI.

    This is the primary function for chunk evaluation. It extracts specific
    measure ranges from MusicXML files (which have proper measure structure)
    and exports them as MIDI for MV2H evaluation.

    Args:
        musicxml_path: Path to input MusicXML file
        start_measure: Starting measure (1-indexed)
        end_measure: Ending measure (inclusive)
        output_midi_path: Path to output MIDI file

    Returns:
        Path to output MIDI file if successful, None otherwise

    Example:
        # Extract measures 10-14 (5-bar chunk) from a MusicXML file
        midi_path = extract_measures_to_midi(
            "score.musicxml",
            start_measure=10,
            end_measure=14,
            output_midi_path="/tmp/chunk_10.mid"
        )
    """
    # Skip if already extracted (cache)
    if Path(output_midi_path).exists():
        return output_midi_path

    try:
        from music21 import converter

        score = converter.parse(musicxml_path)
        extracted = score.measures(start_measure, end_measure)

        Path(output_midi_path).parent.mkdir(parents=True, exist_ok=True)
        extracted.write("midi", fp=output_midi_path)

        if Path(output_midi_path).exists():
            return output_midi_path
        return None

    except ImportError:
        logger.error("music21 required for measure extraction: pip install music21")
        return None

    except Exception as e:
        logger.error(f"Measure extraction to MIDI failed: {e}")
        return None


def extract_chunks_batch(
    musicxml_path: str,
    chunks: List[Tuple[int, int, str]],
) -> Dict[str, Optional[str]]:
    """
    Extract multiple measure ranges from a MusicXML file (parse once).

    This is much faster than calling extract_measures_to_midi() repeatedly,
    as it only parses the MusicXML file once.

    Args:
        musicxml_path: Path to input MusicXML file
        chunks: List of (start_measure, end_measure, output_path) tuples

    Returns:
        Dictionary mapping output_path to actual path (or None if failed)

    Example:
        results = extract_chunks_batch(
            "score.musicxml",
            [
                (10, 14, "/tmp/chunk_10.mid"),
                (11, 15, "/tmp/chunk_11.mid"),
                (12, 16, "/tmp/chunk_12.mid"),
            ]
        )
    """
    results = {}

    # Check which chunks need extraction (skip cached)
    to_extract = []
    for start_m, end_m, output_path in chunks:
        if Path(output_path).exists():
            results[output_path] = output_path
        else:
            to_extract.append((start_m, end_m, output_path))

    if not to_extract:
        return results

    try:
        from music21 import converter

        # Parse once
        score = converter.parse(musicxml_path)

        # Ensure output directory exists
        if to_extract:
            Path(to_extract[0][2]).parent.mkdir(parents=True, exist_ok=True)

        # Extract all chunks
        for start_m, end_m, output_path in to_extract:
            try:
                extracted = score.measures(start_m, end_m)
                extracted.write("midi", fp=output_path)

                if Path(output_path).exists():
                    results[output_path] = output_path
                else:
                    results[output_path] = None
            except Exception as e:
                logger.debug(f"Failed to extract measures {start_m}-{end_m}: {e}")
                results[output_path] = None

        return results

    except ImportError:
        logger.error("music21 required for measure extraction: pip install music21")
        return {output_path: None for _, _, output_path in chunks}

    except Exception as e:
        logger.error(f"Batch measure extraction failed: {e}")
        return {output_path: None for _, _, output_path in chunks}
