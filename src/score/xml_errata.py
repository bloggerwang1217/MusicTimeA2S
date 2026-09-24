"""Apply exact source-MusicXML corrections declared in xml_overrides.csv."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_OVERRIDES_PATH = (Path(__file__).resolve().parent.parent
                   / "datasets" / "asap" / "xml_overrides.csv")
_OVERRIDES: Optional[Dict[str, List[dict]]] = None


def _load() -> Dict[str, List[dict]]:
    global _OVERRIDES
    if _OVERRIDES is None:
        _OVERRIDES = {}
        with open(_OVERRIDES_PATH, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                _OVERRIDES.setdefault(row["file"], []).append(row)
    return _OVERRIDES


def _namespace(root: ET.Element) -> str:
    if root.tag.startswith("{"):
        return root.tag.split("}")[0] + "}"
    return ""


def _pitch(note: ET.Element, namespace: str) -> str:
    pitch = note.find(f"{namespace}pitch")
    if pitch is None:
        return "rest"
    step = pitch.findtext(f"{namespace}step", default="")
    octave = pitch.findtext(f"{namespace}octave", default="")
    alter = pitch.findtext(f"{namespace}alter")
    accidental = {"-1": "b", "1": "#"}.get(alter, "")
    return f"{step}{accidental}{octave}"


def _target(
    measure: ET.Element,
    voice: str,
    staff: str,
    namespace: str,
) -> Tuple[List[ET.Element], ET.Element]:
    children = list(measure)
    notes = [
        element for element in children
        if element.tag == f"{namespace}note"
        and element.find(f"{namespace}chord") is None
        and element.findtext(f"{namespace}voice") == voice
        and element.findtext(f"{namespace}staff") == staff
    ]
    if not notes:
        raise ValueError(
            f"xml_overrides measure {measure.get('number')}: "
            f"voice {voice} staff {staff} has no notes")
    last = children.index(notes[-1])
    following = next(
        (element for element in children[last + 1:]
         if element.tag in {
             f"{namespace}backup",
             f"{namespace}forward",
             f"{namespace}note",
         }),
        None,
    )
    if following is None or following.tag != f"{namespace}backup":
        raise ValueError(
            f"xml_overrides measure {measure.get('number')}: "
            "target voice is not followed by backup")
    return notes, following


def _signature(
    notes: List[ET.Element],
    backup: ET.Element,
    namespace: str,
) -> str:
    pitches = []
    durations = []
    ratios = []
    tuplets = []
    for note in notes:
        pitches.append(_pitch(note, namespace))
        durations.append(note.findtext(f"{namespace}duration", default=""))
        modification = note.find(f"{namespace}time-modification")
        if modification is None:
            ratios.append("-")
        else:
            actual = modification.findtext(
                f"{namespace}actual-notes", default="")
            normal = modification.findtext(
                f"{namespace}normal-notes", default="")
            ratios.append(f"{actual}:{normal}")
        notations = note.find(f"{namespace}notations")
        marks = [] if notations is None else [
            f"{tuplet.get('number', '1')}:{tuplet.get('type', '')}"
            for tuplet in notations.findall(f"{namespace}tuplet")
        ]
        tuplets.append("+".join(marks) or "-")
    backup_duration = backup.findtext(f"{namespace}duration", default="")
    return "|".join((
        f"pitches={','.join(pitches)}",
        f"durations={','.join(durations)}",
        f"ratios={','.join(ratios)}",
        f"tuplets={','.join(tuplets)}",
        f"backup={backup_duration}",
    ))


def _fields(signature: str) -> Dict[str, str]:
    return dict(field.split("=", 1) for field in signature.split("|"))


def _insert_time_modification(
    note: ET.Element,
    ratio: str,
    namespace: str,
) -> None:
    actual, normal = ratio.split(":", 1)
    modification = ET.Element(f"{namespace}time-modification")
    ET.SubElement(modification, f"{namespace}actual-notes").text = actual
    ET.SubElement(modification, f"{namespace}normal-notes").text = normal
    children = list(note)
    anchors = [
        index for index, child in enumerate(children)
        if child.tag in {f"{namespace}type", f"{namespace}dot"}
    ]
    note.insert(max(anchors) + 1, modification)


def _apply_signature(
    notes: List[ET.Element],
    backup: ET.Element,
    signature: str,
    namespace: str,
) -> None:
    fields = _fields(signature)
    durations = fields["durations"].split(",")
    ratios = fields["ratios"].split(",")
    tuplets = fields["tuplets"].split(",")
    if not len(notes) == len(durations) == len(ratios) == len(tuplets):
        raise ValueError("xml_overrides replacement has inconsistent note counts")

    for note, duration, ratio, tuplet_text in zip(
        notes, durations, ratios, tuplets
    ):
        note.find(f"{namespace}duration").text = duration
        modification = note.find(f"{namespace}time-modification")
        if modification is not None:
            note.remove(modification)
        if ratio != "-":
            _insert_time_modification(note, ratio, namespace)

        notations = note.find(f"{namespace}notations")
        if notations is not None:
            for tuplet in list(notations.findall(f"{namespace}tuplet")):
                notations.remove(tuplet)
        if tuplet_text != "-":
            if notations is None:
                notations = ET.Element(f"{namespace}notations")
                note.append(notations)
            for mark in tuplet_text.split("+"):
                number, kind = mark.split(":", 1)
                attributes = {"number": number, "type": kind}
                if kind == "start":
                    attributes.update({"bracket": "no", "show-number": "none"})
                notations.insert(
                    0, ET.Element(f"{namespace}tuplet", attributes))
        if notations is not None and len(notations) == 0:
            note.remove(notations)

    backup.find(f"{namespace}duration").text = fields["backup"]


def apply_xml_overrides(xml_path: str | Path, stem: str) -> List[dict]:
    """Apply every exact XML replacement declared for one corpus file."""
    rows = _load().get(stem, [])
    if not rows:
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespace = _namespace(root)
    measures = list(root.iter(f"{namespace}measure"))
    for row in rows:
        if row["method"] != "export_serialization":
            raise ValueError(
                f"xml_overrides {stem}: unknown method {row['method']!r}")
        hits = [measure for measure in measures
                if measure.get("number") == row["measure"]]
        if len(hits) != 1:
            raise ValueError(
                f"xml_overrides {stem}: measure {row['measure']} occurs "
                f"{len(hits)} times, expected exactly 1")
        notes, backup = _target(
            hits[0], row["voice"], row["staff"], namespace)
        current = _signature(notes, backup, namespace)
        if current != row["old"]:
            raise ValueError(
                f"xml_overrides {stem} measure {row['measure']}: "
                "source no longer matches the declared old value")
        _apply_signature(notes, backup, row["new"], namespace)
        if _signature(notes, backup, namespace) != row["new"]:
            raise ValueError(
                f"xml_overrides {stem} measure {row['measure']}: "
                "replacement did not produce the declared new value")

    tree.write(xml_path, xml_declaration=True, encoding="unicode")
    return rows
