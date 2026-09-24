"""Corrections for notation errors in the published source scores.

A row names the erroneous cell and the corrected one.  Applying it
re-checks that the erroneous text is still there and still unambiguous,
so a corpus that changes underneath fails loudly rather than being
silently mis-edited.  Only whole-cell replacement is supported: an
entry that would add or remove events is a score decision, not an
erratum, and has to be argued separately.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sits with the other per-file decision tables.  Named "overrides" so the
# table reads without music-editing vocabulary; rows are still errata.
_ERRATA_PATH = (Path(__file__).resolve().parent.parent
                / "datasets" / "syn" / "kern_overrides.csv")

_ERRATA: Optional[Dict[str, List[dict]]] = None


def _load() -> Dict[str, List[dict]]:
    global _ERRATA
    if _ERRATA is None:
        _ERRATA = {}
        if _ERRATA_PATH.exists():
            with open(_ERRATA_PATH, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    _ERRATA.setdefault(row["file"], []).append(row)
    return _ERRATA


def errata_files() -> List[str]:
    """Stems the table claims to correct, for the caller's coverage check."""
    return sorted(_load())


def _bar_span(lines: List[str], label: str, stem: str) -> Tuple[int, int]:
    hits = [i for i, line in enumerate(lines)
            if line.startswith("=") and line.split("\t")[0] == label]
    if len(hits) != 1:
        raise ValueError(
            f"kern_errata {stem}: barline {label!r} occurs {len(hits)} "
            f"times in the source, expected exactly 1")
    start = hits[0]
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("=")), len(lines))
    return start, end


def apply_kern_errata(kern_text: str, stem: str) -> str:
    """Correct this file's known notation errors, on the source as published.

    Every row for the stem must land: a row that cannot be placed raises,
    so there is no path where a listed correction quietly does nothing.
    """
    rows = _load().get(stem)
    if not rows:
        return kern_text

    lines = kern_text.split("\n")
    for row in rows:
        label, old, new = row["measure"], row["old"], row["new"]

        if row["spine"] == "line":
            # Whole-line replacement, for a correction that must land on
            # a cell whose text alone is ambiguous within the bar (a
            # null token): the full data line is the unambiguous address.
            start, end = _bar_span(lines, label, stem)
            hits = [i for i in range(start + 1, end)
                    if not lines[i].startswith(("!", "*"))
                    and lines[i] == old]
            if len(hits) != 1:
                raise ValueError(
                    f"kern_errata {stem} {label} line: the line occurs "
                    f"{len(hits)} times in the bar, expected exactly 1")
            if len(new.split("\t")) != len(old.split("\t")):
                raise ValueError(
                    f"kern_errata {stem} {label} line: replacement has "
                    f"a different number of cells")
            lines[hits[0]] = new
            logger.info(
                f"kern_errata {stem} {label} line: {old!r} -> {new!r}")
            continue

        if "\t" in new or "\n" in new:
            raise ValueError(
                f"kern_errata {stem} {label}: 'new' spans more than one "
                f"cell; only cell replacement is supported")

        spine = int(row["spine"]) - 1
        if label == "*":
            # A barline or a playback order belongs to no bar; those are
            # reached file-wide, and still only where the text is unique.
            start, end, skip_interps = -1, len(lines), False
        else:
            start, end = _bar_span(lines, label, stem)
            skip_interps = True
            n_spines = len(lines[start].split("\t"))
            if not 0 <= spine < n_spines:
                raise ValueError(
                    f"kern_errata {stem} {label}: spine {row['spine']} is "
                    f"out of range, the bar has {n_spines}")

        hits = []
        for i in range(start + 1, end):
            # Within a bar, comments and interpretations carry no events.
            if skip_interps and lines[i].startswith(("!", "*")):
                continue
            cells = lines[i].split("\t")
            if spine < len(cells) and cells[spine] == old:
                hits.append(i)
        if len(hits) != 1:
            where = "the file" if label == "*" else "the bar"
            raise ValueError(
                f"kern_errata {stem} {label} spine {row['spine']}: {old!r} "
                f"occurs {len(hits)} times in {where}, expected exactly 1")

        cells = lines[hits[0]].split("\t")
        cells[spine] = new
        lines[hits[0]] = "\t".join(cells)
        logger.info(
            f"kern_errata {stem} {label} spine {row['spine']}: "
            f"{old} -> {new}")

    return "\n".join(lines)
