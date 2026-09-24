"""Shared MusicXML → cleaned-kern chain.

The corpus processors (MuseSyn, ASAP) own file discovery, naming, and
corpus-specific policy (meter-rewrite allowlist, rendering export); the
score-level chain itself lives here so every xml-sourced corpus goes
through the identical Phase 1.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import converter21
import music21 as m21

converter21.register()

from src.score.sanitize_piano_score import (
    normalize_shared_score,
    prepare_humdrum_export,
    preprocess_cue_notes,
    repair_out_of_bar_backups,
    clear_overfull_measures,
    fill_empty_measures,
    infer_missing_meter_score,
    truncate_fermata_holds,
    strip_grace_notes_from_score,
    strip_trills_from_score,
    strip_non_trill_ornaments_from_score,
    strip_arpeggios_from_score,
    preserve_opening_anacrusis,
)
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
from src.score.sanitize_kern import extract_kern_measures
from src.score.expand_repeat import (
    expand_asap_score,
    expand_musesyn_score,
    renumber_kern_barlines,
    renumber_score_measures,
    extract_repeat_structure,
    build_musesyn_repeat_map,
)

logger = logging.getLogger(__name__)



def musicxml_grid(score) -> int:
    """The tick size this score needs so nothing has to be rounded.

    MusicXML measures every duration in whole ticks of a grid declared
    once per part, and the conventional grids are built on powers of two
    with a few small primes.  A note lasting five forty-sixths of a beat
    has nowhere to land on such a grid, so its bar comes back a few ticks
    long; a grid that is a multiple of forty-six holds it exactly.
    """
    from fractions import Fraction
    from math import lcm

    def _frac(value) -> Fraction:
        return Fraction(value).limit_denominator(10 ** 7)

    denominators = {1}
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            for holder in (list(measure.voices) or [measure]):
                for element in holder.notesAndRests:
                    denominators.add(
                        _frac(element.duration.quarterLength).denominator)
                    denominators.add(
                        _frac(holder.elementOffset(element)).denominator)
    needed = lcm(*sorted(denominators))
    usual = m21.defaults.divisionsPerQuarter
    if usual % needed == 0:
        return usual
    # 480 stays in as a floor so the ordinary durations still come out as
    # the round numbers every editor writes.
    return lcm(needed, 480)


def number_cleared_bars(score, kern_content: str, cleared: List[Dict]) -> None:
    """Number the cleared bars the way the kern counts its own.

    A measure holding nothing at all is not written as a bar, so the bar
    count can fall short of the measure count and every such measure
    before a cleared one shifts its position.  The arithmetic is checked
    against the bars the kern actually carries and the ordinal is left
    unset when the two disagree, because quarantining the wrong bar is
    worse than not placing this one.
    """
    if not cleared:
        return
    measures = [list(part.getElementsByClass(m21.stream.Measure))
                for part in score.parts]
    if not measures:
        return
    height = max(len(part) for part in measures)
    silent = set()
    for index in range(height):
        here = [part[index] for part in measures if index < len(part)]
        if here and not any(list(bar.recurse().notesAndRests) for bar in here):
            silent.add(index)

    written = sum(
        1 for line in kern_content.splitlines()
        if line.startswith("=") and not line.startswith("=="))
    if height - len(silent) != written:
        logger.warning(
            f"cleared-bar numbering skipped: {height} measures less "
            f"{len(silent)} empty does not match {written} written bars")
        return

    for entry in cleared:
        index = entry.get("index")
        if index is None:
            continue
        entry["ordinal"] = index - sum(1 for k in silent if k < index)


def phase1_bar_issues(cleared_bars) -> list:
    """Turn cleared bars into the per-bar record phase 1.5 merges in.

    The bar label is the printed number the kern carries, which is what
    locates the bar again once the file has been through the writer.
    """
    return [
        {
            "family": "phase1-metric-timeline",
            "message": (
                f"={c['measure']}: bar holds no notation "
                f"and holds the meter's silence"
                if c.get("empty") else
                f"={c['measure']}: bar outgrew {c['meter']} "
                f"({c['span_qn']:g}qn against {c['meter_qn']:g}qn) "
                f"and holds the meter's silence"),
            "bar_label": str(c["measure"]),
            **({"bar_ordinal": c["ordinal"], "bar_index": c["ordinal"]}
               if c.get("ordinal") is not None else {}),
        }
        for c in cleared_bars
    ]

def standardize_xml(
    xml_path: Path,
    *,
    keep_dynam: bool = True,
    keep_grace: bool = False,
    keep_trill: bool = False,
    keep_non_trill_ornaments: bool = False,
    keep_arpeggio: bool = False,
    fermata_hold_cut: bool = True,
    xml_out: Optional[Path] = None,
    tempo_policy: Optional[Callable[[m21.stream.Score], None]] = None,
    measure_order: Optional[List[int]] = None,
    xml_override_stem: Optional[str] = None,
) -> Tuple[str, List[List[Dict[str, Any]]], Dict, List[Dict]]:
    """Convert one MusicXML score to cleaned, repeat-expanded kern.

    Args:
        xml_path: Path to the MusicXML file.
        keep_dynam: Retain **dynam spines (dynamics render into audio;
            the kern_gt stage strips them before tokenizing).
        keep_grace / keep_trill / keep_non_trill_ornaments / keep_arpeggio:
            retain those signifiers (default: stripped).
        fermata_hold_cut: Cut a fermata bar's written-out hold at the
            barline.  Off for corpora that notate free passages inside a
            fermata bar, where the cut would delete the passage.
        xml_out: If set, also export a canonical rendering MusicXML to
            this path (best-effort, on a deep copy).
        tempo_policy: Callable applied to the export copy when xml_out is
            set, so the rendered audio gets the curated tempo.
        measure_order: Playback order as measure numbers, from a corpus that
            carries its own alignment.  Given, it replaces reading the
            repeat signs; withheld, the score's own signs are followed.
        xml_override_stem: Corpus file name used to address exact source-XML
            replacements before parsing.

    Returns:
        Tuple of (cleaned kern content, visual info, repeat_map,
        cleared bars).  A cleared bar outgrew its meter and holds
        the meter's silence instead; the caller records it so the
        bar is quarantined rather than trusted.
        Raises on failure; corpus processors decide how to record it.
        Visual info is extracted from converter21 output BEFORE cleaning.
    """
    # 1. Pre-process cue notes, then parse MusicXML
    tmp_xml = preprocess_cue_notes(str(xml_path), promote=True)

    # 1a. A cursor excursion driven outside the bar cannot be read back:
    # no parser holds a position before the barline, so the wind-back is
    # pinned there while the wind-forward is spent in full and the rest
    # of the bar lands hundreds of bars late.
    for r in repair_out_of_bar_backups(tmp_xml):
        logger.warning(
            f"{xml_path.stem}: measure {r['measure']} part {r['part']} "
            f"cursor wind-back {r['from_qn']:g}→{r['to_qn']:g}qn, "
            f"wind-forward {r['forward_qn']:g}→0")

    if xml_override_stem is not None:
        from src.score.xml_errata import apply_xml_overrides
        for row in apply_xml_overrides(tmp_xml, xml_override_stem):
            logger.warning(
                f"{xml_override_stem}: corrected source MusicXML measure "
                f"{row['measure']} voice {row['voice']} staff {row['staff']}")

    score = m21.converter.parse(tmp_xml)
    Path(tmp_xml).unlink(missing_ok=True)

    # Normalize musical content before the XML and Humdrum branches split.
    normalize_shared_score(score)

    # 3. Extract repeat structure BEFORE expansion (barlines,
    #    DaCapo/Segno/Fine, volta brackets).
    repeat_structure = extract_repeat_structure(score)

    # 4. Expand repeats at the music21 level if present.
    # converter21 output has repeat barlines (:|! / !|:) but no
    # Humdrum expansion labels (*>[A,A,B,...]), so kern-level
    # expansion via expand_kern_repeats cannot work.
    # Instead, expand in music21 before converting to kern.
    original_measures = len(
        repeat_structure["orig_measure_numbers"]
    )
    if measure_order is not None:
        score_for_kern, has_repeats = expand_asap_score(score, measure_order)
    else:
        score_for_kern, has_repeats = expand_musesyn_score(score)

    # 4b. Cut content overflowing the declared meter (fermata hold
    # written into durations).  The audio chain applies the same
    # cut, so kern bars and rendered measures stay 1:1.
    #
    # The fermata gate is per measure position across all parts, so a bar
    # that opens on a fermata and then carries a free written-out passage
    # loses that passage: the cut cannot tell the hold from the music.
    # Callers whose corpus notates free passages that way turn this off
    # and let the overfull bar reach the ledger instead.
    if fermata_hold_cut:
        for t in truncate_fermata_holds(score_for_kern):
            logger.warning(
                f"{xml_path.stem}: overfull measure {t['measure']} "
                f"({t['meter']}, span {t['span_qn']}qn) — {t['action']} "
                f"{t['what']} {t['from_ql']}→{t['to_ql']}qn "
                f"at beat {t['offset_qn'] + 1:g} (part {t['part']})")

    # A source with no meter at all leaves both exports below
    # without one, and the beat grid is read off the meter.
    declared = infer_missing_meter_score(score_for_kern)
    if declared:
        logger.warning(
            f"{xml_path.stem}: no meter in source, declared {declared}")

    opening_anacrusis = preserve_opening_anacrusis(score_for_kern)
    renumber_score_measures(
        score_for_kern,
        opening_anacrusis=opening_anacrusis,
    )

    # 4e. A bar longer than its meter has no beat grid to sit on.  It is
    # handed the meter's silence and reported, so the barline count stays
    # true and only that bar is withheld.  Last of the repairs, so every
    # station above has had its chance to make the bar fit honestly.
    cleared_bars = clear_overfull_measures(score_for_kern)
    for c in cleared_bars:
        logger.warning(
            f"{xml_path.stem}: measure {c['measure']} outgrew {c['meter']} "
            f"({c['span_qn']:g}qn against {c['meter_qn']:g}qn) — "
            f"cleared to silence")

    # 4f. A bar left with nothing written in it survives in the MusicXML
    # export and disappears from the kern, taking the numbering of every
    # bar after it with it.
    for c in fill_empty_measures(score_for_kern):
        logger.warning(
            f"{xml_path.stem}: measure {c['measure']} holds no notation "
            f"({c['meter']}) — given the meter's silence")
        cleared_bars.append(c)

    # 4d. Export canonical rendering MusicXML (best-effort, on a
    # deep copy — score_for_kern below stays untouched so the
    # kern/ output this function primarily produces cannot be
    # affected by anything in this branch).  Runs on the
    # in-memory object directly: no kern round-trip, so grace/
    # ornament/arpeggio removal here is object-level, not the
    # kern-text strip_* used for kern/ below.
    if xml_out is not None:
        try:
            score_for_xml = copy.deepcopy(score_for_kern)
            if not keep_grace:
                strip_grace_notes_from_score(score_for_xml)
            if not keep_trill:
                strip_trills_from_score(score_for_xml)
            if not keep_non_trill_ornaments:
                strip_non_trill_ornaments_from_score(score_for_xml)
            if not keep_arpeggio:
                strip_arpeggios_from_score(score_for_xml)
            if tempo_policy is not None:
                tempo_policy(score_for_xml)
            # music21 reads the grid off its own defaults at write
            # time, so it is set around the write and put back after.
            grid = musicxml_grid(score_for_xml)
            usual_grid = m21.defaults.divisionsPerQuarter
            if grid != usual_grid:
                logger.warning(
                    f"{xml_path.stem}: writing MusicXML on a {grid}-tick "
                    f"beat so its tuplets land on whole ticks")
            m21.defaults.divisionsPerQuarter = grid
            try:
                score_for_xml.write("musicxml", fp=str(xml_out))
            except m21.musicxml.xmlObjects.MusicXMLExportException:
                # Pathological tuplet (e.g. a cross-barline cadenza
                # fragment whose ratio isn't a power of 2) can make
                # music21's tuplet-bracket consolidation compute an
                # unexportable "normal type". makeNotation=False
                # skips that consolidation — the file still
                # exports, just without a few display-only hidden
                # filler rests.
                logger.warning(
                    f"{xml_path.stem}: retrying MusicXML export "
                    f"with makeNotation=False (tuplet bracket limitation)")
                score_for_xml.write(
                    "musicxml", fp=str(xml_out), makeNotation=False)
            finally:
                m21.defaults.divisionsPerQuarter = usual_grid
        except Exception as e:
            logger.warning(
                f"{xml_path.stem}: MusicXML export failed: {e}")

    # Prepare only the Humdrum branch for converter21.
    prepare_humdrum_export(score_for_kern)

    # 5. Export to Humdrum kern via converter21
    kern_raw = score_for_kern.write("humdrum", makeNotation=False)

    # Read the written file content
    with open(kern_raw, "r", encoding="utf-8") as f:
        kern_content = f.read()

    # Clean up temp file
    Path(kern_raw).unlink(missing_ok=True)

    # 6. Strip non-kern spines
    kern_content = strip_spines(kern_content, keep_dynam=keep_dynam)

    # 6b. Figure-level strips (default off): these signifiers change
    # the sounding figure, so kern/ and kern_gt/ exclude them unless
    # the matching keep flag is set.  Must run before visual
    # extraction — strip_grace_notes drops lines, which would
    # desync visual info.
    if not keep_grace:
        kern_content = strip_grace_notes(kern_content)
        kern_content = strip_grace_designators(kern_content)
    if not keep_trill:
        kern_content = strip_trill_marks(kern_content)
    if not keep_non_trill_ornaments:
        kern_content = strip_non_trill_ornament_marks(kern_content)
    if not keep_arpeggio:
        kern_content = strip_arpeggio_marks(kern_content)

    # 7. Extract visual info BEFORE cleaning (preserves stem/beam/position)
    visual_info = extract_visual_from_sequence(kern_content)

    # 8. Clean kern sequence (remove visual tokens)
    kern_cleaned = clean_kern_sequence(kern_content, warn_tuplet_ratio=False)

    # converter21 may omit or reuse bar numbers.  Number the final
    # linear artifact from its actual playback order.
    kern_cleaned = renumber_kern_barlines(
        kern_cleaned,
        opening_anacrusis=opening_anacrusis,
    )

    # 9. Build repeat_map with rich mapping
    measures_gt = extract_kern_measures(kern_content=kern_cleaned)

    repeat_map = build_musesyn_repeat_map(
        repeat_structure=repeat_structure,
        expanded_score=score_for_kern,
        measures_gt=measures_gt,
        has_repeats=has_repeats,
        original_measure_count=original_measures,
    )

    number_cleared_bars(score_for_kern, kern_cleaned, cleared_bars)
    return kern_cleaned, visual_info, repeat_map, cleared_bars
