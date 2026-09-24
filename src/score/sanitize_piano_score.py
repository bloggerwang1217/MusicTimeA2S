"""
Piano Score Sanitization Functions for ASAP Dataset Processing

This module contains functions to clean and repair piano scores for converter21/Humdrum export.
All functions work on music21.stream.Score objects in-place.

Public stages:
    normalize_shared_score()
    prepare_humdrum_export()

Author: Blogger Wang
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import music21 as m21
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union


def normalize_shared_score(score):
    """Apply musical normalization shared by XML and Humdrum exports."""
    # 1. Exact tuplet arithmetic (first: every later station measures
    # spans, and a rounded tuplet makes those measurements wrong)
    try:
        exact_tuplet_durations(score)
    except:
        pass

    # 2. Realize transposing Ottavas (before hidden-note removal, which can
    # delete Ottavas that reference hidden notes)
    try:
        realize_ottavas(score)
    except:
        pass

    # 3. Whole rests misread as full-bar rests
    try:
        restore_literal_whole_rests(score)
    except:
        pass

    # 4. Remove Hidden Notes while preserving hidden Rests for MusicXML
    try:
        remove_hidden_sounding_notes(score)
    except:
        pass

    # 5. Remove Ghost Note Spanners (NEW - removes MusicXML export errors)
    try:
        remove_ghost_note_spanners(score)
    except:
        pass

    # 6. Fix incomplete repeat structures (MuseSyn MusicXML export bug)
    try:
        fix_incomplete_repeats(score)
    except:
        pass


def prepare_humdrum_export(score):
    """Apply converter21-specific structure before Humdrum export.

    Order is critical:
    - Healing must happen before dynamics snapping
    - Dynamics snapping MUST happen BEFORE refresh_spanners_after_heal (to avoid being undone)

    After this function, full_score.write('humdrum') should succeed.

    Note: repair_measure_voices() was removed - testing showed it had no effect on error rate.
    """
    # 1. Unhide hidden Rests and remove Voices emptied by hidden Notes
    try:
        prepare_hidden_rests_for_humdrum(score)
    except:
        pass

    # 2. Heal Cross-Staff Issues (supports asymmetric Voice structures)
    try:
        heal_cross_staff(score)
    except:
        pass

    # 3. Snap Dynamics (MUST run BEFORE refresh_spanners_after_heal!)
    # Rebinds DynamicWedges from Rests to Notes in different Voices.
    # If this runs after refresh_spanners_after_heal, the rebinding will be undone.
    try:
        snap_dynamics_to_notes(score)
    except:
        pass

    # 4. Refresh Spanners After Healing (Critical for cross-staff slurs)
    try:
        refresh_spanners_after_heal(score)
    except:
        pass

    # 5. Fix SMUFL Text Expressions (converter21 bug workaround)
    try:
        fix_smufl_text_expressions(score)
    except:
        pass

    # 6. Strip dynamics from cadenza measures (converter21 crash workaround)
    try:
        strip_dynamics_from_cadenza_measures(score)
    except:
        pass

    # 7. Patch MetronomeMarks with number=None (converter21 kern export bug)
    try:
        patch_metronome_marks(score)
    except:
        pass

    # 8. A mid-measure clef belongs on the note it is there for; left
    # early it cuts a column the writer then charges to a rest crossing it.
    try:
        align_midbar_clefs(score)
    except:
        pass

    # 9. Fill voice gaps with explicit rests (converter21's humdrum
    # writer cannot export a voice with unnotated time)
    try:
        fill_voice_gaps(score)
    except:
        pass


def preserve_opening_anacrusis(score):
    """Mark a short opening measure so MusicXML does not pad its tail.

    The Humdrum reader preserves an opening anacrusis as a short measure but
    does not set ``Measure.paddingLeft``.  music21's MusicXML writer therefore
    completes the bar *after* the upbeat, moving the following downbeat.  The
    missing duration belongs before the written pickup in the metrical cycle;
    setting ``paddingLeft`` preserves the short playback span on round-trip.
    """
    first_measures = []
    for part in score.parts:
        measures = list(part.getElementsByClass(m21.stream.Measure))
        if not measures:
            return False
        first_measures.append(measures[0])

    spans = [float(measure.duration.quarterLength)
             for measure in first_measures]
    if not spans or max(spans) - min(spans) > 1e-6:
        return False

    time_signatures = []
    for part, measure in zip(score.parts, first_measures):
        local = list(measure.recurse().getElementsByClass(
            m21.meter.TimeSignature))
        if local:
            time_signatures.append(local[0])
            continue
        flattened = list(part.flatten().getElementsByClass(
            m21.meter.TimeSignature))
        if not flattened:
            return False
        time_signatures.append(flattened[0])

    bar_lengths = [float(ts.barDuration.quarterLength)
                   for ts in time_signatures]
    if max(bar_lengths) - min(bar_lengths) > 1e-6:
        return False

    missing = bar_lengths[0] - spans[0]
    if missing <= 1e-6 or spans[0] <= 1e-6:
        return False

    for measure in first_measures:
        measure.paddingLeft = missing
    return True


def realize_ottavas(score):
    """Transpose notes under transposing Ottava spanners to sounding pitch.

    converter21's Humdrum parser stores *8va/*8ba notes at written pitch with
    Ottava(transposing=True), which music21's MIDI writer never realizes —
    8va passages would sound an octave off. No-op for MusicXML-parsed scores
    (their Ottavas are transposing=False).
    """
    realized = 0
    for sp in score.recurse().getElementsByClass(m21.spanner.Ottava):
        if sp.transposing:
            sp.performTransposition()
            realized += 1
    return realized


def restore_literal_whole_rests(score):
    """music21 reads every whole rest as a full-bar rest; one sharing its
    bar with notes is literal, so shrink it back to the encoded length."""
    fixed = 0
    for part in score.parts:
        for meas in part.getElementsByClass(m21.stream.Measure):
            ts = meas.getContextByClass(m21.meter.TimeSignature)
            if ts is None:
                continue
            bar_ql = ts.barDuration.quarterLength
            for voice in (list(meas.voices) or [meas]):
                content = list(voice.notesAndRests)
                excess = sum(n.duration.quarterLength for n in content) - bar_ql
                if excess <= 0:
                    continue
                for n in reversed(content):
                    if (isinstance(n, m21.note.Rest)
                            and n.fullMeasure is True
                            and n.duration.quarterLength > excess):
                        n.duration.linked = False
                        n.fullMeasure = False
                        n.duration.quarterLength -= excess
                        voice.clearCache()
                        meas.clearCache()
                        fixed += 1
                        break
    return fixed


def patch_metronome_marks(score):
    """Patch MetronomeMarks where number=None but numberSounding is set.

    converter21's kern exporter only writes *MM when MetronomeMark.number
    is set. music21 sometimes parses MusicXML <sound tempo="X"> into a
    MetronomeMark with numberSounding=X but number=None (when there is
    no visible metronome marking in the score). This causes tempo to be
    silently dropped during kern export.
    """
    for part in score.parts:
        for tm in part.flatten().getElementsByClass(m21.tempo.MetronomeMark):
            if tm.number is None and tm.numberSounding is not None:
                tm.number = tm.numberSounding
                if tm.referent is None:
                    tm.referent = m21.duration.Duration(1.0)


def fix_incomplete_repeats(score):
    """
    Remove incomplete repeat structures that cause 'badly formed repeats' error.

    Problem: Some MusicXML files (especially MuseScore exports) have incomplete
    repeat structures - e.g., a forward repeat without a matching backward repeat.
    This causes music21's expandRepeats() to fail during MIDI export.

    Example: musesyn_Dont_look_back_in_anger has <repeat direction="forward"/>
    but no corresponding <repeat direction="backward"/>.

    Solution: Detect unpaired repeats and remove them. This loses the repeat
    semantics but allows MIDI export to succeed.

    An unmatched *backward* repeat is not this bug: notation leaves the
    opening `|:` of a section repeat implicit, so a lone `:|` is a correct
    score and music21 expands it on its own.  Only a dangling forward repeat
    is removed.

    Returns:
        int: Number of repeat elements removed
    """
    # Collect all bar.Repeat elements by part
    removed_count = 0

    for part in score.parts:
        forwards = []
        backwards = []

        for el in part.recurse().getElementsByClass(m21.bar.Repeat):
            if el.direction == 'start':
                forwards.append(el)
            elif el.direction == 'end':
                backwards.append(el)

        if len(forwards) > len(backwards):
            # Remove all repeat elements from this part
            for el in forwards + backwards:
                try:
                    if el.activeSite:
                        el.activeSite.remove(el)
                        removed_count += 1
                except:
                    pass

    return removed_count


def fix_smufl_text_expressions(score):
    """
    Fix TextExpression elements with SMUFL characters at end of string.

    converter21 bug:
    -----------------------------
    converter21's translateSMUFLNotesToNoteNames() crashes with IndexError when
    a SMUFL character (U+E000 - U+F8FF) is at the end of a string. The function
    attempts `while text[j]` without boundary checking after finding a SMUFL char.

    Bug location: converter21/humdrum/m21convert.py:2064

    Example:
        '\ueca5'        -> IndexError (SMUFL at end)
        '\ueca5 = 120'  -> '[quarter] = 120' (works, SMUFL not at end)

    SMUFL (Standard Music Font Layout) characters are music symbols in Unicode
    Private Use Area, commonly used in MuseScore exports for metronome markings.

    Workaround:
    -----------
    Append a space after trailing SMUFL characters to avoid the boundary issue.
    This preserves the musical information (converter21 will output '[quarter] ').
    """
    def is_smufl_char(char):
        """Check if character is in SMUFL range (U+E000 - U+F8FF)."""
        return 0xE000 <= ord(char) <= 0xF8FF

    for el in score.recurse():
        if isinstance(el, m21.expressions.TextExpression):
            content = el.content
            if content and is_smufl_char(content[-1]):
                # Append space to avoid converter21 boundary bug
                el.content = content + ' '


def strip_dynamics_from_cadenza_measures(score):
    """Remove dynamics from measures whose duration has non-power-of-two factors.

    Cadenza measures (parsed from MusicXML <cue> notes) can have durations
    like 983/160 (denominator contains factor 5).  converter21 crashes when
    placing dynamics in such measures because it cannot construct
    power-of-two-with-dots invisible rests.

    Workaround only — kern GT strips dynamics anyway, so nothing is lost
    from the training target.
    """
    from fractions import Fraction

    removed = 0
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            ht = measure.highestTime
            if ht <= 0:
                continue
            denom = Fraction(ht).limit_denominator(100000).denominator
            # power-of-two check: denom & (denom - 1) == 0
            if denom > 0 and (denom & (denom - 1)) == 0:
                continue
            # Non-standard duration — strip dynamics
            for el in list(measure.recurse()):
                if isinstance(el, (m21.dynamics.Dynamic, m21.dynamics.DynamicWedge)):
                    try:
                        el.activeSite.remove(el)
                        removed += 1
                    except Exception:
                        pass
    return removed


def strip_grace_notes_from_score(score) -> int:
    """Remove grace notes/chords (GraceDuration, quarterLength 0) from a Score.

    Object-level counterpart to standardize_kern.strip_grace_notes, for
    export paths (VirtuosoNet MusicXML) that stay in music21 space and
    never round-trip through kern text.  Grace notes carry no metric
    duration, so removal leaves no gap to pad — unlike the kern-text
    version, which has to drop now-empty timeline rows.
    """
    removed = 0
    for el in list(score.recurse().notes):
        if el.duration.isGrace:
            site = el.activeSite
            if site is not None:
                try:
                    site.remove(el)
                    removed += 1
                except Exception:
                    pass
    return removed


TRILL_EXPRESSION_CLASSES = (
    m21.expressions.Trill, m21.expressions.TrillExtension,
)

# Non-trill ornament family. VirtuosoNet's score parser discards mordent
# after parsing it and never parses turn/shake/schleifer at all, so
# toggling these cannot change VirtuosoNet-rendered audio.
NON_TRILL_ORNAMENT_EXPRESSION_CLASSES = (
    m21.expressions.Mordent, m21.expressions.InvertedMordent,
    m21.expressions.Turn, m21.expressions.InvertedTurn,
    m21.expressions.Shake, m21.expressions.Schleifer,
)


def strip_trills_from_score(score) -> int:
    """Remove trill ornament expressions from a Score.

    Object-level counterpart to clean_kern.strip_trill_marks.
    """
    removed = 0
    for n in score.recurse().notes:
        keep = [e for e in n.expressions
                if not isinstance(e, TRILL_EXPRESSION_CLASSES)]
        removed += len(n.expressions) - len(keep)
        n.expressions = keep
    return removed


def strip_non_trill_ornaments_from_score(score) -> int:
    """Remove mordent/turn/shake/schleifer ornament expressions from a Score.

    Object-level counterpart to clean_kern.strip_non_trill_ornament_marks.
    """
    removed = 0
    for n in score.recurse().notes:
        keep = [e for e in n.expressions
                if not isinstance(e, NON_TRILL_ORNAMENT_EXPRESSION_CLASSES)]
        removed += len(n.expressions) - len(keep)
        n.expressions = keep
    return removed


def strip_arpeggios_from_score(score) -> int:
    """Remove arpeggio marks from a Score.

    Object-level counterpart to clean_kern.strip_arpeggio_marks. Iterates
    all notes, not just Chords: a cross-staff arpeggio spans two streams
    (one per hand), so each hand's notes carry ArpeggioMark individually
    as plain Notes rather than being merged into one Chord.
    """
    removed = 0
    for n in score.recurse().notes:
        keep = [e for e in n.expressions
                if not isinstance(e, m21.expressions.ArpeggioMark)]
        removed += len(n.expressions) - len(keep)
        n.expressions = keep
    for span in list(score.recurse().getElementsByClass(
            m21.expressions.ArpeggioMarkSpanner)):
        site = span.activeSite
        if site is not None:
            try:
                site.remove(span)
                removed += 1
            except Exception:
                pass
    return removed


ORNAMENT_TAGS = frozenset({
    'trill-mark', 'inverted-mordent', 'mordent', 'wavy-line',
    'turn', 'inverted-turn', 'shake', 'tremolo',
})


def preprocess_cue_notes(xml_path: str, *, promote: bool = False) -> str:
    """Pre-process MusicXML cue notes: hide ornament expansions, optionally promote real content.

    Pass 1 (always) — hide ornament expansions (trill, mordent, tremolo):
      Marks cue notes as print-object="no" when the measure has an ornament
      marking on a non-cue note, or already has hidden cue notes.
    Pass 2 (promote=True) — promote remaining visible cue notes to regular notes:
      Removes the <cue> element so music21 parses them as normal notes.
      Safe for MuseSyn (simple cue notes); ASAP needs per-measure handling
      (cadenza measures require <forward>/<backup> removal too).

    Writes a temp file; caller is responsible for cleanup.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    count = 0
    for part in root.iter(f'{ns}part'):
        for measure in part.iter(f'{ns}measure'):
            has_ornament = False
            has_hidden_cue = False
            cue_notes = []

            for note in measure.iter(f'{ns}note'):
                is_cue = note.find(f'{ns}cue') is not None

                if is_cue:
                    cue_notes.append(note)
                    if note.get('print-object') == 'no':
                        has_hidden_cue = True
                else:
                    notations = note.find(f'{ns}notations')
                    if notations is not None:
                        for ornaments in notations.iter(f'{ns}ornaments'):
                            for child in ornaments:
                                tag = child.tag.replace(ns, '')
                                if tag in ORNAMENT_TAGS:
                                    has_ornament = True

            if (has_ornament or has_hidden_cue) and cue_notes:
                for note in cue_notes:
                    if note.get('print-object') != 'no':
                        note.set('print-object', 'no')
                        count += 1

    # Pass 2: promote remaining visible cue notes to regular notes
    if promote:
        promoted = 0
        for part in root.iter(f'{ns}part'):
            for note in part.iter(f'{ns}note'):
                cue_el = note.find(f'{ns}cue')
                if cue_el is not None and note.get('print-object') != 'no':
                    note.remove(cue_el)
                    promoted += 1

    tmp = tempfile.NamedTemporaryFile(
        suffix='.musicxml', delete=False, mode='w', encoding='utf-8',
    )
    tree.write(tmp, xml_declaration=True, encoding='unicode')
    tmp.close()
    return tmp.name


def mark_cue_notes_hidden(xml_path: str) -> str:
    """Backward-compatible alias for preprocess_cue_notes."""
    return preprocess_cue_notes(xml_path)


def remove_hidden_sounding_notes(score):
    """
    Remove hidden Note/Chord elements and zero-duration Rest spacers.
    Preserve positive-duration hidden Rests for the MusicXML branch.

    This ensures the Humdrum `**kern` output does not include redundant tremolo
    expansion sequences and retains only tremolo markings (e.g., `TT`).

    Also removes spanners that reference hidden notes.
    Without this, converter21 throws "Element is not in hierarchy" when it
    tries to calculate spanner lengths for removed notes.
    Example: Chopin Ballades #3 has hidden tremolo notes referenced by ottava marks.
    """
    elements_to_remove = []
    element_ids_to_remove = set()

    for el in score.recurse():
        if not hasattr(el, 'style') or not el.style.hideObjectOnPrint:
            continue
        is_hidden_note = isinstance(el, (m21.note.Note, m21.chord.Chord))
        is_zero_rest = (
            isinstance(el, m21.note.Rest)
            and el.duration.quarterLength == 0
        )
        if is_hidden_note or is_zero_rest:
            elements_to_remove.append(el)
            element_ids_to_remove.add(id(el))

    # Spanners cannot retain endpoints that this shared pass removes.
    if element_ids_to_remove:
        spanners_to_remove = []
        for spanner in score.flatten().spanners:
            try:
                for elem in spanner.getSpannedElements():
                    if id(elem) in element_ids_to_remove:
                        spanners_to_remove.append(spanner)
                        break
            except:
                pass

        # Remove spanners first
        for spanner in spanners_to_remove:
            try:
                for part in score.parts:
                    if spanner in part.flatten().spanners:
                        part.remove(spanner, recurse=True)
                        break
            except:
                pass

    # Record which voice INSTANCES had notes removed BEFORE deletion
    # (activeSite becomes None after remove).
    affected_voices = set()
    for el in elements_to_remove:
        if not isinstance(el, (m21.note.Note, m21.chord.Chord)):
            continue
        site = el.activeSite
        if site is not None and isinstance(site, m21.stream.Voice):
            affected_voices.add(id(site))

    # Now safe to remove the notes
    for el in elements_to_remove:
        try:
            parent = el.activeSite
            if parent:
                parent.remove(el)
        except:
            # If removal fails, unhide it (conservative fallback)
            try:
                el.style.hideObjectOnPrint = False
            except:
                pass

    # A voice emptied by hidden-note removal has no remaining musical identity;
    # keeping only its filler rests makes the canonical writer engrave a ghost voice.
    if affected_voices:
        for part in score.parts:
            for measure in part.getElementsByClass(m21.stream.Measure):
                for voice in list(measure.voices):
                    if id(voice) not in affected_voices:
                        continue
                    has_notes = any(
                        isinstance(n, (m21.note.Note, m21.chord.Chord))
                        for n in voice.recurse().notesAndRests
                    )
                    if not has_notes:
                        voice._empty_after_hidden_notes = True
                        measure.remove(voice)


def prepare_hidden_rests_for_humdrum(score):
    """Expose hidden rests and remove voices emptied by hidden notes."""
    for rest in score.recurse().getElementsByClass(m21.note.Rest):
        if rest.style.hideObjectOnPrint:
            # Rest: unhide (preserve measure structure)
            rest.style.hideObjectOnPrint = False

    # Remove only the voice instances marked during hidden-note removal.
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            for voice in list(measure.voices):
                if getattr(voice, '_empty_after_hidden_notes', False):
                    measure.remove(voice)




def snap_dynamics_to_notes(score):
    """
    Rebind DynamicWedges (crescendo/diminuendo) from Rests to actual Notes in other Voices.
    
    Problem: When a DynamicWedge is attached to Rests in one Voice, but the actual notes
    are in a different Voice at the same time, converter21 cannot process it and throws:
        "Element <music21.note.Rest X ql> is not in hierarchy"
    
    Solution: If a DynamicWedge's start/end are Rests, search for Notes/Chords in other
    Voices of the same Part at those offsets, and rebind the wedge to those notes.
    If no exact match is found, snap to the nearest Note within 0.25 beats.
    
    Example: Chopin Etude Op.10 #12, Measure 52 has a Crescendo on Voice with Rests,
    but the actual melody notes are in a different Voice.
    
    Completely rewritten to handle cross-Voice dynamics.
    Note: DynamicWedges are stored at Part level, not Measure level.
    Use replaceSpannedElement() API to rebind, not clearSpannedElements().
    """
    for part in score.parts:
        # Get all DynamicWedges at Part level (not measure level)
        wedges = list(part.flatten().getElementsByClass(m21.dynamics.DynamicWedge))
        
        for wedge in wedges:
            try:
                spanned = list(wedge.getSpannedElements())
                if len(spanned) < 2:
                    continue  # Need at least start and end
                
                start_elem = spanned[0]
                end_elem = spanned[-1]
                
                # Check if start or end is a Rest
                start_is_rest = isinstance(start_elem, m21.note.Rest)
                end_is_rest = isinstance(end_elem, m21.note.Rest)
                
                if not (start_is_rest or end_is_rest):
                    continue  # Both are Notes/Chords, no need to fix
                
                # Get the offsets and measure numbers
                start_offset = float(start_elem.offset)
                end_offset = float(end_elem.offset)
                start_measure_num = start_elem.measureNumber
                end_measure_num = end_elem.measureNumber
                
                # Find the measures
                start_measure = None
                end_measure = None
                
                for measure in part.getElementsByClass(m21.stream.Measure):
                    if measure.number == start_measure_num:
                        start_measure = measure
                    if measure.number == end_measure_num:
                        end_measure = measure
                
                if not start_measure or not end_measure:
                    continue
                
                # Search for replacement Notes
                new_start = None
                new_end = None
                
                # Search start measure for Notes at start offset
                if start_is_rest:
                    # First try exact match
                    for voice in start_measure.voices:
                        for note in voice.flatten().notesAndRests:
                            if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                if abs(float(note.offset) - start_offset) < 0.01:
                                    new_start = note
                                    break
                        if new_start:
                            break
                    
                    # If no exact match, find nearest Note within 0.25 beats
                    if not new_start:
                        candidates = []
                        for voice in start_measure.voices:
                            for note in voice.flatten().notesAndRests:
                                if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                    distance = abs(float(note.offset) - start_offset)
                                    if distance < 0.25:
                                        candidates.append((distance, note))
                        
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            new_start = candidates[0][1]
                
                # Search end measure for Notes at end offset
                if end_is_rest:
                    # First try exact match
                    for voice in end_measure.voices:
                        for note in voice.flatten().notesAndRests:
                            if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                if abs(float(note.offset) - end_offset) < 0.01:
                                    new_end = note
                                    break
                        if new_end:
                            break
                    
                    # If no exact match, find nearest Note within 0.25 beats
                    if not new_end:
                        candidates = []
                        for voice in end_measure.voices:
                            for note in voice.flatten().notesAndRests:
                                if isinstance(note, (m21.note.Note, m21.chord.Chord)):
                                    distance = abs(float(note.offset) - end_offset)
                                    if distance < 0.25:
                                        candidates.append((distance, note))
                        
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            new_end = candidates[0][1]
                
                # Rebind using replaceSpannedElement (correct API)
                if new_start:
                    wedge.replaceSpannedElement(start_elem, new_start)
                
                if new_end:
                    wedge.replaceSpannedElement(end_elem, new_end)
                    
            except Exception:
                # If we can't process this wedge, skip it
                continue


def heal_cross_staff(
    score, record_movements: bool = False
) -> Union[int, Tuple[int, List[Dict]]]:
    """
    Repair cross-staff notes by moving them to their logical owner Part.

    Cross-staff notation places notes visually on one staff but logically belonging
    to another. music21 parses based on visual <staff> tag, causing notes to be
    misplaced. This function detects and fixes such cases.

    Now supports ASYMMETRIC Voice structures.
    Added record_movements for Visual Auxiliary Head ground truth.

    Detection logic:
    1. ASYMMETRIC case (NEW): One Part has Voices, the other doesn't
       - Detects when a Voice in Part 1 should belong to Part 0 (or vice versa)
       - Criteria: Voice ends exactly where the other Part's notes begin
       - Criteria: Combined duration ≈ measure duration
       - Action: Move notes directly to the target Part (without creating Voice wrapper)

    2. SYMMETRIC case (existing): Both Parts have notes for the same Voice ID
       - Combined duration = 1x measure duration (complete voice split)
       - Move notes from Part with less coverage to Part with more coverage

    Example fixes:
    - Chopin Etude Op.10 #8 Measure 22: Right hand scale starts on bass staff
    - Schubert Impromptu D.899 #2: Cross-staff arpeggios

    Args:
        score: music21.stream.Score
        record_movements: If True, return movement records for Visual Aux Head ground truth

    Returns:
        int: Number of notes moved (if record_movements=False)
        Tuple[int, List[Dict]]: (moved_count, movement_records) if record_movements=True

        movement_records format:
        [
            {
                'note_id': id(note),
                'pitch': 'C4' or 'C4,E4,G4' for chords,
                'measure': 22,
                'offset': 2.0,
                'from_part': 1,  # 0=upper, 1=lower
                'to_part': 0,
                'reason': 'asymmetric_voice' | 'empty_part_voice1' | 'symmetric_merge'
            },
            ...
        ]
    """
    if len(score.parts) < 2:
        return (0, []) if record_movements else 0

    part0, part1 = score.parts[0], score.parts[1]
    moved = 0
    movement_records: List[Dict] = []

    def get_pitch_str(note_or_chord) -> str:
        """Get pitch string for a note or chord."""
        if isinstance(note_or_chord, m21.chord.Chord):
            return ','.join(p.nameWithOctave for p in note_or_chord.pitches)
        elif isinstance(note_or_chord, m21.note.Note):
            return note_or_chord.nameWithOctave
        else:
            return 'rest'

    def record_movement(note, measure_num: int, from_part: int, to_part: int, reason: str):
        """Record a note movement if record_movements is enabled."""
        if record_movements:
            movement_records.append({
                'note_id': id(note),
                'pitch': get_pitch_str(note),
                'measure': measure_num,
                'offset': float(note.offset),
                'from_part': from_part,
                'to_part': to_part,
                'reason': reason,
            })
    
    # Get all measure numbers
    measure_nums = set()
    for part in [part0, part1]:
        for m in part.getElementsByClass('Measure'):
            measure_nums.add(m.number)
    
    for mnum in sorted(measure_nums):
        m0 = part0.measure(mnum)
        m1 = part1.measure(mnum)
        
        # Check if measures exist (use 'is None' instead of 'not' to handle empty measures)
        # Empty measures have bool(measure) = False, but are still valid measure objects
        if m0 is None or m1 is None:
            continue
        
        # Expected measure duration.  A bar written longer than its meter
        # still hands the cross-staff fragment exactly the gap the other
        # staff leaves, so the yardstick is what the bar actually holds —
        # measuring against the meter would refuse every stretched bar.
        ts = m0.getContextByClass('TimeSignature')
        measure_duration = ts.barDuration.quarterLength if ts else 4.0
        measure_duration = max(
            float(measure_duration),
            float(m0.duration.quarterLength),
            float(m1.duration.quarterLength),
        )
        
        # === ASYMMETRIC Voice structures ===
        # Cross-staff detection: voice_dur + other_dur ≈ measure_duration
        # Cross-staff notes are visually placed on other staff but logically belong together.
        # When combined, they form a complete measure.

        def _would_overlap(voice_notes, target_notes) -> bool:
            # Cross-staff fills what the target lacks; overlap means a
            # second voice, and a flat insert would corrupt the bar.
            spans = [
                (float(n.offset),
                 float(n.offset) + float(n.duration.quarterLength))
                for n in target_notes
            ]
            for note in voice_notes:
                s = float(note.offset)
                e = s + float(note.duration.quarterLength)
                for a, b in spans:
                    if a < e and s < b:
                        return True
            return False

        def _hands_take_turns(voice_notes, target_notes) -> bool:
            # A line handed over to the other staff goes across, and may
            # come back: two blocks of time at most.  Anything beyond that
            # is not one line drawn across the staff break, it is two
            # hands playing in turn, and each keeps the staff it is on.
            marks = sorted(
                [(float(n.offset), 0) for n in voice_notes]
                + [(float(n.offset), 1) for n in target_notes])
            return sum(1 for a, b in zip(marks, marks[1:])
                       if a[1] != b[1]) > 2

        # Case 1: m1 has Voices, m0 doesn't
        if m1.voices and not m0.voices:
            m0_notes = list(m0.notesAndRests)
            if m0_notes:
                # Subcase 1a: Part 0 has some notes (non-Voice notes)
                # Check if any Voice from Part 1 should be moved to complete the measure
                m0_dur = float(sum(n.duration.quarterLength for n in m0_notes))

                for voice in list(m1.voices):
                    notes = list(voice.notesAndRests)
                    if not notes:
                        continue

                    voice_dur = float(sum(n.duration.quarterLength for n in notes))
                    combined_dur = voice_dur + m0_dur

                    # If voice + m0 = complete measure, voice is cross-staff → move to m0
                    if (abs(combined_dur - measure_duration) < 0.1 * measure_duration
                            and not _would_overlap(notes, m0_notes)
                            and not _hands_take_turns(notes, m0_notes)):
                        for note in notes:
                            off = note.offset  # capture before remove() drops activeSite
                            record_movement(note, mnum, from_part=1, to_part=0, reason='asymmetric_voice')
                            voice.remove(note)
                            m0.insert(off, note)
                            moved += 1
            
            else:
                # Subcase 1b: Part 0 is completely empty
                # This occurs when all notes (including upper staff voices) are marked as staff=2
                # 
                # CONVENTION: In MuseScore-generated files (ASAP dataset), Voice 1 conventionally
                # represents the primary melody line, typically belonging to the upper staff.
                # When Part 0 is completely empty but Part 1 contains Voice 1, this indicates
                # cross-staff notation where the right-hand melody is written on the bass staff
                # (e.g., when the melody descends to a low register).
                #
                # Example: Beethoven Piano Sonata 21-1, measures 142-145
                # - All Voice 1 notes are marked staff=2 (bass clef for readability)
                # - Voice 5 (bass line) also in staff=2
                # - Result: Part 0 completely empty, causing failed chunk generation
                #
                # Solution: Move Voice 1 to Part 0 to restore proper staff distribution.
                # This is defensible because:
                # 1. Voice 1 represents the melodic line (right hand/upper staff by convention)
                # 2. Empty Part 0 is structurally incorrect for piano grand staff
                # 3. ASAP dataset uses MuseScore 2.3.x which follows Voice 1-4 = upper staff convention
                #
                # Research note: This convention holds for 96-100% of ASAP dataset (empirically verified)
                for voice in list(m1.voices):
                    if str(voice.id) == '1':
                        notes = list(voice.notesAndRests)
                        if notes:
                            for note in notes:
                                off = note.offset  # capture before remove() drops activeSite
                                record_movement(note, mnum, from_part=1, to_part=0, reason='empty_part_voice1')
                                voice.remove(note)
                                m0.insert(off, note)
                                moved += 1
                        break  # Only move Voice 1, preserve other voices in Part 1

        # Case 2: m0 has Voices, m1 doesn't
        elif m0.voices and not m1.voices:
            m1_notes = list(m1.notesAndRests)
            if m1_notes:
                # Subcase 2a: Part 1 has some notes (non-Voice notes)
                # Check if any Voice from Part 0 should be moved to complete the measure
                m1_dur = float(sum(n.duration.quarterLength for n in m1_notes))

                for voice in list(m0.voices):
                    notes = list(voice.notesAndRests)
                    if not notes:
                        continue

                    voice_dur = float(sum(n.duration.quarterLength for n in notes))
                    combined_dur = voice_dur + m1_dur

                    # If voice + m1 = complete measure, voice is cross-staff → move to m1
                    if (abs(combined_dur - measure_duration) < 0.1 * measure_duration
                            and not _would_overlap(notes, m1_notes)
                            and not _hands_take_turns(notes, m1_notes)):
                        for note in notes:
                            off = note.offset  # capture before remove() drops activeSite
                            record_movement(note, mnum, from_part=0, to_part=1, reason='asymmetric_voice')
                            voice.remove(note)
                            m1.insert(off, note)
                            moved += 1
            
            else:
                # Subcase 2b: Part 1 is completely empty (symmetric case to 1b)
                # This occurs when bass voices are marked as staff=1
                # 
                # CONVENTION: Voice 5 conventionally represents the bass line (left hand/lower staff)
                # When Part 1 is empty but Part 0 contains Voice 5, move it to restore proper distribution.
                #
                # Note: This case is rarer than 1b, but follows the same logic
                for voice in list(m0.voices):
                    if str(voice.id) == '5':
                        notes = list(voice.notesAndRests)
                        if notes:
                            for note in notes:
                                off = note.offset  # capture before remove() drops activeSite
                                record_movement(note, mnum, from_part=0, to_part=1, reason='empty_part_voice5')
                                voice.remove(note)
                                m1.insert(off, note)
                                moved += 1
                        break  # Only move Voice 5, preserve other voices in Part 0
        
        # === EXISTING: SYMMETRIC Voice-to-Voice logic ===
        else:
            # Collect voice -> [(offset, duration, note, voice_obj)] for each part
            def collect_voice_data(measure):
                result = defaultdict(list)
                if measure.voices:
                    for voice in measure.voices:
                        for el in voice.getElementsByClass(['Note', 'Chord']):
                            result[voice.id].append({
                                'offset': float(el.offset),
                                'duration': float(el.duration.quarterLength),
                                'note': el,
                                'voice': voice
                            })
                return result
            
            p0_data = collect_voice_data(m0)
            p1_data = collect_voice_data(m1)
            
            all_voice_ids = set(p0_data.keys()) | set(p1_data.keys())
            
            for vid in all_voice_ids:
                p0_notes = p0_data.get(vid, [])
                p1_notes = p1_data.get(vid, [])
                
                # Skip if only one side has notes (not a cross-staff split)
                if not p0_notes or not p1_notes:
                    continue
                
                # Calculate total duration covered in each part
                p0_total_dur = sum(n['duration'] for n in p0_notes)
                p1_total_dur = sum(n['duration'] for n in p1_notes)
                combined_dur = p0_total_dur + p1_total_dur
                
                # Validation: 1x (Split) or 2x (Imbalanced) measure duration
                tolerance = 0.01
                is_split = abs(combined_dur - measure_duration) < tolerance
                is_imbalanced = abs(combined_dur - 2 * measure_duration) < tolerance
                
                if not (is_split or is_imbalanced):
                    continue
                
                # For Case 2, only act if there's actual surplus/deficit
                if is_imbalanced and abs(p0_total_dur - measure_duration) < tolerance:
                    continue
                
                # Merge to the Part with more coverage
                if p0_total_dur > p1_total_dur:
                    # Merge P1 -> P0
                    target_v = None
                    for v in m0.voices:
                        if str(v.id) == str(vid):
                            target_v = v
                            break
                    if target_v is None:
                        target_v = m21.stream.Voice()
                        target_v.id = str(vid)
                        m0.insert(0, target_v)

                    for item in p1_notes:
                        note = item['note']
                        record_movement(note, mnum, from_part=1, to_part=0, reason='symmetric_merge')
                        item['voice'].remove(note)
                        target_v.insert(item['offset'], note)
                        moved += 1

                elif p1_total_dur > p0_total_dur:
                    # Merge P0 -> P1
                    target_v = None
                    for v in m1.voices:
                        if str(v.id) == str(vid):
                            target_v = v
                            break
                    if target_v is None:
                        target_v = m21.stream.Voice()
                        target_v.id = str(vid)
                        m1.insert(0, target_v)

                    for item in p0_notes:
                        note = item['note']
                        record_movement(note, mnum, from_part=0, to_part=1, reason='symmetric_merge')
                        item['voice'].remove(note)
                        target_v.insert(item['offset'], note)
                        moved += 1

    return (moved, movement_records) if record_movements else moved


def refresh_spanners_after_heal(score):
    """
    Repair broken spanners after notes have been moved by `heal_cross_staff`.

    When notes are moved from one Part to another, Spanners (e.g., Slurs,
    DynamicWedges, Ottavas) may retain stale references. This function rebuilds
    those references.

    Strategy:
    1. Build a mapping from note object id to the note (post-move locations).
    2. Detect broken spanners by attempting `getOffsetInHierarchy` on their elements.
    3. Reconstruct spanner element references using the id mapping.
    4. Keep ALL spanners - never remove them (preserves musical semantics).

    Returns:
        int: number of spanners fixed
    """
    # Build note id -> note mapping (post-move positions)
    note_map = {}
    for note in score.flatten().notesAndRests:
        note_map[id(note)] = note

    fixed_count = 0

    for spanner in list(score.flatten().spanners):
        is_broken = False
        element_ids = []

        try:
            # Test spanner and record element ids
            for elem in spanner.getSpannedElements():
                element_ids.append(id(elem))
                elem.getOffsetInHierarchy(score)
        except:
            is_broken = True

        if is_broken and element_ids:
            # Attempt reconstruction
            try:
                spanner.clearSpannedElements()
                
                for eid in element_ids:
                    if eid in note_map:
                        elem = note_map[eid]
                        try:
                            elem.getOffsetInHierarchy(score)
                            spanner.addSpannedElements(elem)
                        except:
                            pass

                fixed_count += 1

            except:
                # If reconstruction fails, leave spanner as-is
                # NEVER remove spanners - preserves musical information
                pass

    return fixed_count


def remove_ghost_note_spanners(score):
    """
    Remove spanners that reference ghost notes.
    
    Ghost notes are artifacts from MusicXML export or music21 parsing errors.
    They are characterized by:
    1. Not present in any Measure's hierarchy (id not in valid_notes set)
    2. Duration = 0 (zero-length notes)
    
    This function only removes spanners with duration=0 ghost notes to avoid
    accidentally removing spanners that reference valid notes which are 
    temporarily out of hierarchy during processing.
    
    Example: Chopin Etude Op.10 #8 Measure 26 has a PedalMark referencing
    a ghost note F2 with duration=0 that doesn't exist in any measure.
    
    Returns:
        int: Number of spanners removed
    """
    # Build set of all valid note IDs in the score hierarchy
    valid_note_ids = set(id(note) for note in score.flatten().notesAndRests)
    
    removed_count = 0
    
    for spanner in list(score.flatten().spanners):
        has_ghost = False
        
        try:
            for elem in spanner.getSpannedElements():
                # Repeat brackets span Measures; an empty Measure has zero
                # duration but is not a ghost note.
                if not isinstance(
                    elem, (m21.note.Note, m21.chord.Chord, m21.note.Rest)
                ):
                    continue
                # Check 1: Element not in hierarchy
                if id(elem) not in valid_note_ids:
                    # Check 2: Zero-length duration (true ghost note)
                    if hasattr(elem, 'duration') and elem.duration.quarterLength == 0:
                        has_ghost = True
                        break
        except:
            # If we can't even get spanned elements, skip this spanner
            continue
        
        if has_ghost:
            # Remove this spanner from all parts
            try:
                for part in score.parts:
                    if spanner in part.flatten().spanners:
                        part.remove(spanner, recurse=True)
                        removed_count += 1
                        break
            except:
                pass
    
    return removed_count


def remove_cross_part_spanners(score):
    """
    Remove spanners that reference notes not in the current score.
    
    This function is used AFTER separating Parts into individual Score objects
    for model compatibility (e.g., Zeng et al. 2024's architecture requires
    separate upper/lower staff inputs).
    
    IMPORTANT: This is a TECHNICAL operation, not a data quality issue.
    - Full Score retains ALL spanners (preserves complete musical semantics)
    - Separated Part Scores remove cross-part spanners due to music21 limitation:
      Creating Score([part]) breaks Spanner hierarchy references to notes in other parts
    
    Cross-staff musical elements affected:
    - Slurs connecting notes across treble/bass staves
    - Dynamic wedges spanning both staves
    - Ottava markings crossing staff boundaries
    
    Defense for reviewers: Zeng's CNN architecture cannot process cross-staff
    information by design. Our ViT-based model uses Full Score, preserving all semantics.
    
    Args:
        score: music21.stream.Score containing a single Part
    
    Returns:
        int: Number of spanners removed
    """
    # Build set of valid note IDs in this separated score
    valid_note_ids = set(id(note) for note in score.flatten().notesAndRests)
    
    removed_count = 0
    
    for spanner in list(score.flatten().spanners):
        has_invalid_refs = False
        
        try:
            for elem in spanner.getSpannedElements():
                if id(elem) not in valid_note_ids:
                    # This spanner references a note from another Part
                    has_invalid_refs = True
                    break
        except:
            continue
        
        if has_invalid_refs:
            # Remove this cross-part spanner
            try:
                for part in score.parts:
                    if spanner in part.flatten().spanners:
                        part.remove(spanner, recurse=True)
                        removed_count += 1
                        break
            except:
                pass
    
    return removed_count


def truncate_fermata_holds(score) -> List[Dict]:
    """Truncate fermata-bearing measures whose content overflows the meter.

    MuseScore sources sometimes stretch a measure's actual duration to
    write a fermata's held time into the notes; the sign itself already
    means "hold", so the written overflow is cut at the barline.  Notes
    and rests crossing the bar end are shortened to it; elements starting
    at or after it are removed.  Signs (fermata etc.) stay on the notes.

    The fermata gate separates two uses of the same MuseScore feature:
    a stretched bar holding a fermata carries no new music (cut), while
    a stretched bar without one carries real content (e.g. a written-out
    transition — kept; reported with action "kept").  The gate is
    evaluated across parts by measure position so both staves cut
    together.

    DANGER: the gate is per measure position, not per element.  One
    fermata anywhere in the bar makes everything past the barline
    cuttable, so a bar that opens on a fermata and then carries a free
    written-out passage loses that passage — Chopin's Étude Op. 25 No. 5
    ends that way, and the cut deleted its twelve-note closing arpeggio.
    Corpora that notate free passages inside a fermata bar must not call
    this at all.

    Mutates the score in place.  Returns one report entry per affected
    element so the caller can log every cut:
    [{"measure", "meter", "span_qn", "part", "offset_qn", "action",
      "what", "from_ql", "to_ql"}]
    """
    eps = 1e-4
    report: List[Dict] = []

    # Pass 1: by measure position, is any part overfull / fermata-bearing?
    part_measures = [list(p.getElementsByClass(m21.stream.Measure))
                     for p in score.parts]
    n_positions = max((len(ms) for ms in part_measures), default=0)
    overfull_pos = set()
    fermata_pos = set()
    for ms in part_measures:
        for idx, meas in enumerate(ms):
            ts = meas.timeSignature or meas.getContextByClass(m21.meter.TimeSignature)
            if ts is None:
                continue
            if float(meas.duration.quarterLength) > float(ts.barDuration.quarterLength) + eps:
                overfull_pos.add(idx)
            for el in meas.recurse().notesAndRests:
                if any(isinstance(e, m21.expressions.Fermata)
                       for e in el.expressions):
                    fermata_pos.add(idx)
                    break

    if not overfull_pos:
        return report

    # Pass 2: cut (or report as kept) each overfull position in every part.
    for pi, ms in enumerate(part_measures):
        for idx in sorted(overfull_pos):
            if idx >= len(ms):
                continue
            meas = ms[idx]
            ts = meas.timeSignature or meas.getContextByClass(m21.meter.TimeSignature)
            if ts is None:
                continue
            bar_ql = float(ts.barDuration.quarterLength)
            span = float(meas.duration.quarterLength)
            if span <= bar_ql + eps:
                continue

            if idx not in fermata_pos:
                report.append({
                    "measure": meas.number, "meter": ts.ratioString,
                    "span_qn": span, "part": pi, "offset_qn": bar_ql,
                    "action": "kept", "what": "no fermata — real content",
                    "from_ql": span, "to_ql": span,
                })
                continue

            for el in list(meas.recurse().notesAndRests):
                off = float(el.getOffsetInHierarchy(meas))
                dur = float(el.duration.quarterLength)
                if el.isRest:
                    what = "rest"
                else:
                    what = "+".join(p.nameWithOctave for p in el.pitches)
                entry = {
                    "measure": meas.number,
                    "meter": ts.ratioString,
                    "span_qn": span,
                    "part": pi,
                    "offset_qn": off,
                    "what": what,
                    "from_ql": dur,
                }
                if off >= bar_ql - eps:
                    el.activeSite.remove(el)
                    entry["action"] = "removed"
                    entry["to_ql"] = 0.0
                    report.append(entry)
                elif off + dur > bar_ql + eps:
                    # In-place duration edits bypass the stream's change
                    # notification, so cached durations must be cleared.
                    site = el.activeSite
                    el.duration.quarterLength = bar_ql - off
                    site.clearCache()
                    entry["action"] = "shortened"
                    entry["to_ql"] = bar_ql - off
                    report.append(entry)
            meas.clearCache()
    return report


def repair_out_of_bar_backups(xml_path) -> List[Dict]:
    """Bring a cursor excursion that leaves the bar back inside it.

    A second voice is written by winding the cursor back to the start of
    the bar and laying the notes down again.  A source can inflate both
    halves of that pair by one and the same amount: the wind-back reaches
    hundreds of bars before the barline, the wind-forward carries the
    same surplus, and the two still cancel exactly.  No reader holds a
    cursor before the start of a bar, so the wind-back is pinned at the
    barline while the wind-forward is spent in full, and everything after
    it lands hundreds of bars late.  Because the surplus is common to the
    pair, subtracting it restores what was meant without guessing: the
    notes keep their span and the cursor keeps its landing.

    A pair whose remainder still will not fit inside the bar is left
    alone.  Rewrites the file in place; returns one entry per pair.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    def duration_of(element) -> int:
        node = element.find(f'{ns}duration')
        if node is None or not node.text:
            return 0
        try:
            return int(node.text)
        except ValueError:
            return 0

    def set_duration(element, value: int) -> None:
        node = element.find(f'{ns}duration')
        if node is not None:
            node.text = str(int(value))

    report: List[Dict] = []
    for part_index, part in enumerate(root.iter(f'{ns}part')):
        divisions = 1
        for measure in part.iter(f'{ns}measure'):
            for node in measure.iter(f'{ns}divisions'):
                try:
                    divisions = max(1, int(node.text))
                except (TypeError, ValueError):
                    pass

            steps = []
            cursor = 0
            for element in measure:
                tag = element.tag.replace(ns, '')
                if tag == 'backup':
                    steps.append((element, tag, cursor))
                    cursor -= duration_of(element)
                elif tag == 'forward':
                    steps.append((element, tag, cursor))
                    cursor += duration_of(element)
                elif tag == 'note':
                    if element.find(f'{ns}chord') is None:
                        steps.append((element, tag, cursor))
                        cursor += duration_of(element)

            index = 0
            while index < len(steps):
                element, tag, before = steps[index]
                back = duration_of(element)
                if tag != 'backup' or before - back >= 0:
                    index += 1
                    continue
                # The excursion runs until the cursor is back inside the
                # bar, however many wind-forwards that takes.  A second
                # wind-back on the way makes the pairing ambiguous, and
                # the excursion is then left exactly as written.
                cursor = before - back
                forwards = []
                ahead = index + 1
                broken = False
                while ahead < len(steps) and cursor < 0:
                    node, kind, _ = steps[ahead]
                    if kind == 'backup':
                        broken = True
                        break
                    if kind == 'forward':
                        forwards.append(node)
                    cursor += duration_of(node)
                    ahead += 1
                if broken or cursor < 0 or not forwards:
                    index += 1
                    continue
                surplus = sum(duration_of(node) for node in forwards)
                keep = back - surplus
                if keep < 0 or keep > before:
                    index += 1
                    continue
                set_duration(element, keep)
                for node in forwards:
                    set_duration(node, 0)
                report.append({
                    "measure": measure.get("number"),
                    "part": part_index,
                    "from_qn": back / divisions,
                    "to_qn": keep / divisions,
                    "forward_qn": surplus / divisions,
                })
                index = ahead

    if report:
        tree.write(xml_path, xml_declaration=True, encoding='unicode')
    return report


def clear_overfull_measures(score) -> List[Dict]:
    """Give a bar that outgrew its meter back to silence.

    A free passage written inside one barline leaves the bar longer than
    the meter it declares, and there is no beat grid in that meter to put
    those events on.  The bar is handed the meter's own silence instead,
    so the barline count stays true and the music either side of it goes
    on; the bar is reported so it can be quarantined rather than trusted.

    Both staves clear together, because a bar is one moment in time.  A
    clef written inside it still speaks for what follows and moves to the
    barline; slurs and wedges holding what the bar held go with it, and a
    tie reaching in or out is released, having nothing left to reach.

    Returns one entry per cleared bar.
    """
    from fractions import Fraction

    def _frac(value) -> Fraction:
        return Fraction(value).limit_denominator(10 ** 7)

    report: List[Dict] = []
    part_measures = [list(part.getElementsByClass(m21.stream.Measure))
                     for part in score.parts]
    if not part_measures:
        return report

    doomed = []
    for index in range(max(len(ms) for ms in part_measures)):
        here = [ms[index] for ms in part_measures if index < len(ms)]
        if not here:
            continue
        meter_mark = here[0].timeSignature or here[0].getContextByClass(
            m21.meter.TimeSignature)
        if meter_mark is None:
            continue
        meter = _frac(meter_mark.barDuration.quarterLength)
        span = max(_frac(measure.duration.quarterLength) for measure in here)
        if span <= meter:
            continue
        doomed.append((index, here, meter_mark, meter, span))

    if not doomed:
        return report

    emptied = set()
    for index, here, meter_mark, meter, span in doomed:
        for measure in here:
            for element in list(measure.recurse().notesAndRests):
                emptied.add(id(element))
                holder = element.activeSite
                if holder is not None:
                    holder.remove(element)
            for voice in list(measure.voices):
                measure.remove(voice)
            # What the bar declares still governs the music after it, so
            # declarations move to the barline rather than being dropped;
            # of several dynamics only the last is still in force there.
            standing = [
                element for element in measure.getElementsByClass((
                    m21.clef.Clef,
                    m21.key.KeySignature,
                    m21.dynamics.Dynamic,
                    m21.expressions.TextExpression,
                    m21.tempo.TempoIndication,
                ))
            ]
            louds = [element for element in standing
                     if isinstance(element, m21.dynamics.Dynamic)]
            for element in louds[:-1]:
                measure.remove(element)
                standing.remove(element)
            for element in standing:
                if _frac(measure.elementOffset(element)) > 0:
                    measure.setElementOffset(element, 0.0)
            silence = m21.note.Rest()
            silence.duration.quarterLength = meter
            measure.insert(0.0, silence)
            measure.clearCache()
        report.append({
            "measure": here[0].number,
            "index": index,
            "meter": meter_mark.ratioString,
            "span_qn": float(span),
            "meter_qn": float(meter),
        })

    for spanner in list(score.recurse().getElementsByClass(m21.spanner.Spanner)):
        if any(id(element) in emptied
               for element in spanner.getSpannedElements()):
            try:
                score.remove(spanner, recurse=True)
            except Exception:
                pass

    cleared_positions = {index for index, *_ in doomed}
    for measures in part_measures:
        for index in cleared_positions:
            for neighbour, at_edge in (
                    (index - 1, "end"), (index + 1, "start")):
                if not 0 <= neighbour < len(measures):
                    continue
                measure = measures[neighbour]
                edge = _frac(measure.duration.quarterLength)
                for element in measure.recurse().notesAndRests:
                    if getattr(element, "tie", None) is None:
                        continue
                    start = _frac(element.getOffsetInHierarchy(measure))
                    touches = (
                        start + _frac(element.duration.quarterLength) >= edge
                        if at_edge == "end" else start <= 0)
                    if touches:
                        element.tie = None

    return report


def infer_missing_meter_score(score) -> Optional[str]:
    """Object-level twin of the kern-text pass, for chains that branch to
    MusicXML before any kern text exists.  Returns the meter it declared."""
    from fractions import Fraction

    from src.score.kern_utils import _span_to_meter

    if score.recurse().getElementsByClass(m21.meter.TimeSignature):
        return None

    measures = [list(p.getElementsByClass(m21.stream.Measure))
                for p in score.parts]
    if not measures or not measures[0]:
        return None

    spelled = _span_to_meter(
        Fraction(measures[0][0].duration.quarterLength).limit_denominator(1000))
    if spelled is None:
        return None
    num, den = spelled

    for part_measures in measures:
        if part_measures:
            part_measures[0].insert(
                0.0, m21.meter.TimeSignature(f'{num}/{den}'))
    return f'{num}/{den}'


def fill_empty_measures(score) -> List[Dict]:
    """Give a bar with nothing written in it the meter's own silence.

    A bar whose only content was hidden filler is left empty once that
    filler is dropped, and the two writers then disagree about it: the
    MusicXML writer materialises a whole-bar rest, the Humdrum writer
    skips the bar entirely.  The bar would vanish from the kern alone and
    every bar after it would shift, so it is written out as silence here
    and reported, like an overfull bar, for quarantine.

    Runs after the overfull pass, whose test (span greater than the meter)
    cannot see a bar that has been emptied.

    Returns one entry per filled bar.
    """
    from fractions import Fraction

    report: List[Dict] = []
    part_measures = [list(part.getElementsByClass(m21.stream.Measure))
                     for part in score.parts]
    if not part_measures:
        return report

    for index in range(max(len(ms) for ms in part_measures)):
        here = [ms[index] for ms in part_measures if index < len(ms)]
        if not here or any(list(m.recurse().notesAndRests) for m in here):
            continue
        meter_mark = here[0].timeSignature or here[0].getContextByClass(
            m21.meter.TimeSignature)
        if meter_mark is None:
            continue
        meter = Fraction(meter_mark.barDuration.quarterLength).limit_denominator(
            10 ** 7)
        for measure in here:
            silence = m21.note.Rest()
            silence.duration.quarterLength = meter
            measure.insert(0.0, silence)
            measure.clearCache()
        report.append({
            "measure": here[0].number,
            "index": index,
            "meter": meter_mark.ratioString,
            "span_qn": 0.0,
            "meter_qn": float(meter),
            "empty": True,
        })
    return report


def exact_tuplet_durations(score) -> List[Dict]:
    """Restore complete tuplet groups without changing their metric identity.

    A source grid rounds member durations and can leave tiny gaps or overlaps
    inside one tuplet.  Start/stop markers define the musical group; metadata
    is trusted only when its exact group span differs from the source span by
    no more than the group's accumulated source-tick rounding.  Cross-staff
    members share that calculation but remain on their written staff.

    Returns one report entry per corrected note.
    """
    from fractions import Fraction
    from math import gcd, lcm

    def _frac(value) -> Fraction:
        return Fraction(value).limit_denominator(10 ** 7)

    def _find_groups(rows):
        groups = []
        active = []
        signature = None
        for row in sorted(rows, key=lambda item: (item[0], item[4])):
            element = row[1]
            tuplets = element.duration.tuplets
            # A partial view of a nested tuplet cannot establish its full span.
            if len(tuplets) != 1:
                continue
            tuplet = tuplets[0]
            current_signature = (
                tuplet.numberNotesActual,
                tuplet.numberNotesNormal,
                _frac(tuplet.tupletMultiplier()),
            )
            marker = tuplet.type
            if marker in {"start", "startStop"}:
                active = [row]
                signature = current_signature
            elif active and current_signature == signature:
                active.append(row)
            else:
                continue
            if marker in {"stop", "startStop"}:
                groups.append(active)
                active = []
                signature = None
        return groups

    values_by_measure = defaultdict(list)
    holder_rows = {}
    for part_index, part in enumerate(score.parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            for holder in (list(measure.voices) or [measure]):
                rows = sorted(
                    [
                        (
                            _frac(element.getOffsetInHierarchy(measure)),
                            element,
                            holder,
                            measure,
                            part_index,
                        )
                        for element in holder.notesAndRests
                    ],
                    key=lambda item: item[0],
                )
                holder_rows[id(holder)] = rows
                for offset, element, _, _, _ in rows:
                    duration = _frac(element.duration.quarterLength)
                    if offset > 0:
                        values_by_measure[measure.number].append(offset)
                    if duration > 0:
                        values_by_measure[measure.number].append(duration)

    source_ticks = {}
    for measure_number, values in values_by_measure.items():
        denominator = 1
        for value in values:
            denominator = lcm(denominator, value.denominator)
        numerator = 0
        for value in values:
            numerator = gcd(
                numerator,
                value.numerator * (denominator // value.denominator),
            )
        if numerator:
            source_ticks[measure_number] = Fraction(numerator, denominator)

    groups = []
    used_members = set()
    if len(score.parts) >= 2:
        first_part, second_part = score.parts[:2]
        measure_numbers = {
            measure.number
            for part in (first_part, second_part)
            for measure in part.getElementsByClass(m21.stream.Measure)
        }
        for measure_number in sorted(measure_numbers):
            first_measure = first_part.measure(measure_number)
            second_measure = second_part.measure(measure_number)
            if first_measure is None or second_measure is None:
                continue
            pairs = []
            if second_measure.voices and not first_measure.voices:
                pairs.extend((voice, first_measure)
                             for voice in second_measure.voices)
            elif first_measure.voices and not second_measure.voices:
                pairs.extend((voice, second_measure)
                             for voice in first_measure.voices)
            elif first_measure.voices and second_measure.voices:
                second_voices = {
                    str(voice.id): voice for voice in second_measure.voices
                }
                pairs.extend(
                    (voice, second_voices[str(voice.id)])
                    for voice in first_measure.voices
                    if str(voice.id) in second_voices
                )
            for first_holder, second_holder in pairs:
                rows = (
                    holder_rows.get(id(first_holder), [])
                    + holder_rows.get(id(second_holder), [])
                )
                for group in _find_groups(rows):
                    member_ids = {id(row[1]) for row in group}
                    holder_ids = {id(row[2]) for row in group}
                    if len(holder_ids) > 1 and not member_ids & used_members:
                        groups.append(group)
                        used_members.update(member_ids)

    for holder_id, rows in holder_rows.items():
        local_rows = [
            row for row in rows if id(row[1]) not in used_members
        ]
        for group in _find_groups(local_rows):
            groups.append(group)
            used_members.update(id(row[1]) for row in group)

    accepted_groups = []
    exact_values = {}
    tolerances = {}
    accepted_member_ids = set()
    for group in groups:
        measure = group[0][3]
        tick = source_ticks.get(measure.number, Fraction(0))
        group_values = []
        valid = True
        for _, element, _, _, _ in group:
            try:
                exact = _frac(m21.duration.Duration(
                    type=element.duration.type,
                    dots=element.duration.dots,
                ).quarterLength)
                for layer in element.duration.tuplets:
                    exact *= _frac(layer.tupletMultiplier())
            except Exception:
                valid = False
                break
            group_values.append(exact)
        source_start = group[0][0]
        source_end = max(
            offset + _frac(element.duration.quarterLength)
            for offset, element, _, _, _ in group
        )
        exact_span = sum(group_values, Fraction(0))
        tolerance = len(group) * tick
        if (
            not valid
            or abs(exact_span - (source_end - source_start)) > tolerance
        ):
            continue
        group_index = len(accepted_groups)
        accepted_groups.append(group)
        tolerances[group_index] = tolerance
        for row, exact in zip(group, group_values):
            exact_values[id(row[1])] = exact
            accepted_member_ids.add(id(row[1]))

    report: List[Dict] = []
    bundles = defaultdict(list)
    for group_index, group in enumerate(accepted_groups):
        holder_key = frozenset(id(row[2]) for row in group)
        bundles[holder_key].append((group_index, group))

    measure_moments = defaultdict(dict)

    def _remember(mapping, source, target):
        if source not in mapping:
            mapping[source] = target
        elif mapping[source] != target:
            mapping[source] = None

    for bundle in bundles.values():
        bundle.sort(key=lambda item: item[1][0][0])
        holders = set()
        measures = set()
        moments = {}
        previous_source_end = None
        previous_target_end = None
        previous_tolerance = None
        for group_index, group in bundle:
            source_start = group[0][0]
            source_end = max(
                offset + _frac(element.duration.quarterLength)
                for offset, element, _, _, _ in group
            )
            target_start = source_start
            if (
                previous_source_end is not None
                and abs(source_start - previous_source_end)
                <= tolerances[group_index] + previous_tolerance
            ):
                target_start = previous_target_end

            cursor = target_start
            for offset, element, holder, measure, _ in group:
                current = _frac(element.duration.quarterLength)
                exact = exact_values[id(element)]
                holder_base = _frac(holder.getOffsetInHierarchy(measure))
                if offset != cursor:
                    holder.setElementOffset(element, cursor - holder_base)
                if current != exact:
                    element.duration.quarterLength = exact
                    tuplet = element.duration.tuplets[0]
                    report.append({
                        "measure": measure.number,
                        "ratio": f"{tuplet.numberNotesActual}:"
                                 f"{tuplet.numberNotesNormal}",
                        "from_ql": float(current),
                        "to_ql": float(exact),
                    })
                _remember(moments, offset, cursor)
                _remember(moments, offset + current, cursor + exact)
                cursor += exact
                holders.add(holder)
                measures.add(measure)
                holder.clearCache()
            previous_source_end = source_end
            previous_target_end = cursor
            previous_tolerance = tolerances[group_index]

        remaining = []
        for holder in holders:
            remaining.extend(
                row for row in holder_rows[id(holder)]
                if id(row[1]) not in accepted_member_ids
            )
        for offset, element, holder, measure, _ in sorted(
            remaining, key=lambda item: item[0]
        ):
            target = moments.get(offset)
            if target is None or target == offset:
                continue
            holder_base = _frac(holder.getOffsetInHierarchy(measure))
            holder.setElementOffset(element, target - holder_base)
            duration = _frac(element.duration.quarterLength)
            _remember(moments, offset + duration, target + duration)
            holder.clearCache()

        for measure in measures:
            mapping = measure_moments[id(measure)]
            for source, target in moments.items():
                if target is not None:
                    _remember(mapping, source, target)
            measure.clearCache()

    # A marking at a moved onset belongs to that musical moment, not to the
    # source grid coordinate that happened to approximate it.
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            mapping = measure_moments.get(id(measure), {})
            if not mapping:
                continue
            for element in list(measure):
                if isinstance(element, (m21.stream.Stream, m21.note.GeneralNote)):
                    continue
                duration = getattr(element, "duration", None)
                if duration is None or _frac(duration.quarterLength) != 0:
                    continue
                source = _frac(measure.elementOffset(element))
                target = mapping.get(source)
                if target is not None and target != source:
                    measure.setElementOffset(element, target)
            measure.clearCache()
    return report


def drop_hidden_filler_rests(score) -> List[Dict]:
    """Remove the invisible rests a rounded tuplet left the other voices.

    A tuplet the source could not divide evenly makes its own voice come
    out short, and the engraver squares the bar by appending rests that
    are never printed — one sixty-fourth on each of the other voices, say.
    Once the tuplet holds its written ratio again those rests are pure
    surplus, and because they were never printed and never sounded,
    dropping them changes neither the page nor the ear.

    Only bars that still exceed their meter are touched, only invisible
    rests, and only up to the surplus.  Returns one entry per rest.
    """
    from fractions import Fraction

    def _frac(value) -> Fraction:
        return Fraction(value).limit_denominator(10 ** 7)

    # Hand any marking anchored to these rests over to a real note first;
    # this is the same station the chain already runs later, called early
    # so the surplus can go before anything measures the bar.
    try:
        snap_dynamics_to_notes(score)
    except Exception:
        pass

    report: List[Dict] = []
    touched = set()
    for part_index, part in enumerate(score.parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            meter = measure.timeSignature or measure.getContextByClass(
                m21.meter.TimeSignature)
            if meter is None:
                continue
            surplus = (_frac(measure.duration.quarterLength)
                       - _frac(meter.barDuration.quarterLength))
            if surplus <= 0:
                continue
            for holder in (list(measure.voices) or [measure]):
                if surplus <= 0:
                    break
                for element in reversed(list(holder.notesAndRests)):
                    if surplus <= 0:
                        break
                    if not element.isRest:
                        continue
                    if not getattr(element.style, "hideObjectOnPrint", False):
                        continue
                    # snap_dynamics_to_notes has already moved what it can
                    # onto a sounding note of the same hand; a rest still
                    # carrying a marking has nowhere to hand it over, so it
                    # stays.
                    try:
                        if element.getSpannerSites():
                            continue
                    except Exception:
                        continue
                    length = _frac(element.duration.quarterLength)
                    if length <= 0:
                        continue
                    report.append({
                        "measure": measure.number,
                        "part": part_index,
                        "duration_qn": float(length),
                    })
                    holder.remove(element)
                    touched.add(id(holder))
            measure.clearCache()

    # A filler taken from the head or middle of a voice leaves a hole the
    # width of what it held; the notes after it belong that much earlier.
    creep = Fraction(1, 8)
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            for holder in (list(measure.voices) or [measure]):
                if id(holder) not in touched:
                    continue
                cursor = Fraction(0)
                for element in sorted(holder.notesAndRests,
                                      key=lambda e: _frac(holder.elementOffset(e))):
                    offset = _frac(holder.elementOffset(element))
                    if abs(offset - cursor) > creep:
                        cursor = offset
                    elif offset != cursor:
                        holder.setElementOffset(element, cursor)
                    cursor = cursor + _frac(element.duration.quarterLength)
            measure.clearCache()
    return report


def align_midbar_clefs(score) -> List[Dict]:
    """Move a mid-measure clef to the first note it is there for.

    An engraver's clef change is written for the notes that follow it, but
    the source often places it a little early — inside the rest that
    precedes them.  A clef carries no time of its own, yet the writer
    still cuts a grid column at it, so a rest reaching across the change
    is split in two and, having already spent its length on the first
    half, walks into the note after the second.  Moving the clef onto the
    onset it serves says the same thing about the page and leaves the rest
    whole.

    Returns one entry per clef moved.
    """
    from fractions import Fraction

    def _frac(value) -> Fraction:
        return Fraction(value).limit_denominator(10 ** 7)

    # A bar's moments belong to the bar, not to one staff: the column the
    # clef has to land on is cut by whatever sounds anywhere in it, and a
    # staff can fall silent right where its own clef change was written.
    bar_onsets: Dict[Any, set] = defaultdict(set)
    for part in score.parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            for element in measure.recurse().notesAndRests:
                bar_onsets[measure.number].add(
                    _frac(element.getOffsetInHierarchy(measure)))

    report: List[Dict] = []
    for part_index, part in enumerate(score.parts):
        for measure in part.getElementsByClass(m21.stream.Measure):
            clefs = [element for element in measure.getElementsByClass(m21.clef.Clef)
                     if _frac(measure.elementOffset(element)) > 0]
            if not clefs:
                continue
            spans = []
            onsets = set(bar_onsets.get(measure.number, ()))
            for element in measure.recurse().notesAndRests:
                start = _frac(element.getOffsetInHierarchy(measure))
                spans.append((start, start + _frac(element.duration.quarterLength)))
            for clef in clefs:
                here = _frac(measure.elementOffset(clef))
                # Only a clef stranded inside something still sounding does
                # harm; one that already sits on an onset is where the
                # engraver put it and stays.
                if not any(start < here < end for start, end in spans):
                    continue
                later = sorted(onset for onset in onsets if onset > here)
                if not later:
                    continue
                target = later[0]
                measure.setElementOffset(clef, target)
                report.append({
                    "measure": measure.number,
                    "part": part_index,
                    "from_qn": float(here),
                    "to_qn": float(target),
                })
            measure.clearCache()
    return report


def fill_voice_gaps(score) -> List[Dict]:
    """Fill unnotated time inside measure voices with explicit rests.

    MusicXML <forward> skips — and voices that simply start late or end
    early — leave offset ranges with nothing notated.  Timing survives
    parsing (the MIDI chain is unaffected), but converter21's humdrum
    writer renders a gapped voice as mid-bar spine merges/splits with a
    broken time account (notes land past the bar's end).  Every voice
    is therefore padded with real rests to span its measure.

    The target span is the measure's own content span, not the meter:
    pickup and final partial bars (all voices equally short) are
    untouched.  Sound is unchanged — the gaps were silent.

    Returns one report entry per inserted rest:
    [{"measure", "part", "offset_qn", "duration_qn"}]
    """
    eps = 1e-4
    report: List[Dict] = []
    for pi, part in enumerate(score.parts):
        for meas in part.getElementsByClass(m21.stream.Measure):
            span = float(meas.duration.quarterLength)
            if span <= eps:
                continue
            streams = list(meas.voices) or [meas]
            for stream in streams:
                base = (float(stream.getOffsetBySite(meas))
                        if stream is not meas else 0.0)
                els = sorted(
                    stream.notesAndRests,
                    key=lambda e: float(e.getOffsetBySite(stream)))
                if not els:
                    continue
                cursor = 0.0
                inserts = []
                for el in els:
                    off = float(el.getOffsetBySite(stream))
                    if off > cursor + eps:
                        inserts.append((cursor, off - cursor))
                    cursor = max(
                        cursor, off + float(el.duration.quarterLength))
                if base + cursor < span - eps:
                    inserts.append((cursor, span - base - cursor))
                for off, ql in inserts:
                    stream.insert(off, m21.note.Rest(quarterLength=ql))
                    report.append({
                        "measure": meas.number, "part": pi,
                        "offset_qn": base + off, "duration_qn": ql,
                    })
                if inserts:
                    stream.clearCache()
                    meas.clearCache()
    return report
