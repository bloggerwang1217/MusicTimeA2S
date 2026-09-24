"""
Kern Sanitization for Audio Generation
=======================================

Combines kern preprocessing steps needed for converter21/music21 parsing:
1. Repeat expansion (via ``src.score.expand_repeat.expand_kern_repeats``)
2. Cue passage resolution (``sanitize_cue``)
3. Measure extraction (``extract_kern_measures``)

Repeat-related functions have been consolidated into ``src.score.expand_repeat``.

Usage:
    from src.score.sanitize_kern import sanitize_kern_for_audio

    with open("input.krn") as f:
        kern_content = f.read()

    sanitized = sanitize_kern_for_audio(kern_content)
    # Now safe to parse with converter21/music21
"""

import re
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Duration Parsing
# =============================================================================

@lru_cache(maxsize=None)
def parse_kern_duration(token: str) -> Optional[Fraction]:
    """Parse kern token duration, returns Fraction or None.

    Cached: pure text→Fraction over a small vocabulary of tokens, on
    hot paths in standardize and audio sanitization.

    This implementation uses re.search to find duration numbers anywhere
    in the token, which correctly handles tokens like ")4d" where the
    duration number is not at the start.

    Args:
        token: Kern token like "8c#", "16.ee-", "[4f", ")4d"

    Returns:
        Duration as Fraction of quarter notes, or None for invalid tokens
    """
    if not token or token == '.' or token.startswith(('!', '*', '=')):
        return None

    # Grace note (q) has 0 duration
    if 'q' in token.lower() and not token.startswith('='):
        return Fraction(0)

    token_clean = token.split()[0] if ' ' in token else token
    match = re.search(r'(\d+)(?:%(\d+))?', token_clean)
    if not match:
        return None

    recip_num = int(match.group(1))
    recip_den = int(match.group(2)) if match.group(2) else 1
    if recip_num == 0:
        dur = Fraction(8)
    else:
        dur = Fraction(4 * recip_den, recip_num)

    # Handle dots - count all dots in token (dots can appear after pitch letter)
    # e.g., "(4A." = slur + 4 + pitch A + dot, the dot is AFTER the letter
    dots = token_clean.count('.')
    if dots > 0:
        dur = dur * (Fraction(2) - Fraction(1, 2**dots))

    return dur


def extract_kern_measures(kern_path=None, include_timing: bool = False,
                          kern_content: str = None) -> List[Dict]:
    """Extract measure line-ranges from kern file or content string.

    Splits kern content at barlines to find line ranges for each measure region.
    Used for ChunkedDataset — slicing kern_gt tokens at measure boundaries.

    NOTE: The ``measure`` field is a local index (0-based if pickup exists,
    1-based otherwise). It is NOT authoritative — the definitive measure
    number and timing come from the music21 Score via ``extract_measure_times()``.
    ChunkedDataset pairs kern_measures[i] with audio_measures[i] by **index**,
    not by measure number.

    Args:
        kern_path: Path to kern file (str or Path). Mutually exclusive with kern_content.
        include_timing: If True, include start_sec/end_sec (kern-parsed, approximate).
                        For authoritative timing, use extract_measure_times(Score).
        kern_content: Raw kern content string. If provided, kern_path is ignored.

    Returns:
        List of measure info dicts:
        [
            {"measure": 0, "line_start": 21, "line_end": 21},   # pickup (if any)
            {"measure": 1, "line_start": 23, "line_end": 26},
            ...
        ]
        line_start is the first data line of the measure (after the barline)
        line_end is the last data line before the next barline (or end of file)
    """
    measures = []
    current_measure = None
    measure_start_line = None

    # Pattern to match barlines: =N, =N-, =N:|!, ==, etc.
    # Captures the measure number if present
    barline_pattern = re.compile(r'^=(\d+)?')

    # Timing state (only used when include_timing=True)
    current_offset = Fraction(0)       # cumulative offset in quarter notes
    measure_start_offset = Fraction(0) # offset at start of current measure
    tempo_changes: List[Tuple[Fraction, float]] = [(Fraction(0), 120.0)]  # (offset, bpm)

    def offset_to_seconds(offset: Fraction) -> float:
        """Convert quarter note offset to seconds using tempo map."""
        seconds = 0.0
        prev_offset = Fraction(0)
        prev_bpm = tempo_changes[0][1]
        for t_offset, t_bpm in tempo_changes:
            if t_offset >= offset:
                break
            seconds += float(t_offset - prev_offset) * (60.0 / prev_bpm)
            prev_offset = t_offset
            prev_bpm = t_bpm
        seconds += float(offset - prev_offset) * (60.0 / prev_bpm)
        return seconds

    if kern_content is not None:
        lines = kern_content.split('\n')
        # Add newline back for consistency with readlines() format
        lines = [line + '\n' for line in lines]
    else:
        from pathlib import Path as _Path
        kern_path = _Path(kern_path)
        with open(kern_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

    # Pre-scan: find the line number of the LAST final barline (==) in
    # the file.  Only that barline terminates measure tracking; earlier ==
    # barlines (e.g. DaCapo section boundaries) are treated as regular
    # barlines so the music after them is still counted.
    last_final_barline: Optional[int] = None
    for scan_i, scan_line in enumerate(lines, start=1):
        if scan_line.strip().split('\t')[0].startswith('=='):
            last_final_barline = scan_i

    # Detect pickup measure: scan for data lines before the first barline.
    # If found, create measure 0 for the anacrusis content.
    pickup_data_start = None
    for scan_num, scan_line in enumerate(lines, start=1):
        scan_stripped = scan_line.strip()
        if not scan_stripped or scan_stripped.startswith('!') or scan_stripped.startswith('*'):
            continue
        first_tok = scan_stripped.split('\t')[0]
        if barline_pattern.match(first_tok):
            break  # Reached first barline — no pickup (or pickup already handled)
        # Found data before any barline → pickup measure exists
        if pickup_data_start is None:
            pickup_data_start = scan_num

    if pickup_data_start is not None:
        # There's content before the first barline — start tracking measure 0
        current_measure = 0
        measure_start_line = pickup_data_start
        measure_start_offset = Fraction(0)

    # Tracks the 1-indexed line number of the barline that started the current
    # measure (None for pickup measures that have no preceding barline).
    current_measure_barline: Optional[int] = None

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('!'):
            continue

        # Check for barline
        first_token = line.split('\t')[0]
        match = barline_pattern.match(first_token)

        if match:
            # Found a barline
            measure_num_str = match.group(1)

            # Close previous measure (including pickup measure 0)
            if current_measure is not None and measure_start_line is not None:
                line_end = line_num - 1  # End before this barline
                # Skip measures with no actual music data (only comments,
                # interpretations like key/meter changes, or empty lines).
                # music21 does not create Measure objects for these, so
                # including them would cause kern_measures/audio_measures mismatch.
                has_music = False
                if line_end >= measure_start_line:
                    for check_i in range(measure_start_line - 1, line_end):
                        raw = lines[check_i].strip()
                        if raw and not raw.startswith('!') and not raw.startswith('*') and not raw.startswith('='):
                            has_music = True
                            break
                if has_music:
                    entry = {
                        "measure": current_measure,
                        "line_start": measure_start_line,
                        "line_end": line_end,
                    }
                    if current_measure_barline is not None:
                        entry["line_barline"] = current_measure_barline
                    if include_timing:
                        entry["start_sec"] = round(offset_to_seconds(measure_start_offset), 4)
                        entry["end_sec"] = round(offset_to_seconds(current_offset), 4)
                    measures.append(entry)

            # The LAST final barline (==) in the file signals end of
            # piece.  Do NOT start a new measure — any content after it
            # (e.g. tie resolution, reference comments) belongs to the
            # preceding measure, not a new one.  music21 does not create
            # a Measure for post-final-barline content.
            # NOTE: == can also appear as a section double barline
            # mid-piece (e.g. DaCapo point), so we only skip when this
            # is truly the last == in the file.
            if first_token.startswith('==') and line_num == last_final_barline:
                current_measure = None
                measure_start_line = None
                continue

            # Start new measure
            if measure_num_str:
                current_measure = int(measure_num_str)
            elif current_measure is not None:
                current_measure += 1
            else:
                current_measure = 1

            measure_start_line = line_num + 1  # Start after barline
            measure_start_offset = current_offset
            current_measure_barline = line_num  # barline that opens this measure

        elif first_token.startswith('*'):
            # Interpretation line (metadata) - not part of measure content
            # But update measure_start_line if we haven't started the measure yet
            if measure_start_line == line_num:
                measure_start_line = line_num + 1

            # Parse tempo and time signature for timing
            if include_timing:
                if '*MM' in line:
                    tempo_match = re.search(r'\*MM=?(\d+\.?\d*)', line)
                    if tempo_match:
                        bpm = float(tempo_match.group(1))
                        tempo_changes.append((current_offset, bpm))
            continue

        else:
            # Data line - advance offset by max duration across spines
            if include_timing:
                cols = line.split('\t')
                max_dur = Fraction(0)
                for col in cols:
                    if col == '.' or not col:
                        continue
                    # Handle chords (space-separated subtokens)
                    for subtoken in col.split():
                        dur = parse_kern_duration(subtoken)
                        if dur is not None and dur > max_dur:
                            max_dur = dur
                if max_dur > 0:
                    current_offset += max_dur

    # Close the last measure — only if it contains actual music data.
    # After the final barline (==), there are typically only reference
    # comments (!!!) and spine terminators (*-), which should NOT form
    # a phantom measure.
    if current_measure is not None and measure_start_line is not None:
        has_data = False
        last_data_line = measure_start_line
        for i in range(measure_start_line - 1, len(lines)):
            raw = lines[i].strip()
            # Skip empty, comments, spine terminators, interpretation lines
            if not raw or raw.startswith('!') or raw.startswith('*'):
                continue
            # Skip barlines (shouldn't appear, but just in case)
            if raw.startswith('='):
                continue
            # Found actual music data
            has_data = True
            last_data_line = i + 1  # 1-indexed

        if has_data:
            entry = {
                "measure": current_measure,
                "line_start": measure_start_line,
                "line_end": last_data_line,
            }
            if include_timing:
                entry["start_sec"] = round(offset_to_seconds(measure_start_offset), 4)
                entry["end_sec"] = round(offset_to_seconds(current_offset), 4)
            measures.append(entry)

    return measures


def sanitize_cue(kern_content: str, treatment: str) -> str:
    """Apply the cue-passage treatment decided in cue_overrides.csv.

    `*cue`...`*Xcue` spans have two distinct meanings that require opposite
    handling, and Humdrum syntax alone can't tell them apart:

    - "keep" (cadenza / a piacere): the pianist actually plays this passage.
      No-op — leave the notes as-is.
    - "strip": the cue is a reference to what another performer plays
      (concerto orchestral tutti transcribed onto the piano's own staff,
      or another instrument's own part in chamber music). The pianist is
      silent here in real performance. Replace with rests of matching
      duration, same as the ground-truth pipeline already does.

    This is the single place both the audio-synthesis path
    (sanitize_kern_for_audio) and the ground-truth path (standardize_kern)
    resolve cue content, so they can't drift apart again.

    Args:
        kern_content: Kern content (any pipeline stage; *cue markers intact)
        treatment: "strip" or "keep" (default elsewhere is "keep" — a file
            absent from cue_overrides.csv is assumed to be cadenza)

    Returns:
        Kern content with cue passages resolved per treatment
    """
    if treatment == "strip":
        from src.score.clean_kern import strip_cue_passages
        return strip_cue_passages(kern_content)
    return kern_content


# =============================================================================
# Main Entry Point
# =============================================================================


def sanitize_kern_for_audio(kern_content: str, cue_treatment: str = "keep") -> str:
    """Sanitize kern content for audio generation via converter21/music21.

    Applies the full preprocessing pipeline:
    1. expand_kern_repeats - Expand Humdrum expansion labels (*>[A,B,A,...])
    2. sanitize_cue - Resolve *cue passages per cue_overrides.csv (see
       sanitize_cue's docstring); default "keep" leaves cadenza content,
       which the pianist actually plays, untouched.

    Args:
        kern_content: Raw kern file content
        cue_treatment: "strip" or "keep", looked up per-file from
            cue_overrides.csv by the caller

    Returns:
        Sanitized kern content safe for converter21/music21 parsing
    """
    # Lazy import to avoid circular dependency:
    # expand_repeat imports extract_kern_measures from this module.
    from src.score.expand_repeat import expand_kern_repeats

    # Step 1: Expand repeats
    result = expand_kern_repeats(kern_content)

    # Step 2: Resolve cue passages (strip orchestral/chamber reference cues;
    # leave cadenza cues alone)
    result = sanitize_cue(result, cue_treatment)

    return result


# Public API
__all__ = [
    # Main entry point
    'sanitize_kern_for_audio',
    # Cue passage resolution (shared with standardize_kern)
    'sanitize_cue',
    # Measure extraction
    'extract_kern_measures',
    'parse_kern_duration',
]
