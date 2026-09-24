#!/usr/bin/env python3
"""
Kern Token Sanitizer for converter21 Output
============================================

Problem:
    converter21 preserves visual layout information from MusicXML (e.g., rest positions,
    stem directions, articulation marks), producing tokens like:
    - '8rGG' (rest at Great G position)
    - '4g/' (stem up)
    - '8cc\' (stem down)
    - '4.c~' (trill mark)

    Zeng et al. (2024)'s LabelsMultiple vocabulary only recognizes semantic tokens:
    - '8r' (eighth rest, no position)
    - '4g' (quarter note G, no stem direction)

Solution:
    Clean tokens by removing visual layout information while preserving semantic content.

Usage:
    # In training/inference pipeline:
    from clean_kern import clean_kern_token, clean_kern_sequence

    # Single token
    cleaned = clean_kern_token('8rGG')  # -> '8r'

    # Entire sequence (note stems on notes survive Phase 1; they are
    # consumed and stripped later by standardize_kern)
    dirty_seq = "8rGG\t4g/\t8cc\\\t4.c~"
    cleaned_seq = clean_kern_sequence(dirty_seq)  # -> "8r\t4g/\t8cc\\\t4.c~"

Author: bloggerwang
"""

import re
import logging
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# Visual Information Extraction (for Visual Auxiliary Head ground truth)
# ============================================================================

def extract_visual_info(token: str) -> Dict[str, Any]:
    """
    Extract visual layout information from a Kern token.

    This function extracts visual rendering hints that are typically removed
    during cleaning for the main sequence, but preserved for the Visual
    Auxiliary Head as ground truth.

    Args:
        token: Raw Kern token from converter21 (e.g., '8cc>/L', '4g/', '8dd\\J')

    Returns:
        Dict with keys:
        - 'stem': 'up' | 'down' | None
        - 'beam': List of beam markers ['L', 'J', 'k', 'K'] or None
        - 'above_below': 'above' | 'below' | None

    Examples:
        >>> extract_visual_info("8cc>/L")
        {'stem': 'up', 'beam': ['L'], 'above_below': 'above'}
        >>> extract_visual_info("4g/")
        {'stem': 'up', 'beam': None, 'above_below': None}
        >>> extract_visual_info("8dd\\J")
        {'stem': 'down', 'beam': ['J'], 'above_below': None}
    """
    visual_info: Dict[str, Any] = {}

    # Stem direction (cleaned by clean_kern_token)
    # / = stem up, \ = stem down
    if '/' in token:
        visual_info['stem'] = 'up'
    elif '\\' in token:
        visual_info['stem'] = 'down'
    else:
        visual_info['stem'] = None

    # Beam markers (preserved in clean_kern_token by default)
    # L = beam start, J = beam end, k = partial beam back, K = partial beam forward
    beam = []
    for marker in ['L', 'J', 'k', 'K']:
        if marker in token:
            beam.append(marker)
    visual_info['beam'] = beam if beam else None

    # Above/below staff markers (cleaned from note tokens)
    # > = above middle line, < = below middle line
    if '>' in token:
        visual_info['above_below'] = 'above'
    elif '<' in token:
        visual_info['above_below'] = 'below'
    else:
        visual_info['above_below'] = None

    return visual_info


def extract_visual_from_sequence(sequence: str) -> List[List[Dict[str, Any]]]:
    """
    Extract visual info for each token in a sequence.

    This function processes an entire Kern sequence and extracts visual
    information for each token, including the spine index (Voice position).

    Args:
        sequence: Newline-separated lines, each line is tab-separated tokens

    Returns:
        List of lines, each line is list of dicts:
        [
            [  # line 0
                {'stem': 'up', 'beam': ['L'], 'above_below': None, 'spine_index': 0},
                {'stem': None, 'beam': None, 'above_below': None, 'spine_index': 1},
            ],
            [  # line 1
                {'stem': 'down', 'beam': ['J'], 'above_below': None, 'spine_index': 0},
                ...
            ],
        ]

    Note:
        spine_index represents the Voice position within the Part.
        In Humdrum, spines are tab-separated columns representing voices.
        Control tokens (barlines, interpretations, comments) are skipped.

    Examples:
        >>> seq = "8cc/L\\t4g\\n8dd\\\\J\\t4a"
        >>> visual_seq = extract_visual_from_sequence(seq)
        >>> visual_seq[0][0]
        {'stem': 'up', 'beam': ['L'], 'above_below': None, 'spine_index': 0}
    """
    lines = sequence.split('\n')
    result: List[List[Dict[str, Any]]] = []

    for line in lines:
        # Skip control lines (barlines, interpretations, comments)
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            continue

        if '\t' in line:
            tokens = line.split('\t')
        else:
            tokens = [line]

        line_visual: List[Dict[str, Any]] = []
        for spine_index, token in enumerate(tokens):
            # Skip placeholder tokens
            if token in ('.', '', 'q'):
                line_visual.append({
                    'stem': None,
                    'beam': None,
                    'above_below': None,
                    'spine_index': spine_index,
                })
                continue

            visual = extract_visual_info(token)
            visual['spine_index'] = spine_index
            line_visual.append(visual)

        result.append(line_visual)

    return result


def check_tuplet_ratio_notation(token: str) -> bool:
    """
    Check if token contains Humdrum tuplet ratio notation (X%Y).

    These tokens will cause KeyError in Zeng's LabelsMultiple and should be skipped.
    This matches Zeng's original behavior (commit 7bc0bc6, L207-208).

    Examples:
        6%7ryy    → 6 notes in time of 7 (Chopin Ballades #2)
        1920%37   → complex cadenza ratio (Liszt Paganini #6)
        48%13ryy  → 48 notes in time of 13

    Returns:
        True if token contains '%' (tuplet ratio notation)
    """
    if '%' in token:
        logger.warning(
            f"[TUPLET RATIO] Found '{token}' - Humdrum X%Y notation. "
            f"This will cause KeyError in LabelsMultiple. "
            f"Chunk will be skipped (same behavior as Zeng's original pipeline)."
        )
        return True
    return False


def _is_rest_piece(token: str) -> bool:
    """Whether a single (already-cleaned) chord element is a rest."""
    clean = token.lstrip('[({<>')
    pitch_part = re.sub(r'^\d+(?:%\d+)?\.*', '', clean)
    return pitch_part.startswith('r')


def clean_kern_token(token: str) -> str:
    """
    Remove visual layout information from a single Kern token.

    Phase 1 cleanup only: strips visual markers (beam, editorial, rest
    position, unicode). Stem-direction marks on notes are KEPT — they are
    voice-membership evidence consumed (and stripped) downstream by
    standardize_kern. Preserves all musical content (articulation, ties,
    duration values). Content-level decisions (dotted triplet conversion,
    articulation strip, tie strip) belong in Phase 1.5 (standardize_kern).

    Args:
        token: Raw Kern token from converter21 (e.g., '8rGG', '4g/', '8cc\\')

    Returns:
        Cleaned token with visual markers removed but musical content intact.

    Examples:
        >>> clean_kern_token('8rGG')
        '8r'
        >>> clean_kern_token('4g/')
        '4g/'
        >>> clean_kern_token('4.c~')
        '4.c~'
        >>> clean_kern_token('4c_')
        '4c_'
    """
    # 0. Handle chord tokens (space-separated notes within a single token field)
    if ' ' in token and not token.startswith(('=', '*', '!')):
        notes = token.split(' ')
        cleaned_notes = [clean_kern_token(n) for n in notes]
        # A cleaned note can itself expand into multiple space-separated
        # pieces (step 4 below splits a rest glued directly onto a
        # following note); flatten before filtering so those pieces are
        # judged individually, not as one merged string.
        flat_notes = []
        for n in cleaned_notes:
            flat_notes.extend(n.split(' '))
        real_notes = [n for n in flat_notes if not _is_rest_piece(n)]
        # A rest can't sound alongside real notes packed into the same
        # token; drop it once cleanup reveals the mix (rest position-hint
        # markers like "yy" can hide this until stripped, e.g. raw
        # "2rAyy 2D 2EE" only reads as rest+notes after step 4 below).
        if real_notes and len(real_notes) < len(flat_notes):
            return ' '.join(real_notes)
        return ' '.join(flat_notes)

    # 1. Control tokens: barlines (=), interpretations (*), comments (!)
    #    Most are returned as-is, but some need cleaning.
    if token.startswith('='):
        return token

    # 1b. Comment tokens (!) - filter out layout hints that confuse music21
    #     !LO:N:vis=5 → visual duration hint (converter21 can't parse quintuplets)
    #     These cause "Cannot figure out vDurNoDots" errors in music21
    if token.startswith('!'):
        # Remove layout hints for visual duration (vis=), stem direction, etc.
        if ':vis=' in token or ':stem=' in token:
            return '!'  # Keep as empty comment
        return token

    # 1a. Interpretation tokens (*) - clean editorial markers from clefs
    #     *clefG2yy → *clefG2 (yy = editorial/invisible marker)
    #     *clefF4xx → *clefF4 (xx = similar editorial marker)
    if token.startswith('*'):
        if token.startswith('*clef'):
            # Remove editorial markers (y, x suffixes) from clef tokens
            # These indicate invisible/editorial clefs that music21 can't parse
            cleaned = re.sub(r'([FGC]\d)[xy]+$', r'\1', token)
            return cleaned
        return token

    # 2. Empty placeholder or special markers
    if token in ('.', '', 'q'):
        return token

    # =========================================================================
    # =========================================================================
    # 3. TUPLET RATIO NOTATION (%) - WARNING ONLY, NO CONVERSION
    # =========================================================================
    #
    #   converter21 outputs Humdrum's standard tuplet ratio notation:
    #     X%Y = "X notes in the time of Y"
    #
    #   Examples found in ASAP:
    #   ┌─────────────────────────────────────────────────────────────────────┐
    #   │ 6%7ryy    → Chopin Ballades #2, M45 (6 in time of 7)              │
    #   │ 1920%37   → Liszt Paganini #6, M213 (complex cadenza)             │
    #   │ 48%13ryy  → Liszt Paganini #6 (48 in time of 13)                  │
    #   │ 48%17ryy  → Liszt Paganini #6 (48 in time of 17)                  │
    #   └─────────────────────────────────────────────────────────────────────┘
    #
    #   These cause KeyError in Zeng's LabelsMultiple because:
    #   - The tokenizer may strip '%' leaving malformed numbers (6%7 → 67)
    #   - Or the regex doesn't match the X%Y pattern
    #
    #   ⚠️ WE DO NOT CONVERT THESE - Zeng's original pipeline also skips them!
    #   See commit 7bc0bc6, L207-208: `except Exception as e: continue`
    #
    #   These are rare complex tuplets (cadenzas, ornamental passages).
    #   Skipping is the correct behavior for both pipelines.
    #
    # (No conversion - will trigger KeyError → skip chunk, same as Zeng)

    # =========================================================================
    # 4. REST POSITION CLEANUP (CRITICAL FIX for KeyError: 'rGG')
    # =========================================================================
    #    Examples: 4rCC, 8rGG, 2ryy -> 4r, 8r, 2r
    if 'r' in token:
        # Rest position-hint letters can be glued directly onto a
        # following note with no separator (raw "4ryy4G-" = rest + note,
        # the editorial "yy" hides the join). Insert a space first so
        # cleanup doesn't silently concatenate the two into one
        # malformed token ("4r4G-").
        token = re.sub(r'(\d+\.*r)[A-Ga-gy]+(?=\d)', r'\1 ', token)
        # Pattern: (duration)(dot*)r(pitch letters) -> keep only (duration)(dot*)r
        token = re.sub(r'(\d+\.*r)[A-Ga-gy]+', r'\1', token)

    # =========================================================================
    # 5. STEM DIRECTION (kept on notes)
    # =========================================================================
    #    Examples: 4g/ (stem up), 8cc\ (stem down)
    #    Stem marks on notes pass through Phase 1 untouched; standardize
    #    strips them with the articulation table.  Rests have no stem; a
    #    mark left on a rest is engraving noise and goes.
    if _is_rest_piece(token):
        token = re.sub(r'[/\\]', '', token)

    # =========================================================================
    # 5b. EDITORIAL/VISUAL MARKERS (always remove)
    # =========================================================================
    #    X : cautionary/explicit accidental ("show this accidental even if not required")
    #    y : invisible/hidden symbol (visual only)
    #    z : editorial Z / printed rest marker
    #    N : editorial note marker ("editor added this")
    #    * : editorial footnote (when inside token, not at line start)
    #    = : editorial accidental marker (editor added this accidental)
    #    ? : uncertainty marker (editor unsure about this note)
    #    These are pure visual hints with no audible effect
    token = re.sub(r'[XyzN*=?]', '', token)

    # =========================================================================
    # 5c. ACCIDENTAL NORMALIZATION (redundant natural signs)
    # =========================================================================
    #    In music notation, a natural sign can precede an accidental to explicitly
    #    cancel a previous accidental before applying a new one. This is visually
    #    explicit but semantically redundant:
    #
    #    n# or #n = sharp (natural then sharp, or sharp after natural)
    #    n- or -n = flat (natural then flat)
    #    nn = natural (double natural = natural)
    #
    #    Examples from chopin_first_editions:
    #    - ffn# → ff# (F with natural-then-sharp = F#)
    #    - b-n → b- (B-flat with natural = B-flat... actually this might be typo)
    #    - ann → an (A with double-natural = A-natural)
    #
    #    Note: We normalize the accidental AFTER the pitch letters are consumed
    #    Pattern matches: pitch letters followed by redundant accidental combos
    token = re.sub(r'n#', '#', token)   # natural + sharp = sharp
    token = re.sub(r'#n', '#', token)   # sharp + natural = sharp (rare)
    token = re.sub(r'n-', '-', token)   # natural + flat = flat
    token = re.sub(r'-n', '-', token)   # flat + natural = flat (rare)
    token = re.sub(r'nn', 'n', token)   # double natural = natural

    # =========================================================================
    # 6. BEAM MARKERS (visual only)
    # =========================================================================
    token = re.sub(r'[LJkK]', '', token)

    # =========================================================================
    # 9. UNICODE / NON-ASCII CLEANUP (always remove)
    # =========================================================================
    #    Kern standard is ASCII-only. Any Unicode is likely encoding errors
    #    from HumSyn data (e.g., …, π, Ω found in Mozart/Beethoven sonatas).
    token = ''.join(c for c in token if ord(c) < 128)

    return token


def _kern_cell_sounds(cell: str) -> bool:
    """True when a kern data cell carries at least one pitched note."""
    if not cell or cell == '.':
        return False
    for sub in cell.split(' '):
        if not sub or 'r' in sub:
            continue
        if any(ch in 'ABCDEFGabcdefg' for ch in sub):
            return True
    return False


def _sounding_kern_origins(lines: List[str]) -> Set[int]:
    """Declaration index of every **kern spine that ever sounds a note.

    An engraving template can declare more staves than the music uses;
    the unused ones carry a full-length rest column that no later station
    can tell apart from a real silent hand.
    """
    spine_types: List[str] = []
    origins: List[int] = []
    sounding: Set[int] = set()
    initialized = False

    for line in lines:
        if not line.strip() or line.startswith('!!!') or '\t' not in line:
            continue
        cols = line.split('\t')

        if line.startswith('**'):
            spine_types = cols.copy()
            origins = list(range(len(cols)))
            initialized = True
            continue
        if not initialized:
            continue

        if line.startswith('*'):
            spine_types, origins = _advance_spines(cols, spine_types, origins)
            continue
        if line.startswith('!') or line.startswith('='):
            continue

        for i, cell in enumerate(cols):
            if i < len(spine_types) and spine_types[i] == '**kern' \
                    and _kern_cell_sounds(cell):
                sounding.add(origins[i])

    return sounding


def _advance_spines(
    cols: List[str], spine_types: List[str], origins: List[int],
) -> Tuple[List[str], List[int]]:
    """Apply one manipulator line to the spine type / origin tracks."""
    new_types: List[str] = []
    new_origins: List[int] = []
    src_idx = 0
    i = 0
    while i < len(cols):
        if src_idx >= len(spine_types):
            break
        col = cols[i]
        if col == '*^':
            new_types.extend([spine_types[src_idx]] * 2)
            new_origins.extend([origins[src_idx]] * 2)
            src_idx += 1
            i += 1
        elif col == '*v':
            v_count = 1
            while i + v_count < len(cols) and cols[i + v_count] == '*v':
                v_count += 1
            new_types.append(spine_types[src_idx])
            new_origins.append(origins[src_idx])
            src_idx += v_count
            i += v_count
        elif col == '*-':
            src_idx += 1
            i += 1
        else:
            new_types.append(spine_types[src_idx])
            new_origins.append(origins[src_idx])
            src_idx += 1
            i += 1
    return new_types, new_origins


def strip_spines(content: str, keep_dynam: bool = False) -> str:
    """
    Reduce Humdrum content to the spines the ground truth transcribes.

    Handles spine splits (*^), joins (*v), and terminations (*-) correctly
    by tracking which columns correspond to **kern spines throughout the file.

    Two things go: spines whose exclusive interpretation is not wanted, and
    **kern spines that never sound a note anywhere in the file.

    Args:
        content: Raw Humdrum content with multiple spine types
        keep_dynam: If True, preserve **dynam spines (for piano-full).
                    If False, only keep **kern spines (for piano-base).

    Returns:
        Humdrum content with only the sounding **kern (and optionally
        **dynam) spines

    Spine handling:
        - piano-base (keep_dynam=False): Keep only **kern
        - piano-full (keep_dynam=True): Keep **kern and **dynam
        - Always remove: **text, **fing, and other non-kern spines

    Note:
        **text may contain expression markings.
    """
    lines = content.split('\n')
    result_lines = []

    # A file where nothing sounds keeps every spine: the emptiness is then
    # the caller's to report, not this station's to silently erase.
    sounding = _sounding_kern_origins(lines)
    drop_silent = bool(sounding)

    # Track spine types for each column (updated on spine operations)
    spine_types: List[str] = []
    spine_origins: List[int] = []
    initialized = False

    def _wanted(index: int) -> bool:
        spine_type = spine_types[index]
        if spine_type == '**kern':
            return not drop_silent or spine_origins[index] in sounding
        return spine_type == '**dynam' and keep_dynam

    for line in lines:
        # Skip empty lines
        if not line.strip():
            result_lines.append(line)
            continue

        # Global comments (no tabs, start with !!!)
        if line.startswith('!!!'):
            result_lines.append(line)
            continue

        # Check if line has columns
        if '\t' not in line:
            # Single column line - keep if it's a global comment or special
            if line.startswith('!') or line.startswith('*'):
                result_lines.append(line)
            else:
                result_lines.append(line)
            continue

        cols = line.split('\t')

        # Initial spine declaration
        if line.startswith('**'):
            spine_types = cols.copy()
            spine_origins = list(range(len(cols)))
            initialized = True
            # Filter to keep only desired spines
            keep_cols = [i for i in range(len(spine_types)) if _wanted(i)]
            # Build filtered line
            filtered = [cols[i] for i in keep_cols]
            if filtered:
                result_lines.append('\t'.join(filtered))
            continue

        if not initialized:
            # Before spine declaration, keep everything
            result_lines.append(line)
            continue

        # Determine which columns to keep based on current spine_types
        keep_cols = [i for i in range(min(len(spine_types), len(cols)))
                     if _wanted(i)]

        # Handle spine manipulators
        if line.startswith('*'):
            spine_types, spine_origins = _advance_spines(
                cols, spine_types, spine_origins)
            # NOTE: Don't rebuild keep_cols here!
            # The original keep_cols (from before spine operations) is correct
            # for filtering the CURRENT line. The updated spine_types will
            # affect the NEXT line's keep_cols determination.

        # Filter columns
        kept_idx = [i for i in keep_cols if i < len(cols)]
        filtered = [cols[i] for i in kept_idx]

        # Dropping a column that separated two *v groups fuses them into
        # one run, which Humdrum reads as a single all-into-one merge —
        # converter21's spine tracking then derails.  Re-separate such
        # groups onto their own lines (input-adjacent *v cells stay one
        # group; input gaps mark group boundaries).
        if line.startswith('*') and '*v' in filtered:
            fused = any(
                filtered[p] == '*v' and filtered[p + 1] == '*v'
                and kept_idx[p + 1] != kept_idx[p] + 1
                for p in range(len(filtered) - 1)
            )
            if fused:
                groups: List[List[int]] = []
                for p, c in enumerate(filtered):
                    if c != '*v':
                        continue
                    if (groups and filtered[p - 1] == '*v'
                            and kept_idx[p] == kept_idx[p - 1] + 1
                            and groups[-1][-1] == p - 1):
                        groups[-1].append(p)
                    else:
                        groups.append([p])
                shift = 0
                for gi, grp in enumerate(groups):
                    if gi == 0:
                        # First line keeps every non-*v cell's content.
                        row = ['*' if (c == '*v' and p not in grp) else c
                               for p, c in enumerate(filtered)]
                    else:
                        width = len(filtered) - shift
                        row = ['*'] * width
                        for p in grp:
                            row[p - shift] = '*v'
                    result_lines.append('\t'.join(row))
                    shift += len(grp) - 1
                continue

        if filtered:
            result_lines.append('\t'.join(filtered))

    return '\n'.join(result_lines)


def fold_extra_spines(content: str) -> str:
    """Fold every **kern spine above the second one into the second one.

    A score engraved on three or more staves has no place to go in a
    two-hand canon: the lowest staff keeps its own spine and everything
    above it becomes voices of the upper spine.  The columns themselves
    never move — the folded staves already sit immediately to the right,
    so a head *^ and a tail *v run are the whole edit and every interior
    manipulator keeps working on the position it already had.
    """
    lines = content.split('\n')
    decl_index = next(
        (i for i, line in enumerate(lines) if line.startswith('**')), None)
    if decl_index is None:
        return content
    decl = lines[decl_index].split('\t')
    kern_cols = [i for i, cell in enumerate(decl) if cell == '**kern']
    if len(kern_cols) <= 2:
        return content

    folded = kern_cols[1:]
    keep = folded[0]
    dropped = set(folded[1:])

    out: List[str] = lines[:decl_index]
    new_decl = [c for i, c in enumerate(decl) if i not in dropped]
    out.append('\t'.join(new_decl))
    # One split per folded staff, each opening the column that staff
    # already occupies, so the body's width is restored before it starts.
    lane = [i for i, c in enumerate(decl) if i not in dropped].index(keep)
    for step in range(len(dropped)):
        width = len(new_decl) + step
        out.append('\t'.join(
            '*^' if i == lane + step else '*' for i in range(width)))

    spine_types = decl.copy()
    spine_origins = list(range(len(decl)))
    for line in lines[decl_index + 1:]:
        cols = line.split('\t')
        if line.strip() and '\t' in line and cols[0] == '*-':
            run = [i for i, origin in enumerate(spine_origins)
                   if origin in set(folded)]
            out.append('\t'.join(
                '*v' if i in run else '*'
                for i in range(len(spine_origins))))
            out.append('\t'.join(
                ['*-'] * (len(spine_origins) - len(run) + 1)))
            continue
        out.append(line)
        if line.strip() and '\t' in line and line.startswith('*'):
            spine_types, spine_origins = _advance_spines(
                cols, spine_types, spine_origins)
    return '\n'.join(out)


def _cue_token_to_rest(token: str) -> str:
    """Convert a cue data token to a rest with the same duration.

    Handles chords (space-separated notes) by keeping only the first
    note's duration. Grace notes become grace rests.

    Examples:
        16gg  → 16r
        32bb- → 32r
        [4cc  → 4r        (strip tie — rest doesn't tie)
        8ee 8eee → 8r     (chord → single rest)
        16cc#q → 16qr     (grace note)
        272%3e/ → 272%3r   (preserve tuplet ratio)
        8..E] → 8..r      (keep every dot — a lost dot desyncs spine timing)
    """
    if not token or token == '.':
        return token
    # Take first note if chord
    first = token.split()[0] if ' ' in token else token
    # Strip prefix markers (tie, slur, phrase) and stem ink: stems now
    # survive Phase 1, and a stem mark riding before the duration digits
    # would otherwise make the token "unparseable" and silently keep the
    # cue note as a real one.
    clean = first.lstrip('[({<>/\\')
    # Extract duration including dots and tuplet ratio (%N)
    m = re.match(r'(\d+\.*(?:%\d+)?)', clean)
    if not m:
        return token  # unparseable, keep as-is
    dur = m.group(1)
    # Check for grace note (q/Q after duration)
    rest_after = clean[len(dur):]
    if 'q' in rest_after.lower():
        return f'{dur}qr'
    return f'{dur}r'


def strip_cue_passages(sequence: str) -> str:
    """Remove cue passage content from kern sequence.

    Cue notes (*cue...*Xcue) are editorial annotations showing another
    voice's part during rests. They produce no audio in MIDI synthesis.

    Tracks per-column cue state through spine operations (*^ split, *v join)
    so that *cue/*Xcue pairing works even when column indices shift.

    When **dynam spines are present and every **kern column is cue-active
    (the whole instrument is silent), dynam tokens on those lines are
    cleared too — they describe the cue's music, and left in place they
    would set the velocity the instrument re-enters at.  While any kern
    column still plays, dynam marks belong to the playing hand (one dynam
    spine serves the whole grand staff) and are kept.
    """
    lines = sequence.split('\n')

    cue_active: list = []  # per-column bool: is cue active?
    spine_types: list = []  # per-column exclusive interpretation (**kern etc.)
    cue_map: dict = {}     # line_idx -> set of cue column indices
    dynam_clear: dict = {}  # line_idx -> list of dynam column indices to clear
    cue_open_lines: list = []   # (line_idx, col) where *cue opens
    xcue_close_lines: list = [] # (line_idx, col) where *Xcue closes

    for i, line in enumerate(lines):
        if line.startswith('**'):
            spine_types = line.split('\t')
            cue_active = [False] * len(spine_types)
            continue

        if not line.startswith('*') or not line.strip():
            # Data / barline / comment: record current cue state
            if any(cue_active):
                cue_map[i] = {c for c, a in enumerate(cue_active) if a}
                kern_cols = [c for c, st in enumerate(spine_types)
                             if st == '**kern' and c < len(cue_active)]
                if kern_cols and all(cue_active[c] for c in kern_cols):
                    dynam_cols = [c for c, st in enumerate(spine_types)
                                  if st == '**dynam']
                    if dynam_cols:
                        dynam_clear[i] = dynam_cols
            continue

        tokens = line.split('\t')

        # Process *cue / *Xcue BEFORE spine operations
        for si, tok in enumerate(tokens):
            if si >= len(cue_active):
                break
            if re.match(r'^\*cue\b', tok):
                cue_active[si] = True
                cue_open_lines.append((i, si))
            elif re.match(r'^\*Xcue\b', tok):
                if cue_active[si]:
                    cue_active[si] = False
                else:
                    # *Xcue on a non-cue column (spine ops shifted indices).
                    # Close the nearest cue-active column instead.
                    best = None
                    best_dist = len(cue_active) + 1
                    for c, a in enumerate(cue_active):
                        if a and abs(c - si) < best_dist:
                            best = c
                            best_dist = abs(c - si)
                    if best is not None:
                        cue_active[best] = False
                xcue_close_lines.append((i, si))

        # Record tandem lines that have active cue or cue/Xcue markers
        active_cols = {c for c, a in enumerate(cue_active) if a}
        marker_cols = {si for si, tok in enumerate(tokens)
                       if re.match(r'^\*(cue|Xcue)\b', tok)}
        all_cols = active_cols | marker_cols
        if all_cols:
            if i not in cue_map:
                cue_map[i] = set()
            cue_map[i].update(all_cols)

        # Handle spine operations: *^ (split) and *v (join)
        has_split = any(t == '*^' for t in tokens)
        has_join = any(t == '*v' for t in tokens)

        if has_split:
            new_cue = []
            new_types = []
            for si, tok in enumerate(tokens):
                val = cue_active[si] if si < len(cue_active) else False
                st = spine_types[si] if si < len(spine_types) else '**kern'
                new_cue.append(val)
                new_types.append(st)
                if tok == '*^':
                    new_cue.append(val)  # both sub-spines inherit cue
                    new_types.append(st)
            cue_active = new_cue
            spine_types = new_types
        elif has_join:
            new_cue = []
            new_types = []
            si = 0
            while si < len(tokens):
                val = cue_active[si] if si < len(cue_active) else False
                st = spine_types[si] if si < len(spine_types) else '**kern'
                if tokens[si] == '*v':
                    merged = val
                    while si + 1 < len(tokens) and tokens[si + 1] == '*v':
                        si += 1
                        if si < len(cue_active):
                            merged = merged or cue_active[si]
                    new_cue.append(merged)
                else:
                    new_cue.append(val)
                new_types.append(st)
                si += 1
            cue_active = new_cue
            spine_types = new_types

    if not cue_map:
        return sequence

    # Expand: scan backwards from *cue for *rscale on same column
    for cue_line, cue_col in cue_open_lines:
        for j in range(max(0, cue_line - 5), cue_line):
            ln = lines[j]
            if not ln.startswith('*') or ln.startswith('**'):
                continue
            toks = ln.split('\t')
            if cue_col < len(toks) and re.match(r'^\*rscale:', toks[cue_col]):
                if j not in cue_map:
                    cue_map[j] = set()
                cue_map[j].add(cue_col)

    # Expand: scan forwards from *Xcue for *rscale on same column
    for xcue_line, xcue_col in xcue_close_lines:
        for j in range(xcue_line + 1, min(len(lines), xcue_line + 6)):
            ln = lines[j]
            if not ln.startswith('*') or ln.startswith('**'):
                continue
            toks = ln.split('\t')
            if xcue_col < len(toks) and re.match(r'^\*rscale:', toks[xcue_col]):
                if j not in cue_map:
                    cue_map[j] = set()
                cue_map[j].add(xcue_col)

    # --- Pass 2: Strip cue content ---
    for i, line in enumerate(lines):
        if i not in cue_map:
            continue
        cue_spines = cue_map[i]

        if line.startswith('*') and not line.startswith('**'):
            tokens = line.split('\t')
            new_tokens = list(tokens)
            for si in cue_spines:
                if si < len(tokens):
                    tok = tokens[si]
                    if re.match(r'^\*(cue|Xcue|rscale:)\b', tok):
                        new_tokens[si] = '*'
            lines[i] = '\t'.join(new_tokens)

        elif not line.startswith(('=', '!')) and line.strip():
            # Data line: replace cue tokens with rests of same duration.
            # Rest fills the time slot correctly (matching silence in audio).
            tokens = line.split('\t')
            for si in cue_spines:
                if si < len(tokens) and tokens[si] != '.':
                    tokens[si] = _cue_token_to_rest(tokens[si])
            for dc in dynam_clear.get(i, []):
                if dc < len(tokens) and tokens[dc] not in ('.', ''):
                    tokens[dc] = '.'
            lines[i] = '\t'.join(tokens)

    return '\n'.join(lines)


def clean_kern_sequence(
    sequence: str,
    warn_tuplet_ratio: bool = True,
    strip_cue: bool = True,
) -> str:
    """
    Clean an entire Kern sequence: visual-only cleanup.

    Phase 1 cleanup: strips visual markers (beam, editorial, rest
    position, unicode) from all tokens; stem-direction marks on notes
    stay for standardize_kern to consume. Preserves musical content
    (articulation, ties, duration values). Content-level decisions belong
    in Phase 1.5 (standardize_kern).

    Args:
        sequence: Tab-separated or newline-separated Kern tokens
        warn_tuplet_ratio: Whether to log warning for tuplet ratio notation (%)
        strip_cue: Whether to strip *cue...*Xcue passages

    Returns:
        Cleaned sequence with same structure
    """
    if strip_cue:
        sequence = strip_cue_passages(sequence)

    if warn_tuplet_ratio and '%' in sequence:
        for line in sequence.split('\n'):
            for token in line.split('\t'):
                check_tuplet_ratio_notation(token)

    # Data cells are cleaned per spine type: clean_kern_token's kern
    # signifiers (stem /\, editorial Xyz) collide with **dynam content
    # (a lone "/" cell would be emptied into an illegal blank token).
    return _map_kern_cells(sequence, clean_kern_token,
                           control_fn=clean_kern_token)


def _map_kern_cells(content: str, cell_fn, control_fn=None) -> str:
    """Apply cell_fn to **kern data cells only, leaving other spines alone.

    Spine-type tracking mirrors strip_spines (*^ split, *v join,
    *- terminate).  Needed because kern signifier letters collide with the
    **dynam vocabulary (p, <, >): a column-blind strip would delete
    dynamics when dynam spines are retained.

    control_fn, when given, is applied to every cell of comment (!),
    barline (=), and interpretation (*) lines regardless of spine type —
    those cleanups (layout-hint comments, clef editorial marks) are
    column-agnostic.
    """
    def map_control(line: str) -> str:
        if control_fn is None:
            return line
        return '\t'.join(control_fn(c) for c in line.split('\t'))

    lines = content.split('\n')
    result = []
    spine_types: List[str] = []
    initialized = False
    for line in lines:
        if line.startswith('**'):
            spine_types = line.split('\t')
            initialized = True
            result.append(line)
            continue
        if not initialized or not line.strip():
            result.append(line)
            continue
        if line.startswith(('!', '=')):
            result.append(map_control(line))
            continue
        cols = line.split('\t')
        if line.startswith('*'):
            new_spine_types = []
            src_idx = 0
            i = 0
            while i < len(cols):
                if src_idx >= len(spine_types):
                    break
                col = cols[i]
                if col == '*^':
                    new_spine_types.append(spine_types[src_idx])
                    new_spine_types.append(spine_types[src_idx])
                    src_idx += 1
                    i += 1
                elif col == '*v':
                    v_count = 1
                    while i + v_count < len(cols) and cols[i + v_count] == '*v':
                        v_count += 1
                    new_spine_types.append(spine_types[src_idx])
                    src_idx += v_count
                    i += v_count
                elif col == '*-':
                    src_idx += 1
                    i += 1
                else:
                    new_spine_types.append(spine_types[src_idx])
                    src_idx += 1
                    i += 1
            spine_types = new_spine_types
            result.append(map_control(line))
            continue
        new_cols = []
        for i, c in enumerate(cols):
            st = spine_types[i] if i < len(spine_types) else '**kern'
            if st == '**kern' and c not in ('.', ''):
                new_cols.append(cell_fn(c))
            else:
                new_cols.append(c)
        result.append('\t'.join(new_cols))
    return '\n'.join(result)


# Trill signifiers per the **kern spec: t/T. Split from the rest of the
# ornament family because VirtuosoNet's score parser tracks trill-mark as
# a real input feature (is_trill), so this one can change VirtuosoNet-
# rendered audio.
_TRILL_PATTERN = re.compile(r'[Tt]')

# Non-trill ornament signifiers per the **kern spec: mordent (M/m),
# inverted mordent (W/w), turn (S/s/$), generic (O). VirtuosoNet's parser
# discards mordent after parsing it and never parses turn at all, so
# toggling this cannot change VirtuosoNet-rendered audio.
#
# Turn is not simply uppercase S: converter21's humdrum reader
# (_addTurn) treats any maximal run of 2+ characters from {s, S, $} as a
# turn (e.g. plain "ss"), and a lone uppercase S or $ also counts — only
# a lone lowercase s (spiccato, unrelated articulation, stays in
# strip_articulation's table) is not a turn. [sS$]{2,}|[S$] mirrors that
# exactly: runs of 2+ are always a turn; a single S or $ is a turn; a
# single lowercase s is left alone.
_NON_TRILL_ORNAMENT_PATTERN = re.compile(r'[MmWwO]|[sS$]{2,}|[S$]')
_ARPEGGIO_PATTERN = re.compile(r':')
_GRACE_DESIGNATOR_PATTERN = re.compile(r'[pP]')


def strip_trill_marks(content: str) -> str:
    """Strip trill signifiers (t/T) from **kern cells."""
    return _map_kern_cells(content, lambda c: _TRILL_PATTERN.sub('', c))


def strip_non_trill_ornament_marks(content: str) -> str:
    """Strip non-trill ornament signifiers (mordent/turn family) from **kern cells."""
    return _map_kern_cells(content, lambda c: _NON_TRILL_ORNAMENT_PATTERN.sub('', c))


def strip_arpeggio_marks(content: str) -> str:
    """Strip arpeggiation marks (:) from **kern cells."""
    return _map_kern_cells(content, lambda c: _ARPEGGIO_PATTERN.sub('', c))


def strip_grace_designators(content: str) -> str:
    """Strip appoggiatura designators from **kern cells.

    P marks the appoggiatura, p the note it resolves onto (per the **kern
    spec).  Companion to strip_grace_notes, which removes the grace tokens
    themselves; without the grace notes these designators are dangling.
    """
    return _map_kern_cells(content, lambda c: _GRACE_DESIGNATOR_PATTERN.sub('', c))


def strip_articulation(sequence: str) -> str:
    """Strip articulation and stem marks from a kern sequence.

    Removes: s p P ' ~ ^ > < ; ` x v u & @ i j Z + ! | / \\
    Stem marks ride along: they carry no decision weight anywhere
    downstream, and left in place they can sit between the pitch letters
    and the accidental (e.g. A\\-) and derail pitch parsing.
    (Ornament signifiers and arpeggio marks are handled separately by
    strip_trill_marks / strip_non_trill_ornament_marks / strip_arpeggio_marks.)

    This is a Phase 1.5 (standardize) operation — call it when the training
    target does not include articulation tokens.
    """
    pattern = re.compile(r"[spP'~^><;`xvu&@ijZ+!|/\\]")
    lines = sequence.split('\n')
    result = []
    for line in lines:
        if line.startswith(('=', '*', '!')) or not line.strip():
            result.append(line)
            continue
        if '\t' in line:
            tokens = line.split('\t')
            result.append('\t'.join(pattern.sub('', t) if t != '.' else t for t in tokens))
        else:
            result.append(pattern.sub('', line) if line != '.' else line)
    return '\n'.join(result)


def strip_ties(sequence: str) -> str:
    """Strip tie markers ([ ] _) from a kern sequence.

    This is a Phase 1.5 (standardize) operation — only call when the
    training target does not use ties.
    """
    pattern = re.compile(r'[\[\]_]')
    lines = sequence.split('\n')
    result = []
    for line in lines:
        if line.startswith(('=', '*', '!')) or not line.strip():
            result.append(line)
            continue
        if '\t' in line:
            tokens = line.split('\t')
            result.append('\t'.join(pattern.sub('', t) if t != '.' else t for t in tokens))
        else:
            result.append(pattern.sub('', line) if line != '.' else line)
    return '\n'.join(result)


def scan_dirty_tokens(data_dir: str, top_n: int = 50) -> List[Tuple[str, int]]:
    """
    Scan a corpus of .krn files and identify tokens that may cause KeyError.

    This is a diagnostic tool to help you understand what "dirty" tokens
    converter21 is producing in your specific dataset.

    Args:
        data_dir: Path to directory containing .krn files
        top_n: Number of top dirty tokens to return

    Returns:
        List of (token, count) tuples, sorted by frequency

    A "dirty" token is defined as one that contains:
    - Rest position markers (rGG, rCC, etc.)
    - Stem direction (/, \\)
    - Articulation marks (', ~, ^, etc.)
    """
    dirty_tokens = Counter()

    # Pattern for "clean" tokens (only semantic information)
    # Allowed: digits, dot (dotted note), r (rest), A-G/a-g (pitch),
    #          # (sharp), - (flat), n (natural), _ [ ] (ties)
    clean_pattern = re.compile(r'^[0-9\.]+[rA-Ga-g#\-n_\[\]]+$')

    data_path = Path(data_dir)
    krn_files = list(data_path.rglob('*.krn'))

    print(f"Scanning {len(krn_files)} .krn files in {data_dir}...")

    for krn_file in krn_files:
        try:
            with open(krn_file, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    # Skip control lines
                    if line.startswith(('!', '*', '=')):
                        continue

                    # Split by tabs
                    tokens = line.strip().split('\t')
                    for token in tokens:
                        if token in ('.', '', 'q'):
                            continue

                        # Check if token is "dirty"
                        if not clean_pattern.match(token):
                            dirty_tokens[token] += 1
        except Exception as e:
            print(f"Error reading {krn_file}: {e}")

    return dirty_tokens.most_common(top_n)


def generate_cleaning_report(data_dir: str, output_file: str = 'kern_cleaning_report.txt'):
    """
    Generate a comprehensive report of dirty tokens found in the corpus.

    Args:
        data_dir: Path to directory containing .krn files
        output_file: Path to save the report
    """
    dirty_tokens = scan_dirty_tokens(data_dir, top_n=100)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Kern Token Cleaning Report (converter21 Output)\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total dirty token types found: {len(dirty_tokens)}\n\n")

        # Categorize by type
        rest_position = []
        stem_direction = []
        articulation = []
        other = []

        for token, count in dirty_tokens:
            if 'r' in token and re.search(r'r[A-Ga-g]+', token):
                rest_position.append((token, count))
            elif '/' in token or '\\' in token:
                stem_direction.append((token, count))
            elif re.search(r"[';~^:`]", token):
                articulation.append((token, count))
            else:
                other.append((token, count))

        f.write(f"Category Breakdown:\n")
        f.write(f"  Rest position markers: {len(rest_position)}\n")
        f.write(f"  Stem direction: {len(stem_direction)}\n")
        f.write(f"  Articulation marks: {len(articulation)}\n")
        f.write(f"  Other: {len(other)}\n\n")

        f.write("="*70 + "\n")
        f.write("Top 50 Dirty Tokens (by frequency):\n")
        f.write("="*70 + "\n\n")

        for i, (token, count) in enumerate(dirty_tokens[:50], 1):
            cleaned = clean_kern_token(token)
            f.write(f"{i:3d}. {token:20s} ({count:6d} occurrences) -> {cleaned}\n")

    print(f"Report saved to: {output_file}")


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean Kern tokens from converter21 output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan corpus and generate report
  python clean_kern.py --scan workspace/feature.asap/test/kern_upper

  # Clean a single token
  python clean_kern.py --token "8rGG"

  # Clean a file
  python clean_kern.py --file input.krn --output cleaned.krn
        """
    )

    parser.add_argument('--scan', type=str, metavar='DIR',
                        help='Scan directory for dirty tokens')
    parser.add_argument('--token', type=str, metavar='TOKEN',
                        help='Clean a single token')
    parser.add_argument('--file', type=str, metavar='FILE',
                        help='Clean a .krn file')
    parser.add_argument('--output', type=str, metavar='FILE',
                        help='Output file for cleaned content')
    parser.add_argument('--report', type=str, metavar='FILE',
                        default='kern_cleaning_report.txt',
                        help='Output file for scan report (default: kern_cleaning_report.txt)')

    args = parser.parse_args()

    if args.scan:
        generate_cleaning_report(args.scan, args.report)

    elif args.token:
        cleaned = clean_kern_token(args.token)
        print(f"Original: {args.token}")
        print(f"Cleaned:  {cleaned}")

    elif args.file:
        with open(args.file, 'r', encoding='iso-8859-1') as f:
            content = f.read()

        cleaned = clean_kern_sequence(content)

        if args.output:
            with open(args.output, 'w', encoding='iso-8859-1') as f:
                f.write(cleaned)
            print(f"Cleaned file saved to: {args.output}")
        else:
            print(cleaned)

    else:
        parser.print_help()
