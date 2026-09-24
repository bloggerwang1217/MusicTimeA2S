"""Pre-MIDI kern preparation shared between evaluation and inference output.

The helpers here make a sliced kern fragment independently renderable, then
apply the transforms needed before kern → musicxml → MIDI:

1. strip_grace_notes(text): drop any token containing a `q`/`Q` ornament
   marker. Grace notes get exported by music21 as zero-duration notes which
   produce orphan note-off events that MV2H's Java Converter rejects. MV2H
   does not score grace notes, so we strip them on both sides for symmetry.

2. tiefix_kern(text, n_spines): per-spine tie repair following the strategy
   in humextra's tiefix (Zeng's baseline pipeline):
     - Orphan `]`: try to re-pair with the immediately previous same-pitch
       note in the same spine. If pitch doesn't match, drop the `]`.
     - Orphan `[`: try to re-pair with the immediately next same-pitch note
       in the same spine. If no match, drop the `[`.
   This handles slice-boundary ties where one half of the tie was sliced away.
"""
from __future__ import annotations

import re
from typing import List, Set, Tuple

_PITCH_RE = re.compile(r'[a-gA-G]+[#\-]*')
_DATA_LINE_PREFIXES = ('*', '=', '!')
_METER_INTERP_RE = re.compile(r'^\*M\d+/\d+$')
_KEY_INTERP_RE = re.compile(r'^\*k\[[^\]]*\]$')
_SPINE_PATH_TOKENS = frozenset(('*', '*^', '*v', '*x', '*-'))


def _schema_value(line: str, pattern: re.Pattern) -> str | None:
    values = {
        cell.strip()
        for cell in line.rstrip('\r\n').split('\t')
        if pattern.fullmatch(cell.strip())
    }
    if len(values) > 1:
        raise ValueError(f"conflicting schema interpretations: {line.rstrip()}")
    return next(iter(values), None)


def _schema_kind(line: str) -> str | None:
    cells = [
        cell.strip()
        for cell in line.rstrip('\r\n').split('\t')
        if cell.strip() not in ('', '*')
    ]
    if cells and all(_METER_INTERP_RE.fullmatch(cell) for cell in cells):
        return 'meter'
    if cells and all(_KEY_INTERP_RE.fullmatch(cell) for cell in cells):
        return 'key'
    return None


def _active_spine_path_rows(
    lines: list[str], header_end: int, start_line: int,
) -> list[str]:
    """Return path rows needed to recreate the topology at ``start_line``."""
    base_width = len(lines[header_end].rstrip('\r\n').split('\t'))
    width = base_width
    replay: list[str] = []
    can_reset_at_base = True

    for line in lines[header_end:start_line]:
        cells = line.rstrip('\r\n').split('\t')
        if not cells or not all(cell.strip() in _SPINE_PATH_TOKENS for cell in cells):
            continue
        if all(cell.strip() == '*' for cell in cells):
            continue
        if len(cells) != width:
            raise ValueError(
                f"spine-path row has {len(cells)} fields, expected {width}: "
                f"{line.rstrip()}"
            )

        tokens = [cell.strip() for cell in cells]
        width += tokens.count('*^')
        width -= tokens.count('*-')

        i = 0
        while i < len(tokens):
            if tokens[i] != '*v':
                i += 1
                continue
            j = i + 1
            while j < len(tokens) and tokens[j] == '*v':
                j += 1
            if j - i < 2:
                raise ValueError(f"unpaired *v in spine-path row: {line.rstrip()}")
            width -= (j - i) - 1
            i = j

        if tokens.count('*x') % 2:
            raise ValueError(f"unpaired *x in spine-path row: {line.rstrip()}")
        if '*x' in tokens or '*-' in tokens:
            can_reset_at_base = False

        replay.append(line)
        # Split/merge histories that return to the original topology carry no
        # state into a later bar-aligned slice.
        if can_reset_at_base and width == base_width:
            replay.clear()

    opening_width = len(lines[start_line].rstrip('\r\n').split('\t'))
    if width != opening_width:
        raise ValueError(
            f"active spine width {width} does not match opening barline width "
            f"{opening_width}"
        )
    return replay


def slice_kern_for_midi(text: str, start_line: int, end_line: int) -> str:
    """Return a bar-aligned slice carrying active topology and schema state.

    ``start_line`` and ``end_line`` are zero-based Python slice bounds, and
    ``start_line`` must name the opening barline.  Kern interpretations are
    stateful, so copying only the file preamble makes a mid-piece slice inherit
    the wrong default whenever the active schema was declared before the
    window.  The selected bar's own declarations take precedence.
    """
    lines = text.splitlines(keepends=True)
    if not (0 <= start_line < end_line <= len(lines)):
        raise ValueError(
            f"invalid kern slice [{start_line}:{end_line}] for {len(lines)} lines"
        )

    opening = lines[start_line]
    opening_cells = opening.rstrip('\r\n').split('\t')
    if not opening_cells[0].strip().startswith('='):
        raise ValueError(f"kern slice does not start at a barline: {opening.rstrip()}")

    header_end = next(
        (
            i for i, line in enumerate(lines)
            if line.split('\t', 1)[0].strip().startswith('=')
        ),
        None,
    )
    if header_end is None:
        raise ValueError("kern has no barline")

    # Include schema declared at the selected bar head, but stop before its
    # first data row or the following barline if the bar is empty.
    initial_end = start_line + 1
    while initial_end < end_line:
        first = lines[initial_end].split('\t', 1)[0].strip()
        if first.startswith('=') or (
            first and not first.startswith(_DATA_LINE_PREFIXES)
        ):
            break
        initial_end += 1

    active_meter = None
    active_key = None
    for line in lines[:initial_end]:
        meter = _schema_value(line, _METER_INTERP_RE)
        key = _schema_value(line, _KEY_INTERP_RE)
        if meter is not None:
            active_meter = meter
        if key is not None:
            active_key = key

    # Keep only non-stateful file metadata in the preamble.  Spine paths must
    # precede the opening barline so its field count has a valid ancestry.
    header = [line for line in lines[:header_end] if _schema_kind(line) is None]
    spine_path_rows = _active_spine_path_rows(lines, header_end, start_line)
    initial_rows = [
        line for line in lines[start_line + 1:initial_end]
        if _schema_kind(line) is None
    ]
    n_spines = len(opening_cells)
    schema_rows: List[str] = []
    if active_meter is not None:
        schema_rows.append('\t'.join([active_meter] * n_spines) + '\n')
    if active_key is not None:
        schema_rows.append('\t'.join([active_key] * n_spines) + '\n')

    return ''.join(
        header
        + spine_path_rows
        + [opening]
        + schema_rows
        + initial_rows
        + lines[initial_end:end_line]
    )


def _is_data_line(line: str) -> bool:
    s = line.lstrip()
    return bool(s) and not s.startswith(_DATA_LINE_PREFIXES)


def strip_grace_notes(text: str) -> str:
    """Drop any token containing `q` or `Q` from data rows.

    Grace tokens take several humdrum forms (`eq8`, `dd#32q`, `q8e`, ...).
    All are ornaments that MV2H does not evaluate.
    """
    out_lines: List[str] = []
    for line in text.split('\n'):
        if not _is_data_line(line):
            out_lines.append(line)
            continue
        cells = line.split('\t')
        new_cells: List[str] = []
        for cell in cells:
            if not cell or cell == '.':
                new_cells.append(cell if cell else '.')
                continue
            kept = [tok for tok in cell.split() if 'q' not in tok and 'Q' not in tok]
            new_cells.append(' '.join(kept) if kept else '.')
        out_lines.append('\t'.join(new_cells))
    return '\n'.join(out_lines)


def _extract_pitch(tok: str) -> str:
    """Return first pitch+accidental in a kern token, or '' if no pitch."""
    m = _PITCH_RE.search(tok)
    return m.group(0) if m else ''


def tiefix_kern(text: str, n_spines: int = 2) -> str:
    """Per-spine tie repair following humextra/tiefix.cpp.

    Walks each spine in order, tracking open ties on a stack. Orphan opens
    and closes are first attempted to be re-paired with the immediately
    adjacent same-pitch note; if that fails, the bracket is dropped.
    """
    out_lines = text.split('\n')

    # Per spine: list of (line_idx, cell_idx, tok_idx, pitch, has_open, has_close)
    notes_per_spine: List[List[Tuple[int, int, int, str, bool, bool]]] = [[] for _ in range(n_spines)]

    for li, line in enumerate(out_lines):
        if not _is_data_line(line):
            continue
        cells = line.split('\t')
        for ci in range(min(n_spines, len(cells))):
            cell = cells[ci]
            if not cell or cell == '.':
                continue
            for ti, tok in enumerate(cell.split()):
                pitch = _extract_pitch(tok)
                if not pitch:
                    continue
                notes_per_spine[ci].append(
                    (li, ci, ti, pitch, '[' in tok, ']' in tok)
                )

    # Modifications keyed by (li, ci, ti). Values: set of mod tags.
    mods: dict = {}

    def mark(key, tag):
        mods.setdefault(key, set()).add(tag)

    # The decision logic below mirrors humextra/tiefix.cpp lines 320-407
    # branch-for-branch (see https://github.com/craigsapp/humextra). Each
    # comment cites the line(s) it implements so the correspondence is
    # auditable.
    for spine_notes in notes_per_spine:
        n = len(spine_notes)
        stack: List[int] = []  # indices of pending opens (TIESTART)
        for idx, (li, ci, ti, pitch, has_open, has_close) in enumerate(spine_notes):
            if has_close:
                # First try to match a prior open of the same pitch (LIFO).
                # This is the normal "balanced tie" path; tiefix only fires
                # the orphan branch when no matching open exists.
                matched = False
                for j in range(len(stack) - 1, -1, -1):
                    si = stack[j]
                    if spine_notes[si][3] == pitch:
                        stack.pop(j)
                        matched = True
                        break
                if not matched:
                    # Orphan TIESTOP. Mirrors tiefix.cpp:385-405.
                    if idx == 0:
                        # cpp:386-389 — at first position: strip
                        mark((li, ci, ti), 'rm_close')
                    else:
                        prev = spine_notes[idx - 1]
                        prev_tied = prev[4] or prev[5]
                        if prev[3] == pitch and not prev_tied:
                            # cpp:391-394 — re-pair: prev becomes TIESTART
                            mark((prev[0], prev[1], prev[2]), 'add_open')
                        elif prev[3] != pitch:
                            # cpp:395-398 — pitch differs: strip
                            mark((li, ci, ti), 'rm_close')
                        # else: same pitch but prev already tied → do nothing
                        # (matches the cpp:399-404 TIECONT branch's no-op
                        # for our subset; we don't model the `_` continuation
                        # marker.)
            if has_open:
                stack.append(idx)
        # Remaining stack entries are orphan TIESTARTs. Mirrors cpp:321-335.
        for si in stack:
            li, ci, ti, pitch, _, _ = spine_notes[si]
            if si == n - 1:
                # cpp:322-324 — at last position: strip
                mark((li, ci, ti), 'rm_open')
            else:
                nxt = spine_notes[si + 1]
                nxt_tied = nxt[4] or nxt[5]
                if nxt[3] == pitch and not nxt_tied:
                    # cpp:326-329 — re-pair: next becomes TIESTOP
                    mark((nxt[0], nxt[1], nxt[2]), 'add_close')
                elif nxt[3] != pitch:
                    # cpp:330-333 — pitch differs: strip
                    mark((li, ci, ti), 'rm_open')
                # else: pitch matches but next already tied → do nothing
                # (cpp falls through silently in this case)

    if not mods:
        return text

    # Apply modifications
    new_lines = list(out_lines)
    touched_lines: Set[int] = {k[0] for k in mods}
    for li in touched_lines:
        line = out_lines[li]
        if not _is_data_line(line):
            continue
        cells = line.split('\t')
        for ci in range(min(n_spines, len(cells))):
            cell = cells[ci]
            if not cell or cell == '.':
                continue
            toks = cell.split()
            for ti, tok in enumerate(toks):
                tags = mods.get((li, ci, ti))
                if not tags:
                    continue
                if 'rm_open' in tags and tok.startswith('['):
                    tok = tok[1:]
                if 'rm_close' in tags and tok.endswith(']'):
                    tok = tok[:-1]
                if 'add_open' in tags and not tok.startswith('['):
                    tok = '[' + tok
                if 'add_close' in tags and not tok.endswith(']'):
                    tok = tok + ']'
                toks[ti] = tok
            cells[ci] = ' '.join(toks) if toks else '.'
        new_lines[li] = '\t'.join(cells)

    return '\n'.join(new_lines)


def kern_pre_midi_clean(text: str, n_spines: int = 2) -> str:
    """Apply all pre-MIDI cleanups: grace strip → tiefix."""
    text = strip_grace_notes(text)
    text = tiefix_kern(text, n_spines=n_spines)
    return text
