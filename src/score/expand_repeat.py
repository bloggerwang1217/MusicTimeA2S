"""
Repeat Expansion for Kern and Music21 Scores
=============================================

Consolidates all repeat-related functionality:

1. **Kern-level expansion** (Humdrum expansion labels ``*>[A,A,B,...]``):
   - ``expand_kern_repeats()`` -- expand kern content to through-composed
   - ``expand_kern_repeats_with_mapping()`` -- expand + build repeat_map

2. **Music21-level expansion** (MuseSyn MusicXML):
   - ``expand_musesyn_score()`` -- expand Score + detect repeats
   - ``extract_repeat_structure()`` -- extract barlines / DaCapo / volta
   - ``build_musesyn_repeat_map()`` -- build rich repeat_map from Score

3. **Utilities**:
   - ``has_expansion_labels()``, ``parse_expansion_order()``,
     ``parse_section_ranges()``, ``remove_repeat_barlines()``,
     ``get_expansion_info()``
"""

import copy
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import music21 as m21

from src.score.sanitize_kern import extract_kern_measures

logger = logging.getLogger(__name__)

# ============================================================================
# Kern-level utilities
# ============================================================================


def has_expansion_labels(kern_content: str) -> bool:
    """Check if kern content has Humdrum expansion labels.

    Args:
        kern_content: Raw kern file content

    Returns:
        True if expansion labels (*>[...]) are present
    """
    return bool(re.search(r'\*>\[[^\]]+\]', kern_content))


def parse_expansion_order(kern_content: str) -> Optional[List[str]]:
    """Extract the expansion order from kern content.

    Args:
        kern_content: Raw kern file content

    Returns:
        List of section names in playback order, or None if no expansion labels

    Example:
        ``"*>[A,A,B,B1,B,B2]"`` -> ``["A", "A", "B", "B1", "B", "B2"]``
    """
    match = re.search(r'\*>\[([^\]]+)\]', kern_content)
    if not match:
        return None
    return [s.strip() for s in match.group(1).split(',')]


def parse_section_ranges(kern_content: str) -> Dict[str, Tuple[int, int]]:
    """Parse section markers and their line ranges.

    Args:
        kern_content: Raw kern file content

    Returns:
        Dictionary mapping section name to (start_line, end_line) tuple.
        Lines are 0-indexed, start is inclusive, end is exclusive.

    Example:
        ``{"A": (15, 69), "B": (70, 104), "B1": (105, 118), ...}``
    """
    lines = kern_content.split('\n')
    section_ranges: Dict[str, Tuple[int, int]] = {}
    current_section: Optional[str] = None
    current_start: int = 0

    for i, line in enumerate(lines):
        # Section marker: *>A\t*>A or *>A\t*>A\t*>A (one per spine).
        # A spine with no section identity of its own sits the line out
        # with a plain '*' — **dynam does, and requiring every column to
        # carry the label silently dropped those files' repeats.
        # But NOT expansion labels (*>[...]) or norep labels (*>norep[...])
        if line.startswith('*>') and '\t' in line:
            parts = line.split('\t')
            is_section_marker = all(
                (p == '*' or p.startswith('*>')) and
                not p.startswith('*>[') and
                not p.startswith('*>norep')
                for p in parts
            )
            if is_section_marker:
                section = parts[0][2:]  # Extract "A" from "*>A"
                if section:
                    if current_section is not None:
                        section_ranges[current_section] = (current_start, i)
                    current_section = section
                    current_start = i + 1

    if current_section is not None:
        end = len(lines)
        while end > current_start and lines[end - 1].startswith('!!!'):
            end -= 1
        section_ranges[current_section] = (current_start, end)

    return section_ranges


def _clean_repeat_barline(line: str) -> str:
    """Remove repeat markers from barline while preserving measure number.

    Humdrum repeat barlines look like:
        ``=5:|!|:``  -> ``=5``
        ``=:|!|:``   -> ``=``
        ``=10!|:``   -> ``=10``
    """
    if not line.startswith('='):
        return line

    parts = line.split('\t')
    cleaned_parts = []

    for part in parts:
        if part.startswith('='):
            cleaned = re.sub(r'[:\|!]+', '', part)
            if not cleaned.startswith('='):
                cleaned = '=' + cleaned.lstrip('=')
            cleaned_parts.append(cleaned)
        else:
            cleaned_parts.append(part)

    return '\t'.join(cleaned_parts)


def _line_meter_token(line: str) -> Optional[str]:
    """First ``*M<num>/<den>`` token on the line, any column.

    Meter changes are often declared only in the kern columns
    (``*\\t*M4/4\\t...``); a line-anchored match misses them.
    """
    if '*M' not in line:
        return None
    for p in line.split('\t'):
        if re.match(r'\*M\d+/\d+$', p):
            return p
    return None


def _line_key_token(line: str) -> Optional[str]:
    """First ``*k[...]`` token on the line, any column."""
    if '*k[' not in line:
        return None
    for p in line.split('\t'):
        if p.startswith('*k['):
            return p
    return None


class _ColClock:
    """Column-tracking duration clock over kern lines (whole-note units).

    Follows ``*^``/``*v``/``*x``, takes cell durations from the first
    non-grace subtoken (rational ``a%b`` recips included), and steps by
    the minimum remaining duration across columns.  A spine-0-only sum
    undercounts under ``*^`` and misreads rscale values (``4%5`` as a
    quarter), turning full bars into false partials for the seam merge.
    """

    def __init__(self, ncols: int) -> None:
        from fractions import Fraction as F
        self.t = F(0)
        self.rem: List[F] = [F(0)] * ncols

    def feed(self, line: str) -> None:
        from fractions import Fraction as F
        from src.score.kern_utils import _col_first_dur
        cols = line.split('\t')
        if line.startswith(('=', '!')) or not line.strip():
            return
        if line.startswith('*'):
            if any(c in ('*^', '*v', '*x') for c in cols):
                new_rem: List[F] = []
                i = 0
                while i < len(cols):
                    c = cols[i]
                    r = self.rem[i] if i < len(self.rem) else F(0)
                    if c == '*^':
                        new_rem += [r, r]
                        i += 1
                    elif c == '*v':
                        rs = [r]
                        i += 1
                        while i < len(cols) and cols[i] == '*v':
                            rs.append(self.rem[i] if i < len(self.rem)
                                      else F(0))
                            i += 1
                        new_rem.append(max(rs))
                    elif c == '*x':
                        new_rem += [self.rem[i + 1]
                                    if i + 1 < len(self.rem) else F(0), r]
                        i += 2
                    else:
                        new_rem.append(r)
                        i += 1
                self.rem = new_rem
            return
        for ci, cell in enumerate(cols):
            cell = cell.strip()
            if not cell or cell == '.':
                continue
            d = _col_first_dur(cell) / 4
            if d > 0:
                while ci >= len(self.rem):
                    self.rem.append(F(0))
                self.rem[ci] = d
        positive = [r for r in self.rem if r > 0]
        if positive:
            step = min(positive)
            self.t += step
            self.rem = [r - step if r > step else F(0) for r in self.rem]

    def span(self) -> 'Fraction':
        """Elapsed time including still-sounding tails, non-destructive."""
        from fractions import Fraction as F
        return self.t + (max(self.rem) if self.rem else F(0))

    def flush(self) -> None:
        from fractions import Fraction as F
        if self.rem and max(self.rem) > 0:
            self.t += max(self.rem)
            self.rem = [F(0)] * len(self.rem)


def _barline_times(lines: List[str]) -> Dict[Any, 'Fraction']:
    """Global time (whole notes) at every barline line index, plus 'end'."""
    times: Dict[Any, 'Fraction'] = {}
    clock: Optional[_ColClock] = None
    for li, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith('**'):
            clock = _ColClock(len(line.split('\t')))
            continue
        if clock is None:
            continue
        if line.startswith('='):
            clock.flush()
            times[li] = clock.t
            continue
        clock.feed(line)
    if clock is not None:
        times['end'] = clock.span()
    return times


def _merge_seam_bars(lines: List[str], meter: 'Fraction') -> List[str]:
    """Merge consecutive partial bars that sum to meter.

    After expansion, section seams leave partial bars separated by one
    or more barlines (possibly with empty bars from cleaned repeat
    barlines in between).  This pass finds runs of consecutive bars
    whose total duration equals meter and removes the interior barlines.

    Tracks meter changes (``*M``) so mid-piece modulations are handled.
    """
    from fractions import Fraction as F
    if meter <= 0:
        return lines

    barline_indices = [
        i for i, line in enumerate(lines)
        if line.startswith('=') and not line.startswith('==')
    ]
    if len(barline_indices) < 2:
        return lines

    # Compute duration of each bar (between consecutive barlines).
    # Also track meter changes within bars.
    bl_times = _barline_times(lines)
    all_bl = [i for i, line in enumerate(lines) if line.startswith('=')]
    bar_durs = []  # parallel to barline_indices
    cur_meter = meter
    for bi in range(len(barline_indices)):
        bar_start = barline_indices[bi] + 1
        bar_end = (barline_indices[bi + 1]
                   if bi + 1 < len(barline_indices)
                   else len(lines))
        # Check for meter change within this bar (any column: meter is
        # often declared only in the kern spines).
        for line in lines[bar_start:bar_end]:
            t = _line_meter_token(line)
            if t:
                m = re.match(r'\*M(\d+)/(\d+)', t)
                cur_meter = F(int(m.group(1)), int(m.group(2)))
        t0 = bl_times.get(barline_indices[bi], F(0))
        nxt = next((j for j in all_bl if j > barline_indices[bi]), None)
        t1 = bl_times.get(nxt, bl_times['end'])
        bar_durs.append((t1 - t0, cur_meter))

    # A run of partial bars is accepted by re-measuring it as one joined
    # region (interior barlines fed as nothing, so `.` continuations flow
    # across).  Summing fragment durations breaks on ragged seams — a
    # section boundary can cut the two hands at different times, making
    # the fragments overlap and their sum exceed the meter even though
    # the joined bar is exactly full.
    to_remove = set()
    bi = 0
    n_bl = len(barline_indices)
    while bi < n_bl - 1:
        dur_i, meter_i = bar_durs[bi]
        if dur_i <= 0 or dur_i >= meter_i:
            bi += 1
            continue
        clock = _ColClock(len(lines[barline_indices[bi]].split('\t')))
        merged_to = None
        j = bi
        li = barline_indices[bi] + 1
        while j < n_bl:
            end_li = (barline_indices[j + 1] if j + 1 < n_bl
                      else len(lines))
            hit_final = False
            while li < end_li:
                if lines[li].startswith('=='):
                    hit_final = True
                    break
                clock.feed(lines[li])
                li += 1
            joined = clock.span()
            if j > bi and joined == meter_i:
                merged_to = j
                break
            if hit_final or joined >= meter_i:
                break
            j += 1
            if j >= n_bl or bar_durs[j][1] != meter_i:
                break
            li = barline_indices[j] + 1
        if merged_to is not None:
            for k in range(bi + 1, merged_to + 1):
                to_remove.add(barline_indices[k])
            bi = merged_to + 1
        else:
            bi += 1

    if not to_remove:
        return lines
    return [line for i, line in enumerate(lines) if i not in to_remove]


def _renumber_barlines(
    lines: List[str],
    *,
    opening_anacrusis: Optional[bool] = None,
) -> List[str]:
    """Renumber barlines sequentially after expansion.

    Pickup measures (=N-) become =0, subsequent barlines are 1, 2, 3, ...
    Final barlines (==) are preserved as-is.  ``opening_anacrusis`` overrides
    the textual pickup inference when the Humdrum writer omits bar numbers.
    """
    result = []
    measure_counter = 0
    first_barline_seen = False

    for line in lines:
        if not line.startswith('='):
            result.append(line)
            continue

        if line.startswith('=='):
            result.append(line)
            continue

        parts = line.split('\t')

        first_bar_part = next((p for p in parts if p.startswith('=')), None)
        inferred_pickup = (
            first_bar_part
            and '-' in first_bar_part
            and not first_bar_part.startswith('==')
        )
        is_pickup = (
            opening_anacrusis
            if not first_barline_seen and opening_anacrusis is not None
            else inferred_pickup
        )

        if not first_barline_seen:
            first_barline_seen = True
            if is_pickup:
                new_measure_num = 0
            else:
                measure_counter += 1
                new_measure_num = measure_counter
        else:
            measure_counter += 1
            new_measure_num = measure_counter

        new_parts = []
        for part in parts:
            if part.startswith('=') and not part.startswith('=='):
                # Replace the label while retaining the encoded barline style.
                # A letter immediately after the number is a source-label
                # suffix, not a style, so canonical numbering removes it.
                style = re.sub(r'^=(?:\d+[A-Za-z]?)?', '', part)
                new_parts.append(f'={new_measure_num}{style}')
            else:
                new_parts.append(part)

        result.append('\t'.join(new_parts))

    return result


def renumber_kern_barlines(
    kern_content: str,
    *,
    opening_anacrusis: bool,
) -> str:
    """Assign canonical playback-order numbers to final Humdrum barlines."""
    return '\n'.join(_renumber_barlines(
        kern_content.splitlines(),
        opening_anacrusis=opening_anacrusis,
    ))


def renumber_score_measures(
    score: m21.stream.Score,
    *,
    opening_anacrusis: bool,
) -> None:
    """Assign canonical playback-order measure numbers to every part.

    Repeat expansion can preserve source labels, suffix them, or duplicate
    them.  Downstream rendering instead consumes a through-composed score, so
    its measure identity must follow playback order.  A pickup remains measure
    zero; otherwise numbering starts at one.
    """
    start = 0 if opening_anacrusis else 1
    for part in score.parts:
        measures = list(part.getElementsByClass(m21.stream.Measure))
        for index, measure in enumerate(measures):
            measure.number = start + index
            measure.numberSuffix = None


def _count_spine_change(line: str) -> int:
    """Count net spine change from a line containing *^ or *v."""
    if not line.startswith('*') or line.startswith('**'):
        return 0

    parts = line.split('\t')
    change = 0
    i = 0
    while i < len(parts):
        if parts[i] == '*^':
            change += 1
        elif parts[i] == '*v':
            merge_count = 0
            while i < len(parts) and parts[i] == '*v':
                merge_count += 1
                i += 1
            change -= (merge_count - 1)
            continue
        i += 1
    return change


def _get_spine_count(line: str) -> int:
    """Get the number of spines (tab-separated fields) in a line."""
    if not line or line.startswith('!'):
        return 0
    return len(line.split('\t'))


def _apply_ops_to_types(types: List[str], line: str) -> List[str]:
    """Advance per-column spine types across a *^ / *v manipulator line."""
    parts = line.split('\t')
    out: List[str] = []
    src = 0
    i = 0
    while i < len(parts):
        if src >= len(types):
            break
        p = parts[i]
        if p == '*^':
            out.extend([types[src], types[src]])
            src += 1
            i += 1
        elif p == '*v':
            vc = 1
            while i + vc < len(parts) and parts[i + vc] == '*v':
                vc += 1
            out.append(types[src])
            src += vc
            i += vc
        else:
            out.append(types[src])
            src += 1
            i += 1
    out.extend(types[src:])
    return out


def _seam_merge_parts(types: List[str]) -> Optional[Tuple[List[str], int]]:
    """Synthesize a seam merge line targeting the rightmost adjacent
    **kern pair.  Position matters: with a trailing **dynam column, the
    blind last-two-columns merge would join a kern spine into dynam,
    which is illegal and desyncs every downstream spine tracker."""
    for i in range(len(types) - 2, -1, -1):
        if types[i] == '**kern' and types[i + 1] == '**kern':
            parts = ['*'] * i + ['*v', '*v'] + ['*'] * (len(types) - i - 2)
            return parts, i
    return None


def _seam_split_parts(types: List[str]) -> Optional[Tuple[List[str], int]]:
    """Synthesize a seam split line targeting the rightmost **kern column."""
    for i in range(len(types) - 1, -1, -1):
        if types[i] == '**kern':
            parts = ['*'] * i + ['*^'] + ['*'] * (len(types) - i - 1)
            return parts, i
    return None


# ============================================================================
# Kern-level repeat expansion
# ============================================================================


def _derive_expansion_order(kern_content: str,
                            section_ranges: Dict[str, Tuple[int, int]],
                            ) -> Optional[Tuple[List[str], Dict[str, Tuple[int, int]]]]:
    """Playback order for a score that states none and whose repeat sign
    stands inside a bar.

    Such a bar keeps only the half printed before the sign; the rest of
    it sits after the sign and is reached solely by taking the repeat.
    Left unexpanded the two halves never meet.  Where the sign falls on
    a barline the halves are whole bars already, so nothing is missing
    and the score is left as it is.

    The replay starts at the repeat sign, which can stand mid-section,
    so the section carrying it is cut in two.
    """
    from fractions import Fraction as F
    if not section_ranges:
        return None
    lines = kern_content.split('\n')
    times = _barline_times(lines)
    bl = sorted(k for k in times if isinstance(k, int))
    if len(bl) < 3:
        return None
    meter = F(1)
    for line in lines:
        t = _line_meter_token(line)
        if t:
            m = re.match(r'\*M(\d+)/(\d+)', t)
            meter = F(int(m.group(1)), int(m.group(2)))
            break

    opens = []
    for n, i in enumerate(bl[1:-1], start=1):
        if '|:' not in lines[i].split('\t')[0]:
            continue
        before = times[bl[n]] - times[bl[n - 1]]
        after = times[bl[n + 1]] - times[bl[n]]
        if 0 < before < meter and before + after == meter:
            opens.append(i)
    if len(opens) != 1:
        return None
    cut = opens[0]

    host = next((s for s, (a, b) in section_ranges.items() if a <= cut < b), None)
    if host is None or re.match(r'^.+\d+$', host):
        return None
    a, b = section_ranges[host]
    tail = f'{host}-repeat'
    if tail in section_ranges:
        return None
    ranges = dict(section_ranges)
    ranges[host] = (a, cut)
    ranges[tail] = (cut, b)

    order: List[str] = []
    for name, _ in sorted(section_ranges.items(), key=lambda kv: kv[1][0]):
        if re.match(r'^.+\d+$', name):
            continue        # an ending is placed by the section it serves
        if name == host:
            order += [host, tail]
            e1, e2 = f'{host}1', f'{host}2'
            if e1 in section_ranges:
                order += [e1, tail]
                if e2 in section_ranges:
                    order.append(e2)
        else:
            order.append(name)
    return order, ranges


def _resolve_expansion_plan(
    kern_content: str,
) -> Tuple[Optional[List[str]], Dict[str, Tuple[int, int]]]:
    """Resolve explicit and notation-derived playback plans identically."""
    section_ranges = parse_section_ranges(kern_content)
    expansion_order = parse_expansion_order(kern_content)
    if not expansion_order:
        derived = _derive_expansion_order(kern_content, section_ranges)
        if derived:
            expansion_order, section_ranges = derived
    return expansion_order, section_ranges


def expand_kern_repeats(kern_content: str) -> str:
    """Expand kern content according to Humdrum expansion labels.

    This function:
    1. Parses the expansion order (``*>[A,A,B,...]``)
    2. Identifies section boundaries (``*>A``, ``*>B``, etc.)
    3. Reassembles content in playback order
    4. Handles spine count mismatches between sections by inserting merge lines
    5. Removes repeat barlines to avoid music21 expandRepeats issues

    Args:
        kern_content: Raw kern file content with expansion labels

    Returns:
        Expanded kern content (through-composed, no repeats)

    Note:
        If no expansion labels are found, returns the original content
        with repeat barlines removed.
    """
    expansion_order, section_ranges = _resolve_expansion_plan(kern_content)
    if not expansion_order or not section_ranges:
        return _cleanup_unexpanded(kern_content)

    lines = kern_content.split('\n')

    # Trailing reference records are file metadata, not section
    # content — keep them exactly once, after the final barline.
    trailing_refs: List[str] = []
    ti = len(lines) - 1
    while ti >= 0 and (not lines[ti].strip()
                       or lines[ti].startswith('!!!')):
        if lines[ti].startswith('!!!'):
            trailing_refs.insert(0, lines[ti])
        ti -= 1

    # Find header end (everything before first section marker)
    first_section_start = min(start for start, _ in section_ranges.values())
    header_end = first_section_start - 1

    # Collect header lines (exclude expansion and norep labels)
    header_lines = []
    base_spine_count = 2
    base_spine_types: List[str] = ['**kern', '**kern']
    for i, line in enumerate(lines[:header_end]):
        if line.startswith('*>[') or line.startswith('*>norep['):
            continue
        if '\t' in line:
            parts = line.split('\t')
            if all(p.startswith('*>[') or p.startswith('*>norep[') for p in parts):
                continue
        if line.startswith('**'):
            base_spine_count = len(line.split('\t'))
            base_spine_types = line.split('\t')
        header_lines.append(line)

    # Assemble expanded content
    from fractions import Fraction as F
    meter = F(1)  # default 4/4
    for line in lines:
        found = _line_meter_token(line)
        if found:
            m = re.match(r'\*M(\d+)/(\d+)', found)
            meter = F(int(m.group(1)), int(m.group(2)))
            break

    # Per-position meter/key state in the document.  A re-entered
    # section must start under the state its notation was written in,
    # not whatever the previous instance's tail left behind.
    pos_m: List[Optional[str]] = []
    pos_k: List[Optional[str]] = []
    _sm: Optional[str] = None
    _sk: Optional[str] = None
    for line in lines:
        t = _line_meter_token(line)
        if t:
            _sm = t
        t = _line_key_token(line)
        if t:
            _sk = t
        pos_m.append(_sm)
        pos_k.append(_sk)

    sec_state: Dict[str, Tuple] = {}
    for s, (a, b) in section_ranges.items():
        entry_m = pos_m[a - 1] if a > 0 else None
        entry_k = pos_k[a - 1] if a > 0 else None
        exit_m = pos_m[b - 1] if b > 0 else None
        exit_k = pos_k[b - 1] if b > 0 else None
        # Does the section itself re-declare before its first data line?
        redecl_m = redecl_k = False
        for line in lines[a:b]:
            if _line_meter_token(line):
                redecl_m = True
            if _line_key_token(line):
                redecl_k = True
            if line.strip() and not line.startswith(('!', '*', '=')):
                break
        sec_state[s] = (entry_m, entry_k, exit_m, exit_k, redecl_m, redecl_k)

    # Data before a section's own first barline is normally its upbeat.
    # It is the rest of the bar being left behind only where the repeat
    # closes without reopening, the two add up to one bar, and the
    # section is taken again later -- an ending is written as its own
    # section and played once, so it keeps everything it carries.
    src_times = _barline_times(lines)
    src_bl = sorted(k for k in src_times if isinstance(k, int))
    played = Counter(expansion_order)
    lead_barline: Dict[str, Optional[int]] = {}
    for s, (a, b) in section_ranges.items():
        lead_barline[s] = None
        if played[s] < 2:
            continue
        prev_bl = next((lines[j].split('\t')[0]
                        for j in range(a - 2, -1, -1)
                        if lines[j].startswith('=')), '')
        if '|:' in prev_bl or ':|' not in prev_bl:
            continue
        idx = None
        for i in range(a, b):
            if lines[i].startswith('='):
                break
            if lines[i].strip() and not lines[i].startswith(('!', '*')):
                idx = next((j for j in range(i, b)
                            if lines[j].startswith('=')), None)
                break
        if idx is None:
            continue
        before = [i for i in src_bl if i < a]
        if len(before) < 2:
            continue
        cur = meter
        t = pos_m[a] if a < len(pos_m) else None
        if t:
            mm = re.match(r'\*M(\d+)/(\d+)', t)
            if mm:
                cur = F(int(mm.group(1)), int(mm.group(2)))
        prev_bar = src_times[before[-1]] - src_times[before[-2]]
        frag = src_times[idx] - src_times[before[-1]]
        if 0 < prev_bar < cur and prev_bar + frag == cur:
            lead_barline[s] = idx

    expanded_lines = header_lines.copy()
    current_spine_count = base_spine_count
    current_types: List[str] = list(base_spine_types)
    running_m: Optional[str] = None
    running_k: Optional[str] = None

    for si, section in enumerate(expansion_order):
        if section not in section_ranges:
            continue

        start, end = section_ranges[section]

        # Arriving with the bar behind already counted out to its last
        # beat, that remainder has nothing left to finish.
        if lead_barline[section] is not None:
            done = _barline_times(expanded_lines)
            done_bl = sorted(k for k in done if isinstance(k, int))
            if len(done_bl) >= 2 and done['end'] <= done[done_bl[-1]]:
                cur = meter
                if running_m:
                    mm = re.match(r'\*M(\d+)/(\d+)', running_m)
                    if mm:
                        cur = F(int(mm.group(1)), int(mm.group(2)))
                if done[done_bl[-1]] - done[done_bl[-2]] >= cur:
                    start = lead_barline[section] + 1

        section_start_spines = base_spine_count
        for line in lines[start:end]:
            if line and '\t' in line and not line.startswith('!'):
                section_start_spines = len(line.split('\t'))
                break

        # Guarantee a barline at every section seam.  A section whose
        # terminal barline was the piece-final ``==`` (dropped above)
        # re-enters with bare data, gluing two complete bars under one
        # barline.  Insert a plain barline unless one is already there
        # on either side of the seam.
        last_content = next(
            (l for l in reversed(expanded_lines)
             if l.strip() and not l.startswith(('!', '*'))), None)
        first_content = next(
            (l for l in lines[start:end]
             if l.strip() and not l.startswith(('!', '*'))), None)
        if (last_content is not None and not last_content.startswith('=')
                and first_content is not None
                and not first_content.startswith('=')):
            expanded_lines.append('\t'.join(['='] * current_spine_count))

        while current_spine_count > section_start_spines:
            synth = _seam_merge_parts(current_types)
            if synth is None:
                merge_parts = ['*'] * (current_spine_count - 2) + ['*v', '*v']
                current_types = current_types[:-1]
            else:
                merge_parts, mi = synth
                current_types = current_types[:mi + 1] + current_types[mi + 2:]
            expanded_lines.append('\t'.join(merge_parts))
            current_spine_count -= 1
        while current_spine_count < section_start_spines:
            synth = _seam_split_parts(current_types)
            if synth is None:
                split_parts = ['*'] * (current_spine_count - 1) + ['*^']
                current_types = current_types + [current_types[-1]]
            else:
                split_parts, si_ = synth
                current_types = (current_types[:si_ + 1]
                                 + [current_types[si_]] + current_types[si_ + 1:])
            expanded_lines.append('\t'.join(split_parts))
            current_spine_count += 1

        # Restore the meter/key the section's notation was written
        # under: a previous instance's tail may have modulated away,
        # and the section only re-declares when the engraver did.
        entry_m, entry_k, exit_m, exit_k, redecl_m, redecl_k = \
            sec_state[section]
        if start != section_ranges[section][0]:
            # Dropping the remainder can uncover a declaration that
            # stood behind it.
            redecl_m = redecl_k = False
            for line in lines[start:end]:
                if _line_meter_token(line):
                    redecl_m = True
                if _line_key_token(line):
                    redecl_k = True
                if line.strip() and not line.startswith(('!', '*', '=')):
                    break
        if (running_m is not None and entry_m is not None
                and running_m != entry_m and not redecl_m):
            expanded_lines.append(
                '\t'.join([entry_m] * current_spine_count))
        if (running_k is not None and entry_k is not None
                and running_k != entry_k and not redecl_k):
            expanded_lines.append(
                '\t'.join([entry_k] * current_spine_count))
        if exit_m is not None:
            running_m = exit_m
        if exit_k is not None:
            running_k = exit_k

        for line in lines[start:end]:
            if line.startswith('!!!'):
                continue
            if line.strip() and all(p.strip() == '*-' for p in line.split('\t')):
                continue
            if line.startswith('='):
                line = _clean_repeat_barline(line)
                # Double barlines (==) become single after cleaning;
                # the final == is appended at the end of expansion.
                # Cleaning leaves whatever it does not recognise (==:|.
                # keeps its dot), so measure the '=' run, not the tail.
                tok0 = line.split('\t')[0]
                if re.sub(r'[^=]', '', tok0) == '==':
                    parts = line.split('\t')
                    line = '\t'.join(
                        p.replace('==', '=', 1) for p in parts)
            expanded_lines.append(line)

            if line.startswith('*') and not line.startswith('**'):
                current_spine_count += _count_spine_change(line)
                if any(t in ('*^', '*v') for t in line.split('\t')):
                    current_types = _apply_ops_to_types(current_types, line)

    while current_spine_count > base_spine_count:
        synth = _seam_merge_parts(current_types)
        if synth is None:
            merge_parts = ['*'] * (current_spine_count - 2) + ['*v', '*v']
            current_types = current_types[:-1]
        else:
            merge_parts, mi = synth
            current_types = current_types[:mi + 1] + current_types[mi + 2:]
        expanded_lines.append('\t'.join(merge_parts))
        current_spine_count -= 1

    expanded_lines.append('\t'.join(['=='] * base_spine_count))
    expanded_lines.append('\t'.join(['*-'] * base_spine_count))

    expanded_lines = _merge_seam_bars(expanded_lines, meter)
    expanded_lines = _renumber_barlines(expanded_lines)
    expanded_lines.extend(trailing_refs)

    return '\n'.join(expanded_lines)


_CONSUMED_NAVIGATION_TEXT = re.compile(
    r"(?:D\s*\.\s*[CS](?:\s*\.)?(?=$|[^A-Za-z])"
    r"|Da\s+Capo\b|Dal\s+Segno\b"
    r"|^\s*Fine(?=$|[^A-Za-z])"
    r"|^\s*\[?Segno\]?(?=$|[^A-Za-z])"
    r"|^\s*Coda(?=$|[^A-Za-z]))",
    re.IGNORECASE,
)


def _is_consumed_navigation_layout(cell: str) -> bool:
    """Identify playback directives whose effect is already expanded."""
    if "LO:TX" not in cell or ":t=" not in cell:
        return False
    text = cell.split(":t=", 1)[1].strip()
    if text.startswith("P:"):
        return False
    return bool(_CONSUMED_NAVIGATION_TEXT.search(text))


def _strip_consumed_kern_navigation(kern_content: str) -> str:
    """Remove navigation layout tokens without changing Humdrum row shape."""
    cleaned_lines = []
    for line in kern_content.split("\n"):
        if line.startswith("!!") and _is_consumed_navigation_layout(line):
            cleaned_lines.append("!!")
            continue
        if line.startswith("!") and "\t" in line:
            cells = line.split("\t")
            cells = ["!" if _is_consumed_navigation_layout(c) else c
                     for c in cells]
            line = "\t".join(cells)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _cleanup_unexpanded(kern_content: str) -> str:
    """Cleanup for files that skip expansion (no ``*>[...]`` order).

    Sources without an expansion order still carry engraver split
    measures: a section's final partial bar and the next section's
    pickup share one printed bar, separated by a double or invisible
    barline. The expansion path merges these in its seam pass; this
    gives the unexpanded path the same merge.
    """
    from fractions import Fraction as F
    content = remove_repeat_barlines(kern_content)
    lines = content.split('\n')
    # Interior double barlines are section marks; only the last one ends
    # the piece.  The seam pass stops at any of them, so the interior
    # ones must read as ordinary barlines first.
    dbl = [i for i, l in enumerate(lines)
           if l.startswith('==')
           and re.sub(r'[^=]', '', l.split('\t')[0]) == '==']
    for i in dbl[:-1]:
        lines[i] = '\t'.join(p.replace('==', '=', 1)
                             for p in lines[i].split('\t'))
    meter = F(1)  # default 4/4
    for line in lines:
        t = _line_meter_token(line)
        if t:
            m = re.match(r'\*M(\d+)/(\d+)', t)
            meter = F(int(m.group(1)), int(m.group(2)))
            break
    merged = _merge_seam_bars(lines, meter)
    return '\n'.join(_renumber_barlines(merged))


def remove_repeat_barlines(kern_content: str) -> str:
    """Remove repeat barlines from kern content without expanding.

    Use this for files without expansion labels but with repeat barlines
    that cause music21 to fail.
    """
    lines = kern_content.split('\n')
    cleaned_lines = []

    for line in lines:
        if re.match(r'^\*>\[.*\]', line) or re.match(r'^\*>norep\[.*\]', line):
            continue
        if line.startswith('*>') and '\t' in line:
            parts = line.split('\t')
            if all(p.startswith('*>') and not p.startswith('*>[') for p in parts):
                continue
        if line.startswith('='):
            line = _clean_repeat_barline(line)
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def get_expansion_info(kern_content: str) -> Dict:
    """Get information about the repeat structure for debugging."""
    expansion_order = parse_expansion_order(kern_content)
    section_ranges = parse_section_ranges(kern_content)

    info: Dict[str, Any] = {
        'has_expansion': expansion_order is not None,
        'expansion_order': expansion_order,
        'sections': section_ranges,
        'section_count': len(section_ranges),
    }

    if expansion_order and section_ranges:
        original_lines = sum(end - start for start, end in section_ranges.values())
        expanded_lines = sum(
            section_ranges[s][1] - section_ranges[s][0]
            for s in expansion_order
            if s in section_ranges
        )
        info['estimated_expansion_ratio'] = expanded_lines / original_lines if original_lines > 0 else 1.0
    else:
        info['estimated_expansion_ratio'] = 1.0

    return info


def expand_kern_repeats_with_mapping(kern_content: str) -> Tuple[str, Dict]:
    """Expand kern repeats and return measure mapping for visual aux head.

    Expansion is delegated to ``expand_kern_repeats()``.  The mapping is built
    by running ``extract_kern_measures`` on the expanded kern (single source of
    truth for measure boundaries), then annotating each entry with
    section / occurrence / repeat info from the original expansion labels.

    Args:
        kern_content: Raw kern file content with expansion labels

    Returns:
        Tuple of (expanded_kern, repeat_map_dict)
    """
    import tempfile
    from pathlib import Path

    expansion_order, section_ranges = _resolve_expansion_plan(kern_content)
    expanded = expand_kern_repeats(kern_content)
    if expansion_order and section_ranges:
        expanded = _strip_consumed_kern_navigation(expanded)

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.krn', delete=False
    ) as tmp:
        tmp.write(expanded)
        tmp_path = Path(tmp.name)
    try:
        measures_gt = extract_kern_measures(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    n_measures = len(measures_gt)

    if expansion_order and section_ranges:
        lines = kern_content.split('\n')

        section_barline_counts: Dict[str, int] = {}
        section_original_bars: Dict[str, List[Optional[int]]] = {}
        for section_name, (start, end) in section_ranges.items():
            bar_nums: List[Optional[int]] = []
            for line in lines[start:end]:
                if line.startswith('='):
                    parts = line.split('\t')
                    match = re.match(r'=(\d+)', parts[0])
                    bar_nums.append(int(match.group(1)) if match else None)
            section_barline_counts[section_name] = len(bar_nums)
            section_original_bars[section_name] = bar_nums

        def _kern_ending_type(section_name: str) -> int:
            m = re.match(r'^([A-Za-z]+)(\d+)$', section_name)
            if m:
                base, num = m.group(1), int(m.group(2))
                other = f"{base}{2 if num == 1 else 1}"
                if other in section_ranges:
                    return num
            return 0

        section_annotations: List[Dict[str, Any]] = []
        section_occ: Counter = Counter()

        for section_name in expansion_order:
            if section_name not in section_ranges:
                continue
            occurrence = section_occ[section_name]
            section_occ[section_name] += 1
            ending_type = _kern_ending_type(section_name)

            for orig_bar in section_original_bars.get(section_name, []):
                section_annotations.append({
                    "original_measure": orig_bar,
                    "section": section_name,
                    "occurrence": occurrence,
                    "is_repeat": occurrence > 0,
                    "ending_type": ending_type,
                })

        all_original: Set[int] = set()
        for section_name in set(expansion_order):
            for b in section_original_bars.get(section_name, []):
                if b is not None:
                    all_original.add(b)

        measure_mapping = []
        for i, m_gt in enumerate(measures_gt):
            entry: Dict[str, Any] = {"expanded_measure": m_gt["measure"]}
            if i < len(section_annotations):
                entry.update(section_annotations[i])
            else:
                entry.update({
                    "original_measure": None,
                    "section": "",
                    "occurrence": 0,
                    "is_repeat": False,
                    "ending_type": 0,
                })
            measure_mapping.append(entry)

        return expanded, {
            "has_repeats": True,
            "expansion_order": expansion_order,
            "original_measure_count": len(all_original),
            "expanded_measure_count": n_measures,
            "measures": measure_mapping,
        }

    # Fallback: no expansion labels -> identity mapping
    measure_mapping = []
    for m_gt in measures_gt:
        measure_mapping.append({
            "expanded_measure": m_gt["measure"],
            "original_measure": m_gt["measure"],
            "section": "",
            "occurrence": 0,
            "is_repeat": False,
            "ending_type": 0,
        })

    return expanded, {
        "has_repeats": False,
        "expansion_order": None,
        "original_measure_count": n_measures,
        "expanded_measure_count": n_measures,
        "measures": measure_mapping,
    }


# ============================================================================
# Music21-level repeat expansion (MuseSyn)
# ============================================================================

# Navigation marker types detected from music21
_NAVIGATION_TYPES = (
    m21.repeat.DaCapo,
    m21.repeat.DalSegno,
    m21.repeat.Fine,
    m21.repeat.Segno,
    m21.repeat.Coda,
)


def _strip_consumed_music21_repeats(score: m21.stream.Score) -> None:
    """Remove repeat semantics after their playback order has been realized."""
    for element in list(score.recurse()):
        if isinstance(element, _NAVIGATION_TYPES):
            site = element.activeSite
            if site is not None:
                site.remove(element)

    for bracket in list(score.spannerBundle):
        if isinstance(bracket, m21.spanner.RepeatBracket):
            site = bracket.activeSite
            if site is not None:
                site.remove(bracket)

    for measure in score.recurse().getElementsByClass(m21.stream.Measure):
        if isinstance(measure.leftBarline, m21.bar.Repeat):
            measure.leftBarline = m21.bar.Barline("regular")
        if isinstance(measure.rightBarline, m21.bar.Repeat):
            measure.rightBarline = m21.bar.Barline("regular")


def _drop_staff_groups(score: m21.stream.Score) -> None:
    """converter21 raises KeyError on PartStaff for a StaffGroup on a copied score."""
    for group in list(score.getElementsByClass(m21.layout.StaffGroup)):
        score.remove(group)


def _reconnect_spanners(post: m21.stream.Score) -> None:
    """Point slurs and wedges at the copies, the way music21's own expander does."""
    bundle = post.spannerBundle
    for element in post.recurse(includeSelf=False):
        if not element.sites.hasSpannerSite():
            continue
        origin = element.derivation.origin
        if origin is not None and element.derivation.method == '__deepcopy__':
            bundle.replaceSpannedElement(origin, element)


def _finalize_expansion(score: m21.stream.Score) -> None:
    """Everything an expanded score needs once its playback order is fixed."""
    _strip_consumed_music21_repeats(score)
    _drop_staff_groups(score)


def expand_asap_score(
    score: m21.stream.Score,
    measure_order: List[int],
) -> Tuple[m21.stream.Score, bool]:
    """Realise a playback order handed in from outside, as measure numbers.

    The order comes from an alignment against a real performance, not from the
    score's own repeat signs, so it may revisit a bar, skip forward, or run in
    a sequence the notation never spells out.  A score whose order is simply
    its own bars in their own sequence is returned untouched.

    Args:
        score: music21 Score (already sanitized).
        measure_order: measure numbers in playback order.

    Returns:
        ``(score, has_repeats)`` -- the realised score and whether any bar is
        played more than once.
    """
    natural = [m.number for m in
               score.parts[0].getElementsByClass(m21.stream.Measure)]
    if measure_order == natural:
        return score, False

    post = score.cloneEmpty(derivationMethod='expandRepeats')
    post.mergeAttributes(score)
    for element in score.iter().getElementsNotOfClass(m21.stream.Part):
        post.insert(score.elementOffset(element), copy.deepcopy(element))

    for part in score.getElementsByClass(m21.stream.Part):
        by_number = {m.number: m
                     for m in part.getElementsByClass(m21.stream.Measure)}
        new_part = part.cloneEmpty(derivationMethod='expandRepeats')
        for element in part.iter().getElementsNotOfClass(m21.stream.Measure):
            if 'RepeatBracket' in element.classes:
                continue
            new_part.insert(part.elementOffset(element),
                            copy.deepcopy(element))
        offset = 0.0
        for index, number in enumerate(measure_order):
            source = by_number.get(number)
            if source is None:
                raise KeyError(
                    f"measure {number} is not in part {part.id}")
            copied = copy.deepcopy(source)
            copied.number = index + 1
            copied.numberSuffix = None
            new_part.insert(offset, copied)
            offset += source.duration.quarterLength
        post.insert(0, new_part)

    _reconnect_spanners(post)
    _finalize_expansion(post)
    return post, len(measure_order) > len(set(measure_order))


def expand_musesyn_score(
    score: m21.stream.Score,
) -> Tuple[m21.stream.Score, bool]:
    """Expand repeats in a MuseSyn score if present.

    Compares measure counts before/after ``expandRepeats()`` to detect
    all repeat types (barline repeats, DaCapo, DalSegno, Fine, etc.).
    Also removes StaffGroup spanners that crash converter21.

    Args:
        score: music21 Score (already sanitized).

    Returns:
        ``(score, has_repeats)`` -- the expanded score (or original if no
        repeats) and a boolean flag.
    """
    n_before = len(
        score.parts[0].getElementsByClass(m21.stream.Measure)
    )
    expanded = score.expandRepeats()
    n_after = len(
        expanded.parts[0].getElementsByClass(m21.stream.Measure)
    )
    has_repeats = n_after > n_before

    if has_repeats:
        _finalize_expansion(expanded)
        return expanded, True

    return score, False


def extract_repeat_structure(score: m21.stream.Score) -> Dict[str, Any]:
    """Extract repeat structure from the original (unexpanded) score.

    Must be called BEFORE ``expandRepeats()``.

    Returns:
        Dictionary with:
        - repeat_barlines: [{measure, direction}, ...]
        - navigation: [{measure, type}, ...]  (DaCapo, DalSegno, Fine, Segno, Coda)
        - volta_brackets: [{measures, number}, ...]
        - orig_measure_numbers: [int, ...]
        - dacapo_measure: int or None  (measure number of DaCapo/DalSegno)
    """
    part = score.parts[0]
    measures = list(part.getElementsByClass(m21.stream.Measure))
    orig_nums = [m.number for m in measures]

    # 1. Repeat barlines
    repeat_barlines: List[Dict[str, Any]] = []
    seen_barlines: Set[Tuple[int, str]] = set()
    for m_obj in measures:
        for bar in [m_obj.leftBarline, m_obj.rightBarline]:
            if bar and isinstance(bar, m21.bar.Repeat):
                key = (m_obj.number, bar.direction)
                if key not in seen_barlines:
                    seen_barlines.add(key)
                    repeat_barlines.append({
                        "measure": m_obj.number,
                        "direction": bar.direction,
                    })

    # 2. Navigation markers (DaCapo, DalSegno, Fine, Segno, Coda)
    navigation: List[Dict[str, Any]] = []
    dacapo_measure: Optional[int] = None
    seen_nav: Set[Tuple[int, str]] = set()
    for p in score.parts:
        for el in p.flatten():
            if isinstance(el, _NAVIGATION_TYPES):
                m_num = getattr(el, "measureNumber", None)
                if m_num is None:
                    continue
                type_name = el.__class__.__name__
                key = (m_num, type_name)
                if key not in seen_nav:
                    seen_nav.add(key)
                    navigation.append({"measure": m_num, "type": type_name})
                    if type_name in ("DaCapo", "DalSegno") and dacapo_measure is None:
                        dacapo_measure = m_num

    # 3. Volta brackets (RepeatBracket spanners)
    volta_brackets: List[Dict[str, Any]] = []
    seen_volta: Set[Tuple[int, ...]] = set()
    for sp in score.spannerBundle:
        if isinstance(sp, m21.spanner.RepeatBracket):
            spanned = sp.getSpannedElements()
            m_nums = sorted(set(
                s.number for s in spanned
                if isinstance(s, m21.stream.Measure)
            ))
            if m_nums:
                key = tuple(m_nums + [sp.number])
                if key not in seen_volta:
                    seen_volta.add(key)
                    try:
                        bracket_num = int(str(sp.number).strip().rstrip("."))
                    except (ValueError, TypeError):
                        bracket_num = 0
                    volta_brackets.append({
                        "measures": m_nums,
                        "number": bracket_num,
                    })

    return {
        "repeat_barlines": repeat_barlines,
        "navigation": navigation,
        "volta_brackets": volta_brackets,
        "orig_measure_numbers": orig_nums,
        "dacapo_measure": dacapo_measure,
    }


def _get_ending_type(
    measure_num: int, volta_brackets: List[Dict[str, Any]]
) -> int:
    """Return volta bracket number (1=first ending, 2=second ending, 0=none)."""
    for vb in volta_brackets:
        if measure_num in vb["measures"]:
            return vb["number"]
    return 0


def _build_rich_mapping(
    repeat_structure: Dict[str, Any],
    expanded_score: m21.stream.Score,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build a rich expanded->original measure mapping from music21 Scores.

    Handles three cases:
    1. Simple barline repeats: expanded numbers have duplicates.
    2. Pure DaCapo (no barline repeats): sequential renumbered measures.
    3. Mixed (repeats + DaCapo): first pass has duplicates, DaCapo pass
       has sequential renumbered measures.

    Returns:
        (measure_mapping, expansion_order)
    """
    orig_nums = repeat_structure["orig_measure_numbers"]
    dacapo_measure = repeat_structure["dacapo_measure"]
    volta_brackets = repeat_structure["volta_brackets"]
    orig_max = max(orig_nums) if orig_nums else 0

    exp_measures = list(
        expanded_score.parts[0].getElementsByClass(m21.stream.Measure)
    )
    exp_nums = [m.number for m in exp_measures]

    has_duplicates = len(exp_nums) != len(set(exp_nums))

    # Determine DaCapo split point in expanded sequence.
    dacapo_split: Optional[int] = None
    if dacapo_measure is not None:
        threshold = dacapo_measure
    elif not has_duplicates and len(exp_nums) > len(orig_nums):
        threshold = orig_max
    else:
        threshold = None

    if threshold is not None:
        for i, n in enumerate(exp_nums):
            if n > threshold:
                dacapo_split = i
                break

    # --- Build first pass mapping ---
    first_pass_nums = exp_nums[:dacapo_split] if dacapo_split else exp_nums
    occurrence_global: Dict[int, int] = {}
    mapping: List[Dict[str, Any]] = []
    first_pass_orig_sequence: List[int] = []

    section_idx = 0
    prev_num = -1

    for i, n in enumerate(first_pass_nums):
        if n < prev_num:
            section_idx += 1
        prev_num = n

        occ = occurrence_global.get(n, 0)
        occurrence_global[n] = occ + 1
        ending = _get_ending_type(n, volta_brackets)

        base_section = chr(ord("A") + section_idx)
        if ending > 0:
            section = f"{base_section}{ending}"
        else:
            section = base_section

        mapping.append({
            "original_measure": n,
            "section": section,
            "occurrence": occ,
            "is_repeat": occ > 0,
            "ending_type": ending,
        })
        first_pass_orig_sequence.append(n)

    # --- Build DaCapo pass mapping (if present) ---
    if dacapo_split is not None:
        dacapo_nums = exp_nums[dacapo_split:]
        dacapo_offset = dacapo_nums[0] - 1 if dacapo_nums else 0

        for i, n in enumerate(dacapo_nums):
            if i < len(first_pass_orig_sequence):
                orig_n = first_pass_orig_sequence[i]
            else:
                orig_n = n - dacapo_offset

            occ = occurrence_global.get(orig_n, 0)
            occurrence_global[orig_n] = occ + 1
            ending = _get_ending_type(orig_n, volta_brackets)

            section_idx += 1 if i == 0 else 0
            base_section = chr(ord("A") + section_idx)
            if ending > 0:
                section = f"{base_section}{ending}"
            else:
                section = base_section

            mapping.append({
                "original_measure": orig_n,
                "section": section,
                "occurrence": occ,
                "is_repeat": occ > 0,
                "ending_type": ending,
            })

    # Derive expansion_order from section sequence (deduplicate consecutive)
    expansion_order: List[str] = []
    for m in mapping:
        s = m["section"]
        if not expansion_order or expansion_order[-1] != s:
            expansion_order.append(s)

    return mapping, expansion_order


def _align_with_kern_measures(
    mapping: List[Dict[str, Any]],
    measures_gt: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Align music21-based mapping with kern-based extract_kern_measures output.

    ``measures_gt`` is the ground truth for measure count and numbering
    (from kern).  The mapping was built from music21's expanded score.
    """
    aligned: List[Dict[str, Any]] = []
    n_gt = len(measures_gt)
    n_map = len(mapping)

    for i, m_gt in enumerate(measures_gt):
        entry: Dict[str, Any] = {"expanded_measure": m_gt["measure"]}
        if i < n_map:
            entry["original_measure"] = mapping[i]["original_measure"]
            entry["section"] = mapping[i]["section"]
            entry["occurrence"] = mapping[i]["occurrence"]
            entry["is_repeat"] = mapping[i]["is_repeat"]
            entry["ending_type"] = mapping[i]["ending_type"]
        elif aligned:
            # Beyond mapping range (e.g. tie resolution after final barline).
            # Inherit from the last aligned entry.
            prev = aligned[-1]
            entry["original_measure"] = prev["original_measure"]
            entry["section"] = prev["section"]
            entry["occurrence"] = prev["occurrence"]
            entry["is_repeat"] = prev["is_repeat"]
            entry["ending_type"] = prev["ending_type"]
        else:
            entry["original_measure"] = m_gt["measure"]
            entry["section"] = ""
            entry["occurrence"] = 0
            entry["is_repeat"] = False
            entry["ending_type"] = 0

        aligned.append(entry)

    if n_map != n_gt:
        logger.warning(
            f"Measure count mismatch: music21={n_map} vs kern={n_gt}. "
            f"Alignment may be imprecise."
        )

    return aligned


def build_musesyn_repeat_map(
    repeat_structure: Dict[str, Any],
    expanded_score: m21.stream.Score,
    measures_gt: List[Dict[str, Any]],
    has_repeats: bool,
    original_measure_count: int,
) -> Dict[str, Any]:
    """Build a complete repeat_map for a MuseSyn file.

    Combines rich mapping, kern alignment, and original markers into
    the final repeat_map dictionary.

    Args:
        repeat_structure: Output of ``extract_repeat_structure()``.
        expanded_score: The score after ``expand_musesyn_score()``.
        measures_gt: Output of ``extract_kern_measures()`` on the cleaned kern.
        has_repeats: Whether the score had repeats.
        original_measure_count: Number of measures in the original score.

    Returns:
        Complete repeat_map dictionary (same format as HumSyn, plus
        ``original_markers``).
    """
    if has_repeats:
        rich_mapping, expansion_order = _build_rich_mapping(
            repeat_structure, expanded_score
        )
        aligned = _align_with_kern_measures(rich_mapping, measures_gt)
    else:
        # No repeats: expanded == original.
        aligned = []
        for m_gt in measures_gt:
            orig = m_gt["measure"]
            aligned.append({
                "expanded_measure": m_gt["measure"],
                "original_measure": orig,
                "section": "A",
                "occurrence": 0,
                "is_repeat": False,
                "ending_type": 0,
            })
        expansion_order = None

    original_markers = {
        "repeat_barlines": repeat_structure["repeat_barlines"],
        "navigation": repeat_structure["navigation"],
        "volta_brackets": repeat_structure["volta_brackets"],
    }

    return {
        "has_repeats": has_repeats,
        "expansion_order": expansion_order,
        "original_measure_count": original_measure_count,
        "expanded_measure_count": len(measures_gt),
        "original_markers": original_markers,
        "measures": aligned,
    }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Kern-level
    "expand_kern_repeats",
    "expand_kern_repeats_with_mapping",
    "remove_repeat_barlines",
    "has_expansion_labels",
    "parse_expansion_order",
    "parse_section_ranges",
    "get_expansion_info",
    # Music21-level (MuseSyn)
    "expand_musesyn_score",
    "renumber_kern_barlines",
    "renumber_score_measures",
    "extract_repeat_structure",
    "build_musesyn_repeat_map",
]
