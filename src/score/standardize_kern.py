"""
Kern Ground-Truth Standardization

Content-level transformations applied to Phase 1 kern (visual markers
already stripped) before it becomes training-target kern_gt: tie-mark
balancing (orphan halves repaired or stripped), accidental normalization
for the tokenizer's pitch grammar (no naturals, no double accidentals)
and slur/phrase marker removal (no vocab entry for them).
"""

import difflib
import json
import logging
import re
from collections import Counter, defaultdict, deque
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@lru_cache(maxsize=None)
def _cells(line: str) -> tuple:
    """Cached tab-split of a kern line, as an immutable tuple.

    Topology and tie binding revisit the same immutable source lines, so
    without this the same text is re-split millions of times per file."""
    return tuple(line.split('\t'))

logger = logging.getLogger(__name__)


def issue_sidecar_path(kern_path: Path) -> Path:
    return Path(kern_path).with_suffix(".issues.json")


def write_bar_issues(kern_path: Path, issues: Iterable[dict]) -> None:
    path = issue_sidecar_path(kern_path)
    rows = [dict(issue) for issue in issues]
    if not rows:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_bar_issues(kern_path: Path) -> List[dict]:
    path = issue_sidecar_path(kern_path)
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(
            isinstance(item, dict) for item in rows):
        raise ValueError(f"invalid kern issue payload in {path}")
    return rows


class KernTokenizerOOV(ValueError):
    """The standardized score contains content the tokenizer cannot encode."""

    def __init__(
        self, message: str, kern_content: str,
        bar_issues: Optional[List[dict]] = None,
    ):
        super().__init__(message)
        self.kern_content = kern_content
        self.bar_issues = list(bar_issues or ())


class KernIncompleteTuplet(ValueError):
    """The standardized score contains a tuplet run that cannot close."""

    def __init__(
        self, message: str, kern_content: str,
        bar_issues: Optional[List[dict]] = None,
    ):
        super().__init__(message)
        self.kern_content = kern_content
        self.bar_issues = list(bar_issues or ())


class KernMetricTimelineError(ValueError):
    """A standardized bar does not tile the duration declared by its meter."""

    def __init__(
        self, message: str, kern_content: str,
        bar_issues: Optional[List[dict]] = None,
    ):
        super().__init__(message)
        self.kern_content = kern_content
        self.bar_issues = list(bar_issues or ())


class KernWriterContractError(ValueError):
    """The standardizer failed to produce deterministic canonical text."""

    def __init__(
        self,
        message: str,
        kern_content: str,
        readback: Optional[str] = None,
        bar_issues: Optional[List[dict]] = None,
    ):
        super().__init__(message)
        self.kern_content = kern_content
        self.readback = readback
        self.bar_issues = list(bar_issues or ())


def _contract_bar_issue(
    kern_content: str,
    ordinal: int,
    family: str,
    message: str,
) -> List[dict]:
    """Attach a writer/token-scope failure to its data-bar ordinal."""
    label = None
    try:
        from src.a2s.piano.tokenizer import KernTokenizer

        bars = KernTokenizer().parse_kern_bars_isolated(kern_content)
        if 0 <= ordinal < len(bars):
            label = bars[ordinal].get("label")
    except (ValueError, TypeError):
        pass
    return [{
        "family": family,
        "message": message,
        "bar_index": ordinal,
        "bar_label": label,
        "bar_ordinal": ordinal,
    }]


def _tokenizer_bar_issue(error: ValueError, family: str) -> List[dict]:
    """Translate a reader refusal's native bar address into a sidecar row."""
    index = getattr(error, "bar_index", None)
    label = getattr(error, "bar_label", None)
    ordinal = getattr(error, "bar_ordinal", None)
    if index is None and label is None and ordinal is None:
        return []
    return [{
        "family": family,
        "message": str(error),
        "bar_index": index,
        "bar_label": label,
        "bar_ordinal": ordinal,
    }]


# =============================================================================
# Tie-Mark Repair
# =============================================================================
# Kern sources carry unmatched tie halves: a `[` whose pitch never returns,
# or a `]` whose pitch was never opened.  Pairing is resolved per hand and
# pitch in line order — the invariant the ground truth actually needs:
# every close answers an earlier open of the same hand and pitch.  Orphans are either repairable (a same-hand same-pitch note exists
# to receive the missing half — the pair is completed in place) or hopeless
# (no candidate anywhere — the mark is stripped).  Repairs run to a fixpoint
# so the output is always balanced, and every decision is emitted as a log
# record for review.

_PITCH_RE = re.compile(r"[a-gA-G]+[#\-n]*")
_SERIAL_PITCH_RE = re.compile(
    r"(([A-G])\2*|([a-g])\3*)[#\-n]*"
)
_METER_RE = re.compile(r'\*M(\d+)/(\d+)')
_SPINE_OPS = {"*^", "*v", "*-", "*+", "*x"}
_MAX_ROUNDS = 10


def _subtoken_pitch(tok: str) -> Optional[str]:
    """Pitch spelled in one note subtoken; None for rests/graces/non-notes."""
    if "q" in tok:
        return None
    m = _PITCH_RE.search(tok)
    if not m:
        return None
    return m.group(0).replace("n", "")


def _scan_ties(lines: List[str]):
    """Walk all note subtokens with hand attribution through spine ops.

    Returns (notes, orphan_opens, orphan_closes) where notes is a list of
    [line, field, sub, side, pitch, has_open, has_close] and orphans are
    (line, field, sub, side, pitch) of unbalanced tie halves under
    per-(side, pitch) queue pairing in line order.
    """
    sides: Optional[List[int]] = None  # per field: kern hand index or -1
    notes: List[list] = []
    open_q: Dict[Tuple[int, str], deque] = {}
    orphan_opens: List[Tuple[int, int, int, int, str]] = []
    orphan_closes: List[Tuple[int, int, int, int, str]] = []
    carried_to_head: Set[Tuple[int, int, int, int, str]] = set()
    awaiting_bar_head = False

    for ln, line in enumerate(lines):
        if not line or line.startswith("!"):
            continue
        parts = line.split("\t")
        if line.startswith("**"):
            sides = []
            k = 0
            for p in parts:
                sides.append(k if p == "**kern" else -1)
                if p == "**kern":
                    k += 1
            continue
        if sides is None:
            continue
        if line.startswith("*"):
            if any(p in _SPINE_OPS for p in parts):
                new: List[int] = []
                i = 0
                while i < len(parts):
                    op = parts[i]
                    s = sides[i] if i < len(sides) else -1
                    if op == "*^":
                        new.append(s)
                        new.append(s)
                        i += 1
                    elif op == "*v":
                        new.append(s)
                        j = i + 1
                        while j < len(parts) and parts[j] == "*v":
                            j += 1
                        i = j
                    elif op == "*-":
                        i += 1
                    elif op == "*x":
                        if i + 1 < len(sides):
                            new.append(sides[i + 1])
                            new.append(sides[i])
                        i += 2
                    elif op == "*+":
                        new.append(s)
                        new.append(-1)
                        i += 1
                    else:
                        new.append(s)
                        i += 1
                sides = new
            continue
        if line.startswith("="):
            carried_to_head = {
                location for queue in open_q.values() for location in queue
            }
            awaiting_bar_head = True
            continue

        for fi in range(min(len(sides), len(parts))):
            side = sides[fi]
            cell = parts[fi].strip()
            if side < 0 or not cell or cell == ".":
                continue
            for si, tok in enumerate(cell.split(" ")):
                p = _subtoken_pitch(tok)
                if p is None:
                    continue
                has_open = "[" in tok or "_" in tok
                has_close = "]" in tok or "_" in tok
                notes.append([ln, fi, si, side, p, has_open, has_close])
                key = (side, p)
                if has_close:
                    q = open_q.get(key)
                    if q:
                        q.pop()
                    else:
                        orphan_closes.append((ln, fi, si, side, p))
                if has_open:
                    open_q.setdefault(key, deque()).append((ln, fi, si, side, p))

        if awaiting_bar_head:
            for key, queue in list(open_q.items()):
                kept = deque()
                for location in queue:
                    if location in carried_to_head:
                        orphan_opens.append(location)
                    else:
                        kept.append(location)
                open_q[key] = kept
            carried_to_head.clear()
            awaiting_bar_head = False

    orphan_opens.extend(loc for q in open_q.values() for loc in q)
    return notes, orphan_opens, orphan_closes


def _apply_tie_edits(lines: List[str], edits: Dict[Tuple[int, int, int], set]) -> None:
    by_line: Dict[int, List[Tuple[Tuple[int, int, int], set]]] = {}
    for loc, ops in edits.items():
        by_line.setdefault(loc[0], []).append((loc, ops))
    for ln, cell_edits in by_line.items():
        parts = lines[ln].split("\t")
        for (_, fi, si), ops in cell_edits:
            subs = parts[fi].split(" ")
            tok = subs[si]
            had_open = "[" in tok or "_" in tok
            had_close = "]" in tok or "_" in tok
            open_f = had_open and "drop_open" not in ops
            close_f = (had_close and "drop_close" not in ops) or "add_close" in ops
            core = tok.replace("[", "").replace("]", "").replace("_", "")
            if open_f and close_f:
                tok = core + "_"
            elif open_f:
                tok = "[" + core
            elif close_f:
                tok = core + "]"
            else:
                tok = core
            subs[si] = tok
            parts[fi] = " ".join(subs)
        lines[ln] = "\t".join(parts)


_KEY_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def _kern_key(p: str) -> Optional[int]:
    """Piano key of a kern pitch spelling (enharmonic-blind)."""
    m = re.match(r"([a-gA-G])(\1*)([#\-n]*)", p)
    if not m:
        return None
    ch, rest, acc = m.group(1), m.group(2), m.group(3)
    n = 1 + len(rest)
    octv = (3 + n) if ch.islower() else (4 - n)
    return (12 * octv + _KEY_PC[ch.lower()]
            + acc.count("#") - acc.count("-"))


def repair_ties(kern_content: str, *,
                strip_unmatched: bool = True) -> Tuple[str, List[dict]]:
    """Balance explicit same-hand, same-pitch attachments.

    Marked closes pair with the innermost pending open.  This preserves a
    close-and-reopen reattack: the old sound closes there and the new
    departure owns the later landing.  Exact adjacency is used only to repair
    a missing close; a non-adjacent unmarked attack is never promoted into one.
    """
    from src.score.sanitize_kern import parse_kern_duration

    lines = kern_content.split("\n")
    records: List[dict] = []

    for _ in range(_MAX_ROUNDS):
        notes, orphan_opens, orphan_closes = _scan_ties(lines)
        if not orphan_opens and not orphan_closes:
            break
        gt = _global_times(lines)
        edits: Dict[Tuple[int, int, int], set] = {}
        claimed: Set[Tuple[int, int, int]] = set()
        respelled = False

        def note_end(location):
            line, field, sub, _hand, _pitch = location
            onset = gt.get(line) if gt is not None else None
            subtokens = lines[line].split("\t")[field].split(" ")
            duration = (
                parse_kern_duration(subtokens[sub])
                if sub < len(subtokens) else None
            )
            if not duration or duration <= 0:
                duration = next(
                    (value for value in
                     (parse_kern_duration(token) for token in subtokens)
                     if value and value > 0),
                    None,
                )
            return (
                onset + Fraction(duration)
                if onset is not None and duration else None
            )

        for location in orphan_opens:
            line, field, sub, hand, pitch = location
            end = note_end(location)
            exact = []
            enharmonic = []
            for candidate in notes:
                c_line, c_field, c_sub, c_hand, c_pitch, _open, close = candidate
                c_loc = (c_line, c_field, c_sub)
                if c_loc in claimed or c_line <= line or gt is None                         or gt.get(c_line) != end or c_hand != hand:
                    continue
                if c_pitch == pitch and not close:
                    exact.append(candidate)
                elif close and _kern_key(c_pitch) == _kern_key(pitch):
                    enharmonic.append(candidate)
            if exact:
                target = min(
                    exact,
                    key=lambda item: (
                        item[1] != field, item[0], item[1], item[2]),
                )
                target_loc = (target[0], target[1], target[2])
                edits.setdefault(target_loc, set()).add("add_close")
                claimed.add(target_loc)
                records.append({
                    "action": "adopt_close",
                    "pitch": pitch,
                    "line": target[0] + 1,
                })
                continue
            if enharmonic:
                target = min(
                    enharmonic,
                    key=lambda item: (
                        item[1] != field, item[0], item[1], item[2]),
                )
                parts = lines[target[0]].split("\t")
                subtokens = parts[target[1]].split(" ")
                subtokens[target[2]] = subtokens[target[2]].replace(
                    target[4], pitch, 1)
                parts[target[1]] = " ".join(subtokens)
                lines[target[0]] = "\t".join(parts)
                claimed.add((target[0], target[1], target[2]))
                respelled = True
                records.append({
                    "action": "enharmonic_respell_join",
                    "pitch": pitch,
                    "respelled_from": target[4],
                    "dep_line": line + 1,
                    "land_line": target[0] + 1,
                })
                continue
            if strip_unmatched:
                edits.setdefault((line, field, sub), set()).add("drop_open")
                records.append({
                    "action": "strip_open", "pitch": pitch,
                    "line": line + 1,
                })

        if strip_unmatched:
            for line, field, sub, _hand, pitch in orphan_closes:
                location = (line, field, sub)
                if location in claimed:
                    continue
                edits.setdefault(location, set()).add("drop_close")
                records.append({
                    "action": "strip_close", "pitch": pitch,
                    "line": line + 1,
                })

        if edits:
            _apply_tie_edits(lines, edits)
        if not edits and not respelled:
            break

    for record in records:
        if record["action"].startswith("strip_"):
            logger.warning("tie_repair %s", record)
        else:
            logger.info("tie_repair %s", record)
    return "\n".join(lines), records
# Pitch/Accidental Compatibility (tokenizer pitch grammar: only # and -)
# =============================================================================

# q/Q only: P marks an appoggiatura that occupies real metric time.
_GRACE_MARK_RE = re.compile(r"[qQ]")


def strip_grace_notes(kern_content: str) -> str:
    """Remove grace-note cells (any q/Q subtoken) from all data cells.

    Grace notes are out of scope for the ground truth: they carry no
    metric duration, so they cannot participate in the timeline the
    tokenizer transcribes.  A chord cell is one event with one duration
    slot: a q on any member makes the whole event a grace (sources mix
    q-marked and unmarked members in the same ornament chord, e.g.
    "8qgg# 8g#"), so the whole cell goes — stripping only the q-marked
    members would promote the remainder into a timed note the bar has
    no room for.  Lines left with no content in any column are dropped.

    A grace that opens a tie is the attack of the note it is tied into
    ("[8dq" then "2dd d]"), so its landing loses the close along with it
    and becomes a plain attack: same pitch, same onset, same length,
    since a grace holds no metric time.  Whatever the one-cell forward
    look does not resolve is still left to the later repair pass.
    """
    out_lines: List[str] = []
    # field -> pitches whose tie opened on a grace cell just removed
    orphaned: Dict[int, Set[str]] = {}
    for line in kern_content.split("\n"):
        s = line.rstrip("\n")
        if not s or s.startswith(("*", "!", "=")):
            # Field addresses stop meaning the same thing across a spine
            # operation, so anything still unresolved goes to repair.
            if s.startswith("*") and any(
                    p in _SPINE_OPS for p in s.split("\t")):
                orphaned.clear()
            out_lines.append(line)
            continue
        cols = s.split("\t")
        new_cols: List[str] = []
        for field, cell in enumerate(cols):
            if cell.strip() in (".", ""):
                new_cols.append(cell)
                continue
            if any(_GRACE_MARK_RE.search(sub) for sub in cell.split()):
                for sub in cell.split():
                    if "[" not in sub and "_" not in sub:
                        continue
                    # _subtoken_pitch refuses grace tokens by design, and
                    # a grace is exactly what is being read here.
                    spelled = _PITCH_RE.search(sub)
                    if spelled is not None:
                        orphaned.setdefault(field, set()).add(
                            spelled.group(0).replace("n", ""))
                new_cols.append(".")
                continue
            pending = orphaned.pop(field, None)
            if pending:
                cell = _drop_tie_closes(cell, pending)
            new_cols.append(cell)
        if all(c.strip() in (".", "") for c in new_cols):
            continue
        out_lines.append("\t".join(new_cols))
    return "\n".join(out_lines)


def _drop_tie_closes(cell: str, pitches: Set[str]) -> str:
    """Drop the tie close on every named pitch in one cell."""
    subs = cell.split(" ")
    for index, tok in enumerate(subs):
        if "]" not in tok and "_" not in tok:
            continue
        if _subtoken_pitch(tok) not in pitches:
            continue
        still_open = "[" in tok or "_" in tok
        core = tok.replace("[", "").replace("]", "").replace("_", "")
        subs[index] = ("[" + core) if still_open else core
    return " ".join(subs)


def strip_natural_accidentals(token: str) -> str:
    """
    Strip standalone natural signs from a token.

    The pitch vocabulary has no natural-sign token (redundant given the
    key-signature schema): only # and - are accidentals.

    Natural signs (n) are implicit in standard notation.

    Examples:
        ffn → ff (F-natural = F)
        een → ee (E-natural = E)
        CCn → CC (Contra C-natural = C)

    Note: This should be called AFTER n#/n-/nn normalization in clean_kern.py
    """
    token = token.replace('n#', '#').replace('#n', '#')
    token = token.replace('n-', '-').replace('-n', '-')
    while 'nn' in token:
        token = token.replace('nn', 'n')
    # A natural can occur inside a repeated-letter octave spelling.
    token = re.sub(r'([A-Ga-g])n(?=\1)', r'\1', token)
    # Remove 'n' that appears after pitch letters (A-G or a-g)
    # but before tie markers, accidentals, or end of token.
    return re.sub(r'([A-Ga-g])n(?=[^A-Ga-g]|$)', r'\1', token)


_CANONICAL_KEY_SIGS = {
    '*k[]',
    '*k[f#]', '*k[f#c#]', '*k[f#c#g#]', '*k[f#c#g#d#]',
    '*k[f#c#g#d#a#]', '*k[f#c#g#d#a#e#]', '*k[f#c#g#d#a#e#b#]',
    '*k[b-]', '*k[b-e-]', '*k[b-e-a-]', '*k[b-e-a-d-]',
    '*k[b-e-a-d-g-]', '*k[b-e-a-d-g-c-]', '*k[b-e-a-d-g-c-f-]',
}


def normalize_key_signatures(kern_content: str) -> str:
    """Remove cancellation ink while preserving the active key signature."""
    out = []
    for line in kern_content.split('\n'):
        if not line.startswith('*'):
            out.append(line)
            continue
        fields = []
        for field in line.split('\t'):
            if field == '*k[cancel]':
                # Cancellation is engraving ink, not a change to C major.
                # The ancillary-ink station removes it after cue resolution.
                fields.append(field)
                continue
            if not field.startswith('*k[') or not field.endswith(']') \
                    or field in _CANONICAL_KEY_SIGS:
                fields.append(field)
                continue
            inside = field[3:-1]
            accidentals = []
            i = 0
            while i < len(inside):
                ch = inside[i]
                if ch not in 'abcdefg':
                    i += 1
                    continue
                if i + 1 < len(inside) and inside[i + 1] in '#-':
                    accidentals.append(ch + inside[i + 1])
                    i += 2
                    continue
                i += 2 if i + 1 < len(inside) and inside[i + 1] == 'n' else 1
                if i < len(inside) and inside[i] == '-':
                    i += 1
            candidate = f'*k[{"".join(accidentals)}]'
            fields.append(candidate if candidate in _CANONICAL_KEY_SIGS
                          else field)
        out.append('\t'.join(fields))
    return '\n'.join(out)


_ANCILLARY_PEDAL = {'*ped', '*Xped', '*pedp', '*Xaped'}
_ANCILLARY_OTTAVA_RE = re.compile(
    r'^\*(?:X)?(?:8va|8ba|15ma|15ba|coll8ba)$')
_ANCILLARY_TUPLET = {
    '*tuplet', '*Xtuplet', '*brackettup', '*Xbrackettup', '*tupbreak',
}
_ANCILLARY_KEY_CANCEL_RE = re.compile(
    r'^\*(?:[Xx])?(?:kcancel|cancel)$|^\*k\[cancel\]$')
_ANCILLARY_TONAL_RE = re.compile(r'^\*[A-Ga-g](?:[#-])?:$')
_ANCILLARY_METER_DISPLAY_RE = re.compile(r'^\*met\([^)]*\)$')
_ANCILLARY_PART_RE = re.compile(r'^\*part\d+$')
_ANCILLARY_PLACEMENT_RE = re.compile(
    r'^\*(?:above|below|center)(?::\d+)?$|^\*(?:X)?flip$|'
    r'^\*\d+\\(?:left|right)$')
_ANCILLARY_SECTION_RE = re.compile(
    r'^\*(?:X)?rep$|^\*>.*$|^\*[Ss]$|^\*[AB]\d*$|'
    r'^\*\s*\[[^]]+\]$|^\*\d+$')
_ANCILLARY_TREMOLO = {'*tremolo', '*Xtremolo'}


def _ancillary_interpretation_family(token: str) -> str:
    """Classify a tandem token without assigning musical meaning to unknowns."""
    if token == '*' or token.startswith('**') or token in _SPINE_OPS:
        return 'carry'
    if token in _ANCILLARY_PEDAL:
        return 'pedal'
    if token == '*X':
        # A bare cancel names nothing to cancel; corpus sites are slips
        # riding *Xped / *Xtuplet cancel lines.
        return 'stray-cancel'
    if _ANCILLARY_OTTAVA_RE.fullmatch(token):
        return 'ottava'
    if token == '*rscale' or token.startswith('*rscale:'):
        return 'rscale'
    if token in _ANCILLARY_TUPLET:
        return 'tuplet-display'
    if _ANCILLARY_KEY_CANCEL_RE.fullmatch(token):
        return 'key-cancellation'
    if _ANCILLARY_TONAL_RE.fullmatch(token):
        return 'tonal-declaration'
    if _ANCILLARY_METER_DISPLAY_RE.fullmatch(token):
        return 'meter-display'
    if _ANCILLARY_PART_RE.fullmatch(token):
        return 'part-display'
    if token in {'*LH', '*RH'}:
        return 'hand-display'
    if _ANCILLARY_PLACEMENT_RE.fullmatch(token):
        return 'placement-display'
    if _ANCILLARY_SECTION_RE.fullmatch(token):
        return 'repeat-display'
    if token in _ANCILLARY_TREMOLO:
        return 'tremolo-display'
    if token.startswith('*staff'):
        return 'staff'
    if token.startswith('*I'):
        return 'instrument'
    if token.startswith('*clef'):
        return 'clef'
    if token.startswith('*MM'):
        return 'tempo'
    if re.match(r'^\*(?:X)?cue\b', token):
        return 'cue-display'
    if (token.startswith('*k[') and token.endswith(']')) \
            or re.fullmatch(r'\*M\d+/\d+', token):
        return 'carry'
    return 'unknown-interpretation'


def strip_ancillary_ink(
    kern_content: str,
    *,
    trace: Optional[List[dict]] = None,
) -> str:
    """Remove non-musical ink while retaining timed score controls in place.

    Pitch and duration tokens are never rewritten here. Unknown tandem
    interpretations remain in the score and are reported, so this station
    cannot silently erase a control it does not understand.
    """
    lines = kern_content.split('\n')
    kept: List[str] = []
    counts: Counter = Counter()
    unknowns: Counter = Counter()
    body_metadata: List[Tuple[int, int, str, str]] = []
    header = True

    for row, line in enumerate(lines):
        if line.startswith('!!!'):
            counts['reference-record'] += 1
            continue
        if line.startswith('!'):
            counts['comment'] += 1
            continue
        if not line:
            kept.append(line)
            continue
        if line.startswith('=') or not line.startswith('*'):
            header = False
            kept.append(line)
            continue
        if line.startswith('**'):
            kept.append(line)
            continue

        fields = line.split('\t')
        out_fields = list(fields)
        for field, token in enumerate(fields):
            family = _ancillary_interpretation_family(token.strip())
            if family == 'carry':
                continue
            if family == 'unknown-interpretation':
                unknowns[token] += 1
                continue
            out_fields[field] = '*'
            counts[family] += 1
            if token in {'*pedp', '*Xaped'}:
                counts['pedal-typo'] += 1
            if family in {'staff', 'instrument'} and not header:
                body_metadata.append((row + 1, field + 1, token, family))

        if any(token != '*' for token in out_fields):
            kept.append('\t'.join(out_fields))
        elif out_fields != fields:
            counts['empty-interpretation-line'] += 1
        else:
            kept.append(line)

    stripped = {key: value for key, value in sorted(counts.items()) if value}
    if stripped:
        logger.info('ancillary-ink stripped counts=%s', stripped)
    if body_metadata:
        logger.warning('ancillary-ink body metadata stripped=%s', body_metadata)
    if unknowns:
        logger.warning(
            'ancillary-ink unknown interpretations carried=%s',
            dict(sorted(unknowns.items())),
        )
    if trace is not None:
        trace.append({
            'station': 'ancillary-ink',
            'stripped': stripped,
            'body_metadata': body_metadata,
            'unknown_carried': dict(sorted(unknowns.items())),
        })
    return '\n'.join(kept)


def canonicalize_piano_header(kern_content: str) -> str:
    """Install the deterministic two-staff piano header used by the writer."""
    lines = kern_content.split('\n')
    header = next(
        (index for index, line in enumerate(lines) if line.startswith('**')),
        None,
    )
    if header is None or lines[header].split('\t') != ['**kern', '**kern']:
        raise ValueError("canonical piano score requires exactly two **kern spines")
    fixed = [
        '*staff2\t*staff1',
        '*Ipiano\t*Ipiano',
        '*clefF4\t*clefG2',
    ]
    lines[header + 1:header + 1] = fixed
    return '\n'.join(lines)


def relocate_midbar_key_signatures(kern_content: str) -> str:
    """Move a key declaration after bar data to the following bar head."""
    lines = kern_content.split('\n')
    out: List[str] = []
    saw_data = False
    active_key: Optional[str] = None
    pending_key: Optional[str] = None

    for line_number, line in enumerate(lines, start=1):
        if line.startswith('='):
            if (pending_key is not None and pending_key != active_key
                    and line.startswith('==')):
                raise ValueError(
                    "mid-bar key signature has no following measure "
                    f"(before line {line_number})"
                )
            out.append(line)
            saw_data = False
            if pending_key is not None and pending_key != active_key:
                out.append('\t'.join(
                    [pending_key] * len(line.split('\t'))
                ))
                active_key = pending_key
            if pending_key is not None:
                pending_key = None
            continue

        if line.startswith('*'):
            fields = line.split('\t')
            keys = [field for field in fields if field.startswith('*k[')]
            if saw_data and keys:
                if len(set(keys)) != 1:
                    raise ValueError(
                        "simultaneous key signatures disagree at "
                        f"line {line_number}: {sorted(set(keys))}"
                    )
                pending_key = keys[-1]
                fields = [
                    '*' if field.startswith('*k[') else field
                    for field in fields
                ]
                if any(field != '*' for field in fields):
                    out.append('\t'.join(fields))
                continue
            if keys:
                if len(set(keys)) != 1:
                    raise ValueError(
                        "simultaneous key signatures disagree at "
                        f"line {line_number}: {sorted(set(keys))}"
                    )
                active_key = keys[-1]
            out.append(line)
            continue

        out.append(line)
        if line.strip() and not line.startswith('!'):
            saw_data = True

    if pending_key is not None and pending_key != active_key:
        raise ValueError("mid-bar key signature has no following measure")
    return '\n'.join(out)


def strip_redundant_schema_declarations(kern_content: str) -> str:
    """Drop meter/key rows that restate the declaration already in force.

    Sources re-emit `*M`/`*k[` at system breaks and tempo returns; a
    restatement carries no information and can land inside bar data,
    where the token stream would have to invent a bar-head placement.
    Only rows made solely of `*` and one declaration kind are considered;
    the first declaration of each kind always stays.
    """
    lines = kern_content.split('\n')
    out: List[str] = []
    meter: Optional[str] = None
    key: Optional[str] = None
    for line in lines:
        if line.startswith('*'):
            cells = line.split('\t')
            meters = [c for c in cells
                      if c.startswith('*M') and not c.startswith('*MM')]
            keys = [c for c in cells if c.startswith('*k[')]
            plain = all(c == '*' or c in meters or c in keys for c in cells)
            if plain and meters and not keys:
                if all(c == meter for c in meters):
                    continue
                if len(set(meters)) == 1:
                    meter = meters[0]
            elif plain and keys and not meters:
                if all(c == key for c in keys):
                    continue
                if len(set(keys)) == 1:
                    key = keys[0]
        out.append(line)
    return '\n'.join(out)


def _is_merge_row(cells: List[str]) -> bool:
    return cells.count('*v') >= 2 and all(c in ('*', '*v') for c in cells)


def _is_split_row(cells: List[str]) -> bool:
    return '*^' in cells and all(c in ('*', '*^') for c in cells)


def _net_head_topology(
    width: int, op_rows: List[List[str]],
) -> Optional[Tuple[List[Tuple[int, int]], List[int]]]:
    """Reduce a run of spine operations to tail merges plus head splits.

    Each current column is tracked as the set of original columns it
    holds.  The result is ``(merge_runs, split_counts)``: half-open
    original-column ranges to merge before the barline, then per merged
    column how many copies to open after the declarations.  ``None`` when
    an original column both merges with a neighbour and survives
    elsewhere — that shape has no tail/head equivalent.
    """
    cols: List[frozenset] = [frozenset([i]) for i in range(width)]
    for cells in op_rows:
        if len(cells) != len(cols):
            return None
        out: List[frozenset] = []
        i = 0
        while i < len(cells):
            if cells[i] == '*^':
                out.extend([cols[i], cols[i]])
                i += 1
            elif cells[i] == '*v':
                j = i
                while j < len(cells) and cells[j] == '*v':
                    j += 1
                if j - i < 2:
                    return None
                out.append(frozenset().union(*cols[i:j]))
                i = j
            else:
                out.append(cols[i])
                i += 1
        cols = out
    # Original columns that were merged must appear in exactly one output.
    merged_ids = set()
    for group in cols:
        if len(group) > 1:
            merged_ids |= group
    for gid in merged_ids:
        if sum(1 for group in cols if gid in group) != 1:
            return None
    merge_runs: List[Tuple[int, int]] = []
    for group in cols:
        if len(group) > 1:
            lo, hi = min(group), max(group)
            if hi - lo + 1 != len(group):
                return None
            merge_runs.append((lo, hi + 1))
    merge_runs.sort()
    # After the tail merges the columns are the distinct groups in order;
    # every group must occupy one contiguous block of the output.
    groups: List[frozenset] = []
    for group in cols:
        if groups and groups[-1] == group:
            continue
        if group in groups:
            return None
        groups.append(group)
    expected = []
    i = 0
    while i < width:
        run = next((r for r in merge_runs if r[0] == i), None)
        if run is not None:
            expected.append(frozenset(range(run[0], run[1])))
            i = run[1]
        else:
            expected.append(frozenset([i]))
            i += 1
    if groups != expected:
        return None
    split_counts = [sum(1 for g in cols if g == group) for group in groups]
    return merge_runs, split_counts


def _apply_merge_to_row(cells: List[str], runs: List[Tuple[int, int]]) -> List[str]:
    """Collapse each merged column range of an interpretation/barline row."""
    out: List[str] = []
    i = 0
    for start, end in runs:
        out.extend(cells[i:start])
        group = cells[start:end]
        out.append(next((c for c in group if c != '*'), group[0]))
        i = end
    out.extend(cells[i:])
    return out


def normalize_spine_op_placement(kern_content: str) -> str:
    """Put voice merges on the bar tail and splits after the declarations.

    MusicXML-derived kern writes `*v` rows at the head of the next bar
    (after the barline and any declarations) and sometimes splits a
    column only to merge it back before any data row.  Both readers
    require operations on the bar border with data between them.  The
    operations between a barline and its first data row are reduced to
    their net effect: merges move to just before the barline, splits
    follow the declarations, undone splits vanish, and the barline,
    declaration and comment rows in between are re-widthed.  A shape with
    no such equivalent is left as written.  Column content never changes.
    """
    lines = kern_content.split('\n')
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith('='):
            out.append(line)
            i += 1
            continue
        # Collect the head segment: everything up to the first data row.
        j = i + 1
        while j < n and (not lines[j] or lines[j].startswith(('!', '*'))) \
                and not lines[j].startswith('*-'):
            j += 1
        segment = lines[i + 1:j]
        op_idx = [k for k, ln in enumerate(segment)
                  if _is_split_row(ln.split('\t')) or _is_merge_row(ln.split('\t'))]
        head_merge = any(_is_merge_row(segment[k].split('\t')) for k in op_idx)
        if not op_idx or not head_merge:
            out.append(line)
            out.extend(segment)
            i = j
            continue
        width = len(line.split('\t'))
        net = _net_head_topology(width, [segment[k].split('\t') for k in op_idx])
        if net is None:
            out.append(line)
            out.extend(segment)
            i = j
            continue
        merge_runs, split_counts = net
        cur = width - sum(end - start - 1 for start, end in merge_runs)
        # One row per run: adjacent runs in a single row would read as
        # one wider merge.
        shrink = 0
        for start, end in merge_runs:
            row = ['*'] * (width - shrink)
            for c in range(start - shrink, end - shrink):
                row[c] = '*v'
            out.append('\t'.join(row))
            shrink += end - start - 1
        out.append('\t'.join(_apply_merge_to_row(line.split('\t'), merge_runs)))
        # Declaration and comment rows keep their order; each is collapsed
        # onto the post-merge columns from the column sets live at its row.
        groups: List[frozenset] = []
        c = 0
        while c < width:
            run = next((r for r in merge_runs if r[0] == c), None)
            if run is not None:
                groups.append(frozenset(range(run[0], run[1])))
                c = run[1]
            else:
                groups.append(frozenset([c]))
                c += 1
        cols: List[frozenset] = [frozenset([k]) for k in range(width)]
        for k, ln in enumerate(segment):
            cells = ln.split('\t')
            if k in op_idx:
                nxt: List[frozenset] = []
                ci = 0
                while ci < len(cells):
                    if cells[ci] == '*^':
                        nxt.extend([cols[ci], cols[ci]])
                        ci += 1
                    elif cells[ci] == '*v':
                        cj = ci
                        while cj < len(cells) and cells[cj] == '*v':
                            cj += 1
                        nxt.append(frozenset().union(*cols[ci:cj]))
                        ci = cj
                    else:
                        nxt.append(cols[ci])
                        ci += 1
                cols = nxt
                continue
            if ln and ln.startswith(('!', '*')) and len(cells) == len(cols):
                collapsed = []
                for group in groups:
                    cand = [cell for col, cell in zip(cols, cells) if col <= group]
                    collapsed.append(
                        next((x for x in cand if x not in ('*', '!')), cand[0] if cand else '*'))
                out.append('\t'.join(collapsed))
            else:
                out.append(ln)
        # Head splits: right-comb, one row per extra copy.
        cols = [1] * cur
        for c, count in enumerate(split_counts):
            for _ in range(count - 1):
                row = []
                for cc, copies in enumerate(cols):
                    row.extend(['*'] * copies)
                    if cc == c:
                        row[-1] = '*^'
                out.append('\t'.join(row))
                cols[c] += 1
        i = j
    return '\n'.join(out)


def relocate_empty_pickup_schema(kern_content: str) -> str:
    """Attach initial schema to the first real bar when no pickup sounds."""
    lines = kern_content.split('\n')
    first_bar = next(
        (index for index, line in enumerate(lines) if line.startswith('=')),
        None,
    )
    if first_bar is None or any(
            line and not line.startswith(('*', '!', '='))
            for line in lines[:first_bar]):
        return kern_content

    meter_rows = []
    key_rows = []
    remove = set()
    for index, line in enumerate(lines[:first_bar]):
        if not line.startswith('*'):
            continue
        cells = line.split('\t')
        if any(cell.startswith('*M') and not cell.startswith('*MM')
               for cell in cells) \
                and all(cell == '*' or (
                    cell.startswith('*M') and not cell.startswith('*MM'))
                        for cell in cells):
            meter_rows.append(line)
            remove.add(index)
        elif any(cell.startswith('*k[') for cell in cells) \
                and all(cell == '*' or cell.startswith('*k[')
                        for cell in cells):
            key_rows.append(line)
            remove.add(index)
    if not remove:
        return kern_content

    first_bar -= sum(index < first_bar for index in remove)
    kept = [line for index, line in enumerate(lines) if index not in remove]
    kept[first_bar + 1:first_bar + 1] = meter_rows + key_rows
    return '\n'.join(kept)


# Enharmonic conversion tables for double accidentals
# Double sharp: pitch## → next semitone (natural or sharp)
DOUBLE_SHARP_MAP = {
    # Uppercase (Great octave and below)
    'C': 'D', 'D': 'E', 'E': 'F#', 'F': 'G', 'G': 'A', 'A': 'B', 'B': 'C#',
    # Lowercase (Small octave and above)
    'c': 'd', 'd': 'e', 'e': 'f#', 'f': 'g', 'g': 'a', 'a': 'b', 'b': 'c#',
}

# Double flat: pitch-- → prev semitone (natural or flat)
DOUBLE_FLAT_MAP = {
    # Uppercase
    'C': 'B-', 'D': 'C', 'E': 'D', 'F': 'E-', 'G': 'F', 'A': 'G', 'B': 'A',
    # Lowercase
    'c': 'b-', 'd': 'c', 'e': 'd', 'f': 'e-', 'g': 'f', 'a': 'g', 'b': 'a',
}


def convert_double_accidentals(token: str) -> str:
    """
    Convert double sharps (##) and double flats (--) to enharmonic equivalents.

    The pitch vocabulary has only single # and - accidentals.

    Examples:
        4f## → 4g (F double-sharp = G)
        4FF## → 4GG (Great F double-sharp = Great G)
        8b-- → 8a (B double-flat = A)
        8BB-- → 8AA (Great B double-flat = Great A)

    Note: Handles octave notation (repeated letters like CC, ccc)
    """
    # Match: optional prefix + duration + pitch letters + ## or --
    # Pattern: (prefix)(duration)(pitch_letters)(##|--)

    def replace_double_sharp_pitch(pitch_letters: str) -> str:
        """Convert pitch## to enharmonic equivalent (just the pitch part).

        B## crosses an octave boundary (B3→C#4), handled by case/count shift.
        """
        base = pitch_letters[0]
        octave_count = len(pitch_letters)
        is_upper = base.isupper()

        if base.upper() not in DOUBLE_SHARP_MAP:
            return pitch_letters + '##'

        replacement = DOUBLE_SHARP_MAP[base.upper()]
        new_base_letter = replacement[0]
        new_accidental = replacement[1:] if len(replacement) > 1 else ''

        crosses_up = (base.upper() == 'B')
        if crosses_up:
            if is_upper:
                if octave_count == 1:
                    return new_base_letter.lower() + new_accidental
                else:
                    return new_base_letter * (octave_count - 1) + new_accidental
            else:
                return new_base_letter.lower() * (octave_count + 1) + new_accidental
        else:
            if is_upper:
                return new_base_letter * octave_count + new_accidental
            else:
                return new_base_letter.lower() * octave_count + new_accidental

    def replace_double_flat_pitch(pitch_letters: str) -> str:
        """Convert pitch-- to enharmonic equivalent (just the pitch part).

        C-- crosses an octave boundary (C4→Bb3), handled by case/count shift.
        """
        base = pitch_letters[0]
        octave_count = len(pitch_letters)
        is_upper = base.isupper()

        if base.upper() not in DOUBLE_FLAT_MAP:
            return pitch_letters + '--'

        replacement = DOUBLE_FLAT_MAP[base.upper()]
        new_base_letter = replacement[0]
        new_accidental = replacement[1:] if len(replacement) > 1 else ''

        crosses_down = (base.upper() == 'C')
        if crosses_down:
            if is_upper:
                return new_base_letter * (octave_count + 1) + new_accidental
            else:
                if octave_count == 1:
                    return new_base_letter.upper() + new_accidental
                else:
                    return new_base_letter.lower() * (octave_count - 1) + new_accidental
        else:
            if is_upper:
                return new_base_letter * octave_count + new_accidental
            else:
                return new_base_letter.lower() * octave_count + new_accidental

    # Double sharp: pitch## → enharmonic
    # Simplified approach: match pitch letters followed by ## anywhere in the token
    # This handles: 4f##, 4..ff##, <unk>FF##, [BBB##, 16qqf##, etc.
    token = re.sub(
        r'([A-Ga-g]+)##',
        lambda m: replace_double_sharp_pitch(m.group(1)),
        token
    )

    # Double flat: pitch-- → enharmonic
    # Same simplified approach
    token = re.sub(
        r'([A-Ga-g]+)--',
        lambda m: replace_double_flat_pitch(m.group(1)),
        token
    )

    return token


def remove_slur_phrase_markers(token: str) -> str:
    """
    Remove slur and phrase markers from a token.

    The vocabulary has no slur/phrase tokens:
        ( ) = slur start/end (legato phrasing)
        { } = phrase start/end (musical phrase boundaries)
    """
    return re.sub(r'[(){}]', '', token)


_REST_RECIP_RE = re.compile(r'\d+(?:%\d+)?\.*r')


def strip_rest_tie_marks(token: str) -> str:
    """Remove tie syntax from metric rests while preserving signifiers."""
    if not _REST_RECIP_RE.search(token):
        return token
    return token.replace("[", "").replace("]", "").replace("_", "")


def normalize_odd_spellings(token: str) -> str:
    """Canonicalize rare source spellings that read as the wrong sound.

    ``4Cr`` is a rest with its staff-position letter BEFORE the r (the
    usual hint form is ``4rC``); read naively it becomes a note C.
    ``4e]-`` carries its accidental after the tie-closer; a pitch scan
    that stops at the glyph reads E-natural instead of E-flat.  The
    renderer reads both as intended, so the text moves to the plain
    spelling before any pitch-keyed pass runs.
    """
    token = re.sub(r'(\d+(?:%\d+)?\.*)[A-Ga-g]+(?=r)', r'\1', token)
    token = re.sub(r'(\d+(?:%\d+)?)(r)(\.+)', r'\1\3\2', token)
    token = re.sub(r'([a-gA-G]+)([\]_]+)([#\-n]+)', r'\1\3\2', token)
    token = re.sub(
        r'(\d+(?:%\d+)?)([a-gA-G]+[#\-n]*)(\.+)',
        r'\1\3\2',
        token,
    )
    token = re.sub(
        r'^([^\d]*)(\d+(?:%\d+)?\.*)\[',
        r'\1[\2',
        token,
    )

    # A sharp and a flat on the same note cancel to a natural: the
    # renderer already reads the pair that way, and every pitch-keyed
    # pass must agree with what sounds.
    def _cancel(m):
        acc = m.group(2)
        net = acc.count('#') - acc.count('-')
        return m.group(1) + ('#' * net if net > 0 else '-' * -net)

    token = re.sub(r'([a-gA-G]+)((?=[#\-n]*#)(?=[#\-n]*-)[#\-n]+)',
                   _cancel, token)
    return token


# =============================================================================
# Pitch Token Mapping
# =============================================================================

def _map_kern_tokens(sequence: str, fn) -> str:
    """Apply fn to every pitch token in a kern sequence.

    Skips control lines (=, *, !) and null tokens; splits chord tokens
    (space-separated notes in one field) and applies fn per note.
    """
    lines = sequence.split('\n')
    result_lines = []

    for line in lines:
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            result_lines.append(line)
            continue

        tokens = line.split('\t') if '\t' in line else [line]
        processed_tokens = []

        for token in tokens:
            if token == '.' or not token:
                processed_tokens.append(token)
                continue

            if ' ' in token:
                notes = token.split(' ')
                processed_notes = [n if (n == '.' or not n) else fn(n) for n in notes]
                processed_tokens.append(' '.join(processed_notes))
            else:
                processed_tokens.append(fn(token))

        result_lines.append('\t'.join(processed_tokens) if len(processed_tokens) > 1 else processed_tokens[0])

    return '\n'.join(result_lines)


# =============================================================================
# Mixed-duration chord normalization
# =============================================================================


def normalize_chord_durations(kern_content: str) -> str:
    """Canonicalize mixed-duration chord cells through voice resynthesis.

    The source cell already declares one voice.  Its member intervals are
    therefore serialized by the same engine used after a layer fusion:
    every source attack and release remains independent, continuations gain
    ties, and silence covered by a ringing member disappears.
    """
    from src.score.sanitize_kern import parse_kern_duration
    original = kern_content
    lines = kern_content.split('\n')

    def mixed_dyadic(cell: str) -> bool:
        if cell == '.' or ' ' not in cell:
            return False
        vals = []
        for sub in cell.split():
            if re.match(r'[\d.%]*r', sub.lstrip('[({')):
                return False
            dur = parse_kern_duration(sub)
            if dur is None or dur <= 0 or (dur * 128).denominator != 1:
                return False
            vals.append(dur)
        return len(set(vals)) > 1

    attempted: set = set()
    for _ in range(256):
        tracked = _column_lives(lines)
        if tracked is None:
            return original
        lives, _ = tracked
        target = None
        for li, line in enumerate(lines):
            if not line or line.startswith(('!', '*', '=')):
                continue
            for fi, cell in enumerate(line.split('\t')):
                if mixed_dyadic(cell):
                    target = (li, fi)
                    break
            if target is not None:
                break
        if target is None:
            return '\n'.join(lines)

        li, fi = target
        life = next((lv for lv in lives.values()
                     if lv['pos'].get(li) == fi), None)
        if life is None:
            return '\n'.join(lines)
        bar_end = next((j for j in range(li + 1, len(lines))
                        if lines[j].startswith('=')), len(lines))
        dls = [j for j in sorted(life['pos'])
               if li <= j < bar_end and lines[j]
               and not lines[j].startswith(('!', '*', '='))]
        pos = {j: life['pos'][j] for j in dls}
        # The resynthesis writes only into rows this column already has,
        # so an early release landing between two rows has nowhere to put
        # its tie tail.  Give it one: a blank data row is a kern no-op for
        # every other column, and the cell's own clock step (its shortest
        # member) means the row was missing from the bar's arithmetic too.
        inserted = False
        g_all = _global_times(lines)
        if g_all is not None and li in g_all:
            durs = {parse_kern_duration(sub)
                    for sub in lines[li].split('\t')[fi].split(' ')}
            durs = {d for d in durs if d is not None and d > 0}
            if durs:
                have = {g_all[j] for j in dls if j in g_all}
                longest = max(durs)
                want = sorted({g_all[li] + d for d in durs if d < longest}
                              - have - attempted)
                for t_want in reversed(want):
                    attempted.add(t_want)
                    anchor = next((j for j in dls
                                   if j in g_all and g_all[j] > t_want),
                                  bar_end)
                    prev = next((j for j in reversed(range(anchor))
                                 if lines[j] and not lines[j].startswith(
                                     ('!', '*', '='))), None)
                    if prev is None:
                        continue
                    ncols = len(lines[prev].split('\t'))
                    lines.insert(anchor, '\t'.join(['.'] * ncols))
                    inserted = True
        if inserted:
            continue
        cells = _build_resynth_voice_cells(
            lines,
            dls,
            [life['pos']],
            continuation_positions=[
                candidate['pos']
                for candidate in lives.values()
                if candidate['hand'] == life['hand']
            ],
        )
        if cells is None:
            return '\n'.join(lines)
        changed = False
        for dl, cell in cells.items():
            row = lines[dl].split('\t')
            f = pos[dl]
            if row[f] != cell:
                row[f] = cell
                lines[dl] = '\t'.join(row)
                changed = True
        if not changed:
            return '\n'.join(lines)

    raise ValueError("mixed-duration chord normalization did not converge")


# =============================================================================
# Column Lives
# =============================================================================
#
# Voice merging itself lives in merge_voice (account-based engine);
# these column trackers remain as topology readers for the spelling
# passes, which need to know which written column a cell belongs to.

def _column_lives(lines: List[str]):
    """Track every kern column from birth to death through spine ops.

    Returns (lives, runs) or None when an op line desyncs from the
    tracked layout.  lives: id -> {hand, birth, birth_field, parent,
    death, death_field, survivor, pos, touched}.  ``pos`` maps every
    spine-parallel line index (data, barline, local comment, op line
    under its pre-op layout) to the column's field index.  ``touched``
    holds op lines where the column FORKS (it splits, is
    exchanged, or gains an inserted neighbour) — an absorbed column must
    have none inside the fusion window.  ``absorbs`` holds op lines
    where the column survives a *v run mid-life; fusing such a column is
    legal when the fusion target sits immediately left, so the run
    participation can be handed over.
    runs: [{line, members: [(field, id)]}] for every *v run with two or
    more live members.
    """
    lives: Dict[int, dict] = {}
    runs: List[dict] = []
    cur: List[Optional[int]] = []
    next_id = [0]

    def new_life(hand: int, birth: int, bfield: Optional[int],
                 parent: Optional[int]) -> int:
        cid = next_id[0]
        next_id[0] += 1
        lives[cid] = {"id": cid, "hand": hand, "birth": birth,
                      "birth_field": bfield, "parent": parent,
                      "death": None, "death_field": None,
                      "survivor": None, "pos": {}, "touched": set(),
                      "absorbs": set()}
        return cid

    for idx, ln in enumerate(lines):
        if not ln or ln.startswith('!!'):
            continue
        toks = _cells(ln)
        if ln.startswith('**'):
            cur = []
            k = 0
            for t in toks:
                if t == '**kern':
                    cur.append(new_life(k, idx, None, None))
                    k += 1
                else:
                    cur.append(None)
            continue
        if not cur:
            continue
        is_op = ln.startswith('*') and any(
            t in ('*^', '*v', '*-', '*+', '*x') for t in toks)
        for fi, cid in enumerate(cur):
            if cid is not None:
                lives[cid]["pos"][idx] = fi
        if not is_op:
            continue
        if len(toks) != len(cur):
            return None
        new: List[Optional[int]] = []
        fi = 0
        while fi < len(toks):
            t = toks[fi]
            cid = cur[fi] if fi < len(cur) else None
            if t == '*^':
                if cid is None:
                    new += [None, None]
                else:
                    lives[cid]["touched"].add(idx)
                    child = new_life(lives[cid]["hand"], idx, fi, cid)
                    new += [cid, child]
                fi += 1
            elif t == '*v':
                run = []
                while fi < len(toks) and toks[fi] == '*v':
                    run.append((fi, cur[fi] if fi < len(cur) else None))
                    fi += 1
                live_run = [(f, c) for f, c in run if c is not None]
                if live_run:
                    surv = live_run[0][1]
                    for f, c in live_run[1:]:
                        lives[c]["death"] = idx
                        lives[c]["death_field"] = f
                        lives[c]["survivor"] = surv
                    if len(live_run) >= 2:
                        lives[surv]["absorbs"].add(idx)
                        runs.append({"line": idx, "members": live_run})
                    new.append(surv)
                else:
                    new.append(None)
            elif t == '*-':
                if cid is not None:
                    lives[cid]["death"] = idx
                    lives[cid]["death_field"] = fi
                fi += 1
            elif t == '*x':
                nxt = cur[fi + 1] if fi + 1 < len(cur) else None
                for c in (cid, nxt):
                    if c is not None:
                        lives[c]["touched"].add(idx)
                new.append(nxt)
                new.append(cid)
                fi += 2
            elif t == '*+':
                if cid is not None:
                    lives[cid]["touched"].add(idx)
                new.append(cid)
                new.append(None)
                fi += 1
            else:
                new.append(cid)
                fi += 1
        cur = new
    return lives, runs


@lru_cache(maxsize=8)
def _global_times_cached(lines_key: tuple) -> Optional[Dict[int, Fraction]]:
    lines = list(lines_key)
    all_dls = [i for i, ln in enumerate(lines)
               if ln and not ln.startswith(('!', '*', '='))]
    if not all_dls:
        return None
    lt_all = _line_times(lines, all_dls)
    if lt_all is None:
        return None
    return lt_all[0]


def _global_times(lines: List[str]) -> Optional[Dict[int, Fraction]]:
    """Absolute onset of every data line from the file-global clock, or
    None when the clock is unparseable somewhere.  Unlike a clock
    started inside a window, this one knows every column's mid-ring
    state, so mid-bar window rows keep true onsets.

    Memoized because several spelling passes ask for the same clock and
    rebuilding it walks the whole file.  A fresh dict is returned so callers
    cannot mutate the cached one (Fraction values are immutable)."""
    res = _global_times_cached(tuple(lines))
    return dict(res) if res is not None else None


def _in_bar_onsets(lines: List[str],
                   g_times: Optional[Dict[int, Fraction]] = None,
                   ) -> Optional[Dict[int, Fraction]]:
    """In-bar onset of every data line, from the file-global clock."""
    if g_times is None:
        g_times = _global_times(lines)
    if g_times is None:
        return None
    out: Dict[int, Fraction] = {}
    head_t = Fraction(0)
    prev = None
    for dl in sorted(g_times):
        if prev is None or any(lines[r].startswith('=')
                               for r in range(prev + 1, dl)):
            head_t = g_times[dl]
        out[dl] = g_times[dl] - head_t
        prev = dl
    return out


@lru_cache(maxsize=None)
def _is_rest_cell(cell: str) -> bool:
    """True if *cell* is a rest token (not '.' and not a note).

    Kern rests: duration + 'r' + optional pitch-position suffix (e.g.
    ``4rA``, ``8rc``).  The suffix letters are staff placement hints,
    not pitched notes.

    Cached because several spelling passes revisit the same cells.
    """
    if cell == '.' or not cell:
        return False
    first = cell.split()[0] if ' ' in cell else cell
    return bool(re.match(r'[\d.%]*r', first))


def _dur_to_kern_recip(dur: Fraction) -> Optional[str]:
    """Duration (quarter notes) → kern reciprocal string, or None.

    Only sane reciprocals come back — powers of two and their triplet
    row (3·2^k): a stray gap in an irregular-tuplet region would
    otherwise "solve" to a nonsense value like 136."""
    if dur <= 0:
        return None
    if dur == Fraction(8):
        return "0"
    for dots in range(4):
        multiplier = Fraction(2) - Fraction(1, 2**dots)
        recip_frac = Fraction(4) * multiplier / dur
        if recip_frac.denominator == 1 and int(recip_frac) > 0:
            r = int(recip_frac)
            base = r
            while base % 2 == 0:
                base //= 2
            if base not in (1, 3):
                continue
            return str(r) + '.' * dots
    return None


def _token_rhythm(cell: str) -> Optional[Fraction]:
    """Duration of a data cell (note or rest); None for null token '.'."""
    if cell == '.' or not cell:
        return None
    from src.score.sanitize_kern import parse_kern_duration
    parts = cell.split()
    if not parts:
        return None
    return parse_kern_duration(parts[0])


@lru_cache(maxsize=None)
def _cell_step_rhythm(cell: str) -> Optional[Fraction]:
    """Clock advance of a kern cell, including mixed-duration chords.

    Cached because several spelling passes revisit the same cells.
    """
    if cell == '.' or not cell:
        return None
    from src.score.sanitize_kern import parse_kern_duration
    durations = [parse_kern_duration(part) for part in cell.split()]
    durations = [d for d in durations if d is not None and d > 0]
    return min(durations) if durations else None


def _line_meter(line: str) -> Optional[Tuple[int, int]]:
    """Meter declared on an interpretation line, scanning every column."""
    if not line.startswith('*'):
        return None
    for part in line.split('\t'):
        m = _METER_RE.match(part)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def absorb_empty_bars(kern_content: str) -> str:
    """A zero-data bar occupies no time: its successor absorbs it and
    inherits the interpretations declared at its head."""
    lines = kern_content.splitlines()

    def is_barline(line: str) -> bool:
        return line.startswith('=')

    def is_terminal(line: str) -> bool:
        return line.startswith('==')

    def is_data(line: str) -> bool:
        return bool(line) and line[0] not in '*!='

    changed = True
    while changed:
        changed = False
        bar_rows = [row for row, line in enumerate(lines)
                    if is_barline(line)]
        for left, right in zip(bar_rows, bar_rows[1:]):
            if any(is_data(lines[row]) for row in range(left + 1, right)):
                continue
            drop = left if is_terminal(lines[right]) else right
            del lines[drop]
            changed = True
            break
    return '\n'.join(lines) + ('\n' if kern_content.endswith('\n') else '')


def complete_boundary_bars(kern_content: str) -> str:
    """Rest-fill the opening pickup and the final bar.

    Opening pickup: leading rests so the upbeat keeps its metric
    position.  Final bar: trailing rests per hand.  Interior short
    bars are data errors for the tokenizer to refuse, never silently
    repaired.  Returns input unchanged when the target bar cannot be
    measured cleanly.
    """
    from fractions import Fraction
    from src.score.kern_utils import _col_first_dur, _spell_frac

    lines = kern_content.split('\n')

    def is_data(s: str) -> bool:
        return bool(s.strip()) and not s.startswith(('!', '*', '='))

    data_idx = [i for i, l in enumerate(lines) if is_data(l)]
    if not data_idx:
        return kern_content
    bar_idx = [i for i, l in enumerate(lines) if l.startswith('=')]

    def meter_before(idx: int) -> Optional[Tuple[int, int]]:
        cur = None
        for l in lines[:idx]:
            m = _line_meter(l)
            if m:
                cur = m
        return cur

    def region(lo: int, hi: int) -> List[int]:
        return [i for i in data_idx if lo <= i < hi]

    def col_spans(idxs: List[int]) -> Optional[List[Fraction]]:
        ncol = len(lines[idxs[0]].split('\t'))
        if any(len(lines[i].split('\t')) != ncol for i in idxs):
            return None
        clocks = [Fraction(0)] * ncol
        for i in idxs:
            cells = lines[i].split('\t')
            for c, cell in enumerate(cells):
                if cell != '.':
                    duration = _cell_step_rhythm(_STEM_RE.sub('', cell))
                    if duration is None:
                        return None
                    clocks[c] += duration
        return clocks

    def rest_line(recip: str, ncol: int) -> str:
        return '\t'.join([f'{recip}r'] * ncol)

    def split_span(lo: int, hi: int, idxs: List[int],
                   ncol: int) -> Optional[List[Fraction]]:
        """Extent of a bar whose column count changes part way through.

        The per-column clock cannot follow ``*^``/``*v``, so a bar that
        divides a hand mid-way is measured as a whole instead, and only
        where it closes on the layout it opened with — otherwise the
        rests would have nowhere to land.
        """
        from src.score.expand_repeat import _ColClock
        if len(lines[idxs[-1]].split('\t')) != ncol:
            return None
        clock = _ColClock(ncol)
        try:
            for i in range(lo, hi):
                clock.feed(lines[i])
        except Exception:
            return None
        # The clock counts whole notes; everything here is in quarters.
        return [clock.span() * 4] * ncol

    # ── opening pickup ──────────────────────────────────────────────
    first_bl = bar_idx[0] if bar_idx else len(lines)
    if data_idx[0] < first_bl:
        r1 = region(0, first_bl)
    else:
        nxt = bar_idx[1] if len(bar_idx) > 1 else len(lines)
        r1 = region(first_bl, nxt)
    pad_insert = None
    m1 = meter_before(r1[0]) if r1 else None
    if r1 and m1:
        bar_len = Fraction(4 * m1[0], m1[1])
        spans = col_spans(r1)
        if spans is not None:
            span = max(spans)
            pad = bar_len - span
            if 0 < pad < bar_len:
                try:
                    pieces = _spell_frac(pad)
                except ValueError:
                    pieces = None
                if pieces:
                    ncol = len(lines[r1[0]].split('\t'))
                    pad_insert = (
                        r1[0], [rest_line(rc, ncol) for _, rc in pieces],
                    )

    # ── final bar ───────────────────────────────────────────────────
    last_data = data_idx[-1]
    prev_bl = max((i for i in bar_idx if i < last_data), default=-1)
    next_bl = min((i for i in bar_idx if i > last_data), default=len(lines))
    rl = region(prev_bl + 1, next_bl)
    fill_edit = None
    ml = meter_before(rl[0]) if rl else None
    if rl and ml:
        bar_len = Fraction(4 * ml[0], ml[1])
        ends = col_spans(rl)
        if ends is None and prev_bl >= 0:
            ends = split_span(prev_bl + 1, next_bl, rl,
                              len(lines[prev_bl].split('\t')))
        if ends is not None and any(0 < bar_len - e for e in ends) \
                and all(e <= bar_len for e in ends):
            ncol = len(ends)
            if len(set(ends)) == 1:
                try:
                    pieces = _spell_frac(bar_len - ends[0])
                except ValueError:
                    pieces = None
                if pieces:
                    fill_edit = (
                        'append', rl[-1],
                        [rest_line(rc, ncol) for _, rc in pieces],
                    )
            else:
                from src.score.kern_utils import _token_duration_frac
                times: List[Fraction] = []
                clocks = [Fraction(0)] * ncol
                ok = True
                for i in rl:
                    cells = lines[i].split('\t')
                    emit = [c for c, cell in enumerate(cells)
                            if cell != '.']
                    if not emit:
                        ok = False
                        break
                    for c in emit:
                        durs = {_token_duration_frac(p)
                                for p in cells[c].split()
                                if _token_duration_frac(p) > 0}
                        if len(durs) > 1:
                            ok = False
                            break
                    if not ok:
                        break
                    t = min(clocks[c] for c in emit)
                    times.append(t)
                    for c in emit:
                        clocks[c] += _col_first_dur(cells[c])
                if ok:
                    inserts: List[Tuple[Fraction, int, str]] = []
                    for c, e in enumerate(ends):
                        gap = bar_len - e
                        if gap <= 0:
                            continue
                        try:
                            pieces = _spell_frac(gap)
                        except ValueError:
                            ok = False
                            break
                        t = e
                        for pdur, rc in pieces:
                            inserts.append((t, c, f'{rc}r'))
                            t += pdur
                    if ok and inserts:
                        fill_edit = ('timed', rl, times, inserts, ncol)

    if fill_edit:
        if fill_edit[0] == 'append':
            _, at, new_lines = fill_edit
            lines[at + 1:at + 1] = new_lines
        else:
            _, rl, times, inserts, ncol = fill_edit
            by_time: Dict[Fraction, List[Tuple[int, str]]] = {}
            for t, c, tok in inserts:
                by_time.setdefault(t, []).append((c, tok))
            for t in sorted(by_time, reverse=True):
                pending = []
                for c, tok in by_time[t]:
                    patched = False
                    for k, i in enumerate(rl):
                        cells = lines[i].split('\t')
                        if times[k] == t and cells[c] == '.':
                            cells[c] = tok
                            lines[i] = '\t'.join(cells)
                            patched = True
                            break
                    if not patched:
                        pending.append((c, tok))
                if pending:
                    cells = ['.'] * ncol
                    for c, tok in pending:
                        cells[c] = tok
                    k = next((k for k, tt in enumerate(times) if tt > t),
                             None)
                    pos = rl[k] if k is not None else rl[-1] + 1
                    lines.insert(pos, '\t'.join(cells))

    if pad_insert:
        at, new_lines = pad_insert
        lines[at:at] = new_lines

    return '\n'.join(lines)


def _line_times(
    lines: List[str], data_lines: List[int]
) -> Optional[Tuple[Dict[int, Fraction], Fraction]]:
    """Onset time of each data line plus the span's total length,
    derived from every column's durations.

    None when a duration is unparseable or the column count shifts
    mid-span (a spine op moved columns).
    """
    if not data_lines:
        return None
    ncols = len(lines[data_lines[0]].split('\t'))
    rem = [Fraction(0)] * ncols
    t = Fraction(0)
    times: Dict[int, Fraction] = {}
    dset = set(data_lines)
    for idx in range(data_lines[0], data_lines[-1] + 1):
        ln = lines[idx]
        if not ln or ln.startswith('!') or ln.startswith('='):
            continue
        toks = _cells(ln)
        if ln.startswith('*'):
            if '*^' in toks or '*v' in toks:
                # carry each column's remaining ring through the op
                new_rem: List[Fraction] = []
                oi = 0
                while oi < len(toks):
                    tk = toks[oi]
                    if tk == '*^':
                        r = rem[oi] if oi < len(rem) else Fraction(0)
                        new_rem.extend([r, r])
                        oi += 1
                    elif tk == '*v':
                        run: List[Fraction] = []
                        while oi < len(toks) and toks[oi] == '*v':
                            if oi < len(rem):
                                run.append(rem[oi])
                            oi += 1
                        new_rem.append(max(run) if run else Fraction(0))
                    else:
                        new_rem.append(rem[oi] if oi < len(rem) else Fraction(0))
                        oi += 1
                rem = new_rem
                ncols = len(rem)
            continue
        if idx not in dset:
            return None
        if len(toks) != ncols:
            return None
        times[idx] = t
        for ci, cell in enumerate(toks):
            if cell != '.':
                d = _cell_step_rhythm(cell)
                if d is not None:
                    rem[ci] = d
        pending = [r for r in rem if r > 0]
        if not pending:
            return None
        step = min(pending)
        rem = [r - step if r > 0 else r for r in rem]
        t += step
    # remaining ring after the last line belongs to the span
    tail = max((r for r in rem if r > 0), default=Fraction(0))
    return times, t + tail


def _kern_pitch_height(pitch: str) -> int:
    """Semitone height of a kern pitch string, for chord-cell ordering."""
    m = re.match(r'([a-gA-G]+)([#\-n]*)', pitch)
    letters, accs = m.group(1), m.group(2)
    ch = letters[0]
    octave = 3 + len(letters) if ch.islower() else 4 - len(letters)
    semis = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
    return octave * 12 + semis[ch.lower()] + accs.count('#') - accs.count('-')


def _column_events(
    lines: List[str], data_lines: List[int],
    times: Dict[int, Fraction], pos: Dict[int, int],
    span_end: Optional[Fraction]
) -> Optional[list]:
    """Sound events of one column: (pitch, start, end, chain openness).

    Tie chains collapse into one event; a chain still open at the span
    edge keeps its openness (the merged column's neighbours continue
    it).  Rests carry no sound and yield no event.  None on anything
    unparseable — the caller then leaves the pair untouched.
    """
    from src.score.sanitize_kern import parse_kern_duration
    events: list = []
    open_chains: Dict[str, dict] = {}
    for dl in data_lines:
        toks = _cells(lines[dl])
        p = pos[dl]
        if p >= len(toks):
            return None
        cell = toks[p]
        if cell == '.':
            continue
        t = times[dl]
        subs = cell.split(' ')
        fixed = []
        lead = None
        for sub in subs:
            if not sub:
                return None
            if re.search(r"\d", sub):
                # Prefix class includes stem ink: a mark before the
                # digits must not turn a timed member into a "bare
                # pitch" that inherits a second duration.
                m0 = re.match(r"^[\[({/\\]*(\d+(?:%\d+)?\.*)", sub)
                lead = m0.group(1) if m0 else None
                fixed.append(sub)
            elif lead and re.search(r"[a-gA-Gr]", sub):
                # chord shorthand: a bare pitch inherits the lead recip
                fixed.append(re.sub(r"^([\[({/\\]*)", r"\g<1>" + lead,
                                    sub, count=1))
            else:
                fixed.append(sub)
        for sub in fixed:
            dur = parse_kern_duration(sub)
            if dur is None or dur <= 0:
                return None
            if re.match(r'[\d.%]*r', sub):
                continue
            m = _PITCH_RE.search(sub)
            if m is None:
                return None
            pitch = m.group(0)
            stem = '/' if '/' in sub else ('\\' if '\\' in sub else '')
            opens = '[' in sub
            closes = ']' in sub
            cont = '_' in sub
            if opens and (closes or cont):
                # '[X]' / '[X_' — the glyph both arrives and departs
                # (canonical mid-chain spelling): extend an open chain
                # and keep it open, or start one if none is.
                ev = open_chains.get(pitch)
                if ev is not None:
                    if ev['end'] != t:
                        return None
                    ev['end'] = t + dur
                else:
                    ev = {'pitch': pitch, 'start': t, 'end': t + dur,
                          'open_start': False, 'open_end': False,
                          'stem': stem}
                    events.append(ev)
                    open_chains[pitch] = ev
                continue
            if cont or closes:
                ev = open_chains.get(pitch)
                if ev is not None:
                    if ev['end'] != t:
                        return None
                    ev['end'] = t + dur
                    if closes:
                        open_chains.pop(pitch)
                else:
                    # chain entered from before the span
                    ev = {'pitch': pitch, 'start': t, 'end': t + dur,
                          'open_start': True, 'open_end': False,
                          'stem': stem}
                    events.append(ev)
                    if not closes:
                        open_chains[pitch] = ev
            else:
                if opens and pitch in open_chains:
                    return None
                ev = {'pitch': pitch, 'start': t, 'end': t + dur,
                      'open_start': False, 'open_end': False,
                      'stem': stem}
                events.append(ev)
                if opens:
                    open_chains[pitch] = ev
    for ev in open_chains.values():
        # Unclosed here — may be a hand-off to the sibling column, a
        # continuation past the span, or an orphan; resolved by the
        # caller after cross-column stitching.
        ev['open_chain'] = True
    if span_end is not None:
        for ev in events:
            if ev['end'] > span_end:
                return None
    return events


def _chain_continues(lines: List[str], last_dl: int, pitch: str,
                     positions: Optional[List[Dict[int, int]]] = None
                     ) -> bool:
    """True if the first data line after *last_dl* continues *pitch*.

    The supplied positions cover one hand: a pending pitch attachment may
    land in any of that hand's written columns.  A kept mark whose mate is
    truly absent stays acoustically inert and is cleared by the exit
    engine's global hygiene; dropping it here would fabricate a re-strike."""
    for idx in range(last_dl + 1, len(lines)):
        ln = lines[idx]
        if not ln or ln.startswith('!') or ln.startswith('*') \
                or ln.startswith('='):
            continue
        cells = ln.split('\t')
        if positions is not None:
            fields = {p[idx] for p in positions if idx in p}
            cells = [c for f, c in enumerate(cells) if f in fields]
        for cell in cells:
            if cell == '.':
                continue
            for sub in cell.split(' '):
                if ('_' in sub or ']' in sub):
                    m = _PITCH_RE.search(sub)
                    if m and m.group(0) == pitch:
                        return True
        return False
    return False


def _build_resynth_voice_cells(
    lines: List[str], data_lines: List[int],
    positions: List[Dict[int, int]],
    wt: Optional[Dict[int, Fraction]] = None,
    required_end: Optional[Fraction] = None,
    continuation_positions: Optional[List[Dict[int, int]]] = None,
) -> Optional[Dict[int, str]]:
    """Re-serialize one or more written layers as one voice of chord cells.

    Same-pitch same-onset entries are one physical strike (the longer ring
    wins).  A later strike remains independent while an earlier sound may
    stay pending; neither event changes the other's release.  Silence
    survives only where nothing sounds, and every ring is split into tie
    pieces at the union onset grid.

    Row onsets come from *wt* (window-relative, cut from the global
    clock) when given: a clock started inside a mid-bar window cannot
    see sibling columns' mid-ring state and mistimes every row.  The
    local clock is only trusted without wt, where the caller's seams
    are bar-quantized and every column starts the window fresh.
    """
    from src.score.sanitize_kern import parse_kern_duration
    if not data_lines:
        return None
    if wt is not None:
        times: Dict[int, Fraction] = wt
        span_end = None
    else:
        lt = _line_times(lines, data_lines)
        if lt is None:
            return None
        times, span_end = lt
    # Fill silence only to where these two layers reach — the global
    # clock's tail can ride other columns' rings.  Only sound counts: a
    # rest carries no voice identity, so letting one stretch the reach
    # would demand filler silence past the window's last row.
    pair_end = Fraction(0)
    for dl in data_lines:
        toks = lines[dl].split('\t')
        for layer_pos in positions:
            pos = layer_pos[dl]
            if pos < len(toks) and toks[pos] not in ('.', ''):
                for sub in toks[pos].split(' '):
                    d = parse_kern_duration(sub)
                    if d and d > 0:
                        pair_end = max(pair_end, times[dl] + d)
    if pair_end <= 0:
        return None
    if required_end is not None:
        if span_end is not None and required_end > span_end:
            return None
        span_end = required_end
    else:
        span_end = (pair_end if span_end is None
                    else min(span_end, pair_end))

    evs: list = []
    for pos in positions:
        got = _column_events(lines, data_lines, times, pos, span_end)
        if got is None:
            return None
        evs.extend(got)
    if not evs:
        return None

    by_pitch: Dict[str, list] = {}
    for ev in evs:
        by_pitch.setdefault(ev['pitch'], []).append(ev)
    final: list = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda e: (e['start'], -e['end'], not e['open_end']))
        # Stitch cross-column chain hand-offs: an unclosed chain whose
        # end meets an entering continuation is ONE sound.
        stitched: list = []
        for ev in group:
            prev = stitched[-1] if stitched else None
            if (prev is not None and prev.get('open_chain')
                    and ev['open_start'] and ev['start'] == prev['end']):
                prev['end'] = ev['end']
                prev['open_chain'] = ev.get('open_chain', False)
                continue
            stitched.append(ev)
        for ev in stitched:
            if ev.get('open_chain'):
                # continues past the span in the merged column's
                # downstream cells — or an orphan that just stops
                ev['open_end'] = _chain_continues(
                    lines,
                    data_lines[-1],
                    ev['pitch'],
                    continuation_positions or positions,
                )
        group = stitched
        # Stitching extends events, so the longest-first order must be
        # recomputed — the unison dedup below keeps the FIRST of a
        # same-onset pair, and a chain grown by stitching must not
        # lose to the shorter layer it duplicates.
        group.sort(key=lambda e: (e['start'], -e['end'], not e['open_end']))
        kept: list = []
        for ev in group:
            if kept and ev['start'] == kept[-1]['start']:
                continue                     # unison duplicate: one strike
            kept.append(ev)
        final.extend(kept)

    # cut at every onset, every event end, AND every source cell
    # boundary — cells must stay duration-uniform, and the source's
    # own segmentation (its beat spelling) must survive
    src_bounds = set()
    for dl in data_lines:
        toks = lines[dl].split('\t')
        for layer_pos in positions:
            pos = layer_pos[dl]
            if pos < len(toks) and toks[pos] != '.':
                src_bounds.add(times[dl])
                break
    grid = sorted({ev['start'] for ev in final}
                  | {ev['end'] for ev in final} | src_bounds)
    t2dl = {times[dl]: dl for dl in data_lines}
    cells: Dict[int, list] = {}

    def emit(ev_pitch, height, start, end, open_start, open_end,
             stem=''):
        cuts = [t for t in grid if start < t < end]
        bnds = [start] + cuts + [end]
        n = len(bnds) - 1
        for i in range(n):
            dur = bnds[i + 1] - bnds[i]
            recip = _dur_to_kern_recip(dur)
            if recip is None or bnds[i] not in t2dl:
                return False
            chain_start = (i == 0 and not open_start)
            chain_end = (i == n - 1 and not open_end)
            if chain_start and chain_end:
                tok = recip + ev_pitch
            elif chain_start:
                tok = '[' + recip + ev_pitch
            elif chain_end:
                tok = recip + ev_pitch + ']'
            else:
                tok = recip + ev_pitch + '_'
            # Stem ink survives the rewrite: it is the engraver's
            # voice-membership evidence and later windows still read
            # it (the exit strip removes it once surgery is over).
            cells.setdefault(t2dl[bnds[i]], []).append(
                (height, tok + stem))
        return True

    for ev in final:
        if not emit(ev['pitch'], _kern_pitch_height(ev['pitch']),
                    ev['start'], ev['end'],
                    ev['open_start'], ev['open_end'],
                    stem=ev.get('stem', '')):
            return None

    # silence survives only where nothing sounds
    ivs = sorted([ev['start'], ev['end']] for ev in final)
    merged_iv: list = []
    for s, e in ivs:
        if merged_iv and s <= merged_iv[-1][1]:
            merged_iv[-1][1] = max(merged_iv[-1][1], e)
        else:
            merged_iv.append([s, e])
    prev = Fraction(0)
    gaps: list = []
    for s, e in merged_iv:
        if s > prev:
            gaps.append((prev, s))
        prev = max(prev, e)
    if prev < span_end:
        gaps.append((prev, span_end))
    for g0, g1 in gaps:
        cuts = [t for t in grid if g0 < t < g1]
        bnds = [g0] + cuts + [g1]
        for i in range(len(bnds) - 1):
            dur = bnds[i + 1] - bnds[i]
            recip = _dur_to_kern_recip(dur)
            if recip is None or bnds[i] not in t2dl:
                return None
            cells.setdefault(t2dl[bnds[i]], []).append((-1, recip + 'r'))

    new_cells: Dict[int, str] = {}
    for dl in data_lines:
        toks = cells.get(dl)
        if toks:
            toks.sort(key=lambda x: x[0])
            new_cells[dl] = ' '.join(tok for _, tok in toks)
        else:
            new_cells[dl] = '.'
    return new_cells


_STEM_RE = re.compile(r'[/\\]')


def _strip_stem_marks(token: str) -> str:
    return _STEM_RE.sub('', token)


# =============================================================================
# Note-Spelling Normalization
# =============================================================================
# Respell every voice column to the serialization's own duration grammar:
# within a bar, a tie chain whose every link carries the full pitch set
# collapses to its total duration; a total (or any single cell) whose
# duration is a glyph prints as one notehead regardless of position, and
# anything else splits per the metric tree.  The exit text therefore depends
# on sounding spans rather than the source engraver's equivalent tie choices.

_PRODUCER_GRID = 128
_PRODUCER_RECIP_TO_GRID = {
    '128': 1, '64': 2, '64.': 3, '32': 4, '32.': 6,
    '16': 8, '16.': 12, '8': 16, '8.': 24,
    '4': 32, '4.': 48, '2': 64, '2.': 96,
    '1': 128, '1.': 192,
}
_PRODUCER_GRID_TO_RECIP = {
    duration: recip for recip, duration in _PRODUCER_RECIP_TO_GRID.items()
}
_PRODUCER_TREE_CACHE: Dict[Tuple[int, int], dict] = {}


def _producer_metric_tree(num: int, den: int) -> dict:
    """Build the producer's independent metric hierarchy."""
    if num not in {2, 3, 4, 5, 6, 9, 12} \
            or den not in {2, 4, 8, 16}:
        raise ValueError(f'unsupported meter {num}/{den}')
    key = (num, den)
    if key in _PRODUCER_TREE_CACHE:
        return _PRODUCER_TREE_CACHE[key]

    bar_length = num * (_PRODUCER_GRID // den)
    cells = {(0, bar_length)}
    lines: Dict[int, int] = {}
    walls = set()

    def add_line(position: int, rank: int, wall: bool = False) -> None:
        lines[position] = min(lines.get(position, rank), rank)
        if wall:
            walls.add(position)

    def add_cells(parts) -> None:
        cells.update(parts)

    def thirds(start: int, end: int, rank: int):
        unit = (end - start) // 3
        left, right = start + unit, start + 2 * unit
        add_line(left, rank)
        add_line(right, rank)
        parts = [(start, left), (left, right), (right, end)]
        add_cells(parts)
        return parts

    if num == 2:
        middle = bar_length // 2
        add_line(middle, 1, True)
        leaves = [(0, middle), (middle, bar_length)]
        add_cells(leaves)
        binary_rank = 2
    elif num == 3:
        leaves = thirds(0, bar_length, 1)
        binary_rank = 2
    elif num == 4:
        middle = bar_length // 2
        quarter = bar_length // 4
        add_line(middle, 1, True)
        add_line(quarter, 2)
        add_line(3 * quarter, 2)
        add_cells([(0, middle), (middle, bar_length)])
        leaves = [
            (0, quarter), (quarter, middle),
            (middle, middle + quarter), (middle + quarter, bar_length),
        ]
        add_cells(leaves)
        binary_rank = 3
    elif num == 5:
        beat = bar_length // 5
        group_line = 3 * beat
        add_line(group_line, 1, True)
        add_cells([(0, group_line), (group_line, bar_length)])
        leaves = thirds(0, group_line, 2)
        two_line = group_line + beat
        add_line(two_line, 2)
        pair = [(group_line, two_line), (two_line, bar_length)]
        add_cells(pair)
        leaves += pair
        binary_rank = 3
    elif num == 6:
        middle = bar_length // 2
        add_line(middle, 1, True)
        add_cells([(0, middle), (middle, bar_length)])
        leaves = thirds(0, middle, 2) + thirds(middle, bar_length, 2)
        binary_rank = 3
    elif num == 9:
        compound = bar_length // 3
        add_line(compound, 1, True)
        add_line(2 * compound, 1, True)
        add_cells([
            (0, compound), (compound, 2 * compound),
            (2 * compound, bar_length),
        ])
        leaves = []
        for start in (0, compound, 2 * compound):
            leaves += thirds(start, start + compound, 2)
        binary_rank = 3
    else:
        middle = bar_length // 2
        compound = bar_length // 4
        add_line(middle, 1, True)
        add_line(compound, 2, True)
        add_line(3 * compound, 2, True)
        add_cells([(0, middle), (middle, bar_length)])
        add_cells([
            (0, compound), (compound, middle),
            (middle, middle + compound),
            (middle + compound, bar_length),
        ])
        leaves = []
        for start in (0, compound, middle, middle + compound):
            leaves += thirds(start, start + compound, 3)
        binary_rank = 4

    def subdivide(start: int, end: int, rank: int) -> None:
        cells.add((start, end))
        if end - start <= 1 or (end - start) % 2:
            return
        middle = (start + end) // 2
        add_line(middle, rank)
        cells.add((start, middle))
        cells.add((middle, end))
        subdivide(start, middle, rank + 1)
        subdivide(middle, end, rank + 1)

    for start, end in leaves:
        subdivide(start, end, binary_rank)
    tree = {
        'num': num,
        'den': den,
        'bar_length': bar_length,
        'cells': cells,
        'lines': lines,
        'walls': walls,
    }
    _PRODUCER_TREE_CACHE[key] = tree
    return tree


def _producer_spell(onset: int, duration: int, tree: dict) -> List[str]:
    """Spell one producer duration without consulting the reader."""
    if duration <= 0:
        raise ValueError(f'duration must be positive, got {duration}')
    end = onset + duration
    glyph = _PRODUCER_GRID_TO_RECIP.get(duration)
    cells = tree['cells']
    lines = tree['lines']

    def rank(position: int) -> int:
        if position <= 0 or position >= tree['bar_length']:
            return 0
        return lines.get(position, 999)

    licensed = False
    if glyph is not None and (onset, end) in cells:
        licensed = True
    elif glyph is not None:
        blocked = any(
            onset < wall < end and rank(onset) >= lines[wall]
            for wall in tree['walls']
        )
        if not blocked and glyph.endswith('.'):
            end_rank = min(rank(onset), rank(end))
            licensed = all(
                line_rank >= end_rank
                for position, line_rank in lines.items()
                if onset < position < end
            )
        if not blocked and not licensed:
            licensed = any(
                position - onset == end - position
                and (onset, position) in cells
                and (position, end) in cells
                for position in lines
                if onset < position < end
            )
    if licensed:
        return [glyph]

    interior = [
        (line_rank, -position, position)
        for position, line_rank in lines.items()
        if onset < position < end
    ]
    if not interior:
        raise ValueError(
            f'no metric cut in [{onset}, {end}) for '
            f'{tree["num"]}/{tree["den"]}'
        )
    cut = min(interior)[2]
    return (
        _producer_spell(onset, cut - onset, tree)
        + _producer_spell(cut, end - cut, tree)
    )


_NNS_RECIP_RE = re.compile(r'(\d+\.*)')


def _nns_parse_cell(tok: str):
    """Parse one data cell for respelling: (dur, recip, subs) or None.

    None marks a cell this pass must leave alone: grace content, hidden
    print-layer rests, tuplet-ratio durations, or any residue the
    rebuild could not reproduce."""
    subs = []
    recip0 = None
    for sub in tok.split():
        if 'q' in sub or 'yy' in sub or '%' in sub:
            return None
        rest = 'r' in sub  # pitch letters exclude r
        stems = ''.join(c for c in sub if c in '/\\')
        core = sub
        for c in '[]_/\\':
            core = core.replace(c, '')
        m = _NNS_RECIP_RE.match(core)
        if m:
            recip, core = m.group(1), core[m.end():]
            if recip0 is None:
                recip0 = recip
            elif recip != recip0:
                return None
        elif recip0 is None:
            return None
        if rest:
            if core != 'r':
                return None
            pitch = None
        else:
            if not re.fullmatch(_PITCH_RE, core):
                return None
            pitch = core
        subs.append({
            'pitch': pitch, 'rest': rest, 'stems': stems,
            'open': '[' in sub, 'close': ']' in sub,
            'cont': '_' in sub or ('[' in sub and ']' in sub),
        })
    if not subs:
        return None
    dur = _token_rhythm(tok)
    if dur is None:
        return None
    return dur, recip0, subs


def _nns_role(subs) -> Optional[str]:
    """Chain-link role of a cell; None = leave as written (partial or
    mixed per-member ties cannot merge as one chain)."""
    if any(s['rest'] for s in subs):
        return 'rest' if all(s['rest'] for s in subs) else None
    if all(not s['open'] and not s['close'] and not s['cont']
           for s in subs):
        return 'plain'
    if all(s['cont'] for s in subs):
        return 'cont'
    if all(s['open'] and not s['cont'] for s in subs):
        return 'start'
    if all(s['close'] and not s['cont'] for s in subs):
        return 'end'
    return None


def _elide_hanging_tie_portals(kern_content: str) -> str:
    """Keep one tied head per bar and the final marked exit."""
    lines = kern_content.split("\n")
    notes, _orphan_opens, _orphan_closes = _scan_ties(lines)
    bar_at: Dict[int, int] = {}
    bar_index = -1
    for row, line in enumerate(lines):
        if line.startswith("="):
            bar_index += 1
        bar_at[row] = bar_index

    grouped: Dict[Tuple[int, int, str], List[list]] = defaultdict(list)
    for note in notes:
        row, _field, _sub, hand, pitch, has_open, has_close = note
        if has_open or has_close:
            grouped[(bar_at[row], hand, pitch)].append(note)

    remove: Set[Tuple[int, int, int]] = set()
    for members in grouped.values():
        chains: List[Tuple[List[list], bool]] = []
        chain: List[list] = []
        active = False
        ambiguous = False
        for note in members:
            has_open, has_close = note[5], note[6]
            if has_open and not has_close:
                if active:
                    ambiguous = True
                    break
                chain = [note]
                active = True
                continue
            if has_open and has_close:
                if not active:
                    chain = [note]
                    active = True
                else:
                    chain.append(note)
                continue
            if has_close:
                if active:
                    chain.append(note)
                    chains.append((chain, True))
                    chain = []
                    active = False
                else:
                    chains.append(([note], True))
        if ambiguous:
            continue
        if chain:
            chains.append((chain, False))
        for tied_chain, has_exit in chains:
            keep = {0}
            if has_exit:
                keep.add(len(tied_chain) - 1)
            # A tuplet member's slot reading needs the pitch present in
            # every slot; elision is for chains that live entirely on the
            # dyadic lattice.
            dyadic_chain = True
            for note in tied_chain:
                cell = lines[note[0]].split("\t")[note[1]]
                token = cell.split(" ")[note[2]]
                recip = re.search(r"\d+", token)
                base = int(recip.group(0)) if recip else 0
                if base and (base & (base - 1)):
                    dyadic_chain = False
                    break
            if not dyadic_chain:
                continue
            for index, note in enumerate(tied_chain):
                if index not in keep and note[5] and note[6]:
                    remove.add((note[0], note[1], note[2]))

    if not remove:
        return kern_content
    for row in sorted({location[0] for location in remove}):
        fields = lines[row].split("\t")
        by_field: Dict[int, Set[int]] = defaultdict(set)
        for remove_row, field, sub in remove:
            if remove_row == row:
                by_field[field].add(sub)
        for field, indexes in by_field.items():
            subtokens = fields[field].split(" ")
            kept = [
                token for index, token in enumerate(subtokens)
                if index not in indexes
            ]
            if kept:
                fields[field] = " ".join(kept)
                continue
            reciprocal = re.search(
                r"\d+(?:%\d+)?\.*", subtokens[0])
            if reciprocal is None:
                return kern_content
            fields[field] = f"{reciprocal.group(0)}r"
        lines[row] = "\t".join(fields)
    return "\n".join(lines)


def normalize_note_spelling(kern_content: str) -> str:
    """Respell columns to canonical duration spelling, bar by bar.

    Sound-invariant by construction: chain merging and re-splitting
    preserve every keystroke and its full ringing span.  Any bar/column
    the pass cannot prove safe is left exactly as written."""
    from src.score.merge_voice import (
        GRID, GRID_TO_RECIP, RECIP_TO_GRID, get_metric_tree, spell)

    lines = kern_content.split('\n')
    lv = _column_lives(lines)
    if lv is None:
        return kern_content
    lives, _runs = lv
    in_bar = _in_bar_onsets(lines)
    if in_bar is None:
        return kern_content
    q = GRID // 4  # grid units per quarter note

    def is_data(r: int) -> bool:
        ln = lines[r]
        return bool(ln) and not ln.startswith(('!', '*', '='))

    meters: Dict[int, Optional[Tuple[int, int]]] = {}
    cur_m = None
    for i, ln in enumerate(lines):
        got = _line_meter(ln)
        if got:
            cur_m = got
        meters[i] = cur_m

    boundaries = [-1] + [i for i, ln in enumerate(lines)
                         if ln.startswith('=')] + [len(lines)]
    out: List[str] = []

    for s, e in zip(boundaries, boundaries[1:]):
        if s >= 0:
            out.append(lines[s])
        span = list(range(s + 1, e))
        dls = [r for r in span if is_data(r)]
        if not dls:
            out.extend(lines[r] for r in span)
            continue
        struct_rows = [r for r in span
                       if lines[r].startswith('*')
                       and any(t in ('*^', '*v')
                               for t in lines[r].split('\t'))]
        meter = meters.get(dls[0])
        tree = None
        if meter is not None:
            try:
                tree = get_metric_tree(*meter)
            except ValueError:
                tree = None
        if tree is None:
            out.extend(lines[r] for r in span)
            continue

        # integer-grid onset per data row (tuplet-interior rows excluded)
        row_onset: Dict[int, int] = {}
        onset_row: Dict[int, int] = {}
        ok_clock = True
        for r in dls:
            t = in_bar.get(r)
            if t is None:
                ok_clock = False
                break
            tg = t * q
            if tg.denominator == 1:
                row_onset[r] = int(tg)
                onset_row[int(tg)] = r
        if not ok_clock:
            out.extend(lines[r] for r in span)
            continue

        clears: set = set()                 # (row, field)
        writes: Dict[Tuple[int, int], str] = {}
        inserts: Dict[int, Dict[int, str]] = {}  # onset -> {cid: text}

        for cid, life in lives.items():
            cells = []
            for r in dls:
                f = life['pos'].get(r)
                if f is None:
                    continue
                tok = lines[r].split('\t')[f]
                if tok == '.':
                    continue
                if r not in row_onset:
                    cells.append({
                        'row': r, 'field': f, 'unsupported': True,
                    })
                    continue
                parsed = _nns_parse_cell(tok)
                if parsed is None:
                    cells.append({
                        'row': r, 'field': f, 'unsupported': True,
                    })
                    continue
                dur, recip, subs = parsed
                dg = dur * q
                if dg.denominator != 1:
                    cells.append({
                        'row': r, 'field': f, 'unsupported': True,
                    })
                    continue
                o = row_onset[r]
                if o + int(dg) > tree.bar_length:
                    cells.append({
                        'row': r, 'field': f, 'unsupported': True,
                    })
                    continue
                cells.append({'row': r, 'field': f, 'onset': o,
                              'dur': int(dg), 'recip': recip, 'subs': subs,
                              'role': _nns_role(subs)})
            if not cells:
                continue

            # ── chain accumulation ──────────────────────────────────
            events = []       # merged/respellable events
            active = None     # one chain at a time in a sequential column

            def close_active(tie_out: bool) -> None:
                nonlocal active
                if active is not None:
                    active['tie_out'] = tie_out
                    events.append(active)
                    active = None

            def break_active() -> None:
                # A chain interrupted before its close is a source
                # anomaly: keep every link exactly as written.
                nonlocal active
                if active is not None:
                    active['broken'] = True
                    close_active(True)

            for c in cells:
                if c.get('unsupported'):
                    break_active()
                    events.append({'cells': [c], 'broken': True})
                    continue
                role = c['role']
                pitches = tuple(sorted(
                    (s['pitch'] for s in c['subs'] if s['pitch']),
                    key=_kern_pitch_height))
                stems = ''.join(sorted({ch for s in c['subs']
                                        for ch in s['stems']}))
                if role == 'rest':
                    break_active()
                    events.append({'cells': [c], 'onset': c['onset'],
                                   'dur': c['dur'], 'rest': True,
                                   'pitches': (), 'stems': '',
                                   'tie_in': False, 'tie_out': False,
                                   'broken': False})
                elif role == 'plain':
                    break_active()
                    events.append({'cells': [c], 'onset': c['onset'],
                                   'dur': c['dur'], 'rest': False,
                                   'pitches': pitches, 'stems': stems,
                                   'tie_in': False, 'tie_out': False,
                                   'broken': False})
                elif role == 'start':
                    break_active()
                    active = {'cells': [c], 'onset': c['onset'],
                              'dur': c['dur'], 'rest': False,
                              'pitches': pitches, 'stems': stems,
                              'tie_in': False, 'tie_out': True,
                              'broken': False}
                elif role in ('cont', 'end'):
                    if (active is not None
                            and active['pitches'] == pitches
                            and active['onset'] + active['dur']
                            == c['onset']):
                        active['cells'].append(c)
                        active['dur'] += c['dur']
                        active['stems'] = ''.join(sorted(
                            set(active['stems']) | set(stems)))
                        if role == 'end':
                            close_active(False)
                    elif active is None:
                        # chain entering from the previous bar
                        active = {'cells': [c], 'onset': c['onset'],
                                  'dur': c['dur'], 'rest': False,
                                  'pitches': pitches, 'stems': stems,
                                  'tie_in': True, 'tie_out': True,
                                  'broken': False}
                        if role == 'end':
                            close_active(False)
                    else:
                        # chain the parse cannot follow: keep both sides
                        break_active()
                        events.append({'cells': [c], 'broken': True})
                elif all(not s['rest'] for s in c['subs']):
                    break_active()
                    members = sorted(
                        (
                            s['pitch'],
                            s['close'] or s['cont'],
                            s['open'] or s['cont'],
                        )
                        for s in c['subs']
                        if s['pitch']
                    )
                    events.append({
                        'cells': [c], 'onset': c['onset'],
                        'dur': c['dur'], 'rest': False,
                        'pitches': tuple(member[0] for member in members),
                        'stems': stems,
                        'tie_ins': tuple(member[1] for member in members),
                        'tie_outs': tuple(member[2] for member in members),
                        'tie_in': False, 'tie_out': False,
                        'broken': False,
                    })
                else:
                    break_active()
                    events.append({'cells': [c], 'broken': True})
            close_active(True)   # still open at bar end: exits the bar

            # A rest has no attack boundary.  Canonicalize one contiguous
            # silence as a unit unless a source topology row separates it.
            merged_events = []
            for ev in events:
                previous = merged_events[-1] if merged_events else None
                separated = False
                if previous is not None \
                        and previous.get('cells') and ev.get('cells'):
                    left_row = previous['cells'][-1]['row']
                    right_row = ev['cells'][0]['row']
                    separated = any(
                        left_row < row < right_row for row in struct_rows)
                if previous is not None \
                        and previous.get('rest') and ev.get('rest') \
                        and not previous.get('broken') \
                        and not ev.get('broken') \
                        and previous['onset'] + previous['dur'] == ev['onset'] \
                        and not separated:
                    previous['cells'].extend(ev['cells'])
                    previous['dur'] += ev['dur']
                else:
                    merged_events.append(ev)
            events = merged_events

            # A *^/*v line re-books the column mid-bar; an event whose
            # ringing crosses it must keep the engraver's own split.
            if struct_rows:
                for ev in events:
                    if ev.get('broken') or not ev.get('cells'):
                        continue
                    r0 = ev['cells'][0]['row']
                    end_t = ev['onset'] + ev['dur']
                    for sr in struct_rows:
                        if sr <= r0:
                            continue
                        nr = next((r for r in dls if r > sr), None)
                        if nr is None:
                            continue
                        nt = row_onset.get(nr)
                        if nt is None or nt < end_t:
                            ev['broken'] = True
                            break

            # ── piece plans ─────────────────────────────────────────
            col_clear: List[Tuple[int, int]] = []
            col_write: Dict[int, Tuple[int, str]] = {}   # onset -> text
            feasible = True
            for ev in events:
                if ev.get('broken'):
                    continue
                single_glyph = (
                    len(ev['cells']) == 1
                    and ev['dur'] in GRID_TO_RECIP
                    and ev['cells'][0]['recip'] == GRID_TO_RECIP[ev['dur']]
                )
                if single_glyph:
                    continue      # already canonical: untouched
                if ev['dur'] in GRID_TO_RECIP:
                    glyphs = [GRID_TO_RECIP[ev['dur']]]
                else:
                    try:
                        glyphs = spell(ev['onset'], ev['dur'], tree)
                    except ValueError:
                        feasible = False
                        break
                pieces = []
                po = ev['onset']
                for g in glyphs:
                    pieces.append((po, RECIP_TO_GRID[g], g))
                    po += RECIP_TO_GRID[g]
                n = len(pieces)
                for k, (po2, pg, pr) in enumerate(pieces):
                    if ev['rest']:
                        text = f"{pr}r"
                    else:
                        parts = []
                        per_pitch_ins = ev.get('tie_ins')
                        per_pitch_outs = ev.get('tie_outs')
                        for pitch_index, p in enumerate(ev['pitches']):
                            # converter21 hears the bracket-pair form
                            # [X] as a fresh attack; underscore is the
                            # only continue spelling it reads silently.
                            pitch_ti = (
                                (per_pitch_ins[pitch_index]
                                 if per_pitch_ins is not None else ev['tie_in'])
                                or k > 0
                            )
                            pitch_to = (
                                (per_pitch_outs[pitch_index]
                                 if per_pitch_outs is not None else ev['tie_out'])
                                or k < n - 1
                            )
                            if pitch_ti and pitch_to:
                                parts.append(f"{pr}{p}_")
                            elif pitch_to:
                                parts.append(f"[{pr}{p}")
                            elif pitch_ti:
                                parts.append(f"{pr}{p}]")
                            else:
                                parts.append(f"{pr}{p}")
                        if ev['stems']:
                            parts[0] += ev['stems']
                        text = ' '.join(parts)
                    col_write[po2] = (pg, text)
                for c in ev['cells']:
                    col_clear.append((c['row'], c['field']))
            if not feasible or not col_write:
                continue

            # ── placement feasibility (before committing anything) ──
            placeable = True
            cleared_here = set(col_clear)
            for po2, (_pg, _text) in col_write.items():
                r = onset_row.get(po2)
                if r is not None:
                    f = life['pos'].get(r)
                    if f is None:
                        placeable = False
                        break
                    tok = lines[r].split('\t')[f]
                    if tok != '.' and (r, f) not in cleared_here:
                        placeable = False
                        break
                else:
                    r_b = next((rr for rr in dls
                                if in_bar[rr] * q > po2), None)
                    anchor = r_b if r_b is not None else dls[-1]
                    if life['pos'].get(anchor) is None:
                        placeable = False
                        break
            if not placeable:
                continue

            clears.update((r, f) for r, f in col_clear)
            for po2, (_pg, text) in col_write.items():
                r = onset_row.get(po2)
                if r is not None:
                    writes[(r, life['pos'][r])] = text
                else:
                    inserts.setdefault(po2, {})[cid] = text

        if not clears and not writes and not inserts:
            out.extend(lines[r] for r in span)
            continue

        # ── rebuild the bar ─────────────────────────────────────────
        pending = sorted(inserts.items())
        comment_buf: List[str] = []

        def flush_inserts(before_onset: Optional[Fraction],
                          anchor: int) -> None:
            nonlocal pending
            while pending and (before_onset is None
                               or pending[0][0] < before_onset):
                po2, colmap = pending.pop(0)
                width = len(lines[anchor].split('\t'))
                row = ['.'] * width
                ok = True
                for cid2, text in colmap.items():
                    f2 = lives[cid2]['pos'].get(anchor)
                    if f2 is None or f2 >= width:
                        ok = False
                        break
                    row[f2] = text
                if ok:
                    out.append('\t'.join(row))

        for r in span:
            ln = lines[r]
            if not is_data(r):
                if ln.startswith('!'):
                    comment_buf.append(ln)
                else:
                    out.extend(comment_buf)
                    comment_buf = []
                    out.append(ln)
                continue
            flush_inserts(in_bar[r] * q, r)
            out.extend(comment_buf)
            comment_buf = []
            toks = ln.split('\t')
            for fi in range(len(toks)):
                if (r, fi) in clears:
                    toks[fi] = '.'
            for (wr, wf), text in writes.items():
                if wr == r:
                    toks[wf] = text
            if all(t == '.' for t in toks):
                continue
            out.append('\t'.join(toks))
        flush_inserts(None, dls[-1])
        out.extend(comment_buf)

    new_content = _elide_hanging_tie_portals('\n'.join(out))
    # Structural safety net: refuse the whole rewrite rather than emit a
    # file the downstream passes cannot clock.
    new_lines = new_content.split('\n')
    if _column_lives(new_lines) is None or _global_times(new_lines) is None:
        logger.warning("note-spelling normalization refused: "
                       "rewritten file fails structural checks")
        return kern_content
    return new_content


def normalize_rest_spelling(kern_content: str) -> str:
    """Canonicalize each contiguous silence independently of note parsing."""
    lines = kern_content.split('\n')
    tracked = _column_lives(lines)
    if tracked is None:
        return kern_content
    lives, _ = tracked
    q = _PRODUCER_GRID // 4
    meter = None

    barlines = [row for row, line in enumerate(lines) if line.startswith('=')]
    segments = list(zip(barlines, barlines[1:]))
    first_data = next(
        (row for row, line in enumerate(lines)
         if line and not line.startswith(('!', '*', '='))),
        None,
    )
    if first_data is not None and (not barlines or first_data < barlines[0]):
        segments.insert(0, (-1, barlines[0] if barlines else len(lines)))

    for b0, b1 in segments:
        dls = [row for row in range(b0 + 1, b1)
               if lines[row] and not lines[row].startswith(('!', '*', '='))]
        if not dls:
            continue
        timed = _line_times(lines, dls)
        if timed is None:
            continue
        in_bar, _ = timed
        for row in range(dls[0] + 1):
            got = _line_meter(lines[row])
            if got:
                meter = got
        if meter is None:
            continue
        try:
            tree = _producer_metric_tree(*meter)
        except ValueError:
            continue
        onset_row = {
            in_bar[row] * q: row for row in dls
        }
        clears = set()
        writes: Dict[Tuple[int, int], str] = {}

        for life in lives.values():
            rest_cells = []
            for row in dls:
                field = life['pos'].get(row)
                if field is None:
                    continue
                cells = lines[row].split('\t')
                if field >= len(cells) or not _is_rest_cell(cells[field]):
                    continue
                onset_grid = in_bar[row] * q
                duration = _cell_step_rhythm(cells[field])
                if duration is None:
                    continue
                rest_cells.append({
                    'row': row,
                    'field': field,
                    'onset': onset_grid,
                    'dur': duration * q,
                })
            rest_cells.sort(key=lambda cell: cell['onset'])

            index = 0
            while index < len(rest_cells):
                run = [rest_cells[index]]
                while index + len(run) < len(rest_cells):
                    nxt = rest_cells[index + len(run)]
                    prev = run[-1]
                    if prev['onset'] + prev['dur'] != nxt['onset']:
                        break
                    run.append(nxt)
                cursor = 0
                while cursor < len(run):
                    selected = None
                    for stop in range(len(run), cursor, -1):
                        onset = run[cursor]['onset']
                        total = sum(
                            (cell['dur'] for cell in run[cursor:stop]),
                            Fraction(0),
                        )
                        if onset.denominator != 1 or total.denominator != 1:
                            continue
                        onset_int = int(onset)
                        total_int = int(total)
                        if total_int in _PRODUCER_GRID_TO_RECIP:
                            glyphs = [_PRODUCER_GRID_TO_RECIP[total_int]]
                        else:
                            try:
                                glyphs = _producer_spell(
                                    onset_int, total_int, tree)
                            except ValueError:
                                continue
                        placements = []
                        placement_onset = onset
                        for glyph in glyphs:
                            row = onset_row.get(placement_onset)
                            field = (life['pos'].get(row)
                                     if row is not None else None)
                            if row is None or field is None:
                                placements = []
                                break
                            placements.append((row, field, f'{glyph}r'))
                            placement_onset += _PRODUCER_RECIP_TO_GRID[glyph]
                        if placements:
                            selected = (stop, glyphs, placements)
                            break
                    if selected is None:
                        cursor += 1
                        continue
                    stop, glyphs, placements = selected
                    actual = [cell['dur'] for cell in run[cursor:stop]]
                    expected = [
                        _PRODUCER_RECIP_TO_GRID[glyph] for glyph in glyphs
                    ]
                    if actual != expected:
                        clears.update(
                            (cell['row'], cell['field'])
                            for cell in run[cursor:stop]
                        )
                        for row, field, token in placements:
                            writes[(row, field)] = token
                    cursor = stop
                index += len(run)

        for row, field in clears:
            fields = lines[row].split('\t')
            fields[field] = '.'
            lines[row] = '\t'.join(fields)
        for (row, field), token in writes.items():
            fields = lines[row].split('\t')
            fields[field] = token
            lines[row] = '\t'.join(fields)
    return '\n'.join(
        line for line in lines
        if not (line and not line.startswith(('!', '*', '='))
                and all(cell == '.' for cell in line.split('\t')))
    )


def normalize_chord_order(kern_content: str) -> str:
    """Make kern member order agree with the unique low-to-high serialization."""
    lines = kern_content.split('\n')
    for row, line in enumerate(lines):
        if not line or line.startswith(('!', '*', '=')):
            continue
        fields = line.split('\t')
        for field, cell in enumerate(fields):
            parts = cell.split()
            if len(parts) < 2:
                continue
            expanded = []
            lead = None
            for part in parts:
                # Prefix class includes stem ink: a mark before the
                # digits must not turn a timed member into a "bare
                # pitch" that inherits a second duration.
                recip = re.match(
                    r'^[\[({/\\]*(\d+(?:%\d+)?\.*)', part)
                if recip is not None:
                    lead = recip.group(1)
                elif lead is not None and _PITCH_RE.search(part):
                    part = re.sub(
                        r'^([\[({/\\]*)', rf'\g<1>{lead}', part, count=1)
                expanded.append(part)
            parts = expanded
            pitches = [_SERIAL_PITCH_RE.search(part) for part in parts]
            if any(pitch is None for pitch in pitches):
                continue
            def tie_role(part: str) -> int:
                if "_" in part:
                    return 1
                if "]" in part:
                    return 2
                return 0
            fields[field] = ' '.join(part for _key, part in sorted(
                ((
                    _kern_pitch_height(pitch.group(0)),
                    tie_role(part),
                    pitch.group(0).replace("n", ""),
                ), part)
                for pitch, part in zip(pitches, parts)
            ))
        lines[row] = '\t'.join(fields)
    return '\n'.join(lines)


def standardize_kern(
    raw_kern: str,
    *,
    strip_cue: bool = True,
    keep_grace: bool = False,
    keep_trill: bool = False,
    keep_non_trill_ornaments: bool = False,
    keep_arpeggio: bool = False,
    canonical_trace: Optional[dict] = None,
) -> str:
    """Full standardization pipeline: raw kern → ground-truth kern.

    Shared entry point for prepare_syn (HumSyn/MuseSyn) and
    prepare_asap (ASAP MusicXML).  Callers produce a raw kern string;
    this function does everything after that.

    Input is expected to be Phase 1 output (visual markers already
    stripped by clean_kern_sequence). This function handles content-level
    decisions: cue stripping, articulation stripping, pitch-vocab
    compatibility (naturals, double accidentals, slur/phrase markers),
    and canonical duration spelling (normalize_note_spelling), so the
    stored ground truth is already in the serialization's own grammar.
    Tuplet and rscale display controls are removed without changing their
    literal duration tokens or grouping.

    strip_cue is a blanket True/False here; callers that need per-file
    accuracy (cadenza cue must be kept, orchestral/chamber reference cue
    must be stripped) should resolve the per-file treatment from
    cue_overrides.csv and pass strip_cue=(treatment == "strip"). The
    actual dispatch happens in sanitize_kern.sanitize_cue, shared with
    sanitize_kern_for_audio so both pipelines agree.

    keep_grace / keep_trill / keep_non_trill_ornaments / keep_arpeggio
    retain those signifiers in the ground truth (default: stripped).
    Articulation is always stripped.
    """
    # Memo caches are per-file working sets; clearing at entry bounds
    # memory over a corpus run (results are pure, so this cannot change
    # output).
    _cells.cache_clear()
    _is_rest_cell.cache_clear()
    _cell_step_rhythm.cache_clear()
    _global_times_cached.cache_clear()

    from src.score.sanitize_kern import sanitize_cue
    from src.score.clean_kern import (
        strip_articulation, strip_trill_marks,
        strip_non_trill_ornament_marks, strip_arpeggio_marks)

    content = raw_kern
    # Internal double barlines (==) left by Phase 1 expansion become
    # single barlines; only the final == at EOF stays double.
    lines_tmp = content.split('\n')
    for li in range(len(lines_tmp)):
        ln = lines_tmp[li]
        if ln.startswith('==') and not ln.startswith('==-'):
            parts = ln.split('\t')
            if any(lines_tmp[j].strip() and not lines_tmp[j].startswith('!')
                   for j in range(li + 1, min(li + 5, len(lines_tmp)))
                   if not lines_tmp[j].startswith('*-')):
                lines_tmp[li] = '\t'.join(
                    p.replace('==', '=', 1) for p in parts)
    content = '\n'.join(lines_tmp)
    content = _map_kern_tokens(content, normalize_odd_spellings)
    content = normalize_key_signatures(content)
    content = relocate_midbar_key_signatures(content)
    content = strip_redundant_schema_declarations(content)
    content = normalize_spine_op_placement(content)
    if not keep_grace:
        content = strip_grace_notes(content)
    content = _map_kern_tokens(content, strip_rest_tie_marks)
    # Source-orphan ties are paired on the real-note layout, before any
    # later pass can insert same-pitch pieces that would change which
    # candidate is nearest (repair-only: nothing is stripped here).
    content, _ = repair_ties(content, strip_unmatched=False)
    content = sanitize_cue(content, "strip" if strip_cue else "keep")
    content = strip_ancillary_ink(content)
    content = canonicalize_piano_header(content)
    content = relocate_empty_pickup_schema(content)
    content = strip_articulation(content)
    if not keep_trill:
        content = strip_trill_marks(content)
    if not keep_non_trill_ornaments:
        content = strip_non_trill_ornament_marks(content)
    if not keep_arpeggio:
        content = strip_arpeggio_marks(content)
    content = _map_kern_tokens(content, strip_natural_accidentals)
    content = _map_kern_tokens(content, convert_double_accidentals)
    content = _map_kern_tokens(content, remove_slur_phrase_markers)
    # Tie balancing runs last: cue stripping deletes notes and accidental
    # normalization respells pitches, so any earlier pairing could be
    # orphaned again.  The invariant is per hand and pitch.
    content, _ = repair_ties(content)
    content = normalize_chord_durations(content)
    content = complete_boundary_bars(content)
    content = absorb_empty_bars(content)
    from src.score.merge_voice import kern_sound_snapshot, merge_voice

    try:
        source_sound_snapshot = kern_sound_snapshot(content)
    except ValueError:
        source_sound_snapshot = None

    voice_issues = []
    voice_outcomes = []
    # Voice and playing-hand decisions share one immutable parsed source;
    # writer-filled rests never become decision evidence.
    content = merge_voice(
        content, trace=voice_outcomes, issues=voice_issues,
        restore_hand=True)
    content = normalize_note_spelling(content)
    content, _ = repair_ties(content)
    content = normalize_chord_order(content)
    content = normalize_rest_spelling(content)
    prewriter = content
    if canonical_trace is not None:
        canonical_trace['prewriter'] = prewriter
    if voice_issues:
        bar_issues = [
            {
                "family": issue.family,
                "message": issue.message,
                "bar_index": issue.bar_index,
                "bar_label": issue.bar_label,
                "bar_ordinal": issue.bar_ordinal,
            }
            for issue in voice_issues
        ]
        if canonical_trace is not None:
            canonical_trace['bar_issues'] = bar_issues
        # Keep the best text the finalizer can produce, while the structured
        # issue remains authoritative even if surface normalization removes
        # the syntax which first exposed the held bar.
        try:
            content = finalize_voice_notation(content)
        except (
            KernTokenizerOOV,
            KernIncompleteTuplet,
            KernMetricTimelineError,
            KernWriterContractError,
        ) as error:
            # Canonical readback is secondary once source-backed voice
            # analysis has already located the bar that must be isolated.
            # Preserve the finalizer's best text without replacing that
            # primary issue family.
            content = error.kern_content
        issue = voice_issues[0]
        if issue.family == "oov":
            raise KernTokenizerOOV(
                issue.message, content, bar_issues=bar_issues)
        if issue.family == "incomplete_tuplet":
            raise KernIncompleteTuplet(
                issue.message, content, bar_issues=bar_issues)
        if issue.family == "metric_timeline":
            raise KernMetricTimelineError(
                issue.message, content, bar_issues=bar_issues)
        raise KernWriterContractError(
            f"standardizer/voice-canonicalization: {issue.message}",
            content,
            bar_issues=bar_issues,
        )
    content = finalize_voice_notation(content)
    if source_sound_snapshot is None:
        raise KernWriterContractError(
            "standardizer/final-sound: source sound account is unreadable",
            content,
        )
    source_sound, source_bars = source_sound_snapshot
    try:
        final_sound, _final_bars = kern_sound_snapshot(content)
    except ValueError as error:
        raise KernWriterContractError(
            f"standardizer/final-sound: canonical sound account is unreadable: {error}",
            content,
        ) from error
    if final_sound != source_sound:
        source_count = Counter(source_sound)
        final_count = Counter(final_sound)
        differences = list((source_count - final_count).elements()) + list(
            (final_count - source_count).elements())
        mismatch_time = min(
            (item[1] for item in differences), default=Fraction(0))
        located = next((
            (index, label)
            for index, label, start, end in source_bars
            if start <= mismatch_time < end
        ), None)
        if located is None and source_bars:
            located = source_bars[-1][:2]
        message = (
            "final notation changed the sounding pitch/attack/release union"
        )
        bar_issues = [] if located is None else [{
            "family": "sound-invariant",
            "message": message,
            "bar_index": located[0],
            "bar_label": located[1] or None,
            "bar_ordinal": located[0],
        }]
        raise KernWriterContractError(
            f"standardizer/final-sound: {message}",
            content,
            bar_issues=bar_issues,
        )
    if canonical_trace is not None:
        canonical_trace['canonical'] = content
    return content


def finalize_voice_notation(kern_content: str) -> str:
    """Write the tokenized VoiceState back in deterministic canonical form."""
    from src.a2s.piano.tokenizer import (
        IncompleteTupletError,
        MetricTimelineError,
        TokenizerOOVError,
        tokenize_kern,
    )
    from src.score.reconstruct_kern import (
        reconstruct_kern_from_bracket_tokens,
    )

    try:
        tokens = tokenize_kern(kern_content)
    except TokenizerOOVError as error:
        raise KernTokenizerOOV(
            str(error),
            kern_content,
            bar_issues=_tokenizer_bar_issue(error, "oov"),
        ) from error
    except IncompleteTupletError as error:
        raise KernIncompleteTuplet(
            str(error),
            kern_content,
            bar_issues=_tokenizer_bar_issue(error, "incomplete_tuplet"),
        ) from error
    except MetricTimelineError as error:
        raise KernMetricTimelineError(
            str(error),
            kern_content,
            bar_issues=_tokenizer_bar_issue(error, "metric_timeline"),
        ) from error
    except ValueError as error:
        message = str(error)
        if "bar data before any meter declaration" in message:
            raise KernTokenizerOOV(
                f"schema has no meter for bar data: {message}",
                kern_content,
                bar_issues=_tokenizer_bar_issue(error, "oov"),
            ) from error
        if "voice: desync" in message:
            family = "voice-topology"
        elif "data row" in message and "active spines" in message:
            family = "spine-topology"
        elif "split/merge should be on the border of bars" in message:
            family = "voice-operation-border"
        elif "adjacent rests are not canonical" in message:
            family = "rest-spelling"
        elif "duration spelling is not canonical" in message:
            family = "duration-spelling"
        else:
            family = "prewriter-validation"
        bar_issues = _tokenizer_bar_issue(error, family)
        if not bar_issues:
            token_bar = re.search(r"token scope: bar (\d+):", message)
            if token_bar is not None:
                bar_issues = _contract_bar_issue(
                    kern_content,
                    int(token_bar.group(1)) - 1,
                    family,
                    message,
                )
        raise KernWriterContractError(
            f"standardizer/{family}: {message}",
            kern_content,
            bar_issues=bar_issues,
        ) from error

    try:
        canonical = reconstruct_kern_from_bracket_tokens(tokens)
        repeated = reconstruct_kern_from_bracket_tokens(tokens)
        if repeated != canonical:
            raise ValueError(
                "canonical writer is not deterministic for one token stream")
        readback_tokens = tokenize_kern(canonical)
    except Exception as error:
        message = str(error)
        writer_bar = re.match(r"bar (\d+):", message)
        token_bar = re.search(r"token scope: bar (\d+):", message)
        if writer_bar is not None:
            bar_issues = _contract_bar_issue(
                kern_content,
                int(writer_bar.group(1)),
                "canonical_writer",
                message,
            )
        elif token_bar is not None:
            bar_issues = _contract_bar_issue(
                kern_content,
                int(token_bar.group(1)) - 1,
                "canonical_readback",
                message,
            )
        else:
            bar_issues = []
        raise KernWriterContractError(
            message, kern_content, bar_issues=bar_issues,
        ) from error
    if readback_tokens != tokens:
        raise KernWriterContractError(
            "canonical writer changed the token stream",
            kern_content,
            canonical,
        )
    logger.info("voice-canonical-writer status=pass")
    return canonical


def normalize_voice_notation(kern_content: str) -> str:
    """Compatibility name for the canonical voice writer."""
    return finalize_voice_notation(kern_content)


def _phase1_bar_ordinals(issues: List[dict], kern_content: str) -> List[dict]:
    """Give phase 1's bars the ordinal the ground truth counts them by.

    The ground truth writes bare barlines, so the printed number that
    located a bar upstream cannot find it again.  The kern still carries
    the numbers, and a bar is a stretch of data lines between barlines —
    a stretch with no data at all (a header before the first barline) is
    not a bar — so counting them here gives the position the bar parser
    will arrive at.
    """
    ordinals: Dict[str, int] = {}
    index = 0
    has_data = False
    label = None
    for line in kern_content.splitlines():
        text = line.strip()
        if not text or text.startswith(("!", "*")):
            continue
        if text.startswith("="):
            if has_data and label is not None:
                ordinals.setdefault(label, index)
                index += 1
            elif has_data:
                index += 1
            label = text.split("\t")[0].lstrip("=").rstrip("-;:|!")
            has_data = False
            continue
        has_data = True
    if has_data and label is not None:
        ordinals.setdefault(label, index)

    located = []
    for issue in issues:
        row = dict(issue)
        if row.get("bar_ordinal") is None:
            ordinal = ordinals.get(str(issue.get("bar_label")))
            if ordinal is not None:
                row["bar_ordinal"] = ordinal
                row["bar_index"] = ordinal
        located.append(row)
    return located


def create_single_gt(
    kern_path: Path,
    kern_gt_dir: Path,
    inventory_dir: Path,
    *,
    keep_grace: bool = False,
    keep_trill: bool = False,
    keep_non_trill_ornaments: bool = False,
    keep_arpeggio: bool = False,
    strip_cue: bool = False,
) -> Tuple[str, str, str]:
    """One kern file → kern_gt: standardize, gate, and inventory the diff.

    Files are independent (no shared state), so the same body serves a
    sequential loop and a process pool.  Returns
    (stem, status, diff_families); the ground-truth text is written here so
    only small strings cross a process boundary.
    """
    from src.a2s.piano.tokenizer import tokenize_kern
    from src.score.clean_kern import fold_extra_spines, strip_spines
    from src.score.expand_repeat import (
        expand_kern_repeats, has_expansion_labels)
    from src.score.reconstruct_kern import reconstruct_kern_from_bracket_tokens

    stem = kern_path.stem
    target = kern_gt_dir / f"{stem}.krn"
    diff_path = inventory_dir / f"{stem}.diff"
    best_available: Optional[str] = None

    # Bars phase 1 could not fit in their meter and handed back to
    # silence.  They travel with the kern file and are merged into every
    # outcome below, so a bar withheld upstream stays withheld.
    phase1_issues = read_bar_issues(kern_path)

    def write_target(content: str, issues=()) -> None:
        target.write_text(content, encoding="utf-8")
        write_bar_issues(target, list(phase1_issues) + list(issues))

    def outcome(status: str, families: str = '') -> Tuple[str, str, str]:
        if phase1_issues:
            status = "phase1-metric-timeline: " + str(
                phase1_issues[0].get("message", "bar cleared in phase 1"))
        return stem, status, families

    write_bar_issues(target, list(phase1_issues))

    def diff_families(before: str, after: str) -> List[str]:
        families = set()
        for delta in difflib.ndiff(before.splitlines(), after.splitlines()):
            if delta.startswith('  ') or delta.startswith('? '):
                continue
            line = delta[2:]
            fields = line.split('\t')
            if line.startswith('**') or any(
                    field.startswith(('*staff', '*I', '*clef'))
                    for field in fields):
                families.add('header')
            elif line.startswith('='):
                families.add('barline')
            elif any(field in {'*^', '*v'} for field in fields):
                families.add('voice-operation')
            elif any(field.startswith('*k[') for field in fields):
                families.add('key')
            elif any(re.fullmatch(r'\*M\d+/\d+', field) for field in fields):
                families.add('meter')
            elif any(field == '*-' for field in fields):
                families.add('terminator')
            elif line.startswith('*'):
                families.add('interpretation')
            elif line.startswith('!'):
                families.add('comment')
            elif line:
                families.add('data')
        return sorted(families)

    def write_diff(before: Optional[str], after: Optional[str]) -> str:
        if before is None or after is None or before == after:
            diff_path.unlink(missing_ok=True)
            return ''
        diff = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{stem}:standardized",
            tofile=f"{stem}:canonical",
            lineterm="",
        )
        diff_path.write_text("\n".join(diff) + "\n", encoding="utf-8")
        return ','.join(diff_families(before, after))

    try:
        raw_kern = kern_path.read_text(encoding="utf-8")
        best_available = raw_kern
        # kern/ may retain **dynam for audio rendering; the ground
        # truth transcribes **kern only.
        raw_kern = strip_spines(raw_kern, keep_dynam=False)
        # Hand attribution downstream is spine index, so a third staff has
        # to become voices before repair_ties reads a hand off it.
        raw_kern = fold_extra_spines(raw_kern)
        best_available = raw_kern
        if has_expansion_labels(raw_kern):
            raw_kern = expand_kern_repeats(raw_kern)
        best_available = raw_kern
        if phase1_issues:
            phase1_issues = _phase1_bar_ordinals(phase1_issues, raw_kern)
            write_bar_issues(target, list(phase1_issues))
        trace: Dict[str, str] = {}
        try:
            ground_truth = standardize_kern(
                raw_kern, strip_cue=strip_cue,
                keep_grace=keep_grace,
                keep_trill=keep_trill,
                keep_non_trill_ornaments=keep_non_trill_ornaments,
                keep_arpeggio=keep_arpeggio,
                canonical_trace=trace,
            )
        except KernTokenizerOOV as error:
            write_target(error.kern_content, error.bar_issues)
            write_diff(error.kern_content, None)
            return outcome(f"oov: tokenizer: {error}")
        except KernIncompleteTuplet as error:
            write_target(error.kern_content, error.bar_issues)
            write_diff(error.kern_content, None)
            return outcome(f"incomplete_tuplet: tokenizer: {error}")
        except KernMetricTimelineError as error:
            write_target(error.kern_content, error.bar_issues)
            write_diff(error.kern_content, None)
            return outcome(f"metric_timeline: tokenizer: {error}")
        except KernWriterContractError as error:
            write_target(error.kern_content, error.bar_issues)
            families = write_diff(error.kern_content, error.readback)
            return outcome(f"writer_error: {error}", families)
        except Exception as error:
            fallback = trace.get('prewriter') or best_available
            if fallback is not None:
                write_target(fallback, trace.get('bar_issues', ()))
            diff_path.unlink(missing_ok=True)
            return outcome(f"writer_error: standardizer exception: {error}")

        try:
            tokens = tokenize_kern(ground_truth)
            readback = reconstruct_kern_from_bracket_tokens(tokens)
        except Exception as error:
            write_target(ground_truth)
            write_diff(ground_truth, None)
            return outcome(f"writer_error: exact-gate exception: {error}")
        if readback != ground_truth:
            write_target(ground_truth)
            families = write_diff(ground_truth, readback)
            return outcome("writer_error: exact text differs", families)

        write_target(ground_truth)
        families = write_diff(trace.get('prewriter'), trace.get('canonical'))
        return outcome("success", families)

    except Exception as error:
        if best_available is not None:
            write_target(best_available)
        diff_path.unlink(missing_ok=True)
        return outcome(f"writer_error: worker exception: {error}")
