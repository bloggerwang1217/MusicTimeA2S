"""
Kern-to-kern voice canonicalization.

Reads standardized two-staff kern into per-key attack/release accounts,
decides voice identity with explicit MergeOutcome records, and writes
canonical kern back at whatever voice width the source needs.  A bar the rules
cannot settle is paused: its source layout ships unchanged and the
strict tokenizer downstream reports it as noncanonical.
"""

import logging
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.a2s.piano.tokenizer import (
    IncompleteTupletError,
    MetricTimelineError,
    TokenizerOOVError,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceIssue:
    """A source-side refusal that must survive the string-only writer API."""

    family: str
    message: str
    bar_index: Optional[int] = None
    bar_label: Optional[str] = None
    bar_ordinal: Optional[int] = None


def _voice_issue_from_error(error: ValueError) -> Optional[VoiceIssue]:
    if isinstance(error, TokenizerOOVError):
        family = "oov"
    elif isinstance(error, IncompleteTupletError):
        family = "incomplete_tuplet"
    elif isinstance(error, MetricTimelineError):
        family = "metric_timeline"
    else:
        return None
    return VoiceIssue(
        family=family,
        message=str(error),
        bar_index=getattr(error, "bar_index", None),
        bar_label=getattr(error, "bar_label", None),
        bar_ordinal=getattr(error, "bar_ordinal", None),
    )


_TUPLET_ISSUE_MARKERS = (
    "tuplet group has no member slots",
    "does not align with its tuplet host",
    "falls outside its selected tuplet host",
    "tuplet groups overlap without a shared window",
    "tuplet boundary cannot carry a voice rest",
)


def _issue_family_for_message(message: str) -> str:
    """Tuplet-shaped writer failures belong to the tuplet ledger family."""
    if any(marker in message for marker in _TUPLET_ISSUE_MARKERS):
        return "incomplete_tuplet"
    return "writer_topology"


def _extend_voice_issues(
    target: Optional[List[VoiceIssue]], source: List[VoiceIssue],
) -> None:
    if target is None:
        return
    seen = set(target)
    for issue in source:
        if issue not in seen:
            target.append(issue)
            seen.add(issue)


def _add_voice_issue(
    state: "VoiceState", family: str, message: str, bar_index: int,
) -> None:
    issue = VoiceIssue(
        family=family,
        message=message,
        bar_index=bar_index,
        bar_label=state.bars[bar_index].label or "pickup",
        bar_ordinal=sum(
            bool(bar.data_rows) for bar in state.bars[:bar_index + 1]
        ) - 1,
    )
    if issue not in state.issues:
        state.issues.append(issue)


# =============================================================================
# Duration Grid (128th-note resolution)
# =============================================================================

GRID = 128

RECIP_TO_GRID: Dict[str, int] = {
    "128": 1, "64": 2, "64.": 3, "32": 4, "32.": 6,
    "16": 8, "16.": 12, "8": 16, "8.": 24,
    "4": 32, "4.": 48, "2": 64, "2.": 96,
    "1": 128, "1.": 192,
}

GRID_TO_RECIP: Dict[int, str] = {v: k for k, v in RECIP_TO_GRID.items()}

# =============================================================================
# Pitch Utilities
# =============================================================================

_KERN_BASE_SEMITONES: Dict[str, int] = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
}


# Kern pitch to MIDI mapping
def kern_pitch_to_midi(token: str) -> int:
    """Convert a kern pitch token to MIDI number.

    Uppercase = octave 3 and below, lowercase = octave 4 and above.
    Examples: CCCC=12(C0), CCC=24(C1), CC=36(C2), C=48(C3),
              c=60(C4), cc=72(C5), ccc=84(C6), cccc=96(C7), ccccc=108(C8)
    Accidentals: # = +1, - = -1.
    Returns -1 for rest (r) or unrecognized tokens.
    """
    if not token or token == 'r':
        return -1

    # Strip accidentals from the end
    accidental = 0
    core = token
    while core.endswith('#'):
        accidental += 1
        core = core[:-1]
    while core.endswith('-'):
        accidental -= 1
        core = core[:-1]
    if not core:
        return -1
    letter = core[0].upper()
    if letter not in _KERN_BASE_SEMITONES:
        return -1
    base = _KERN_BASE_SEMITONES[letter]
    if core[0].isupper():
        # Uppercase: C=48(oct3), CC=36(oct2), CCC=24(oct1), CCCC=12(oct0)
        midi = (5 - len(core)) * 12 + base
    else:
        # Lowercase: c=60(oct4), cc=72(oct5), ccc=84(oct6), cccc=96(oct7)
        midi = (4 + len(core)) * 12 + base
    return midi + accidental


# =============================================================================
# Metric Tree
# =============================================================================

class MetricTree:
    """Hierarchical metric structure for a time signature.

    Builds the cell/line hierarchy used by spell() to determine
    position-dependent duration spelling.

    Lines are ranked by importance (lower rank = more important):
      rank 0 = barline (implicit endpoints)
      rank 1 = first interior division (midline, group line, or comp-beat)
      rank 2+ = deeper levels, down to individual 128th-note ticks

    Wall lines (barline, midline, comp-beat lines, 5/4 group line) block
    shapes whose anchor does not strictly outrank the wall.
    """

    SUPPORTED_NUM = frozenset({2, 3, 4, 5, 6, 9, 12})
    SUPPORTED_DEN = frozenset({2, 4, 8, 16})

    def __init__(self, num: int, den: int) -> None:
        if num not in self.SUPPORTED_NUM or den not in self.SUPPORTED_DEN:
            raise TokenizerOOVError(f"Unsupported meter {num}/{den}")
        self.num = num
        self.den = den
        self.bar_length: int = num * (GRID // den)
        self.cells: Set[Tuple[int, int]] = set()
        self.lines: Dict[int, int] = {}
        self.walls: Set[int] = set()
        self._build()

    # ── tree construction ────────────────────────────────────────────────

    def _build(self) -> None:
        self.cells.add((0, self.bar_length))
        leaves, next_rank = self._build_top()
        for start, end in leaves:
            self._binary_sub(start, end, next_rank)

    def _add_line(self, pos: int, rank: int, *, wall: bool = False) -> None:
        if pos not in self.lines or self.lines[pos] > rank:
            self.lines[pos] = rank
        if wall:
            self.walls.add(pos)

    def _add_cells(self, cells: List[Tuple[int, int]]) -> None:
        for c in cells:
            self.cells.add(c)

    def _ternary_split(
        self, start: int, end: int, rank: int,
    ) -> List[Tuple[int, int]]:
        """Split [start, end) into 3 equal parts.  No intermediate cells."""
        u = (end - start) // 3
        p1, p2 = start + u, start + 2 * u
        self._add_line(p1, rank)
        self._add_line(p2, rank)
        cells = [(start, p1), (p1, p2), (p2, end)]
        self._add_cells(cells)
        return cells

    def _build_top(self) -> Tuple[List[Tuple[int, int]], int]:
        """Build the named levels above binary beat subdivision.

        Returns (leaf_cells, next_rank_for_binary_sub).
        """
        BL = self.bar_length
        n = self.num

        if n == 2:
            mid = BL // 2
            self._add_line(mid, 1, wall=True)
            self._add_cells([(0, mid), (mid, BL)])
            return [(0, mid), (mid, BL)], 2

        if n == 3:
            leaves = self._ternary_split(0, BL, 1)
            return leaves, 2

        if n == 4:
            mid = BL // 2
            self._add_line(mid, 1, wall=True)
            self._add_cells([(0, mid), (mid, BL)])
            q = BL // 4
            self._add_line(q, 2)
            self._add_line(3 * q, 2)
            beats = [(0, q), (q, mid), (mid, mid + q), (mid + q, BL)]
            self._add_cells(beats)
            return beats, 3

        if n == 5:
            b = BL // 5
            gl = 3 * b
            self._add_line(gl, 1, wall=True)
            self._add_cells([(0, gl), (gl, BL)])
            g3 = self._ternary_split(0, gl, 2)
            g2_mid = gl + b
            self._add_line(g2_mid, 2)
            g2 = [(gl, g2_mid), (g2_mid, BL)]
            self._add_cells(g2)
            return g3 + g2, 3

        if n == 6:
            mid = BL // 2
            self._add_line(mid, 1, wall=True)
            self._add_cells([(0, mid), (mid, BL)])
            leaves: List[Tuple[int, int]] = []
            for cs in (0, mid):
                leaves.extend(self._ternary_split(cs, cs + mid, 2))
            return leaves, 3

        if n == 9:
            cb = BL // 3
            self._add_line(cb, 1, wall=True)
            self._add_line(2 * cb, 1, wall=True)
            self._add_cells([(0, cb), (cb, 2 * cb), (2 * cb, BL)])
            leaves = []
            for cs in (0, cb, 2 * cb):
                leaves.extend(self._ternary_split(cs, cs + cb, 2))
            return leaves, 3

        # n == 12
        mid = BL // 2
        self._add_line(mid, 1, wall=True)
        self._add_cells([(0, mid), (mid, BL)])
        cb = BL // 4
        self._add_line(cb, 2, wall=True)
        self._add_line(3 * cb, 2, wall=True)
        self._add_cells([(0, cb), (cb, mid), (mid, mid + cb), (mid + cb, BL)])
        leaves = []
        for cs in (0, cb, mid, mid + cb):
            leaves.extend(self._ternary_split(cs, cs + cb, 3))
        return leaves, 4

    def _binary_sub(self, start: int, end: int, rank: int) -> None:
        """Recursively binary-subdivide a cell down to size 1."""
        self.cells.add((start, end))
        size = end - start
        if size <= 1 or size % 2 != 0:
            return
        mid = (start + end) // 2
        self._add_line(mid, rank)
        self.cells.add((start, mid))
        self.cells.add((mid, end))
        self._binary_sub(start, mid, rank + 1)
        self._binary_sub(mid, end, rank + 1)

    # ── queries ──────────────────────────────────────────────────────────

    def rank_at(self, pos: int) -> int:
        """Rank of the line at *pos* (0 = barline, higher = less important)."""
        if pos <= 0 or pos >= self.bar_length:
            return 0
        return self.lines.get(pos, 999)

    def is_cell(self, t: int, d: int) -> bool:
        return (t, t + d) in self.cells

    def biggest_interior_line(self, t: int, d: int) -> Optional[int]:
        """Position of the most-important line strictly inside (t, t+d).

        Ties broken rightmost (front-heavy cut).
        """
        best_pos: Optional[int] = None
        best_rank = 999
        end = t + d
        for pos, rank in self.lines.items():
            if t < pos < end:
                if rank < best_rank or (rank == best_rank and pos > best_pos):
                    best_pos, best_rank = pos, rank
        return best_pos

    # ── licensed-shape tests ────────────────────────────────────────────

    def is_licensed(self, t: int, d: int) -> bool:
        """Is (t, d) a licensed shape that can be emitted as one glyph?"""
        if self._blocked_by_wall(t, d):
            return False
        if self._dotted_test(t, d):
            return True
        if self._two_cell_test(t, d):
            return True
        return False

    def _blocked_by_wall(self, t: int, d: int) -> bool:
        """True if a wall line inside (t, t+d) blocks this shape."""
        r_start = self.rank_at(t)
        end = t + d
        for pos in self.walls:
            if t < pos < end:
                if r_start >= self.lines[pos]:
                    return True
        return False

    def _dotted_test(self, t: int, d: int) -> bool:
        """Larger-end test: dot covers no line bigger than the larger end."""
        if d not in GRID_TO_RECIP or not GRID_TO_RECIP[d].endswith('.'):
            return False
        r_start = self.rank_at(t)
        r_end = self.rank_at(t + d)
        larger_end = min(r_start, r_end)
        end = t + d
        for pos, rank in self.lines.items():
            if t < pos < end:
                if rank < larger_end:
                    return False
        return True

    def _two_cell_test(self, t: int, d: int) -> bool:
        """Two equal-sized cells at one interior line (2-of-3, off-beat)."""
        end = t + d
        for pos in self.lines:
            if t < pos < end:
                if (pos - t) == (end - pos):
                    if (t, pos) in self.cells and (pos, end) in self.cells:
                        return True
        return False

# Module-level cache
_TREE_CACHE: Dict[Tuple[int, int], MetricTree] = {}


def get_metric_tree(num: int, den: int) -> MetricTree:
    """Get or create the MetricTree for a time signature."""
    key = (num, den)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = MetricTree(num, den)
    return _TREE_CACHE[key]


# =============================================================================
# Duration Spelling
# =============================================================================

def spell(t: int, d: int, tree: MetricTree) -> List[str]:
    """Spell duration *d* at position *t* into its glyph sequence.

    Returns a list of duration token strings.  Multiple tokens means the
    value is split (tied) at beat-visibility boundaries.

    All arithmetic is in 128th-note grid units.
    """
    if d <= 0:
        raise ValueError(f"spell: d must be positive, got {d}")

    glyph = GRID_TO_RECIP.get(d)

    # Step 1: cell fill
    if glyph is not None and tree.is_cell(t, d):
        return [glyph]

    # Step 2: licensed shape
    if glyph is not None and tree.is_licensed(t, d):
        return [glyph]

    # Step 3: cut at the biggest interior line, recurse
    cut = tree.biggest_interior_line(t, d)
    if cut is None:
        raise ValueError(
            f"spell: no interior line in [{t}, {t + d}) for "
            f"{tree.num}/{tree.den}"
        )
    left = spell(t, cut - t, tree)
    right = spell(cut, t + d - cut, tree)
    return left + right


# =============================================================================
# Voice-bracket serialization (kern_gt → token sequence)
# =============================================================================

_RECIP_RE = re.compile(r"(\d+)(?:%(\d+))?(\.*)")
# Unanchored: engravers may write the stem before the tie ([2.c#\ vs
# \[2.c#), and a kern note token has no other meaning for '['.
_TIE_START_RE = re.compile(r"\[")
_TIE_END_RE = re.compile(r"\]")
_PITCH_RE = re.compile(r"(([A-G])\2*|([a-g])\3*)[#\-n]*|r")
# q/Q are true graces (no metric duration).  P is an appoggiatura
# designation on a note that DOES occupy metric time — dropping it
# punches a hole in the bar.
_GRACE_RE = re.compile(r"[qQ]")


def _parse_note_token(tok: str):
    """Parse one kern data token.

    Returns:
      ('note', grid_dur, pitch, tie_in, tie_out)
      ('tup', recip_int, pitch, tie_in, tie_out, duration)
      None                                          — unparseable / grace / rational
    """
    if not tok or tok == ".":
        return None
    if _GRACE_RE.search(tok):
        return None
    tie_out = bool(_TIE_START_RE.search(tok))
    tie_in = bool(_TIE_END_RE.search(tok))
    if "_" in tok:
        # Kern tie continuation: the middle fragment of a 3+ fragment
        # chain both lands and departs.
        tie_in = True
        tie_out = True
    m = _RECIP_RE.search(tok)
    if not m:
        return None
    pm = _PITCH_RE.search(tok)
    if not pm:
        return None
    pitch = pm.group(0).replace("n", "")
    if not pitch:
        return None
    if "#" in pitch and "-" in pitch:
        pitch = pitch[: pitch.index("#") + 1]
    if pitch == "r" and (tie_in or tie_out):
        raise TokenizerOOVError(
            f"tie mark on rest is not allowed: {tok!r}")
    from fractions import Fraction as F
    recip_num = int(m.group(1))
    if recip_num == 0:
        if pitch != "r" or m.group(1) != "0":
            # Pitched breve (and longa) are out of vocabulary: fail loudly
            # so the caller skips the chunk instead of silently losing the
            # note.
            raise TokenizerOOVError(
                f"breve duration not in vocabulary: {tok!r}")
        # A breve rest is legal ink for a whole-bar rest in long meters
        # (e.g. 4/2); only its length matters, since the writer re-spells
        # rest spans from scratch.
        dots = len(m.group(3) or "")
        dur = F(2) * (F(2) - F(1, 2 ** dots)) if dots else F(2)
        return ("note", int(dur * GRID), pitch, tie_in, tie_out)
    # Rational recip a%b = the fraction a/b, duration b/a whole notes
    # (e.g. 232%5 = 5/232 whole note).
    recip_den = int(m.group(2)) if m.group(2) else 1
    dots = len(m.group(3) or "")
    dur = F(recip_den, recip_num)
    if dots:
        dur = dur * (F(2) - F(1, 2 ** dots))
    grid = dur * GRID
    base_grid = F(recip_den, recip_num) * GRID
    if grid.denominator == 1 and base_grid.denominator == 1:
        # Integer grid duration AND undotted base is also integer:
        # ordinary note event; non-glyph values are split by spell().
        return ("note", int(grid), pitch, tie_in, tie_out)
    # Non-integer grid, OR a dotted tuplet member whose dot happens
    # to land on an integer grid (e.g. 12. = 1/8 whole = 16 grid):
    # equal-division group member.
    return (
        "tup", F(recip_num, recip_den), pitch,
        tie_in, tie_out, dur,
    )


def _flush_tuplet(
    accum: dict,
    events: List[dict],
) -> None:
    """Emit tuplet groups from accumulator using slot-based layout.

    Flushes whenever accumulated duration reaches a dyadic span.
    A single member whose accumulated span is regular is a plain event.
    Multi-member groups produce is_tup events with per-slot info for
    <.>/</.>.
    """
    from fractions import Fraction as F
    members = accum["members"]
    if not members:
        return

    onset = accum["onset"]
    hand = accum["hand"]
    recip = accum["recip"]
    # All durations in exact 128th-grid Fractions; whole-note units mixed
    # with grid units here once truncated quintuplet runs into fake
    # dotted-32nds with onsets off by 128x.
    slot_dur = F(GRID, 1) / F(recip)

    buf: list = []
    buf_dur = F(0)

    def _flush_group():
        nonlocal onset
        if not buf:
            return
        total = sum(m["dur"] for m in buf)
        grid = int(total)  # caller guarantees integrality

        # There is no equal-division relationship to preserve with only one
        # member; its exact regular span determines the ordinary glyph.
        if len(buf) == 1:
            m = buf[0]
            events.append({
                "onset": onset, "dur": grid, "hand": hand,
                "chain": accum.get("chain", 0),
                "pitches": m["pitches"],
                "tie_ins": m["tie_ins"],
                "tie_outs": m["tie_outs"],
                "_source_row": m.get("source_row"),
                "_source_rows": tuple(accum.get("source_rows", ())),
            })
            onset += total
            buf.clear()
            return

        # Silence has no tuplet identity: a group of nothing but rests
        # (e.g. engraved 3r + 6r) becomes one plain rest span for the
        # writer's canonical rest spelling to redraw.
        if all(m["pitches"] == ["r"] for m in buf):
            events.append({
                "onset": onset, "dur": grid, "hand": hand,
                "chain": accum.get("chain", 0),
                "pitches": ["r"],
                "tie_ins": [False],
                "tie_outs": [False],
                "_source_row": buf[0].get("source_row"),
                "_source_rows": tuple(accum.get("source_rows", ())),
            })
            onset += total
            buf.clear()
            return

        # Build slot layout
        n_slots_f = total / slot_dur
        if n_slots_f.denominator != 1 or n_slots_f <= 0:
            raise IncompleteTupletError(
                f"tuplet span {total} grid units is not a whole number of "
                f"1/{recip} slots"
            )
        n_slots = int(n_slots_f)
        slots = []
        mi = 0
        rem_dur = F(0)
        rem_member = None

        for k in range(n_slots):
            slot = {"cont": [], "cont_ti": [], "cont_to": [], "attack": [],
                    "attack_ti": [], "attack_to": [], "departs": False}
            avail = slot_dur

            if rem_dur > 0 and rem_member is not None:
                slot["cont"] = list(rem_member["pitches"])
                slot["cont_ti"] = [True] * len(rem_member["pitches"])
                if rem_dur > avail:
                    slot["cont_to"] = [True] * len(rem_member["pitches"])
                    rem_dur -= avail
                    slot["departs"] = True
                    avail = F(0)
                elif rem_dur == avail:
                    slot["cont_to"] = list(rem_member["tie_outs"])
                    slot["departs"] = any(rem_member["tie_outs"])
                    rem_dur = F(0)
                    rem_member = None
                    avail = F(0)
                else:
                    slot["cont_to"] = list(rem_member["tie_outs"])
                    slot["departs"] = any(rem_member["tie_outs"])
                    avail -= rem_dur
                    rem_dur = F(0)
                    rem_member = None

            # When a continuation filled the slot (avail=0) but another
            # member ATTACKS at this same slot time, it is a concurrent
            # voice (e.g. F# sounding while D# continues).  Let it in
            # without consuming time.  The onset gate keeps sequential
            # members (a run following a long note) in their own slots.
            if avail == 0 and mi < len(buf) and rem_member is not None:
                m = buf[mi]
                if (m["dur"] <= slot_dur
                        and m.get("onset") == onset + k * slot_dur):
                    mi += 1
                    slot["attack"].extend(m["pitches"])
                    slot["attack_ti"].extend(m["tie_ins"])
                    slot["attack_to"].extend(m["tie_outs"])

            while avail > 0 and mi < len(buf):
                m = buf[mi]
                mi += 1
                first_attack = len(slot["attack"])
                slot["attack"].extend(m["pitches"])
                slot["attack_ti"].extend(m["tie_ins"])
                slot["attack_to"].extend(m["tie_outs"])
                if m["dur"] <= avail:
                    if m["dur"] == avail and any(m["tie_outs"]):
                        slot["departs"] = True
                    avail -= m["dur"]
                else:
                    slot["attack_to"][first_attack:] = [
                        True
                    ] * len(m["pitches"])
                    rem_dur = m["dur"] - avail
                    rem_member = m
                    slot["departs"] = True
                    avail = F(0)

            slots.append(slot)

        for slot in slots:
            pitches = list(slot.get("attack", ())) + list(
                slot.get("cont", ()))
            if "r" not in pitches or any(pitch != "r" for pitch in pitches):
                continue
            slot["cont"] = []
            slot["cont_ti"] = []
            slot["cont_to"] = []
            slot["attack"] = ["r"]
            slot["attack_ti"] = [False]
            slot["attack_to"] = [False]
            slot["departs"] = False

        events.append({
            "onset": onset, "dur": grid, "hand": hand,
            "chain": accum.get("chain", 0),
            "is_tup": True, "tup_n": n_slots, "slots": slots,
            "_source_row": min(accum.get("source_rows", [0])),
            "_source_rows": tuple(accum.get("source_rows", ())),
        })
        onset += total
        buf.clear()

    for m in members:
        buf.append(m)
        buf_dur += m["dur"]
        if buf_dur.denominator == 1 and int(buf_dur) in GRID_TO_RECIP:
            _flush_group()
            buf_dur = F(0)

    if buf:
        raise IncompleteTupletError(
            f"tuplet group leftover does not reach a dyadic span: "
            f"{buf_dur} grid units of 1/{recip} members"
        )


def _parse_bar_events(data_lines: List[str],
                      row_channels: Optional[List[list]] = None,
                      aux: Optional[dict] = None,
                      fold_map: Optional[Dict[tuple, tuple]] = None,
                      row_rebinds: Optional[List[Optional[dict]]] = None,
                      ) -> List[dict]:
    """Parse kern data lines for one bar into event dicts.

    Per-channel time tracking: a channel is (hand, chain) — chain 0 is a
    hand's single line or LEFT sub-column, chain 1 the RIGHT sub-column.
    Each channel advances independently by its own min duration, which
    handles polymetric passages (e.g. triplets vs dotted notes across
    channels).  Tuplet members accumulate into one is_tup event whose
    total span is always dyadic.

    Without row_channels, columns 0/1 are the two hands' single lines
    (the flattened two-spine layout).  With it, each row's columns map to
    channels; a chain-1 channel appearing mid-bar starts at chain 0's
    clock (both sub-columns continue the single line's timeline), and
    when it disappears the merged line resumes at the later clock.
    """
    from fractions import Fraction as F
    events: List[dict] = []
    spine_time: Dict[tuple, F] = {}
    tup_accum: Dict[tuple, Optional[dict]] = {}
    default_map = [(0, 0), (1, 0)]
    fold_map = fold_map or {}

    def _base(ch: tuple) -> tuple:
        # a fold column continues its mother line's timeline, the way a
        # chain-1 sub-column continues the single line's
        return fold_map.get(ch, (ch[0], 0))

    def _close_tup(ch: tuple):
        acc = tup_accum.get(ch)
        if acc is None:
            return
        n_before = len(events)
        _flush_tuplet(acc, events)
        n_after = len(events)
        if n_after > n_before:
            last = events[-1]
            spine_time[ch] = last["onset"] + last["dur"]
        tup_accum[ch] = None

    prev_present: Optional[set] = None
    for ri, line in enumerate(data_lines):
        parts = line.split("\t")
        chans = row_channels[ri] if row_channels is not None else default_map

        present = {c for c in chans
                   if c is not None and (c[1] < 2 or c in fold_map)}
        for ch in present:
            spine_time.setdefault(ch, F(0))
            tup_accum.setdefault(ch, None)
            if aux is not None and ch[1] == 1:
                aux.setdefault("present", set()).add(ch[0])
        if prev_present is not None:
            for ch in sorted(present - prev_present):
                spine_time[ch] = spine_time.get(_base(ch), F(0))
                tup_accum[ch] = None
                if aux is not None and ch[1] == 1:
                    aux.setdefault("starts", []).append(
                        (ch[0], spine_time[ch]))
            rb0 = row_rebinds[ri] if row_rebinds is not None else None
            flowed = set()
            if rb0:
                for srcs in rb0.values():
                    flowed.update(srcs)
            for ch in sorted(prev_present - present):
                if ch in flowed:
                    # timeline continues under a new name; the flow
                    # block below carries clock and tuplet run
                    continue
                _close_tup(ch)
                base = _base(ch)
                spine_time[base] = max(spine_time.get(base, F(0)),
                                       spine_time.get(ch, F(0)))
                if aux is not None and ch[1] == 1:
                    aux.setdefault("ends", []).append(
                        (ch[0], spine_time[base]))
        prev_present = present

        # Apply the scan's clock flow: a channel name here received its
        # timeline from these sources this instant (survivor of a *v
        # run resumes at the latest member clock; a renamed column
        # carries its own clock and tuplet run to the new name).
        rb = row_rebinds[ri] if row_rebinds is not None else None
        if rb:
            vals = {}
            accs = {}
            for name, srcs in rb.items():
                ts = [spine_time[s2] if s2 in spine_time
                      else spine_time.get(_base(s2), F(0))
                      for s2 in srcs]
                vals[name] = max(ts) if ts else F(0)
                if len(srcs) == 1:
                    accs[name] = tup_accum.get(srcs[0])
                    tup_accum[srcs[0]] = None
                else:
                    for s2 in srcs:
                        _close_tup(s2)
                    accs[name] = None
            for name in vals:
                if name in tup_accum and tup_accum.get(name) is not None \
                        and accs.get(name) is not tup_accum.get(name):
                    _close_tup(name)
                spine_time[name] = vals[name]
                tup_accum[name] = accs.get(name)

        if aux is not None:
            row_clock = min(
                (spine_time[channel] for channel in present),
                default=F(0),
            )
            aux.setdefault("row_onsets", {})[ri] = row_clock

        for ci in range(min(len(chans), len(parts))):
            ch = chans[ci]
            if ch is None or (ch[1] >= 2 and ch not in fold_map):
                continue
            sp = parts[ci].strip()
            if sp == ".":
                continue
            toks = sp.split()
            # Kern chord shorthand: a chord note may omit the recip and
            # inherit it from the nearest preceding note ("4F FF" =
            # octave).  Without this the duration-less notes vanish
            # silently.  Inheritance is local: after a grace, a bare
            # pitch is that grace chord's member, not a real note.
            fixed = []
            lead: Optional[str] = None
            lead_grace = False
            for t in toks:
                if re.search(r"\d", t):
                    # Prefix class includes stem ink: a mark before
                    # the digits must not turn a timed member into a
                    # "bare pitch" that inherits a second duration.
                    m2 = re.match(r"^[\[({/\\]*(\d+\.*)", t)
                    lead = m2.group(1) if m2 else None
                    lead_grace = bool(_GRACE_RE.search(t))
                    fixed.append(t)
                elif (lead and not lead_grace
                      and not _GRACE_RE.search(t)
                      and re.search(r"[a-gA-Gr]", t)):
                    fixed.append(re.sub(r"^([\[({/\\]*)",
                                        r"\g<1>" + lead, t, count=1))
                else:
                    fixed.append(t)
            toks = fixed
            parsed = [_parse_note_token(t) for t in toks]
            parsed = [p for p in parsed if p is not None]
            if not parsed:
                continue

            note_list = [p for p in parsed if p[0] == "note"]
            tup_list = [p for p in parsed if p[0] == "tup"]

            if tup_list:
                recip = tup_list[0][1]
                # Flush if switching tuplet family (odd core differs)
                def _odd_core(n):
                    if isinstance(n, F):
                        n = n.numerator
                    while n % 2 == 0:
                        n //= 2
                    return n
                if tup_accum[ch] is not None:
                    acc = tup_accum[ch]
                    expected = acc["onset"] + sum(
                        m["dur"] for m in acc["members"]
                    )
                    if spine_time[ch] != expected:
                        t_now = spine_time[ch]
                        _close_tup(ch)
                        spine_time[ch] = t_now
                if tup_accum[ch] is not None:
                    old_core = _odd_core(tup_accum[ch]["recip"])
                    new_core = _odd_core(recip)
                    if old_core != new_core:
                        t_now = spine_time[ch]
                        _close_tup(ch)
                        spine_time[ch] = t_now
                    elif recip > tup_accum[ch]["recip"]:
                        tup_accum[ch]["recip"] = recip
                if tup_accum[ch] is None:
                    tup_accum[ch] = {
                        "onset": spine_time[ch], "recip": recip,
                        "hand": ch[0], "chain": ch[1], "members": [],
                        "source_rows": [],
                    }
                tup_accum[ch]["source_rows"].append(ri)
                member_pitches = [p[2] for p in tup_list]
                member_ti = [p[3] for p in tup_list]
                member_to = [p[4] for p in tup_list]
                member_dur = max(p[5] for p in tup_list) * GRID
                acc_members = tup_accum[ch]["members"]
                # Keep explicit tie links as separate members until group
                # boundaries have been recovered.  The account layer joins
                # their sounding identity after the hidden grid is stable.
                acc_members.append({
                    "pitches": member_pitches,
                    "tie_ins": member_ti,
                    "tie_outs": member_to,
                    "dur": member_dur,
                    "onset": spine_time[ch],
                    "source_row": ri,
                })

            if note_list:
                by_dur: Dict[tuple, dict] = {}
                rest_seq = 0
                for _, grid_dur, pitch, ti, to in note_list:
                    # Rests never join chords: each rest is its own event
                    # even at a shared duration.
                    if pitch == "r":
                        key = (grid_dur, rest_seq)
                        rest_seq += 1
                    else:
                        key = (grid_dur, -1)
                    if key not in by_dur:
                        by_dur[key] = {
                            "onset": spine_time[ch], "dur": grid_dur,
                            "hand": ch[0], "chain": ch[1],
                            "pitches": [], "tie_ins": [],
                            "tie_outs": [],
                            "_source_row": ri,
                            "_source_rows": (ri,),
                        }
                    by_dur[key]["pitches"].append(pitch)
                    by_dur[key]["tie_ins"].append(ti)
                    by_dur[key]["tie_outs"].append(to)
                for ev in by_dur.values():
                    zipped = sorted(
                        zip(
                            ev["pitches"], ev["tie_ins"],
                            ev["tie_outs"],
                        ),
                        key=lambda x: kern_pitch_to_midi(x[0]),
                    )
                    ev["pitches"] = [z[0] for z in zipped]
                    ev["tie_ins"] = [z[1] for z in zipped]
                    ev["tie_outs"] = [z[2] for z in zipped]
                    events.append(ev)

            # One time advance per cell, over ALL parsed items.
            cell_durs = []
            for p in parsed:
                if p[0] == "note":
                    cell_durs.append(F(p[1]))
                else:
                    cell_durs.append(p[5] * GRID)
            if cell_durs:
                spine_time[ch] += min(cell_durs)

    for ch in list(tup_accum):
        _close_tup(ch)

    return events


def _expand_spell(events: List[dict], tree: MetricTree) -> List[dict]:
    """Expand events whose duration is not a glyph into tied glyph sequences.

    Glyph durations pass through as-is regardless of metric position —
    sub-beat spelling is a post-processing concern (reconstruction),
    not something the model needs to learn.  Only non-glyph durations
    (from polyphonic merge or other sources) require spell() to produce
    valid vocab tokens.
    """
    expanded: List[dict] = []
    for ev in events:
        if ev.get("is_tup"):
            if ev["onset"] >= tree.bar_length:
                raise ValueError(
                    f"tuplet onset {ev['onset']} >= bar_length {tree.bar_length} "
                    f"({tree.num}/{tree.den})"
                )
            if ev["onset"] + ev["dur"] > tree.bar_length:
                raise ValueError(
                    f"tuplet exceeds bar: onset={ev['onset']} dur={ev['dur']} "
                    f"bar_length={tree.bar_length}"
                )
            expanded.append(ev)
            continue
        if ev["onset"] >= tree.bar_length:
            raise ValueError(
                f"onset {ev['onset']} >= bar_length {tree.bar_length} "
                f"({tree.num}/{tree.den}), pitches={ev['pitches']}"
            )
        if ev["onset"] + ev["dur"] > tree.bar_length:
            raise ValueError(
                f"event exceeds bar: onset={ev['onset']} dur={ev['dur']} "
                f"bar_length={tree.bar_length} ({tree.num}/{tree.den}), "
                f"pitches={ev['pitches']}"
            )
        if ev["dur"] <= 0:
            raise ValueError(
                f"dur {ev['dur']} <= 0, onset={ev['onset']}, "
                f"bar_length={tree.bar_length}"
            )
        if ev["dur"] in GRID_TO_RECIP:
            expanded.append(ev)
            continue
        glyphs = spell(ev["onset"], ev["dur"], tree)
        pos = ev["onset"]
        # Rests never tie: a multi-glyph silence is adjacent rest brackets
        # with no marks.
        is_rest = ev["pitches"] == ["r"]
        for gi, g in enumerate(glyphs):
            gd = RECIP_TO_GRID[g]
            mid_in = [False] if is_rest else [True] * len(ev["pitches"])
            mid_out = [False] if is_rest else [True] * len(ev["pitches"])
            new_ev = {
                "onset": pos, "dur": gd, "hand": ev["hand"],
                "chain": ev.get("chain", 0),
                "pitches": list(ev["pitches"]),
                "tie_ins": ev["tie_ins"][:] if gi == 0 else mid_in,
                "tie_outs": ev["tie_outs"][:] if gi == len(glyphs) - 1
                    else mid_out,
            }
            if "voice" in ev:
                new_ev["voice"] = ev["voice"]
            expanded.append(new_ev)
            pos += gd
    return expanded


def _regroup_same_onset(events: List[dict]) -> List[dict]:
    """Merge same-(hand, onset, duration) note events into one chord.

    Tie-chain merging can leave a merged single-pitch event colliding
    with a native event at the same onset and duration; serializing them
    as two brackets reads as sequential playing. Rests never join chords.
    """
    groups: Dict[tuple, dict] = {}
    out: List[dict] = []
    for ev in events:
        if ev.get("is_tup") or ev["pitches"] == ["r"]:
            out.append(ev)
            continue
        key = (ev["hand"], ev.get("chain", 0), ev["onset"], ev["dur"])
        if key in groups:
            g = groups[key]
            g["pitches"].extend(ev["pitches"])
            g["tie_ins"].extend(ev["tie_ins"])
            g["tie_outs"].extend(ev["tie_outs"])
        else:
            ev = dict(
                ev,
                pitches=list(ev["pitches"]),
                tie_ins=list(ev["tie_ins"]),
                tie_outs=list(ev["tie_outs"]),
            )
            groups[key] = ev
            out.append(ev)
    for g in groups.values():
        zipped = sorted(
            zip(g["pitches"], g["tie_ins"], g["tie_outs"]),
            key=lambda x: kern_pitch_to_midi(x[0]),
        )
        g["pitches"] = [z[0] for z in zipped]
        g["tie_ins"] = [z[1] for z in zipped]
        g["tie_outs"] = [z[2] for z in zipped]
    return out


def _chan(ev: dict) -> tuple:
    """Channel key: (hand, chain).  Flattened input has no chain field, so
    every event falls in chain 0 and per-channel logic reduces to the
    per-hand behavior."""
    return (ev["hand"], ev.get("chain", 0))


def _close_rest_run(run: List[dict], tree: MetricTree) -> List[dict]:
    if len(run) < 2:
        return list(run)
    total = sum(e["dur"] for e in run)
    if total in GRID_TO_RECIP:
        # Rests take the note path: a span whose value
        # is one glyph is ONE rest token, position-free — same
        # pass-through as notes.
        piece = dict(run[0], onset=run[0]["onset"], dur=total,
                     tie_ins=[False], tie_outs=[False])
        piece.pop("birth", None)
        piece.pop("death", None)
        if any(e.get("birth") for e in run):
            piece["birth"] = True
        if any(e.get("death") for e in run):
            piece["death"] = True
        return [piece]
    glyphs = spell(run[0]["onset"], total, tree)
    pos = run[0]["onset"]
    pieces = []
    for g in glyphs:
        piece = dict(run[0], onset=pos, dur=RECIP_TO_GRID[g],
                     tie_ins=[False], tie_outs=[False])
        piece.pop("birth", None)
        piece.pop("death", None)
        pieces.append(piece)
        pos += piece["dur"]
    if any(e.get("birth") for e in run):
        pieces[0]["birth"] = True
    if any(e.get("death") for e in run):
        pieces[-1]["death"] = True
    return pieces


def _merge_rest_spans(events: List[dict], tree: MetricTree) -> List[dict]:
    """Collapse a channel's contiguous rests into one silence, respelled.

    A silence has no attack, so no printed mark delimits it and the span
    itself is the unit; f names the glyphs again.  This runs after the
    voice pass because that is where a voice's silent stretches first get
    their printed rests — merging earlier disagrees with reading a printed
    bar back.  A shared onset (flattened voices) leaves its rest as read.
    """
    by_chan: Dict[tuple, List[dict]] = {}
    for ev in events:
        key = _chan(ev)
        by_chan.setdefault(key, []).append(ev)

    out: List[dict] = []
    for chan in by_chan:
        evs = sorted(by_chan[chan], key=lambda e: e["onset"])
        n_at: Dict[object, int] = {}
        for e in evs:
            n_at[e["onset"]] = n_at.get(e["onset"], 0) + 1
        run: List[dict] = []
        for ev in evs:
            solo_rest = (not ev.get("is_tup") and ev["pitches"] == ["r"]
                         and n_at[ev["onset"]] == 1)
            if (solo_rest and run
                    and ev["onset"] == run[-1]["onset"] + run[-1]["dur"]
                    and not ev.get("_preserve_rest_boundary")
                    and not run[-1].get("_preserve_rest_boundary")):
                run.append(ev)
                continue
            out.extend(_close_rest_run(run, tree))
            run = [ev] if solo_rest else []
            if not solo_rest:
                out.append(ev)
        out.extend(_close_rest_run(run, tree))
    return out


# =============================================================================
# Kern-to-kern voice canonicalization
# =============================================================================


@dataclass(frozen=True)
class MergeOutcome:
    """One explicit musical decision made by the voice canonicalizer."""

    decision: str
    reason: str
    source: object = None
    target: object = None
    span: object = None
    execution: str = "applied"
    evidence_span: object = None
    effect_span: object = None


@dataclass
class WrittenColumn:
    """One stable source voice life, from its birth until its retirement."""

    channel: tuple
    parent: Optional[tuple] = None
    merge_parents: tuple = ()
    birth_row: Optional[int] = None
    death_row: Optional[int] = None
    survivor: Optional[tuple] = None


@dataclass
class TopologyNode:
    """One explicit leaf segment between source split/merge operations."""

    node_id: int
    hand: int
    parents: tuple = ()
    birth_row: Optional[int] = None
    death_row: Optional[int] = None
    positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class AttackReleaseAccount:
    """One piano-key account, including all of its tie pieces."""

    account_id: int
    hand: int
    pitch: Optional[str]
    start: Fraction
    end: Fraction
    source: tuple
    attacked: bool
    piece_ids: List[int] = field(default_factory=list)
    owner: Optional[tuple] = None
    retired: bool = False


@dataclass
class _PitchPiece:
    piece_id: int
    event: dict
    pitch: Optional[str]
    hand: int
    start: Fraction
    end: Fraction
    tie_in: bool
    tie_out: bool
    source: tuple
    source_row: Optional[int]
    owner: tuple


@dataclass
class _CarriedRow:
    row: int
    onset: Fraction
    text: str
    channels: tuple
    after_data: bool = False


@dataclass
class _VoiceBar:
    index: int
    label: str
    closing_label: str
    start_row: int
    stop_row: int
    tree: MetricTree
    offset: Fraction
    events: List[dict]
    data_rows: List[int]
    carried: List[_CarriedRow]


@dataclass
class _WriterBar:
    events: List[dict]
    widths: Dict[int, int]
    lanes: tuple
    lane_hands: Dict[tuple, int]
    incoming: tuple
    outgoing: tuple
    head_ops: List[str]
    tail_ops: List[str]
    source_rows: Optional[List[str]] = None


@dataclass(frozen=True)
class _SeamCancellation:
    rows: tuple
    lanes: tuple
    operations: tuple


@dataclass
class VoiceState:
    """Musical state used by all merge rules and by the canonical writer."""

    original: str
    lines: List[str]
    header: List[str]
    trailer: List[str]
    bars: List[_VoiceBar]
    columns: Dict[tuple, WrittenColumn]
    row_channels: Dict[int, list]
    topology_nodes: Dict[int, TopologyNode] = field(default_factory=dict)
    row_nodes: Dict[int, list] = field(default_factory=dict)
    op_layouts: Dict[int, Tuple[tuple, tuple, tuple]] = field(
        default_factory=dict)
    initial_channels: tuple = ()
    outcomes: List[MergeOutcome] = field(default_factory=list)
    pieces: List[_PitchPiece] = field(default_factory=list)
    accounts: List[AttackReleaseAccount] = field(default_factory=list)
    paused_bars: Set[int] = field(default_factory=set)
    decision_blocked: Set[int] = field(default_factory=set)
    issues: List[VoiceIssue] = field(default_factory=list)
    written_hands: Dict[Tuple[int, tuple], int] = field(
        default_factory=dict)
    suppressed_tuple_atoms: Dict[
        int, Set[Tuple[str, Fraction, Fraction]]
    ] = field(default_factory=dict)
    seam_cancellations: Tuple[_SeamCancellation, ...] = ()
    source_topology: Optional["SourceTopology"] = None
    canonical_order: bool = False
    decision_mutated: bool = False


class _CanonicalizationPause(ValueError):
    pass


class _WriterDecisionConflict(ValueError):
    """The writer held bars that this round's decisions relied on.

    A held bar keeps its source column layout, so any decision that
    changes lane counts at or next to it writes two disagreeing
    topologies into one file.  The round must be redecided with those
    bars blocked."""

    def __init__(self, bars: Set[int]):
        super().__init__(
            f"writer hold invalidates decisions at bars {sorted(bars)}")
        self.bars = set(bars)


def _material_hold_bars(state: VoiceState) -> Set[int]:
    """Return bars whose source text must remain topologically intact."""
    result = set(state.decision_blocked)
    for bar_index in state.paused_bars:
        bar = state.bars[bar_index]
        # A zero-time segment can carry key or meter interpretations without
        # carrying a voice life.  Only a real topology operation inside such
        # a segment constrains the adjacent writer layout.
        if bar.data_rows or any(
                _is_spine_operation(state.lines[row])
                for row in range(bar.start_row + 1, bar.stop_row)):
            result.add(bar_index)
    return result


_SPINE_OPS = {"*^", "*v", "*x", "*+", "*-"}


def _zero_time_operation_groups(lines: List[str]) -> Dict[int, tuple]:
    """Group structural rows separated only by zero-time annotations."""
    structural = {
        row for row, line in enumerate(lines)
        if _is_spine_operation(line)
    }
    result: Dict[int, tuple] = {}
    claimed: Set[int] = set()
    for start in sorted(structural):
        if start in claimed:
            continue
        rows = []
        for row in range(start, len(lines)):
            line = lines[row]
            if row > start and (
                    line.startswith("=")
                    or (line and not line.startswith(("!", "*")))):
                break
            if row in structural:
                rows.append(row)
        if rows:
            group = tuple(rows)
            result[start] = group
            claimed.update(group)
    return result


def _normalized_zero_time_group(
    lines: List[str],
    rows: tuple,
    columns: tuple,
) -> Optional[Tuple[_SeamCancellation, Dict[int, str]]]:
    """Recognize one count-neutral close/open component in source space.

    Each temporary slot carries the set of source positions that reached it.
    A seam disappears only when the final bipartite components contain the
    same contiguous input and output positions in the same hand.  This
    admits nested close/open trees but rejects exchanges and operations on
    unrelated branches.
    """
    if len(rows) < 2 or any(column is None for column in columns):
        return None
    first = rows[0]
    last = rows[-1]
    provenance = [frozenset({index}) for index in range(len(columns))]
    snapshots: List[Tuple[int, tuple, tuple]] = []
    operations = []
    merge_rows = []
    split_rows = []

    for row in range(first, last + 1):
        line = lines[row]
        if not line or line.startswith("!!"):
            continue
        fields = tuple(line.split("\t"))
        if row not in rows:
            if line.startswith(("!", "*")):
                if len(fields) != len(provenance):
                    return None
                snapshots.append((row, fields, tuple(provenance)))
            continue
        if len(fields) != len(provenance) \
                or any(token in {"*x", "*+", "*-"} for token in fields):
            return None
        operations.append((row, fields))
        output = []
        index = 0
        while index < len(fields):
            token = fields[index]
            if token == "*v":
                stop = index + 1
                while stop < len(fields) and fields[stop] == "*v":
                    stop += 1
                if stop - index < 2:
                    return None
                merged = frozenset().union(*provenance[index:stop])
                output.append(merged)
                merge_rows.append(row)
                index = stop
                continue
            if token == "*^":
                output.extend((provenance[index], provenance[index]))
                split_rows.append(row)
            elif token in _SPINE_OPS:
                return None
            else:
                output.append(provenance[index])
            index += 1
        provenance = output

    if not merge_rows or not split_rows \
            or max(merge_rows) >= min(split_rows) \
            or len(provenance) != len(columns):
        return None

    parent = list(range(len(columns)))

    def find(member: int) -> int:
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for sources in provenance:
        members = sorted(sources)
        if not members:
            return None
        for member in members[1:]:
            union(members[0], member)
    components: Dict[int, Set[int]] = defaultdict(set)
    for member in range(len(columns)):
        components[find(member)].add(member)
    changed = False
    for component in components.values():
        outputs = [
            position for position, sources in enumerate(provenance)
            if sources & component
        ]
        inputs = sorted(component)
        if outputs != inputs \
                or any(not sources <= component
                       for sources in (provenance[position]
                                       for position in outputs)) \
                or len({columns[position][0] for position in inputs}) != 1:
            return None
        changed |= len(component) > 1
    if not changed:
        return None

    expansions: Dict[int, str] = {}
    for row, fields, snapshot in snapshots:
        expanded = []
        for source_position in range(len(columns)):
            candidates = [
                position for position, sources in enumerate(snapshot)
                if source_position in sources
            ]
            tokens = {fields[position] for position in candidates}
            if not candidates or len(tokens) != 1:
                return None
            expanded.append(next(iter(tokens)))
        expansions[row] = "\t".join(expanded)

    return (
        _SeamCancellation(
            rows=rows,
            lanes=tuple(columns),
            operations=tuple(operations),
        ),
        expansions,
    )


class SourceTopology:
    """Boundary-normalized reading of the source lane operations.

    Built once per parsed state; owns every interpretation of raw
    ``*^``/``*v`` row positions: which bar an operation belongs to once
    pushed to its legal boundary, and which bars still need relocation.
    Material holds are decision state, so callers pass them in.
    """

    def __init__(self, state: "VoiceState") -> None:
        self.tail_split_targets = self._scan_tail_splits(state)
        self.raw_tail_merge_targets = self._scan_head_merges(state)
        self.relocation_bars = self._scan_relocation_bars(state)
        self._state = state
        self._columns = state.columns
        self._bar_lanes_cache: Dict[int, tuple] = {}
        self._written_lanes_cache: Dict[int, tuple] = {}
        self._row_births_cache: Dict[int, Tuple[tuple, ...]] = {}
        self._node_span_cache: Dict[int, Optional[Tuple[int, int]]] = {}
        self._row_bar = {
            row: bar.index
            for bar in state.bars
            for row in bar.data_rows
        }

    @staticmethod
    def _scan_tail_splits(state: "VoiceState") -> Dict[int, int]:
        """Map source tail splits to the bar whose data they introduce."""
        targets: Dict[int, int] = {}
        for bar in state.bars[:-1]:
            if not bar.data_rows:
                continue
            last_data = max(bar.data_rows)
            for row, (_before, tokens, _after) in state.op_layouts.items():
                if last_data < row < bar.stop_row and "*^" in tokens:
                    targets[row] = bar.index + 1
        return targets

    @staticmethod
    def _scan_head_merges(state: "VoiceState") -> Dict[int, int]:
        """Map source head-only joins to the preceding canonical bar tail."""
        targets: Dict[int, int] = {}
        for bar in state.bars[1:]:
            if not bar.data_rows:
                continue
            first_data = min(bar.data_rows)
            leading = [
                (row, layout)
                for row, layout in sorted(state.op_layouts.items())
                if bar.start_row < row < first_data
            ]
            if not leading or any(
                    token not in {"*", "*v"}
                    for _row, (_before, tokens, _after) in leading
                    for token in tokens):
                continue
            for row, (_before, tokens, _after) in leading:
                if "*v" in tokens:
                    targets[row] = bar.index - 1
        return targets

    @staticmethod
    def _scan_relocation_bars(state: "VoiceState") -> Set[int]:
        """Bars whose source split/merge rows are not at their legal edge."""
        result: Set[int] = set()
        for bar in state.bars:
            if not bar.data_rows:
                continue
            first_data = min(bar.data_rows)
            last_data = max(bar.data_rows)
            for row in range(bar.start_row + 1, bar.stop_row):
                cells = state.lines[row].split("\t")
                if ("*^" in cells and row > first_data) \
                        or ("*v" in cells and row < last_data):
                    result.add(bar.index)
                    break
        return result

    def layout_before_deferred_splits(self, row: int, after: tuple) -> tuple:
        """Keep same-row merges while postponing newborn split lanes."""
        children = {
            channel for channel, column in self._columns.items()
            if column.birth_row == row and column.parent is not None
        }
        return tuple(channel for channel in after if channel not in children)

    def incoming_layout(
        self,
        bar: "_VoiceBar",
        tail_merge_targets: Optional[Dict[int, int]] = None,
    ) -> tuple:
        """Source lanes entering a bar under one explicit seam policy."""
        merge_targets = (
            self.raw_tail_merge_targets
            if tail_merge_targets is None else tail_merge_targets
        )
        incoming = self._state.initial_channels
        for row, (_before, _tokens, after) in sorted(
                self._state.op_layouts.items()):
            if row > bar.start_row or (row == bar.start_row and bar.label):
                if merge_targets.get(row) == bar.index - 1:
                    incoming = after
                    continue
                break
            incoming = (
                self.layout_before_deferred_splits(row, after)
                if self.tail_split_targets.get(row) == bar.index else after
            )
        return tuple(incoming)

    def _row_births(self, row: int) -> Tuple[tuple, ...]:
        """Read actual births from one positional operation row."""
        cached = self._row_births_cache.get(row)
        if cached is not None:
            return cached
        before, tokens, after = self._state.op_layouts[row]
        births: List[tuple] = []
        out_pos = 0
        index = 0
        while index < len(tokens):
            token = tokens[index]
            inp = before[index] if index < len(before) else None
            if token == "*v":
                stop = index + 1
                while stop < len(tokens) and tokens[stop] == "*v":
                    stop += 1
                index = stop
                out_pos += 1
            elif token in {"*^", "*+"}:
                first = after[out_pos] if out_pos < len(after) else None
                second = (after[out_pos + 1]
                          if out_pos + 1 < len(after) else None)
                if inp is not None and second == inp and first is not None:
                    births.append(first)
                else:
                    if second is not None:
                        births.append(second)
                out_pos += 2
                index += 1
            elif token == "*-":
                index += 1
            else:
                out_pos += 1
                index += 1
        result = tuple(births)
        self._row_births_cache[row] = result
        return result

    def bar_lanes(self, bar: "_VoiceBar") -> tuple:
        """Ordered source lives present in this bar, every op at a boundary.

        Pure source view: no account decisions or hand-move overlays.  Mid-bar
        births advance to the bar head, mid-bar deaths delay to the bar
        tail, so every listed life spans the whole bar.  Presence comes
        from the layout replay itself: channel ids are recycled, so a
        life's rows are not one contiguous interval.  Two sibling lives
        that share one written slot inside the bar both appear here; slot
        economy is the writer's concern, not identity evidence.
        """
        cached = self._bar_lanes_cache.get(bar.index)
        if cached is not None:
            return cached
        lanes = list(self.incoming_layout(bar))
        if any(channel is None for channel in lanes):
            raise _CanonicalizationPause(
                "voice writer cannot move a non-kern spine operation")
        for row in sorted(self._state.op_layouts):
            deferred_head = self.tail_split_targets.get(row) == bar.index
            if not deferred_head:
                if not bar.start_row < row < bar.stop_row:
                    continue
                if self.raw_tail_merge_targets.get(row) == bar.index - 1:
                    continue
                if self.tail_split_targets.get(row) == bar.index + 1:
                    continue
            _before, _tokens, after = self._state.op_layouts[row]
            for channel in self._row_births(row):
                if channel in lanes:
                    continue
                position = after.index(channel)
                anchor = next(
                    (after[left] for left in range(position - 1, -1, -1)
                     if after[left] in lanes), None)
                if anchor is None:
                    lanes.insert(0, channel)
                else:
                    lanes.insert(lanes.index(anchor) + 1, channel)
        result = tuple(lanes)
        self._bar_lanes_cache[bar.index] = result
        return result

    def logical_order(
        self, bar: "_VoiceBar", members: Iterable[tuple],
    ) -> Optional[Tuple[tuple, ...]]:
        """Return bar-local voice order from the normalized source layout."""
        unique = tuple(dict.fromkeys(members))
        if not unique or len({member[0] for member in unique}) != 1:
            return None
        lanes = self.bar_lanes(bar)
        if any(member not in lanes for member in unique):
            return None
        member_set = set(unique)
        physical = tuple(lane for lane in lanes if lane in member_set)
        if len(physical) != len(unique):
            return None
        return physical

    def canonical_source_lanes(self, bar: "_VoiceBar") -> tuple:
        """Project source lives into the canonical physical column order."""
        lanes = self.bar_lanes(bar)
        left = tuple(lane for lane in lanes if lane[0] == 0)
        right = tuple(lane for lane in lanes if lane[0] == 1)
        if not self._state.canonical_order:
            left = tuple(reversed(left))
        return left + right

    def written_hand(self, bar: "_VoiceBar", lane: tuple) -> int:
        """Return the performing hand assigned to one bar-local source lane."""
        return self._state.written_hands.get((bar.index, lane), lane[0])

    def _lane_pitch_span(
        self, bar: "_VoiceBar", lane: tuple,
    ) -> Optional[Tuple[int, int]]:
        """Return the sounding register of one decided lane in this bar."""
        bar_start = bar.offset
        bar_end = bar_start + bar.tree.bar_length
        pitches = []
        for account in self._state.accounts:
            if account.retired:
                continue
            if not any(
                    self._state.pieces[piece_id].event.get("_bar") == bar.index
                    and self._state.pieces[piece_id].owner == lane
                    for piece_id in account.piece_ids):
                continue
            pitches.extend(
                kern_pitch_to_midi(pitch)
                for pitch, start, end
                in _account_sound_atoms(self._state, account)
                if start < bar_end and bar_start < end
                and kern_pitch_to_midi(pitch) >= 0
            )
        return (min(pitches), max(pitches)) if pitches else None

    def _ordered_hand_lanes(
        self, bar: "_VoiceBar", hand: int, lanes: Iterable[tuple],
    ) -> List[tuple]:
        """Order non-crossing whole voices by register, stably otherwise."""
        base = list(lanes)
        if hand == 0 and not self._state.canonical_order:
            base.reverse()
        spans = {
            lane: self._lane_pitch_span(bar, lane)
            for lane in base
        }
        outgoing: Dict[tuple, Set[tuple]] = {
            lane: set() for lane in base
        }
        indegree = {lane: 0 for lane in base}
        for left in base:
            left_span = spans[left]
            if left_span is None:
                continue
            for right in base:
                right_span = spans[right]
                if left == right or right_span is None:
                    continue
                if left_span[1] < right_span[0] \
                        and right not in outgoing[left]:
                    outgoing[left].add(right)
                    indegree[right] += 1

        rank = {lane: index for index, lane in enumerate(base)}
        ready = sorted(
            (lane for lane in base if indegree[lane] == 0),
            key=rank.__getitem__,
        )
        ordered = []
        while ready:
            lane = ready.pop(0)
            ordered.append(lane)
            for follower in sorted(outgoing[lane], key=rank.__getitem__):
                indegree[follower] -= 1
                if indegree[follower] == 0:
                    ready.append(follower)
                    ready.sort(key=rank.__getitem__)
        if len(ordered) != len(base):
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: register order contains a cycle")
        return ordered

    def _sounding_owners(self, bar: "_VoiceBar") -> Set[tuple]:
        """Return written owners with source-backed sound in one bar."""
        result: Set[tuple] = set()
        bar_start = bar.offset
        bar_end = bar_start + bar.tree.bar_length
        for account in self._state.accounts:
            if account.retired:
                continue
            atoms = _account_sound_atoms(self._state, account)
            if not any(
                    start < bar_end and bar_start < end
                    for _pitch, start, end in atoms):
                continue
            for piece_id in account.piece_ids:
                piece = self._state.pieces[piece_id]
                if piece.event.get("_bar") == bar.index:
                    result.add(piece.owner)
        return result

    def written_lanes(self, bar: "_VoiceBar") -> tuple:
        """Return the sole ordered column table consumed by the writer.

        Source lives remain immutable.  A non-base column is present only
        when it owns sound in this bar; hand-move lives are inserted on the
        receiving hand's boundary side.  One column of a silent hand is
        retained as its whole-rest staff baseline.
        """
        cached = self._written_lanes_cache.get(bar.index)
        if cached is not None:
            return cached

        if not bar.data_rows or bar.index in _material_hold_bars(self._state):
            source = self.bar_lanes(bar)
            self._written_lanes_cache[bar.index] = source
            return source
        source = self.bar_lanes(bar)
        sounding = self._sounding_owners(bar)
        result: List[tuple] = []
        for hand in (0, 1):
            hand_lanes = [
                lane for lane in source
                if lane in sounding and self.written_hand(bar, lane) == hand
            ]
            hand_lanes = self._ordered_hand_lanes(
                bar, hand, hand_lanes)
            if not hand_lanes:
                base = next(
                    (lane for lane in source
                     if lane[0] == hand and lane not in sounding),
                    next(
                        (lane for lane in self._state.initial_channels
                         if lane is not None and lane[0] == hand
                         and lane not in sounding),
                        (hand, ("baseline", bar.index)),
                    ),
                )
                hand_lanes = [base]
            result.extend(dict.fromkeys(hand_lanes))

        lanes = tuple(result)
        if len(lanes) != len(set(lanes)):
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: written column table duplicates a lane")
        self._written_lanes_cache[bar.index] = lanes
        return lanes

    def verify_written_invariants(self) -> None:
        """Check the bar-local written projection after all decisions."""
        live_accounts_by_piece: Dict[int, Set[int]] = defaultdict(set)
        for account in self._state.accounts:
            if account.retired or not _account_sound_atoms(
                    self._state, account):
                continue
            for piece_id in account.piece_ids:
                live_accounts_by_piece[piece_id].add(account.account_id)
        if any(len(accounts) != 1
               for accounts in live_accounts_by_piece.values()):
            raise _CanonicalizationPause(
                "a note piece belongs to multiple live sound accounts")

        for bar in self._state.bars:
            if not bar.data_rows:
                continue
            lanes = self.written_lanes(bar)
            hands = tuple(self.written_hand(bar, lane) for lane in lanes)
            if hands != tuple(sorted(hands)) \
                    or any(hand not in hands for hand in (0, 1)):
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: written columns lose staff order")
            missing = self._sounding_owners(bar) - set(lanes)
            if missing:
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: sounding owners have no "
                    f"written column: {sorted(missing)!r}")

    def normalized_node_span(
        self, node_id: int,
    ) -> Optional[Tuple[int, int]]:
        """Whole-bar life of one source topology segment."""
        if node_id in self._node_span_cache:
            return self._node_span_cache[node_id]
        node = self._state.topology_nodes.get(node_id)
        bars = sorted({
            self._row_bar[row]
            for row, _field in (node.positions if node is not None else ())
            if row in self._row_bar
        })
        result = (bars[0], bars[-1]) if bars else None
        self._node_span_cache[node_id] = result
        return result

    def verify_invariants(self) -> None:
        """Validate the normalized source forest before any decision."""
        for bar in self._state.bars:
            self.validate_bar(bar)

        for cancellation in self._state.seam_cancellations:
            checked = _normalized_zero_time_group(
                self._state.lines,
                cancellation.rows,
                cancellation.lanes,
            )
            if checked is None or checked[0] != cancellation:
                raise _CanonicalizationPause(
                    "a seam cancellation has no reproducible source pairing")
            containing = {
                bar.index for bar in self._state.bars
                if all(bar.start_row < row < bar.stop_row
                       for row in cancellation.rows)
            }
            if len(containing) != 1:
                raise _CanonicalizationPause(
                    "a seam cancellation crosses a metric bar boundary")

        for channel, column in self._columns.items():
            if column.parent is not None:
                parent = self._columns.get(column.parent)
                if parent is None or parent.channel[0] != channel[0]:
                    raise _CanonicalizationPause(
                        "normalized topology has an invalid split parent")
            if column.merge_parents:
                parents = [self._columns.get(parent)
                           for parent in column.merge_parents]
                if any(parent is None or parent.channel[0] != channel[0]
                       for parent in parents):
                    raise _CanonicalizationPause(
                        "normalized topology has an invalid merge predecessor")
            if column.parent is not None and column.merge_parents:
                raise _CanonicalizationPause(
                    "a normalized lane has two birth mechanisms")
            if column.survivor is not None:
                survivor = self._columns.get(column.survivor)
                if survivor is None or survivor.channel[0] != channel[0]:
                    raise _CanonicalizationPause(
                        "normalized topology has an invalid merge survivor")

        seen_positions: Dict[Tuple[int, int], int] = {}
        for node_id, node in self._state.topology_nodes.items():
            for position in node.positions:
                if position in seen_positions:
                    raise _CanonicalizationPause(
                        "a source fragment belongs to two normalized nodes")
                seen_positions[position] = node_id
        for row, channels in self._state.row_channels.items():
            nodes = self._state.row_nodes.get(row, ())
            if len(channels) != len(nodes):
                raise _CanonicalizationPause(
                    "a source row has mismatched lane and node provenance")
            for field, (channel, node_id) in enumerate(zip(channels, nodes)):
                if channel is None:
                    continue
                if node_id is None \
                        or seen_positions.get((row, field)) != node_id:
                    raise _CanonicalizationPause(
                        "a source fragment has no unique normalized node")
        for bar in self._state.bars:
            for event in bar.events:
                row = event.get("_source_row")
                channel = event.get("_source_channel")
                channels = self._state.row_channels.get(row, ())
                matches = [index for index, candidate in enumerate(channels)
                           if candidate == channel]
                if len(matches) != 1:
                    raise _CanonicalizationPause(
                        "a source event has no unique lane provenance")
                node_id = self._state.row_nodes[row][matches[0]]
                if node_id is None:
                    raise _CanonicalizationPause(
                        "a source event has no normalized topology node")

        visiting: Set[int] = set()
        visited: Set[int] = set()

        def visit(node_id: int) -> None:
            if node_id in visited:
                return
            if node_id in visiting:
                raise _CanonicalizationPause(
                    "normalized topology contains a parent cycle")
            visiting.add(node_id)
            node = self._state.topology_nodes[node_id]
            for parent_id in node.parents:
                parent = self._state.topology_nodes.get(parent_id)
                if parent is None or parent.hand != node.hand:
                    raise _CanonicalizationPause(
                        "normalized topology has an invalid parent edge")
                visit(parent_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in sorted(self._state.topology_nodes):
            visit(node_id)

        data_bars = {bar.index for bar in self._state.bars if bar.data_rows}
        for node_id, node in sorted(self._state.topology_nodes.items()):
            span = self.normalized_node_span(node_id)
            if span is None:
                continue
            occupied = {
                self._row_bar[row]
                for row, _field in node.positions
                if row in self._row_bar
            }
            expected = {
                bar_index for bar_index in data_bars
                if span[0] <= bar_index <= span[1]
            }
            if occupied != expected:
                raise _CanonicalizationPause(
                    "normalized topology gives a lane a discontinuous life")

        ancestry: Dict[Tuple[int, int], bool] = {}

        def descends(node_id: int, ancestor_id: int) -> bool:
            key = (node_id, ancestor_id)
            if key in ancestry:
                return ancestry[key]
            answer = node_id == ancestor_id or any(
                descends(parent_id, ancestor_id)
                for parent_id in self._state.topology_nodes[node_id].parents
            )
            ancestry[key] = answer
            return answer

        for row, nodes in self._state.row_nodes.items():
            present = [node_id for node_id in nodes if node_id is not None]
            ancestors = set(present)
            pending = list(present)
            while pending:
                node_id = pending.pop()
                for parent_id in self._state.topology_nodes[node_id].parents:
                    if parent_id not in ancestors:
                        ancestors.add(parent_id)
                        pending.append(parent_id)
            for ancestor_id in ancestors:
                positions = [
                    index for index, node_id in enumerate(present)
                    if descends(node_id, ancestor_id)
                ]
                if positions and positions != list(
                        range(positions[0], positions[-1] + 1)):
                    raise _CanonicalizationPause(
                        "normalized topology interleaves sibling subtrees")

        for row, (before, tokens, _after) in self._state.op_layouts.items():
            index = 0
            while index < len(tokens):
                if tokens[index] != "*v":
                    index += 1
                    continue
                stop = index + 1
                while stop < len(tokens) and tokens[stop] == "*v":
                    stop += 1
                members = [lane for lane in before[index:stop]
                           if lane is not None]
                member_hands = tuple(lane[0] for lane in members)
                if len(members) < 2 \
                        or member_hands != tuple(sorted(member_hands)):
                    raise _CanonicalizationPause(
                        "a source merge interleaves the two staff roots")
                index = stop

    def validate_bar(self, bar: "_VoiceBar") -> tuple:
        """Check invariants visible in one normalized source bar."""
        lanes = self.bar_lanes(bar)
        if len(lanes) != len(set(lanes)):
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: normalized topology duplicates a lane")
        hands = tuple(lane[0] for lane in lanes)
        if any(hand not in {0, 1} for hand in hands) \
                or hands != tuple(sorted(hands)):
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: normalized topology reorders hands")
        if any(hand not in hands for hand in (0, 1)):
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: normalized topology loses a staff root")
        lane_set = set(lanes)
        if any(
                event.get("_source_channel") not in lane_set
                for event in bar.events):
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: a source event has no normalized lane")
        return lanes

    def source_bar_boundaries(
        self, bar: "_VoiceBar",
    ) -> Tuple[tuple, tuple]:
        """Return the source layout around one held bar."""
        incoming = self._state.initial_channels
        internal = []
        for row, layout in sorted(self._state.op_layouts.items()):
            if row < bar.start_row or (row == bar.start_row and not bar.label):
                incoming = layout[2]
            elif bar.start_row < row < bar.stop_row:
                internal.append(layout)
        return incoming, internal[-1][2] if internal else incoming




def _source_topology(state: "VoiceState") -> SourceTopology:
    """Build the boundary-normalized source view once per parsed state."""
    if state.source_topology is None:
        state.source_topology = SourceTopology(state)
    return state.source_topology


def _record(
    state: VoiceState,
    decision: str,
    reason: str,
    *,
    source=None,
    target=None,
    span=None,
    execution: str = "applied",
    evidence_span=None,
    effect_span=None,
) -> None:
    state.outcomes.append(MergeOutcome(
        decision=decision,
        reason=reason,
        source=source,
        target=target,
        span=span,
        execution=execution,
        evidence_span=evidence_span,
        effect_span=effect_span,
    ))


def _is_spine_operation(line: str) -> bool:
    return line.startswith("*") and any(
        token in {"*^", "*v", "*x", "*+", "*-"}
        for token in line.split("\t")
    )


def _scan_written_columns(
    lines: List[str],
):
    """Track writer lanes separately from explicit topology-node lives.

    Both children of a split and every join are fresh topology nodes.  Writer
    lane reuse is only an address choice and never supplies merge eligibility.
    """
    columns: Optional[List[Optional[tuple]]] = None
    nodes: Optional[List[Optional[int]]] = None
    row_channels: Dict[int, list] = {}
    row_nodes: Dict[int, list] = {}
    birth_base: Dict[tuple, tuple] = {}
    clock_flow: Dict[int, Dict[tuple, list]] = {}
    pending_flow: Dict[tuple, Set[tuple]] = {}
    written: Dict[tuple, WrittenColumn] = {}
    topology_nodes: Dict[int, TopologyNode] = {}
    op_layouts: Dict[int, Tuple[tuple, tuple, tuple]] = {}
    seam_cancellations: List[_SeamCancellation] = []
    carried_expansions: Dict[int, str] = {}
    zero_time_groups = _zero_time_operation_groups(lines)
    cancelled_rows: Set[int] = set()
    initial_channels: tuple = ()
    next_id = {0: 1, 1: 1}
    next_node_id = 0

    def new_node(
        hand: int, *, parents: tuple = (), birth_row: Optional[int] = None,
    ) -> int:
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        topology_nodes[node_id] = TopologyNode(
            node_id=node_id, hand=hand, parents=parents,
            birth_row=birth_row)
        return node_id

    for row, line in enumerate(lines):
        if not line or line.startswith("!"):
            continue
        fields = line.split("\t")
        if line.startswith("**"):
            columns = []
            nodes = []
            hand = 0
            for token in fields:
                if token == "**kern":
                    if hand > 1:
                        raise ValueError("more than two kern staves")
                    channel = (hand, 0)
                    columns.append(channel)
                    nodes.append(new_node(hand))
                    written[channel] = WrittenColumn(channel=channel)
                    hand += 1
                else:
                    columns.append(None)
                    nodes.append(None)
            initial_channels = tuple(columns)
            continue
        if columns is None or nodes is None:
            continue
        if row in cancelled_rows:
            continue
        if row in zero_time_groups:
            normalized = _normalized_zero_time_group(
                lines, zero_time_groups[row], tuple(columns))
            if normalized is not None:
                cancellation, expansions = normalized
                seam_cancellations.append(cancellation)
                carried_expansions.update(expansions)
                cancelled_rows.update(cancellation.rows)
                continue
        if line.startswith("*") and any(
                token in {"*^", "*v", "*x", "*+", "*-"}
                for token in fields):
            before = tuple(columns)
            new_columns: List[Optional[tuple]] = []
            new_nodes: List[Optional[int]] = []
            next_flow: Dict[tuple, Set[tuple]] = {}
            merge_inputs: Dict[tuple, tuple] = {}
            index = 0
            while index < len(fields):
                token = fields[index]
                column = columns[index] if index < len(columns) else None
                node = nodes[index] if index < len(nodes) else None
                if token == "*^":
                    if column is None:
                        new_columns.extend([None, None])
                        new_nodes.extend([None, None])
                    else:
                        hand = column[0]
                        child = (hand, next_id[hand])
                        next_id[hand] += 1
                        birth_base[child] = birth_base.get(column, column)
                        written[child] = WrittenColumn(
                            channel=child, parent=column, birth_row=row)
                        next_flow[child] = {column}
                        new_columns.extend([column, child])
                        if node is None:
                            new_nodes.extend([None, None])
                        else:
                            topology_nodes[node].death_row = row
                            left_node = new_node(
                                hand, parents=(node,), birth_row=row)
                            right_node = new_node(
                                hand, parents=(node,), birth_row=row)
                            new_nodes.extend([left_node, right_node])
                    index += 1
                    continue
                if token == "*v":
                    stop = index + 1
                    while stop < len(fields) and fields[stop] == "*v":
                        stop += 1
                    members = [columns[position]
                               for position in range(index, stop)
                               if position < len(columns)
                               and columns[position] is not None]
                    member_nodes = [nodes[position]
                                    for position in range(index, stop)
                                    if position < len(nodes)
                                    and nodes[position] is not None]
                    hands = {member[0] for member in members}
                    survivor = (
                        next(
                            (member for member in members
                             if any(written[other].parent == member
                                    for other in members
                                    if other != member)),
                            None,
                        )
                        if len(hands) == 1 else None
                    )
                    if survivor is None and members:
                        hand = next(iter(hands)) if len(hands) == 1 \
                            else members[0][0]
                        survivor = (hand, next_id[hand])
                        next_id[hand] += 1
                        birth_base[survivor] = birth_base.get(
                            members[0], members[0])
                        written[survivor] = WrittenColumn(
                            channel=survivor, birth_row=row,
                            merge_parents=tuple(
                                member for member in members
                                if member[0] == hand))
                    if survivor is not None:
                        merge_inputs[survivor] = tuple(members)
                    new_columns.append(survivor)
                    if survivor is None or not member_nodes:
                        new_nodes.append(None)
                    else:
                        for member_node in member_nodes:
                            topology_nodes[member_node].death_row = row
                        new_nodes.append(new_node(
                            survivor[0], parents=tuple(member_nodes),
                            birth_row=row))
                    if survivor is not None:
                        sources: Set[tuple] = set()
                        for member in members:
                            sources.update(pending_flow.get(member, {member}))
                        next_flow[survivor] = sources
                        for member in members:
                            if member != survivor:
                                written[member].death_row = row
                                written[member].survivor = (
                                    survivor
                                    if member[0] == survivor[0] else None)
                    index = stop
                    continue
                if token == "*x" and index + 1 < len(fields):
                    right = columns[index + 1] \
                        if index + 1 < len(columns) else None
                    new_columns.extend([right, column])
                    right_node = nodes[index + 1] \
                        if index + 1 < len(nodes) else None
                    new_nodes.extend([right_node, node])
                    index += 2
                    continue
                if token == "*+":
                    new_columns.extend([column, None])
                    new_nodes.extend([node, None])
                    index += 1
                    continue
                if token == "*-":
                    if column is not None:
                        written[column].death_row = row
                    if node is not None:
                        topology_nodes[node].death_row = row
                    index += 1
                    continue
                new_columns.append(column)
                new_nodes.append(node)
                index += 1
            active_positions = [
                position for position, channel in enumerate(new_columns)
                if channel is not None
            ]
            if len(active_positions) == 2:
                # A piano topology that has returned to its two physical
                # staves starts a fresh bar-local address basis.  Source *v
                # trees may cross the original staff line; carrying their
                # leftmost survivor forward would turn both remaining spines
                # into one hand even though their physical order is again
                # staff 2 / staff 1.
                for hand, position in enumerate(active_positions):
                    desired = (hand, 0)
                    current = new_columns[position]
                    if current == desired:
                        continue
                    sources = next_flow.pop(current, {current})
                    next_flow.setdefault(desired, set()).update(sources)
                    if current is not None and current in written:
                        written[current].death_row = row
                        written[current].survivor = (
                            desired if current[0] == hand else None)
                        for member in merge_inputs.get(current, ()):
                            if member == current:
                                continue
                            written[member].survivor = (
                                desired if member[0] == hand else None)
                    root = written.setdefault(
                        desired, WrittenColumn(channel=desired))
                    root.death_row = None
                    root.survivor = None
                    birth_base[desired] = desired
                    new_columns[position] = desired
            for position, (channel, node_id) in enumerate(
                    zip(new_columns, new_nodes)):
                if channel is None or node_id is None:
                    continue
                node = topology_nodes[node_id]
                if node.birth_row != row:
                    continue
                node.hand = channel[0]
                node.parents = tuple(
                    parent_id for parent_id in node.parents
                    if topology_nodes[parent_id].hand == channel[0])
            columns = new_columns
            nodes = new_nodes
            op_layouts[row] = (before, tuple(fields), tuple(columns))
            for column in columns:
                if column is None or column in next_flow:
                    continue
                if column in pending_flow:
                    next_flow[column] = set(pending_flow[column])
            pending_flow = next_flow
            continue
        if line.startswith(("*", "=")):
            continue
        row_channels[row] = list(columns)
        row_nodes[row] = list(nodes)
        for field_index, node in enumerate(nodes):
            if node is not None:
                topology_nodes[node].positions.append((row, field_index))
        if pending_flow:
            flow = {
                channel: sorted(sources)
                for channel, sources in pending_flow.items()
                if sources != {channel}
            }
            if flow:
                clock_flow[row] = flow
        pending_flow = {}
    return (row_channels, row_nodes, topology_nodes, birth_base,
            clock_flow, written, op_layouts, initial_channels,
            tuple(seam_cancellations), carried_expansions)


def _parse_voice_state(
    kern_content: str,
) -> VoiceState:
    """Read source topology once, then parse every written channel as itself."""
    lines = kern_content.splitlines()
    (row_channels, row_nodes, topology_nodes, birth_base,
     clock_flow, written, op_layouts, initial_channels, seam_cancellations,
     carried_expansions) = _scan_written_columns(lines)
    barlines = [i for i, line in enumerate(lines) if line.startswith("=")]
    if len(barlines) < 2:
        raise ValueError("voice canonicalization requires a complete bar")

    first, last = barlines[0], barlines[-1]
    # A pickup is a complete bar (its head is already rest-padded); it
    # enters the engine like any bar so its rests and voices canonicalize.
    pickup_rows = [row for row in range(first)
                   if lines[row]
                   and not lines[row].startswith(("!", "*", "="))]
    header_stop = pickup_rows[0] if pickup_rows else first
    state = VoiceState(
        original=kern_content,
        lines=lines,
        header=list(lines[:header_stop]),
        trailer=list(lines[last + 1:]),
        bars=[],
        columns=written,
        row_channels=row_channels,
        topology_nodes=topology_nodes,
        row_nodes=row_nodes,
        op_layouts=op_layouts,
        initial_channels=initial_channels,
        seam_cancellations=seam_cancellations,
    )

    meter: Optional[Tuple[int, int]] = None
    for line in lines[:header_stop]:
        for token in line.split("\t"):
            match = re.fullmatch(r"\*M(\d+)/(\d+)", token)
            if match:
                meter = (int(match.group(1)), int(match.group(2)))

    segments: List[Tuple[range, str, str, int, int]] = []
    if pickup_rows:
        segments.append((range(header_stop, first), "",
                         lines[first].split("\t")[0], header_stop - 1, first))
    for start, stop in zip(barlines, barlines[1:]):
        segments.append((range(start + 1, stop),
                         lines[start].split("\t")[0],
                         lines[stop].split("\t")[0], start, stop))

    offset = Fraction(0)
    event_uid = 0
    for bar_index, (segment, label, closing_label, start_row, stop_row) \
            in enumerate(segments):
        data_rows = [row for row in segment
                     if lines[row]
                     and not lines[row].startswith(("!", "*", "="))]
        bar_ordinal = (
            sum(bool(previous.data_rows) for previous in state.bars)
            if data_rows else None
        )
        for row in segment:
            if not lines[row].startswith("*"):
                continue
            for token in lines[row].split("\t"):
                match = re.fullmatch(r"\*M(\d+)/(\d+)", token)
                if match:
                    meter = (int(match.group(1)), int(match.group(2)))
        if meter is None:
            raise TokenizerOOVError(
                f"{label or 'pickup'}: data before meter",
                bar_index=bar_index,
                bar_label=label or "pickup",
                bar_ordinal=bar_ordinal,
            )
        try:
            tree = get_metric_tree(*meter)
        except TokenizerOOVError as error:
            raise TokenizerOOVError(
                str(error),
                bar_index=bar_index,
                bar_label=label or "pickup",
                bar_ordinal=bar_ordinal,
            ) from error
        channels = [row_channels.get(row, [(0, 0), (1, 0)])
                    for row in data_rows]
        identities = {
            channel: birth_base.get(channel, channel)
            for row_map in channels
            for channel in row_map
            if channel is not None
        }
        parse_aux: dict = {}
        try:
            events = _parse_bar_events(
                [lines[row] for row in data_rows],
                row_channels=channels,
                aux=parse_aux,
                fold_map=identities,
                row_rebinds=[clock_flow.get(row) for row in data_rows],
            )
        except (IncompleteTupletError, TokenizerOOVError) as error:
            issue = _voice_issue_from_error(error)
            assert issue is not None
            state.issues.append(replace(
                issue, bar_index=bar_index, bar_label=label or "pickup",
                bar_ordinal=bar_ordinal))
            state.paused_bars.add(bar_index)
            _record(
                state, "refuse", issue.family,
                source=None, span=label or "pickup", execution="paused",
            )
            events = []
            parse_aux = {}
        if not data_rows:
            state.paused_bars.add(bar_index)
            _record(
                state, "refuse", "zero-time-barline-seam",
                source=None, span=label or "pickup",
                execution="paused",
            )
        overfull = [event for event in events
                    if event["onset"] + event["dur"] > tree.bar_length]
        if overfull:
            # An engraved note ringing past its barline is out of the
            # canonical vocabulary; the bar ships as written so the strict
            # reader reports it, never repaired into a tie — overfull bars
            # stay visibly OOV.
            state.paused_bars.add(bar_index)
            _record(
                state, "refuse", "overfull-bar",
                source=sorted({event.get("chain", 0) for event in overfull}),
                span=label or "pickup",
                execution="paused",
            )
            state.issues.append(VoiceIssue(
                family="metric_timeline",
                message=(f"{label or 'pickup'}: event exceeds "
                         f"bar length {tree.bar_length}"),
                bar_index=bar_index,
                bar_label=label or "pickup",
                bar_ordinal=bar_ordinal,
            ))
        elif events and max(
                event["onset"] + event["dur"] for event in events
        ) < tree.bar_length:
            # A short metric timeline is missing source data, not silence the
            # voice writer may complete with rests.
            state.paused_bars.add(bar_index)
            _record(
                state, "refuse", "underfull-bar",
                source=None, span=label or "pickup",
                execution="paused",
            )
            max_end = max(event["onset"] + event["dur"] for event in events)
            state.issues.append(VoiceIssue(
                family="metric_timeline",
                message=(f"{label or 'pickup'}: score ends at {max_end}, "
                         f"bar ends at {tree.bar_length}"),
                bar_index=bar_index,
                bar_label=label or "pickup",
                bar_ordinal=bar_ordinal,
            ))

        row_onsets = {
            data_rows[local_row]: Fraction(onset)
            for local_row, onset in parse_aux.get("row_onsets", {}).items()
            if local_row < len(data_rows)
        }
        for event in events:
            local_row = event.get("_source_row")
            if local_row is not None and local_row < len(data_rows):
                absolute_row = data_rows[local_row]
                event["_source_row"] = absolute_row
            source = (event["hand"], event.get("chain", 0))
            event["_source_channel"] = source
            event["_bar"] = bar_index
            event["_uid"] = event_uid
            event_uid += 1

        carried: List[_CarriedRow] = []
        for row in segment:
            text = carried_expansions.get(row, lines[row])
            if not text or row in data_rows or text.startswith("="):
                continue
            if _is_spine_operation(text):
                continue
            following = next((data for data in data_rows if data > row), None)
            preceding = next((data for data in reversed(data_rows)
                              if data < row), None)
            anchor = following if following is not None else preceding
            if anchor is None:
                onset = Fraction(0)
                anchor_channels: tuple = ()
            else:
                onset = row_onsets.get(anchor, Fraction(0))
                anchor_channels = tuple(row_channels.get(anchor, ()))
            carried.append(_CarriedRow(
                row=row, onset=onset, text=text,
                channels=anchor_channels,
                after_data=following is None and preceding is not None,
            ))

        state.bars.append(_VoiceBar(
            index=bar_index,
            label=label,
            closing_label=closing_label,
            start_row=start_row,
            stop_row=stop_row,
            tree=tree,
            offset=offset,
            events=events,
            data_rows=data_rows,
            carried=carried,
        ))
        if data_rows:
            offset += tree.bar_length
    return state


def _build_accounts(state: VoiceState) -> None:
    pieces: List[_PitchPiece] = []
    for bar in state.bars:
        for event in bar.events:
            source = event["_source_channel"]
            start = bar.offset + Fraction(event["onset"])
            end = start + Fraction(event["dur"])
            source_row = event.get("_source_row")
            if event.get("is_tup"):
                piece = _PitchPiece(
                    piece_id=len(pieces), event=event,
                    pitch=None, hand=event["hand"], start=start, end=end,
                    tie_in=False, tie_out=False, source=source,
                    source_row=source_row, owner=source,
                )
                pieces.append(piece)
                event["_piece_ids"] = [piece.piece_id]
                continue
            if event["pitches"] == ["r"]:
                event["_piece_ids"] = []
                continue
            ids = []
            for member, pitch in enumerate(event["pitches"]):
                piece = _PitchPiece(
                    piece_id=len(pieces), event=event,
                    pitch=pitch, hand=event["hand"], start=start, end=end,
                    tie_in=bool(event["tie_ins"][member]),
                    tie_out=bool(event["tie_outs"][member]),
                    source=source, source_row=source_row, owner=source,
                )
                pieces.append(piece)
                ids.append(piece.piece_id)
            event["_piece_ids"] = ids

    accounts: List[AttackReleaseAccount] = []
    open_accounts: Dict[Tuple[int, Optional[str]], List[int]] = defaultdict(list)

    for piece in sorted(
            pieces,
            key=lambda item: (item.start, item.hand, item.pitch or "",
                              0 if item.tie_in else 1, item.piece_id)):
        key = (piece.hand, piece.pitch)
        current_bar = piece.event["_bar"]
        current_onset = Fraction(piece.event["onset"])
        open_accounts[key] = [
            account_id for account_id in open_accounts[key]
            if (
                pieces[accounts[account_id].piece_ids[-1]].event["_bar"]
                == current_bar
                or (
                    pieces[
                        accounts[account_id].piece_ids[-1]
                    ].event["_bar"] == current_bar - 1
                    and current_onset == 0
                )
            )
        ]
        candidates = [
            account_id for account_id in open_accounts[key]
            if pieces[accounts[account_id].piece_ids[-1]].tie_out
        ] if piece.tie_in else []
        if candidates:
            # One landing closes one account.  Same-lane pending sounds
            # outrank the rest (kern's lineage pairing), then the latest
            # departure wins.
            best = max(
                candidates,
                key=lambda account_id: (
                    pieces[
                        accounts[account_id].piece_ids[-1]
                    ].source == piece.source,
                    accounts[account_id].start,
                    -account_id,
                ),
            )
            chosen = [best]
            for account_id in chosen:
                account = accounts[account_id]
                account.piece_ids.append(piece.piece_id)
                account.end = max(account.end, piece.end)
            if not piece.tie_out:
                open_accounts[key] = [
                    account_id for account_id in open_accounts[key]
                    if account_id not in chosen
                ]
            continue
        account = AttackReleaseAccount(
            account_id=len(accounts),
            hand=piece.hand,
            pitch=piece.pitch,
            start=piece.start,
            end=piece.end,
            source=piece.source,
            attacked=not piece.tie_in,
            piece_ids=[piece.piece_id],
            owner=piece.source,
        )
        accounts.append(account)
        if piece.tie_out:
            open_accounts[key].append(account.account_id)

    state.pieces = pieces
    state.accounts = accounts


def _account_sound_atoms(
    state: VoiceState,
    account: AttackReleaseAccount,
    *,
    source: bool = False,
) -> Set[Tuple[str, Fraction, Fraction]]:
    """Return the keyed spans that remain after settled owner decisions."""
    if account.retired and not source:
        return set()
    if account.pitch is not None:
        return {(account.pitch, account.start, account.end)}

    atoms: Set[Tuple[str, Fraction, Fraction]] = set()
    for piece_id in account.piece_ids:
        piece = state.pieces[piece_id]
        slots = piece.event.get("slots") or []
        if not slots:
            continue
        grain = (piece.end - piece.start) / len(slots)
        active: Dict[str, Fraction] = {}
        for index, slot in enumerate(slots):
            onset = piece.start + index * grain
            continuations = {
                pitch for pitch in slot.get("cont", ()) if pitch != "r"
            }
            attacks = slot.get("attack", ())
            tie_ins = slot.get("attack_ti", (False,) * len(attacks))
            tied = {
                pitch for pitch, tie_in in zip(attacks, tie_ins)
                if pitch != "r" and tie_in
            }
            fresh = {
                pitch for pitch, tie_in in zip(attacks, tie_ins)
                if pitch != "r" and not tie_in
            }
            sounding = continuations | tied | fresh
            for pitch in list(active):
                if pitch not in sounding or pitch in fresh:
                    atoms.add((pitch, active.pop(pitch), onset))
            for pitch in sounding:
                active.setdefault(pitch, onset)
        for pitch, onset in active.items():
            atoms.add((pitch, onset, piece.end))
    return (
        atoms
        if source else
        atoms - state.suppressed_tuple_atoms.get(account.account_id, set())
    )


def _account_attack_atoms(
    state: VoiceState, account: AttackReleaseAccount, *, source: bool = True,
) -> Set[Tuple[str, Fraction, Fraction]]:
    """Return immutable fresh attacks and their complete sounding spans."""
    if account.pitch is not None:
        if not account.attacked:
            return set()
        return {(account.pitch, account.start, account.end)}

    strikes: Set[Tuple[str, Fraction]] = set()
    for piece_id in account.piece_ids:
        piece = state.pieces[piece_id]
        slots = piece.event.get("slots") or []
        if not slots:
            continue
        grain = (piece.end - piece.start) / len(slots)
        for index, slot in enumerate(slots):
            onset = piece.start + index * grain
            attacks = slot.get("attack", ())
            tie_ins = slot.get("attack_ti", (False,) * len(attacks))
            strikes.update(
                (pitch, onset)
                for pitch, tie_in in zip(attacks, tie_ins)
                if pitch != "r" and not tie_in
            )

    result = set()
    for pitch, onset in strikes:
        matching = [
            end for atom_pitch, start, end
            in _account_sound_atoms(state, account, source=source)
            if atom_pitch == pitch and start == onset
        ]
        if matching:
            result.add((pitch, onset, max(matching)))
    return result


def _bar_at_time(
    state: VoiceState, instant: Fraction,
) -> Optional[_VoiceBar]:
    return next(
        (bar for bar in state.bars
         if bar.offset <= instant < bar.offset + bar.tree.bar_length),
        None,
    )


def _decide_voices(state: VoiceState) -> None:
    """Settle duplicate physical attacks without merging source voices."""
    topology = _source_topology(state)
    topology.verify_invariants()
    for bar in state.bars:
        if bar.index in state.paused_bars:
            continue
        try:
            topology.validate_bar(bar)
        except _CanonicalizationPause as error:
            state.paused_bars.add(bar.index)
            _add_voice_issue(
                state, _issue_family_for_message(str(error)),
                f"{bar.label or 'pickup'}: {error}", bar.index)
            _record(
                state, "refuse", "writer-bar-hold",
                source=None, target=None,
                span=bar.label or "pickup", execution="paused",
                evidence_span=str(error),
            )

    frozen_accounts = {
        account.account_id
        for account in state.accounts
        if any(
            state.pieces[piece_id].event["_bar"] in state.paused_bars
            for piece_id in account.piece_ids)
    }
    source_accounts = [
        account for account in state.accounts
        if not account.retired
        and account.account_id not in frozen_accounts
    ]

    strike_groups: Dict[
        Tuple[int, str, Fraction],
        List[Tuple[AttackReleaseAccount, Fraction]],
    ] = defaultdict(list)
    for account in source_accounts:
        for pitch, onset, end in _account_attack_atoms(state, account):
            strike_groups[(account.hand, pitch, onset)].append(
                (account, end))

    duplicate_groups = {
        key: list({item[0].account_id: item
                   for item in members}.values())
        for key, members in strike_groups.items()
        if len({item[0].account_id for item in members}) > 1
    }

    # A simultaneous duplicate is one physical key attack.  Keep the account
    # with the longest sounding extent; equal extents prefer the smallest
    # normalized logical address.  This does not by itself move a voice.
    for (hand, pitch, onset), members in sorted(
            duplicate_groups.items(), key=lambda item: item[0]):
        evidence_bar = _bar_at_time(state, onset)
        sources = tuple(dict.fromkeys(
            account.source for account, _end in members))
        source_rank: Dict[tuple, int] = {}
        if evidence_bar is not None:
            ordered = topology.logical_order(evidence_bar, sources)
            if ordered is not None:
                source_rank = {
                    source: rank
                    for rank, source in enumerate(ordered)
                }
        keeper, keeper_end = max(
            members,
            key=lambda item: (
                item[1],
                -source_rank.get(item[0].source, len(source_rank)),
                any(state.pieces[piece_id].tie_out
                    for piece_id in item[0].piece_ids),
                -item[0].account_id,
            ),
        )
        changed = []
        for account, end in members:
            if account.account_id == keeper.account_id:
                continue
            if account.pitch is not None:
                if not account.retired:
                    account.retired = True
                    changed.append((account.source, account.start, account.end))
            else:
                atom = (pitch, onset, end)
                suppressed = state.suppressed_tuple_atoms.setdefault(
                    account.account_id, set())
                if atom not in suppressed:
                    suppressed.add(atom)
                    changed.append((account.source, onset, end))
                if not _account_sound_atoms(state, account):
                    account.retired = True
        if changed:
            _record(
                state, "absorb", "same-pitch-attack",
                source=tuple(sorted({source for source, _start, _end
                                     in changed})),
                target=keeper.source,
                span=(onset, keeper_end),
                evidence_span=(hand, pitch, onset),
                effect_span=tuple(sorted(changed, key=repr)),
            )
            state.decision_mutated = True

def _event_cell(event: dict) -> str:
    recip = GRID_TO_RECIP.get(event["dur"])
    if recip is None:
        raise ValueError(
            f"duration {event['dur']} has no canonical kern reciprocal")
    parts = []
    for pitch, tie_in, tie_out in zip(
            event["pitches"], event["tie_ins"], event["tie_outs"]):
        if pitch == "r":
            parts.append(f"{recip}r")
        elif tie_in and tie_out:
            parts.append(f"{recip}{pitch}_")
        elif tie_out:
            parts.append(f"[{recip}{pitch}")
        elif tie_in:
            parts.append(f"{recip}{pitch}]")
        else:
            parts.append(f"{recip}{pitch}")
    return " ".join(parts)


def _member_recip(n: int, total_grid: int) -> str:
    reciprocal = Fraction(128 * n, total_grid)
    if reciprocal.denominator == 1:
        return str(reciprocal.numerator)
    return f"{reciprocal.numerator}%{reciprocal.denominator}"


def _tuplet_rows(event: dict) -> List[Tuple[Fraction, str]]:
    slots = event.get("slots") or []
    if not slots:
        return []
    n = event["tup_n"]
    reciprocal = _member_recip(n, event["dur"])
    grain = Fraction(event["dur"], len(slots))
    onset = Fraction(event["onset"])
    rows = []
    # Plan rest ink first: a rest ringing through several slots prints
    # one merged token (reconstruct-parity; silence draws no arc).
    rest_tokens: Dict[int, str] = {}
    index = 0
    while index < len(slots):
        if "r" in slots[index].get("attack", []):
            span = 1
            while (index + span < len(slots)
                   and "r" in slots[index + span].get("cont", [])):
                span += 1
            rest_tokens[index] = \
                f"{_member_recip(n, event['dur'] * span)}r"
            index += span
        else:
            index += 1

    for index, slot in enumerate(slots):
        note_parts = []
        departs = bool(slot.get("departs"))
        continuations = slot.get("cont", [])
        continuation_outs = slot.get(
            "cont_to", [departs] * len(continuations))
        for pitch, tie_out in zip(continuations, continuation_outs):
            if pitch != "r":
                note_parts.append(
                    f"{reciprocal}{pitch}{'_' if tie_out else ']'}")
        attacks = slot.get("attack", [])
        tie_ins = slot.get("attack_ti", [False] * len(attacks))
        tie_outs = slot.get("attack_to", [departs] * len(attacks))
        for pitch, tie_in, tie_out in zip(attacks, tie_ins, tie_outs):
            if pitch == "r":
                continue
            if tie_in and tie_out:
                note_parts.append(f"{reciprocal}{pitch}_")
            elif tie_out:
                note_parts.append(f"[{reciprocal}{pitch}")
            elif tie_in:
                note_parts.append(f"{reciprocal}{pitch}]")
            else:
                note_parts.append(f"{reciprocal}{pitch}")
        # A slot that sounds needs no rest ink beside it.
        parts = note_parts
        if not parts and index in rest_tokens:
            parts = [rest_tokens[index]]
        if parts:
            rows.append((onset + index * grain, " ".join(parts)))
    return rows


def _fill_channel_rests(
    events: List[dict], tree: MetricTree, widths: Dict[int, int],
) -> List[dict]:
    out = list(events)
    for hand in (0, 1):
        for role in range(1, widths[hand] + 1):
            spans = []
            for event in events:
                if event["hand"] != hand:
                    continue
                event_role = event.get("voice") or 1
                if event_role != role:
                    continue
                spans.append((Fraction(event["onset"]),
                              Fraction(event["onset"] + event["dur"])))
            merged = []
            for start, end in sorted(spans):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            gaps = []
            cursor = Fraction(0)
            for start, end in merged:
                if start > cursor:
                    gaps.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < tree.bar_length:
                gaps.append((cursor, Fraction(tree.bar_length)))
            for start, end in gaps:
                if start.denominator != 1 or end.denominator != 1:
                    raise _CanonicalizationPause(
                        "tuplet boundary cannot carry a voice rest")
                position = int(start)
                for reciprocal in spell(position, int(end - start), tree):
                    duration = RECIP_TO_GRID[reciprocal]
                    rest = {
                        "onset": position,
                        "dur": duration,
                        "hand": hand,
                        "chain": role - 1,
                        "pitches": ["r"],
                        "tie_ins": [False],
                        "tie_outs": [False],
                    }
                    if widths[hand] > 1:
                        rest["voice"] = role
                    out.append(rest)
                    position += duration
    return _merge_rest_spans(out, tree)


def _flatten_slot_flags(group: dict) -> None:
    """Rewrite slot-level departs into per-pitch tie marks.

    A ringing pitch prints identically as an attack entry with tie_in
    True, so cont notes move into the attack arrays; cont rests stay,
    because the serializer's merged-rest lookahead reads them there.  Existing
    attack_to flags are already pitch-specific; departs applies only to the
    continuation members promoted here.
    """
    for slot in group["slots"]:
        departs = bool(slot.get("departs"))
        keep_cont, keep_ti, keep_to = [], [], []
        cont_ti = slot.get("cont_ti", [True] * len(slot["cont"]))
        cont_to = slot.get("cont_to", [departs] * len(slot["cont"]))
        for pitch, tie_in, tie_out in zip(
                slot["cont"], cont_ti, cont_to):
            if pitch == "r":
                keep_cont.append(pitch)
                keep_ti.append(tie_in)
                keep_to.append(tie_out)
            else:
                slot["attack"].append(pitch)
                slot["attack_ti"].append(True)
                slot["attack_to"].append(tie_out)
        slot["cont"], slot["cont_ti"], slot["cont_to"] = (
            keep_cont, keep_ti, keep_to)
        slot["departs"] = False


def _merge_tuple_slots(target: dict, source: dict) -> None:
    """Union two decided tuple voices without duplicating one piano key."""
    attacks: Dict[Tuple[str, bool], bool] = {}
    order: List[Tuple[str, bool]] = []
    has_rest = any(
        pitch == "r"
        for slot in (target, source)
        for pitch in slot.get("cont", []) + slot.get("attack", [])
    )
    for slot in (target, source):
        pitches = slot.get("attack", [])
        tie_ins = slot.get("attack_ti", [False] * len(pitches))
        tie_outs = slot.get("attack_to", [False] * len(pitches))
        for pitch, tie_in, tie_out in zip(pitches, tie_ins, tie_outs):
            if pitch == "r":
                continue
            key = (pitch, bool(tie_in))
            if key not in attacks:
                attacks[key] = bool(tie_out)
                order.append(key)
                continue
            # Equal tie-in roles refer to the same physical occurrence.
            # A pending continuation and a fresh strike remain two entries.
            attacks[key] = attacks[key] or bool(tie_out)

    order.sort(key=lambda item: (kern_pitch_to_midi(item[0]), not item[1]))
    # A rest has no identity inside a sounding slot.  Keep exactly one only
    # when both decided voices are silent at this instant.
    silent = not order
    target["attack"] = (
        ["r"] if silent and has_rest
        else [pitch for pitch, _tie_in in order]
    )
    target["attack_ti"] = (
        [False] if target["attack"] == ["r"]
        else [tie_in for _pitch, tie_in in order]
    )
    target["attack_to"] = (
        [False] if target["attack"] == ["r"]
        else [attacks[key] for key in order]
    )
    target["cont"] = []
    target["cont_ti"] = []
    target["cont_to"] = []


def _embed_note_in_tuple(group: dict, event: dict) -> None:
    """Project a regular note onto an enclosing tuplet's slot grid."""
    slots = group.get("slots") or []
    if not slots:
        raise _CanonicalizationPause("a tuplet group has no member slots")
    group_start = Fraction(group["onset"])
    grain = Fraction(group["dur"], len(slots))
    start = (Fraction(event["onset"]) - group_start) / grain
    end = (
        Fraction(event["onset"]) + Fraction(event["dur"]) - group_start
    ) / grain
    if start.denominator != 1 or end.denominator != 1:
        raise _CanonicalizationPause(
            "a note boundary does not align with its tuplet host")
    first, stop = int(start), int(end)
    if first < 0 or stop > len(slots) or first >= stop:
        raise _CanonicalizationPause(
            "a note falls outside its selected tuplet host")

    _flatten_slot_flags(group)
    pitches = list(event["pitches"])
    tie_ins = list(event.get("tie_ins", [False] * len(pitches)))
    tie_outs = list(event.get("tie_outs", [False] * len(pitches)))
    for index in range(first, stop):
        source = {
            "attack": pitches,
            "attack_ti": [
                tie_in if index == first else True
                for tie_in in tie_ins
            ],
            "attack_to": [
                tie_out if index == stop - 1 else True
                for tie_out in tie_outs
            ],
            "cont": [],
            "cont_ti": [],
            "cont_to": [],
            "departs": False,
        }
        _merge_tuple_slots(slots[index], source)


def _slice_channel_overlaps(events: List[dict]) -> List[dict]:
    """Slice already-decided chord voices at every attack or release."""
    by_channel: Dict[tuple, List[dict]] = defaultdict(list)
    for event in events:
        by_channel[(event["hand"], event.get("voice") or 1)].append(event)
    out = []
    for channel_events in by_channel.values():
        sounding = [
            (Fraction(e["onset"]), Fraction(e["onset"]) + Fraction(e["dur"]))
            for e in channel_events
            if e.get("is_tup") or (e.get("pitches") or ["r"]) != ["r"]
        ]
        kept = []
        for event in channel_events:
            start = Fraction(event["onset"])
            end = start + Fraction(event["dur"])
            if (event.get("pitches") or []) == ["r"] and not event.get(
                    "is_tup") and any(s < end and start < e
                                      for s, e in sounding):
                # A rest absorbed into a channel that is sounding there is
                # redundant ink; true silences are rebuilt by
                # _fill_channel_rests.
                continue
            kept.append(event)
        # Rests have no say over where notes are cut, so boundaries come
        # from the surviving events only.
        boundaries = sorted({
            Fraction(point)
            for event in kept
            for point in (event["onset"], event["onset"] + event["dur"])
        })
        pieces = []
        for event in kept:
            start = Fraction(event["onset"])
            end = start + Fraction(event["dur"])
            cuts = [point for point in boundaries if start < point < end]
            if not cuts:
                pieces.append(event)
                continue
            if event.get("is_tup"):
                # The group's own slot grid remains authoritative.  Other
                # voices are projected onto it after their regular events
                # have been sliced at the shared boundaries.
                pieces.append(event)
                continue
            points = [start, *cuts, end]
            for index, (left, right) in enumerate(zip(points, points[1:])):
                piece = dict(
                    event,
                    onset=left,
                    dur=right - left,
                    pitches=list(event["pitches"]),
                    tie_ins=(list(event["tie_ins"])
                             if index == 0
                             else [True] * len(event["pitches"])),
                    tie_outs=(list(event["tie_outs"])
                              if index == len(points) - 2
                              else [True] * len(event["pitches"])),
                )
                pieces.append(piece)
        by_span: Dict[tuple, List[dict]] = defaultdict(list)
        for piece in pieces:
            if piece.get("is_tup"):
                by_span[(Fraction(piece["onset"]), Fraction(piece["dur"]),
                         piece.get("tup_n"))].append(piece)
        groups = []
        for members in by_span.values():
            head = members[0]
            if len(members) > 1:
                # Two decided voices sharing one channel and one tuplet
                # window are one written group: chord the slots.  The
                # slot-level departs flag is homogeneous, so each side's
                # flags flatten to per-pitch tie marks before the union.
                for group in members:
                    _flatten_slot_flags(group)
                for other in members[1:]:
                    for slot_a, slot_b in zip(head["slots"], other["slots"]):
                        _merge_tuple_slots(slot_a, slot_b)
            groups.append(head)

        # Two tuplet records of one decided voice on the same hidden grid
        # are one written group: the shorter fuses into its host chordwise.
        absorbed_groups: Set[int] = set()
        for guest in sorted(
                groups, key=lambda group: Fraction(group["dur"])):
            if id(guest) in absorbed_groups:
                continue
            guest_start = Fraction(guest["onset"])
            guest_end = guest_start + Fraction(guest["dur"])
            guest_slots = guest.get("slots") or []
            if not guest_slots:
                continue
            guest_grain = Fraction(guest["dur"], len(guest_slots))
            hosts = []
            for host in groups:
                if host is guest or id(host) in absorbed_groups:
                    continue
                host_start = Fraction(host["onset"])
                host_end = host_start + Fraction(host["dur"])
                host_slots = host.get("slots") or []
                if not host_slots or not (
                        host_start <= guest_start
                        and guest_end <= host_end):
                    continue
                host_grain = Fraction(host["dur"], len(host_slots))
                offset = (guest_start - host_start) / host_grain
                if host_grain != guest_grain or offset.denominator != 1:
                    continue
                hosts.append((Fraction(host["dur"]), host, int(offset)))
            if not hosts:
                continue
            _duration, host, offset = min(hosts, key=lambda item: item[0])
            _flatten_slot_flags(host)
            _flatten_slot_flags(guest)
            for index, slot in enumerate(guest_slots):
                _merge_tuple_slots(host["slots"][offset + index], slot)
            absorbed_groups.add(id(guest))
        groups = [group for group in groups
                  if id(group) not in absorbed_groups]

        for index, group in enumerate(groups):
            gs = Fraction(group["onset"])
            ge = gs + Fraction(group["dur"])
            for other in groups[index + 1:]:
                hs = Fraction(other["onset"])
                he = hs + Fraction(other["dur"])
                if gs < he and hs < ge:
                    raise _CanonicalizationPause(
                        "tuplet groups overlap without a shared window")
        kept_group_ids = {id(group) for group in groups}
        for piece in pieces:
            if piece.get("is_tup"):
                if id(piece) in kept_group_ids:
                    out.append(piece)
                continue
            if not piece.get("pitches") or piece["pitches"] == ["r"]:
                out.append(piece)
                continue
            start = Fraction(piece["onset"])
            end = start + Fraction(piece["dur"])
            host = next(
                (group for group in groups
                 if Fraction(group["onset"]) <= start
                 and end <= Fraction(group["onset"]) + Fraction(group["dur"])),
                None)
            if host is None:
                out.append(piece)
                continue
            # The tuplet owns the time grid.  A decided regular voice can
            # share it only by becoming tied members on that exact grid.
            _embed_note_in_tuple(host, piece)
    return out


def _freeze_paused_outcomes(state: VoiceState) -> None:
    if not state.paused_bars:
        return
    intervals = [
        (state.bars[index].offset,
         state.bars[index].offset + state.bars[index].tree.bar_length,
         state.bars[index].label)
        for index in state.paused_bars
    ]
    frozen = []
    for outcome in state.outcomes:
        if outcome.execution != "applied":
            frozen.append(outcome)
            continue
        touches = False
        if isinstance(outcome.span, tuple) and len(outcome.span) >= 2 \
                and isinstance(outcome.span[0], Fraction) \
                and isinstance(outcome.span[1], Fraction):
            touches = any(outcome.span[0] < end and start < outcome.span[1]
                          for start, end, _label in intervals)
        elif isinstance(outcome.span, str):
            touches = any(outcome.span == label
                          for _start, _end, label in intervals)
        frozen.append(replace(outcome, execution="paused")
                      if touches else outcome)
    state.outcomes = frozen


def _freeze_all_outcomes(state: VoiceState, execution: str) -> None:
    state.outcomes = [
        replace(outcome, execution=execution)
        if outcome.execution == "applied" else outcome
        for outcome in state.outcomes
    ]


def _writer_slot_for_owner(
    lanes: Iterable[tuple],
    owner: tuple,
) -> Optional[tuple]:
    """Return the owner's bar-local written column without lineage repair."""
    return owner if owner in set(lanes) else None


def _structural_head_ops(
    state: VoiceState, bar: _VoiceBar,
    incoming: tuple, target: tuple, lane_hands: Dict[tuple, int],
) -> List[str]:
    """Expand each hand's right edge to the bar's written count."""
    del state
    lanes = list(incoming)
    operations: List[str] = []
    for hand in (0, 1):
        target_count = sum(lane_hands[lane] == hand for lane in target)
        current_count = sum(lane_hands[lane] == hand for lane in lanes)
        if current_count > target_count:
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: a bar head would need to close a lane")
        while current_count < target_count:
            positions = [
                index for index, lane in enumerate(lanes)
                if lane_hands[lane] == hand
            ]
            if not positions:
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: a hand has no split baseline")
            anchor = positions[-1]
            operations.append("\t".join(
                "*^" if index == anchor else "*"
                for index in range(len(lanes))))
            current_count += 1
            placeholder = (hand, f"__split_{current_count}")
            lanes.insert(anchor + 1, placeholder)
            lane_hands[placeholder] = hand
    return operations


def _structural_tail_ops(
    state: VoiceState, bar: _VoiceBar,
    lanes_in_bar: tuple, outgoing: tuple, lane_hands: Dict[tuple, int],
) -> List[str]:
    """Contract each hand's right edge to the next written count."""
    del state
    lanes = list(lanes_in_bar)
    operations: List[str] = []
    for hand in (0, 1):
        target_count = sum(lane_hands[lane] == hand for lane in outgoing)
        current_count = sum(lane_hands[lane] == hand for lane in lanes)
        if current_count < target_count:
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: a bar tail would need to open a lane")
        while current_count > target_count:
            positions = [
                index for index, lane in enumerate(lanes)
                if lane_hands[lane] == hand
            ]
            if len(positions) < 2:
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: a hand cannot merge below baseline")
            left, right = positions[-2:]
            if right != left + 1:
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: hand columns are not contiguous")
            operations.append("\t".join(
                "*v" if index in {left, right} else "*"
                for index in range(len(lanes))))
            lanes.pop(right)
            current_count -= 1
    return operations


def _layout_prefix(
    lanes: tuple, counts: Dict[int, int], lane_hands: Dict[tuple, int],
) -> tuple:
    """Keep the leftmost written slots needed at a count-only boundary."""
    return tuple(
        lane
        for hand in (0, 1)
        for lane in [item for item in lanes if lane_hands[item] == hand][
            :counts[hand]
        ]
    )


def _bar_topology_plan(
    state: VoiceState, bar: _VoiceBar,
) -> Tuple[tuple, tuple, tuple, List[str], List[str]]:
    """Derive one writer seam solely from adjacent normalized lane tables."""
    position = state.bars.index(bar)
    preceding = (
        state.bars[position - 1] if position > 0 else None
    )
    following = (
        state.bars[position + 1]
        if position + 1 < len(state.bars) else None
    )
    topology = _source_topology(state)
    lanes = topology.written_lanes(bar)
    previous_lanes = (
        topology.written_lanes(preceding)
        if preceding is not None else tuple()
    )
    next_lanes = (
        topology.written_lanes(following)
        if following is not None else tuple()
    )
    lane_hands = {
        lane: topology.written_hand(bar, lane)
        for lane in lanes
    }
    current_counts = {
        hand: sum(lane_hands[lane] == hand for lane in lanes)
        for hand in (0, 1)
    }
    previous_counts = {
        hand: (sum(topology.written_hand(preceding, lane) == hand
                   for lane in previous_lanes)
               if preceding is not None else 1)
        for hand in (0, 1)
    }
    next_counts = {
        hand: (sum(topology.written_hand(following, lane) == hand
                   for lane in next_lanes)
               if following is not None else 1)
        for hand in (0, 1)
    }
    incoming_counts = {
        hand: min(previous_counts[hand], current_counts[hand])
        for hand in (0, 1)
    }
    outgoing_counts = {
        hand: min(current_counts[hand], next_counts[hand])
        for hand in (0, 1)
    }
    incoming = _layout_prefix(lanes, incoming_counts, lane_hands)
    outgoing = _layout_prefix(lanes, outgoing_counts, lane_hands)
    return (
        incoming,
        lanes,
        outgoing,
        _structural_head_ops(
            state, bar, incoming, lanes, dict(lane_hands)),
        _structural_tail_ops(
            state, bar, lanes, outgoing, dict(lane_hands)),
    )


def _source_writer_bar(state: VoiceState, bar: _VoiceBar) -> _WriterBar:
    incoming, outgoing = _source_topology(state).source_bar_boundaries(bar)
    return _WriterBar(
        events=[], widths={0: 0, 1: 0}, lanes=incoming,
        lane_hands={lane: lane[0] for lane in incoming},
        incoming=incoming, outgoing=outgoing,
        head_ops=[], tail_ops=[],
        source_rows=list(state.lines[bar.start_row + 1:bar.stop_row + 1]),
    )


def _suppress_tuple_atoms(
    event: dict,
    piece: _PitchPiece,
    suppressions: Set[Tuple[str, Fraction, Fraction]],
) -> None:
    """Replace locally absorbed duplicate tuplet notes with equal-slot rests."""
    slots = event.get("slots") or []
    if not slots:
        return
    grain = (piece.end - piece.start) / len(slots)
    affected: Set[int] = set()

    for index, slot in enumerate(slots):
        slot_start = piece.start + index * grain
        slot_end = slot_start + grain
        pitches = {
            pitch
            for pitch, start, end in suppressions
            if start < slot_end and slot_start < end
        }
        if not pitches:
            continue

        attacks = list(slot.get("attack", []))
        tie_ins = list(slot.get("attack_ti", [False] * len(attacks)))
        tie_outs = list(slot.get("attack_to", [False] * len(attacks)))
        kept_attacks = [
            (pitch, tie_in, tie_out)
            for pitch, tie_in, tie_out
            in zip(attacks, tie_ins, tie_outs)
            if pitch not in pitches
        ]
        slot["attack"] = [item[0] for item in kept_attacks]
        slot["attack_ti"] = [item[1] for item in kept_attacks]
        slot["attack_to"] = [item[2] for item in kept_attacks]

        continuations = list(slot.get("cont", []))
        continuation_ties = list(
            slot.get("cont_ti", [True] * len(continuations)))
        continuation_outs = list(
            slot.get("cont_to", [False] * len(continuations)))
        kept_continuations = [
            (pitch, tie_in, tie_out)
            for pitch, tie_in, tie_out in zip(
                continuations, continuation_ties, continuation_outs)
            if pitch not in pitches
        ]
        slot["cont"] = [item[0] for item in kept_continuations]
        slot["cont_ti"] = [item[1] for item in kept_continuations]
        slot["cont_to"] = [item[2] for item in kept_continuations]
        if not any(
            pitch != "r"
            for pitch in slot["attack"] + slot["cont"]
        ):
            affected.add(index)
            slot["departs"] = False

    index = 0
    while index < len(slots):
        if index not in affected:
            index += 1
            continue
        stop = index + 1
        while stop < len(slots) and stop in affected:
            stop += 1
        for member in range(index, stop):
            slot = slots[member]
            slot["attack"] = ["r"] if member == index else []
            slot["attack_ti"] = [False] if member == index else []
            slot["attack_to"] = [False] if member == index else []
            slot["cont"] = [] if member == index else ["r"]
            slot["cont_ti"] = [] if member == index else [False]
            slot["cont_to"] = [] if member == index else [False]
            slot["departs"] = False
        index = stop


def _prepare_writer_bar(
    state: VoiceState,
    bar: _VoiceBar,
    live_by_piece: Dict[int, AttackReleaseAccount],
) -> _WriterBar:
    incoming, lanes, outgoing, head_ops, tail_ops = \
        _bar_topology_plan(state, bar)
    lane_role: Dict[tuple, int] = {}
    widths = {0: 0, 1: 0}
    topology = _source_topology(state)
    lane_hands = {
        lane: topology.written_hand(bar, lane)
        for lane in lanes
    }
    for lane in lanes:
        hand = lane_hands[lane]
        widths[hand] += 1
        lane_role[lane] = widths[hand]
    if any(widths[hand] == 0 for hand in (0, 1)):
        raise _CanonicalizationPause(
            "source topology leaves a hand without a staff lane")
    def active_owner(owner: tuple) -> tuple:
        slot = _writer_slot_for_owner(lanes, owner)
        if slot is None:
            raise _CanonicalizationPause(
                "a decided voice has no source lane in its bar")
        return slot

    events = []
    for source_event in bar.events:
        if source_event.get("is_tup"):
            piece_ids = source_event.get("_piece_ids", [])
            if not piece_ids:
                continue
            account = live_by_piece.get(piece_ids[0])
            if account is None:
                continue
            owner = (source_event["_source_channel"]
                     if bar.index in state.paused_bars
                     else state.pieces[piece_ids[0]].owner)
            grouped = [(active_owner(owner), None)]
        else:
            grouped_members: Dict[tuple, List[int]] = defaultdict(list)
            for member, piece_id in enumerate(
                    source_event.get("_piece_ids", [])):
                account = live_by_piece.get(piece_id)
                if account is None:
                    continue
                owner = (source_event["_source_channel"]
                         if bar.index in state.paused_bars
                         else state.pieces[piece_id].owner)
                grouped_members[active_owner(owner)].append(member)
            grouped = list(grouped_members.items())
        for owner, members in grouped:
            event = (
                deepcopy(source_event)
                if source_event.get("is_tup")
                else dict(source_event)
            )
            if source_event.get("is_tup") and account is not None:
                suppressions = state.suppressed_tuple_atoms.get(
                    account.account_id, set())
                if suppressions:
                    _suppress_tuple_atoms(
                        event, state.pieces[piece_ids[0]], suppressions)
            if members is not None:
                event["pitches"] = [source_event["pitches"][index]
                                    for index in members]
                event["tie_ins"] = [source_event["tie_ins"][index]
                                     for index in members]
                event["tie_outs"] = [source_event["tie_outs"][index]
                                      for index in members]
                if not event["pitches"]:
                    continue
            hand = lane_hands[owner]
            role = lane_role[owner]
            event["hand"] = hand
            event["voice"] = role
            event["chain"] = role - 1
            event["_owner_source"] = owner
            events.append(event)

    events = _slice_channel_overlaps(events)
    # A decided voice is serialized from the union of its source grids.  Tie
    # attachments remain pending independently of that grid and therefore
    # must not be collapsed into a duration span here.
    events = _regroup_same_onset(events)
    events = _expand_spell(events, bar.tree)
    events = _regroup_same_onset(events)
    events = _fill_channel_rests(events, bar.tree, widths)
    return _WriterBar(
        events=events, widths=widths, lanes=lanes,
        lane_hands=lane_hands,
        incoming=incoming, outgoing=outgoing,
        head_ops=head_ops, tail_ops=tail_ops,
    )


def _column_ledger_error(text: str) -> Optional[str]:
    """Walk spine ops and report the first row whose column count breaks.

    Mutation-agnostic fail-closed backstop: a rewrite that splices bars with
    mixed column layouts cannot pass this walk, no matter which decision path
    produced it.
    """
    width: Optional[int] = None
    seen_header = False
    terminated = False
    for number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("!!"):
            continue
        tokens = line.split("\t")
        if not seen_header:
            if terminated:
                return f"line {number}: content after spine termination"
            if all(token.startswith("**") for token in tokens):
                width = len(tokens)
                seen_header = True
                continue
            return f"line {number}: content before exclusive interpretation"
        if terminated:
            return f"line {number}: content after spine termination"
        if width is None:
            return f"line {number}: spine ledger has no active width"
        if len(tokens) != width:
            return (f"line {number}: {len(tokens)} columns, "
                    f"ledger says {width}: {line!r}")
        if all(token.startswith("*") for token in tokens):
            width += sum(1 for token in tokens if token in ("*^", "*+"))
            width -= sum(1 for token in tokens if token == "*-")
            if sum(1 for token in tokens if token == "*x") % 2:
                return f"line {number}: unpaired spine exchange"
            run = 0
            for token in tokens + ["*"]:
                if token == "*v":
                    run += 1
                    continue
                if run == 1:
                    return f"line {number}: solitary spine merge"
                if run >= 2:
                    width -= run - 1
                run = 0
            if width < 0:
                return f"line {number}: spine width became negative"
            if width == 0:
                width = None
                terminated = True
    if not seen_header:
        return "missing exclusive interpretation"
    if not terminated:
        return f"end of file: {width} active spines remain"
    return None


def _prepare_writer_bars(state: VoiceState) -> List[_WriterBar]:
    held_bars = _material_hold_bars(state)
    writer_piece_ids = {
        piece_id
        for bar in state.bars
        if bar.index not in held_bars
        for event in bar.events
        for piece_id in event.get("_piece_ids", [])
    }
    live_by_piece: Dict[int, AttackReleaseAccount] = {}
    for account in state.accounts:
        if account.retired:
            continue
        for piece_id in account.piece_ids:
            if piece_id not in writer_piece_ids:
                continue
            if piece_id in live_by_piece:
                raise _CanonicalizationPause(
                    "one note piece belongs to multiple live voice accounts")
            live_by_piece[piece_id] = account

    prepared: List[_WriterBar] = []
    for bar in state.bars:
        if bar.index in held_bars or not bar.data_rows:
            # A zero-data segment occupies no time; filling it with rests
            # would materialize a bar and shift the whole timeline.
            prepared.append(_source_writer_bar(state, bar))
            continue
        try:
            plan = _prepare_writer_bar(state, bar, live_by_piece)
        except ValueError as error:
            state.paused_bars.add(bar.index)
            _add_voice_issue(
                state, _issue_family_for_message(str(error)),
                f"{bar.label or 'pickup'}: {error}", bar.index)
            _record(
                state, "refuse", "writer-bar-hold",
                source=None, target=None,
                span=bar.label or "pickup", execution="paused",
            )
            plan = _source_writer_bar(state, bar)
            logger.info("%s writer held: %s", bar.label or "pickup", error)
        prepared.append(plan)
    return prepared


def _kern_data_rows(
    events: List[dict], widths: Dict[int, int],
) -> Tuple[List[tuple], List[tuple]]:
    channels = [(hand, role)
                for hand in (0, 1)
                for role in range(widths[hand])]
    cells: Dict[Fraction, Dict[tuple, List[str]]] = defaultdict(
        lambda: defaultdict(list))
    for event in sorted(events, key=lambda item: (
            Fraction(item["onset"]), item["hand"], item.get("voice") or 1,
            -Fraction(item["dur"]),
    )):
        channel = (event["hand"], (event.get("voice") or 1) - 1)
        if event.get("is_tup"):
            for onset, cell in _tuplet_rows(event):
                cells[onset][channel].append(cell)
        else:
            cells[Fraction(event["onset"])][channel].append(
                _event_cell(event))
    rows = []
    for onset in sorted(cells):
        fields = [" ".join(cells[onset].get(channel, [])) or "."
                  for channel in channels]
        rows.append((onset, "\t".join(fields)))
    return rows, channels


def _remap_carried(
    carried: _CarriedRow,
    lanes: tuple,
) -> Optional[str]:
    if carried.text.startswith("!!"):
        return carried.text
    fields = carried.text.split("\t")
    if carried.text.startswith("!"):
        return "\t".join("!" for _ in lanes)
    if not carried.text.startswith("*"):
        return carried.text
    selected = []
    for lane in lanes:
        candidates = [index for index, source in enumerate(carried.channels)
                      if source == lane]
        index = candidates[0] if candidates else None
        token = fields[index] if index is not None and index < len(fields) \
            else "*"
        selected.append("*" if token in {"*^", "*v", "*x", "*+", "*-"}
                        else token)
    if selected and all(token == "*" for token in selected):
        return None
    return "\t".join(selected)


def _canonical_header_rows(
    state: VoiceState,
    first_lanes: tuple,
) -> Tuple[List[str], List[str]]:
    """Separate the fixed header from rows owned by the first bar seam."""
    prefix: List[str] = []
    after_head: List[str] = []
    source_lanes = state.initial_channels
    topology_started = False

    for row, line in enumerate(state.header):
        layout = state.op_layouts.get(row)
        if _is_spine_operation(line):
            topology_started = True
            if layout is not None:
                source_lanes = layout[2]
            continue
        if not topology_started:
            prefix.append(line)
            continue
        mapped = _remap_carried(
            _CarriedRow(
                row=row,
                onset=Fraction(0),
                text=line,
                channels=tuple(source_lanes),
            ),
            first_lanes,
        )
        if mapped is not None:
            after_head.append(mapped)
    return prefix, after_head


def _canonical_trailer_rows(
    state: VoiceState,
    final_lanes: tuple,
) -> List[str]:
    """Write the final seam once, then terminate its normalized columns."""
    if not state.trailer:
        return []

    trailer_start = len(state.lines) - len(state.trailer)
    source_lanes = state.initial_channels
    for row, layout in sorted(state.op_layouts.items()):
        if row >= trailer_start:
            break
        source_lanes = layout[2]

    before_termination: List[str] = []
    after_termination: List[str] = []
    terminated = False
    for offset, line in enumerate(state.trailer):
        row = trailer_start + offset
        layout = state.op_layouts.get(row)
        if _is_spine_operation(line):
            tokens = line.split("\t")
            if tokens and all(token == "*-" for token in tokens):
                if not terminated:
                    before_termination.append("\t".join(
                        "*-" for _lane in final_lanes))
                    terminated = True
            if layout is not None:
                source_lanes = layout[2]
            continue
        if terminated:
            after_termination.append(line)
            continue
        mapped = _remap_carried(
            _CarriedRow(
                row=row,
                onset=Fraction(0),
                text=line,
                channels=tuple(source_lanes),
            ),
            final_lanes,
        )
        if mapped is not None:
            before_termination.append(mapped)
    return before_termination + after_termination


def _write_voice_state(
    state: VoiceState,
    prepared: List[_WriterBar],
) -> str:
    topology = _source_topology(state)
    if prepared[0].source_rows is not None:
        output = list(state.header)
        first_bar_rows: List[str] = []
    else:
        output, first_bar_rows = _canonical_header_rows(
            state, prepared[0].lanes)
    if state.bars[0].label:
        output.append("\t".join(
            state.bars[0].label for _lane in prepared[0].incoming))
    for position, (bar, plan) in enumerate(zip(state.bars, prepared)):
        relocated = [
            row for row in range(bar.start_row + 1, bar.stop_row)
            if topology.raw_tail_merge_targets.get(row) == position - 1
        ]
        if relocated:
            if not output or not output[-1].startswith("="):
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: head merge has no seam")
            before, _tokens, _after = state.op_layouts[relocated[0]]
            _before, _tokens, after = state.op_layouts[relocated[-1]]
            seam_width = len(output[-1].split("\t"))
            if seam_width == len(before):
                if position == 0 \
                        or prepared[position - 1].source_rows is None:
                    raise _CanonicalizationPause(
                        f"{bar.label or 'pickup'}: canonical bar tail "
                        "did not apply its normalized merge")
                output.pop()
                output.extend(state.lines[row] for row in relocated)
                output.append("\t".join(bar.label for _lane in after))
            elif seam_width != len(after):
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: head merge disagrees "
                    "with the preceding seam")
        if plan.source_rows is not None:
            source_rows = list(plan.source_rows)
            if relocated:
                relocated_set = set(relocated)
                source_rows = [
                    line for row, line in zip(
                        range(bar.start_row + 1, bar.stop_row + 1),
                        source_rows)
                    if row not in relocated_set
                ]
            output.extend(source_rows)
            continue
        output.extend(plan.head_ops)
        if position == 0:
            output.extend(first_bar_rows)
        data_rows, expected_channels = _kern_data_rows(
            plan.events, plan.widths)
        expected_lanes = tuple(
            lane for hand in (0, 1)
            for lane in plan.lanes
            if plan.lane_hands[lane] == hand)
        if plan.lanes != expected_lanes:
            raise ValueError(
                f"{bar.label}: source voice order changed")
        expected_roles = [(hand, role)
                          for hand in (0, 1)
                          for role in range(plan.widths[hand])]
        if expected_channels != expected_roles:
            raise ValueError(f"{bar.label}: writer role disagreement")
        carried_before: Dict[Fraction, List[_CarriedRow]] = defaultdict(list)
        carried_after: Dict[Fraction, List[_CarriedRow]] = defaultdict(list)
        for carried in bar.carried:
            destination = carried_after if carried.after_data \
                else carried_before
            destination[carried.onset].append(carried)
        emitted_carried = set()
        for onset, row in data_rows:
            for carried in carried_before.get(onset, []):
                mapped = _remap_carried(carried, plan.lanes)
                if mapped is not None:
                    output.append(mapped)
                emitted_carried.add(carried.row)
            output.append(row)
            for carried in carried_after.get(onset, []):
                mapped = _remap_carried(carried, plan.lanes)
                if mapped is not None:
                    output.append(mapped)
                emitted_carried.add(carried.row)
        for carried in bar.carried:
            if carried.row in emitted_carried:
                continue
            mapped = _remap_carried(carried, plan.lanes)
            if mapped is not None:
                output.append(mapped)

        output.extend(plan.tail_ops)
        output.append("\t".join(
            bar.closing_label for _lane in plan.outgoing))

    if prepared[-1].source_rows is not None:
        output.extend(state.trailer)
    else:
        output.extend(_canonical_trailer_rows(
            state, prepared[-1].outgoing))
    return "\n".join(output) + "\n"


def _restore_accounts(state: VoiceState) -> bool:
    """Move a uniquely identified overreaching source episode.

    Performing-hand brackets describe who plays the notes while the source
    staves describe page layout.  Only simultaneous fresh attacks establish
    an overreach; sustained tails do not occupy the hand again.
    """
    moved_accounts: Set[int] = set()
    move_requests: List[
        Tuple[int, tuple, int, Set[int], Fraction, Fraction]
    ] = []

    def log_refusal(bar_indexes, message: str) -> None:
        """Keep engraved hands when no unique playable rewrite exists."""
        for bar_index in bar_indexes:
            logger.info(
                "%s hand restoration retained source: %s",
                state.bars[bar_index].label or "pickup", message,
            )

    account_atoms = {
        account.account_id: tuple(
            (kern_pitch_to_midi(pitch), start, end)
            for pitch, start, end in _account_sound_atoms(
                state, account)
        )
        for account in state.accounts
    }
    attack_atoms = {
        account.account_id: tuple(
            (kern_pitch_to_midi(pitch), start, end)
            for pitch, start, end in _account_attack_atoms(
                state, account, source=False)
        )
        for account in state.accounts
    }
    live_accounts_by_piece: Dict[int, Set[int]] = defaultdict(set)
    for account in state.accounts:
        if not account_atoms[account.account_id]:
            continue
        for piece_id in account.piece_ids:
            live_accounts_by_piece[piece_id].add(account.account_id)
    event_accounts: Dict[Tuple[int, int], Set[int]] = {}
    account_events: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for bar in state.bars:
        for event_index, event in enumerate(bar.events):
            key = (bar.index, event_index)
            members = {
                account_id
                for piece_id in event.get("_piece_ids", ())
                for account_id in live_accounts_by_piece.get(piece_id, ())
            }
            event_accounts[key] = members
            for account_id in members:
                account_events[account_id].add(key)
    accounts_by_bar = {
        bar.index: [
            account for account in state.accounts
            if any(
                start < bar.offset + bar.tree.bar_length
                and bar.offset < end
                for _pitch, start, end
                in account_atoms[account.account_id]
            )
        ]
        for bar in state.bars
    }
    topology = _source_topology(state)

    for bar in state.bars:
        lo = bar.offset
        hi = lo + bar.tree.bar_length
        for hand in (0, 1):
            candidates_accounts = [
                account for account in accounts_by_bar[bar.index]
                if account.source[0] == hand
                and account.account_id not in moved_accounts
            ]
            sources = sorted({account.owner for account in candidates_accounts})
            if len(sources) < 2:
                continue
            # Only keys struck together occupy one hand at once: a note
            # counts at its attack instant alone, while a sounding tail
            # rides the pedal and an earlier attack has already freed the
            # hand.  The strike must involve two voices, or moving a voice
            # cannot relieve it.
            attack_instants = sorted({
                start
                for account in candidates_accounts
                for _pitch, start, _end
                in attack_atoms[account.account_id]
                if lo <= start < hi
            })
            overreach_instants = []
            for instant in attack_instants:
                struck = [
                    (account.owner, pitch)
                    for account in candidates_accounts
                    for pitch, start, _end
                    in attack_atoms[account.account_id]
                    if start == instant
                ]
                pitches = [pitch for _owner, pitch in struck]
                if len({owner for owner, _pitch in struck}) >= 2 \
                        and max(pitches) - min(pitches) >= 19:
                    overreach_instants.append(instant)
            if not overreach_instants:
                continue
            outer = []
            for source in sources:
                valid = True
                compared = False
                for instant in overreach_instants:
                    mine = [
                        pitch
                        for account in candidates_accounts
                        if account.owner == source
                        for pitch, start, _end
                        in attack_atoms[account.account_id]
                        if start == instant
                    ]
                    other = [
                        pitch
                        for account in candidates_accounts
                        if account.owner != source
                        for pitch, start, _end
                        in attack_atoms[account.account_id]
                        if start == instant
                    ]
                    if not mine or not other:
                        continue
                    compared = True
                    if hand == 1 and max(mine) >= min(other):
                        valid = False
                        break
                    if hand == 0 and min(mine) <= max(other):
                        valid = False
                        break
                if valid and compared:
                    outer.append(source)
            if len(outer) != 1:
                _record(
                    state, "refuse", "hand-crossing"
                    if not outer else "hand-candidate-ambiguous",
                    source=tuple(sources), target=None,
                    span=bar.label, execution="refused",
                )
                log_refusal(
                    [bar.index],
                    "an overreached hand has no unique outer voice",
                )
                continue
            source = outer[0]
            # The evidence and the moved voice must share the strike: the
            # candidate has to be part of the overreach it relieves.
            if not any(
                    any(start == instant
                        for account in candidates_accounts
                        if account.owner == source
                        for _pitch, start, _end
                        in attack_atoms[account.account_id])
                    for instant in overreach_instants):
                _record(
                    state, "refuse", "hand-candidate-outside-overreach",
                    source=source, target=None,
                    span=bar.label, execution="refused",
                )
                log_refusal(
                    [bar.index],
                    "the outer voice does not sound at the overreach "
                    "instant",
                )
                continue
            # The moved unit is the picked voice's material inside the
            # triggering bar: bar lines and spine operations bound a move,
            # and each bar re-triggers on its own physical evidence.
            selected = [
                account for account in candidates_accounts
                if account.owner == source
                and any(lo <= start < hi
                        for _pitch, start, _end
                        in attack_atoms[account.account_id])
            ]
            if not selected:
                continue
            selected_ids = {account.account_id for account in selected}
            episode_bars = [
                candidate for candidate in state.bars
                if any(
                    start < candidate.offset + candidate.tree.bar_length
                    and candidate.offset < end
                    for account in selected
                    for _pitch, start, end
                    in account_atoms[account.account_id])
            ]
            destination_hand = 1 - hand
            # The destination hand may also overreach with this voice.  A
            # whole-life move would only trade one overreach for another
            # (and re-decision would trade it back), so when both hands pin
            # the voice somewhere, hands are assigned piecewise: each
            # uninterrupted run of origin pins moves as one span, and the
            # handover lands where the origin hand's blocking notes
            # release, at the voice's own next strike.
            selected_strikes = sorted({
                start
                for account in selected
                for _pitch, start, _end in attack_atoms[account.account_id]
            })

            def pinned_instants(partner_hand: int) -> List[Fraction]:
                partners = [
                    partner for partner in state.accounts
                    if not partner.retired
                    and partner.account_id not in selected_ids
                    and partner.source[0] == partner_hand
                    and account_atoms[partner.account_id]
                ]
                pinned = []
                for instant in selected_strikes:
                    mine = [
                        pitch
                        for account in selected
                        for pitch, start, _end
                        in attack_atoms[account.account_id]
                        if start == instant
                    ]
                    theirs = [
                        pitch
                        for partner in partners
                        for pitch, start, _end
                        in attack_atoms[partner.account_id]
                        if start == instant
                    ]
                    if mine and theirs and \
                            max(mine + theirs) - min(mine + theirs) >= 19:
                        pinned.append(instant)
                return pinned

            destination_pins = pinned_instants(destination_hand)
            if destination_pins:
                origin_pins = pinned_instants(hand)
                anchor = next(
                    (instant for instant in overreach_instants
                     if any(
                         start == instant
                         for account in selected
                         for _pitch, start, _end
                         in attack_atoms[account.account_id])),
                    None,
                )
                destination_set = set(destination_pins)
                runs: List[List[Fraction]] = []
                current_run: List[Fraction] = []
                for instant in sorted(set(origin_pins) | destination_set):
                    if instant in destination_set:
                        if current_run:
                            runs.append(current_run)
                        current_run = []
                    else:
                        current_run.append(instant)
                if current_run:
                    runs.append(current_run)
                run = next(
                    (candidate for candidate in runs if anchor in candidate),
                    None,
                )
                handover = None
                if run is not None:
                    last_pin = run[-1]
                    release = max(
                        (end
                         for partner in state.accounts
                         if not partner.retired
                         and partner.account_id not in selected_ids
                         and partner.source[0] == hand
                         for _pitch, start, end
                         in account_atoms[partner.account_id]
                         if start == last_pin),
                        default=last_pin,
                    )
                    handover = next(
                        (instant for instant in selected_strikes
                         if instant >= release),
                        None,
                    )
                voice_end = max(
                    end
                    for account in selected
                    for _pitch, _start, end
                    in account_atoms[account.account_id]
                )
                segment_start = run[0] if run is not None else anchor
                bound = handover if handover is not None else voice_end
                if run is None or any(
                        segment_start <= blocker < bound
                        for blocker in destination_pins):
                    # The origin blockers are still sounding when the
                    # destination's own overreach arrives: no single-hand
                    # assignment exists anywhere in between.
                    _record(
                        state, "refuse", "hand-move-hands-conflict",
                        source=source, target=destination_hand,
                        span=(bar.label, segment_start, bound),
                        execution="refused",
                    )
                    log_refusal(
                        [candidate.index for candidate in episode_bars
                         if candidate.offset < bound
                         and segment_start
                         < candidate.offset + candidate.tree.bar_length],
                        "the hands' reaches conflict with no free "
                        "handover instant",
                    )
                    continue
                attacks = {
                    account.account_id: min(
                        (start for _pitch, start, _end
                         in attack_atoms[account.account_id]),
                        default=None)
                    for account in selected
                }
                segment = [
                    account for account in selected
                    if attacks[account.account_id] is not None
                    and segment_start <= attacks[account.account_id]
                    and (handover is None
                         or attacks[account.account_id] < handover)
                ]
                if segment and len(segment) != len(selected):
                    segment_ids = {
                        account.account_id for account in segment
                    }
                    selected = segment
                    selected_ids = segment_ids
                    segment_lo = min(
                        start
                        for account in selected
                        for _pitch, start, _end
                        in attack_atoms[account.account_id]
                    )
                    segment_hi = max(
                        end
                        for account in selected
                        for _pitch, _start, end
                        in account_atoms[account.account_id]
                    )
                    episode_bars = [
                        candidate for candidate in episode_bars
                        if candidate.offset < segment_hi
                        and segment_lo
                        < candidate.offset + candidate.tree.bar_length
                    ]
                    if not episode_bars:
                        continue
            touched_events = {
                key for account_id in selected_ids
                for key in account_events.get(account_id, ())
            }
            partial_key = next((
                key for key in sorted(touched_events)
                if any(
                    len({
                        destination_hand
                        if account_id in selected_ids
                        else state.accounts[account_id].hand
                        for account_id in owner_accounts
                    }) > 1
                    for owner_accounts in (
                        {
                            account_id for account_id in event_accounts[key]
                            if state.accounts[account_id].owner == owner
                        }
                        for owner in {
                            state.accounts[account_id].owner
                            for account_id in event_accounts[key]
                        }
                    )
                )
            ), None)
            if partial_key is not None:
                bar_index, event_index = partial_key
                event = state.bars[bar_index].events[event_index]
                _record(
                    state, "refuse", "hand-move-partial-chord",
                    source=source, target=destination_hand,
                    span=(state.bars[bar_index].label, event["onset"]),
                    execution="refused",
                )
                log_refusal(
                    [bar_index],
                    "moving the overreached voice would divide a chord",
                )
                continue
            move_start = episode_bars[0].offset
            move_end = (
                episode_bars[-1].offset
                + episode_bars[-1].tree.bar_length
            )
            hold_bars = _material_hold_bars(state)
            if any(
                    state.pieces[piece_id].event["_bar"] in hold_bars
                    for account in selected
                    for piece_id in account.piece_ids):
                # A held bar keeps its source text, so an account moved
                # across hands there would sound in two places at once.
                _record(
                    state, "keep", "hand-move-crosses-held-bar",
                    source=source, target=None,
                    span=(bar.label, move_start, move_end),
                    execution="held",
                )
                continue
            moved_accounts.update(selected_ids)
            move_requests.append((
                bar.index, source, destination_hand, set(selected_ids),
                move_start, move_end,
            ))

    if not move_requests:
        return False

    column_accounts: Dict[Tuple[int, tuple], Set[int]] = defaultdict(set)
    for account in state.accounts:
        if account.retired or not account_atoms[account.account_id]:
            continue
        for piece_id in account.piece_ids:
            piece = state.pieces[piece_id]
            key = (piece.event["_bar"], piece.source)
            column_accounts[key].add(account.account_id)

    def account_column_chain(account_ids: Iterable[int]) \
            -> Set[Tuple[int, tuple]]:
        return {
            (state.pieces[piece_id].event["_bar"],
             state.pieces[piece_id].source)
            for account_id in account_ids
            for piece_id in state.accounts[account_id].piece_ids
        }

    def strict_boundary_gaps(
        bar: _VoiceBar,
        source: tuple,
        destination_hand: int,
        proposed: Dict[Tuple[int, tuple], int],
    ) -> Tuple[tuple, ...]:
        """Find only wholly separated columns inside a forced hand move.

        A real voice crossing is not an ordering instruction.  Extra columns
        move only when their whole-bar registers form a strict chain between
        the forced column and the nearest column already played by the
        destination hand.
        """
        lanes = topology.bar_lanes(bar)
        if source not in lanes:
            return ()
        spans = {
            lane: topology._lane_pitch_span(bar, lane)
            for lane in lanes
        }
        source_span = spans[source]
        if source_span is None:
            return ()
        rank = {lane: index for index, lane in enumerate(lanes)}

        def written_hand(lane: tuple) -> int:
            return proposed.get((bar.index, lane), lane[0])

        if destination_hand == 0:
            anchors = [
                lane for lane in lanes
                if lane != source and written_hand(lane) == 0
                and spans[lane] is not None
                and source_span[1] < spans[lane][0]
            ]
            if not anchors:
                return ()
            anchor = min(
                anchors, key=lambda lane: (spans[lane][0], rank[lane]))
            gaps = [
                lane for lane in lanes
                if written_hand(lane) == 1 and spans[lane] is not None
                and source_span[1] < spans[lane][0]
                and spans[lane][1] < spans[anchor][0]
            ]
            gaps.sort(key=lambda lane: (spans[lane][0], rank[lane]))
            chain = [source, *gaps, anchor]
        else:
            anchors = [
                lane for lane in lanes
                if lane != source and written_hand(lane) == 1
                and spans[lane] is not None
                and spans[lane][1] < source_span[0]
            ]
            if not anchors:
                return ()
            anchor = max(
                anchors, key=lambda lane: (spans[lane][1], -rank[lane]))
            gaps = [
                lane for lane in lanes
                if written_hand(lane) == 0 and spans[lane] is not None
                and spans[anchor][1] < spans[lane][0]
                and spans[lane][1] < source_span[0]
            ]
            gaps.sort(key=lambda lane: (spans[lane][0], rank[lane]))
            chain = [anchor, *gaps, source]

        if any(
                spans[left][1] >= spans[right][0]
                for left, right in zip(chain, chain[1:])):
            logger.info(
                "%s hand boundary retained crossing columns",
                bar.label or "pickup",
            )
            return ()
        return tuple(gaps)

    direct_requests = []
    for (bar_index, source, destination_hand, selected_ids,
         move_start, move_end) in move_requests:
        bar = state.bars[bar_index]
        if source not in topology.bar_lanes(bar):
            _record(
                state, "refuse", "hand-move-column-unavailable",
                source=source, target=destination_hand,
                span=(bar.label, move_start, move_end),
                execution="refused",
            )
            log_refusal(
                [bar_index],
                "the selected voice has no source column in this bar",
            )
            continue

        # Direct evidence carries only the selected accounts.  All direct
        # requests are combined before shared-column consistency is checked,
        # so two independently justified sounds may occupy one moved column
        # without turning either landing into evidence for a third sound.
        selected_columns = account_column_chain(selected_ids)
        direct_requests.append((
            bar_index, source, destination_hand, move_start, move_end,
            selected_columns,
        ))

    if not direct_requests:
        return False

    active_requests = set(range(len(direct_requests)))
    rejected_requests: Set[int] = set()
    while active_requests:
        forced_columns: Dict[Tuple[int, tuple], int] = {}
        column_causes: Dict[Tuple[int, tuple], Set[int]] = defaultdict(set)
        conflicting_requests: Set[int] = set()
        for request_index in sorted(active_requests):
            (_bar_index, _source, destination_hand, _move_start, _move_end,
             selected_columns) = direct_requests[request_index]
            for key in selected_columns:
                if key in forced_columns \
                        and forced_columns[key] != destination_hand:
                    conflicting_requests |= column_causes[key]
                    conflicting_requests.add(request_index)
                    continue
                forced_columns[key] = destination_hand
                column_causes[key].add(request_index)

        if conflicting_requests:
            rejected_requests |= conflicting_requests
            active_requests -= conflicting_requests
            continue

        boundary_moves: Set[Tuple[int, tuple, int]] = set()
        for request_index in sorted(active_requests):
            (bar_index, source, destination_hand, _move_start, _move_end,
             _selected_columns) = direct_requests[request_index]
            bar = state.bars[bar_index]
            gaps = strict_boundary_gaps(
                bar, source, destination_hand, forced_columns)
            boundary_columns = {
                (bar.index, lane) for lane in gaps
            }
            # Boundary evidence begins at the gap column itself.  Its own
            # accounts follow their attachment chains; sharing a later
            # landing column does not adopt another account's chain.
            boundary_columns |= account_column_chain({
                account_id
                for lane in gaps
                for account_id in column_accounts.get(
                    (bar.index, lane), ())
            })
            if any(
                    key in forced_columns
                    and forced_columns[key] != destination_hand
                    for key in boundary_columns):
                logger.info(
                    "%s hand boundary retained conflicting columns",
                    bar.label or "pickup",
                )
                continue
            for key in boundary_columns:
                if key not in forced_columns:
                    forced_columns[key] = destination_hand
                    column_causes[key].add(request_index)
            boundary_moves.update(
                (bar_index, lane, destination_hand) for lane in gaps)

        for account in state.accounts:
            if account.retired or not account.piece_ids:
                continue
            keys = {
                (state.pieces[piece_id].event["_bar"],
                 state.pieces[piece_id].source)
                for piece_id in account.piece_ids
            }
            hands = {
                forced_columns.get(key, key[1][0]) for key in keys
            }
            if len(hands) != 1:
                conflicting_requests |= {
                    request_index
                    for key in keys
                    for request_index in column_causes.get(key, ())
                }

        if not conflicting_requests:
            break
        rejected_requests |= conflicting_requests
        active_requests -= conflicting_requests

    for request_index in sorted(rejected_requests):
        (bar_index, source, destination_hand, move_start, move_end,
         _selected_columns) = direct_requests[request_index]
        bar = state.bars[bar_index]
        _record(
            state, "refuse", "hand-move-column-conflict",
            source=source, target=destination_hand,
            span=(bar.label, move_start, move_end),
            execution="refused",
        )
        log_refusal(
            [bar_index],
            "the selected source column cannot move without splitting "
            "a tied sound or conflicting with another hand move",
        )

    if not active_requests:
        return False

    accepted_requests = [
        direct_requests[index][:5] for index in sorted(active_requests)
    ]

    account_hands: Dict[int, int] = {}
    for account in state.accounts:
        if account.retired or not account.piece_ids:
            continue
        hands = {
            forced_columns.get(
                (state.pieces[piece_id].event["_bar"],
                 state.pieces[piece_id].source),
                state.pieces[piece_id].source[0],
            )
            for piece_id in account.piece_ids
        }
        if len(hands) != 1:
            raise _CanonicalizationPause(
                "accepted hand moves split a tied sound")
        account_hands[account.account_id] = next(iter(hands))

    planned_live_by_piece: Dict[int, Set[int]] = defaultdict(set)
    for account in state.accounts:
        if not account_atoms[account.account_id] \
                or account.account_id not in account_hands:
            continue
        for piece_id in account.piece_ids:
            planned_live_by_piece[piece_id].add(
                account_hands[account.account_id])
    for bar in state.bars:
        for event in bar.events:
            piece_hands = [
                planned_live_by_piece.get(piece_id, set())
                for piece_id in event.get("_piece_ids", [])
                if planned_live_by_piece.get(piece_id)
            ]
            hands = {
                hand for members in piece_hands for hand in members
            }
            if not hands:
                continue
            if any(len(members) != 1 for members in piece_hands) \
                    or len(hands) != 1:
                _record(
                    state, "refuse", "hand-move-partial-chord",
                    source=event.get("_source_channel"),
                    target=tuple(sorted(hands)),
                    span=(bar.label, event["onset"]),
                    execution="refused",
                )
                log_refusal(
                    [bar.index],
                    "moving the selected source column would divide a "
                    "chord",
                )
                return False

    state.written_hands = {
        key: hand
        for key, hand in forced_columns.items()
        if key[1][0] != hand
    }
    topology._written_lanes_cache.clear()

    for account in state.accounts:
        if account.account_id not in account_hands:
            continue
        written_hand = account_hands[account.account_id]
        account.hand = written_hand
        for piece_id in account.piece_ids:
            state.pieces[piece_id].hand = written_hand

    for (bar_index, source, destination_hand,
         move_start, move_end) in accepted_requests:
        _record(
            state, "move", "playing-hand",
            source=source, target=destination_hand,
            span=(state.bars[bar_index].label, move_start, move_end),
        )

    for bar_index, source, destination_hand in sorted(boundary_moves):
        bar = state.bars[bar_index]
        _record(
            state, "move", "playing-hand-boundary",
            source=source, target=destination_hand,
            span=(bar.label, bar.offset,
                  bar.offset + bar.tree.bar_length),
        )

    return True


def _owner_decision_gate(state: VoiceState) -> None:
    """Every writer-owner change needs an applied identity decision."""
    decisions: Dict[tuple, Set[Tuple[tuple, str]]] = defaultdict(set)
    owner_decisions: Dict[tuple, Set[tuple]] = defaultdict(set)
    for outcome in state.outcomes:
        if outcome.execution != "applied" \
                or outcome.decision != "move":
            continue
        sources = outcome.source if isinstance(outcome.source, tuple) \
            and outcome.source and isinstance(outcome.source[0], tuple) \
            else (outcome.source,)
        for source in sources:
            if isinstance(source, tuple) and isinstance(outcome.target, tuple):
                decisions[source].add(
                    (outcome.target, outcome.decision))
                owner_decisions[source].add(outcome.target)

    visiting: Set[tuple] = set()
    visited: Set[tuple] = set()

    def verify_acyclic(source: tuple) -> None:
        if source in visited:
            return
        if source in visiting:
            raise _CanonicalizationPause(
                "the applied owner-decision graph contains a cycle")
        visiting.add(source)
        for target in owner_decisions.get(source, ()):
            verify_acyclic(target)
        visiting.remove(source)
        visited.add(source)

    for source in tuple(owner_decisions):
        verify_acyclic(source)

    def decision_reaches(
        source: tuple, target: tuple, *, require_move: bool = False,
    ) -> bool:
        pending = [(source, False)]
        seen = set()
        while pending:
            current, moved = pending.pop()
            if current == target and (moved or not require_move):
                return True
            key = (current, moved)
            if key in seen:
                continue
            seen.add(key)
            pending.extend(
                (next_target, moved or decision == "move")
                for next_target, decision in decisions.get(current, ())
            )
        return False

    for account in state.accounts:
        if account.retired:
            continue
        # Pitch attachments may cross columns without changing either written
        # owner.  An actual owner change still needs an explicit decision.
        if account.owner != account.source \
                and not decision_reaches(
                    account.source, account.owner,
                    require_move=(account.source[0] != account.owner[0])):
            raise _CanonicalizationPause(
                "a voice owner changed without an applied decision")
        for piece_id in account.piece_ids:
            piece = state.pieces[piece_id]
            piece_source = piece.source
            if piece.owner != piece_source \
                    and not decision_reaches(
                        piece_source, piece.owner,
                        require_move=(piece_source[0] != piece.owner[0])) \
                    and not decision_reaches(
                        account.source, piece.owner,
                        require_move=(account.source[0]
                                      != piece.owner[0])):
                raise _CanonicalizationPause(
                    "a note piece moved without an applied whole-life decision")
        written_hands = {
            state.written_hands.get(
                (state.pieces[piece_id].event["_bar"],
                 state.pieces[piece_id].source),
                state.pieces[piece_id].source[0],
            )
            for piece_id in account.piece_ids
        }
        if written_hands and (
                len(written_hands) != 1
                or account.hand != next(iter(written_hands))
                or any(state.pieces[piece_id].hand != account.hand
                       for piece_id in account.piece_ids)):
            raise _CanonicalizationPause(
                "a tied sound disagrees with its bar-local hand projection")


def _sound_union_signature(state: VoiceState) -> tuple:
    intervals: Dict[
        int, List[Tuple[Fraction, Fraction, bool]]
    ] = defaultdict(list)
    for account in state.accounts:
        if account.retired:
            continue
        if account.pitch is not None:
            pitch = kern_pitch_to_midi(account.pitch)
            if pitch >= 0:
                intervals[pitch].append(
                    (account.start, account.end, account.attacked))
            continue
        for piece_id in account.piece_ids:
            piece = state.pieces[piece_id]
            slots = piece.event.get("slots") or []
            if not slots:
                continue
            grain = (piece.end - piece.start) / len(slots)
            active: Dict[str, Tuple[Fraction, bool]] = {}
            for index, slot in enumerate(slots):
                onset = piece.start + index * grain
                continuations = {
                    pitch for pitch in slot.get("cont", ()) if pitch != "r"
                }
                attacks = slot.get("attack", ())
                tie_ins = slot.get("attack_ti", (False,) * len(attacks))
                tied_attacks = {
                    pitch for pitch, tie_in in zip(attacks, tie_ins)
                    if pitch != "r" and tie_in
                }
                fresh_attacks = {
                    pitch for pitch, tie_in in zip(attacks, tie_ins)
                    if pitch != "r" and not tie_in
                }
                sounding = continuations | tied_attacks | fresh_attacks
                for pitch in list(active):
                    if pitch not in sounding or pitch in fresh_attacks:
                        midi = kern_pitch_to_midi(pitch)
                        if midi >= 0:
                            attack_start, attacked = active[pitch]
                            intervals[midi].append(
                                (attack_start, onset, attacked))
                        del active[pitch]
                for pitch in sounding:
                    active.setdefault(
                        pitch, (onset, pitch in fresh_attacks))
            for pitch, (onset, attacked) in active.items():
                midi = kern_pitch_to_midi(pitch)
                if midi >= 0:
                    intervals[midi].append((onset, piece.end, attacked))
    result = []
    for pitch, spans in sorted(intervals.items()):
        merged: List[List[Fraction]] = []
        for start, end, attacked in sorted(spans):
            if not merged:
                merged.append([start, end])
                continue
            previous = merged[-1]
            if start < previous[1]:
                if attacked and start > previous[0]:
                    inherited_end = max(previous[1], end)
                    previous[1] = start
                    merged.append([start, inherited_end])
                else:
                    previous[1] = max(previous[1], end)
                continue
            if start == previous[1] and not attacked:
                previous[1] = max(previous[1], end)
                continue
            merged.append([start, end])
        result.extend((pitch, start, end) for start, end in merged)
    return tuple(result)


def _event_lineage_atoms(event: dict) -> List[tuple]:
    """Keyed attack/release spans at one decided lane address."""
    start = Fraction(event["onset"])
    end = start + Fraction(event["dur"])
    channel = (
        event["hand"],
        event.get("chain", (event.get("voice") or 1) - 1),
    )
    if not event.get("is_tup"):
        pitches = event.get("pitches", ())
        tie_ins = event.get("tie_ins", (False,) * len(pitches))
        return [
            (*channel, pitch, start, end, not bool(tie_in))
            for pitch, tie_in in zip(pitches, tie_ins)
            if pitch != "r"
        ]

    slots = event.get("slots") or []
    if not slots:
        return []
    grain = Fraction(event["dur"]) / len(slots)
    active: Dict[str, Tuple[Fraction, bool]] = {}
    atoms: List[tuple] = []
    for index, slot in enumerate(slots):
        onset = start + index * grain
        continuations = {
            pitch for pitch in slot.get("cont", ()) if pitch != "r"
        }
        attacks = slot.get("attack", ())
        tie_ins = slot.get("attack_ti", (False,) * len(attacks))
        tied = {
            pitch for pitch, tie_in in zip(attacks, tie_ins)
            if pitch != "r" and tie_in
        }
        fresh = {
            pitch for pitch, tie_in in zip(attacks, tie_ins)
            if pitch != "r" and not tie_in
        }
        sounding = continuations | tied | fresh
        for pitch in list(active):
            if pitch not in sounding or pitch in fresh:
                attack_start, attacked = active.pop(pitch)
                atoms.append(
                    (*channel, pitch, attack_start, onset, attacked))
        for pitch in sounding:
            active.setdefault(pitch, (onset, pitch in fresh))
    for pitch, (attack_start, attacked) in active.items():
        atoms.append((*channel, pitch, attack_start, end, attacked))
    return atoms


def _writer_identity_gate(
    source: VoiceState,
    prepared: List[_WriterBar],
    written: VoiceState,
) -> None:
    """The serialized text must reconstruct the decided lane/event mapping."""
    def canonical_atoms(events: List[dict]) -> List[tuple]:
        by_identity: Dict[tuple, List[tuple]] = defaultdict(list)
        for event in events:
            for atom in _event_lineage_atoms(event):
                by_identity[atom[:3]].append(atom)
        merged = []
        for identity, atoms in by_identity.items():
            runs: List[List[object]] = []
            for atom in sorted(atoms, key=lambda item: (
                    item[3], item[4], item[5])):
                start, end, attacked = atom[3:]
                if (runs and start == runs[-1][1]
                        and not attacked):
                    runs[-1][1] = max(runs[-1][1], end)
                    continue
                runs.append([start, end, attacked])
            merged.extend(
                (*identity, start, end, attacked)
                for start, end, attacked in runs
            )
        return sorted(merged, key=repr)

    if len(written.bars) != len(source.bars):
        raise _CanonicalizationPause(
            "writer changed the number of metric bars")
    for bar, plan, written_bar in zip(source.bars, prepared, written.bars):
        if plan.source_rows is not None:
            continue
        _incoming, written_lanes, _outgoing, _head, _tail = \
            _bar_topology_plan(written, written_bar)
        written_roles = {}
        widths = {0: 0, 1: 0}
        written_topology = _source_topology(written)
        for lane in written_lanes:
            hand = written_topology.written_hand(written_bar, lane)
            written_roles[lane] = widths[hand]
            widths[hand] += 1
        actual_events = []
        for event in written_bar.events:
            lane = event.get("_source_channel")
            if lane not in written_roles:
                raise _CanonicalizationPause(
                    f"{bar.label or 'pickup'}: written event has no bar-local lane")
            actual_events.append(dict(
                event, chain=written_roles[lane],
                voice=written_roles[lane] + 1,
            ))
        expected = canonical_atoms(plan.events)
        actual = canonical_atoms(actual_events)
        if actual != expected:
            mismatch = next(
                ((left, right) for left, right in zip(expected, actual)
                 if left != right),
                (expected[len(actual)] if len(expected) > len(actual) else None,
                 actual[len(expected)] if len(actual) > len(expected) else None),
            )
            raise _CanonicalizationPause(
                f"{bar.label or 'pickup'}: writer changed decided lane "
                f"identity: {mismatch[0]!r} != {mismatch[1]!r}")


def _finish_voice_state(
    state: VoiceState, *, restore_hand: bool = False,
) -> str:
    source_sound = _sound_union_signature(state)
    state.decision_mutated = False
    _decide_voices(state)
    decision_changed = state.decision_mutated
    hand_changed = restore_hand and _restore_accounts(state)
    if hand_changed:
        decision_changed = True
    topology = _source_topology(state)
    topology.verify_written_invariants()
    projection_changed = any(
        bar.data_rows
        and (
            topology.written_lanes(bar) != topology.bar_lanes(bar)
            or any(
                sum(topology.written_hand(bar, lane) == hand
                    for lane in topology.written_lanes(bar)) > 1
                for hand in (0, 1)
            )
        )
        for bar in state.bars
    )
    decision_changed |= projection_changed
    relocation_bars = (
        topology.relocation_bars
        - state.paused_bars
        - state.decision_blocked
    )
    if relocation_bars:
        _record(
            state, "write", "topology-relocation",
            source="source-operations", target="bar-boundaries",
            span=tuple(state.bars[index].label or "pickup"
                       for index in sorted(relocation_bars)),
        )
        decision_changed = True
    if not decision_changed:
        return state.original
    _owner_decision_gate(state)
    decide_pauses = set(state.paused_bars)
    prepared = _prepare_writer_bars(state)
    writer_pauses = (state.paused_bars - decide_pauses
                     - state.decision_blocked)
    if writer_pauses and decision_changed:
        # These holds surfaced only while writing, after decisions were
        # already baked into every other bar's plan; splicing now would mix
        # two column layouts, so the round is redecided with them blocked.
        raise _WriterDecisionConflict(writer_pauses)
    _freeze_paused_outcomes(state)
    result = _write_voice_state(state, prepared)
    if result != state.original:
        ledger_error = _column_ledger_error(result)
        if ledger_error is not None:
            raise _CanonicalizationPause(
                f"writer column ledger desync: {ledger_error}")
        written_state = _read_voice_state(result, canonical_order=True)
        if _sound_union_signature(written_state) != source_sound:
            raise _CanonicalizationPause(
                "writer changed the sounding pitch/attack/release union")
        _writer_identity_gate(state, prepared, written_state)
        _record(
            state, "write", "canonical-writer",
            source="VoiceState", target="kern", span="file")
    return result


def _read_voice_state(
    kern_content: str,
    *,
    blocked: Optional[Set[int]] = None,
    canonical_order: bool = False,
) -> VoiceState:
    """Parse one immutable source; ties never choose topology identities."""
    state = _parse_voice_state(kern_content)
    state.decision_blocked = set(blocked or ())
    state.canonical_order = canonical_order
    _build_accounts(state)
    return state


def kern_sound_snapshot(kern_content: str) -> Tuple[tuple, tuple]:
    """Return the keyed sound union and its metric-bar address table."""
    state = _read_voice_state(kern_content)
    bars = tuple(
        (bar.index, bar.label, bar.offset, bar.offset + bar.tree.bar_length)
        for bar in state.bars
    )
    return _sound_union_signature(state), bars


def merge_voice(
    kern_content: str,
    *,
    trace: Optional[List[MergeOutcome]] = None,
    issues: Optional[List[VoiceIssue]] = None,
    restore_hand: bool = False,
) -> str:
    """Project immutable source voices into canonical bar-local columns."""
    outcomes: List[MergeOutcome] = []
    found_issues: List[VoiceIssue] = []
    blocked: Set[int] = set()
    state = None
    result = kern_content
    try:
        source_state = _read_voice_state(kern_content)
        while True:
            state = deepcopy(source_state)
            state.decision_blocked = set(blocked)
            _extend_voice_issues(found_issues, state.issues)
            try:
                result = _finish_voice_state(
                    state, restore_hand=restore_hand)
            except _WriterDecisionConflict as conflict:
                grown = conflict.bars - blocked
                if not grown:
                    raise _CanonicalizationPause(
                        "writer holds keep invalidating decisions")
                _extend_voice_issues(found_issues, state.issues)
                _freeze_all_outcomes(state, "retried")
                outcomes.extend(state.outcomes)
                blocked |= grown
                continue
            _extend_voice_issues(found_issues, state.issues)
            outcomes.extend(state.outcomes)
            break
    except _CanonicalizationPause as error:
        if state is None:
            logger.warning("voice canonicalization paused: %s", error)
        else:
            _freeze_all_outcomes(state, "paused")
            _record(
                state, "refuse", "writer-pause",
                source=None, target=None, span=str(error),
                execution="paused")
            outcomes.extend(state.outcomes)
            # A pause must never fall back silently: the bar it names (or
            # the file head) leaves the canonical set with it.
            message = str(error)
            paused_bar = next(
                (candidate for candidate in state.bars
                 if candidate.label and f"{candidate.label}:" in message),
                None,
            )
            if state.bars:
                _add_voice_issue(
                    state, _issue_family_for_message(message),
                    message if paused_bar is not None
                    else f"file: {message}",
                    paused_bar.index if paused_bar is not None else 0,
                )
            _extend_voice_issues(found_issues, state.issues)
            logger.info("voice canonicalization paused outcomes=%s",
                        state.outcomes)
        result = kern_content
    except ValueError as error:
        issue = _voice_issue_from_error(error)
        if issue is not None:
            _extend_voice_issues(found_issues, [issue])
        if state is not None:
            _freeze_all_outcomes(state, "refused")
            _record(
                state, "refuse", "writer-refusal",
                source=None, target=None, span=str(error),
                execution="refused")
            outcomes.extend(state.outcomes)
            logger.info("voice canonicalization refused outcomes=%s",
                        state.outcomes)
        else:
            logger.warning("voice canonicalization unreadable: %s", error)
        result = kern_content

    if trace is not None:
        trace.extend(outcomes)
    _extend_voice_issues(issues, found_issues)
    return result
