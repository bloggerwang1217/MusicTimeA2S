"""
Kern Tokenizer — voice-bracket vocabulary and duration spelling
===============================================================

Vocabulary (authoritative count = VOCAB_SIZE, built in build_vocab):
    Special:    <pad> <sos> <eos>
    Structural: <bar> <grid> <pl> </pl> <pr> </pr> <tup> </tup>
                <tie> </tie>
    Schema:     <num:N> + <den:D> + <key:K>
    Duration:   plain and single-dotted binary glyphs + triplet family
    Pitch:      kern letters x accidentals x octaves + r
    Voice:      repeatable <v> addresses; per-bar width is max depth + 1

Two time regimes: glyph-valued durations (dyadic + triplet family) are
measured events interleaved in true time order; content past the measured
boundary is a hidden-time capsule <tup> <grid> ... </tup> whose interior
is equal-division slots walled inside one beat unit.
"""

import re
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple


class _LocatedTokenizerError(ValueError):
    """A tokenizer refusal with an optional data-bar address."""

    def __init__(
        self,
        message: str,
        *,
        bar_index: Optional[int] = None,
        bar_label: Optional[str] = None,
        bar_ordinal: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.bar_index = bar_index
        self.bar_label = bar_label
        self.bar_ordinal = bar_ordinal


class TokenizerOOVError(_LocatedTokenizerError):
    """A score element has no representation in the current vocabulary."""


class IncompleteTupletError(_LocatedTokenizerError):
    """A tuplet run cannot be closed as a complete representable group."""


class MetricTimelineError(_LocatedTokenizerError):
    """A voice timeline does not tile its meter-governed bar exactly."""


_TYPED_TOKENIZER_ERRORS = (
    TokenizerOOVError,
    IncompleteTupletError,
    MetricTimelineError,
)


def _locate_tokenizer_error(
    error: ValueError,
    *,
    bar_index: Optional[int],
    bar_label: Optional[str],
    bar_ordinal: Optional[int],
) -> ValueError:
    """Attach the parser's known bar address without changing the message."""
    if getattr(error, "bar_index", None) is None:
        error.bar_index = bar_index
    if getattr(error, "bar_label", None) is None:
        error.bar_label = bar_label
    if getattr(error, "bar_ordinal", None) is None:
        error.bar_ordinal = bar_ordinal
    return error


def _with_tokenizer_context(
    context: str,
    error: ValueError,
    *,
    bar_index: Optional[int] = None,
    bar_label: Optional[str] = None,
    bar_ordinal: Optional[int] = None,
) -> ValueError:
    error_type = (
        type(error)
        if isinstance(error, _TYPED_TOKENIZER_ERRORS)
        else ValueError
    )
    contextual = error_type(f"{context}: {error}")
    return _locate_tokenizer_error(
        contextual,
        bar_index=(bar_index if bar_index is not None
                   else getattr(error, "bar_index", None)),
        bar_label=(bar_label if bar_label is not None
                   else getattr(error, "bar_label", None)),
        bar_ordinal=(bar_ordinal if bar_ordinal is not None
                     else getattr(error, "bar_ordinal", None)),
    )

# =============================================================================
# Duration Grid (128th-note resolution)
# =============================================================================

GRID = 128

# Triplet-family tokens carry real durations that are non-integral on the
# 128th grid; exact Fractions keep the account honest.  Integral Fractions
# hash like ints, so mixed keys stay interchangeable in these tables.
RECIP_TO_GRID: Dict[str, "int | Fraction"] = {
    "128": 1, "64": 2, "64.": 3, "32": 4, "32.": 6,
    "16": 8, "16.": 12, "8": 16, "8.": 24,
    "4": 32, "4.": 48, "2": 64, "2.": 96,
    "1": 128, "1.": 192,
    # measured triplet family (real values: 12 = 1/12 whole note)
    "6": Fraction(64, 3), "12": Fraction(32, 3),
    "24": Fraction(16, 3), "48": Fraction(8, 3),
}

GRID_TO_RECIP: Dict["int | Fraction", str] = {
    v: k for k, v in RECIP_TO_GRID.items()
}

# =============================================================================
# Vocabulary Definition
# =============================================================================

SPECIAL_TOKENS: Dict[str, int] = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
}

STRUCTURAL_TOKENS_LIST: List[str] = [
    "<bar>", "<grid>",
    "<pl>", "</pl>",
    "<pr>", "</pr>",
    "<tup>", "</tup>",
    "<tie>", "</tie>",
]

# Repeating <v> before an ordinary hand bracket selects a deeper voice; the
# maximum depth used in a bar is that hand's complete written width.
VOICE_TOKENS_LIST: List[str] = [
    "<v>",
]

METER_NUM_TOKENS: List[str] = [
    "<num:2>", "<num:3>", "<num:4>",
    "<num:6>", "<num:9>", "<num:12>",
]
METER_DEN_TOKENS: List[str] = [
    "<den:2>", "<den:4>", "<den:8>", "<den:16>",
]
_VALID_NUM: Dict[str, str] = {t[5:-1]: t for t in METER_NUM_TOKENS}   # "3" -> "<num:3>"
_VALID_DEN: Dict[str, str] = {t[5:-1]: t for t in METER_DEN_TOKENS}   # "4" -> "<den:4>"

# Schema tokens: key signatures (circle of fifths)
KEY_TOKENS: List[str] = [
    "<key:0>",                                          # C major / A minor
    "<key:1#>", "<key:2#>", "<key:3#>", "<key:4#>",    # sharps
    "<key:5#>", "<key:6#>", "<key:7#>",
    "<key:1b>", "<key:2b>", "<key:3b>", "<key:4b>",    # flats
    "<key:5b>", "<key:6b>", "<key:7b>",
]

# Mapping from kern key signature to token
_KEY_SIG_TO_TOKEN: Dict[str, str] = {
    '*k[]': '<key:0>',
    '*k[f#]': '<key:1#>', '*k[f#c#]': '<key:2#>', '*k[f#c#g#]': '<key:3#>',
    '*k[f#c#g#d#]': '<key:4#>', '*k[f#c#g#d#a#]': '<key:5#>',
    '*k[f#c#g#d#a#e#]': '<key:6#>', '*k[f#c#g#d#a#e#b#]': '<key:7#>',
    '*k[b-]': '<key:1b>', '*k[b-e-]': '<key:2b>', '*k[b-e-a-]': '<key:3b>',
    '*k[b-e-a-d-]': '<key:4b>', '*k[b-e-a-d-g-]': '<key:5b>',
    '*k[b-e-a-d-g-c-]': '<key:6b>', '*k[b-e-a-d-g-c-f-]': '<key:7b>',
}
# Reverse mapping: token to kern key signature
_TOKEN_TO_KEY_SIG: Dict[str, str] = {v: k for k, v in _KEY_SIG_TO_TOKEN.items()}

# Duration tokens (plain + single-dotted dyadic glyphs + triplet family)
DURATION_TOKENS: List[str] = [
    # Binary
    "1", "2", "4", "8", "16", "32", "64", "128",
    # Dotted
    "1.", "2.", "4.", "8.", "16.", "32.", "64.",
    # Measured triplet family (dotted triplet values split in-family:
    # 12. -> 12 tied to 24; 96 and 3 route outside the measured regime)
    "6", "12", "24", "48",
]

# Pitch tokens - Humdrum kern notation
# Octave notation: CC=C2, C=C3, c=C4, cc=C5, ccc=C6, cccc=C7
PITCH_LETTERS: List[str] = ["C", "D", "E", "F", "G", "A", "B"]
ACCIDENTALS: List[str] = ["-", "", "#"]  # flat, none, sharp


def generate_pitch_tokens() -> List[str]:
    """7 letters x 3 accidentals x 9 octaves + rest = 190 tokens."""
    pitches: List[str] = []
    for n in range(4, 0, -1):
        for letter in PITCH_LETTERS:
            for acc in ACCIDENTALS:
                pitches.append(letter * n + acc)
    for n in range(1, 6):
        for letter in PITCH_LETTERS:
            for acc in ACCIDENTALS:
                pitches.append(letter.lower() * n + acc)
    pitches.append("r")
    return pitches


PITCH_TOKENS: List[str] = generate_pitch_tokens()


# =============================================================================
# Build Vocabulary
# =============================================================================

def build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build the token-to-id mapping from the category lists."""
    token_to_id: Dict[str, int] = {}
    idx = 0

    for token, special_idx in SPECIAL_TOKENS.items():
        token_to_id[token] = special_idx
        idx = max(idx, special_idx + 1)

    for token in STRUCTURAL_TOKENS_LIST:
        token_to_id[token] = idx
        idx += 1

    for token in METER_NUM_TOKENS + METER_DEN_TOKENS + KEY_TOKENS:
        token_to_id[token] = idx
        idx += 1

    for token in DURATION_TOKENS:
        token_to_id[token] = idx
        idx += 1

    for token in PITCH_TOKENS:
        token_to_id[token] = idx
        idx += 1

    for token in VOICE_TOKENS_LIST:
        token_to_id[token] = idx
        idx += 1

    id_to_token = {v: k for k, v in token_to_id.items()}
    return token_to_id, id_to_token


VOCAB, ID_TO_TOKEN = build_vocab()
VOCAB_SIZE = len(VOCAB)


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

    Wall lines (barline, midline, comp-beat lines) block
    shapes whose anchor does not strictly outrank the wall.
    """

    SUPPORTED_NUM = frozenset({2, 3, 4, 6, 9, 12})
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

    def grid_positions(self) -> List[int]:
        """Meter-unit gridlines: num cells, each of duration 128/den."""
        step = GRID // self.den
        return [i * step for i in range(self.num)]

    def beat_count(self) -> int:
        """Felt-pulse count: N/3 in compound meters (N in 6/9/12), else N."""
        return self.num // 3 if self.num in (6, 9, 12) else self.num

    def beat_positions(self) -> List[int]:
        """Derived beat-unit lattice (dotted unit in compound meters)."""
        n_beats = self.beat_count()
        step = self.bar_length // n_beats
        return [i * step for i in range(n_beats)]


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

    # Token-first law (NOTES): any glyph-valued duration is one token,
    # position-blind; the print layer re-splits for beat visibility.
    # Rests are exempt and use spell_positioned below.
    if glyph is not None:
        return [glyph]

    # No token: cut at the biggest interior line, recurse
    cut = tree.biggest_interior_line(t, d)
    if cut is None:
        raise ValueError(
            f"spell: no interior line in [{t}, {t + d}) for "
            f"{tree.num}/{tree.den}"
        )
    left = spell(t, cut - t, tree)
    right = spell(cut, t + d - cut, tree)
    return left + right


def spell_positioned(t: int, d: int, tree: MetricTree) -> List[str]:
    """Position-dependent spelling: cell fill, licensed shape, then cut.

    This is the metric-tree law — the print layer's re-split, and the
    spelling law for RESTS (rests never take the token-first shortcut).
    """
    if d <= 0:
        raise ValueError(f"spell_positioned: d must be positive, got {d}")

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
            f"spell_positioned: no interior line in [{t}, {t + d}) for "
            f"{tree.num}/{tree.den}"
        )
    left = spell_positioned(t, cut - t, tree)
    right = spell_positioned(cut, t + d - cut, tree)
    return left + right


# =============================================================================
# Verification
# =============================================================================

def verify_spell_exhaustive() -> Dict[str, object]:
    """Exhaustively verify spell() for every (t, d) on the 1/128 grid.

    Checks totality (every pair returns a result), single-valuedness
    (deterministic), and consistency (returned tokens sum to d).
    """
    results: Dict[str, object] = {"pass": True, "meters": {}, "total_pairs": 0}

    for num in sorted(MetricTree.SUPPORTED_NUM):
        for den in sorted(MetricTree.SUPPORTED_DEN):
            tree = get_metric_tree(num, den)
            BL = tree.bar_length
            errors: List[str] = []
            count = 0
            max_splits = 0

            for t in range(BL):
                for d in range(1, BL - t + 1):
                    count += 1
                    try:
                        glyphs = spell(t, d, tree)
                    except Exception as e:
                        errors.append(f"  t={t} d={d}: {e}")
                        continue

                    total = sum(RECIP_TO_GRID[g] for g in glyphs)
                    if total != d:
                        errors.append(
                            f"  t={t} d={d}: sum={total} != d "
                            f"(glyphs={glyphs})"
                        )

                    for g in glyphs:
                        if g not in RECIP_TO_GRID:
                            errors.append(
                                f"  t={t} d={d}: unknown glyph {g!r}"
                            )

                    max_splits = max(max_splits, len(glyphs))

            meter_key = f"{num}/{den}"
            results["meters"][meter_key] = {
                "bar_length": BL,
                "pairs": count,
                "max_splits": max_splits,
                "errors": errors[:20],
            }
            results["total_pairs"] += count
            if errors:
                results["pass"] = False

    return results


# =============================================================================
# Walkthrough Verification
# =============================================================================

def verify_walkthroughs() -> List[str]:
    """Check every walkthrough example from the spec.  Returns error list."""
    errors: List[str] = []
    t44 = get_metric_tree(4, 4)
    t34 = get_metric_tree(3, 4)

    cases = [
        (t44, 0, 128, ["1"], "whole note in 4/4"),
        (t44, 32, 64, ["2"], "half from beat 2 of 4/4 (token-first)"),
        (t44, 24, 40, ["16", "4"], "5/16 from last 16th of beat 1"),
        (t44, 0, 40, ["4", "16"], "5/16 from beat head (front-heavy cut)"),
        (t44, 0, 48, ["4."], "4. from beat 1 of 4/4"),
        (t44, 32, 48, ["4."], "3/8 from beat 2 of 4/4 (token-first)"),
        (t44, 32, 56, ["4", "8."], "7/16 from beat 2 (no double-dot token)"),
        (t44, 0, 96, ["2."], "2. from beat 1 of 4/4"),
        (t34, 0, 64, ["2"], "waltz half note (beats 1-2 of 3/4)"),
        (t34, 32, 64, ["2"], "mirror half note (beats 2-3 of 3/4)"),
    ]

    for tree, t, d, expected, label in cases:
        result = spell(t, d, tree)
        if result != expected:
            errors.append(f"{label}: expected {expected}, got {result}")

    return errors


# =============================================================================
# Voice-bracket serialization (kern_gt → token sequence)
# =============================================================================

_RECIP_RE = re.compile(r"(\d+)(?:%(\d+))?(\.*)")
_TIE_START_RE = re.compile(r"^\[")
_TIE_END_RE = re.compile(r"\]")
_PITCH_RE = re.compile(r"(([A-G])\2*|([a-g])\3*)[#\-n]*|r")
# q/Q are true graces (no metric duration).  P is an appoggiatura
# designation on a note that DOES occupy metric time — dropping it
# punches a hole in the bar.
_GRACE_RE = re.compile(r"[qQ]")


def _parse_note_token(tok: str):
    """Parse one kern data token.

    Returns:
      ('note', grid_dur, pitch, tie_in, tie_out)      — glyph-valued duration
                                                        (dyadic or triplet family;
                                                        grid_dur is int or Fraction)
      ('tri_dotted', base_grid, pitch, tie_in, tie_out) — dotted triplet value:
                                                        splits in-family as
                                                        base tied to base/2
      ('tup', recip, pitch, tie_in, tie_out, dur)     — capsule member
      None                                            — unparseable
    """
    if not tok or tok == ".":
        return None
    if _GRACE_RE.search(tok):
        raise TokenizerOOVError(
            f"grace token survived standardization: {tok!r}"
        )
    tie_out = bool(_TIE_START_RE.search(tok))
    tie_in = bool(_TIE_END_RE.search(tok))
    if "_" in tok:
        # Kern tie continuation: the middle fragment of a 3+ fragment
        # chain both lands and departs.
        tie_in = True
        tie_out = True
    m = _RECIP_RE.search(tok)
    if not m:
        raise TokenizerOOVError(f"note has no readable duration: {tok!r}")
    pm = _PITCH_RE.search(tok)
    if not pm:
        raise TokenizerOOVError(f"note has no readable pitch: {tok!r}")
    pitch = pm.group(0)
    if not pitch:
        raise ValueError(f"note has empty pitch: {tok!r}")
    if "n" in pitch:
        raise ValueError(f"natural sign survived standardization: {tok!r}")
    if "#" in pitch and "-" in pitch:
        raise ValueError(f"contradictory accidental spelling: {tok!r}")
    if pitch == "r" and (tie_in or tie_out):
        raise ValueError(f"tie mark on rest is not allowed: {tok!r}")
    F = Fraction
    recip_num = int(m.group(1))
    if recip_num == 0:
        # Breve is out of vocabulary: fail loudly so the caller skips the
        # chunk instead of silently losing the note.
        raise TokenizerOOVError(
            f"breve duration not in vocabulary: {tok!r}"
        )
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
        grid_int = int(grid)
        if grid_int not in GRID_TO_RECIP:
            raise ValueError(
                f"duration spelling is not canonical: {tok!r}"
            )
        return ("note", grid_int, pitch, tie_in, tie_out)
    if base_grid in GRID_TO_RECIP:
        # Measured triplet family: a real duration token, an ordinary
        # event interleaved in true time order.
        if dots == 0:
            return ("note", base_grid, pitch, tie_in, tie_out)
        if dots == 1:
            half = base_grid / 2
            if half not in GRID_TO_RECIP:
                raise TokenizerOOVError(
                    f"dotted triplet value splits outside the "
                    f"vocabulary: {tok!r}"
                )
            return ("tri_dotted", base_grid, pitch, tie_in, tie_out)
        raise TokenizerOOVError(
            f"multi-dotted triplet value not representable: {tok!r}"
        )
    # Past the measured boundary: hidden-time capsule member.
    return ("tup", F(recip_num, recip_den), pitch, tie_in, tie_out, dur)




def _flush_tuplet(accum: dict, events: List[dict],
                  beat_step: Optional[Fraction] = None) -> None:
    """Emit hidden-time capsules from the accumulator.

    A group closes at the first glyph-valued span (minimal closure) and
    never crosses a beat line: a member straddling the line is split
    there with a boundary tie.  A closed multi-member group must be an
    equal division of its span with ties at its edges only; single
    members and all-rest groups degrade to ordinary events.
    """
    F = Fraction
    members = accum["members"]
    if not members:
        return

    onset = F(accum["onset"])
    hand = accum["hand"]

    buf: list = []
    buf_dur = F(0)

    def _closable(span: F) -> bool:
        # Minimal DYADIC closure: a capsule span is measured against the
        # dyadic lattice; triplet-valued tokens never close a capsule.
        return span.denominator == 1 and int(span) in GRID_TO_RECIP

    def _flush_group():
        nonlocal onset, buf_dur
        if not buf:
            return
        total = buf_dur
        dur_val = int(total) if total.denominator == 1 else total

        # An all-rest group carries no gesture identity: canonical
        # notation for a glyph-valued silence is the plain rest.
        if all(
                pitch == "r"
                for member in buf
                for pitch in member["pitches"]):
            events.append({
                "onset": onset, "dur": dur_val, "hand": hand,
                "chain": accum.get("chain", 0),
                "pitches": ["r"], "tie_ins": [False],
                "tie_outs": [False],
            })
            onset += total
            buf.clear()
            buf_dur = F(0)
            return

        # One member that fills a glyph-valued span is not a capsule;
        # retain its attack/tie identity as an ordinary event.
        if len(buf) == 1:
            member = buf[0]
            events.append({
                "onset": onset, "dur": dur_val, "hand": hand,
                "chain": accum.get("chain", 0),
                "pitches": list(member["pitches"]),
                "tie_ins": list(member["tie_ins"]),
                "tie_outs": list(member["tie_outs"]),
                "source_row": member.get("source_row"),
            })
            onset += total
            buf.clear()
            buf_dur = F(0)
            return

        n_slots = len(buf)
        slot_dur = total / n_slots
        for index, member in enumerate(buf):
            if member["dur"] != slot_dur:
                raise IncompleteTupletError(
                    f"capsule members are not an equal division: member "
                    f"{index + 1} of {n_slots} spans {member['dur']} "
                    f"against slot {slot_dur}"
                )
            if index > 0 and any(member["tie_ins"]):
                raise IncompleteTupletError(
                    "capsule tie lands on an interior member; ties "
                    "attach at capsule edges only"
                )
            if index < n_slots - 1 and any(member["tie_outs"]):
                raise IncompleteTupletError(
                    "capsule tie departs from an interior member; ties "
                    "attach at capsule edges only"
                )
        slots = [
            {"attack": list(member["pitches"]),
             "attack_ti": list(member["tie_ins"])}
            for member in buf
        ]
        events.append({
            "onset": onset, "dur": dur_val, "hand": hand,
            "chain": accum.get("chain", 0),
            "is_tup": True, "tup_n": n_slots, "slots": slots,
            "tie_out": any(buf[-1]["tie_outs"]),
        })
        onset += total
        buf.clear()
        buf_dur = F(0)

    for member in members:
        rem = dict(member)
        rem_dur = F(rem["dur"])
        while True:
            pos = onset + buf_dur
            if beat_step is not None:
                line = (pos // beat_step + 1) * beat_step
                if pos + rem_dur > line:
                    part = line - pos
                    buf.append({
                        "pitches": list(rem["pitches"]),
                        "tie_ins": list(rem["tie_ins"]),
                        "tie_outs": [p != "r" for p in rem["pitches"]],
                        "dur": part,
                    })
                    buf_dur += part
                    if not _closable(buf_dur):
                        raise IncompleteTupletError(
                            f"capsule span {buf_dur} closing at the beat "
                            f"line is not a duration token"
                        )
                    _flush_group()
                    rem = {
                        "pitches": list(rem["pitches"]),
                        "tie_ins": [p != "r" for p in rem["pitches"]],
                        "tie_outs": list(rem["tie_outs"]),
                        "dur": rem_dur - part,
                    }
                    rem_dur = rem_dur - part
                    continue
            buf.append(dict(rem, dur=rem_dur))
            buf_dur += rem_dur
            break
        if _closable(buf_dur):
            _flush_group()
        elif (beat_step is not None
                and (onset + buf_dur) % beat_step == 0):
            raise IncompleteTupletError(
                f"capsule span {buf_dur} closing at the beat line is "
                f"not a duration token"
            )

    if buf:
        raise IncompleteTupletError(
            f"capsule leftover does not reach a duration token: "
            f"{buf_dur} grid units"
        )


def _parse_bar_events(data_lines: List[str],
                      row_channels: Optional[List[list]] = None,
                      row_rebinds: Optional[List[Optional[dict]]] = None,
                      beat_step: Optional[Fraction] = None,
                      ) -> List[dict]:
    """Parse kern data lines for one bar into event dicts.

    Per-channel time tracking: a channel is `(hand, depth)`, where depth is
    the current left-to-right physical address.
    Each channel advances independently by its own min duration, which
    handles polymetric passages (e.g. triplets vs dotted notes across
    channels).  Capsule members accumulate into one is_tup event whose
    total span is glyph-valued and beat-walled (`beat_step`, when known).

    Without row_channels, columns 0/1 are the two hands' single lines
    (the flattened two-spine layout).  With it, each row's columns map to
    channels; a newly split channel starts at the adjacent channel's clock, and
    after a merge the surviving channel resumes at the latest member clock.
    """
    F = Fraction
    events: List[dict] = []
    spine_time: Dict[tuple, F] = {}
    tup_accum: Dict[tuple, Optional[dict]] = {}
    default_map = [(0, 0), (1, 0)]

    def _base(ch: tuple) -> tuple:
        return (ch[0], 0)

    def _close_tup(ch: tuple):
        acc = tup_accum.get(ch)
        if acc is None:
            return
        n_before = len(events)
        _flush_tuplet(acc, events, beat_step)
        n_after = len(events)
        if n_after > n_before:
            last = events[-1]
            spine_time[ch] = last["onset"] + last["dur"]
        tup_accum[ch] = None

    prev_present: Optional[set] = None
    for ri, line in enumerate(data_lines):
        parts = line.split("\t")
        chans = row_channels[ri] if row_channels is not None else default_map
        if len(parts) != len(chans):
            raise ValueError(
                f"data row {ri} has {len(parts)} fields for "
                f"{len(chans)} active spines"
            )

        present = {c for c in chans if c is not None}
        for ch in present:
            spine_time.setdefault(ch, F(0))
            tup_accum.setdefault(ch, None)
        if prev_present is not None:
            for ch in sorted(present - prev_present):
                spine_time[ch] = spine_time.get(_base(ch), F(0))
                tup_accum[ch] = None
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

        # A kern data record has one global onset.  A null token is only
        # legal while that voice's preceding event still covers the
        # beat; accepting a null on a ready voice would silently move
        # its next attack earlier and hide a missing explicit rest.
        row_time = min((spine_time[ch] for ch in present), default=F(0))
        for ci, ch in enumerate(chans):
            if ch is None:
                continue
            cell = parts[ci].strip()
            if cell == ".":
                if spine_time[ch] <= row_time:
                    raise MetricTimelineError(
                        f"voice_gap: channel {ch} is uncovered at "
                        f"data row {ri}"
                    )
            elif spine_time[ch] != row_time:
                raise MetricTimelineError(
                    f"voice overlap: channel {ch} has a new cell at "
                    f"data row {ri} before its clock is ready"
                )

        for ci in range(len(chans)):
            ch = chans[ci]
            if ch is None:
                continue
            sp = parts[ci].strip()
            if sp == ".":
                continue
            toks = sp.split()
            parsed = [_parse_note_token(t) for t in toks]
            parsed = [p for p in parsed if p is not None]
            if not parsed:
                continue

            note_list = [p for p in parsed if p[0] == "note"]
            tri_list = [p for p in parsed if p[0] == "tri_dotted"]
            tup_list = [p for p in parsed if p[0] == "tup"]

            pitched_notes = [p for p in note_list if p[2] != "r"]
            duration_groups: List[Tuple[int, List[str]]] = []
            for note in pitched_notes:
                if not duration_groups or duration_groups[-1][0] != note[1]:
                    duration_groups.append((note[1], []))
                duration_groups[-1][1].append(note[2])
            duration_order = [duration for duration, _ in duration_groups]
            if duration_order != sorted(duration_order, reverse=True):
                raise ValueError(
                    f"chord durations not canonical at data row {ri}"
                )
            chord_groups = [pitches for _duration, pitches
                            in duration_groups]
            if tri_list:
                chord_groups.append([p[2] for p in tri_list])
            if tup_list:
                chord_groups.append([p[2] for p in tup_list])
            if any(pitches != sorted(pitches, key=kern_pitch_to_midi)
                   for pitches in chord_groups):
                raise ValueError(
                    f"chord pitches not canonical at data row {ri}"
                )

            # A measured event on a channel with a pending capsule closes
            # the capsule first; a leftover that reaches no duration token
            # is refused there.
            if (note_list or tri_list) and not tup_list \
                    and tup_accum[ch] is not None:
                _close_tup(ch)

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
                    }
                member_pitches = [p[2] for p in tup_list]
                member_ti = [p[3] for p in tup_list]
                member_to = [p[4] for p in tup_list]
                member_dur = max(p[5] for p in tup_list) * GRID
                acc_members = tup_accum[ch]["members"]
                # Tied same-pitch member runs are legitimate: duplicate
                # resolution slices sustains at the union onset grid, and
                # inside a tuplet those slices stay separate members.
                acc_members.append({
                    "pitches": member_pitches,
                    "tie_ins": member_ti,
                    "tie_outs": member_to,
                    "dur": member_dur,
                    "onset": spine_time[ch],
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
                            "source_row": ri,
                            "pitches": [], "tie_ins": [],
                            "tie_outs": [],
                        }
                    by_dur[key]["pitches"].append(pitch)
                    by_dur[key]["tie_ins"].append(ti)
                    by_dur[key]["tie_outs"].append(to)
                for ev in by_dur.values():
                    zipped = sorted(
                        zip(ev["pitches"], ev["tie_ins"], ev["tie_outs"]),
                        key=lambda x: kern_pitch_to_midi(x[0]),
                    )
                    ev["pitches"] = [z[0] for z in zipped]
                    ev["tie_ins"] = [z[1] for z in zipped]
                    ev["tie_outs"] = [z[2] for z in zipped]
                    events.append(ev)

            if tri_list:
                # A dotted triplet value splits inside its family with a
                # tie: 12. -> 12 tied to 24 (front-heavy, the double-dot
                # precedent).  Rests split without tie marks.
                by_base: Dict[tuple, dict] = {}
                rest_seq = 0
                for _, base_grid, pitch, ti, to in tri_list:
                    if pitch == "r":
                        key = (base_grid, rest_seq)
                        rest_seq += 1
                    else:
                        key = (base_grid, -1)
                    entry = by_base.setdefault(key, {
                        "pitches": [], "tie_ins": [], "tie_outs": [],
                    })
                    entry["pitches"].append(pitch)
                    entry["tie_ins"].append(ti)
                    entry["tie_outs"].append(to)
                for (base_grid, _seq), entry in by_base.items():
                    zipped = sorted(
                        zip(entry["pitches"], entry["tie_ins"],
                            entry["tie_outs"]),
                        key=lambda x: kern_pitch_to_midi(x[0]),
                    )
                    pitches = [z[0] for z in zipped]
                    seam = [p != "r" for p in pitches]
                    events.append({
                        "onset": spine_time[ch], "dur": base_grid,
                        "hand": ch[0], "chain": ch[1], "source_row": ri,
                        "pitches": pitches,
                        "tie_ins": [z[1] for z in zipped],
                        "tie_outs": seam,
                    })
                    events.append({
                        "onset": spine_time[ch] + base_grid,
                        "dur": base_grid / 2,
                        "hand": ch[0], "chain": ch[1], "source_row": ri,
                        "pitches": list(pitches),
                        "tie_ins": list(seam),
                        "tie_outs": [z[2] for z in zipped],
                    })

            # One time advance per cell, over ALL parsed items.
            cell_durs = []
            for p in parsed:
                if p[0] == "note":
                    cell_durs.append(F(p[1]))
                elif p[0] == "tri_dotted":
                    cell_durs.append(p[1] * F(3, 2))
                else:
                    cell_durs.append(p[5] * GRID)
            if cell_durs:
                spine_time[ch] += min(cell_durs)

    for ch in list(tup_accum):
        _close_tup(ch)

    return events


def _absorb_spelling_runs(
    events: List[dict],
    tree: Optional[MetricTree] = None,
    structural_rows: Optional[Set[int]] = None,
) -> List[dict]:
    """Merge glyph-valued in-bar tie chains back to single values.

    Token-first readback (NOTES only): a maximal whole-chord tie chain
    whose total is a duration token of the SAME family collapses to one
    event — this also absorbs same-pitch collision seams left by the
    voice-merge machinery.  Family typing is preserved: triplet segments
    never merge into a dyadic value, so the in-family dotted split (12
    tied to 24) survives as a tie pair.  RESTS are never absorbed: rest
    spelling always follows the metric tree (spell_positioned).
    """
    structural_rows = structural_rows or set()

    def crosses_structure(right: dict) -> bool:
        row = right.get("source_row")
        return row is not None and row in structural_rows

    def mergeable_total(run: List[dict]):
        durations = [Fraction(event["dur"]) for event in run]
        total = sum(durations)
        if total not in GRID_TO_RECIP:
            return None
        integral = [d.denominator == 1 for d in durations]
        if all(integral):
            return int(total)
        if not any(integral) and total.denominator != 1:
            return total
        return None

    by_channel: Dict[tuple, List[dict]] = {}
    passthrough: List[dict] = []
    for event in events:
        if event.get("is_tup") or event["pitches"] == ["r"]:
            passthrough.append(event)
        else:
            by_channel.setdefault(
                (event["hand"], event.get("chain", 0)), []
            ).append(event)

    result: List[dict] = list(passthrough)
    for channel_events in by_channel.values():
        ordered = sorted(
            channel_events,
            key=lambda event: (Fraction(event["onset"]),
                               Fraction(event["dur"])),
        )
        index = 0
        while index < len(ordered):
            first = ordered[index]
            run = [first]
            while index + len(run) < len(ordered):
                prev = run[-1]
                nxt = ordered[index + len(run)]
                if (nxt["pitches"] != first["pitches"]
                        or prev["onset"] + prev["dur"] != nxt["onset"]
                        or crosses_structure(nxt)
                        or not prev.get("tie_outs")
                        or not all(prev["tie_outs"])
                        or not nxt.get("tie_ins")
                        or not all(nxt["tie_ins"])):
                    break
                run.append(nxt)
            if any(first.get("tie_ins", [])) \
                    and not all(first.get("tie_ins", [])):
                # A partial-tie head is a structural slice, never the
                # first glyph of a whole-chord value.
                run = [first]
            total = mergeable_total(run) if len(run) > 1 else None
            if total is None:
                result.extend(run)
            else:
                last = run[-1]
                result.append(dict(
                    first,
                    dur=total,
                    tie_ins=list(first.get("tie_ins", [])),
                    tie_outs=list(last.get("tie_outs", [])),
                ))
            index += len(run)
    return result


def _validate_canonical_spelling(
    events: List[dict],
    tree: MetricTree,
    structural_rows: Optional[Set[int]] = None,
    *,
    assigned_voices: bool = False,
) -> None:
    """Validate producer-owned duration spelling without rewriting events."""
    structural_rows = structural_rows or set()

    # Family lock (one content, one form): a note on the triplet grid
    # must use a family token; the same-length dyadic spelling is illegal
    # at that position.
    for event in events:
        if event.get("is_tup"):
            continue
        if (Fraction(event["dur"]).denominator == 1
                and Fraction(event["onset"]).denominator != 1):
            raise ValueError(
                f"cross-family spelling: dyadic value at triplet-grid "
                f"onset {event['onset']}"
            )

    def expected(onset: int, duration: int) -> Optional[List[int]]:
        # Rest spans keep the producer's single-glyph convention; the
        # wall-legality of lone rest glyphs is an open spec question.
        if duration in GRID_TO_RECIP:
            return [duration]
        # Rests never take the token-first shortcut: multi-glyph silence
        # follows the metric-tree spelling.
        try:
            return [
                RECIP_TO_GRID[g]
                for g in spell_positioned(onset, duration, tree)
            ]
        except ValueError:
            return None

    def expected_spelled(onset: int, duration: int) -> Optional[List[int]]:
        # Token-first (notes): a glyph-valued duration is one token
        # anywhere; only tokenless values carry a metric-tree cut list.
        try:
            return [RECIP_TO_GRID[g] for g in spell(onset, duration, tree)]
        except ValueError:
            return None

    def outside_dyadic_account(run: List[dict]) -> bool:
        # Triplet-family segments are governed by the in-family split
        # rule (12. -> 12 tied to 24), not the metric-tree cut list; the
        # grid arithmetic of the bar account guards them instead.
        return any(
            Fraction(event["dur"]).denominator != 1
            or Fraction(event["onset"]).denominator != 1
            for event in run
        )

    def crosses_structure(left: dict, right: dict) -> bool:
        row = right.get("source_row")
        return row is not None and row in structural_rows

    by_channel: Dict[tuple, List[dict]] = {}
    for event in events:
        if not event.get("is_tup"):
            channel = (event["hand"], event.get("chain", 0))
            if assigned_voices:
                channel = (event["hand"], event.get("voice") or 0)
            by_channel.setdefault(
                channel, []
            ).append(event)

    for channel, channel_events in by_channel.items():
        ordered = sorted(
            channel_events,
            key=lambda event: (event["onset"], event["dur"]),
        )

        index = 0
        while index < len(ordered):
            first = ordered[index]
            if first["pitches"] != ["r"]:
                index += 1
                continue
            run = [first]
            while index + len(run) < len(ordered):
                nxt = ordered[index + len(run)]
                prev = run[-1]
                if (nxt["pitches"] != ["r"]
                        or prev["onset"] + prev["dur"] != nxt["onset"]
                        or crosses_structure(prev, nxt)):
                    break
                run.append(nxt)
            actual = [event["dur"] for event in run]
            if outside_dyadic_account(run):
                index += len(run)
                continue
            total = sum(actual)
            canon = expected(first["onset"], total)
            if canon is not None and actual != canon:
                raise ValueError(
                    f"adjacent rests are not canonical in channel {channel}"
                )
            index += len(run)

        index = 0
        while index < len(ordered):
            first = ordered[index]
            if first["pitches"] == ["r"]:
                index += 1
                continue
            first_tie_ins = first.get("tie_ins", [])
            if any(first_tie_ins) and not all(first_tie_ins):
                # A changed chord membership is a structural slice, not
                # the first glyph of a whole-chord spelling run.
                index += 1
                continue
            run = [first]
            while index + len(run) < len(ordered):
                prev = run[-1]
                nxt = ordered[index + len(run)]
                if (nxt["pitches"] == ["r"]
                        or nxt["pitches"] != first["pitches"]
                        or prev["onset"] + prev["dur"] != nxt["onset"]
                        or not prev.get("tie_outs")
                        or not all(prev["tie_outs"])
                        or not nxt.get("tie_ins")
                        or not all(nxt["tie_ins"])
                        or crosses_structure(prev, nxt)):
                    break
                run.append(nxt)
            actual = [event["dur"] for event in run]
            if (len(run) > 1 and not any(run[-1].get("tie_outs", []))
                    and not outside_dyadic_account(run)):
                total = sum(actual)
                canon = expected_spelled(first["onset"], total)
                if canon is not None and actual != canon:
                    raise ValueError(
                        f"tie spelling is not canonical in channel {channel}"
                    )
            index += len(run)


def _validate_channel_tiling(
    events: List[dict], tree: MetricTree, channels: Set[tuple]
) -> None:
    """Require every active voice to tile the declared bar exactly."""
    from fractions import Fraction as F

    by_channel: Dict[tuple, List[dict]] = {channel: [] for channel in channels}
    for event in events:
        by_channel.setdefault(
            (event["hand"], event.get("chain", 0)), []
        ).append(event)

    for channel, channel_events in by_channel.items():
        cursor = F(0)
        for event in sorted(
                channel_events,
                key=lambda item: (F(item["onset"]), F(item["dur"]))):
            onset = F(event["onset"])
            if onset < cursor:
                raise MetricTimelineError(
                    f"voice overlap: channel {channel} starts at {onset} "
                    f"before {cursor}"
                )
            if onset > cursor:
                raise MetricTimelineError(
                    f"voice_gap: channel {channel} is uncovered from "
                    f"{cursor} to {onset}"
                )
            cursor = onset + F(event["dur"])
        if cursor != tree.bar_length:
            raise MetricTimelineError(
                f"voice_gap: channel {channel} ends at {cursor}, "
                f"bar ends at {tree.bar_length}"
            )


def _scan_voice_columns(
    lines: List[str], *, recover_desync: bool = False,
):
    """Read spine topology and map data columns to ordered voice channels.

    Canonical physical fields are already ordered from low to high.  Channel
    zero is therefore the leftmost field of each hand.  Binary split
    parenthesization is deliberately absent from the resulting channel name.
    """
    cols: Optional[list] = None
    line_channels: Dict[int, list] = {}
    cases: List[dict] = []
    clock_flow: Dict[int, Dict[tuple, list]] = {}
    label = "start"
    next_temp = 2
    pending: Optional[dict] = None
    bar_ops: List[dict] = []
    current_ops = {
        "head_splits": {0: 0, 1: 0},
        "tail_merges": {0: 0, 1: 0},
    }
    saw_bar_data = False

    def close_instant(data_idx: Optional[int] = None):
        nonlocal pending
        if pending is None:
            return

        renames: Dict[tuple, tuple] = {}
        kern_fields = [field for field, channel in enumerate(cols)
                       if channel is not None]
        if recover_desync and pending.get("cross_hand") \
                and len(kern_fields) == 2:
            # A source cross-staff contraction can leave both surviving
            # physical spines carrying one scanner hand.  At the two-spine
            # piano baseline their order uniquely restores left/right staff.
            for hand, field in enumerate(kern_fields):
                old = cols[field]
                new = (hand, 0)
                if old != new:
                    renames[old] = new
                cols[field] = new
        for hand in (0, 1):
            fields = [field for field, channel in enumerate(cols)
                      if channel is not None and channel[0] == hand]
            for chain, field in enumerate(fields):
                channel = cols[field]
                new_name = (hand, chain)
                if channel != new_name:
                    renames[channel] = new_name
                cols[field] = new_name

        if data_idx is not None:
            flow: Dict[tuple, list] = {}
            for survivor, members in pending.get("merges", []):
                sources: list = []
                for member in members:
                    sources.extend(flow.pop(member, [member]))
                flow[survivor] = sources

            final: Dict[tuple, list] = {}
            for name, sources in flow.items():
                final[renames.get(name, name)] = sources
            for old, new in renames.items():
                if old not in flow and old != new:
                    final.setdefault(new, [old])
            final = {
                name: sources for name, sources in final.items()
                if not (len(sources) == 1 and sources[0] == name)
            }
            if final:
                clock_flow[data_idx] = final
        pending = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        tokens = stripped.split("\t")

        if stripped.startswith("**"):
            cols = []
            hand = 0
            for token in tokens:
                if token == "**kern":
                    cols.append((hand, 0))
                    hand += 1
                else:
                    cols.append(None)
            if hand != 2:
                cases.append({
                    "kind": "staves",
                    "label": label,
                    "detail": f"{hand} kern staves",
                })
            continue
        if cols is None:
            continue
        if tokens[0].startswith("="):
            if saw_bar_data:
                bar_ops.append(current_ops)
            current_ops = {
                "head_splits": {0: 0, 1: 0},
                "tail_merges": {0: 0, 1: 0},
            }
            saw_bar_data = False
            new_label = tokens[0].lstrip("=").rstrip("-;:|!")
            if new_label:
                label = new_label
            continue
        if stripped.startswith("*"):
            if not any(t in ("*^", "*v", "*-", "*+", "*x")
                       for t in tokens):
                continue
            if len(tokens) != len(cols):
                cases.append({
                    "kind": "desync",
                    "label": label,
                    "detail": f"{len(tokens)} vs {len(cols)}",
                })
                if recover_desync:
                    continue
                return line_channels, cases, clock_flow
            if pending is None:
                pending = {"merges": []}
            if "*x" in tokens:
                cases.append({"kind": "exchange", "label": label})

            new_cols = []
            field = 0
            while field < len(tokens):
                token, channel = tokens[field], cols[field]
                if token == "*^":
                    if channel is not None and not saw_bar_data:
                        current_ops["head_splits"][channel[0]] += 1
                    new_cols.append(channel)
                    if channel is None:
                        new_cols.append(None)
                    else:
                        new_cols.append((channel[0], next_temp))
                        next_temp += 1
                    field += 1
                elif token == "*v":
                    end = field
                    while end < len(tokens) and tokens[end] == "*v":
                        end += 1
                    run = [cols[pos] for pos in range(field, end)]
                    live = [c for c in run if c is not None]
                    if len(run) > 1 and live:
                        if saw_bar_data and len({c[0] for c in live}) == 1:
                            current_ops["tail_merges"][live[0][0]] += \
                                len(live) - 1
                        survivor = min(live, key=lambda c: c[1])
                        new_cols.append(survivor)
                        if len(live) > 1:
                            pending["merges"].append(
                                (survivor, list(live)))
                            if len({channel[0] for channel in live}) > 1:
                                pending["cross_hand"] = True
                    else:
                        new_cols.append(cols[field])
                    field = end
                elif token == "*-":
                    field += 1
                elif token == "*+":
                    if channel is not None:
                        cases.append({
                            "kind": "add_spine",
                            "label": label,
                        })
                    new_cols.extend([channel, None])
                    field += 1
                else:
                    new_cols.append(channel)
                    field += 1
            cols = new_cols
            continue

        close_instant(idx)
        saw_bar_data = True
        line_channels[idx] = list(cols)

    close_instant()
    if saw_bar_data:
        bar_ops.append(current_ops)
    return line_channels, cases, clock_flow, bar_ops
def _is_sounding(ev: dict) -> bool:
    if ev.get("is_tup"):
        return any(p != "r" for s in ev["slots"] for p in s["attack"])
    return ev["pitches"] != ["r"]


def _assign_voices(bars: List[dict]) -> List[dict]:
    """Attach the scanner's ordered leaf address without re-deciding voices."""
    cases: List[dict] = []
    for bar in bars:
        counts = bar["voice_counts"]
        seen = {0: set(), 1: set()}
        for event in bar["events"]:
            hand = event["hand"]
            depth = event.get("chain", 0)
            event["voice"] = depth
            seen[hand].add(depth)
        for hand in (0, 1):
            expected = set(range(counts[hand]))
            if seen[hand] != expected:
                cases.append({
                    "kind": "voice_address",
                    "hand": hand,
                    "label": bar["label"],
                })
    return cases


def _voice_operation_boundary_cases(lines: List[str]) -> List[dict]:
    """Locate source voice operations written inside bar data."""
    label = "pickup"
    bar_ordinal = 0
    saw_data = False
    pending_merges: List[int] = []
    cases: List[dict] = []

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if stripped.startswith("="):
            if saw_data:
                bar_ordinal += 1
            label = stripped.split("\t", 1)[0]
            saw_data = False
            pending_merges = []
            continue
        if stripped.startswith("*"):
            fields = stripped.split("\t")
            if "*^" in fields and saw_data:
                cases.append({
                    "label": label,
                    "bar_ordinal": bar_ordinal,
                    "message": (
                        f"{label}: split/merge should be on the border of "
                        f"bars (*^ at line {index + 1} follows bar data)"
                    ),
                })
            if "*v" in fields:
                pending_merges.append(index + 1)
            continue

        if pending_merges:
            cases.append({
                "label": label,
                "bar_ordinal": bar_ordinal,
                "message": (
                    f"{label}: split/merge should be on the border of bars "
                    f"(*v at line {pending_merges[0]} precedes more bar data)"
                ),
            })
            pending_merges = []
        saw_data = True
    return cases


def _validate_voice_operation_boundaries(lines: List[str]) -> None:
    """Require source voice operations at the edges of their written bar."""
    cases = _voice_operation_boundary_cases(lines)
    if cases:
        raise ValueError(cases[0]["message"])


def _kind_rank(e: dict) -> int:
    """Outermost-first tie-break at equal duration: note >= rest > tup.

    Mandatory nesting forces a tup's slot instants, which fall strictly
    inside any covering span, into that bracket; a rest nests inside an
    equal note (silence has no interior structure to force the other
    way)."""
    if e.get("is_tup"):
        return 2
    return 1 if e["pitches"] == ["r"] else 0


def _serialize_bar(
    events: List[dict], tree: MetricTree, schema_tokens: List[str],
) -> List[str]:
    """Serialize a bar's events into voice-bracket tokens.

    Token position is time: opens at attack and closes at release.
    Same instant: closes due, then opens.
    """
    tokens: List[str] = ["<bar>"] + schema_tokens
    def _address(ev) -> List[str]:
        return ["<v>"] * int(ev.get("voice", 0))

    def _open_tok(ev):
        return "<pl>" if ev["hand"] == 0 else "<pr>"

    def _close_tok(ev):
        return "</pl>" if ev["hand"] == 0 else "</pr>"

    max_end = max((e["onset"] + e["dur"]) for e in events) if events else 0

    # A bar lasts as long as its meter says.  One that stops short puts
    # a barline where the meter does not, and nothing in the stream says
    # where the missing time went — refuse rather than teach that.
    if events and max_end != tree.bar_length:
        raise MetricTimelineError(
            f"bar falls short: holds {max_end} of bar_length "
            f"{tree.bar_length} ({tree.num}/{tree.den})"
        )

    # One channel is one timeline: overlapping events have no kern
    # column to print into and no single-voice reading — refuse rather
    # than emit nested brackets the reader cannot bind.
    by_chan: Dict[tuple, List[dict]] = {}
    for ev in events:
        by_chan.setdefault((ev["hand"], int(ev.get("voice", 0))),
                           []).append(ev)
    for (h, v), evs in by_chan.items():
        evs = sorted(evs, key=lambda e: (e["onset"], e["dur"]))
        for a, b in zip(evs, evs[1:]):
            if a["onset"] + a["dur"] > b["onset"]:
                pa = a.get("pitches") or "a tuplet group"
                pb = b.get("pitches") or "a tuplet group"
                raise MetricTimelineError(
                    f"voice overlap: hand {h} voice {v} holds "
                    f"{pa} through {a['onset'] + a['dur']}"
                    f" while {pb} starts at {b['onset']}")

    # Attack and release clocks uniquely determine the token timeline.
    all_times_set: Set[int] = {ev["onset"] for ev in events}
    for ev in events:
        all_times_set.add(ev["onset"] + ev["dur"])
    all_times = sorted(all_times_set)

    events_at: Dict[int, List[dict]] = {}
    for ev in events:
        events_at.setdefault(ev["onset"], []).append(ev)

    open_brs: List[dict] = []
    def _emit_open(ev):
        # A capsule's slot list is glued to its open (hidden time): the
        # interior carries no stream time, so external events due during
        # the span serialize after the capsule content.
        tokens.extend(_address(ev))
        tokens.append(_open_tok(ev))
        if ev.get("is_tup"):
            tokens.append("<tup>")
            for slot in ev["slots"]:
                tokens.append("<grid>")
                for j, p in enumerate(slot["attack"]):
                    tokens.append(p)
                    if slot["attack_ti"][j]:
                        tokens.append("</tie>")
            tokens.append("</tup>")
        else:
            for i, pitch in enumerate(ev["pitches"]):
                tokens.append(pitch)
                if ev["tie_ins"][i]:
                    tokens.append("</tie>")

    def _emit_close(ev):
        tokens.extend(_address(ev))
        tokens.append(_close_tok(ev))
        tokens.append(GRID_TO_RECIP[ev["dur"]])
        if ev.get("is_tup"):
            if ev.get("tie_out"):
                tokens.append("<tie>")
        elif any(ev.get("tie_outs", [])):
            tokens.append("<tie>")

    for t in all_times:
        # 1. Closes due at t: a close sits at its release instant.
        # LH before RH; within a hand, higher addresses close first.
        expired = [br for br in open_brs if br["onset"] + br["dur"] <= t]
        expired.sort(
            key=lambda e: (e["hand"], -int(e.get("voice", 0)),
                           -e["onset"], e["dur"], -_kind_rank(e))
        )
        for br in expired:
            _emit_close(br)
            open_brs.remove(br)

        # 2. Open new brackets: LH first, longest first, pitch ascending
        if t in events_at:
            # Outermost-first at a shared onset: longest first; at equal
            # duration, _kind_rank orders who stays outside.
            starting = sorted(
                events_at[t],
                key=lambda e: (e["hand"], e.get("voice") or 0,
                               -e["dur"], _kind_rank(e)),
            )
            for ev in starting:
                _emit_open(ev)
                open_brs.append(ev)

    # Close anything still open at bar end
    open_brs.sort(
        key=lambda e: (e["hand"], -int(e.get("voice", 0)),
                       -e["onset"], e["dur"], -_kind_rank(e))
    )
    for br in open_brs:
        _emit_close(br)
    open_brs.clear()

    return tokens


def _event_signature(ev: dict) -> tuple:
    """Token-layer content of one event, comparable across the writer and
    the reader (field spellings differ: dur/tie_outs/attack_to on the
    writer side, dur_grid/tie_out on the reader side).  Per-pitch tie-outs
    project to the bracket-level <tie> the grammar actually carries.
    """
    F = Fraction
    dur = F(ev["dur_grid"] if "dur_grid" in ev else ev["dur"])
    if ev.get("is_tup"):
        slots = ev.get("slots") or []
        tie_out = bool(ev.get("tie_out"))
        slot_sig = tuple(
            (tuple(s.get("attack", [])),
             tuple(bool(x) for x in s.get("attack_ti", [])))
            for s in slots
        )
        return (ev["hand"], int(ev.get("voice", 0)),
                F(ev["onset"]), dur, True,
                slot_sig, tie_out)
    if "tie_outs" in ev:
        tie_out = any(ev["tie_outs"])
    else:
        tie_out = bool(ev.get("tie_out"))
    return (ev["hand"], int(ev.get("voice", 0)),
            F(ev["onset"]), dur, False,
            tuple(ev["pitches"]),
            tuple(bool(t) for t in ev.get("tie_ins", [])),
            tie_out)


def _verify_tokens(tokens: List[str], bars: List[dict]) -> None:
    """Require the complete balanced scope to decode to the same events."""
    from src.score.reconstruct_kern import _bracket_tokens_to_events
    decoded_bars = list(_bracket_tokens_to_events(tokens))
    if len(decoded_bars) != len(bars):
        raise ValueError(
            f"serialized scope decoded into {len(decoded_bars)} bars; "
            f"source has {len(bars)}"
        )
    for source, decoded in zip(bars, decoded_bars):
        want = sorted(_event_signature(e) for e in source["events"])
        got = sorted(_event_signature(e) for e in decoded[0])
        if want == got:
            continue
        for w, g in zip(want, got):
            if w != g:
                raise ValueError(
                    f"tokens decode differently from source: {w} vs {g}"
                )
        raise ValueError(
            f"tokens decode to {len(got)} events, source has {len(want)}"
        )


def _parse_kern_bars(
    kern_content: str, *, isolate_invalid: bool = False,
) -> List[dict]:
    """Parse canonical bars, optionally quarantining invalid bars in place."""
    lines = kern_content.split("\n")
    boundary_cases = _voice_operation_boundary_cases(lines)
    if not isolate_invalid:
        if boundary_cases:
            case = boundary_cases[0]
            ordinal = case.get("bar_ordinal")
            raise _locate_tokenizer_error(
                ValueError(case["message"]),
                bar_index=ordinal,
                bar_label=case.get("label"),
                bar_ordinal=ordinal,
            )
    line_channels, vcases, clock_flow, bar_ops = _scan_voice_columns(
        lines, recover_desync=isolate_invalid)
    if vcases and not isolate_invalid:
        raise ValueError("voice: " + _voice_case_summary(vcases))

    voice_errors: Dict[str, List[str]] = {}
    global_voice_errors: List[str] = []
    if isolate_invalid:
        for case in boundary_cases:
            label = str(case["label"]).lstrip("=").rstrip("-;:|!")
            voice_errors.setdefault(label, []).append(case["message"])
        for case in vcases:
            message = "voice: " + _voice_case_summary([case])
            label = str(case.get("label", "start"))
            if label == "start":
                global_voice_errors.append(message)
            else:
                voice_errors.setdefault(label, []).append(message)

    current_meter: Optional[str] = None
    current_key: Optional[str] = None
    tree: Optional[MetricTree] = None
    schema_error: Optional[ValueError] = None
    pending_meter: Optional[str] = None
    pending_tree: Optional[MetricTree] = None
    pending_schema_error: Optional[ValueError] = None
    bar_data: List[tuple] = []
    bar_errors: List[ValueError] = []
    bars: List[dict] = []
    cur_label = "pickup"

    def locate_current(error: ValueError) -> ValueError:
        return _locate_tokenizer_error(
            error,
            bar_index=len(bars),
            bar_label=cur_label,
            bar_ordinal=len(bars),
        )

    def _schema():
        toks: List[str] = []
        if current_meter:
            m = re.match(r"\*M(\d+)/(\d+)", current_meter)
            if m:
                nt = _VALID_NUM.get(m.group(1))
                dt = _VALID_DEN.get(m.group(2))
                if nt and dt:
                    toks.extend([nt, dt])
        if current_key:
            kt = _KEY_SIG_TO_TOKEN.get(current_key)
            if kt:
                toks.append(kt)
        return toks

    def flush():
        nonlocal bar_data, bar_errors
        if not bar_data:
            return
        errors = list(bar_errors)
        errors.extend(ValueError(message) for message in global_voice_errors)
        normalized_label = cur_label.lstrip("=").rstrip("-;:|!")
        errors.extend(
            ValueError(message)
            for message in voice_errors.get(normalized_label, ())
        )
        if schema_error is not None:
            errors.append(schema_error)

        chans: List[list] = []
        evts: List[dict] = []
        try:
            if tree is None:
                raise ValueError("bar data before any meter declaration")
            rows = [s for _, s in bar_data]
            missing = [li for li, _ in bar_data if li not in line_channels]
            if missing:
                raise ValueError(
                    f"data rows have no readable spine topology: {missing}")
            chans = [line_channels[li] for li, _ in bar_data]
            rrebinds = [clock_flow.get(li) for li, _ in bar_data]
            structural = {
                row for row, rebind in enumerate(rrebinds) if rebind
            }
            evts = _parse_bar_events(
                rows, row_channels=chans, row_rebinds=rrebinds,
                beat_step=Fraction(tree.bar_length, tree.beat_count()))
            evts = _absorb_spelling_runs(evts, tree, structural)
            channels = {
                channel for row in chans for channel in row
                if channel is not None
            }
            _validate_channel_tiling(evts, tree, channels)
            _validate_canonical_spelling(evts, tree, structural)
        except ValueError as error:
            errors.append(_with_tokenizer_context(
                cur_label,
                error,
                bar_index=len(bars),
                bar_label=cur_label,
                bar_ordinal=len(bars),
            ))

        row_counts = [
            {
                hand: sum(channel is not None and channel[0] == hand
                          for channel in row)
                for hand in (0, 1)
            }
            for row in chans
        ]
        counts = row_counts[0] if row_counts else {0: 1, 1: 1}
        if row_counts and any(
                item != counts for item in row_counts[1:]):
            errors.append(ValueError(
                f"{cur_label}: voice topology changes inside bar data"))
        if any(counts[hand] < 1 for hand in (0, 1)):
            errors.append(ValueError(f"{cur_label}: missing hand spine"))
        if errors:
            error = _locate_tokenizer_error(
                errors[0],
                bar_index=len(bars),
                bar_label=cur_label,
                bar_ordinal=len(bars),
            )
            if not isolate_invalid:
                raise error
            errors[0] = error
        operation_counts = (
            bar_ops[len(bars)] if len(bars) < len(bar_ops) else {
                "head_splits": {0: 0, 1: 0},
                "tail_merges": {0: 0, 1: 0},
            }
        )
        bars.append({
            "label": cur_label,
            "tree": tree,
            "schema": _schema(),
            "events": evts,
            "voice_counts": counts,
            "head_splits": operation_counts["head_splits"],
            "tail_merges": operation_counts["tail_merges"],
            "error": errors[0] if errors else None,
        })
        bar_data = []
        bar_errors = []

    def _apply_pending():
        nonlocal current_meter, tree, schema_error
        nonlocal pending_meter, pending_tree, pending_schema_error
        if pending_meter is not None:
            current_meter = pending_meter
            tree = pending_tree
            schema_error = pending_schema_error
            pending_meter = None
            pending_tree = None
            pending_schema_error = None

    for li, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("!"):
            error = ValueError(
                f"{cur_label}: comment/reference record not canonical "
                f"at line {li + 1}")
            if not isolate_invalid:
                raise locate_current(error)
            bar_errors.append(error)
            continue
        if s.startswith("*"):
            for p in s.split("\t"):
                recognized = p in {
                    "*", "**kern", "*staff1", "*staff2", "*Ipiano",
                    "*clefF4", "*clefG2", "*^", "*v", "*-",
                }
                m = re.fullmatch(r"\*M(\d+)/(\d+)", p)
                if m:
                    recognized = True
                    new_meter = p
                    num, den = int(m.group(1)), int(m.group(2))
                    if (num not in MetricTree.SUPPORTED_NUM
                            or den not in MetricTree.SUPPORTED_DEN):
                        error = TokenizerOOVError(
                            f"meter {num}/{den} not in vocabulary")
                        if not isolate_invalid:
                            raise locate_current(error)
                        new_tree = None
                    else:
                        error = None
                        new_tree = get_metric_tree(num, den)
                    if tree is None or not bar_data:
                        current_meter = new_meter
                        tree = new_tree
                        schema_error = error
                        pending_meter = None
                        pending_tree = None
                        pending_schema_error = None
                    else:
                        pending_meter = new_meter
                        pending_tree = new_tree
                        pending_schema_error = error
                elif p.startswith("*M") and not p.startswith("*MM"):
                    error = ValueError(
                        f"meter spelling not canonical: {p!r}")
                    if not isolate_invalid:
                        raise locate_current(error)
                    bar_errors.append(error)
                if p.startswith("*k["):
                    recognized = True
                    if p not in _KEY_SIG_TO_TOKEN:
                        error = TokenizerOOVError(
                            f"key signature not canonical: {p!r}")
                        if not isolate_invalid:
                            raise locate_current(error)
                        bar_errors.append(error)
                        continue
                    if bar_data:
                        error = ValueError(
                            "key signature must be at the head of its bar: "
                            f"{p!r} at line {li + 1}")
                        if not isolate_invalid:
                            raise locate_current(error)
                        bar_errors.append(error)
                    current_key = p
                if not recognized:
                    error = TokenizerOOVError(
                        f"interpretation OOV at line {li + 1}: {p!r}")
                    if not isolate_invalid:
                        raise locate_current(error)
                    bar_errors.append(error)
            continue
        if s.startswith("="):
            flush()
            cur_label = s.split("\t")[0]
            _apply_pending()
            continue
        bar_data.append((li, s))

    flush()

    valid_bars = [bar for bar in bars if bar["error"] is None]
    assigned_cases = _assign_voices(valid_bars)
    if assigned_cases and not isolate_invalid:
        case = assigned_cases[0]
        label = str(case.get("label", "pickup"))
        ordinal = next(
            (index for index, bar in enumerate(bars)
             if bar.get("label") == label),
            None,
        )
        raise _locate_tokenizer_error(
            ValueError("voice: " + _voice_case_summary(assigned_cases)),
            bar_index=ordinal,
            bar_label=label,
            bar_ordinal=ordinal,
        )
    if assigned_cases:
        by_label = {bar["label"]: bar for bar in valid_bars}
        for case in assigned_cases:
            label = str(case.get("label", "pickup"))
            bar = by_label.get(label) or by_label.get(f"={label}")
            if bar is not None and bar["error"] is None:
                bar["error"] = ValueError(
                    "voice: " + _voice_case_summary([case]))

    # Recheck duration spelling after physical fields have become canonical
    # left-to-right addresses.
    for ordinal, bar in enumerate(bars):
        if bar["error"] is not None:
            continue
        try:
            _validate_canonical_spelling(
                bar["events"], bar["tree"], assigned_voices=True)
        except ValueError as error:
            contextual = _with_tokenizer_context(
                bar["label"],
                error,
                bar_index=ordinal,
                bar_label=bar["label"],
                bar_ordinal=ordinal,
            )
            if not isolate_invalid:
                raise contextual from None
            bar["error"] = contextual
    # Source tie pairing is not validated here: the strict reconstruct
    # in the roundtrip gate rejects unlanded departures, so a second
    # tokenizer-side pairing check would duplicate that account.

    return bars


def _tokenize_parsed_bars(
    parsed_bars: List[dict],
    bar_slice: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Serialize an optional inclusive slice of already validated bars."""
    bars = parsed_bars

    if bar_slice is not None:
        first, last = bar_slice
        if first < 0 or last < first or last >= len(bars):
            raise ValueError(
                f"bar slice {bar_slice} outside {len(bars)} parsed bars"
            )
        bars = bars[first:last + 1]

    invalid = next(
        (bar for bar in bars if bar.get("error") is not None), None)
    if invalid is not None:
        raise ValueError(str(invalid["error"]))

    result = ["<sos>"]
    for b in bars:
        try:
            bar_tokens = _serialize_bar(
                b["events"], b["tree"], b["schema"],
            )
        except ValueError as e:
            raise _with_tokenizer_context(b['label'], e) from None
        result.extend(bar_tokens)
    result.append("<eos>")
    try:
        _verify_tokens(result, bars)
    except ValueError as e:
        raise _with_tokenizer_context("token scope", e) from None
    return result


def tokenize_kern(
    kern_content: str,
    bar_slice: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Tokenize canonical kern, optionally selecting an inclusive bar range."""
    return _tokenize_parsed_bars(_parse_kern_bars(kern_content), bar_slice)


def _voice_case_summary(cases: List[dict]) -> str:
    parts = []
    for c in cases[:20]:
        loc = c.get("label") or c.get("label_start", "?")
        if c["kind"] == "width":
            parts.append(
                f"width{c['depth']} m{c['label_start']}-{c['label_end']}")
        else:
            parts.append(f"{c['kind']} m{loc}")
    if len(cases) > 20:
        parts.append(f"+{len(cases) - 20} more")
    return "; ".join(parts)


# =============================================================================
# KernTokenizer
# =============================================================================

class KernTokenizer:
    """Strict voice-bracket tokenizer for standardized **kern.

    Example:
        >>> tokenizer = KernTokenizer()
        >>> tokenizer.encode("4c# 8d")
        [token_ids...]
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or VOCAB
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def tokenize(self, kern_content: str) -> List[str]:
        """Tokenize entire kern content into voice-bracket token sequence.

        The module-level reader parses the canonical voices, validates
        their topology and timing, then serializes them without score edits.
        """
        return tokenize_kern(kern_content)

    def encode(self, kern_content: str, strict: bool = True) -> List[int]:
        """Encode kern content to token IDs.

        Args:
            kern_content: Kern file content or sequence
            strict: If True (default), raise ValueError on OOV tokens.
        """
        tokens = self.tokenize(kern_content)
        return self._encode_tokens(tokens, strict)

    def _encode_tokens(self, tokens: List[str], strict: bool) -> List[int]:
        ids = []
        for t in tokens:
            if t not in self.vocab:
                if strict:
                    raise TokenizerOOVError(
                        f"Token {t!r} not in vocabulary"
                    )
                continue
            ids.append(self.vocab[t])
        return ids

    def parse_kern_bars(self, kern_content: str) -> List[dict]:
        """Parse canonical kern for repeated chunk serialization."""
        return _parse_kern_bars(kern_content)

    def parse_kern_bars_isolated(self, kern_content: str) -> List[dict]:
        """Parse all bars while retaining local failures as quarantines."""
        return _parse_kern_bars(kern_content, isolate_invalid=True)

    def encode_parsed_chunk(
        self,
        parsed_bars: List[dict],
        first_bar: int,
        last_bar: int,
        strict: bool = True,
    ) -> List[int]:
        """Encode a bar range without reparsing its complete source score."""
        tokens = _tokenize_parsed_bars(parsed_bars, (first_bar, last_bar))
        return self._encode_tokens(tokens, strict)

    def encode_chunk(
        self,
        kern_content: str,
        first_bar: int,
        last_bar: int,
        strict: bool = True,
    ) -> List[int]:
        """Encode an inclusive bar range with balanced topology wrappers."""
        tokens = tokenize_kern(kern_content, (first_bar, last_bar))
        return self._encode_tokens(tokens, strict)



# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Vocabulary size: {VOCAB_SIZE}")
    expected = (len(SPECIAL_TOKENS) + len(STRUCTURAL_TOKENS_LIST)
                + len(METER_NUM_TOKENS) + len(METER_DEN_TOKENS)
                + len(KEY_TOKENS) + len(DURATION_TOKENS)
                + len(PITCH_TOKENS) + len(VOICE_TOKENS_LIST))
    assert VOCAB_SIZE == expected, (
        f"vocab has {VOCAB_SIZE} entries but the category lists sum to "
        f"{expected} — duplicate or clobbered token"
    )

    print("\n--- Walkthrough verification ---")
    wt_errors = verify_walkthroughs()
    if wt_errors:
        for e in wt_errors:
            print(f"  FAIL: {e}")
    else:
        print("  All walkthroughs pass.")

    print("\n--- Exhaustive cell enumeration ---")
    results = verify_spell_exhaustive()
    for meter, info in sorted(results["meters"].items()):
        status = "PASS" if not info["errors"] else "FAIL"
        print(
            f"  {meter:>5s}: {info['pairs']:>6d} pairs, "
            f"max_splits={info['max_splits']}, {status}"
        )
        for e in info["errors"]:
            print(f"    {e}")

    print(f"\nTotal pairs checked: {results['total_pairs']}")
    print(f"Overall: {'PASS' if results['pass'] else 'FAIL'}")
