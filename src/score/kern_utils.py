"""Kern cell utilities: recip spelling, duration parsing, tie-splitting,
and reading a missing meter off the first bar.

Shared by standardize_kern, reconstruct_kern, expand_repeat, and the
sanitize stations that need the same arithmetic on a score object.
"""

import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


_SUBTOK_RE = re.compile(
    # pre includes stem ink: a mark before the digits must not hide a
    # subtoken's duration (kern/ carries stems through Phase 1).
    r'^(?P<pre>[\[({/\\]*)(?P<dur>\d+\.*)?(?P<rest>.*)$')
_PITCH_RE = re.compile(r'([a-gA-G]+[#\-n]*)')
_GRACE_RE_FLAT = re.compile(r'[qQ]')
_METER_PATTERN = re.compile(r'^\*M(\d+)/(\d+)')

# Recip spellings a kern source may write, as quarter notes.  Kept as a
# table because it also answers whether a string is a legal recip at all,
# which parsing cannot: an unknown string parses to zero rather than
# failing.
_DUR_TO_FRAC: Dict[str, Fraction] = {}
for _b in [1, 2, 4, 8, 16, 32, 64, 128]:
    _DUR_TO_FRAC[str(_b)] = Fraction(4, _b)
    _DUR_TO_FRAC[f'{_b}.'] = Fraction(4, _b) * Fraction(3, 2)
_DUR_TO_FRAC['0'] = Fraction(8)
_DUR_TO_FRAC['0.'] = Fraction(12)
_DUR_TO_FRAC['1..'] = Fraction(7)

for _base, _val in [('3', Fraction(4, 3)), ('6', Fraction(2, 3)),
                     ('12', Fraction(1, 3)), ('24', Fraction(1, 6)),
                     ('48', Fraction(1, 12)), ('96', Fraction(1, 24))]:
    _DUR_TO_FRAC[_base] = _val


def _spell_frac(d: Fraction) -> List[Tuple[Fraction, str]]:
    """Spell a duration (in quarter notes) as kern recip strings.

    Returns [(piece_duration, recip_str), ...] whose durations sum to d.
    Prefers one plain or dotted glyph; otherwise peels dyadic pieces.
    """
    pieces: List[Tuple[Fraction, str]] = []
    rem = Fraction(d)
    if rem <= 0:
        raise ValueError(f"cannot spell non-positive duration {d}")
    while rem > 0:
        r = Fraction(4) / rem
        if r.denominator == 1:
            pieces.append((rem, str(r.numerator)))
            break
        r = Fraction(6) / rem
        if r.denominator == 1:
            pieces.append((rem, f"{r.numerator}."))
            break
        k = 0
        while Fraction(4, 2 ** k) > rem:
            k += 1
            if k > 10:
                unit = Fraction(1, rem.denominator)
                recip = str(4 * rem.denominator)
                if rem.numerator > 64:
                    raise ValueError(f"cannot spell duration {d}")
                for _ in range(rem.numerator):
                    pieces.append((unit, recip))
                return pieces
        piece = Fraction(4, 2 ** k)
        pieces.append((piece, str(2 ** k)))
        rem -= piece
    return pieces


def _split_cell_token(tok: str, head: Fraction, tail: Fraction
                      ) -> Optional[Tuple[List[str], List[str]]]:
    """Split a sounding cell token into tied head/tail piece token lists.

    Returns (head_tokens, tail_tokens) or None when the token cannot be
    split (grace content).
    """
    subs = tok.split()
    if any(_GRACE_RE_FLAT.search(s) for s in subs):
        return None
    head_sp = _spell_frac(head)
    tail_sp = _spell_frac(tail)
    spellings = head_sp + tail_sp
    n_pieces = len(spellings)

    piece_tokens: List[List[str]] = [[] for _ in range(n_pieces)]
    for sub in subs:
        m = _SUBTOK_RE.match(sub)
        is_rest = _is_rest_token(sub)
        pm = _PITCH_RE.search(m.group('rest') or '')
        body = (m.group('rest') or '')
        opens = '[' in (m.group('pre') or '')
        lands = body.endswith(']')
        mids = body.endswith('_')
        core = body[:-1] if (lands or mids) else body
        for pi, (pdur, recip) in enumerate(spellings):
            first, last = pi == 0, pi == n_pieces - 1
            if is_rest:
                piece_tokens[pi].append(f"{recip}{core}")
                continue
            if pm is None:
                return None
            if first:
                pre = '[' if (opens or not (lands or mids)) else ''
                suf = '_' if (lands or mids) else ''
            else:
                pre = ''
                if last:
                    suf = ']' if not (opens or mids) else '_'
                else:
                    suf = '_'
            piece_tokens[pi].append(f"{pre}{recip}{core}{suf}")
    heads = [' '.join(piece_tokens[i]) for i in range(len(head_sp))]
    tails = [' '.join(piece_tokens[i + len(head_sp)])
             for i in range(len(tail_sp))]
    return heads, tails


def _is_rest_token(token: str) -> bool:
    """Whether a kern data token's pitch component is a rest ('r')."""
    clean = token.lstrip('[({')
    first = clean.split(' ', 1)[0]
    pitch_part = re.sub(r'^\d+(?:%\d+)?\.*', '', first)
    return pitch_part.startswith('r')


def _token_duration_frac(token: str) -> Fraction:
    """Parse a kern data token's duration as a Fraction of quarter notes.

    Handles rational recips (``a%b``) and multiple dots.
    """
    clean = token.lstrip('[({')
    first = clean.split(' ', 1)[0]
    m = re.search(r'(\d+)(?:%(\d+))?(\.*)', first)
    if not m:
        return Fraction(0)
    recip = int(m.group(1))
    if recip == 0:
        return Fraction(8)
    den = int(m.group(2)) if m.group(2) else 1
    dur = Fraction(4 * den, recip)
    dots = len(m.group(3) or '')
    if dots:
        dur *= (Fraction(2) - Fraction(1, 2 ** dots))
    return dur


def _col_first_dur(token: str) -> Fraction:
    """First non-grace subtoken's duration — kern chord semantics."""
    for part in token.split():
        if _GRACE_RE_FLAT.search(part):
            continue
        d = _token_duration_frac(part)
        if d > 0:
            return d
    return Fraction(0)


def _span_to_meter(span: Fraction) -> Optional[Tuple[int, int]]:
    """Spell a bar span as num/den, or None when no denominator fits."""
    for den in (4, 8, 16):
        num = span * den / 4
        if num.denominator == 1 and num >= 1:
            return int(num), den
    return None


def infer_missing_meter(kern_content: str) -> str:
    """When a file declares no meter at all, declare the first bar's
    span as *M; unrepresentable spans insert nothing and fail loudly."""
    lines = kern_content.split('\n')
    for l in lines:
        if l.startswith('*') and any(_METER_PATTERN.match(pt)
                                     for pt in l.split('\t')):
            return kern_content

    data_idx = [i for i, l in enumerate(lines)
                if l.strip() and not l.startswith(('!', '*', '='))]
    if not data_idx:
        return kern_content
    bar_idx = [i for i, l in enumerate(lines) if l.startswith('=')]
    first_bl = bar_idx[0] if bar_idx else len(lines)
    if data_idx[0] < first_bl:
        region = [i for i in data_idx if i < first_bl]
    else:
        nxt = bar_idx[1] if len(bar_idx) > 1 else len(lines)
        region = [i for i in data_idx if first_bl <= i < nxt]
    if not region:
        return kern_content

    ncol = len(lines[region[0]].split('\t'))
    rem = [Fraction(0)] * ncol
    t = Fraction(0)
    for i in region:
        cells = lines[i].split('\t')
        if len(cells) != ncol:
            return kern_content
        for c, cell in enumerate(cells):
            if cell.strip() and cell != '.':
                d = _col_first_dur(cell)
                if d > 0:
                    rem[c] = d
        pending = [r for r in rem if r > 0]
        if not pending:
            return kern_content
        step = min(pending)
        rem = [r - step if r > 0 else r for r in rem]
        t += step
    span = t + (max(rem) if rem else Fraction(0))

    spelled = _span_to_meter(span)
    if spelled is None:
        return kern_content
    num, den = spelled

    insert_at = min(data_idx[0], first_bl)
    ncols_at = len(lines[insert_at].split('\t'))
    lines.insert(insert_at, '\t'.join([f'*M{num}/{den}'] * ncols_at))
    return '\n'.join(lines)
