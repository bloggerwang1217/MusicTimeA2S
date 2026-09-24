#!/usr/bin/env python3
"""
Reconstruct **kern from MusicTime-A2S Model Predictions
========================================================

Converts factorized token predictions back to valid **kern format.

The MusicTime-A2S model predicts factorized tokens with barlines (<bar>) and column
separators (<coc>). This module reassembles them into proper **kern notation.

Usage:
    from src.score.reconstruct_kern import reconstruct_kern_from_bracket_tokens

    tokens = ['4', 'c', '<bar>', '8', 'e', '8', 'g', '<bar>', ...]
    kern_str = reconstruct_kern_from_bracket_tokens(tokens)
"""

import re
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from src.a2s.piano.tokenizer import _TOKEN_TO_KEY_SIG


def extract_kern_metadata(kern_content: str) -> Dict[str, str]:
    """Extract essential metadata from a kern file for reconstruction.

    Extracts interpretation lines that converter21 needs for correct parsing:
    time signature, key signature, clef, instrument, staff assignment.
    Also extracts the starting measure number for proper barline numbering.

    Args:
        kern_content: Original kern file content

    Returns:
        Dict with keys like 'time_sig', 'key_sig', 'clef', 'first_measure_num', etc.
        Each value is a full tab-separated interpretation line (or a number string for first_measure_num).
    """
    metadata = {}
    found_first_measure = False
    
    for line in kern_content.split('\n'):
        stripped = line.strip()
        
        # Extract the first measure number (format: =0, =1, =1-, etc.)
        if not found_first_measure and stripped.startswith('='):
            first_field = stripped.split('\t')[0]
            # Extract just the number part (=0, =1, =1-, etc.)
            match = re.match(r'^=([0-9]+)', first_field)
            if match:
                metadata['first_measure_num'] = match.group(1)
                found_first_measure = True
        
        if not stripped.startswith('*'):
            continue
        # Skip spine structure and terminators
        if stripped.startswith('**') or stripped.startswith('*-'):
            continue
        if stripped.startswith('*^') or stripped.startswith('*v'):
            continue

        first_field = stripped.split('\t')[0]
        if first_field.startswith('*M') and '/' in first_field and 'time_sig' not in metadata:
            metadata['time_sig'] = line
        elif first_field.startswith('*k[') and 'key_sig' not in metadata:
            metadata['key_sig'] = line
        elif first_field.startswith('*clef') and 'clef' not in metadata:
            metadata['clef'] = line
        elif first_field.startswith('*staff') and 'staff' not in metadata:
            metadata['staff'] = line
        elif first_field.startswith('*I') and 'instrument' not in metadata:
            metadata['instrument'] = line
        elif first_field.startswith('*MM') and 'tempo' not in metadata:
            metadata['tempo'] = line

    return metadata


# =============================================================================
# Voice-bracket reconstruction (bracket tokens → **kern)
# =============================================================================

from src.a2s.piano.tokenizer import (
    RECIP_TO_GRID,
    get_metric_tree,
    kern_pitch_to_midi,
    spell_positioned,
)

_OPENS = {"<pl>": 0, "<pr>": 1}
_CLOSES = {"</pl>": 0, "</pr>": 1}
def _bracket_tokens_to_events(
    tokens: List[str],
    release_normalizations: Optional[List[dict]] = None,
    tail_rest_completions: Optional[List[dict]] = None,
    tail_rest_completion_bars: Optional[Set[int]] = None,
):
    """Parse voice-bracket tokens into per-bar event lists.

    Pure transcription + verification: repeating <v> before a hand bracket
    selects its left-to-right address.  Each bar's voice count is the maximum
    addressed depth plus one; adjacent counts later determine kern spine
    operations.  Structural tokens verify clocks and never synthesize score
    content in strict mode.  The optional final-bar repair only materializes
    metrically exact rests after every voice has stopped at one shared time.
    Semantically equivalent permutations within an exact same-release group
    are normalized to the canonical hand/voice order.

    Yields (bar_events, meter_str, key_sig_str, topology) for each bar.
    bar_events: list of dicts, each with 'onset' (Fraction).
    """
    from fractions import Fraction

    cur_events: List[dict] = []
    meter_str: Optional[str] = None
    key_str: Optional[str] = None
    pending_num: Optional[str] = None
    started = False

    stack: List[dict] = []
    clock: Dict[tuple, Fraction] = {}
    bar_voice_count: Dict[int, int] = {0: 1, 1: 1}
    address_depth = 0
    num: Optional[int] = None
    den: Optional[int] = None
    stream_time = Fraction(0)
    bar_no = 0

    def _clock(h: int, v: int) -> Fraction:
        return clock.get((h, v), Fraction(0))

    def _bar_length() -> Optional[Fraction]:
        if num is None or den is None:
            return None
        return Fraction(num * 128, den)

    def _beat_count() -> Optional[int]:
        if num is None:
            return None
        return num // 3 if num in (6, 9, 12) else num

    def _beat_step() -> Optional[Fraction]:
        bar_len = _bar_length()
        if bar_len is None:
            return None
        return bar_len / _beat_count()

    def _finish_bar(*, allow_tail_rest_completion: bool = False):
        nonlocal stream_time
        if stack:
            raise ValueError(
                f"bar {bar_no}: {len(stack)} bracket(s) still open at <bar>"
            )
        if address_depth:
            raise ValueError(
                f"bar {bar_no}: <v> prefix not followed by a hand bracket"
            )
        bar_len = _bar_length()
        if allow_tail_rest_completion and bar_len is not None:
            channels = [
                (hand, voice)
                for hand in (0, 1)
                for voice in range(bar_voice_count[hand])
            ]
            positions = [_clock(hand, voice) for hand, voice in channels]
            if positions and any(position > bar_len for position in positions):
                raise ValueError(
                    f"bar {bar_no}: a voice ends past bar length {bar_len}"
                )
            if positions and any(position != positions[0] for position in positions):
                raise ValueError(
                    f"bar {bar_no}: tail-rest fallback requires aligned "
                    f"voice clocks, got {positions}"
                )
            if positions and positions[0] < bar_len:
                onset = positions[0]
                missing = bar_len - onset
                if onset.denominator != 1 or missing.denominator != 1:
                    raise ValueError(
                        f"bar {bar_no}: tail-rest fallback cannot spell "
                        f"fractional grid span [{onset}, {bar_len})"
                    )
                glyphs = spell_positioned(
                    int(onset),
                    int(missing),
                    get_metric_tree(num, den),
                )
                fill_record = {
                    "kind": "tail_rest_completion",
                    "bar_index": bar_no - 1,
                    "from_tick": str(onset),
                    "bar_length_tick": str(bar_len),
                    "voices": [],
                }
                for hand, voice in channels:
                    position = onset
                    for glyph in glyphs:
                        duration = Fraction(RECIP_TO_GRID[glyph])
                        cur_events.append({
                            "hand": hand,
                            "voice": voice,
                            "onset": position,
                            "dur_grid": duration,
                            "dur_tok": glyph,
                            "pitches": ["r"],
                            "tie_ins": [False],
                            "tie_out": False,
                            "is_tup": False,
                        })
                        position += duration
                    clock[(hand, voice)] = bar_len
                    fill_record["voices"].append({
                        "hand": "l" if hand == 0 else "r",
                        "voice": voice,
                        "duration_tokens": list(glyphs),
                    })
                stream_time = bar_len
                tail_rest_completions.append(fill_record)
        releases: Dict[Fraction, List[tuple[int, dict]]] = {}
        for position, event in enumerate(cur_events):
            release = event["onset"] + event["dur_grid"]
            releases.setdefault(release, []).append((position, event))
        for release, group in releases.items():
            positions = [position for position, _event in group]
            actual = [event for _position, event in group]
            ordered = sorted(
                actual,
                key=lambda event: (event["hand"], -event["voice"]),
            )
            actual_order = [
                (event["hand"], event["voice"]) for event in actual
            ]
            canonical_order = [
                (event["hand"], event["voice"]) for event in ordered
            ]
            if actual_order != canonical_order and release_normalizations is not None:
                original_position = {
                    id(event): position for position, event in group
                }

                def event_ref(event: dict) -> dict:
                    return {
                        "event_index": original_position[id(event)],
                        "hand": "l" if event["hand"] == 0 else "r",
                        "voice": int(event["voice"]),
                        "onset_tick": str(event["onset"]),
                        "duration_tick": str(event["dur_grid"]),
                    }

                release_normalizations.append({
                    "bar_index": bar_no - 1,
                    "release_tick": str(release),
                    "original": [event_ref(event) for event in actual],
                    "canonical": [event_ref(event) for event in ordered],
                })
            for position, event in zip(positions, ordered):
                cur_events[position] = event
        for hand in (0, 1):
            addressed = {
                int(event["voice"])
                for event in cur_events
                if int(event["hand"]) == hand
            }
            expected = set(range(bar_voice_count[hand]))
            if addressed != expected:
                raise ValueError(
                    f"bar {bar_no}: hand {hand} addresses "
                    f"{sorted(addressed)}, expected {sorted(expected)}"
                )
        if bar_len is not None:
            for hand in (0, 1):
                for depth in range(bar_voice_count[hand]):
                    if clock.get((hand, depth), Fraction(0)) != bar_len:
                        raise ValueError(
                            f"bar {bar_no}: hand {hand} voice {depth} ends at "
                            f"{clock.get((hand, depth), Fraction(0))}, "
                            f"bar ends at {bar_len}"
                        )

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if address_depth and t not in ("<v>", *tuple(_OPENS), *tuple(_CLOSES)):
            raise ValueError(
                f"bar {bar_no}: <v> prefix must directly address a bracket"
            )
        if t in ("<pad>", "<sos>", "<eos>"):
            i += 1
            continue

        if t == "<bar>":
            # Every <bar> delimits a bar, including empty ones — dropping
            # them would shift measure numbering.
            if started:
                _finish_bar(
                    allow_tail_rest_completion=(
                        tail_rest_completion_bars is not None
                        and bar_no - 1 in tail_rest_completion_bars
                    ),
                )
                yield cur_events, meter_str, key_str, {
                    "voice_counts": dict(bar_voice_count),
                    "next_counts": dict(bar_voice_count),
                }
                cur_events = []
            started = True
            clock = {(0, 0): Fraction(0), (1, 0): Fraction(0)}
            bar_voice_count = {0: 1, 1: 1}
            address_depth = 0
            stream_time = Fraction(0)
            bar_no += 1
            i += 1
            continue

        if t == "<grid>":
            # Hidden-time slot marker: legal only inside an open capsule.
            frame = stack[-1] if stack else None
            if frame is None or not frame.get("in_tup"):
                raise ValueError(
                    f"bar {bar_no}: <grid> outside a capsule"
                )
            if frame["slots"] and not frame["slots"][-1]["attack"]:
                raise ValueError(
                    f"bar {bar_no}: capsule slot holds no pitch"
                )
            frame["tup_n"] += 1
            frame["cur_slot"] = {"attack": [], "attack_ti": []}
            frame["slots"].append(frame["cur_slot"])
            i += 1
            continue

        if t.startswith("<num:"):
            pending_num = t[5:-1]
            i += 1
            continue
        if t.startswith("<den:"):
            if pending_num is None:
                raise ValueError(
                    f"bar {bar_no}: <den:> without preceding <num:>"
                )
            try:
                num = int(pending_num)
                den = int(t[5:-1])
            except ValueError:
                raise ValueError(
                    f"bar {bar_no}: malformed meter tokens "
                    f"<num:{pending_num}> {t}"
                )
            meter_str = f"*M{num}/{den}"
            pending_num = None
            i += 1
            continue
        if t.startswith("<key:"):
            if t not in _TOKEN_TO_KEY_SIG:
                raise ValueError(f"bar {bar_no}: unknown key token {t}")
            key_str = _TOKEN_TO_KEY_SIG[t]
            i += 1
            continue

        if t == "<v>":
            address_depth += 1
            i += 1
            continue

        if t in _OPENS:
            h = _OPENS[t]
            v = address_depth
            address_depth = 0
            if v > bar_voice_count[h]:
                raise ValueError(
                    f"bar {bar_no}: address {v} skips hand {h} voice "
                    f"depth {bar_voice_count[h]}"
                )
            if v == bar_voice_count[h]:
                bar_voice_count[h] = v + 1
                clock[(h, v)] = Fraction(0)
            if any(frame["hand"] == h and frame["voice"] == v
                   for frame in stack):
                raise ValueError(
                    f"bar {bar_no}: hand {h} voice {v} already open"
                )
            onset = _clock(h, v)
            if onset != stream_time:
                raise ValueError(
                    f"bar {bar_no}: {t} at hand clock {onset}, expected "
                    f"stream time {stream_time}"
                )
            prior = []
            if h == 1:
                prior.extend((0, depth)
                             for depth in range(bar_voice_count[0]))
            prior.extend((h, depth) for depth in range(v))
            for prior_hand, prior_voice in prior:
                if (_clock(prior_hand, prior_voice) == onset
                        and not any(frame["hand"] == prior_hand
                                    and frame["voice"] == prior_voice
                                    for frame in stack)):
                    raise ValueError(
                        f"bar {bar_no}: hand {h} voice {v} opens before "
                        f"hand {prior_hand} voice {prior_voice} at {onset}"
                    )
            stack.append({
                "hand": h,
                "voice": v,
                "_open_time": onset,
                "tie_out": False, "is_tup": False,
                "in_tup": False,
                "tup_n": 0, "slots": [],
                "cur_slot": None,
                "pitches": [], "tie_ins": [],
            })
            i += 1
            continue

        if t == "<tup>":
            if not stack:
                raise ValueError(f"bar {bar_no}: <tup> outside any bracket")
            frame = stack[-1]
            if frame["is_tup"]:
                raise ValueError(
                    f"bar {bar_no}: <tup> inside an open capsule"
                )
            if frame["pitches"]:
                raise ValueError(
                    f"bar {bar_no}: <tup> after plain pitches in the "
                    f"same bracket"
                )
            frame["is_tup"] = True
            frame["in_tup"] = True
            i += 1
            continue

        if t == "</tup>":
            frame = stack[-1] if stack else None
            if frame is None or not frame.get("in_tup"):
                raise ValueError(
                    f"bar {bar_no}: </tup> without an open capsule"
                )
            if not frame["slots"]:
                raise ValueError(
                    f"bar {bar_no}: capsule closes with no <grid> slot"
                )
            if not frame["slots"][-1]["attack"]:
                raise ValueError(
                    f"bar {bar_no}: capsule slot holds no pitch"
                )
            frame["in_tup"] = False
            frame["cur_slot"] = None
            i += 1
            continue

        if t in _CLOSES:
            close_hand = _CLOSES[t]
            close_voice = address_depth
            address_depth = 0
            frame = None
            for si in range(len(stack) - 1, -1, -1):
                if (stack[si]["hand"] == close_hand
                        and stack[si].get("voice") == close_voice):
                    frame = stack.pop(si)
                    break
            if frame is None:
                raise ValueError(
                    f"bar {bar_no}: {t} without a matching open"
                )
            if frame.get("in_tup"):
                raise ValueError(
                    f"bar {bar_no}: {t} closes a capsule still missing "
                    f"</tup>"
                )
            # Next token = duration
            i += 1
            dur_tok = tokens[i] if i < len(tokens) else None
            if dur_tok not in RECIP_TO_GRID:
                raise ValueError(
                    f"bar {bar_no}: expected duration token after {t}, "
                    f"got {dur_tok!r}"
                )
            dur_grid = RECIP_TO_GRID[dur_tok]
            frame["dur_grid"] = dur_grid
            frame["dur_tok"] = dur_tok

            h = frame["hand"]
            frame["onset"] = frame["_open_time"]
            end = frame["_open_time"] + dur_grid
            previous_stream_time = stream_time
            if end < previous_stream_time:
                raise ValueError(
                    f"bar {bar_no}: {t} closes event at {end} before "
                    f"stream time {previous_stream_time}"
                )
            if num is not None and den is not None:
                bar_len = Fraction(num * 128, den)
                if end > bar_len:
                    raise ValueError(
                        f"bar {bar_no}: event ends at {end} past bar "
                        f"length {bar_len} ({num}/{den})"
                    )
                step = _beat_step()
                if frame["is_tup"] and step is not None:
                    beat_end = (frame["_open_time"] // step + 1) * step
                    if end > beat_end:
                        raise ValueError(
                            f"bar {bar_no}: capsule closes at {end} past "
                            f"its derived beat boundary {beat_end}"
                        )
            clock[(h, frame["voice"])] = end
            if end > previous_stream_time:
                due = [
                    (hand, voice)
                    for (hand, voice), position in clock.items()
                    if position == previous_stream_time
                    and not any(
                        opened["hand"] == hand and opened["voice"] == voice
                        for opened in stack
                    )
                ]
                if due:
                    raise ValueError(
                        f"bar {bar_no}: close advances to {end} before "
                        f"due voice channels {due} act at "
                        f"{previous_stream_time}"
                    )
            stream_time = end

            # Check for <tie> after duration
            if i + 1 < len(tokens) and tokens[i + 1] == "<tie>":
                if frame["is_tup"]:
                    # A capsule departure refers to the last member, the
                    # only one still sounding at close.
                    sounds = any(
                        p != "r" for p in frame["slots"][-1]["attack"]
                    ) if frame["slots"] else False
                else:
                    sounds = any(p != "r" for p in frame["pitches"])
                if not sounds:
                    raise ValueError(
                        f"bar {bar_no}: <tie> cannot depart from a rest"
                    )
                frame["tie_out"] = True
                i += 1
            frame.pop("cur_slot", None)
            frame.pop("_open_time", None)
            cur_events.append(frame)
            i += 1
            continue

        if t == "</tie>":
            if not stack:
                raise ValueError(
                    f"bar {bar_no}: </tie> outside any bracket"
                )
            frame = stack[-1]
            if frame["is_tup"]:
                slot = frame.get("cur_slot")
                if not (frame.get("in_tup") and slot and slot["attack"]):
                    raise ValueError(
                        f"bar {bar_no}: </tie> with no preceding pitch "
                        f"in slot"
                    )
                if len(frame["slots"]) != 1:
                    raise ValueError(
                        f"bar {bar_no}: </tie> lands on an interior "
                        f"capsule member; ties attach at edges only"
                    )
                if slot["attack"][-1] == "r":
                    raise ValueError(
                        f"bar {bar_no}: </tie> cannot land on a rest"
                    )
                slot["attack_ti"][-1] = True
            else:
                if not frame["pitches"]:
                    raise ValueError(
                        f"bar {bar_no}: </tie> with no preceding pitch"
                    )
                if frame["pitches"][-1] == "r":
                    raise ValueError(
                        f"bar {bar_no}: </tie> cannot land on a rest"
                    )
                frame["tie_ins"][-1] = True
            i += 1
            continue

        if t == "<tie>":
            # A legal <tie> is consumed by the close handler right after
            # the duration token; reaching here means it is misplaced.
            raise ValueError(
                f"bar {bar_no}: <tie> not directly after a close + duration"
            )

        # Pitch token
        if not stack:
            raise ValueError(
                f"bar {bar_no}: pitch token {t!r} outside any bracket"
            )
        frame = stack[-1]
        if frame["is_tup"]:
            slot = frame.get("cur_slot")
            if not frame.get("in_tup") or slot is None:
                raise ValueError(
                    f"bar {bar_no}: pitch {t!r} in a capsule bracket "
                    f"outside a <grid> slot"
                )
            slot["attack"].append(t)
            slot["attack_ti"].append(False)
        else:
            frame["pitches"].append(t)
            frame["tie_ins"].append(False)
        i += 1

    final_bar_index = bar_no - 1
    _finish_bar(allow_tail_rest_completion=(
        (
            tail_rest_completion_bars is not None
            and final_bar_index in tail_rest_completion_bars
        )
        or (
            tail_rest_completion_bars is None
            and tail_rest_completions is not None
        )
    ))
    if started or cur_events:
        yield cur_events, meter_str, key_str, {
            "voice_counts": dict(bar_voice_count),
            "next_counts": dict(bar_voice_count),
        }


def _member_recip_str(n: int, total_grid: int) -> str:
    """Kern recip for one of n equal members spanning total_grid units.

    Integer when it divides evenly, rational a%b syntax otherwise
    (e.g. 13 slots over a dotted span) — integer floor division here
    silently corrupted 13-tuplet members.
    """
    from fractions import Fraction
    if total_grid <= 0 or n <= 0:
        raise ValueError(
            f"cannot derive member recip: {n} members over "
            f"{total_grid} grid units"
        )
    r = Fraction(128 * n) / Fraction(total_grid)
    if r.denominator == 1:
        return str(r.numerator)
    return f"{r.numerator}%{r.denominator}"


def _run_recip_str(n: int, total_grid: int, slots: int) -> str:
    """Recip for `slots` contiguous members of an n-member tuplet span.

    A partial three-slot run prints as the dotted two-slot value and a
    seven-slot run as the double-dotted four-slot value: the combined
    duration can be plainly dyadic, but a binary reciprocal inside the
    tuplet window breaks the group's readback.
    """
    if 1 < slots < n:
        if slots == 3:
            base = _member_recip_str(n, total_grid * 2)
            if "%" not in base:
                return f"{base}."
        if slots == 7:
            base = _member_recip_str(n, total_grid * 4)
            if "%" not in base:
                return f"{base}.."
    return _member_recip_str(n, total_grid * slots)


def _printed_lines_readable(data_lines: List[str],
                            bar_events: List[dict],
                            channels: Optional[List[tuple]] = None,
                            meter: Optional[tuple] = None,
                            ) -> bool:
    """Whether printed kern lines parse back to exactly these events."""
    from src.a2s.piano.tokenizer import (
        _event_signature, _parse_bar_events)
    beat_step = None
    if meter is not None:
        m_num, m_den = meter
        n_beats = m_num // 3 if m_num in (6, 9, 12) else m_num
        beat_step = Fraction(m_num * 128, m_den) / n_beats
    try:
        if channels:
            evs = _parse_bar_events(
                data_lines, row_channels=[list(channels)] * len(data_lines),
                beat_step=beat_step)
        else:
            evs = _parse_bar_events(data_lines, beat_step=beat_step)
    except ValueError:
        return False

    def _sig(e: dict, reparse: bool) -> tuple:
        if reparse:
            e = dict(e, voice=e.get("chain", 0))
        return _event_signature(e)

    return (_acoustic_form(_sig(e, True) for e in evs)
            == _acoustic_form(_sig(e, False) for e in bar_events))


def _event_sound_profile(events: List[dict], *, reparse: bool = False) -> tuple:
    """Return attack identities and the sounding-pitch timeline."""
    from src.a2s.piano.tokenizer import kern_pitch_to_midi

    attacks = []
    segments = []
    for event in events:
        hand = int(event["hand"])
        voice = int(
            event.get("chain", 0) if reparse else event.get("voice", 0)
        )
        onset = Fraction(event["onset"])
        duration = Fraction(event.get("dur", event.get("dur_grid")))
        if event.get("is_tup"):
            slots = event.get("slots") or []
            grain = duration / len(slots)
            for slot_index, slot in enumerate(slots):
                start = onset + slot_index * grain
                end = start + grain
                pitches = list(slot.get("attack", ()))
                tie_ins = list(
                    slot.get("attack_ti", [False] * len(pitches))
                )
                for pitch, tied in zip(pitches, tie_ins, strict=True):
                    if pitch != "r" and not tied:
                        attacks.append((
                            hand, voice, start, kern_pitch_to_midi(pitch),
                        ))
                for pitch in pitches:
                    if pitch != "r":
                        segments.append((
                            hand, voice, start, end,
                            kern_pitch_to_midi(pitch),
                        ))
            continue

        pitches = list(event.get("pitches", ()))
        tie_ins = list(event.get("tie_ins", [False] * len(pitches)))
        for pitch, tied in zip(pitches, tie_ins, strict=True):
            if pitch != "r" and not tied:
                attacks.append((
                    hand, voice, onset, kern_pitch_to_midi(pitch),
                ))
        for pitch in pitches:
            if pitch != "r":
                segments.append((
                    hand, voice, onset, onset + duration,
                    kern_pitch_to_midi(pitch),
                ))

    boundaries = sorted({
        point for segment in segments for point in segment[2:4]
    })
    sounding = []
    for start, end in zip(boundaries, boundaries[1:]):
        active = sorted(
            (hand, voice, midi)
            for hand, voice, segment_start, segment_end, midi in segments
            if segment_start <= start and end <= segment_end
        )
        if active:
            active_tuple = tuple(active)
            if (sounding and sounding[-1][1] == start
                    and sounding[-1][2] == active_tuple):
                sounding[-1] = (sounding[-1][0], end, active_tuple)
            else:
                sounding.append((start, end, active_tuple))
    return tuple(sorted(attacks)), tuple(sounding)


def _readback_mismatch_record(
    data_lines: List[str],
    bar_events: List[dict],
    channels: Optional[List[tuple]] = None,
) -> dict:
    """Classify a rejected print without changing the emitted score."""
    from src.a2s.piano.tokenizer import _parse_bar_events

    tuple_events = [event for event in bar_events if event.get("is_tup")]
    cardinalities = [
        int(event.get("tup_n", 0)) for event in tuple_events
    ]
    adjacent_tuplets = False
    for left in tuple_events:
        left_end = Fraction(left["onset"]) + Fraction(left["dur_grid"])
        for right in tuple_events:
            if left is right:
                continue
            same_channel = (
                left["hand"], left.get("voice", 0)
            ) == (
                right["hand"], right.get("voice", 0)
            )
            if same_channel and left_end == Fraction(right["onset"]):
                adjacent_tuplets = True
                break
        if adjacent_tuplets:
            break

    mixed_ordinary_rest_pitch = any(
        not event.get("is_tup")
        and "r" in event.get("pitches", ())
        and any(pitch != "r" for pitch in event.get("pitches", ()))
        for event in bar_events
    )
    repeated_ordinary_rest = any(
        not event.get("is_tup")
        and list(event.get("pitches", ())).count("r") > 1
        for event in bar_events
    )
    all_rest_tuple = any(
        all(
            all(
                pitch == "r"
                for pitch in (
                    tuple(slot.get("cont", ()))
                    + tuple(slot.get("attack", ()))
                )
            )
            for slot in event.get("slots", ())
        )
        for event in tuple_events
    )
    runaway_tuple = any(cardinality > 16 for cardinality in cardinalities)

    try:
        if channels:
            readback_events = _parse_bar_events(
                data_lines,
                row_channels=[list(channels)] * len(data_lines),
            )
        else:
            readback_events = _parse_bar_events(data_lines)
    except ValueError as error:
        return {
            "subtype": "printed_kern_parse_error",
            "severity": "parse_error",
            "attack_equal": None,
            "sounding_equal": None,
            "n_decoded_events": len(bar_events),
            "n_readback_events": None,
            "n_tuple_events": len(tuple_events),
            "tuple_cardinalities": cardinalities,
            "has_adjacent_tuplets": adjacent_tuplets,
            "printed_parse_error": str(error)[:200],
        }

    decoded_attacks, decoded_sounding = _event_sound_profile(bar_events)
    readback_attacks, readback_sounding = _event_sound_profile(
        readback_events, reparse=True,
    )
    attack_equal = decoded_attacks == readback_attacks
    sounding_equal = decoded_sounding == readback_sounding
    sound_equivalent = attack_equal and sounding_equal

    if not sound_equivalent:
        subtype = (
            "acoustic_change_adjacent_tuplet_boundary"
            if adjacent_tuplets
            else "acoustic_change_other"
        )
        severity = "acoustic_change"
    elif runaway_tuple:
        subtype = "sound_equivalent_runaway_tuplet_cardinality"
        severity = "sound_equivalent"
    elif mixed_ordinary_rest_pitch:
        subtype = "sound_equivalent_rest_pitch_event_split"
        severity = "sound_equivalent"
    elif repeated_ordinary_rest:
        subtype = "sound_equivalent_repeated_rest_collapse"
        severity = "sound_equivalent"
    elif any(cardinality in {1, 2, 4} for cardinality in cardinalities):
        subtype = "sound_equivalent_dyadic_tuplet_identity"
        severity = "sound_equivalent"
    elif all_rest_tuple:
        subtype = "sound_equivalent_all_rest_tuplet_identity"
        severity = "sound_equivalent"
    elif adjacent_tuplets:
        subtype = "sound_equivalent_adjacent_tuplet_regrouping"
        severity = "sound_equivalent"
    else:
        subtype = "sound_equivalent_redundant_tuplet_grouping"
        severity = "sound_equivalent"

    return {
        "subtype": subtype,
        "severity": severity,
        "attack_equal": attack_equal,
        "sounding_equal": sounding_equal,
        "n_decoded_events": len(bar_events),
        "n_readback_events": len(readback_events),
        "n_tuple_events": len(tuple_events),
        "tuple_cardinalities": cardinalities,
        "has_adjacent_tuplets": adjacent_tuplets,
    }


def _acoustic_form(sigs) -> list:
    """Reduce tuplet signatures to per-slot (attacked, sounding) sets.

    The printed form of a sustained tuplet member is a run of tie
    slices, which a literal re-parse groups at dyadic spans; the token
    form carries one group with continuation slots.  Both are the same
    sound, so the gate compares the merged per-slot acoustics instead
    of the grouping.
    """
    from fractions import Fraction as F
    out, tup_by_channel = [], {}
    for s in sigs:
        if not s[4]:
            out.append(s)
            continue
        tup_by_channel.setdefault((s[0], s[1]), []).append(s)
    runs_by_channel = {}
    for channel, group in tup_by_channel.items():
        group.sort(key=lambda s: s[2])
        runs = runs_by_channel.setdefault(channel, [])
        for s in group:
            grain = F(s[3]) / len(s[5]) if s[5] else None
            if (runs and runs[-1]["end"] == s[2]
                    and runs[-1]["grain"] == grain):
                run = runs[-1]
                run["slots"].extend(s[5])
                run["dur"] += s[3]
                run["end"] += s[3]
                run["tie_out"] = s[6]
            else:
                runs.append({"onset": s[2], "dur": s[3],
                             "end": s[2] + s[3], "grain": grain,
                             "slots": list(s[5]), "tie_out": s[6]})
    for (hand, voice), runs in runs_by_channel.items():
        for run in runs:
            acoustic = tuple(
                (tuple(sorted(p for p, ti in zip(slot[0], slot[1])
                              if p != "r" and not ti)),
                 tuple(sorted(p for p in slot[0] if p != "r")))
                for slot in run["slots"])
            out.append((hand, voice, run["onset"], run["dur"], True,
                        acoustic, run["tie_out"]))
    return sorted(out)


def _bar_channels(voice_counts: Dict[int, int]) -> List[tuple]:
    """Physical kern columns in canonical low-to-high address order."""
    channels: List[tuple] = []
    for h in (0, 1):
        channels.extend((h, depth) for depth in range(voice_counts[h]))
    return channels


def _events_to_kern_lines(
    events: List[dict], voice_counts: Dict[int, int],
) -> List[str]:
    """Convert a bar's events into kern data lines (tab-separated).

    Uses explicit onset from each event (set by _bracket_tokens_to_events)
    so addressed brackets produce correct concurrent onsets.  Columns follow
    `_bar_channels`'s fixed physical order for the current ordered leaves.
    """
    from fractions import Fraction

    channels = _bar_channels(voice_counts)
    rows: List[dict] = []

    for ev in events:
        h = ev["hand"]
        ch = (h, int(ev.get("voice", 0)))
        onset = Fraction(ev.get("onset", 0))

        if ev.get("is_tup") and ev.get("slots"):
            # A capsule prints as an equal division of its span: one
            # engraved member per slot, chord members sharing a slot.
            # Ties attach at the edges only (leading landings on slot 0,
            # trailing departures from the last slot).
            slots = ev["slots"]
            n = ev["tup_n"]
            total_grid = Fraction(ev["dur_grid"])
            member_dur = total_grid / n
            reciprocal = _member_recip_str(n, total_grid)

            tip = ev.get("tie_in_pitches")
            last_top = slots[-1].get("tie_out_pitches", set())

            for si, slot in enumerate(slots):
                attacks = slot.get("attack", [])
                attack_ti = slot.get(
                    "attack_ti", [False] * len(attacks))
                for pitch, tied in zip(attacks, attack_ti):
                    if pitch == "r":
                        kern = f"{reciprocal}r"
                    else:
                        tie_in_piece = bool(
                            tied and si == 0
                            and (tip is None or pitch in tip)
                        )
                        tie_out_piece = (si == n - 1
                                         and pitch in last_top)
                        if tie_in_piece and tie_out_piece:
                            kern = f"{reciprocal}{pitch}_"
                        elif tie_in_piece:
                            kern = f"{reciprocal}{pitch}]"
                        elif tie_out_piece:
                            kern = f"[{reciprocal}{pitch}"
                        else:
                            kern = f"{reciprocal}{pitch}"
                    rows.append({
                        "onset": onset + si * member_dur,
                        "hand": h, "chan": ch, "kern": kern,
                        "dur": member_dur,
                        "sound": pitch != "r", "tup": True,
                    })
        else:
            kern = _event_to_kern_note(ev)
            dur = Fraction(ev["dur_grid"])
            rows.append({"onset": onset, "hand": h, "chan": ch,
                         "kern": kern,
                         "dur": dur, "sound": ev["pitches"] != ["r"]})

    if not rows:
        return []

    all_onsets = sorted(set(r["onset"] for r in rows))
    by_onset_chan: Dict[Fraction, Dict[tuple, List[str]]] = {}
    for r in rows:
        by_onset_chan.setdefault(r["onset"], {}).setdefault(
            r["chan"], []).append(r["kern"])

    def _kern_part_key(part: str) -> tuple:
        reciprocal = re.search(r"(\d+)(?:%(\d+))?(\.*)", part)
        pitch_match = re.search(r"(([A-G])\2*|([a-g])\3*)[#\-n]*|r", part)
        if reciprocal is None or pitch_match is None:
            return (Fraction(0), -1, part)
        numerator = int(reciprocal.group(1))
        denominator = int(reciprocal.group(2) or 1)
        duration = Fraction(denominator, numerator)
        dots = len(reciprocal.group(3) or "")
        if dots:
            duration *= Fraction(2) - Fraction(1, 2 ** dots)
        pitch = pitch_match.group(0).replace("n", "")
        tie_role = 1 if "_" in part else 2 if "]" in part else 0
        return (-duration, kern_pitch_to_midi(pitch), tie_role, pitch)

    lines: List[str] = []
    for t in all_onsets:
        spines = []
        for ch in channels:
            kerns = by_onset_chan.get(t, {}).get(ch, [])
            if kerns:
                parts = [part for kern in kerns for part in kern.split()]
                parts.sort(key=_kern_part_key)
                spines.append(" ".join(parts))
            else:
                spines.append(".")
        lines.append("\t".join(spines))

    return lines


def _event_to_kern_note(ev: dict) -> str:
    """Convert one event dict to a kern note token string."""
    dur_tok = ev.get("dur_tok")
    if dur_tok is None:
        raise ValueError(
            f"event at onset {ev.get('onset')} missing its duration token"
        )
    pitches = ev["pitches"]
    tie_ins = ev.get("tie_ins", [False] * len(pitches))
    tie_out = ev.get("tie_out", False)

    parts = []
    top = ev.get("tie_out_pitches", set())
    tip = ev.get("tie_in_pitches")
    for pi, pitch in enumerate(pitches):
        ti = (
            tie_ins[pi] if pi < len(tie_ins) else False
        ) and (tip is None or pitch in tip)
        to = tie_out and pitch in top
        # converter21 hears the bracket-pair form [X] as a fresh attack;
        # underscore is the only continue spelling it reads silently.
        if to and ti:
            parts.append(f"{dur_tok}{pitch}_")
        elif to:
            parts.append(f"[{dur_tok}{pitch}")
        elif ti:
            parts.append(f"{dur_tok}{pitch}]")
        else:
            parts.append(f"{dur_tok}{pitch}")
    return " ".join(parts)


def reconstruct_kern_from_bracket_tokens(
    tokens: List[str],
    metadata: Optional[Dict[str, str]] = None,
    release_normalizations: Optional[List[dict]] = None,
    tie_normalizations: Optional[List[dict]] = None,
    readback_mismatches: Optional[List[dict]] = None,
    tail_rest_completions: Optional[List[dict]] = None,
    tail_rest_completion_bars: Optional[Set[int]] = None,
) -> str:
    """Reconstruct **kern from voice-bracket token sequence.

    The canonical surface is determined entirely by the typed events.  Close
    permutations at one exact release normalize to the same surface.  Invalid
    ties remain fatal unless an audit list explicitly enables normalization.
    Supplying ``tail_rest_completions`` likewise enables only final-bar tail
    completion and records every synthesized rest span.
    ``metadata`` remains an ignored compatibility argument for older callers;
    it never changes canonical output.
    """
    from fractions import Fraction

    n_spines = 2
    output_lines: List[str] = []
    prev_meter = None
    prev_key = None

    bars = list(_bracket_tokens_to_events(
        tokens,
        release_normalizations,
        tail_rest_completions,
        tail_rest_completion_bars,
    ))
    for bar_index, (_events, _meter, _key, topology) in enumerate(bars):
        following = (
            bars[bar_index + 1][3]["voice_counts"]
            if bar_index + 1 < len(bars) else {0: 1, 1: 1}
        )
        topology["next_counts"] = {
            hand: min(topology["voice_counts"][hand], following[hand])
            for hand in (0, 1)
        }

    def _landing_tied_pitches(
        ev2: dict, slot_index: Optional[int] = None,
    ) -> List[str]:
        if ev2.get("is_tup"):
            slots = ev2.get("slots") or []
            selected = (
                slots if slot_index is None
                else slots[slot_index:slot_index + 1]
            )
            return [
                pitch
                for slot in selected
                for pitch, tied in zip(
                    slot.get("attack", ()), slot.get("attack_ti", ()))
                if tied
            ]
        return [p for p, ti in zip(ev2.get("pitches", []),
                                   ev2.get("tie_ins", [])) if ti]

    def _event_pitches(ev: dict) -> Set[str]:
        if ev.get("is_tup"):
            ps: Set[str] = set()
            for slot in ev.get("slots") or []:
                ps.update(slot.get("attack", []))
            return ps
        return set(ev.get("pitches", []))

    def _bar_length(meter: Optional[str]):
        match = re.fullmatch(r"\*M(\d+)/(\d+)", meter or "")
        if match is None:
            return None
        from fractions import Fraction
        return Fraction(int(match.group(1)) * 128, int(match.group(2)))

    # A tie is a same-hand, same-pitch attachment.  Because departure syntax
    # is bracket-level, a landing first witnesses an otherwise-unwitnessed
    # bracket; pitches in one landing chord share that choice.  Intervening
    # fresh attacks are independent and voice addresses never enter pairing.
    pending: Dict[Tuple[int, str], List[Tuple[int, dict]]] = defaultdict(list)
    departures: List[dict] = []
    unmatched_landings: List[
        Tuple[int, dict, Optional[int], set]
    ] = []
    matched_departures: Set[int] = set()
    # Where a landing stood, whether or not it consumed this departure:
    # one key can carry several open departures at once.
    landings_at: Dict[Tuple[int, str], List[int]] = defaultdict(list)

    event_order: Dict[int, int] = {}

    def departure_sites(bi: int, ev: dict) -> List[dict]:
        """Return the bracket-level sites that still need a real landing."""
        sites = []
        ev["tie_out_pitches"] = set()
        if not ev.get("is_tup"):
            if ev.get("tie_out"):
                pitches = {
                    pitch for pitch in _event_pitches(ev) if pitch != "r"
                }
                if pitches:
                    sites.append({
                        "bar": bi,
                        "event": ev,
                        "hand": ev["hand"],
                        "pitches": pitches,
                        "assigned": set(),
                        "slot": None,
                        "serial": event_order[id(ev)],
                    })
            return sites

        slots = ev.get("slots") or []
        for slot in slots:
            slot["tie_out_pitches"] = set()
        # A capsule departure refers to the last member only (ties attach
        # at capsule edges).
        if ev.get("tie_out") and slots:
            slot_index = len(slots) - 1
            slot = slots[slot_index]
            pitches = {
                pitch for pitch in slot.get("attack", ())
                if pitch != "r"
            }
            if pitches:
                sites.append({
                    "bar": bi,
                    "event": ev,
                    "hand": ev["hand"],
                    "pitches": pitches,
                    "assigned": set(),
                    "slot": slot,
                    "slot_index": slot_index,
                    "serial": event_order[id(ev)],
                })
        return sites

    actions_by_bar: List[List[dict]] = [[] for _ in bars]
    landing_atoms: List[Tuple[tuple, int, dict, str, Optional[int]]] = []
    landing_position: Dict[Tuple[int, Optional[int], str], int] = {}
    landing_by_key_bar: Dict[Tuple[int, str, int], List[int]] = defaultdict(list)
    serial = 0
    for bi, (bar_events, _m, _k, _topology) in enumerate(bars):
        for event_index, ev in sorted(
                enumerate(bar_events),
                key=lambda item: (item[1]["onset"], item[0])):
            event_order[id(ev)] = serial
            ev["tie_in_pitches"] = set()
            slots = ev.get("slots") or []
            slot_indexes: List[Optional[int]] = (
                list(range(len(slots))) if ev.get("is_tup") else [None]
            )
            member_dur = (
                Fraction(ev["dur_grid"], len(slots)) if slots else None
            )
            for slot_index in slot_indexes:
                landing_time = Fraction(ev["onset"])
                if slot_index is not None:
                    landing_time += slot_index * member_dur
                tied = list(dict.fromkeys(
                    _landing_tied_pitches(ev, slot_index)
                ))
                if tied:
                    actions_by_bar[bi].append({
                        "kind": "landing",
                        "bar": bi,
                        "time": landing_time,
                        "event": ev,
                        "slot_index": slot_index,
                        "pitches": tied,
                        "source_order": (event_index, slot_index or 0),
                    })
            for site in departure_sites(bi, ev):
                slot_index = site.get("slot_index")
                departure_time = Fraction(ev["onset"])
                if slot_index is not None:
                    departure_time = (
                        Fraction(ev["onset"])
                        + slot_index * member_dur
                    )
                site["time"] = departure_time
                actions_by_bar[bi].append({
                    "kind": "departure",
                    "bar": bi,
                    "time": departure_time,
                    "site": site,
                    "source_order": (event_index, slot_index or 0),
                })
            serial += 1

    for bi, actions in enumerate(actions_by_bar):
        actions.sort(key=lambda action: (
            action["time"],
            0 if action["kind"] == "landing" else 1,
            action["source_order"],
        ))
        for action_index, action in enumerate(actions):
            action["order"] = (
                bi,
                action["time"],
                0 if action["kind"] == "landing" else 1,
                action_index,
            )
            if action["kind"] == "departure":
                action["site"]["order"] = action["order"]
                continue
            ev = action["event"]
            slot_index = action["slot_index"]
            for pitch in action["pitches"]:
                key = (id(ev), slot_index, pitch)
                landing_position[key] = len(landing_atoms)
                landing_atoms.append((
                    action["order"], bi, ev, pitch, slot_index,
                ))

    landing_atoms.sort(key=lambda atom: atom[0])
    landing_position.clear()
    for atom_index, atom in enumerate(landing_atoms):
        _order, bi, ev, pitch, slot_index = atom
        landing_position[(id(ev), slot_index, pitch)] = atom_index
        landing_by_key_bar[(ev["hand"], pitch, bi)].append(atom_index)

    def site_can_land(
        site: dict,
        atom: Tuple[tuple, int, dict, str, Optional[int]],
    ) -> bool:
        atom_order, landing_bar, landing, pitch, _slot_index = atom
        if atom_order <= site["order"]:
            return False
        if landing["hand"] != site["hand"] or pitch not in site["pitches"]:
            return False
        if landing_bar == site["bar"]:
            return True
        return (
            landing_bar == site["bar"] + 1
            and atom_order[1] == 0
        )

    def pending_can_still_close(
        candidate: dict,
        current_atom: Tuple[tuple, int, dict, str, Optional[int]],
    ) -> bool:
        """Do not spend a landing that another bracket uniquely needs."""
        (_current_order, _current_bar, current_event,
         current_pitch, current_slot) = current_atom
        required: Dict[int, dict] = {}
        for queue in pending.values():
            for _bar, site in queue:
                if id(site) not in matched_departures:
                    required[id(site)] = site
        required.pop(id(candidate), None)
        if not required:
            return True

        current_position = landing_position[
            (id(current_event), current_slot, current_pitch)
        ]
        edges = {}
        for site_id, site in required.items():
            indexes = set()
            for pitch in site["pitches"]:
                for bar_index in (site["bar"], site["bar"] + 1):
                    for atom_index in landing_by_key_bar.get(
                            (site["hand"], pitch, bar_index), ()):
                        if atom_index <= current_position:
                            continue
                        if site_can_land(site, landing_atoms[atom_index]):
                            indexes.add(atom_index)
            edges[site_id] = sorted(indexes)
        if any(not indexes for indexes in edges.values()):
            return False

        atom_owner: Dict[int, int] = {}

        def assign(site_id: int, seen: Set[int]) -> bool:
            for atom_index in edges[site_id]:
                if atom_index in seen:
                    continue
                seen.add(atom_index)
                previous = atom_owner.get(atom_index)
                if previous is None or assign(previous, seen):
                    atom_owner[atom_index] = site_id
                    return True
            return False

        return all(
            assign(site_id, set())
            for site_id in sorted(
                required,
                key=lambda item: (len(edges[item]), required[item]["serial"]),
            )
        )

    for bi, actions in enumerate(actions_by_bar):
        def expire_prior_bar_departures() -> None:
            for key, queue in list(pending.items()):
                pending[key] = [
                    item for item in queue if item[0] >= bi
                ]

        head_actions = [action for action in actions if action["time"] == 0]
        later_actions = [action for action in actions if action["time"] != 0]
        if not head_actions:
            expire_prior_bar_departures()

        for action in (*head_actions, *later_actions):
            if action["kind"] == "landing":
                ev = action["event"]
                slot_index = action["slot_index"]
                landing_departures: Set[int] = set()
                missing = set()
                for pitch in action["pitches"]:
                    landings_at[(ev["hand"], pitch)].append(bi)
                    queue = pending.get((ev["hand"], pitch), [])
                    if queue:
                        ranked = sorted(
                            range(len(queue)),
                            key=lambda index: (
                                id(queue[index][1]) not in landing_departures,
                                id(queue[index][1]) in matched_departures,
                                -index,
                            ),
                        )
                        atom = landing_atoms[landing_position[
                            (id(ev), slot_index, pitch)
                        ]]
                        choice = next((
                            index for index in ranked
                            if pending_can_still_close(
                                queue[index][1], atom)
                        ), ranked[0])
                        _departure_bar, departure = queue.pop(choice)
                        departure["assigned"].add(pitch)
                        ev["tie_in_pitches"].add(pitch)
                        landing_departures.add(id(departure))
                        matched_departures.add(id(departure))
                    elif bi == 0 and action["time"] == 0:
                        ev["tie_in_pitches"].add(pitch)
                    else:
                        missing.add(pitch)
                if missing:
                    unmatched_landings.append(
                        (bi, ev, slot_index, missing)
                    )
            else:
                departure = action["site"]
                departures.append(departure)
                for pitch in sorted(departure["pitches"]):
                    pending[(departure["hand"], pitch)].append(
                        (bi, departure)
                    )

            if head_actions and action is head_actions[-1]:
                expire_prior_bar_departures()

    for departure in departures:
        bi = departure["bar"]
        ev = departure["event"]
        pitches = departure["pitches"]
        assigned = departure["assigned"]
        slot = departure.get("slot")
        if assigned:
            if slot is None:
                ev["tie_out_pitches"].update(assigned)
            else:
                slot["tie_out_pitches"].update(assigned)
                if departure.get("slot_index") == len(ev.get("slots") or []) - 1:
                    ev["tie_out_pitches"].update(assigned)
            continue
        at_scope_end = bi == len(bars) - 1
        if at_scope_end:
            if slot is None:
                ev["tie_out_pitches"].update(pitches)
            else:
                slot["tie_out_pitches"].update(pitches)
                if departure.get("slot_index") == len(ev.get("slots") or []) - 1:
                    ev["tie_out_pitches"].update(pitches)
        elif tie_normalizations is not None:
            tie_normalizations.append({
                "kind": "departure_without_matching_landing",
                "bar_index": bi,
                "hand": "l" if ev["hand"] == 0 else "r",
                "departure_voice": int(ev.get("voice", 0)),
                "departure_pitches": sorted(pitches),
                "candidate_landings": [],
                "stripped_markers": 1,
            })
            # Capsule departures exist only at the last member, so the
            # bracket-level flag is the whole record either way.
            ev["tie_out"] = False
        else:
            # Same hand and same pitch is one key.  Striking it again while
            # it sustains leaves two open departures and a single landing,
            # and that landing ends both soundings.
            shared = {
                pitch for pitch in pitches
                if any(bar in (bi, bi + 1)
                       for bar in landings_at.get(
                           (departure["hand"], pitch), []))
            }
            if shared:
                if slot is None:
                    ev["tie_out_pitches"].update(shared)
                else:
                    slot["tie_out_pitches"].update(shared)
                    if departure.get("slot_index") == len(
                            ev.get("slots") or []) - 1:
                        ev["tie_out_pitches"].update(shared)
                continue
            raise ValueError(
                f"bar {bi}: hand {ev['hand']} voice "
                f"{ev.get('voice', 0)} tie departure has no "
                "same-pitch landing"
            )

    for bi, ev, slot_index, missing in unmatched_landings:
        if tie_normalizations is not None:
            tie_normalizations.append({
                "kind": "landing_without_matching_departure",
                "bar_index": bi,
                "hand": "l" if ev["hand"] == 0 else "r",
                "landing_voice": int(ev.get("voice", 0)),
                "onset_tick": str(ev["onset"]),
                "landing_pitches": sorted(missing),
                "stripped_markers": len(missing),
            })
            if ev.get("is_tup"):
                landing_slot = (ev.get("slots") or [])[slot_index]
                landing_slot["attack_ti"] = [
                    bool(tied_in and pitch not in missing)
                    for pitch, tied_in in zip(
                        landing_slot.get("attack", []),
                        landing_slot.get("attack_ti", []),
                        strict=True,
                    )
                ]
            else:
                ev["tie_ins"] = [
                    bool(tied_in and pitch not in missing)
                    for pitch, tied_in in zip(
                        ev.get("pitches", []),
                        ev.get("tie_ins", []),
                        strict=True,
                    )
                ]
        else:
            raise ValueError(
                f"bar {bi}: hand {ev['hand']} voice "
                f"{ev.get('voice', 0)} tie landing has no same-pitch "
                f"departure for {sorted(missing)}"
            )
    seam_plan_cache: Dict[
        Tuple[int, int, int],
        Tuple[Tuple[int, ...], Tuple[int, ...]],
    ] = {}

    def _seam_plan(
        old_count: int,
        seam_count: int,
        new_count: int,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Choose the deterministic right-comb boundary operations."""
        cache_key = (old_count, seam_count, new_count)
        cached = seam_plan_cache.get(cache_key)
        if cached is not None:
            return cached
        default_merges = tuple(
            range(old_count - 2, seam_count - 2, -1))
        default_splits = tuple(range(seam_count - 1, new_count - 1))
        result = (default_merges, default_splits)
        seam_plan_cache[cache_key] = result
        return result

    boundary_plans: Dict[
        Tuple[int, int], Tuple[Tuple[int, ...], Tuple[int, ...]]
    ] = {}
    for bar_index, (_events, _meter, _key, topology) in enumerate(bars):
        next_topology = (
            bars[bar_index + 1][3]
            if bar_index + 1 < len(bars) else None
        )
        for hand in (0, 1):
            old_count = topology["voice_counts"][hand]
            seam_count = topology["next_counts"][hand]
            new_count = (
                next_topology["voice_counts"][hand]
                if next_topology is not None else seam_count
            )
            boundary_plans[(bar_index, hand)] = _seam_plan(
                old_count,
                seam_count,
                new_count,
            )

    # Every token scope starts and ends with one spine per hand.  Boundary
    # count deltas always use the same right-comb form; ties do not choose
    # structural addresses.
    cur_counts: Dict[int, int] = {0: 1, 1: 1}
    cur_cols: List[tuple] = _bar_channels(cur_counts)

    def _split_hand(hand: int, voice: int) -> None:
        nonlocal cur_cols
        target = (hand, voice)
        if target not in cur_cols:
            raise ValueError(
                f"cannot split missing hand {hand} voice {voice}")
        output_lines.append("\t".join(
            "*^" if channel == target else "*" for channel in cur_cols
        ))
        cur_counts[hand] += 1
        cur_cols = _bar_channels(cur_counts)

    def _merge_hand(hand: int, voice: int) -> None:
        nonlocal cur_cols
        if cur_counts[hand] <= 1:
            raise ValueError(f"cannot merge hand {hand} below one voice")
        left = (hand, voice)
        right = (hand, voice + 1)
        if left not in cur_cols or right not in cur_cols:
            raise ValueError(
                f"cannot merge missing hand {hand} voices "
                f"{voice}/{voice + 1}"
            )
        output_lines.append("\t".join(
            "*v" if channel in (left, right) else "*"
            for channel in cur_cols
        ))
        cur_counts[hand] -= 1
        cur_cols = _bar_channels(cur_counts)

    for bar_index, (bar_events, meter_str, key_str, topology) in enumerate(bars):
        voice_counts = topology["voice_counts"]
        next_counts = topology["next_counts"]

        # Barline
        output_lines.append("\t".join(["="] * len(cur_cols)))

        # Schema state is bar-local in the token stream.  Kern prints only
        # changes, at the governed bar head and before any voice expansion.
        if meter_str and meter_str != prev_meter:
            output_lines.append("\t".join([meter_str] * len(cur_cols)))
        if key_str and key_str != prev_key:
            output_lines.append("\t".join([key_str] * len(cur_cols)))
        prev_meter = meter_str or prev_meter
        prev_key = key_str or prev_key

        for hand in (0, 1):
            if voice_counts[hand] < cur_counts[hand]:
                raise ValueError(
                    f"bar {bar_index}: hand {hand} count contracts "
                    f"before its preceding bar tail"
                )
            head_splits = (
                boundary_plans[(bar_index - 1, hand)][1]
                if bar_index > 0 else _seam_plan(
                    1, 1, voice_counts[hand])[1]
            )
            for voice in head_splits:
                _split_hand(hand, voice)
            if cur_counts[hand] != voice_counts[hand]:
                raise ValueError(
                    f"bar {bar_index}: hand {hand} split plan reaches "
                    f"{cur_counts[hand]}, expected {voice_counts[hand]}"
                )
        channels = _bar_channels(voice_counts)
        if cur_cols != channels:
            raise ValueError(
                f"bar {bar_index}: printed layout {cur_cols} does "
                f"not match the bar's channels {channels}"
            )

        # Canonical callers remain strict; generation callers may retain the
        # printable score while recording the non-injective representation.
        meter = None
        m = re.match(r"\*M(\d+)/(\d+)", prev_meter or "")
        if m:
            meter = (int(m.group(1)), int(m.group(2)))
        data_lines = _events_to_kern_lines(bar_events, voice_counts)
        if not _printed_lines_readable(data_lines, bar_events, channels,
                                       meter):
            if readback_mismatches is None:
                raise ValueError(
                    f"bar {bar_index}: printed lines do not read back to the "
                    "decoded events"
                )
            mismatch = _readback_mismatch_record(
                data_lines, bar_events, channels,
            )
            mismatch["bar_index"] = bar_index
            readback_mismatches.append(mismatch)
        output_lines.extend(data_lines)

        for hand in (0, 1):
            if next_counts[hand] > cur_counts[hand]:
                raise ValueError(
                    f"bar {bar_index}: hand {hand} expands at a "
                    f"bar tail"
                )
            for voice in boundary_plans[(bar_index, hand)][0]:
                _merge_hand(hand, voice)
            if cur_counts[hand] != next_counts[hand]:
                raise ValueError(
                    f"bar {bar_index}: hand {hand} merge plan reaches "
                    f"{cur_counts[hand]}, expected {next_counts[hand]}"
                )

    if cur_counts != {0: 1, 1: 1}:
        raise ValueError(f"unbalanced final topology {cur_counts}")
    output_lines.append("\t".join(["=="] * len(cur_cols)))

    header_parts = [
        "\t".join(["**kern"] * n_spines),
        "*staff2\t*staff1",
        "*Ipiano\t*Ipiano",
        "*clefF4\t*clefG2",
    ]
    footer = "\t".join(["*-"] * len(cur_cols))
    header = "\n".join(header_parts)
    body = "\n".join(output_lines)
    return f"{header}\n{body}\n{footer}\n"


if __name__ == "__main__":
    # Voice-bracket test
    test_tokens = [
        "<sos>",
        "<bar>", "<num:4>", "<den:4>", "<key:0>",
        "<pl>", "C", "</pl>", "4", "<pr>", "e", "</pr>", "4",
        "<pl>", "D", "</pl>", "4", "<pr>", "f", "</pr>", "4",
        "<pl>", "E", "</pl>", "4", "<pr>", "g", "</pr>", "4",
        "<pl>", "F", "</pl>", "4", "<pr>", "a", "</pr>", "4",
        "<eos>",
    ]

    print("Test voice-bracket tokens:")
    print(" ".join(test_tokens))
    print("\nReconstructed kern:")
    print(reconstruct_kern_from_bracket_tokens(test_tokens))
