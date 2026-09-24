"""Score and MIDI timing shared by dataset rendering pipelines."""

import hashlib
import json
import math
import os
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import music21 as m21

from src.audio.render_epr import EPRTempoPoint


_EPR_RENDER_FINGERPRINT_PREFIX = "epr_render_sha256="


def build_tempo_map(score: m21.stream.Score) -> List[Tuple[float, float]]:
    """Tempo marks as [(quarter offset, quarter-note BPM)], always in qpm.

    numberSounding gives quarter-note BPM regardless of beat unit, but
    converter21 often leaves it None even when referent != quarter (!!!OMD
    "[half-dot] = 60" -> number=60, referent=dotted-half, numberSounding=None),
    so the beat unit has to be folded in by hand.
    """
    marks = sorted(
        score.flatten().getElementsByClass(m21.tempo.MetronomeMark),
        key=lambda t: t.offset,
    )
    tempo_map: List[Tuple[float, float]] = []
    for tm in marks:
        if tm.numberSounding:
            qpm = tm.numberSounding
        elif tm.number is not None and tm.referent is not None:
            qpm = tm.number * tm.referent.quarterLength
        else:
            qpm = tm.number
        if qpm is None:
            # converter21 also builds marks from !!!OMD section labels
            # ("TRIO"), which carry no tempo.
            continue
        tempo_map.append((float(tm.offset), float(qpm)))
    return tempo_map or [(0.0, 120.0)]


def initial_qpm(score: m21.stream.Score) -> float:
    """Quarter-note BPM the piece starts at."""
    return build_tempo_map(score)[0][1]


@contextmanager
def scaled_score_tempo(score: m21.stream.Score, scaling: float):
    """Hold the score at a scaled tempo for the duration of the block.

    Speed belongs to the score before a MIDI exists; stretching tick deltas
    afterwards would rescale a reading that was written for another tempo.
    """
    if scaling == 1.0:
        yield
        return
    marks = list(score.flatten().getElementsByClass(m21.tempo.MetronomeMark))
    saved = [(tm, tm.number, tm.numberSounding) for tm in marks]
    try:
        for tm in marks:
            if tm.number is not None:
                tm.number = tm.number * scaling
            if tm.numberSounding is not None:
                tm.numberSounding = tm.numberSounding * scaling
        yield
    finally:
        for tm, number, sounding in saved:
            if number is not None:
                tm.number = number
            if sounding is not None:
                tm.numberSounding = sounding


def extract_measure_offsets(
    score: m21.stream.Score,
) -> List[Dict[str, Any]]:
    """Extract measure boundaries as quarter-note offsets from a music21 Score.

    These offsets are used to inject MIDI marker events at exact tick
    positions, avoiding the tempo-map mismatch between music21's internal
    offset_to_seconds and its MIDI writer.

    Args:
        score: music21 Score object

    Returns:
        List of {"measure": int, "start_qn": float, "end_qn": float,
        "overfull": bool}.
    """
    if not score.parts:
        return []

    part = score.parts[0]
    offsets = []
    time_signature: Optional[m21.meter.TimeSignature] = None
    for measure in part.getElementsByClass(m21.stream.Measure):
        try:
            local_ts = measure.getElementsByClass(m21.meter.TimeSignature)
            if local_ts:
                time_signature = local_ts[0]
            elif time_signature is None:
                time_signature = measure.getContextByClass(
                    m21.meter.TimeSignature)
            start = float(measure.offset)
            duration = float(measure.duration.quarterLength)
            end = start + duration
            metrical_span = (
                4.0 * time_signature.numerator / time_signature.denominator
                if time_signature is not None else None
            )
            overfull = (
                metrical_span is not None
                and duration > metrical_span + _BEAT_GRID_QN_TOLERANCE
            )

            offsets.append({
                "measure": measure.number,
                "start_qn": start,
                "end_qn": end,
                "overfull": overfull,
            })
        except Exception:
            continue
    return offsets


def _beat_grid(ts: m21.meter.TimeSignature) -> tuple:
    """(beats_per_bar, beat_quarterLength) for a time signature.

    Compound when the numerator is a multiple of three above three, simple
    otherwise — so 6/8 is two dotted-quarter beats while 3/8 stays three
    eighth beats.  This is the convention the ASAP annotators used
    (`asap_annotations.json`, beats-per-bar field): notation-driven, not
    tapping-rate-driven, so a fast 3/8 still gets three beats.  music21's
    own `beatDuration` disagrees on 3/8, and ASAP supplies the beat
    supervision for the real half of the training mix — two conventions in
    one training set would label the same meter two ways.
    """
    unit_ql = 4.0 / ts.denominator
    if ts.numerator > 3 and ts.numerator % 3 == 0:
        return ts.numerator // 3, 3.0 * unit_ql
    return ts.numerator, unit_ql


_BEAT_GRID_QN_TOLERANCE = 1.0 / 512.0


def extract_phase_grid_offsets(
    beat_offsets: List[Dict[str, Any]],
    tempo_boundary_qn: Sequence[float],
) -> List[Dict[str, Any]]:
    """Place phase anchors at beats and actual tempo-segment boundaries.

    Phase advances by one cycle from one project beat to the next.  A tempo
    boundary inside that interval gets its fractional cycle from score
    position; this preserves every bend the performance model actually used
    without inventing subdivisions from the written meter.
    """
    if len(beat_offsets) < 2:
        return []

    beat_qn = [float(beat["start_qn"]) for beat in beat_offsets]
    if any(right <= left for left, right in zip(beat_qn, beat_qn[1:])):
        raise ValueError("beat offsets must be strictly increasing")

    anchors = {round(qn, 9): (qn, float(index))
               for index, qn in enumerate(beat_qn)}
    for boundary in tempo_boundary_qn:
        qn = float(boundary)
        if not (beat_qn[0] < qn < beat_qn[-1]):
            continue
        left = bisect_right(beat_qn, qn) - 1
        span = beat_qn[left + 1] - beat_qn[left]
        cycle = left + (qn - beat_qn[left]) / span
        anchors[round(qn, 9)] = (qn, cycle)

    return [
        {"start_qn": qn, "cycle": cycle}
        for qn, cycle in sorted(anchors.values())
    ]


def _merge_grid_cycles(current: float, candidate: float) -> float:
    """Choose one phase when timing quantization merges two grid anchors."""
    current_is_beat = math.isclose(
        current, round(current), rel_tol=0.0, abs_tol=1e-9)
    candidate_is_beat = math.isclose(
        candidate, round(candidate), rel_tol=0.0, abs_tol=1e-9)
    if current_is_beat != candidate_is_beat:
        return candidate if candidate_is_beat else current
    if current_is_beat and not math.isclose(
            current, candidate, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"distinct beat phases share one timing position: "
            f"{current:g}, {candidate:g}")
    return max(current, candidate)


def epr_render_fingerprint(
    xml_path: Path,
    style: str,
    qpm_primo: float,
    *,
    interval_in_16th: int,
) -> str:
    """Identify the score and EPR controls that produced a reusable MIDI."""
    payload = {
        "interval_in_16th": interval_in_16th,
        "qpm_primo": f"{qpm_primo:.12g}",
        "style": style,
        "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_beat_offsets(score: m21.stream.Score) -> List[Dict[str, Any]]:
    """Extract the beat grid as quarter-note offsets from a music21 Score.

    Companion to ``extract_measure_offsets``: same offset domain, same
    marker-injection route, so beat times survive tempo scaling and are read
    back from the MIDI tempo map rather than interpolated between barlines.

    Args:
        score: music21 Score object

    Returns:
        List of {"measure": int, "beat_index": int, "start_qn": float,
        "score_derivable": bool}.  ``beat_index`` is the position in the
        metrical cycle, so index 0 marks a downbeat.

        ``score_derivable`` is False where the bar does not fill its meter,
        so the grid inside it cannot be read off the notation — cadenzas and
        mid-piece partial bars.  ASAP labels such beats "bR" rather than
        dropping them, and this mirrors that: the beat is still emitted, the
        flag says its position is not notation-derived.  The opening
        anacrusis is excluded — the barline after it fixes its grid.
    """
    if not score.parts:
        return []

    part = score.parts[0]
    beats: List[Dict[str, Any]] = []
    ts: Optional[m21.meter.TimeSignature] = None

    for idx, measure in enumerate(part.getElementsByClass(m21.stream.Measure)):
        local_ts = measure.getElementsByClass(m21.meter.TimeSignature)
        if local_ts:
            ts = local_ts[0]
        elif ts is None:
            ts = measure.getContextByClass(m21.meter.TimeSignature)
        if ts is None:
            continue

        try:
            start = float(measure.offset)
            actual = float(measure.duration.quarterLength)
        except Exception:
            continue

        n_beats, beat_ql = _beat_grid(ts)
        if beat_ql <= 0 or actual <= 0:
            continue

        metrical_span = n_beats * beat_ql
        # Sub-1/2048-note discrepancies are serialization noise rather than
        # extra metrical time.  Larger overfills remain explicitly irregular.
        irregular = abs(actual - metrical_span) > _BEAT_GRID_QN_TOLERANCE

        if idx == 0 and actual < metrical_span - _BEAT_GRID_QN_TOLERANCE:
            # Anacrusis: a short opening bar carries the LAST beats of the
            # cycle, so the grid is anchored to the barline that follows it
            # and no downbeat is emitted.  Short bars elsewhere are section
            # joins, not pickups, and keep the normal forward anchoring.
            # A pickup shorter than one beat holds no beat onset at all —
            # an eighth before a 2/4 bar is the second half of beat two.
            k = min(n_beats, int(
                (actual + _BEAT_GRID_QN_TOLERANCE) / beat_ql))
            for j in range(k):
                beats.append({
                    "measure": measure.number,
                    "beat_index": n_beats - k + j,
                    "start_qn": start + actual - (k - j) * beat_ql,
                    "score_derivable": True,
                })
        else:
            # A bar can carry notated overflow, but the next metric cycle
            # still begins at its barline.  Never invent an (N+1)th onset.
            for j in range(n_beats):
                if (j * beat_ql
                        >= actual - _BEAT_GRID_QN_TOLERANCE):
                    break
                beats.append({
                    "measure": measure.number,
                    "beat_index": j,
                    "start_qn": start + j * beat_ql,
                    "score_derivable": not irregular,
                })

    return beats


def epr_seconds_at(
    tempo_points: Sequence[EPRTempoPoint],
    quarter_offset: float,
) -> float:
    """Map a score position through the performance's exact tempo function."""
    if not tempo_points:
        raise ValueError("need at least one tempo point")
    starts = [point.quarter_offset for point in tempo_points]
    index = max(0, bisect_right(starts, quarter_offset) - 1)
    point = tempo_points[index]
    return point.seconds + (quarter_offset - point.quarter_offset) * 60.0 / point.qpm


def inject_measure_markers(
    midi_path: str,
    measure_offsets: List[Dict[str, Any]],
    beat_offsets: Optional[List[Dict[str, Any]]] = None,
    seconds_at: Optional[Any] = None,
    grid_offsets: Optional[List[Dict[str, Any]]] = None,
    epr_fingerprint: Optional[str] = None,
) -> None:
    """Inject marker meta-events at measure boundaries into a MIDI file.

    Markers are placed on a dedicated track so they don't interfere with
    note data.  FluidSynth ignores all meta-events, so the audio output
    is unaffected.  The markers can be read back with
    ``read_measure_times_from_midi`` to obtain precise measure boundary times
    in seconds.

    Args:
        midi_path: Path to the MIDI file (modified in place).
        measure_offsets: Output of ``extract_measure_offsets``.
        beat_offsets: Output of ``extract_beat_offsets``, or None for
            boundaries only.
        grid_offsets: Beat and tempo-boundary phase anchors from the exact
            timing function used by either rendering route.
        seconds_at: Score offset -> seconds.  Required for a performance
            MIDI, whose timing lives in the note events rather than in a
            tempo map, so score offsets no longer locate anything.
        epr_fingerprint: Exact EPR rendering recipe expected when this MIDI is
            reused.  Absent for a direct score render.
    """
    from mido import MidiFile, MidiTrack, MetaMessage, second2tick

    mid = MidiFile(midi_path)
    tpb = mid.ticks_per_beat

    if seconds_at is None:
        def to_tick(quarter_offset: float) -> int:
            return int(round(quarter_offset * tpb))
    else:
        tempos = {msg.tempo for track in mid.tracks for msg in track
                  if msg.type == "set_tempo"}
        if len(tempos) != 1:
            raise RuntimeError(
                f"{Path(midi_path).name}: expected one tempo in a performance "
                f"MIDI, found {len(tempos)}")
        tempo = tempos.pop()

        def to_tick(quarter_offset: float) -> int:
            return int(round(second2tick(
                seconds_at(quarter_offset), tpb, tempo)))

    # Bars first so that a bar and its downbeat, which land on the same tick,
    # keep boundary-before-beat order under the stable sort below.
    events = [(to_tick(m["start_qn"]), f"bar_{m['measure']}")
              for m in measure_offsets]
    # Trailing "_R" echoes ASAP's "bR" label: beat present, position not
    # derivable from the notation.
    events += [(to_tick(b["start_qn"]),
                f"beat_{b['measure']}_{b['beat_index']}"
                + ("" if b.get("score_derivable", True) else "_R"))
               for b in (beat_offsets or [])]
    # Score offsets can be distinct yet land on one discrete MIDI tick.  A
    # beat anchor owns that instant; otherwise the later phase is the
    # right-continuous value after the collapsed boundary.
    grid_by_tick: Dict[int, float] = {}
    for grid_point in grid_offsets or []:
        tick = to_tick(grid_point["start_qn"])
        cycle = float(grid_point["cycle"])
        if tick in grid_by_tick:
            cycle = _merge_grid_cycles(grid_by_tick[tick], cycle)
        grid_by_tick[tick] = cycle
    events += [(tick, f"grid_{cycle:.6f}")
               for tick, cycle in grid_by_tick.items()]
    # End-of-piece marker at the last measure's end
    if measure_offsets:
        events.append((to_tick(measure_offsets[-1]["end_qn"]), "end"))
    events.sort(key=lambda e: e[0])

    marker_track = MidiTrack()
    if seconds_at is not None:
        if not epr_fingerprint:
            raise ValueError("an EPR MIDI requires a rendering fingerprint")
        marker_track.append(MetaMessage(
            "marker",
            text=f"{_EPR_RENDER_FINGERPRINT_PREFIX}{epr_fingerprint}",
            time=0,
        ))
    prev_tick = 0
    for tick, text in events:
        marker_track.append(
            MetaMessage("marker", text=text, time=max(0, tick - prev_tick))
        )
        prev_tick = tick

    mid.tracks.append(marker_track)
    # Publish atomically so a kill mid-write cannot leave a short MIDI standing
    # in for the real one.
    midi_path = Path(midi_path)
    tmp_path = midi_path.with_name(f".{midi_path.name}")
    mid.save(tmp_path)
    os.replace(tmp_path, midi_path)


def read_epr_render_fingerprint(midi_path: str) -> Optional[str]:
    """Read the render provenance embedded in a complete EPR timing MIDI."""
    from mido import MidiFile

    markers = {
        msg.text
        for track in MidiFile(midi_path).tracks
        for msg in track
        if msg.type == "marker"
    }
    fingerprints = {
        marker[len(_EPR_RENDER_FINGERPRINT_PREFIX):]
        for marker in markers
        if marker.startswith(_EPR_RENDER_FINGERPRINT_PREFIX)
    }
    if len(fingerprints) != 1:
        return None
    return fingerprints.pop()


def has_epr_render_fingerprint(
    midi_path: str, expected_fingerprint: str,
) -> bool:
    """Whether a MIDI carries the expected EPR timing and render recipe."""
    return read_epr_render_fingerprint(midi_path) == expected_fingerprint


def read_measure_times_from_midi(midi_path: str) -> List[Dict[str, Any]]:
    """Read measure boundary times (in seconds) from MIDI marker events.

    mido's file iterator automatically applies the MIDI tempo map when
    converting tick deltas to seconds, so the returned times are guaranteed
    to match what FluidSynth renders.

    Args:
        midi_path: Path to a MIDI file containing ``bar_*`` / ``end`` markers.

    Returns:
        List of {"measure": int, "start_sec": float, "end_sec": float}.
        Returns [] if no markers are found.
    """
    from mido import MidiFile

    mid = MidiFile(midi_path)

    # Collect absolute times for each marker
    markers: List[tuple] = []  # (abs_sec, text)
    abs_time = 0.0
    for msg in mid:
        abs_time += msg.time
        if msg.type == "marker":
            markers.append((abs_time, msg.text))

    if not markers:
        return []

    # Keep only boundary markers before pairing: the beat grid also lives on
    # this track, and a beat marker taken as "the next marker" would become
    # the measure's end time, collapsing every bar to zero length.
    bounds = [(t, txt) for t, txt in markers
              if txt.startswith("bar_") or txt == "end"]

    # Build measure list from consecutive bar_* markers
    measures = []
    for i, (t, text) in enumerate(bounds):
        if not text.startswith("bar_"):
            continue
        measure_num = int(text[4:])
        # end_sec is the next boundary marker's time (bar_* or "end")
        if i + 1 < len(bounds):
            end_sec = bounds[i + 1][0]
        else:
            end_sec = t  # fallback: zero-length last measure
        measures.append({
            "measure": measure_num,
            "start_sec": round(t, 4),
            "end_sec": round(end_sec, 4),
        })

    return measures


def attach_measure_supervision_flags(
    audio_measures: List[Dict[str, Any]],
    measure_offsets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach score-derived supervision flags to rendered measure times."""
    if len(audio_measures) != len(measure_offsets):
        raise ValueError(
            "rendered and score measure counts differ: "
            f"{len(audio_measures)} != {len(measure_offsets)}"
        )

    annotated = []
    for timed, scored in zip(audio_measures, measure_offsets, strict=True):
        if timed["measure"] != scored["measure"]:
            raise ValueError(
                "rendered and score measure order differs: "
                f"{timed['measure']} != {scored['measure']}"
            )
        item = dict(timed)
        item["overfull"] = bool(scored.get("overfull", False))
        annotated.append(item)
    return annotated


def read_grid_times_from_midi(midi_path: str) -> List[Dict[str, Any]]:
    """Read the subdivided phase grid (seconds) from ``grid_*`` markers.

    Returns [] for a MIDI without them, which is the normal case for a score
    render.
    """
    from mido import MidiFile

    grid = []
    abs_time = 0.0
    for msg in MidiFile(midi_path):
        abs_time += msg.time
        if msg.type == "marker" and msg.text.startswith("grid_"):
            try:
                cycle = float(msg.text[5:])
            except ValueError:
                continue
            sec = round(abs_time, 6)
            if grid and sec < grid[-1]["sec"]:
                raise ValueError("MIDI phase-grid markers must be time ordered")
            if grid and sec == grid[-1]["sec"]:
                grid[-1]["cycle"] = _merge_grid_cycles(
                    grid[-1]["cycle"], cycle)
            else:
                grid.append({"sec": sec, "cycle": cycle})
    return grid


def read_beat_times_from_midi(midi_path: str) -> List[Dict[str, Any]]:
    """Read beat grid times (in seconds) from MIDI marker events.

    Same route as ``read_measure_times_from_midi``: mido's iterator applies
    the MIDI tempo map, so a written accelerando lands on the beats exactly
    as FluidSynth renders it.

    Args:
        midi_path: Path to a MIDI file containing ``beat_*`` markers.

    Returns:
        List of {"sec": float, "measure": int, "beat_index": int,
        "is_downbeat": bool, "score_derivable": bool}.  Returns [] if no
        beat markers are found (e.g. a MIDI written before the beat grid
        was added).
    """
    from mido import MidiFile

    mid = MidiFile(midi_path)

    beats = []
    abs_time = 0.0
    for msg in mid:
        abs_time += msg.time
        if msg.type != "marker" or not msg.text.startswith("beat_"):
            continue
        fields = msg.text.split("_")
        if len(fields) < 3:
            continue
        try:
            measure_num, beat_index = int(fields[1]), int(fields[2])
        except ValueError:
            continue
        beats.append({
            "sec": round(abs_time, 4),
            "measure": measure_num,
            "beat_index": beat_index,
            "is_downbeat": beat_index == 0,
            "score_derivable": fields[3:] != ["R"],
        })

    return beats
