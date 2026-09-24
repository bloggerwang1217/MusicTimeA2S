"""Frame-level beat, phase, and beat-position supervision."""

import logging
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch

from .foundation import HFT_MEL

logger = logging.getLogger(__name__)

MEL_FPS = HFT_MEL.sample_rate / HFT_MEL.hop_length  # 62.5


BEAT_POSITION_IGNORE_INDEX = -100
BEAT_POSITION_LABELS = (
    "D",
    "W2",
    "W3_2",
    "W3_3",
    "W4_2",
    "S",
    "W4_4",
)
BEAT_POSITION_TO_CLASS = {
    label: class_index
    for class_index, label in enumerate(BEAT_POSITION_LABELS)
}
BEAT_POSITION_DOWNBEAT_CLASS = BEAT_POSITION_TO_CLASS["D"]


@dataclass(frozen=True)
class CycleSpec:
    """One metrical cycle: legacy seven-way classes plus acoustic roles."""

    name: str
    large_beats: int
    compound: bool
    classes: tuple[str, ...]
    roles: tuple[str, ...]


# Role sequences follow the accent profile measured on human performances:
# bass-note placement is the strongest cue, velocity second, onset density
# has none. Duple second beats sit closer to the bar head than 4/4's own
# secondary strong, and triple meters show no stable internal hierarchy.
BEAT_POSITION_CYCLES = (
    CycleSpec("2s", 2, False, ("D", "W2"), ("D", "S")),
    CycleSpec("2c", 2, True, ("D", "W2"), ("D", "S")),
    CycleSpec(
        "3s", 3, False,
        ("D", "W3_2", "W3_3"), ("D", "W", "W"),
    ),
    CycleSpec(
        "3c", 3, True,
        ("D", "W3_2", "W3_3"), ("D", "W", "W"),
    ),
    CycleSpec(
        "4s", 4, False,
        ("D", "W4_2", "S", "W4_4"), ("D", "W", "S", "W"),
    ),
    CycleSpec(
        "4c", 4, True,
        ("D", "W4_2", "S", "W4_4"), ("D", "W", "S", "W"),
    ),
    CycleSpec(
        "5s", 5, False,
        ("D", "W3_2", "W3_3", "S", "W4_4"), ("D", "W", "W", "S", "W"),
    ),
)
BEAT_POSITION_CYCLE_INDEX = {
    (spec.large_beats, spec.compound): index
    for index, spec in enumerate(BEAT_POSITION_CYCLES)
}
if len(BEAT_POSITION_CYCLE_INDEX) != len(BEAT_POSITION_CYCLES):
    raise RuntimeError("Beat Position cycles must have distinct meter keys")
for _spec in BEAT_POSITION_CYCLES:
    if len(_spec.classes) != _spec.large_beats:
        raise RuntimeError(f"{_spec.name} class sequence length mismatch")
    if len(_spec.roles) != _spec.large_beats:
        raise RuntimeError(f"{_spec.name} role sequence length mismatch")
    if len(set(_spec.classes)) != len(_spec.classes):
        # (cycle, class) must identify a position uniquely; the CRF resolves
        # gold latent states through that pair.
        raise RuntimeError(f"{_spec.name} repeats a class inside one bar")

_METER_RE = re.compile(r"^\*M(\d+)/(\d+)$")


def gt_curves(points, start: int, capacity: int):
    """Piecewise tempo and unwrapped phase on the mel-frame grid."""
    if len(points) < 2:
        raise ValueError("at least two annotated phase points are required")
    secs = np.array([point["sec"] for point in points])
    cycles = np.array([
        point["cycle"] if "cycle" in point else float(i)
        for i, point in enumerate(points)
    ])
    frame_times = (start + np.arange(capacity)) / MEL_FPS
    segments = np.searchsorted(secs, frame_times, side="right") - 1
    # A chunk opens at floor(bar_start * fps) while a beat lands on round(),
    # so a piece's very first frame can sit a fraction of a frame ahead of the
    # first annotated point and lose the bar's own first beat. Anything that
    # rounds to that point belongs to the opening segment.
    at_left_edge = (segments < 0) & (
        frame_times > secs[0] - 0.5 / MEL_FPS
    )
    segments = np.where(at_left_edge, 0, segments)
    valid = (segments >= 0) & (segments < len(secs) - 1)
    clipped = np.clip(segments, 0, len(secs) - 2)
    spans = secs[clipped + 1] - secs[clipped]
    cycle_deltas = cycles[clipped + 1] - cycles[clipped]
    omega = 2.0 * math.pi * cycle_deltas / spans
    fractions = (frame_times - secs[clipped]) / spans
    phi = 2.0 * math.pi * (
        cycles[clipped] + cycle_deltas * fractions
    )
    return (
        torch.from_numpy(omega.astype(np.float32)),
        torch.from_numpy(phi.astype(np.float32)),
        torch.from_numpy(valid),
    )


def _trusted_measure_clock(measures, beats) -> list[bool]:
    """Return bars whose two boundaries and interior beats are score-grounded."""
    trusted = []
    for index, measure in enumerate(measures):
        # A crop edge without a beat annotation does not establish a metrical span.
        if (not measure.get("start_is_annotated", True)
                or not measure.get("end_is_annotated", True)):
            trusted.append(False)
            continue
        m_lo = measure["start_sec"] - 1e-3
        m_hi = measure["end_sec"] - 1e-3
        beat_lo = bisect_left(beats, m_lo, key=lambda beat: beat["sec"])
        beat_hi = bisect_left(beats, m_hi, key=lambda beat: beat["sec"])
        bar_beats = beats[beat_lo:beat_hi]
        start_lo = bisect_left(
            beats, measure["start_sec"] - 1e-3,
            key=lambda beat: beat["sec"],
        )
        start_hi = bisect_right(
            beats, measure["start_sec"] + 1e-3,
            key=lambda beat: beat["sec"],
        )
        end_lo = bisect_left(
            beats, measure["end_sec"] - 1e-3,
            key=lambda beat: beat["sec"],
        )
        end_hi = bisect_right(
            beats, measure["end_sec"] + 1e-3,
            key=lambda beat: beat["sec"],
        )
        start_events = beats[start_lo:start_hi]
        end_events = beats[end_lo:end_hi]
        opening_pickup = (
            index == 0
            and bool(bar_beats)
            and min(bar_beats, key=lambda beat: beat["sec"]).get(
                "beat_index", 0
            ) > 0
        )
        start_grounded = opening_pickup or any(
            beat.get("is_downbeat", False)
            and beat.get("score_derivable", True)
            for beat in start_events
        )
        end_grounded = index == len(measures) - 1 or any(
            beat.get("is_downbeat", False)
            and beat.get("score_derivable", True)
            for beat in end_events
        )
        trusted.append(
            not measure.get("overfull", False)
            and bool(bar_beats)
            and all(
                beat.get("score_derivable", True) for beat in bar_beats
            )
            and start_grounded
            and end_grounded
        )
    return trusted


def downbeat_phase_curve(
    grid,
    measures,
    beats,
    start: int,
    capacity: int,
) -> torch.Tensor:
    """One circular cycle per measure, with beats at rational positions.

    The curve is a restriction of the same piecewise-linear beat count used
    by gt_curves. This keeps metrical positions fixed under rubato instead of
    interpolating uniformly in physical time between downbeats.
    """
    secs = np.array([point["sec"] for point in grid])
    cycles = np.array([
        point["cycle"] if "cycle" in point else float(i)
        for i, point in enumerate(grid)
    ])
    frame_times = (start + np.arange(capacity)) / MEL_FPS
    frame_cycles = np.interp(frame_times, secs, cycles)
    bounds = np.array(
        [measure["start_sec"] for measure in measures]
        + [measures[-1]["end_sec"]]
    )
    bound_cycles = np.interp(bounds, secs, cycles)

    # The closing barline follows the final annotated beat by exactly one
    # metrical beat; interpolation would clamp and shorten the final measure.
    if bounds[-1] > secs[-1]:
        bound_cycles[-1] = cycles[-1] + 1.0
    first_bar_beats = [
        beat for beat in beats
        if bounds[0] <= beat["sec"] < bounds[1]
    ]
    anchor = (
        min(first_bar_beats, key=lambda beat: beat["sec"])
        if first_bar_beats else None
    )
    opening_pickup = anchor is not None and anchor.get("beat_index", 0) > 0
    if bounds[0] < secs[0] or opening_pickup:
        if first_bar_beats:
            if "beat_index" not in anchor:
                raise ValueError(
                    "a pickup beat requires its score-derived beat_index"
                )
            anchor_cycle = np.interp(anchor["sec"], secs, cycles)
            bound_cycles[0] = anchor_cycle - anchor["beat_index"]
        else:
            # The masked pickup prefix still needs a finite coordinate.
            bound_cycles[0] = bound_cycles[1] - 1.0

    trusted = np.asarray(_trusted_measure_clock(measures, beats), dtype=bool)
    required_boundaries = np.zeros(len(bounds), dtype=bool)
    required_boundaries[:-1] |= trusted
    required_boundaries[1:] |= trusted
    snapped = np.round(bound_cycles)
    deviation = np.abs(bound_cycles - snapped)
    if np.any(deviation[required_boundaries] > 0.01):
        raise ValueError(
            "trusted measure boundary off the beat grid by "
            f"{float(deviation[required_boundaries].max()):.3f} cycles"
        )

    segment = np.clip(
        np.searchsorted(bounds, frame_times, side="right") - 1,
        0,
        len(measures) - 1,
    )
    # Masked bars still need finite tensors, but their unsupported clock must
    # not influence a later trusted coordinate.
    fraction = np.zeros(capacity, dtype=np.float64)
    for measure_index in np.flatnonzero(trusted):
        span = snapped[measure_index + 1] - snapped[measure_index]
        if span <= 0:
            raise ValueError(
                f"trusted measure {measure_index} has a non-positive beat span"
            )
        frames = segment == measure_index
        fraction[frames] = (
            frame_cycles[frames] - snapped[measure_index]
        ) / span
    return torch.from_numpy(
        (2.0 * math.pi * fraction).astype(np.float32)
    )


@cache
def _measure_meters(
    kern_path: str,
) -> tuple[tuple[int, int, int, bool], ...]:
    """Return (numerator, denominator, large-beat count) per kern measure."""
    from src.score.sanitize_kern import extract_kern_measures

    path = Path(kern_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    measures = extract_kern_measures(path)
    declarations = []
    for line_index, line in enumerate(lines):
        for token in line.split("\t"):
            match = _METER_RE.match(token.strip())
            if match:
                declarations.append(
                    (line_index, int(match.group(1)), int(match.group(2)))
                )
                break
    if not declarations:
        raise ValueError(f"{path} has no kern time signature")

    result = []
    declaration_index = 0
    active = None
    for measure in measures:
        data_line_index = measure["line_start"] - 1
        while (
            declaration_index < len(declarations)
            and declarations[declaration_index][0] < data_line_index
        ):
            _, numerator, denominator = declarations[declaration_index]
            active = (numerator, denominator)
            declaration_index += 1
        # Some kern pickups precede their first *M declaration. The first
        # written meter governs that anacrusis.
        if active is None:
            _, numerator, denominator = declarations[0]
        else:
            numerator, denominator = active
        compound = numerator > 3 and numerator % 3 == 0
        large_beats = numerator // 3 if compound else numerator
        result.append((numerator, denominator, large_beats, compound))
    return tuple(result)


def _beat_position_class(cycle_index: int, bar_beat_index: int):
    spec = BEAT_POSITION_CYCLES[cycle_index]
    if not 0 <= bar_beat_index < len(spec.classes):
        return None
    return BEAT_POSITION_TO_CLASS[spec.classes[bar_beat_index]]


def beat_position_targets(
    entry: dict,
    beats,
    measures,
    start: int,
    capacity: int,
    first_measure: int,
    last_measure: int,
    supervision_mask: torch.Tensor,
    coordinate_domain: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build beat-position classes and cycle ids from kern meter geometry."""
    kern_path = entry.get("kern_path") or entry.get("kern_gt_path")
    if kern_path:
        meter_timeline = _measure_meters(str(Path(kern_path).resolve()))
        if len(meter_timeline) < len(measures):
            raise ValueError(
                f"{entry.get('id', '?')}: kern/audio measure mismatch "
                f"{len(meter_timeline)} != {len(measures)}"
            )
        # The alignment contract admits kern surplus at the end only (the
        # performed terminal bar has no audio_measures entry); index pairing
        # holds for every audio measure, so the prefix is the timeline.
        meter_timeline = meter_timeline[:len(measures)]
    else:
        meter = entry.get("meter")
        match = re.fullmatch(r"(\d+)/(\d+)", str(meter or ""))
        if match is None:
            raise ValueError(
                f"{entry.get('id', '?')} lacks kern_path and explicit meter"
            )
        numerator, denominator = map(int, match.groups())
        compound = numerator > 3 and numerator % 3 == 0
        large_beats = numerator // 3 if compound else numerator
        meter_timeline = (
            (numerator, denominator, large_beats, compound),
        ) * len(measures)

    target = torch.full(
        (capacity,), BEAT_POSITION_IGNORE_INDEX, dtype=torch.long
    )
    cycle_target = torch.full(
        (capacity,), BEAT_POSITION_IGNORE_INDEX, dtype=torch.long
    )
    defined = torch.zeros(capacity, dtype=torch.bool)
    frame_defined = torch.zeros(capacity, dtype=torch.bool)
    for measure_index in range(first_measure, last_measure + 1):
        measure = measures[measure_index]
        measure_lo = max(
            0, round(measure["start_sec"] * MEL_FPS) - start
        )
        measure_hi = min(
            capacity, round(measure["end_sec"] * MEL_FPS) - start
        )
        if (
            measure_hi <= measure_lo
            or not supervision_mask[measure_lo:measure_hi].any()
        ):
            continue
        _, _, large_beats, compound = meter_timeline[measure_index]
        cycle_index = BEAT_POSITION_CYCLE_INDEX.get((large_beats, compound))
        if cycle_index is None:
            continue

        measure_start = measure["start_sec"]
        measure_end = measure["end_sec"]
        beat_lo = bisect_left(
            beats, measure_start - 1e-3, key=lambda beat: beat["sec"]
        )
        beat_hi = bisect_left(
            beats, measure_end - 1e-3, key=lambda beat: beat["sec"]
        )
        onsets = [
            beat["sec"]
            for beat in beats[beat_lo:beat_hi]
            if measure_start - 1e-3 <= beat["sec"] < measure_end - 1e-3
        ]
        starts_on_boundary = bool(
            onsets and abs(onsets[0] - measure_start) <= 1e-3
        )
        meter_conflict = (
            len(onsets) != large_beats or not starts_on_boundary
        )
        if measure_index > 0 and meter_conflict:
            return None
        if len(onsets) > large_beats:
            return None
        # A bar whose own downbeat is unsupervised gives no position anchor:
        # an anacrusis states only the tail of a bar the audio never plays,
        # and a piece's opening bar can fall outside the phase domain. Both
        # stay unlabelled, which keeps every supervised chunk opening on a
        # bar head.
        head_frame = round(onsets[0] * MEL_FPS) - start if onsets else None
        if meter_conflict or head_frame is None or not (
            0 <= head_frame < capacity and supervision_mask[head_frame]
        ):
            continue

        frame_defined[measure_lo:measure_hi] = coordinate_domain[
            measure_lo:measure_hi
        ]
        first_bar_beat = large_beats - len(onsets)
        for local_index, onset in enumerate(onsets):
            bar_beat_index = first_bar_beat + local_index
            class_index = _beat_position_class(cycle_index, bar_beat_index)
            if class_index is None:
                continue
            frame = round(onset * MEL_FPS) - start
            if 0 <= frame < capacity:
                target[frame] = class_index
                cycle_target[frame] = cycle_index
                defined[frame] = True

    defined &= supervision_mask
    target[~defined] = BEAT_POSITION_IGNORE_INDEX
    cycle_target[~defined] = BEAT_POSITION_IGNORE_INDEX
    return target, cycle_target, defined, frame_defined


def build_emission_targets(
    entry: dict,
    start: int,
    scope_end: int,
    capacity: int,
    n_real: int,
    first_measure: int,
    last_measure: int,
    include_downbeat: bool = False,
) -> dict:
    """Build sparse event targets and their shared supervision domain."""
    beats = entry.get("audio_beats") or []
    measures = entry.get("audio_measures") or []
    if len(beats) < 2:
        raise ValueError(f"{entry.get('id', '?')} lacks clock annotations")
    if not measures:
        raise ValueError(f"{entry.get('id', '?')} lacks audio measures")

    scope_len = max(0, min(capacity, scope_end - start, n_real))
    beat_target = torch.zeros(capacity)
    loss_weight = torch.zeros(capacity)
    loss_weight[:scope_len] = 1.0
    frame_valid = torch.zeros(capacity, dtype=torch.bool)
    frame_valid[:scope_len] = True
    down_target = torch.zeros(capacity) if include_downbeat else None

    # Kern measure geometry remains authoritative for split bars. The opening
    # rendered scope can begin after metric silence, so its retained beat
    # index is needed to avoid turning an anacrusis into a downbeat.
    measure_starts = np.array([m["start_sec"] for m in measures])

    # Long pieces can contain thousands of beats. Restrict the scan to a
    # deliberately over-wide time slice, then retain the existing rounded
    # frame test below so this is exactly equivalent at half-frame boundaries.
    candidate_lo = bisect_left(
        beats, (start - 1) / MEL_FPS, key=lambda beat: beat["sec"]
    )
    candidate_hi = bisect_right(
        beats,
        (start + scope_len + 1) / MEL_FPS,
        key=lambda beat: beat["sec"],
    )
    for beat in beats[candidate_lo:candidate_hi]:
        relative_frame = round(beat["sec"] * MEL_FPS) - start
        if not 0 <= relative_frame < scope_len:
            continue
        # A silent beat is still a beat — rests keep full supervision; trust
        # is decided per bar below, never per beat.
        beat_target[relative_frame] = 1.0
        if include_downbeat:
            at_measure_start = (
                np.abs(measure_starts - beat["sec"]).min() <= 1e-3
            )
            opening_pickup = (
                abs(measure_starts[0] - beat["sec"]) <= 1e-3
                and beat.get("beat_index", 0) > 0
            )
            if at_measure_start and not opening_pickup:
                down_target[relative_frame] = 1.0

    if not (0 <= first_measure <= last_measure < len(measures)):
        raise ValueError(
            f"invalid measure scope {first_measure}:{last_measure} for "
            f"{entry.get('id', '?')}"
        )
    # Bar-level trust requires explicit score-grounded boundaries. Pickups and
    # all-rest bars remain supervised when their clock events are present.
    trusted_measures = _trusted_measure_clock(measures, beats)
    for mi in range(first_measure, last_measure + 1):
        measure = measures[mi]
        if trusted_measures[mi]:
            continue
        # Crop-only tails include the opening fractional frame, matching the
        # score chunk's floor convention without leaking a partial beat label.
        first_frame = (
            round(measure["start_sec"] * MEL_FPS)
            if (measure.get("start_is_annotated", True)
                and measure.get("end_is_annotated", True))
            else int(measure["start_sec"] * MEL_FPS)
        )
        lo = max(0, first_frame - start)
        hi = min(scope_len, round(measure["end_sec"] * MEL_FPS) - start)
        if hi <= lo:
            continue
        beat_target[lo:hi] = 0.0
        if include_downbeat:
            down_target[lo:hi] = 0.0
        loss_weight[lo:hi] = 0.0

    targets = {
        "beat_target": beat_target,
        "loss_weight": loss_weight,
        "frame_valid": frame_valid,
    }
    if include_downbeat:
        targets["down_target"] = down_target
    return targets


def build_clock_targets(
    entry: dict,
    start: int,
    scope_end: int,
    capacity: int,
    n_real: int,
    first_measure: int,
    last_measure: int,
    include_downbeat: bool = False,
) -> dict | None:
    """Project rendered beat/phase annotations onto one score-aligned chunk.

    Supervision trust is decided per bar, not per beat.
    """
    beats = entry.get("audio_beats") or []
    grid = entry.get("audio_grid") or beats
    measures = entry.get("audio_measures") or []
    if len(grid) < 2:
        raise ValueError(f"{entry.get('id', '?')} lacks clock annotations")

    targets = build_emission_targets(
        entry,
        start=start,
        scope_end=scope_end,
        capacity=capacity,
        n_real=n_real,
        first_measure=first_measure,
        last_measure=last_measure,
        include_downbeat=include_downbeat,
    )
    gt_omega, gt_phi, phase_supervision_mask = gt_curves(
        grid, start, capacity
    )
    phase_supervision_mask &= targets["frame_valid"]
    phase_supervision_mask &= targets["loss_weight"] > 0
    targets.update(
        gt_omega=gt_omega,
        gt_phi=gt_phi,
        gt_downbeat_phi=downbeat_phase_curve(
            grid,
            measures,
            beats,
            start,
            capacity,
        ),
        phase_supervision_mask=phase_supervision_mask,
        n_real_frames=n_real,
    )
    beat_position_readout = beat_position_targets(
        entry,
        beats,
        measures,
        start,
        capacity,
        first_measure,
        last_measure,
        phase_supervision_mask,
        targets["loss_weight"] > 0,
    )
    if beat_position_readout is None:
        return None
    (
        gt_beat_position,
        gt_beat_cycle,
        beat_position_mask,
        beat_position_frame_mask,
    ) = beat_position_readout
    if (beat_position_mask & ~(targets["beat_target"] > 0)).any():
        raise ValueError(
            f"{entry.get('id', '?')}: beat-position supervision is not a "
            "subset of beat supervision"
        )
    if (beat_position_mask & ~beat_position_frame_mask).any():
        raise ValueError(
            f"{entry.get('id', '?')}: beat-position event supervision is "
            "outside the Beat PE frame domain"
        )
    if include_downbeat:
        derived_downbeat = (
            gt_beat_position == BEAT_POSITION_DOWNBEAT_CLASS
        )
        raw_downbeat = targets["down_target"] > 0
        if ((derived_downbeat != raw_downbeat) & beat_position_mask).any():
            raise ValueError(
                f"{entry.get('id', '?')}: beat-position D disagrees with "
                "the authoritative measure boundary"
            )
    # Deferred: the loss module reads this module's cycle vocabulary.
    from .tempo_losses import beat_position_gold_path_defect

    defect = beat_position_gold_path_defect(
        gt_beat_position,
        gt_beat_cycle,
        beat_position_mask,
        beat_position_frame_mask,
        targets["beat_target"],
    )
    if defect is not None:
        # Skipping the whole chunk keeps every surviving chunk's gold path
        # legal, so training never has to drop a segment mid-run.
        logger.warning(
            "%s frames %d+%d: beat-position chunk skipped (%s)",
            entry.get("id", "?"),
            start,
            capacity,
            defect,
        )
        return None
    targets.update(
        gt_beat_position=gt_beat_position,
        gt_beat_cycle=gt_beat_cycle,
        beat_position_mask=beat_position_mask,
        beat_position_frame_mask=beat_position_frame_mask,
    )

    audio_bar_index = torch.zeros(capacity, dtype=torch.long)
    for local_index, measure_index in enumerate(
        range(first_measure, last_measure + 1)
    ):
        measure = measures[measure_index]
        lo = max(0, round(measure["start_sec"] * MEL_FPS) - start)
        hi = min(capacity, round(measure["end_sec"] * MEL_FPS) - start)
        if hi <= lo:
            raise ValueError(
                f"{entry.get('id', '?')}: measure {measure_index} has no "
                "frames in its bar-aligned clock scope"
            )
        audio_bar_index[lo:hi] = local_index
    targets["gt_audio_bar_index"] = audio_bar_index
    return targets


