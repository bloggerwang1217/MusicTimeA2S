"""Phase supervision for joint score training and the beat-position target check."""

import logging
import math

import torch
import torch.nn.functional as F

from .tempo_data import (
    BEAT_POSITION_CYCLES,
    BEAT_POSITION_LABELS,
    BEAT_POSITION_TO_CLASS,
)

logger = logging.getLogger(__name__)


PHASE_BLUR_WEIGHTS = (1.0, 0.75, 0.50, 0.25)


# Internal cycle states keep meter history while tying every observation back
# to the existing seven-way acoustic vocabulary. Simple and compound cycles of
# the same large-beat count are distinct states, so the grammar can charge a
# cycle change when a reading slides between them.
BEAT_POSITION_CYCLE_NAMES = tuple(
    spec.name for spec in BEAT_POSITION_CYCLES
)
BEAT_POSITION_CYCLE_LENGTHS = tuple(
    spec.large_beats for spec in BEAT_POSITION_CYCLES
)
BEAT_POSITION_STATE_TO_CYCLE_ID = tuple(
    cycle_id
    for cycle_id, spec in enumerate(BEAT_POSITION_CYCLES)
    for _ in spec.classes
)
BEAT_POSITION_STATE_TO_POSITION = tuple(
    position
    for spec in BEAT_POSITION_CYCLES
    for position in range(len(spec.classes))
)
BEAT_POSITION_STATE_TO_CLASS = tuple(
    BEAT_POSITION_TO_CLASS[label]
    for spec in BEAT_POSITION_CYCLES
    for label in spec.classes
)
BEAT_POSITION_STATE_IS_DOWNBEAT = tuple(
    position == 0 for position in BEAT_POSITION_STATE_TO_POSITION
)
# (cycle, seven-way class) identifies a bar position uniquely, which is how a
# gold latent path is recovered from the two supervision streams.
BEAT_POSITION_STATE_FOR_CYCLE_CLASS = tuple(
    tuple(
        next(
            (
                state
                for state, (cycle_id, class_index) in enumerate(
                    zip(
                        BEAT_POSITION_STATE_TO_CYCLE_ID,
                        BEAT_POSITION_STATE_TO_CLASS,
                        strict=True,
                    )
                )
                if cycle_id == wanted_cycle and class_index == wanted_class
            ),
            -1,
        )
        for wanted_class in range(len(BEAT_POSITION_LABELS))
    )
    for wanted_cycle in range(len(BEAT_POSITION_CYCLES))
)
# Per-bar meter-change rate estimated from the training corpus, counting a
# simple/compound switch at equal large-beat count as a change.
BEAT_POSITION_CYCLE_CHANGE_PROBABILITY = 0.00469


def _build_beat_position_state_edges():
    starts = []
    tails = []
    offset = 0
    successors = [[] for _ in BEAT_POSITION_STATE_TO_CLASS]
    successor_scores = [[] for _ in BEAT_POSITION_STATE_TO_CLASS]
    for length in BEAT_POSITION_CYCLE_LENGTHS:
        starts.append(offset)
        tails.append(offset + length - 1)
        for state in range(offset, offset + length - 1):
            successors[state].append(state + 1)
            successor_scores[state].append(0.0)
        offset += length

    same_score = math.log1p(-BEAT_POSITION_CYCLE_CHANGE_PROBABILITY)
    change_score = math.log(
        BEAT_POSITION_CYCLE_CHANGE_PROBABILITY
        / (len(BEAT_POSITION_CYCLE_LENGTHS) - 1)
    )
    for source_cycle, tail in enumerate(tails):
        for destination_cycle, start in enumerate(starts):
            successors[tail].append(start)
            successor_scores[tail].append(
                same_score
                if source_cycle == destination_cycle
                else change_score
            )

    predecessors = [[] for _ in BEAT_POSITION_STATE_TO_CLASS]
    predecessor_scores = [[] for _ in BEAT_POSITION_STATE_TO_CLASS]
    for source, destinations in enumerate(successors):
        for destination, score in zip(
            destinations, successor_scores[source], strict=True
        ):
            predecessors[destination].append(source)
            predecessor_scores[destination].append(score)
    return tuple(
        tuple(values) for values in (
            *predecessors,
            *predecessor_scores,
            *successors,
            *successor_scores,
        )
    )


_state_edges = _build_beat_position_state_edges()
_n_states = len(BEAT_POSITION_STATE_TO_CLASS)
BEAT_POSITION_STATE_SUCCESSORS = _state_edges[2 * _n_states:3 * _n_states]

if _n_states != 23:
    raise RuntimeError(
        "Beat Position CRF must contain exactly twenty-three states"
    )
if sum(map(len, BEAT_POSITION_STATE_SUCCESSORS)) != 65:
    raise RuntimeError("Beat Position state grammar must contain 65 edges")


def phase_class_centers(gt_phi: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Nearest circular class for a dense phase curve."""
    cycles = torch.remainder(gt_phi.float() / (2.0 * math.pi), 1.0)
    return torch.remainder(torch.round(cycles * n_classes).long(), n_classes)


def gold_phase_log_probs(gt_phi: torch.Tensor, n_classes: int) -> torch.Tensor:
    """[B, T] gold phase -> [B, T, n_classes] log-probabilities carrying the
    same circular blur as the classification target, so a consumer that reads
    the branch's logits through a softmax sees the gold curve in its own units."""
    centers = phase_class_centers(gt_phi, n_classes)
    weight_sum = PHASE_BLUR_WEIGHTS[0] + 2.0 * sum(PHASE_BLUR_WEIGHTS[1:])
    probs = torch.zeros(
        (*centers.shape, n_classes), dtype=torch.float32, device=gt_phi.device
    )
    for offset, weight in enumerate(PHASE_BLUR_WEIGHTS):
        offsets = (0,) if offset == 0 else (-offset, offset)
        for signed_offset in offsets:
            classes = torch.remainder(centers + signed_offset, n_classes)
            probs.scatter_add_(
                -1,
                classes.unsqueeze(-1),
                probs.new_full((*centers.shape, 1), weight / weight_sum),
            )
    return probs.clamp_min(torch.finfo(torch.float32).tiny).log()


def dense_phase_classification_loss(
    phase_logits: torch.Tensor,
    gt_phi: torch.Tensor,
    phase_supervision_mask: torch.Tensor,
) -> torch.Tensor:
    """Oyama-style blurry circular CE on every annotated frame."""
    if phase_logits.ndim != 3:
        logger.warning(
            "phase loss skipped: logits have shape %s, expected [B,T,C]",
            tuple(phase_logits.shape),
        )
        return phase_logits.float().sum() * 0.0
    if (
        phase_logits.shape[:2] != gt_phi.shape
        or gt_phi.shape != phase_supervision_mask.shape
    ):
        logger.warning(
            "phase loss skipped: shape mismatch logits=%s target=%s mask=%s",
            tuple(phase_logits.shape), tuple(gt_phi.shape),
            tuple(phase_supervision_mask.shape),
        )
        return phase_logits.float().sum() * 0.0
    n_classes = phase_logits.shape[-1]
    if n_classes < 7:
        logger.warning(
            "phase loss skipped: classification has %d classes", n_classes
        )
        return phase_logits.float().sum() * 0.0
    centers = phase_class_centers(gt_phi, n_classes)
    log_prob = F.log_softmax(phase_logits.float(), dim=-1)
    per_frame = torch.zeros_like(gt_phi, dtype=torch.float32)
    weight_sum = PHASE_BLUR_WEIGHTS[0] + 2.0 * sum(PHASE_BLUR_WEIGHTS[1:])
    for offset, weight in enumerate(PHASE_BLUR_WEIGHTS):
        offsets = (0,) if offset == 0 else (-offset, offset)
        for signed_offset in offsets:
            classes = torch.remainder(centers + signed_offset, n_classes)
            per_frame -= (weight / weight_sum) * log_prob.gather(
                -1, classes.unsqueeze(-1)
            ).squeeze(-1)
    supervision = phase_supervision_mask.float()
    return (
        (per_frame * supervision).sum()
        / supervision.sum().clamp(min=1.0)
    )


def _beat_position_segments(
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
    frame_domain: torch.Tensor,
    beat_targets: torch.Tensor,
) -> list[torch.Tensor]:
    """Return event-index chains that never cross a supervision hole."""
    if not (
        targets.ndim == 1
        and targets.shape == supervision_mask.shape
        and targets.shape == frame_domain.shape
        and targets.shape == beat_targets.shape
    ):
        raise ValueError("CRF segmentation inputs must be one-dimensional")
    active = supervision_mask.bool() & (targets >= 0)
    frames = torch.nonzero(active, as_tuple=False).flatten()
    if frames.numel() == 0:
        return []

    previous = frames[:-1]
    current = frames[1:]
    frame_hole_prefix = (~frame_domain.bool()).long().cumsum(dim=0)
    holes_before = torch.where(
        previous > 0,
        frame_hole_prefix[(previous - 1).clamp_min(0)],
        torch.zeros_like(previous),
    )
    frame_holes = frame_hole_prefix[current] - holes_before

    beat_prefix = (beat_targets > 0).long().cumsum(dim=0)
    interior_beats = beat_prefix[current - 1] - beat_prefix[previous]
    cuts = torch.nonzero(
        (frame_holes > 0) | (interior_beats > 0), as_tuple=False
    ).flatten() + 1
    boundaries = [0, *cuts.cpu().tolist(), frames.numel()]
    return [
        frames[left:right]
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def beat_position_gold_path_defect(
    targets: torch.Tensor,
    cycle_targets: torch.Tensor,
    supervision_mask: torch.Tensor,
    frame_domain: torch.Tensor,
    beat_targets: torch.Tensor,
) -> str | None:
    """Describe the first gold-path defect in a chunk, or None if it is clean.

    Runs the same segmentation the loss uses, so a chunk that passes here can
    never be silently dropped inside training.
    """
    successors = BEAT_POSITION_STATE_SUCCESSORS
    for order, frames in enumerate(_beat_position_segments(
        targets, supervision_mask, frame_domain, beat_targets
    )):
        classes = targets[frames].long().tolist()
        cycles = cycle_targets[frames].long().tolist()
        if not classes:
            continue
        states = []
        for position, (class_index, cycle_index) in enumerate(
            zip(classes, cycles, strict=True)
        ):
            if not 0 <= cycle_index < len(BEAT_POSITION_CYCLES):
                return f"segment {order} event {position} lacks a cycle id"
            if LEGACY_CYCLE_TO_METER_CYCLE[cycle_index] < 0:
                return (
                    f"segment {order} event {position}: cycle "
                    f"{BEAT_POSITION_CYCLE_NAMES[cycle_index]} has no meter "
                    "state"
                )
            state = BEAT_POSITION_STATE_FOR_CYCLE_CLASS[cycle_index][
                class_index
            ]
            if state < 0:
                return (
                    f"segment {order} event {position}: class {class_index} "
                    f"is not part of cycle "
                    f"{BEAT_POSITION_CYCLE_NAMES[cycle_index]}"
                )
            states.append(state)
        if order == 0 and not BEAT_POSITION_STATE_IS_DOWNBEAT[states[0]]:
            # An anacrusis is left unlabelled upstream, so a chunk's first
            # supervised event must be a bar head.
            return (
                "chunk-opening segment does not start on a bar head "
                f"(class {classes[0]})"
            )
        for position, (source, destination) in enumerate(
            zip(states[:-1], states[1:], strict=True)
        ):
            if destination not in successors[source]:
                return (
                    f"segment {order} event {position}: illegal transition "
                    f"{source} -> {destination}"
                )
    return None


# Bar lengths, in large beats, that carry a meter state.
METER_CYCLE_LENGTHS = (2, 3, 4)


# The legacy supervision streams stay authoritative: cycle ids come from the
# seven-cycle vocabulary and classes from the seven-way labels, so the gold
# path maps through them instead of re-deriving meter geometry.
# A legacy cycle whose bar length has no meter state maps to -1; every consumer
# turns that into the ignore index or an undefined state, and the dataset layer
# skips the chunk before training ever sees it.
LEGACY_CYCLE_TO_METER_CYCLE = tuple(
    METER_CYCLE_LENGTHS.index(spec.large_beats)
    if spec.large_beats in METER_CYCLE_LENGTHS else -1
    for spec in BEAT_POSITION_CYCLES
)


