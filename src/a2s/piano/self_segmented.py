"""Self-segmented whole-piece inference.

The bar mode in `inference.py` takes every scope origin from annotated
`audio_measures`. By default, origins here come from the model's own downbeat-phase
posterior.  Overlapping windows are read at the trained stride; each window's
circular first moment is taken in the window's own frame (the branch is
trained with a run-in prefix, so a window is grounded wherever it opens) and
the moments of the windows covering a frame are averaged. Forward crossings
of phase zero are retained in time order with a minimum spacing of 400 ms.
Separate meter/key Viterbi paths constrain the schema before note generation.
Each decoder window recomputes its native local phase posterior. External
tracker and annotation conditions can supply alternative boundaries and
coordinates.

One whole-piece kern per performance is written by concatenating the
five-bar scope outputs at barlines. Retained suffixes carry the spine paths
active at their opening boundary. A scope the decoder cannot reconstruct leaves its
bars as whole-bar rests so the bar count still follows the predicted
timeline; the timeline log records every such bar.

Usage:
    poetry run python -m src.a2s.piano.self_segmented \
        --checkpoint checkpoints/<arm>/best.pt \
        --config configs/piano_2gpu.yaml \
        --manifest data/experiments/asap102/test_manifest.json \
        --manifest-dir data/experiments/asap102 \
        --output-dir data/experiments/asap102/test_kern_pred_self_segmented
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.a2s.piano.inference import (
    COORDINATE_INTERVENTIONS,
    DECODE_STATS,
    MEL_FPS,
    CoordinateIntervention,
    _METER_RE,
    _accepted_prefix_token_ids,
    _bars_available_from_downbeats,
    _build_decode_constraints,
    _parse_fragment_text,
    _postprocess_bar_decode_entry,
    generate_kern_chunk,
    iter_kern_piece_from_downbeats,
    load_model,
)
from src.a2s.piano.tokenizer import KernTokenizer
from src.a2s.piano.schema_inference import (
    SCOPE_BARS, _slice_with_margin, decide_sticky_timeline, scope_plan,
)

logger = logging.getLogger(__name__)

EVENT_TOLERANCE_FRAMES = 4
MIN_DOWNBEAT_GAP_FRAMES = 25
PREDICTED_BOUNDARY_SOURCE = "forward_crossings_400ms"
TIMELINE_FILE = "_downbeat_timeline.jsonl"
SUMMARY_FILE = "_self_segmented_summary.json"
DECODE_LOG_FILE = "_decode_log.jsonl"
MODEL_DECODE_FILE = "_model_decode.jsonl"
ACR_ROWS_FILE = "acr_rows.json"


def acr_rows_from_timeline(timeline_path: Path, manifest: List[dict]) -> dict:
    """ACR census rows for a written downbeat timeline.

    Reference barlines are `audio_measures` start times, the same source the
    pre-segmented five-bar windows and the bar-aligned chunker cut on; the
    per-bar `is_downbeat` beat flags are a different list (they carry no beat
    inside an opening pickup bar, and on ASAP they disagree by seconds).  Bar
    level only: the phase branch predicts one cycle per bar, so there is no
    predicted beat sequence to census.
    """
    by_id = {item["id"]: item for item in manifest}
    rows, skipped = [], []
    for line in Path(timeline_path).read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        item = by_id.get(entry["id"])
        if item is None:
            skipped.append({"id": entry["id"], "reason": "not in manifest"})
            continue
        reference = [
            float(measure["start_sec"])
            for measure in (item.get("audio_measures") or [])
        ]
        if not reference:
            skipped.append({"id": entry["id"], "reason": "no annotated measures"})
            continue
        rows.append({
            "id": entry["id"],
            "work": re.sub(r"_v\d+$", "", entry["id"].split("~", 1)[0]),
            "level": "bar",
            "status": entry.get("status"),
            "predicted_sec": [float(v) for v in entry.get("downbeat_sec", [])],
            "reference_sec": reference,
        })
    return {"rows": rows, "skipped": skipped}


# =============================================================================
# Phase timeline
# =============================================================================


def phase_first_moment(logits: torch.Tensor) -> torch.Tensor:
    """Expected e^{i phi} per frame from categorical phase logits."""
    probability = logits.float().softmax(-1)
    theta = 2 * math.pi * torch.arange(
        logits.shape[-1], device=logits.device
    ) / logits.shape[-1]
    real = probability @ torch.cos(theta)
    imag = probability @ torch.sin(theta)
    return torch.complex(real, imag)


def _window_vectors(
    model,
    mel: torch.Tensor,
    starts: List[int],
    chunk_frames: int,
    batch_size: int,
    device: str,
) -> List[torch.Tensor]:
    vectors: List[torch.Tensor] = []
    use_autocast = str(device).startswith("cuda")
    for batch_start in range(0, len(starts), batch_size):
        selected = starts[batch_start:batch_start + batch_size]
        scopes = torch.cat(
            [_slice_with_margin(mel, start, chunk_frames) for start in selected],
            dim=0,
        ).to(device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=use_autocast
        ):
            encoded = model.encode_frames(scopes)
            logits = model.tempo(encoded)
        vectors.extend(phase_first_moment(logits).cpu())
    return vectors


def averaged_first_moment(
    model,
    mel: torch.Tensor,
    n_frames: int,
    chunk_frames: int,
    stride_frames: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Per-frame expected e^{i phi}, averaged over the windows covering the frame.

    No window is rotated onto its neighbour: each window keeps its own phase
    origin, so the average carries no seam that could accumulate along the
    piece.
    """
    starts = list(range(0, n_frames, stride_frames))
    vectors = _window_vectors(model, mel, starts, chunk_frames, batch_size, device)
    total = torch.zeros(n_frames, dtype=torch.complex64)
    count = torch.zeros(n_frames)
    for start, vector in zip(starts, vectors):
        length = min(len(vector), n_frames - start)
        total[start:start + length] += vector[:length]
        count[start:start + length] += 1.0
    return total / count.clamp_min(1.0)


def forward_downbeat_frames(moment: np.ndarray) -> List[int]:
    """Read phase-zero crossings without imposing a complete turn between them."""
    z = np.asarray(moment, dtype=np.complex128)
    index = np.flatnonzero((z[:-1].imag <= 0) & (z[1:].imag > 0))
    fraction = -z[index].imag / (z[index + 1].imag - z[index].imag)
    real = z[index].real * (1 - fraction) + z[index + 1].real * fraction
    candidates = np.unique(np.rint(index[real > 0] + fraction[real > 0]).astype(int))
    kept = []
    for frame in candidates:
        if not kept or frame - kept[-1] >= MIN_DOWNBEAT_GAP_FRAMES:
            kept.append(int(frame))
    return kept


def predict_downbeat_timeline(
    model,
    mel: torch.Tensor,
    n_frames: int,
    chunk_frames: int,
    stride_frames: int,
    batch_size: int,
    device: str,
) -> dict:
    """Read downbeats from the uniformly averaged circular first moment."""
    moment = averaged_first_moment(
        model, mel, n_frames, chunk_frames, stride_frames, batch_size, device
    ).numpy()
    return {
        "windows": len(range(0, n_frames, stride_frames)),
        "downbeat_frames": forward_downbeat_frames(moment),
        "mean_concentration": float(np.abs(moment).mean()),
    }


def event_f1(predicted: List[int], target: List[int], tolerance: int) -> dict:
    remaining = list(range(len(target)))
    matches = 0
    for prediction in predicted:
        candidates = [
            index for index in remaining
            if abs(target[index] - prediction) <= tolerance
        ]
        if not candidates:
            continue
        selected = min(candidates, key=lambda index: abs(target[index] - prediction))
        remaining.remove(selected)
        matches += 1
    precision = matches / len(predicted) if predicted else 0.0
    recall = matches / len(target) if target else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matches": matches}


# =============================================================================
# Whole-piece assembly
# =============================================================================


_RECIP_TABLE: List[Tuple[Fraction, str]] = sorted(
    [
        (Fraction(3, 2 * base), f"{base}.") for base in (1, 2, 4, 8, 16, 32)
    ] + [
        (Fraction(1, base), f"{base}") for base in (1, 2, 4, 8, 16, 32)
    ],
    key=lambda item: item[0],
    reverse=True,
)


def rest_recips_for_meter(meter_line: str) -> List[str]:
    """Kern rest durations that exactly fill one bar of the given meter."""
    match = _METER_RE.match(meter_line)
    if match is None:
        return ["1"]
    remaining = Fraction(int(match.group(1)), int(match.group(2)))
    output: List[str] = []
    while remaining > 0:
        for value, recip in _RECIP_TABLE:
            if value <= remaining:
                output.append(recip)
                remaining -= value
                break
        else:
            break
    return output or ["1"]


def _block_interpretations(block: List[str]) -> Tuple[Optional[str], Optional[str]]:
    meter = key = None
    for row in block[1:]:
        first = row.split("\t", 1)[0]
        if first.startswith("*M") and _METER_RE.match(first):
            meter = first
        elif first.startswith("*k["):
            key = first
    return meter, key


DEFAULT_HEADER = [
    "**kern\t**kern",
    "*staff2\t*staff1",
    "*Ipiano\t*Ipiano",
    "*clefF4\t*clefG2",
]


def _rest_block(meter: str, key: str) -> List[str]:
    rows = ["=\t=", f"{meter}\t{meter}", f"{key}\t{key}"]
    for recip in rest_recips_for_meter(meter):
        rows.append(f"{recip}r\t{recip}r")
    return rows


def _fragment_header(text: str) -> List[str]:
    """Interpretation rows the reconstructor writes before its first barline."""
    header = []
    for raw in text.splitlines():
        first = raw.split("\t", 1)[0].strip()
        if first.startswith("="):
            break
        if raw and first != "*-":
            header.append(raw)
    return header


def _bar_blocks(text: str) -> List[List[str]]:
    """Measure blocks of a fragment without its closing `==` terminator."""
    _key, _meter, blocks = _parse_fragment_text(text)
    return [
        block for block in blocks
        if not block[0].split("\t", 1)[0].startswith("==")
    ]


def assemble_whole_piece(
    scope_outputs: List[dict],
    n_bars: int,
) -> Tuple[str, List[int]]:
    """Concatenate scope kern texts into one piece; return text and rest-filled bars.

    Each scope output is `{"start", "keep_from", "kern"}` plus an optional
    `"bars"` (default five) where `kern` is the reconstructed fragment of that
    many bars or None when the scope failed.  Bars below `keep_from` belong to
    an earlier scope (the tail scope and previous-bar retries overlap).  Barlines
    stay unnumbered and the piece closes with `==`, as the reconstructor writes
    them; the reconstructor's meter/key rows travel inside each bar block.
    A retained suffix inherits the meter/key in force at its opening: the
    reconstructor prints them only where they change, so their declarations
    may sit in a discarded bar.
    """
    from src.score.kern_postprocess import _active_spine_path_rows

    bars: Dict[int, Optional[List[str]]] = {index: None for index in range(n_bars)}
    prefixes: Dict[int, List[str]] = {}
    carried: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
    header: Optional[List[str]] = None
    for scope in scope_outputs:
        start = int(scope["start"])
        keep_from = int(scope["keep_from"])
        text = scope["kern"]
        if not text:
            continue
        blocks = _bar_blocks(text)
        expected_bars = int(scope.get("bars", SCOPE_BARS))
        if len(blocks) != expected_bars:
            raise ValueError(
                f"scope at bar {start} reconstructed {len(blocks)} bars, "
                f"expected {expected_bars}"
            )
        if header is None:
            header = _fragment_header(text) or list(DEFAULT_HEADER)
        if keep_from > start:
            source_lines = text.splitlines(keepends=True)
            bar_lines = [
                i for i, line in enumerate(source_lines)
                if line.split("\t", 1)[0].strip().startswith("=")
            ]
            # A retained tail may begin inside a split started in its discarded prefix.
            prefixes[keep_from] = [
                row.rstrip("\r\n") for row in _active_spine_path_rows(
                    source_lines, bar_lines[0], bar_lines[keep_from - start],
                )
            ]
            meter = key = None
            for block in blocks[:keep_from - start]:
                block_meter, block_key = _block_interpretations(block)
                meter = block_meter or meter
                key = block_key or key
            carried[keep_from] = (meter, key)
        for offset, block in enumerate(blocks):
            bar_index = start + offset
            if bar_index < keep_from or bar_index >= n_bars:
                continue
            bars[bar_index] = block

    lines = list(header or DEFAULT_HEADER)
    current_meter, current_key = "*M4/4", "*k[]"
    rest_filled: List[int] = []
    for bar_index in range(n_bars):
        block = bars[bar_index]
        if block is None:
            rest_filled.append(bar_index)
            lines.extend(_rest_block(current_meter, current_key))
            continue
        meter, key = _block_interpretations(block)
        if bar_index in carried:
            block = list(block)
            width = len(block[0].split("\t"))
            carried_meter, carried_key = carried[bar_index]
            row = 1
            if meter is None and carried_meter and carried_meter != current_meter:
                block.insert(row, "\t".join([carried_meter] * width))
                meter = carried_meter
            if row < len(block) and block[row].split("\t", 1)[0].startswith("*M"):
                row += 1
            if key is None and carried_key and carried_key != current_key:
                block.insert(row, "\t".join([carried_key] * width))
                key = carried_key
        current_meter = meter or current_meter
        current_key = key or current_key
        lines.extend(prefixes.get(bar_index, []))
        lines.extend(block)
    lines.append("==\t==")
    lines.append("*-\t*-")
    return "\n".join(lines) + "\n", rest_filled


# =============================================================================
# Driver
# =============================================================================


# =============================================================================
# Whole-piece cursor decoding
# =============================================================================


def _decode_from_bar(
    *,
    model,
    mel: torch.Tensor,
    downbeat_frames: List[int],
    bar: int,
    requested_bars: int,
    tokenizer: KernTokenizer,
    chunk_frames: int,
    max_len: int,
    device: str,
    coordinate_intervention: Optional[CoordinateIntervention],
    manifest_item: dict,
) -> Tuple[List[int], List[float], dict]:
    """One grammar-constrained greedy decode from BOS on the chunk opening at `bar`."""
    mel_chunk = _slice_with_margin(mel, downbeat_frames[bar], chunk_frames)
    mel_input = mel_chunk if mel_chunk.dim() == 3 else mel_chunk.unsqueeze(0)
    DECODE_STATS.update({
        "masked_argmax": 0, "steps": 0, "max_voice_width": 1,
        "boundary_completed_bars": 0, "stop_reason": "max_len", "seam_repairs": 0,
    })
    token_ids, log_probs = generate_kern_chunk(
        model, mel_input, tokenizer, max_len, 1,
        device=device,
        stop_after_n_bars=requested_bars,
        memory_moments_override=(
            coordinate_intervention.moments(
                manifest_item, downbeat_frames[bar], chunk_frames, device,
            ) if coordinate_intervention is not None else None
        ),
    )
    stats = {
        "boundary_completed_bars": int(DECODE_STATS["boundary_completed_bars"]),
        "stop_reason": DECODE_STATS["stop_reason"],
        "masked_argmax": DECODE_STATS["masked_argmax"],
        "seam_repairs": DECODE_STATS["seam_repairs"],
        "steps": DECODE_STATS["steps"],
    }
    return token_ids, log_probs, stats


def _longest_reconstructable_prefix(
    entry: dict,
    tokenizer: KernTokenizer,
    constraints_cpu: dict,
    first_bar: int,
    most_bars: int,
    fewest_bars: int,
) -> Tuple[int, str, Optional[dict], List[dict]]:
    """Longest bar prefix, from `most_bars` down to `fewest_bars`, whose tokens
    reconstruct on their own into exactly that many bar blocks."""
    rejected = []
    for bars in range(most_bars, fewest_bars - 1, -1):
        candidate = dict(entry)
        candidate["measures"] = [first_bar, first_bar + bars - 1]
        candidate["skip_trailing_bar_dedup"] = True
        kern, _, log, _ = _postprocess_bar_decode_entry(
            candidate, tokenizer, constraints_cpu,
        )
        try:
            _accepted_prefix_token_ids(entry["token_ids"], tokenizer, bars)
        except ValueError as error:
            rejected.append({"bars": bars, "reason": str(error)})
            continue
        # The piece is assembled from these fragments, so a fragment must hold
        # exactly the bars it claims.
        blocks = _bar_blocks(kern) if kern else []
        if kern and len(blocks) == bars:
            return bars, kern, log, rejected
        rejected.append({
            "bars": bars,
            "reason": log.get("fail", f"reconstruction returned {len(blocks)} bars"),
        })
    return 0, "", None, rejected


def decode_piece_with_cursor(
    *,
    perf_id: str,
    model,
    mel: torch.Tensor,
    downbeat_frames: List[int],
    tokenizer: KernTokenizer,
    constraints_cpu: dict,
    chunk_frames: int,
    trust_frames: int,
    max_len: int,
    device: str,
    coordinate_intervention: Optional[CoordinateIntervention],
    manifest_item: dict,
    model_decode_log,
    decode_log,
) -> Tuple[List[dict], List[int], int]:
    """Decode a whole piece with one cursor over its bars.

    Each chunk opens at the cursor's bar, asks for every bar whose closing
    boundary lies inside the trust region (at least one), and the cursor moves
    past the accepted bars.  Nothing carries over between chunks except the
    cursor: every chunk decodes from BOS.  A chunk that yields no bar falls
    back to a decode from the previous bar that discards its first bar; if
    that also fails, the cursor's bar alone is left as a gap.

    Returns (fragments for `assemble_whole_piece`, failed bars, decode calls).
    """
    n_bars = len(downbeat_frames)
    if any(right <= left for left, right in zip(downbeat_frames, downbeat_frames[1:])):
        raise ValueError(f"{perf_id}: downbeat positions must be strictly increasing")
    fragments: List[dict] = []
    failed_bars: List[int] = []
    calls = 0
    cursor = 0

    def decode_record(bar, requested, token_ids, log_probs, stats, **extra):
        return {
            "record_type": "decode_attempt",
            "piece": perf_id,
            "chunk": calls,
            "artifact_name": f"{perf_id}.c{calls}",
            "decode_scope": "piece_cursor",
            "reset_downbeat_bar": bar,
            "requested_measures": [bar, bar + requested - 1],
            "requested_bars": requested,
            **stats,
            "selection": "grammar_constrained_greedy",
            "token_ids": list(token_ids),
            "tokens": [tokenizer.id_to_token.get(t, "") for t in token_ids],
            "log_probs": log_probs,
            **extra,
        }

    def write(record, log):
        model_decode_log.write(json.dumps(record) + "\n")
        log_row = dict(log or {})
        log_row.update({
            key: record[key] for key in (
                "record_type", "piece", "chunk", "artifact_name", "decode_scope",
                "reset_downbeat_bar", "requested_measures", "requested_bars",
                "boundary_completed_bars", "stop_reason", "accepted_bars",
            ) if key in record
        })
        for key in ("attempt_kind", "target_reset_bar", "discarded_prefix_bars",
                    "fallback_trigger", "rejected_bar_prefixes", "fail"):
            if key in record:
                log_row[key] = record[key]
        decode_log.write(json.dumps(log_row) + "\n")

    common = dict(
        model=model, mel=mel, downbeat_frames=downbeat_frames, tokenizer=tokenizer,
        chunk_frames=chunk_frames, max_len=max_len, device=device,
        coordinate_intervention=coordinate_intervention, manifest_item=manifest_item,
    )
    while cursor < n_bars:
        requested = _bars_available_from_downbeats(
            downbeat_frames, cursor, trust_frames, n_bars - cursor,
        )
        token_ids, log_probs, stats = _decode_from_bar(
            bar=cursor, requested_bars=requested, **common,
        )
        record = decode_record(cursor, requested, token_ids, log_probs, stats)
        calls += 1
        boundary = stats["boundary_completed_bars"]
        accepted = None
        log = None
        if boundary <= 0:
            # The decoder closed no bar: complete its partial first bar with
            # rests, as the scope decoder does.
            fallback = dict(record)
            fallback.update({
                "measures": [cursor, cursor],
                "completion_policy": "tail_rest_fallback",
                "tail_rest_completion_bars": [0],
                "skip_trailing_bar_dedup": True,
            })
            kern, _, log, fills = _postprocess_bar_decode_entry(
                fallback, tokenizer, constraints_cpu,
            )
            if kern and fills and len(_bar_blocks(kern)) == 1:
                accepted = {"start": cursor, "keep_from": cursor, "bars": 1, "kern": kern}
                record.update({
                    "measures": [cursor, cursor],
                    "accepted_bars": 1,
                    "completion_policy": "tail_rest_fallback",
                    "fallback_trigger": "decoder_completed_zero_bars",
                    "tail_rest_completion_bars": [0],
                    "tail_rest_completions": log["tail_rest_completions"],
                })
            else:
                record.update({
                    "measures": [],
                    "accepted_bars": 0,
                    "fallback_attempted": True,
                    "fail": log.get("fail", "partial bar is not eligible for tail-rest completion"),
                })
        else:
            if boundary > requested:
                raise RuntimeError(
                    f"{record['artifact_name']}: completed {boundary} bars "
                    f"with a limit of {requested}"
                )
            bars, kern, log, rejected = _longest_reconstructable_prefix(
                record, tokenizer, constraints_cpu, cursor, boundary, 1,
            )
            if bars:
                accepted = {"start": cursor, "keep_from": cursor, "bars": bars, "kern": kern}
                record.update({"measures": [cursor, cursor + bars - 1], "accepted_bars": bars})
            else:
                record.update({"measures": [], "accepted_bars": 0,
                               "fail": "no complete reconstructable bar prefix"})
            if rejected:
                record["rejected_bar_prefixes"] = rejected
        write(record, log)

        if accepted is None and cursor > 0:
            retry_start = cursor - 1
            # The retry must close the discarded bar and at least one more.
            retry_requested = max(2, _bars_available_from_downbeats(
                downbeat_frames, retry_start, trust_frames, n_bars - retry_start,
            ))
            token_ids, log_probs, stats = _decode_from_bar(
                bar=retry_start, requested_bars=retry_requested, **common,
            )
            record = decode_record(
                retry_start, retry_requested, token_ids, log_probs, stats,
                attempt_kind="previous_downbeat_retry",
                fallback_trigger="first_bar_unreconstructable",
                target_reset_bar=cursor,
                discarded_prefix_bars=1,
            )
            calls += 1
            boundary = stats["boundary_completed_bars"]
            bars, kern, log, rejected = (0, "", None, [])
            if boundary >= 2:
                bars, kern, log, rejected = _longest_reconstructable_prefix(
                    record, tokenizer, constraints_cpu, retry_start,
                    min(boundary, retry_requested), 2,
                )
            if bars:
                accepted = {
                    "start": retry_start, "keep_from": cursor, "bars": bars,
                    "kern": kern,
                }
                record.update({
                    "measures": [retry_start, retry_start + bars - 1],
                    "accepted_bars": bars,
                    "target_accepted_bars": bars - 1,
                })
            else:
                record.update({
                    "measures": [], "accepted_bars": 0, "target_accepted_bars": 0,
                    "fail": "previous-downbeat retry made no progress",
                })
            if rejected:
                record["rejected_bar_prefixes"] = rejected
            write(record, log)

        if accepted is None:
            failed_bars.append(cursor)
            cursor += 1
            continue
        fragments.append(accepted)
        cursor = accepted["start"] + accepted["bars"]
    model_decode_log.flush()
    decode_log.flush()
    return fragments, failed_bars, calls


def _load_mel(path: Path) -> torch.Tensor:
    if str(path).endswith(".npy"):
        mel = torch.from_numpy(np.load(str(path), mmap_mode="r")).float()
    else:
        mel = torch.load(str(path), map_location="cpu", weights_only=True)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    return mel


def _completed_ids(output_dir: Path, key_penalty, meter_penalty, *, boundary_source=None) -> set:
    timeline_path = output_dir / TIMELINE_FILE
    done = set()
    if not timeline_path.is_file():
        return done
    for line in timeline_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (output_dir / f"{row['id']}.krn").is_file():
            if boundary_source is not None and row.get("boundary_source") != boundary_source:
                raise ValueError(
                    f"Cannot resume {row['id']}: saved boundary source "
                    f"{row.get('boundary_source')!r} differs from {boundary_source!r}; "
                    "use a separate output directory"
                )
            schema = row.get("sticky_schema") or {}
            actual = (schema.get("key_penalty"), schema.get("meter_penalty"))
            if actual != (key_penalty, meter_penalty):
                raise ValueError(
                    f"Cannot resume {row['id']}: saved key/meter penalties {actual} "
                    f"differ from requested {(key_penalty, meter_penalty)}; "
                    "use a separate output directory"
                )
            done.add(row["id"])
    return done


def run(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_source = (
        PREDICTED_BOUNDARY_SOURCE if args.bar_boundaries == "predicted"
        else args.bar_boundaries
    )
    completed = _completed_ids(
        output_dir, args.key_switch_penalty, args.meter_switch_penalty,
        boundary_source=boundary_source,
    ) if args.resume else set()

    model, raw_cfg = load_model(args.checkpoint, args.config, args.device)
    if not hasattr(model, "tempo"):
        raise ValueError(
            "self-segmented inference needs the phase branch; the loaded "
            "checkpoint has no tempo module"
        )
    tokenizer = KernTokenizer()
    constraints_cpu = _build_decode_constraints(tokenizer, torch.device("cpu"))
    chunk_cfg = raw_cfg.get("data", {}).get("chunking", {})
    chunk_frames = int(chunk_cfg.get("chunk_frames", 1280))
    trust_frames = int(chunk_cfg.get("overlap_frames", 640))
    max_len = args.max_len or int(raw_cfg.get("model", {}).get("max_seq_len", 2048))

    with open(args.manifest) as handle:
        manifest = json.load(handle)
    coordinate_intervention = None
    if getattr(args, "coordinate_intervention", "none") != "none":
        # Donor works come from the whole manifest, before any id filter.
        coordinate_intervention = CoordinateIntervention(
            args.coordinate_intervention, model, manifest,
            beatthis_dir=getattr(args, "beatthis_dir", None),
        )
    if args.id_regex:
        pattern = re.compile(args.id_regex)
        manifest = [item for item in manifest if pattern.search(item["id"])]
    start_idx = max(0, args.start_idx or 0)
    if start_idx:
        manifest = manifest[start_idx:]
    if args.max_samples:
        manifest = manifest[:args.max_samples]
    manifest_dir = Path(args.manifest_dir)

    timeline_log = open(output_dir / TIMELINE_FILE, "a")
    decode_log = open(output_dir / DECODE_LOG_FILE, "a")
    model_decode_log = open(output_dir / MODEL_DECODE_FILE, "a")

    totals = {
        "performances": 0, "resumed": 0, "written": 0,
        "too_few_bars": 0, "scopes": 0, "failed_scopes": 0,
        "rest_filled_bars": 0, "predicted_bars": 0, "reference_bars": 0,
        "tracker_failures": 0,
        "decode_calls": 0, "failed_bars": 0,
    }
    for item in tqdm(manifest, desc="Inference (self-segmented)"):
        perf_id = item["id"]
        totals["performances"] += 1
        if perf_id in completed:
            totals["resumed"] += 1
            continue
        mel_path = manifest_dir / item["mel_path"]
        if not mel_path.exists():
            logger.warning("Mel missing: %s, skipping", mel_path)
            continue
        started = time.time()
        mel = _load_mel(mel_path)
        n_frames = int(item.get("n_frames") or mel.shape[-1])
        boundary_mode = getattr(args, "bar_boundaries", "predicted")
        if boundary_mode in ("annotated", "beatthis"):
            # Tracker boundaries and decoder coordinates must come from the same events.
            if boundary_mode == "annotated":
                seconds = [
                    float(measure["start_sec"])
                    for measure in (item.get("audio_measures") or [])
                ]
            else:
                if coordinate_intervention is None or coordinate_intervention.mode != "beatthis":
                    raise ValueError("tracker boundaries require beatthis coordinates")
                try:
                    tracker_source = coordinate_intervention.tracker_source(item)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    totals["tracker_failures"] += 1
                    timeline_log.write(json.dumps({
                        "id": perf_id, "n_frames": n_frames,
                        "status": "tracker_input_failed", "downbeat_frames": [],
                        "error": f"{type(error).__name__}: {error}",
                    }) + "\n")
                    timeline_log.flush()
                    logger.error("%s: tracker input failed (%s)", perf_id, error)
                    continue
                seconds = [m["start_sec"] for m in tracker_source["audio_measures"]]
            frames = [int(s * MEL_FPS) for s in seconds]
            # Two downbeats inside one frame would break the strictly increasing
            # origin sequence the scope decoder relies on.
            frames = [f for i, f in enumerate(frames) if i == 0 or f > frames[i - 1]]
            timeline = {
                "windows": 0,
                "downbeat_frames": frames,
                "mean_concentration": None,
            }
        else:
            timeline = predict_downbeat_timeline(
                model, mel, n_frames, chunk_frames, trust_frames,
                args.batch_size, args.device,
            )
        downbeat_frames = timeline["downbeat_frames"]
        n_bars = len(downbeat_frames)
        totals["predicted_bars"] += n_bars

        reference = item.get("audio_measures") or []
        reference_frames = [
            int(round(float(measure["start_sec"]) * MEL_FPS)) for measure in reference
        ]
        diagnostics = None
        if reference_frames:
            totals["reference_bars"] += len(reference_frames)
            diagnostics = {
                "reference_bars": len(reference_frames),
                "count_ratio": n_bars / len(reference_frames),
                **event_f1(downbeat_frames, reference_frames, EVENT_TOLERANCE_FRAMES),
            }

        row = {
            "id": perf_id,
            "boundary_source": boundary_source,
            "n_frames": n_frames,
            "windows": timeline["windows"],
            "predicted_bars": n_bars,
            "downbeat_frames": downbeat_frames,
            "downbeat_sec": [frame / MEL_FPS for frame in downbeat_frames],
            "mean_concentration": timeline["mean_concentration"],
            "reference_diagnostics": diagnostics,
        }

        if args.piece_scope == "cursor":
            if not n_bars:
                totals["too_few_bars"] += 1
                row.update({
                    "status": "too_few_bars",
                    "decode_scope": "piece_cursor",
                    "elapsed_sec": time.time() - started,
                })
                timeline_log.write(json.dumps(row) + "\n")
                timeline_log.flush()
                logger.warning("%s: no predicted bars, no whole-piece output", perf_id)
                continue
            fragments, failed_bars, calls = decode_piece_with_cursor(
                perf_id=perf_id,
                model=model,
                mel=mel,
                downbeat_frames=downbeat_frames,
                tokenizer=tokenizer,
                constraints_cpu=constraints_cpu,
                chunk_frames=chunk_frames,
                trust_frames=trust_frames,
                max_len=max_len,
                device=args.device,
                coordinate_intervention=coordinate_intervention,
                manifest_item=item,
                model_decode_log=model_decode_log,
                decode_log=decode_log,
            )
            totals["decode_calls"] += calls
            totals["failed_bars"] += len(failed_bars)
            try:
                piece_kern, rest_filled = assemble_whole_piece(fragments, n_bars)
            except ValueError as error:
                totals["assembly_errors"] = totals.get("assembly_errors", 0) + 1
                row.update({
                    "status": "assembly_error",
                    "error": str(error)[:200],
                    "decode_scope": "piece_cursor",
                    "decode_calls": calls,
                    "failed_bars": failed_bars,
                    "rest_filled_bars": [],
                    "elapsed_sec": time.time() - started,
                })
                timeline_log.write(json.dumps(row) + "\n")
                timeline_log.flush()
                logger.warning("%s: whole-piece assembly failed: %s", perf_id, error)
                continue
            (output_dir / f"{perf_id}.krn").write_text(piece_kern)
            totals["written"] += 1
            totals["rest_filled_bars"] += len(rest_filled)
            row.update({
                "status": "ready",
                "decode_scope": "piece_cursor",
                "decode_calls": calls,
                "accepted_chunks": len(fragments),
                "failed_bars": failed_bars,
                "rest_filled_bars": rest_filled,
                "elapsed_sec": time.time() - started,
            })
            timeline_log.write(json.dumps(row) + "\n")
            timeline_log.flush()
            logger.info(
                "%s: %d bars (ref %s) → %d chunks, %d decode calls, %d failed bars, %.0fs",
                perf_id, n_bars,
                diagnostics["reference_bars"] if diagnostics else "n/a",
                len(fragments), calls, len(failed_bars), time.time() - started,
            )
            continue

        scopes = scope_plan(perf_id, n_bars)
        if not scopes:
            totals["too_few_bars"] += 1
            row.update({
                "status": "too_few_bars",
                "scopes": 0,
                "failed_scopes": [],
                "rest_filled_bars": [],
                "elapsed_sec": time.time() - started,
            })
            timeline_log.write(json.dumps(row) + "\n")
            timeline_log.flush()
            logger.warning(
                "%s: only %d predicted bars, no whole-piece output", perf_id, n_bars
            )
            continue

        schema_timeline = None
        meter_switch_penalty = getattr(args, "meter_switch_penalty", None)
        key_switch_penalty = getattr(args, "key_switch_penalty", None)
        if meter_switch_penalty is not None or key_switch_penalty is not None:
            schema_timeline, sticky_row = decide_sticky_timeline(
                model, mel, downbeat_frames, scopes, n_bars, tokenizer,
                chunk_frames, args.device, meter_switch_penalty, key_switch_penalty,
                coordinate_intervention=coordinate_intervention, manifest_item=item,
            )
            row["sticky_schema"] = sticky_row
        scope_outputs = []
        failed_scopes = []
        results = iter_kern_piece_from_downbeats(
            perf_id=perf_id,
            model=model,
            mel=mel,
            downbeat_frames=downbeat_frames,
            tokenizer=tokenizer,
            constraints_cpu=constraints_cpu,
            chunk_frames=chunk_frames,
            trust_frames=trust_frames,
            max_bars_per_chunk=SCOPE_BARS,
            max_len=max_len,
            num_beams=1,
            device=args.device,
            output_scopes=scopes,
            schema_timeline=schema_timeline,
            coordinate_intervention=coordinate_intervention,
            manifest_item=item,
            inference_batch_size=args.inference_batch_size,
            encoder_batch_size=args.encoder_batch_size,
            stop_on_reconstruction_failure=False,
        )
        for scope, result in zip(scopes, results, strict=True):
            totals["scopes"] += 1
            kern_text = None if result["failed"] else result["pred_kern"]
            for attempt in result["attempts"]:
                model_decode_log.write(json.dumps(attempt["model_decode_entry"]) + "\n")
                decode_log.write(json.dumps(attempt["log_entry"]) + "\n")
            if result.get("scope_entry") is not None:
                model_decode_log.write(json.dumps(result["scope_entry"]) + "\n")
            if result.get("output_log_entry") is not None:
                decode_log.write(json.dumps(result["output_log_entry"]) + "\n")
            model_decode_log.flush()
            decode_log.flush()
            if kern_text is None:
                failed_scopes.append(scope["artifact_name"])
                totals["failed_scopes"] += 1
            scope_outputs.append({
                "start": scope["start"],
                "keep_from": scope["keep_from"],
                "kern": kern_text,
            })

        try:
            piece_kern, rest_filled = assemble_whole_piece(scope_outputs, n_bars)
        except ValueError as error:
            # One malformed piece must not end an unattended run; the timeline
            # row keeps the reason and the piece counts as missing downstream.
            totals["assembly_errors"] = totals.get("assembly_errors", 0) + 1
            row.update({
                "status": "assembly_error",
                "error": str(error)[:200],
                "scopes": len(scopes),
                "failed_scopes": failed_scopes,
                "rest_filled_bars": [],
                "elapsed_sec": time.time() - started,
            })
            timeline_log.write(json.dumps(row) + "\n")
            timeline_log.flush()
            logger.warning("%s: whole-piece assembly failed: %s", perf_id, error)
            continue
        (output_dir / f"{perf_id}.krn").write_text(piece_kern)
        totals["written"] += 1
        totals["rest_filled_bars"] += len(rest_filled)
        row.update({
            "status": "ready",
            "scopes": len(scopes),
            "failed_scopes": failed_scopes,
            "rest_filled_bars": rest_filled,
            "elapsed_sec": time.time() - started,
        })
        timeline_log.write(json.dumps(row) + "\n")
        timeline_log.flush()
        logger.info(
            "%s: %d predicted bars (ref %s) → %d scopes, %d failed, %d rest bars, %.0fs",
            perf_id, n_bars,
            diagnostics["reference_bars"] if diagnostics else "n/a",
            len(scopes), len(failed_scopes), len(rest_filled),
            time.time() - started,
        )

    timeline_log.close()
    decode_log.close()
    model_decode_log.close()
    census_rows = acr_rows_from_timeline(output_dir / TIMELINE_FILE, manifest)
    (output_dir / ACR_ROWS_FILE).write_text(
        json.dumps(census_rows, indent=1) + "\n"
    )
    summary = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "manifest": args.manifest,
        "chunk_frames": chunk_frames,
        "trust_frames": trust_frames,
        "boundary_source": (
            "annotated bar starts of the recording"
            if getattr(args, "bar_boundaries", "predicted") == "annotated" else
            "Beat This! predicted downbeats"
            if getattr(args, "bar_boundaries", "predicted") == "beatthis" else
            "forward phase-zero crossings of the averaged first moment, minimum spacing 400 ms"
        ),
        "bar_boundaries": getattr(args, "bar_boundaries", "predicted"),
        "piece_scope": args.piece_scope,
        "key_switch_penalty": args.key_switch_penalty,
        "meter_switch_penalty": args.meter_switch_penalty,
        "coordinate_intervention": (
            {"mode": coordinate_intervention.mode, **coordinate_intervention.stats,
             "tracker_sources": coordinate_intervention.source_metadata}
            if coordinate_intervention is not None else None
        ),
        "totals": totals,
    }
    (output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Done — %s", json.dumps(totals))


def _optional_penalty(value: str):
    return None if value.lower() == "none" else float(value)


def _penalty_name(value) -> str:
    """How a penalty is spelled in output names: 8.3, 4, inf or none."""
    return "none" if value is None else f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-segmented whole-piece inference for the joint model"
    )
    parser.add_argument("--checkpoint", default="checkpoints/full_seed42.pt")
    parser.add_argument("--config", default="configs/piano_2gpu.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved inference settings without loading the model")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Phase windows encoded per forward pass")
    parser.add_argument("--inference-batch-size", type=int, default=4,
                        help="Independent five-bar scopes decoded together")
    parser.add_argument("--encoder-batch-size", type=int, default=1,
                        help="Audio chunks encoded per note-generation forward pass")
    parser.add_argument("--resume", action="store_true",
                        help="Skip performances with a written whole-piece kern")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--id-regex", default=None,
                        help="Only manifest ids matching this regular expression")
    parser.add_argument(
        "--coordinate-intervention", default="none", choices=COORDINATE_INTERVENTIONS,
        help="Replace the audio-side metrical-position posterior the scope decoder "
             "reads at its cross-attention (oracle, other-work, zero, beatthis); the bar "
             "boundary source itself is untouched",
    )
    parser.add_argument(
        "--bar-boundaries", default="predicted",
        choices=("predicted", "annotated", "beatthis"),
        help="Scope origins: forward phase-zero crossings spaced at least 400 ms (predicted), the "
             "recording's annotated bar starts (annotated), or an external "
             "tracker's downbeats read from --beatthis-dir (beatthis)",
    )
    parser.add_argument(
        "--piece-scope", default="five_bar", choices=("five_bar", "cursor"),
        help="five_bar: independent five-bar scopes plus a tail scope; cursor: one "
             "cursor walks the whole piece chunk by chunk, each chunk asking for "
             "every bar inside its trust region",
    )
    parser.add_argument(
        "--beatthis-dir", default=None,
        help="Directory of <id>.json tracker outputs with beat/downbeat times in seconds",
    )
    parser.add_argument(
        "--meter-switch-penalty", type=_optional_penalty, default=8.3,
        help="Decide the meter alone, scope by scope, along the best path through the "
             "scopes' meter log-probabilities where each change of meter costs this many "
             "nats (inf = one meter per piece); the default 8.3 and the key default come from "
             "the training corpus's change rates (src.analysis.schema_switch_penalty); "
             "none leaves meter to free generation",
    )
    parser.add_argument(
        "--key-switch-penalty", type=_optional_penalty, default=6.3,
        help="Same decision for the key signature alone (inf = one key per piece); "
             "default 6.3; none leaves key to free generation; combines with --meter-switch-penalty",
    )
    args = parser.parse_args()
    if args.inference_batch_size < 1 or args.encoder_batch_size < 1:
        parser.error("inference and encoder batch sizes must be positive")
    sticky = args.meter_switch_penalty is not None or args.key_switch_penalty is not None
    if sticky and args.piece_scope == "cursor":
        parser.error("--piece-scope cursor decodes without a schema timeline")
    if args.bar_boundaries == "beatthis" and args.coordinate_intervention != "beatthis":
        parser.error("--bar-boundaries beatthis requires --coordinate-intervention beatthis")
    if args.coordinate_intervention == "beatthis" and not args.beatthis_dir:
        parser.error("--coordinate-intervention beatthis needs --beatthis-dir")
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    if args.manifest_dir is None:
        args.manifest_dir = config["paths"]["manifest_dir"]
    if args.manifest is None:
        args.manifest = f"{args.manifest_dir}/test_manifest.json"
    if args.output_dir is None:
        args.output_dir = (
            f"{args.manifest_dir}/test_kern_pred_self_segmented_forward400"
            f"_meter{_penalty_name(args.meter_switch_penalty)}"
            f"_key{_penalty_name(args.key_switch_penalty)}"
        )

    if args.dry_run:
        chunking = config.get("data", {}).get("chunking", {})
        print(json.dumps({**vars(args), "chunk_frames": chunking.get("chunk_frames", 1280),
                          "trust_frames": chunking.get("overlap_frames", 640)}, indent=2))
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
