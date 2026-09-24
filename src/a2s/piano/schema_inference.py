"""Scope observations and separate meter/key Viterbi timelines."""

from typing import List, Optional, Tuple

import numpy as np
import torch

from src.a2s.piano.foundation import HFT_ARCH, HFT_PAD_VALUE
from src.a2s.piano.tokenizer import (
    KEY_TOKENS, METER_DEN_TOKENS, METER_NUM_TOKENS, KernTokenizer,
)

SCOPE_BARS = 5


def merge_memory_moments(
    predicted: torch.Tensor, override: torch.Tensor, dtype: torch.dtype,
) -> torch.Tensor:
    """Replacement Fourier moments where they are finite; NaN frames keep the
    predicted moments."""
    override = override.to(predicted.device, dtype)
    return torch.where(torch.isnan(override), predicted.to(dtype), override)


def _slice_with_margin(
    mel: torch.Tensor, start_frame: int, chunk_frames: int,
) -> torch.Tensor:
    """Slice mel with hFT margin on both sides, padding with silence at boundaries."""
    n_margin = HFT_ARCH.n_margin
    total_frames = mel.shape[-1]
    sl_start = start_frame - n_margin
    sl_end = start_frame + chunk_frames + n_margin
    pad_left = max(0, -sl_start)
    pad_right = max(0, sl_end - total_frames)
    sl_start = max(0, sl_start)
    sl_end = min(total_frames, sl_end)
    chunk = mel[..., sl_start:sl_end]
    if pad_left + pad_right > 0:
        chunk = torch.nn.functional.pad(
            chunk, (pad_left, pad_right), value=HFT_PAD_VALUE,
        )
    return chunk


@torch.no_grad()
def scope_schema_scorecard(
    model,
    mel_input: torch.Tensor,
    tokenizer: KernTokenizer,
    device: str,
    memory_moments_override: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[int]]:
    """Decoder log-probabilities of the schema prefix for one scope.

    ``memory_moments_override`` replaces the audio-side Fourier moments the
    way the scope decoder's coordinate intervention does, so the scorecard
    reads the same coordinates as the decode it pins.

    Returns meter log-probs [24] over (num, den) pairs (num-major), key
    log-probs [24, 15] conditioned on each pair, the pair token ids and the
    key token ids."""
    vocab = tokenizer.vocab
    sos, bar = vocab["<sos>"], vocab["<bar>"]
    num_ids = [vocab[t] for t in METER_NUM_TOKENS]
    den_ids = [vocab[t] for t in METER_DEN_TOKENS]
    key_ids = [vocab[t] for t in KEY_TOKENS]
    fourierpe = getattr(model, "coordinate_delivery", "memory") == "cross_attention"
    mel_input = mel_input.to(device)
    if fourierpe:
        memory, moments = model.encode_with_fourierpe(mel_input)
    else:
        memory, moments = model.encode(mel_input), None
    if memory_moments_override is not None:
        if moments is None:
            raise ValueError("memory moments can only be overridden under cross-attention delivery")
        moments = merge_memory_moments(moments, memory_moments_override, memory.dtype)

    def last_logits(prefixes: List[List[int]]) -> torch.Tensor:
        ids = torch.tensor(prefixes, dtype=torch.long, device=memory.device)
        mem = memory.expand(len(prefixes), -1, -1)
        extra = {}
        if fourierpe:
            # No meter is known inside the schema prefix: the score-side phase is 0.
            phases = torch.zeros(ids.shape, device=memory.device)
            extra = {
                "memory_moments": moments.expand(len(prefixes), -1, -1),
                "fourierpe_feats": model.score_fourierpe_features(phases),
            }
        hidden = model.decode_hidden(mem, ids, **extra)
        return model.output_proj(hidden[:, -1]).float()

    num_logits = last_logits([[sos, bar]])[0]
    lp_num = torch.log_softmax(num_logits[num_ids], dim=-1)
    lp_den = torch.log_softmax(
        last_logits([[sos, bar, n] for n in num_ids])[:, den_ids], dim=-1,
    )
    meter_logp = (lp_num[:, None] + lp_den).reshape(-1)
    pairs = [(n, d) for n in num_ids for d in den_ids]
    lp_key = torch.log_softmax(
        last_logits([[sos, bar, n, d] for n, d in pairs])[:, key_ids], dim=-1,
    )
    return meter_logp.cpu().numpy(), lp_key.cpu().numpy(), pairs, key_ids


def sticky_path(obs: np.ndarray, switch_penalty: float) -> np.ndarray:
    """One state index for every scope: the best path through the per-scope
    log-likelihoods [S, K] when every change of state costs ``switch_penalty``."""
    n_scopes, n_states = obs.shape
    score = obs[0].astype(np.float64)
    back = np.zeros((n_scopes, n_states), dtype=np.int64)
    states = np.arange(n_states)
    for t in range(1, n_scopes):
        best = int(score.argmax())
        jump = score[best] - switch_penalty
        back[t] = np.where(score >= jump, states, best)
        score = np.maximum(score, jump) + obs[t]
    path = [int(score.argmax())]
    for t in range(n_scopes - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return np.array(path[::-1])


def decide_sticky_timeline(
    model,
    mel: torch.Tensor,
    downbeat_frames: List[int],
    scopes: List[dict],
    n_bars: int,
    tokenizer: KernTokenizer,
    chunk_frames: int,
    device: str,
    meter_penalty: Optional[float],
    key_penalty: Optional[float],
    coordinate_intervention=None,
    manifest_item: Optional[dict] = None,
) -> Tuple[List[tuple], dict]:
    """Per-bar (num, den, key) triples for the piece, plus a log row.

    Meter and key are decided separately, each along the sticky path over the
    scopes' scorecards, so each holds until the evidence against it outweighs
    its penalty.  A ``None`` penalty leaves that slot to the decoder, scope by
    scope.  The key evidence is marginal over the meter, so it does not
    depend on how the meter was decided.  Under a coordinate intervention each
    scorecard reads the coordinates that scope's decode reads; those lookups
    are reported in the log row, not in the intervention's decode counts."""
    cards = []
    decode_stats = dict(coordinate_intervention.stats) if coordinate_intervention is not None else None
    for scope in scopes:
        mel_chunk = _slice_with_margin(mel, downbeat_frames[scope["start"]], chunk_frames)
        mel_input = mel_chunk if mel_chunk.dim() == 3 else mel_chunk.unsqueeze(0)
        override = (
            coordinate_intervention.moments(
                manifest_item, downbeat_frames[scope["start"]], chunk_frames, device,
            ) if coordinate_intervention is not None else None
        )
        cards.append(scope_schema_scorecard(
            model, mel_input, tokenizer, device, memory_moments_override=override,
        ))
    scorecard_coordinates = None
    if coordinate_intervention is not None:
        scorecard_coordinates = {
            "mode": coordinate_intervention.mode,
            **{k: coordinate_intervention.stats[k] - decode_stats[k] for k in decode_stats},
        }
        coordinate_intervention.stats.update(decode_stats)
    pairs, key_ids = cards[0][2], cards[0][3]
    meter_obs = np.stack([card[0] for card in cards])
    joint = np.stack([card[0][:, None] + card[1] for card in cards])
    peak = joint.max(axis=1, keepdims=True)
    key_obs = (peak + np.log(np.exp(joint - peak).sum(axis=1, keepdims=True)))[:, 0, :]
    meter_idx = sticky_path(meter_obs, meter_penalty) if meter_penalty is not None else None
    key_idx = sticky_path(key_obs, key_penalty) if key_penalty is not None else None
    name = tokenizer.id_to_token
    timeline: List[Optional[tuple]] = [None] * n_bars
    decisions = []
    for index, scope in enumerate(scopes):
        num = den = key = None
        decision = {"start": scope["start"]}
        if meter_idx is not None:
            num, den = pairs[meter_idx[index]]
            own = int(meter_obs[index].argmax())
            decision["meter"] = [name[num], name[den]]
            decision["independent_meter"] = [name[pairs[own][0]], name[pairs[own][1]]]
        if key_idx is not None:
            key = key_ids[key_idx[index]]
            decision["key"] = name[key]
            decision["independent_key"] = name[key_ids[int(key_obs[index].argmax())]]
        for b in range(scope["keep_from"], min(scope["start"] + SCOPE_BARS, n_bars)):
            timeline[b] = (num, den, key)
        decisions.append(decision)
    last = None
    for b in range(n_bars):
        if timeline[b] is None:
            timeline[b] = last
        last = timeline[b]
    row = {"meter_penalty": meter_penalty, "key_penalty": key_penalty, "scopes": decisions}
    if scorecard_coordinates is not None:
        row["scorecard_coordinates"] = scorecard_coordinates
    return timeline, row


def scope_plan(perf_id: str, n_bars: int) -> List[dict]:
    """Non-overlapping five-bar scopes plus one tail scope for the remainder."""
    if n_bars < SCOPE_BARS:
        return []
    scopes = []
    for chunk, start in enumerate(range(0, n_bars - SCOPE_BARS + 1, SCOPE_BARS)):
        scopes.append({
            "start": start,
            "keep_from": start,
            "chunk": chunk,
            "artifact_name": f"{perf_id}.{chunk}",
        })
    last_end = scopes[-1]["start"] + SCOPE_BARS
    if last_end < n_bars:
        scopes.append({
            "start": n_bars - SCOPE_BARS,
            "keep_from": last_end,
            "chunk": len(scopes),
            "artifact_name": f"{perf_id}.tail",
        })
    return scopes
