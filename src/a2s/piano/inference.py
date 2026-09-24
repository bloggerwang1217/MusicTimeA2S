"""
Inference for PianoModel (hFT foundation + Transformer decoder).

Bar mode — downbeat-guided direct five-bar inference:
  1. Feed 1280 frames (chunk_frames) to encoder.
  2. Decode tokens, keep bars whose next downbeat < chunk_start + 640
     (trust_frames = overlap_frames from training).
  3. Shift to the next unfinished bar's downbeat until the needed bar table
     is complete.
  4. Reconstruct each requested five-bar scope exactly once.

Usage:
    poetry run python -m src.a2s.piano.inference \
        --checkpoint checkpoints/<arm>/best.pt \
        --config configs/piano_2gpu.yaml \
        --metadata data/experiments/syn/augmentation_metadata.json

ASAP manifests carry ``audio_measures`` directly, so they do not require a
separate metadata file.
"""

import argparse
import hashlib
import json
import math
import logging
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple

import converter21
import torch
import yaml
from tqdm import tqdm

from src.a2s.piano.model import PianoModel, build_piano_model
from src.a2s.piano.foundation import HFT_MEL
from src.a2s.piano.tokenizer import KernTokenizer
from src.a2s.piano.schema_inference import _slice_with_margin, merge_memory_moments
from src.datasets.syn.metadata import index_augmentation_metadata_by_render
from src.score.reconstruct_kern import reconstruct_kern_from_bracket_tokens

converter21.register()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MEL_FPS = HFT_MEL.sample_rate / HFT_MEL.hop_length  # 62.5

COORDINATE_INTERVENTIONS = ("none", "oracle", "other-work", "zero", "beatthis")


class CoordinateIntervention:
    """Swap the audio-side metrical-position posterior that the decoder's
    cross-attention keys and values read, leaving the encoder memory and the
    score-side (grammar) position untouched.

    oracle      the annotated curve of the recording itself, blurred like the
                training target, so the substituted feature is the learned
                embedding of that phase class and nothing else;
    other-work  the annotated curve of a recording from a different work
                (works sorted by id, each takes the next one's first recording,
                same chunk start frame, wrapped into the donor's length);
    zero        no coordinate contribution at the keys and values.
    beatthis    tracker-derived beat counts, normalized within each predicted bar.
    Chunks whose annotation cannot yield a curve keep the predicted posterior
    and are counted in ``stats``. Invalid tracker inputs never use that fallback.
    Under oracle, frames the annotation does not ground (off the beat grid, or
    in a bar whose clock is not score-grounded) keep the predicted posterior,
    as the gold coordinate is only swapped in on those frames in training;
    they come back as NaN rows and are counted in ``predicted_frames``."""

    def __init__(self, mode: str, model, manifest: List[dict], beatthis_dir=None):
        if mode not in COORDINATE_INTERVENTIONS:
            raise ValueError(f"unknown coordinate intervention {mode!r}")
        if mode != "none" and getattr(model, "coordinate_delivery", "memory") != "cross_attention":
            raise ValueError("coordinate interventions need cross-attention delivery")
        self.mode = mode
        self.model = model
        self.stats = {"attempts": 0, "substituted": 0, "fallback_predicted": 0,
                      "predicted_frames": 0}
        self.beatthis_dir = Path(beatthis_dir) if beatthis_dir is not None else None
        if mode == "beatthis" and self.beatthis_dir is None:
            raise ValueError("beatthis coordinates require a tracker output directory")
        self.tracker_sources: Dict[str, dict] = {}
        self.source_metadata: Dict[str, dict] = {}
        self.donor: Dict[str, dict] = {}
        if mode == "other-work":
            by_work: Dict[str, List[dict]] = {}
            for item in manifest:
                by_work.setdefault(str(item.get("piece_id", item["id"])), []).append(item)
            works = sorted(by_work)
            if len(works) < 2:
                raise ValueError("other-work needs at least two works in the manifest")
            for index, work in enumerate(works):
                donor_item = min(by_work[works[(index + 1) % len(works)]], key=lambda i: i["id"])
                for item in by_work[work]:
                    self.donor[item["id"]] = donor_item

    def tracker_source(self, item: dict) -> dict:
        """Convert saved tracker events to the same beat-count inputs as annotations."""
        perf_id = item["id"]
        if perf_id in self.tracker_sources:
            return self.tracker_sources[perf_id]
        path = self.beatthis_dir / f"{perf_id}.json"
        payload = path.read_bytes()
        tracker = json.loads(payload)
        if tracker.get("status") != "ready" or tracker.get("artifact_stem") != perf_id:
            raise ValueError(f"{path}: tracker output is not ready for {perf_id}")
        beats = [float(sec) for sec in tracker["beats"]]
        downbeats = [float(sec) for sec in tracker["downbeats"]]
        recording_end = float(item["n_frames"]) / MEL_FPS
        for name, events in (("beats", beats), ("downbeats", downbeats)):
            if (len(events) < 2 or not all(math.isfinite(t) for t in events)
                    or events[0] < 0 or events[-1] >= recording_end
                    or any(b <= a for a, b in zip(events, events[1:]))):
                raise ValueError(f"{path}: invalid {name} on the recording time axis")
        if not set(downbeats).issubset(beats):
            raise ValueError(f"{path}: downbeats must be a subset of beats")
        downbeat_set = set(downbeats)
        source = {
            "audio_beats": [
                {"sec": sec, "cycle": float(i), "is_downbeat": sec in downbeat_set}
                for i, sec in enumerate(beats)
            ],
            "audio_measures": [
                {"start_sec": start, "end_sec": end}
                for start, end in zip(downbeats, downbeats[1:] + [recording_end])
            ],
        }
        self.tracker_sources[perf_id] = source
        self.source_metadata[perf_id] = {
            "path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "checkpoint": tracker.get("checkpoint"),
            "checkpoint_mapping": tracker.get("checkpoint_mapping"),
        }
        return source

    @staticmethod
    def _curve(item: dict, start_frame: int, capacity: int) -> torch.Tensor:
        from src.a2s.piano.tempo_data import downbeat_phase_curve
        beats = item.get("audio_beats") or []
        grid = item.get("audio_grid") or beats
        return downbeat_phase_curve(grid, item["audio_measures"], beats, start_frame, capacity)

    @staticmethod
    def _grounded_frames(item: dict, start_frame: int, capacity: int):
        """Frames on the annotated beat grid inside a score-grounded bar."""
        import numpy as np
        from src.a2s.piano.tempo_data import _trusted_measure_clock, gt_curves
        beats = item.get("audio_beats") or []
        grid = item.get("audio_grid") or beats
        measures = item["audio_measures"]
        _, _, on_grid = gt_curves(grid, start_frame, capacity)
        trusted = np.asarray(_trusted_measure_clock(measures, beats), dtype=bool)
        bounds = np.array([m["start_sec"] for m in measures] + [measures[-1]["end_sec"]])
        times = (start_frame + np.arange(capacity)) / MEL_FPS
        segment = np.clip(np.searchsorted(bounds, times, side="right") - 1, 0, len(measures) - 1)
        return on_grid.numpy() & trusted[segment]

    def moments(self, item: dict, start_frame: int, capacity: int, device) -> Optional[torch.Tensor]:
        """[1, capacity, 24] replacement moments, or None to keep the prediction."""
        if self.mode == "none":
            return None
        self.stats["attempts"] += 1
        basis = self.model._fourierpe_basis
        if self.mode == "zero":
            self.stats["substituted"] += 1
            return torch.zeros((1, capacity, basis.shape[-1]), device=device)
        if self.mode == "beatthis":
            source = self.tracker_source(item)
        else:
            source = item if self.mode == "oracle" else self.donor[item["id"]]
        frame = start_frame
        if self.mode == "other-work":
            donor_frames = int(source["audio_measures"][-1]["end_sec"] * MEL_FPS)
            frame = start_frame % max(1, donor_frames - capacity)
        try:
            phi = self._curve(source, frame, capacity)
        except (ValueError, KeyError, IndexError) as exc:
            if self.mode == "beatthis":
                raise ValueError(f"{item['id']} @{start_frame}: invalid tracker phase") from exc
            self.stats["fallback_predicted"] += 1
            logger.warning("%s @%d: coordinate intervention fell back to the prediction (%s)", item["id"], start_frame, exc)
            return None
        from src.a2s.piano.tempo_losses import gold_phase_log_probs
        log_probs = gold_phase_log_probs(phi[None].to(device), basis.shape[0])
        moments = torch.softmax(log_probs, dim=-1) @ basis.to(log_probs.dtype)
        if self.mode == "oracle":
            grounded = self._grounded_frames(source, frame, capacity)
            if not grounded.any():
                self.stats["fallback_predicted"] += 1
                logger.warning("%s @%d: no annotated frame in the chunk; coordinate intervention fell back to the prediction", item["id"], start_frame)
                return None
            moments[0, torch.from_numpy(~grounded).to(moments.device)] = float("nan")
            self.stats["predicted_frames"] += int((~grounded).sum())
        self.stats["substituted"] += 1
        return moments


# =============================================================================
# Model loading
# =============================================================================


def load_model(
    checkpoint_path: str,
    config_path: str = "configs/piano_2gpu.yaml",
    device: str = "cuda",
) -> Tuple[PianoModel, dict]:
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    m_cfg = raw_cfg["model"]

    tokenizer = KernTokenizer()

    model = build_piano_model(
        m_cfg,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.vocab["<pad>"],
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    return model, raw_cfg


# =============================================================================
# Decode constraints (grammar enforcement)
# =============================================================================
#
# Next-token grammar: schema triple, bracket/tie placement, the voice-family
# state channel, bar-time accounting and chord member ordering. The model
# decides substance (which pitch, how long), the grammar guarantees form.
# Time rules mirror reconstruct_kern's clock arithmetic exactly, including
# balanced chunk topology and compositional voice addresses. Unknown duration
# accounting still fails open; glyph spelling legality stays unmasked.
#
# A dead end (no legal successor at all) means an open bracket can no longer
# close with any glyph: the model kept it open past the metric-tree seam
# where the tokenizer's spelling would have cut it. The decoder then rewinds
# to that seam and writes the cut itself (close, first glyph, <tie>); the
# model's own onset and release stand, only the spelling is supplied. A dead
# end with no derivable seam falls open exactly as before.

_HANDS = ('l', 'r')

# Per-chunk decode observability: how often the grammar mask overrode the
# model's raw argmax. Reset/read by run_bar_mode around each chunk.
DECODE_STATS = {
    'masked_argmax': 0,
    'steps': 0,
    'max_voice_width': 1,
    'boundary_completed_bars': 0,
    'stop_reason': 'max_len',
    'seam_repairs': 0,
}

_BRACKETS = (('l', '<pl>'), ('r', '<pr>'))


def _build_decode_constraints(tokenizer: KernTokenizer, device: torch.device) -> dict:
    from src.a2s.piano.tokenizer import (
        PITCH_TOKENS, DURATION_TOKENS, METER_NUM_TOKENS, METER_DEN_TOKENS, KEY_TOKENS,
        RECIP_TO_GRID, kern_pitch_to_midi,
    )

    vocab = tokenizer.vocab

    def idx(names) -> torch.Tensor:
        return torch.tensor([vocab[t] for t in names if t in vocab],
                            dtype=torch.long, device=device)

    def ids(names) -> set:
        return {vocab[t] for t in names if t in vocab}

    opens: dict = {}      # open id -> hand
    close_of: dict = {}   # open id -> close id
    for hand, name in _BRACKETS:
        close = '</' + name[1:]
        if name in vocab and close in vocab:
            opens[vocab[name]] = hand
            close_of[vocab[name]] = vocab[close]

    dur_names = [t for t in DURATION_TOKENS if t in vocab]
    return {
        'vocab_size': len(vocab),
        'device': device,
        'pitch_idx': idx(PITCH_TOKENS),
        'dur_idx': idx(dur_names),
        'dur_grids': [RECIP_TO_GRID[t] for t in dur_names],
        'pitch_midi_vec': torch.tensor(
            [kern_pitch_to_midi(t) for t in PITCH_TOKENS if t in vocab],
            dtype=torch.long, device=device),
        'dur_grid': {vocab[t]: RECIP_TO_GRID[t] for t in dur_names},
        'num_val': {vocab[t]: int(t[5:-1]) for t in METER_NUM_TOKENS if t in vocab},
        'den_val': {vocab[t]: int(t[5:-1]) for t in METER_DEN_TOKENS if t in vocab},
        'pitch_midi': {vocab[t]: kern_pitch_to_midi(t)
                       for t in PITCH_TOKENS if t in vocab},
        'num_idx': idx(METER_NUM_TOKENS),
        'den_idx': idx(METER_DEN_TOKENS),
        'key_idx': idx(KEY_TOKENS),
        'pitch_ids': ids(PITCH_TOKENS),
        'dur_ids': ids(DURATION_TOKENS),
        'num_ids': ids(METER_NUM_TOKENS),
        'den_ids': ids(METER_DEN_TOKENS),
        'key_ids': ids(KEY_TOKENS),
        'opens': opens,
        'close_of': close_of,
        'open_of_close': {c: o for o, c in close_of.items()},
        'rest_id': vocab.get('r'),
        'bar_id': vocab['<bar>'],
        'grid_id': vocab['<grid>'],       # capsule slot marker only
        'tup_id': vocab['<tup>'],
        'tup_close_id': vocab['</tup>'],
        'tie_id': vocab['<tie>'],
        'untie_id': vocab['</tie>'],
        'voice_id': vocab.get('<v>'),
        'eos_id': vocab['<eos>'],
    }


def _initial_beam_state() -> dict:
    """Fresh decode scope; each bar discovers its depth from addresses."""
    return {
        'prev': 'start',
        'open': [],
        'address_depth': 0,
        'bar_voice_count': {'l': 1, 'r': 1},
        'last_pitch_rest': False,
        'last_close_sounding': False,
        'num': None,                # meter numerator (int)
        'den': None,                # meter denominator (int)
        'pending_num': None,
        'stream_time': Fraction(0), # latest emitted attack/release instant
        'clock': {},                # (hand, outer-to-inner depth) -> grid ticks
        'tuncert': {'l': False, 'r': False},  # True = clock unknown, fail open
        'last_close': None,         # close metadata awaiting its duration
        'last_member_midi': None,   # chord members ascend strictly low->high
        'pending_ties': {},         # (hand, pitch_id) -> departure age
                                    # (0 = this bar, 1 = previous bar)
        'tie_context': None,        # (hand, pitch_ids) after a close+dur
        'last_pitch_id': None,
        'bars_seen': 0,
    }


def _beat_count(num) -> int:
    return num // 3 if num in (6, 9, 12) else num


def bar_phase(state: dict) -> float:
    """Bar phase (radians) of the grammar's current stream instant; 0 before a meter."""
    if state['num'] is None or state['den'] is None:
        return 0.0
    bar_len = Fraction(int(state['num']) * 128, int(state['den']))
    if bar_len <= 0:
        return 0.0
    return float(2.0 * math.pi * float(state['stream_time'] % bar_len) / float(bar_len))


def _beat_step(state):
    """Beat-unit length in grid ticks, None while the meter is unknown."""
    if state['num'] is None or state['den'] is None:
        return None
    return (Fraction(state['num'] * 128, state['den'])
            / _beat_count(state['num']))


def _pending_landing(state, hand, pid, onset) -> bool:
    """Whether pid has a departure record inside the pairing window: the
    same hand+pitch departed in this bar or the previous one (a portal
    landing may sit anywhere later in the departure's bar or in
    the following bar).  This validates that a landing has a possible
    source; exact multiplicity and assignment belong to reconstruct's
    pairing engine.  Chord departures are partial (>=1 member lands, the
    rest release), so pendings validate landings — they never block
    attacks or barlines."""
    return (hand, pid) in state['pending_ties']


def _chan_clock(state: dict, hand: str, voice):
    return state['clock'].get((hand, voice), Fraction(0))


def _open_time_ok(state: dict, hand: str, voice: int) -> bool:
    """An attack sits at the current derived stream instant."""
    if state['tuncert'][hand]:
        return True
    onset = _chan_clock(state, hand, voice)
    return onset == state['stream_time']


def _close_duration_ok(
    state: dict,
    hand: str,
    voice: int,
    open_time,
    dur_grid,
    is_tup: bool = False,
) -> bool:
    """A release preserves derived time order and the meter boundary.

    Family lock: from a triplet-grid onset only a triplet-family value can
    keep the account reachable — one content, one form."""
    if open_time is None or state['tuncert'][hand]:
        return True
    if (Fraction(open_time).denominator == 3
            and Fraction(dur_grid).denominator == 1):
        return False
    end = open_time + dur_grid
    if end < state['stream_time']:
        return False
    if state['num'] is not None and state['den'] is not None:
        bar_len = Fraction(state['num'] * 128, state['den'])
        if end > bar_len:
            return False
    if is_tup:
        step = _beat_step(state)
        if step is not None:
            beat_end = (open_time // step + 1) * step
            if end > beat_end:
                return False
    if end > state['stream_time']:
        for channel, position in state['clock'].items():
            if channel == (hand, voice) or position != state['stream_time']:
                continue
            if not any(
                frame['hand'] == channel[0] and frame['voice'] == channel[1]
                for frame in state['open']
            ):
                return False
    return True


def _open_allowed(state: dict, hand: str, voice: int) -> bool:
    count = state['bar_voice_count'][hand]
    if voice > count:
        return False
    if any(f['hand'] == hand and f['voice'] == voice for f in state['open']):
        return False
    onset = _chan_clock(state, hand, voice)
    if state['num'] is not None and state['den'] is not None:
        bar_len = Fraction(state['num'] * 128, state['den'])
        if onset >= bar_len:
            return False
    prior = []
    if hand == 'r':
        prior.extend(('l', depth)
                     for depth in range(state['bar_voice_count']['l']))
    prior.extend((hand, depth) for depth in range(voice))
    for prior_hand, prior_voice in prior:
        if (_chan_clock(state, prior_hand, prior_voice) == onset
                and not any(frame['hand'] == prior_hand
                            and frame['voice'] == prior_voice
                            for frame in state['open'])):
            return False
    return _open_time_ok(state, hand, voice)


def _bar_accounts_settled(state: dict) -> bool:
    """Every voice's clock has reached the bar length, so a barline is due.

    Unknown accounting (no meter yet, uncertain clocks) fails open, and so
    does an already-overshot clock — the bar is unprintable either way, and
    blocking the barline there would only burn tokens.
    """
    if state['num'] is None or state['den'] is None:
        return True
    bar_len = Fraction(state['num'] * 128, state['den'])
    for hand in _HANDS:
        if state['tuncert'][hand]:
            continue
        for depth in range(state['bar_voice_count'][hand]):
            if state['clock'].get((hand, depth), Fraction(0)) < bar_len:
                return False
    return True


def _address_has_action(state: dict, c: dict, depth: int) -> bool:
    for hand in _HANDS:
        if _open_allowed(state, hand, depth):
            return True
        frame = next((f for f in reversed(state['open'])
                      if f['hand'] == hand and f['voice'] == depth), None)
        if frame is not None and frame.get('has_tup') \
                and not frame.get('tup_sealed'):
            continue
        if frame is not None and any(
                _close_duration_ok(
                    state,
                    hand,
                    depth,
                    frame['open_time'],
                    grid,
                    frame.get('has_tup', False),
                )
                for grid in c['dur_grids']):
            return True
    return False


def _address_prefix_has_action(state: dict, c: dict, minimum_depth: int) -> bool:
    """Whether more `<v>` prefixes can reach a contiguous legal address."""
    maximum_depth = max(state['bar_voice_count'].values())
    return any(
        _address_has_action(state, c, depth)
        for depth in range(minimum_depth, maximum_depth + 1)
    )


def _decode_allow_mask(state: dict, c: dict) -> torch.Tensor:
    allow = torch.zeros(c['vocab_size'], dtype=torch.bool, device=c['device'])
    prev = state['prev']

    def _allow_pitches() -> None:
        """Pitch successors with strict chord ordering."""
        if state['last_member_midi'] is not None:
            # Chord members are ordered low->high by sounding pitch.
            # Same-key duplicates are absorbed before tokenization, so a
            # repeated sounding pitch cannot be another chord member.
            higher = c['pitch_midi_vec'] > state['last_member_midi']
            allow[c['pitch_idx'][higher]] = True
        else:
            allow[c['pitch_idx']] = True

    def _untie_legal(frame) -> bool:
        """A landing needs a same-hand same-pitch pending at this seam;
        capsule slots land at the first slot only.  Unknown accounting and
        the scope head (departures outside the chunk) fail open."""
        if state['last_pitch_rest'] or frame is None:
            return False
        if frame.get('has_tup') and frame.get('slot_count', 0) > 1:
            return False
        onset = frame.get('open_time')
        if state['bars_seen'] <= 1 and (onset == 0 or onset is None):
            return True                   # chunk head: departure out of scope
        return _pending_landing(
            state, frame['hand'], state['last_pitch_id'], onset)

    # --- positions with a single legal successor class ------------------------
    if prev == 'start':
        allow[c['bar_id']] = True
        return allow
    if prev == 'bar':
        allow[c['num_idx']] = True
        return allow
    if prev == 'num':
        allow[c['den_idx']] = True
        return allow
    if prev == 'den':
        allow[c['key_idx']] = True
        return allow
    if prev == 'open':
        _allow_pitches()
        allow[c['tup_id']] = True         # capsule opens right after the hand
        return allow
    if prev == 'tup':
        allow[c['grid_id']] = True        # a capsule starts with its first slot
        return allow
    if prev == 'tup_grid':
        _allow_pitches()                  # a slot holds at least one pitch
        return allow
    if prev == 'close':
        lc = state['last_close']
        if lc is None:
            allow[c['dur_idx']] = True    # accounting unknown: fail open
        else:
            hand, voice, open_time, _pids, is_tup = lc
            fits = [
                _close_duration_ok(
                    state, hand, voice, open_time, g, is_tup
                )
                for g in c['dur_grids']
            ]
            if any(fits):
                allow[c['dur_idx'][torch.tensor(fits, device=c['device'])]] = True
            else:
                # Filters durations; never lifts "a close is followed by one"
                # — an empty mask would, since the caller drops it wholesale.
                allow[c['dur_idx']] = True
        return allow
    if prev == 'voice':
        depth = state['address_depth']
        if _address_prefix_has_action(state, c, depth + 1):
            allow[c['voice_id']] = True
        for open_id, hand in c['opens'].items():
            if _open_allowed(state, hand, depth):
                allow[open_id] = True
        for close_id, open_id in c['open_of_close'].items():
            hand = c['opens'][open_id]
            frame = next(
                (f for f in reversed(state['open'])
                 if f['hand'] == hand and f['voice'] == depth), None,
            )
            if frame is not None and frame.get('has_tup') \
                    and not frame.get('tup_sealed'):
                continue
            if (frame is not None and any(_close_duration_ok(
                    state,
                    hand,
                    depth,
                    frame['open_time'],
                    grid,
                    frame.get('has_tup', False),
                )
                    for grid in c['dur_grids'])):
                allow[close_id] = True
        return allow

    # --- bracket content continuing -------------------------------------------
    if prev in ('pitch', 'untie'):
        frame = state['open'][-1] if state['open'] else None
        if frame is not None and frame.get('has_tup') \
                and not frame.get('tup_sealed'):
            # Capsule interior: more of this slot, a new slot, or the seal.
            # A slot is pitches or a lone rest — after r nothing joins it.
            if not state['last_pitch_rest']:
                _allow_pitches()
            if prev == 'pitch' and _untie_legal(frame):
                allow[c['untie_id']] = True
            step = _beat_step(state)
            if step is None or frame.get('slot_count', 0) < int(step):
                allow[c['grid_id']] = True
            allow[c['tup_close_id']] = True
            return allow
        # Rests never join chords: after r the bracket holds nothing more.
        if not state['last_pitch_rest']:
            _allow_pitches()
        if prev == 'pitch' and _untie_legal(frame):
            allow[c['untie_id']] = True
    if prev == 'dur' and state['last_close_sounding']:
        allow[c['tie_id']] = True         # <tie> trails a duration

    # --- structural events ----------------------------------------------------
    for close_id, open_id in c['open_of_close'].items():
        hand = c['opens'][open_id]
        voice = 0
        frame = next(
            (f for f in reversed(state['open'])
             if f['hand'] == hand and f['voice'] == voice),
            None,
        )
        if frame is not None and frame.get('has_tup') \
                and not frame.get('tup_sealed'):
            continue                      # a capsule seals before it closes
        if (frame is not None
                and any(_close_duration_ok(
                    state,
                    hand,
                    voice,
                    frame['open_time'],
                    g,
                    frame.get('has_tup', False),
                )
                    for g in c['dur_grids'])):
            allow[close_id] = True
    for open_id, hand in c['opens'].items():
        if _open_allowed(state, hand, 0):
            allow[open_id] = True
    if not state['open'] and _bar_accounts_settled(state):
        allow[c['bar_id']] = True
        allow[c['eos_id']] = True
    if c['voice_id'] is not None and _address_prefix_has_action(state, c, 1):
        allow[c['voice_id']] = True
    return allow


def _apply_decode_constraints(
    log_p: torch.Tensor, beam_states: list, constraints: dict, neg_inf: float,
) -> None:
    for b in range(log_p.shape[0]):
        allow = _decode_allow_mask(beam_states[b], constraints)
        if bool(allow.any()):
            log_p[b].masked_fill_(~allow, neg_inf)


def _update_beam_state(state: dict, tok: int, c: dict) -> dict:
    new = {
        'prev': 'free',
        'open': [dict(f) for f in state['open']],
        'address_depth': state['address_depth'],
        'bar_voice_count': dict(state['bar_voice_count']),
        'last_pitch_rest': state['last_pitch_rest'],
        'last_close_sounding': state['last_close_sounding'],
        'num': state['num'],
        'den': state['den'],
        'pending_num': state['pending_num'],
        'stream_time': state['stream_time'],
        'clock': dict(state['clock']),
        'tuncert': dict(state['tuncert']),
        'last_close': None,
        'last_member_midi': state['last_member_midi'],
        'pending_ties': dict(state['pending_ties']),
        'tie_context': None,
        'last_pitch_id': state['last_pitch_id'],
        'bars_seen': state['bars_seen'],
    }

    if tok == c['bar_id']:
        new['prev'] = 'bar'
        new['stream_time'] = Fraction(0)
        new['clock'] = {(hand, 0): Fraction(0) for hand in _HANDS}
        new['bar_voice_count'] = {'l': 1, 'r': 1}
        new['address_depth'] = 0
        new['tuncert'] = {'l': False, 'r': False}
        new['bars_seen'] = state['bars_seen'] + 1
        # The pairing window is two bars: this bar's departure records
        # age by one, older ones drop (an unlanded chord member simply
        # released at its close).
        new['pending_ties'] = {
            key: 1
            for key, age in state['pending_ties'].items()
            if age == 0
        }
        return new
    if tok in c['num_ids']:
        new['prev'] = 'num'
        new['pending_num'] = c['num_val'][tok]
        return new
    if tok in c['den_ids']:
        new['prev'] = 'den'
        new['num'] = state['pending_num']
        new['den'] = c['den_val'][tok]
        new['pending_num'] = None
        return new
    if tok in c['key_ids']:
        new['prev'] = 'key'
        return new
    if tok == c['grid_id']:
        # Capsule slot marker: open the next hidden-time slot.
        if new['open']:
            frame = new['open'][-1]
            frame['slot_count'] = frame.get('slot_count', 0) + 1
            frame['last_slot_pitches'] = []
            frame['last_slot_sounding'] = False
        new['last_member_midi'] = None
        new['prev'] = 'tup_grid'
        return new
    if tok == c['voice_id']:
        new['address_depth'] = state['address_depth'] + 1
        new['prev'] = 'voice'
        return new
    if tok in c['opens']:
        hand = c['opens'][tok]
        voice = state['address_depth']
        new['address_depth'] = 0
        if voice == state['bar_voice_count'][hand]:
            new['bar_voice_count'][hand] = voice + 1
            stats = c.get('stats', DECODE_STATS)
            stats['max_voice_width'] = max(stats['max_voice_width'], voice + 1)
            new['clock'][(hand, voice)] = Fraction(0)
        onset = _chan_clock(new, hand, voice)
        new['open'].append({
            'id': tok, 'hand': hand, 'voice': voice,
            'has_tup': False, 'sounding': False, 'tup_sealed': False,
            'slot_count': 0, 'last_slot_pitches': [],
            'last_slot_sounding': False, 'pitch_ids': [],
            'open_time': None if new['tuncert'][hand] else onset,
        })
        new['last_member_midi'] = None
        new['prev'] = 'open'
        return new
    if tok in c['open_of_close']:
        open_id = c['open_of_close'][tok]
        hand = c['opens'][open_id]
        voice = state['address_depth']
        new['address_depth'] = 0
        info = None
        for si in range(len(new['open']) - 1, -1, -1):
            if new['open'][si]['hand'] == hand and new['open'][si]['voice'] == voice:
                info = new['open'].pop(si)
                break
        if info is not None and info['has_tup']:
            # A capsule departure refers to the last member only.
            new['last_close_sounding'] = bool(info['last_slot_sounding'])
            close_pitches = tuple(
                pid for pid in info['last_slot_pitches']
                if pid != c['rest_id'])
        else:
            new['last_close_sounding'] = bool(info and info['sounding'])
            close_pitches = tuple(
                pid for pid in (info['pitch_ids'] if info else ())
                if pid != c['rest_id'])
        new['last_close'] = (
            hand,
            voice,
            info['open_time'] if info else None,
            close_pitches,
            bool(info and info.get('has_tup')),
        )
        new['last_member_midi'] = None
        new['prev'] = 'close'
        return new
    if tok in c['dur_ids']:
        new['prev'] = 'dur'
        lc = state['last_close']
        if lc is not None:
            hand, voice, open_time, close_pitches, _is_tup = lc
            if open_time is None:
                new['tuncert'][hand] = True
            else:
                end = open_time + c['dur_grid'][tok]
                new['clock'][(hand, voice)] = end
                new['stream_time'] = end
            if close_pitches:
                # A <tie> after this duration departs these pitches.
                new['tie_context'] = (hand, close_pitches)
        return new
    if tok == c['tie_id']:
        ctx = state['tie_context']
        if ctx is not None:
            hand, close_pitches = ctx
            for pid in close_pitches:
                new['pending_ties'][(hand, pid)] = 0   # departed this bar
        return new
    if tok == c['tup_id']:
        if new['open']:
            new['open'][-1]['has_tup'] = True
        new['last_member_midi'] = None
        new['prev'] = 'tup'
        return new
    if tok == c['tup_close_id']:
        if new['open']:
            new['open'][-1]['tup_sealed'] = True
        new['last_member_midi'] = None
        new['prev'] = 'tup_close'
        return new
    if tok in c['pitch_ids']:
        is_rest = tok == c['rest_id']
        new['last_pitch_rest'] = is_rest
        new['last_pitch_id'] = tok
        frame = new['open'][-1] if new['open'] else None
        if frame is not None:
            if frame['has_tup']:
                frame['last_slot_pitches'] = (
                    list(frame['last_slot_pitches']) + [tok])
                if not is_rest:
                    frame['last_slot_sounding'] = True
                    frame['sounding'] = True
            else:
                frame['pitch_ids'] = list(frame['pitch_ids']) + [tok]
                if not is_rest:
                    frame['sounding'] = True
        # Strict low->high holds in plain chords and within capsule slots.
        new['last_member_midi'] = (
            None if is_rest else c['pitch_midi'][tok])
        # Landings are validated, never forced: same-hand same-pitch
        # overlaps across voices make a plain re-strike legal even while
        # a tie is pending.
        new['prev'] = 'pitch'
        return new
    if tok == c['untie_id']:
        new['prev'] = 'untie'
        return new
    return new


def _stuck_frames(state: dict, c: dict) -> List[dict]:
    """Open brackets that no duration token can close at this instant."""
    return [
        frame for frame in state['open']
        if frame.get('open_time') is not None
        and not (frame.get('has_tup') and not frame.get('tup_sealed'))
        and not any(
            _close_duration_ok(
                state, frame['hand'], frame['voice'], frame['open_time'],
                grid, frame.get('has_tup', False),
            )
            for grid in c['dur_grids']
        )
    ]


def _frame_open_in(state: dict, frame: dict) -> bool:
    return any(
        f['hand'] == frame['hand'] and f['voice'] == frame['voice']
        and f.get('open_time') == frame['open_time']
        for f in state['open']
    )


def _forced_tokens_legal(state: dict, tokens: List[int], c: dict) -> bool:
    for tok in tokens:
        allow = _decode_allow_mask(state, c)
        if not bool(allow.any()) or not bool(allow[tok]):
            return False
        state = _update_beam_state(state, tok, c)
    return True


def _seam_repair(
    states: List[dict], dead_step: int, c: dict,
) -> Optional[Tuple[int, List[int]]]:
    """Derive the seam of a bracket stuck at a dead end.

    Returns the step to resume from and the tokens to force there (the
    stuck hand's close, the seam glyph, <tie>), or None when no seam can
    be derived: tuplet-grid onsets (spell has no interior line there),
    spans that already have a single glyph (stuck for another reason), or
    seams at which the forced close is not grammar-legal.
    """
    from src.a2s.piano.tokenizer import (
        RECIP_TO_GRID, get_metric_tree, spell,
    )
    state = states[dead_step]
    if state['num'] is None or state['den'] is None:
        return None
    tree = get_metric_tree(state['num'], state['den'])
    stream = state['stream_time']
    for frame in sorted(_stuck_frames(state, c), key=lambda f: f['open_time']):
        open_time = frame['open_time']
        span = stream - open_time
        if (Fraction(open_time).denominator != 1
                or Fraction(span).denominator != 1 or span <= 0):
            continue
        try:
            glyphs = spell(int(open_time), int(span), tree)
        except ValueError:
            continue
        if len(glyphs) < 2:
            continue
        seam = open_time + RECIP_TO_GRID[glyphs[0]]
        if not (open_time < seam < stream):
            continue
        close_id = next(
            cid for cid, oid in c['open_of_close'].items()
            if c['opens'][oid] == frame['hand']
        )
        glyph_id = next(
            tid for tid, grid in c['dur_grid'].items()
            if grid == RECIP_TO_GRID[glyphs[0]]
        )
        forced = [close_id, glyph_id, c['tie_id']]
        # Canonical position: the first step at which the stream stands on
        # the seam with the bracket open (closes due precede opens).  When
        # the stream jumped over the seam instead, the close is inserted at
        # the last step before the jump.
        # Stream time restarts every bar, so the bracket is identified
        # within the dead end's bar only.
        same_bar = [
            j for j in range(dead_step)
            if states[j]['bars_seen'] == state['bars_seen']
            and _frame_open_in(states[j], frame)
        ]
        at_seam = [j for j in same_bar if states[j]['stream_time'] == seam]
        before_seam = [j for j in same_bar if states[j]['stream_time'] < seam]
        for j in sorted(at_seam) + sorted(before_seam, reverse=True):
            if states[j]['address_depth'] != frame['voice']:
                continue
            if _forced_tokens_legal(states[j], forced, c):
                return j, forced
    return None


# =============================================================================
# Greedy decode (Transformer autoregressive)
# =============================================================================


def _greedy_decode(
    model: PianoModel,
    memory: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    tokenizer: KernTokenizer,
    stop_after_n_bars: Optional[int] = None,
    memory_moments: Optional[torch.Tensor] = None,
    forced_schema: Optional[List[tuple]] = None,
) -> Tuple[List[int], List[float]]:
    """Greedy decode with KV cache + grammar constraints."""
    constraints = _build_decode_constraints(tokenizer, torch.device('cpu'))
    log_probs: List[float] = []
    generated = _greedy_decode_cached(
        model, memory, bos_id, eos_id, max_len,
        constraints, log_probs, stop_after_n_bars,
        memory_moments=memory_moments,
        forced_schema=forced_schema,
    )
    return generated, log_probs


_SCHEMA_SLOT = {'bar': 0, 'num': 1, 'den': 2}


def _apply_forced_schema(
    allow: torch.Tensor, state: dict, forced_schema: List[tuple],
) -> torch.Tensor:
    """Narrow a schema slot to the bar's decided token.

    ``forced_schema[b]`` is ``(num_id, den_id, key_id)`` for the b-th bar the
    decoder opens; bars past the end reuse the last entry.  Every other slot
    keeps the grammar mask, so the notes stay the decoder's own."""
    slot = _SCHEMA_SLOT.get(state['prev'])
    if slot is None or not forced_schema:
        return allow
    entry = forced_schema[min(max(state['bars_seen'] - 1, 0), len(forced_schema) - 1)]
    tok = entry[slot]
    if tok is None or not bool(allow[tok]):
        return allow
    only = torch.zeros_like(allow)
    only[tok] = True
    return only


def _greedy_decode_cached(
    model: PianoModel,
    memory: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    constraints: dict,
    log_probs: List[float],
    stop_after_n_bars: Optional[int] = None,
    memory_moments: Optional[torch.Tensor] = None,
    forced_schema: Optional[List[tuple]] = None,
) -> List[int]:
    ids, probabilities, stats = _greedy_decode_cached_batch(
        model, memory, bos_id, eos_id, max_len, constraints,
        [stop_after_n_bars], memory_moments=memory_moments,
        forced_schemas=[forced_schema],
    )
    log_probs.extend(probabilities[0])
    DECODE_STATS.update(stats[0])
    return ids[0]


def _greedy_decode_cached_batch(
    model: PianoModel,
    memory: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    constraints: dict,
    stop_after_n_bars: List[Optional[int]],
    memory_moments: Optional[torch.Tensor] = None,
    forced_schemas: Optional[List[Optional[List[tuple]]]] = None,
) -> Tuple[List[List[int]], List[List[float]], List[dict]]:
    """Keep grammar, rewind history and cache positions independent per item."""
    import torch.nn.functional as F

    batch_size, t_enc, dim = memory.shape
    if len(stop_after_n_bars) != batch_size:
        raise ValueError('stop_after_n_bars must have one entry per item')
    forced_schemas = forced_schemas or [None] * batch_size
    if len(forced_schemas) != batch_size:
        raise ValueError('forced_schemas must have one entry per item')
    device = memory.device
    # Grammar is a CPU state machine; scalar GPU mask writes force tiny launches.
    constraints = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in constraints.items()
    }
    constraints['device'] = torch.device('cpu')
    layers = model.decoder.layers
    n_heads = layers[0].self_attn.num_heads
    head_dim = dim // n_heads
    fourierpe = getattr(model, 'coordinate_delivery', 'memory') == 'cross_attention'
    if fourierpe and memory_moments is None and (
        model.fourierpe_key_proj is not None or model.fourierpe_value_proj is not None
    ):
        raise ValueError('cross-attention coordinate delivery needs memory_moments')

    cross_kv = []
    for i, layer in enumerate(layers):
        ca = layer.multihead_attn
        weight, bias = ca.in_proj_weight, ca.in_proj_bias
        key = F.linear(memory, weight[dim:2*dim], bias[dim:2*dim])
        value = F.linear(memory, weight[2*dim:], bias[2*dim:])
        if fourierpe and model.fourierpe_key_proj is not None:
            key = key + model.fourierpe_key_proj[i](memory_moments.to(key.dtype))
        if fourierpe and model.fourierpe_value_proj is not None:
            value = value + model.fourierpe_value_proj[i](memory_moments.to(value.dtype))
        cross_kv.append((
            key.view(batch_size, t_enc, n_heads, head_dim).transpose(1, 2),
            value.view(batch_size, t_enc, n_heads, head_dim).transpose(1, 2),
        ))

    stats = [dict(masked_argmax=0, steps=0, max_voice_width=1,
                  boundary_completed_bars=0, stop_reason='max_len', seam_repairs=0)
             for _ in range(batch_size)]
    item_constraints = [dict(constraints, stats=item) for item in stats]
    states = [_initial_beam_state() for _ in range(batch_size)]
    histories = [[] for _ in range(batch_size)]
    generated = [[bos_id] for _ in range(batch_size)]
    log_probs = [[] for _ in range(batch_size)]
    forced = [[] for _ in range(batch_size)]
    repaired = [set() for _ in range(batch_size)]
    steps = [0] * batch_size
    bars = [0] * batch_size
    active = [max_len > 1] * batch_size
    self_kv = [None] * len(layers)
    row_ids = torch.arange(batch_size, device=device)
    cache_positions = torch.arange(max(1, max_len - 1), device=device)

    while any(active):
        masks, choices = [], []
        for b in range(batch_size):
            while active[b]:
                state = states[b]
                allow = _decode_allow_mask(state, item_constraints[b])
                if forced_schemas[b] is not None:
                    allow = _apply_forced_schema(allow, state, forced_schemas[b])
                histories[b].append(state)
                if bool(allow.any()) or forced[b]:
                    break
                repair_key = (
                    state['bars_seen'], state['stream_time'],
                    tuple(sorted((f['hand'], f['voice'], f['open_time'])
                                 for f in state['open'])),
                )
                repair = None if repair_key in repaired[b] else _seam_repair(
                    histories[b], steps[b], item_constraints[b],
                )
                if repair is None:
                    break
                repaired[b].add(repair_key)
                resume, forced[b] = repair
                del generated[b][resume + 1:]
                del log_probs[b][resume:]
                del histories[b][resume:]
                bars[b] = generated[b].count(constraints['bar_id'])
                stats[b]['boundary_completed_bars'] = max(0, bars[b] - 1)
                stats[b]['seam_repairs'] += 1
                states[b] = _initial_beam_state()
                for tok in generated[b][1:]:
                    states[b] = _update_beam_state(states[b], tok, item_constraints[b])
                steps[b] = resume
            if not active[b]:
                allow = torch.ones(constraints['vocab_size'], dtype=torch.bool)
            masks.append(allow)
            choices.append(forced[b][0] if active[b] and forced[b] else -1)

        last_active_step = max(step for step, live in zip(steps, active) if live)
        compute_steps = [step if live else last_active_step for step, live in zip(steps, active)]
        current = torch.tensor([g[-1] for g in generated], dtype=torch.long, device=device)
        x = model.token_embedding(current[:, None])
        if fourierpe and model.fourierpe_token_proj is not None:
            phase = torch.tensor([[bar_phase(state)] for state in states], device=device)
            score_feats = model.score_fourierpe_features(phase)
            x = x + model.fourierpe_token_proj(score_feats.to(x.dtype))
        cache_length = max(compute_steps) + 1
        # A rewind changes only its row's visible prefix; stale suffixes stay masked.
        attention_mask = None
        if min(compute_steps) != max(compute_steps):
            positions = torch.tensor(compute_steps, dtype=torch.long, device=device)
            x = x + model.pos_encoding.pe[0, positions, :][:, None, :]
            attention_mask = (cache_positions[:cache_length][None, :] <= positions[:, None])[:, None, None, :]
        else:
            x = x + model.pos_encoding.pe[:, cache_length - 1:cache_length]

        for i, layer in enumerate(layers):
            sa = layer.self_attn
            weight, bias = sa.in_proj_weight, sa.in_proj_bias
            query = F.linear(x, weight[:dim], bias[:dim])
            key = F.linear(x, weight[dim:2*dim], bias[dim:2*dim])
            value = F.linear(x, weight[2*dim:], bias[2*dim:])
            query = query.view(batch_size, 1, n_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, 1, n_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, 1, n_heads, head_dim).transpose(1, 2)
            if self_kv[i] is None:
                # Zero the unused suffix so masked slots cannot introduce NaNs.
                self_kv[i] = (
                    key.new_zeros((batch_size, n_heads, max_len - 1, head_dim)),
                    value.new_zeros((batch_size, n_heads, max_len - 1, head_dim)),
                )
            key_cache, value_cache = self_kv[i]
            if attention_mask is None:
                key_cache[:, :, cache_length - 1:cache_length].copy_(key)
                value_cache[:, :, cache_length - 1:cache_length].copy_(value)
            else:
                key_cache[row_ids, :, positions, :] = key[:, :, 0, :]
                value_cache[row_ids, :, positions, :] = value[:, :, 0, :]
            sa_out = F.scaled_dot_product_attention(
                query, key_cache[:, :, :cache_length], value_cache[:, :, :cache_length],
                attn_mask=attention_mask,
            )
            sa_out = sa.out_proj(sa_out.transpose(1, 2).reshape(batch_size, 1, dim))
            x = layer.norm1(x + sa_out)
            ca = layer.multihead_attn
            query = F.linear(x, ca.in_proj_weight[:dim], ca.in_proj_bias[:dim])
            query = query.view(batch_size, 1, n_heads, head_dim).transpose(1, 2)
            ca_out = F.scaled_dot_product_attention(query, *cross_kv[i])
            ca_out = ca.out_proj(ca_out.transpose(1, 2).reshape(batch_size, 1, dim))
            x = layer.norm2(x + ca_out)
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))

        logits = model.output_proj(x[:, 0]).float()
        raw_argmax = logits.argmax(dim=-1)
        allow = torch.stack(masks).to(device)
        has_choices = allow.any(dim=-1)
        logits.masked_fill_(~allow & has_choices[:, None], float('-inf'))
        selected = logits.argmax(dim=-1)
        forced_ids = torch.tensor(choices, device=device)
        selected = torch.where(forced_ids >= 0, forced_ids, selected)
        probability = torch.log_softmax(logits, dim=-1).gather(1, selected[:, None])[:, 0]
        masked = has_choices & ~allow.gather(1, raw_argmax[:, None])[:, 0]
        # One device-to-host synchronization for all items, including diagnostics.
        decisions = torch.stack((selected.float(), probability, masked.float()), dim=-1).cpu().tolist()
        for b, (token, probability, was_masked) in enumerate(decisions):
            if not active[b]:
                continue
            token = int(token)
            if forced[b]:
                forced[b].pop(0)
            stats[b]['masked_argmax'] += int(was_masked)
            stats[b]['steps'] += 1
            generated[b].append(token)
            log_probs[b].append(probability)
            states[b] = _update_beam_state(states[b], token, item_constraints[b])
            steps[b] += 1
            if token == eos_id:
                stats[b]['boundary_completed_bars'] = bars[b]
                stats[b]['stop_reason'] = 'eos'
                active[b] = False
            elif token == constraints['bar_id']:
                bars[b] += 1
                stats[b]['boundary_completed_bars'] = max(0, bars[b] - 1)
                if stop_after_n_bars[b] is not None and bars[b] > stop_after_n_bars[b]:
                    stats[b]['stop_reason'] = 'bar_limit'
                    active[b] = False
            if steps[b] >= max_len - 1:
                active[b] = False
            if not active[b]:
                # Finished rows remain padded in this batch, with a valid cache index.
                steps[b] = max(0, steps[b] - 1)
    return generated, log_probs, stats


def _beam_search(
    model: PianoModel,
    memory: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    num_beams: int,
    tokenizer: KernTokenizer,
    length_penalty: float = 1.0,
) -> Tuple[List[int], List[float]]:
    """Beam search with grammar constraints.

    TODO: add KV cache (currently naive O(S²) per step).
    Batches all beams into one model.decode call per step.
    """
    NEG_INF = float("-inf")
    device = memory.device

    constraints = _build_decode_constraints(tokenizer, device)

    seqs = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    beam_states = [_initial_beam_state()]
    log_probs_per_beam: List[List[float]] = [[]]

    mem_expanded = memory  # will expand after first step

    for _ in range(max_len - 1):
        B_cur = seqs.shape[0]
        if B_cur > 1:
            mem_b = memory.expand(B_cur, -1, -1)
        else:
            mem_b = memory

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model.decode(mem_b, seqs)
        log_p = torch.log_softmax(logits[:, -1, :].float(), dim=-1)

        # Constraints first: a finished beam must keep absorbing on EOS, which the
        # grammar mask would otherwise forbid (its brackets are irrelevant now).
        _apply_decode_constraints(log_p, beam_states, constraints, NEG_INF)

        done = seqs[:, -1].eq(eos_id)
        log_p[done, :] = NEG_INF
        log_p[done, eos_id] = 0.0

        cand = scores.unsqueeze(1) + log_p
        flat = cand.reshape(-1)
        V = logits.shape[-1]
        top_scores, top_idx = flat.topk(min(num_beams, flat.numel()))
        beam_idx = top_idx // V
        token_idx = top_idx % V

        chosen_lp = log_p[beam_idx, token_idx].tolist()

        seqs = torch.cat([seqs[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = top_scores

        beam_states = [
            _update_beam_state(beam_states[beam_idx[b].item()], token_idx[b].item(), constraints)
            for b in range(len(top_scores))
        ]
        log_probs_per_beam = [list(log_probs_per_beam[beam_idx[b].item()]) for b in range(len(top_scores))]
        for b in range(len(top_scores)):
            log_probs_per_beam[b].append(float(chosen_lp[b]))

        if seqs[:, -1].eq(eos_id).all():
            break

    pen_scores = scores / (seqs.shape[1] ** length_penalty)
    best = int(pen_scores.argmax())
    return seqs[best].tolist(), log_probs_per_beam[best]


# =============================================================================
# Chunk-level generation
# =============================================================================


@torch.no_grad()
def generate_kern_chunk(
    model: PianoModel,
    mel_chunk: torch.Tensor,
    tokenizer: KernTokenizer,
    max_len: int = 2048,
    num_beams: int = 1,
    device: str = "cuda",
    stop_after_n_bars: Optional[int] = None,
    forced_schema: Optional[List[tuple]] = None,
    memory_moments_override: Optional[torch.Tensor] = None,
) -> Tuple[List[int], List[float]]:
    """Encode + decode a single mel chunk.

    stop_after_n_bars only applies to greedy decoding (beam prefixes are
    not invariant to early stopping, so beam search ignores it).
    forced_schema pins each bar's (num, den, key) tokens to a decision made
    before decoding; greedy only.

    Returns:
        token_ids: BOS/EOS stripped.
        log_probs: per-token log-probability (aligned with token_ids).
    """
    bos_id = tokenizer.vocab["<sos>"]
    eos_id = tokenizer.vocab["<eos>"]

    mel_chunk = mel_chunk.to(device)
    memory_moments = None
    if getattr(model, 'coordinate_delivery', 'memory') == 'cross_attention':
        memory, memory_moments = model.encode_with_fourierpe(mel_chunk)
        if num_beams > 1:
            raise NotImplementedError(
                'beam search does not carry the cross-attention Fourier PE'
            )
    else:
        memory = model.encode(mel_chunk)
    if memory_moments_override is not None:
        if memory_moments is None:
            raise ValueError('memory moments can only be overridden under cross-attention delivery')
        memory_moments = merge_memory_moments(memory_moments, memory_moments_override, memory.dtype)

    if num_beams > 1:
        if forced_schema is not None:
            raise NotImplementedError('beam search does not take a forced schema')
        ids, log_probs = _beam_search(
            model, memory, bos_id, eos_id, max_len, num_beams, tokenizer,
        )
    else:
        ids, log_probs = _greedy_decode(
            model, memory, bos_id, eos_id, max_len, tokenizer,
            stop_after_n_bars=stop_after_n_bars,
            memory_moments=memory_moments,
            forced_schema=forced_schema,
        )

    # Strip BOS
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    # Strip EOS and everything after
    for i, tok in enumerate(ids):
        if tok == eos_id:
            ids = ids[:i]
            log_probs = log_probs[:i] if log_probs else log_probs
            break
    if log_probs and len(log_probs) > len(ids):
        log_probs = log_probs[:len(ids)]
    return ids, log_probs


@torch.no_grad()
def generate_kern_chunks_batch(
    model: PianoModel,
    memory: torch.Tensor,
    tokenizer: KernTokenizer,
    max_len: int,
    stop_after_n_bars: List[int],
    memory_moments: Optional[torch.Tensor] = None,
    forced_schemas: Optional[List[Optional[List[tuple]]]] = None,
) -> Tuple[List[List[int]], List[List[float]], List[dict]]:
    """Greedily decode a batch of independently encoded score chunks."""
    bos_id = tokenizer.vocab["<sos>"]
    eos_id = tokenizer.vocab["<eos>"]

    constraints = _build_decode_constraints(tokenizer, torch.device('cpu'))
    ids_batch, log_probs_batch, stats = _greedy_decode_cached_batch(
        model,
        memory,
        bos_id,
        eos_id,
        max_len,
        constraints,
        stop_after_n_bars,
        memory_moments=memory_moments, forced_schemas=forced_schemas,
    )

    cleaned_ids = []
    cleaned_log_probs = []
    for ids, log_probs in zip(ids_batch, log_probs_batch):
        if ids and ids[0] == bos_id:
            ids = ids[1:]
        if eos_id in ids:
            eos_pos = ids.index(eos_id)
            ids = ids[:eos_pos]
            log_probs = log_probs[:eos_pos]
        if len(log_probs) > len(ids):
            log_probs = log_probs[:len(ids)]
        cleaned_ids.append(ids)
        cleaned_log_probs.append(log_probs)
    return cleaned_ids, cleaned_log_probs, stats


# =============================================================================
# Kern reconstruction (no schema heads — direct from token stream)
# =============================================================================


def reconstruct_kern(
    token_ids: List[int],
    tokenizer: KernTokenizer,
    n_spines: int = 2,
    release_normalizations: Optional[List[dict]] = None,
    tie_normalizations: Optional[List[dict]] = None,
    readback_mismatches: Optional[List[dict]] = None,
    tail_rest_completions: Optional[List[dict]] = None,
    tail_rest_completion_bars: Optional[Set[int]] = None,
) -> str:
    """Reconstruct **kern from decoder token IDs."""
    sos_id = tokenizer.vocab["<sos>"]
    eos_id = tokenizer.vocab["<eos>"]

    raw_tokens: List[str] = []
    for tid in token_ids:
        if tid == sos_id or tid == eos_id:
            continue
        token = tokenizer.id_to_token.get(tid, "")
        if not token or token == "<pad>":
            continue
        raw_tokens.append(token)

    return reconstruct_kern_from_bracket_tokens(
        raw_tokens,
        release_normalizations=release_normalizations,
        tie_normalizations=tie_normalizations,
        readback_mismatches=readback_mismatches,
        tail_rest_completions=tail_rest_completions,
        tail_rest_completion_bars=tail_rest_completion_bars,
    )


# =============================================================================
# Token / kern utilities
# =============================================================================


def truncate_token_ids_to_n_bars(token_ids: List[int], bar_id: int, n_bars: int) -> List[int]:
    count = 0
    for i, tok in enumerate(token_ids):
        if tok == bar_id:
            count += 1
            if count > n_bars:
                return token_ids[:i]
    return token_ids


def split_tokens_by_bar(token_ids: List[int], bar_id: int) -> List[List[int]]:
    measures: List[List[int]] = []
    current: List[int] = []
    for tok in token_ids:
        current.append(tok)
        if tok == bar_id:
            measures.append(current)
            current = []
    if current:
        measures.append(current)
    return measures


def dedup_trailing_bars(token_ids: List[int], bar_id: int) -> List[int]:
    measures = split_tokens_by_bar(token_ids, bar_id)
    if len(measures) < 2:
        return token_ids
    while len(measures) >= 2 and measures[-1] == measures[-2]:
        measures.pop()
    return [tok for m in measures for tok in m]


def kern_to_midi(kern_content: str, output_midi_path: str) -> bool:
    import tempfile
    import music21 as _music21
    from src.score.generate_score import kern_to_musicxml

    tmp_krn = tmp_xml = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".krn", delete=False, mode="w") as f:
            tmp_krn = f.name
            f.write(kern_content)
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            tmp_xml = f.name
        kern_to_musicxml(tmp_krn, tmp_xml)
        score = _music21.converter.parse(tmp_xml)
        score.write("midi", fp=output_midi_path)
        return True
    except Exception as e:
        logger.warning(f"kern → MIDI failed: {e}")
        return False
    finally:
        for p in [tmp_krn, tmp_xml]:
            if p and os.path.exists(p):
                os.remove(p)


# =============================================================================
# Zeng baseline chunk filter
# =============================================================================

# Piano-A2S's pretrain.score results/chunks.csv; no default.
ZENG_CSV_PATH = os.environ.get("ZENG_CSV_PATH")


def _norm_zeng_piece(p: str) -> str:
    p = p.strip()
    if p[0].isupper():
        return "musesyn_" + p.lower()
    return p.lower().replace("#", "_")


def _norm_our_piece(p: str) -> str:
    p = p.strip().lower()
    for suf in ("~timgm6mb", "~salamandergrand", "~sgm", "~ydp"):
        if p.endswith(suf):
            return p[:-len(suf)]
    return p


def load_zeng_chunk_set() -> Set[Tuple[str, str, int]]:
    import csv
    if not ZENG_CSV_PATH:
        raise ValueError("set ZENG_CSV_PATH to Piano-A2S's pretrain.score results/chunks.csv")
    zeng_set: Set[Tuple[str, str, int]] = set()
    try:
        with open(ZENG_CSV_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                piece = _norm_zeng_piece(row["piece"])
                sf = row.get("soundfont", "").strip()
                measure = int(row["measure"])
                zeng_set.add((piece, sf, measure))
    except FileNotFoundError:
        logger.warning(f"Zeng CSV not found: {ZENG_CSV_PATH}")
    return zeng_set


# =============================================================================
# Reconstructed-fragment inspection
# =============================================================================

_METER_RE = re.compile(r"^\*M(\d+)/(\d+)$")
_KEYSIG_RE = re.compile(r"^\*k\[[^\]]*\]$")


def _parse_fragment_text(text: str):
    """Split a fragment into (key_sig, meter, measure blocks).

    A block is the lines from one barline row up to (not including) the
    next barline row. Header rows before the first barline provide the
    fragment's key/meter; '*-' terminators and orphan content rows before
    the first barline are dropped (orphans would corrupt alignment).
    """
    key_sig = meter = None
    blocks: List[List[str]] = []
    current: Optional[List[str]] = None

    for raw in text.splitlines():
        if not raw:
            continue
        first = raw.split("\t", 1)[0].strip()
        if first.startswith("="):
            if current is not None:
                blocks.append(current)
            current = [raw]
            continue
        if first == "*-":
            continue
        if current is None:
            if _KEYSIG_RE.match(first):
                key_sig = first
            else:
                m = _METER_RE.match(first)
                if m:
                    meter = first
            continue
        current.append(raw)
    if current is not None:
        blocks.append(current)
    return key_sig, meter, blocks


# =============================================================================
# Bar mode — GT-downbeat-guided sliding window
# =============================================================================


def _bars_available_from_downbeats(
    downbeat_frames: List[int],
    cursor: int,
    trust_frames: int,
    max_bars: int,
) -> int:
    """Maximum bars whose following boundary remains in the trust region."""
    chunk_start = downbeat_frames[cursor]
    available = 0
    for measure in range(cursor, min(cursor + max_bars, len(downbeat_frames))):
        next_measure = measure + 1
        if next_measure >= len(downbeat_frames):
            available += 1
            break
        if downbeat_frames[next_measure] - chunk_start > trust_frames:
            break
        available += 1
    return max(1, available)


def _accepted_prefix_token_ids(
    token_ids: List[int],
    tokenizer: KernTokenizer,
    n_bars: int,
) -> List[int]:
    bar_id = tokenizer.vocab["<bar>"]
    accepted = truncate_token_ids_to_n_bars(token_ids, bar_id, n_bars)
    if accepted.count(bar_id) != n_bars:
        raise ValueError(
            f"accepted prefix has {accepted.count(bar_id)} bar heads, "
            f"expected {n_bars}"
        )
    return accepted


def _bar_token_blocks(
    token_ids: List[int],
    tokenizer: KernTokenizer,
) -> List[List[int]]:
    bar_id = tokenizer.vocab["<bar>"]
    starts = [index for index, token in enumerate(token_ids) if token == bar_id]
    return [
        token_ids[start:(starts[index + 1] if index + 1 < len(starts)
                          else len(token_ids))]
        for index, start in enumerate(starts)
    ]


@dataclass
class _DecodeRequest:
    mel: torch.Tensor
    stop_after_n_bars: int
    forced_schema: Optional[List[tuple]]
    memory_moments_override: Optional[torch.Tensor]


@torch.no_grad()
def _generate_decode_requests(
    model: PianoModel,
    requests: List[_DecodeRequest],
    tokenizer: KernTokenizer,
    max_len: int,
    device: str,
    encoder_batch_size: int,
) -> Tuple[List[List[int]], List[List[float]], List[dict]]:
    memories, moments = [], []
    for start in range(0, len(requests), encoder_batch_size):
        items = requests[start:start + encoder_batch_size]
        mel = torch.cat([item.mel for item in items]).to(device)
        if getattr(model, 'coordinate_delivery', 'memory') == 'cross_attention':
            memory, moment = model.encode_with_fourierpe(mel)
        else:
            memory, moment = model.encode(mel), None
        memories.append(memory)
        if moment is not None:
            for index, item in enumerate(items):
                override = item.memory_moments_override
                moments.append(moment[index:index + 1] if override is None
                               else merge_memory_moments(moment[index:index + 1], override, memory.dtype))
        elif any(item.memory_moments_override is not None for item in items):
            raise ValueError('memory moments can only be overridden under cross-attention delivery')
    return generate_kern_chunks_batch(
        model, torch.cat(memories), tokenizer, max_len,
        [item.stop_after_n_bars for item in requests],
        memory_moments=torch.cat(moments) if moments else None,
        forced_schemas=[item.forced_schema for item in requests],
    )


def iter_kern_piece_from_downbeats(
    *,
    perf_id: str,
    model: PianoModel,
    mel: torch.Tensor,
    downbeat_frames: List[int],
    tokenizer: KernTokenizer,
    constraints_cpu: dict,
    chunk_frames: int,
    trust_frames: int,
    max_bars_per_chunk: int,
    max_len: int,
    num_beams: int,
    device: str,
    output_scopes: Optional[List[dict]] = None,
    schema_timeline: Optional[List[tuple]] = None,
    coordinate_intervention: Optional[CoordinateIntervention] = None,
    manifest_item: Optional[dict] = None,
    inference_batch_size: int = 1,
    encoder_batch_size: int = 1,
    stop_on_reconstruction_failure: bool = True,
) -> Iterator[dict]:
    """Batch independent scopes while each scope retains its attempt dependencies."""
    if inference_batch_size < 1 or encoder_batch_size < 1:
        raise ValueError('inference and encoder batch sizes must be positive')
    scopes = output_scopes or [
        dict(start=start, chunk=index, artifact_name=f'{perf_id}.{index}')
        for index, start in enumerate(range(0, len(downbeat_frames) - max_bars_per_chunk + 1,
                                            max_bars_per_chunk))
    ]
    scopes = [dict(scope, chunk=scope.get('chunk', index),
                   artifact_name=scope.get('artifact_name', f"{perf_id}.{scope.get('chunk', index)}"))
              for index, scope in enumerate(scopes)]
    for start in range(0, len(scopes), inference_batch_size):
        group = scopes[start:start + inference_batch_size]
        workers = [_iter_scope_attempts(
            perf_id=perf_id, model=model, mel=mel, downbeat_frames=downbeat_frames,
            tokenizer=tokenizer, constraints_cpu=constraints_cpu, chunk_frames=chunk_frames,
            trust_frames=trust_frames, max_bars_per_chunk=max_bars_per_chunk,
            max_len=max_len, num_beams=num_beams, device=device, output_scopes=[scope],
            schema_timeline=schema_timeline, coordinate_intervention=coordinate_intervention,
            manifest_item=manifest_item,
        ) for scope in group]
        pending = [next(worker) for worker in workers]
        while any(isinstance(item, _DecodeRequest) for item in pending):
            indices = [i for i, item in enumerate(pending) if isinstance(item, _DecodeRequest)]
            ids, log_probs, stats = _generate_decode_requests(
                model, [pending[i] for i in indices], tokenizer, max_len, device, encoder_batch_size,
            )
            for i, tokens, probabilities, item_stats in zip(indices, ids, log_probs, stats):
                pending[i] = workers[i].send((tokens, probabilities, item_stats))
        for worker in workers:
            worker.close()
        # Preserve artifact/log order, including the original pre-segmented stop policy.
        for result in pending:
            yield result
            if stop_on_reconstruction_failure and result['failed'] and 'output_log_entry' in result:
                return


def _iter_scope_attempts(
    *,
    perf_id: str,
    model: PianoModel,
    mel: torch.Tensor,
    downbeat_frames: List[int],
    tokenizer: KernTokenizer,
    constraints_cpu: dict,
    chunk_frames: int,
    trust_frames: int,
    max_bars_per_chunk: int,
    max_len: int,
    num_beams: int,
    device: str,
    output_scopes: Optional[List[dict]] = None,
    schema_timeline: Optional[List[tuple]] = None,
    coordinate_intervention: Optional[CoordinateIntervention] = None,
    manifest_item: Optional[dict] = None,
) -> Generator[_DecodeRequest | dict, Tuple[List[int], List[float], dict], None]:
    """Decode each five-bar scope independently from its own downbeat.

    ``schema_timeline[b]`` (optional) is the (num, den, key) token triple
    decided for piece bar b before decoding; a scope opening at bar
    ``cursor`` is forced to ``schema_timeline[cursor:]``."""
    if not downbeat_frames:
        raise ValueError(f"{perf_id}: downbeat sequence is empty")
    if num_beams != 1:
        raise ValueError("dynamic downbeat inference currently requires greedy decoding")
    if any(
        right <= left
        for left, right in zip(downbeat_frames, downbeat_frames[1:])
    ):
        raise ValueError(f"{perf_id}: downbeat positions must be strictly increasing")

    scopes = output_scopes or [
        {
            "start": start,
            "chunk": index,
            "artifact_name": f"{perf_id}.{index}",
        }
        for index, start in enumerate(
            range(0, len(downbeat_frames) - max_bars_per_chunk + 1,
                  max_bars_per_chunk)
        )
    ]
    if not scopes:
        return
    for scope_ordinal, scope in enumerate(scopes):
        output_start = int(scope["start"])
        output_end = output_start + max_bars_per_chunk
        output_chunk = int(scope.get("chunk", scope_ordinal))
        artifact_name = str(
            scope.get("artifact_name", f"{perf_id}.{output_chunk}")
        )
        if output_end > len(downbeat_frames):
            raise ValueError(
                f"{artifact_name}: requested scope ends at {output_end}, "
                f"but only {len(downbeat_frames)} downbeats exist"
            )

        bar_tokens: Dict[int, List[int]] = {}
        bar_repairs: Dict[int, dict] = {}
        attempts: List[dict] = []
        cursor = output_start
        attempt_index = 0
        scope_failed = False

        while cursor < output_end:
            requested_bars = min(
                output_end - cursor,
                _bars_available_from_downbeats(
                    downbeat_frames,
                    cursor,
                    trust_frames,
                    max_bars_per_chunk,
                ),
            )
            mel_chunk = _slice_with_margin(
                mel,
                downbeat_frames[cursor],
                chunk_frames,
            )
            mel_input = (
                mel_chunk if mel_chunk.dim() == 3
                else mel_chunk.unsqueeze(0)
            )

            token_ids, log_probs, decode_stats = yield _DecodeRequest(
                mel=mel_input,
                stop_after_n_bars=requested_bars,
                forced_schema=(
                    schema_timeline[cursor:] if schema_timeline is not None
                    else None
                ),
                memory_moments_override=(
                    coordinate_intervention.moments(
                        manifest_item, downbeat_frames[cursor], chunk_frames, device,
                    ) if coordinate_intervention is not None else None
                ),
            )
            boundary_bars = int(decode_stats['boundary_completed_bars'])
            model_entry = {
                "record_type": "decode_attempt",
                "piece": perf_id,
                "chunk": output_chunk,
                "artifact_name": artifact_name,
                "scope_measures": [output_start, output_end - 1],
                "attempt": attempt_index,
                "reset_downbeat_bar": cursor,
                "requested_measures": [
                    cursor,
                    cursor + requested_bars - 1,
                ],
                "requested_bars": requested_bars,
                "boundary_completed_bars": boundary_bars,
                "stop_reason": decode_stats['stop_reason'],
                "selection": "grammar_constrained_greedy",
                "token_ids": list(token_ids),
                "tokens": [
                    tokenizer.id_to_token.get(token_id, "")
                    for token_id in token_ids
                ],
                "log_probs": log_probs,
                "masked_argmax": decode_stats['masked_argmax'],
                "seam_repairs": decode_stats['seam_repairs'],
                "steps": decode_stats['steps'],
            }
            attempt_index += 1

            accepted_bars = 0
            accepted_ids: List[int] = []
            attempt_log = None
            attempt_repairs: List[dict] = []

            if boundary_bars <= 0:
                fallback_entry = dict(model_entry)
                fallback_entry["measures"] = [cursor, cursor]
                fallback_entry["completion_policy"] = "tail_rest_fallback"
                fallback_entry["tail_rest_completion_bars"] = [0]
                fallback_entry["skip_trailing_bar_dedup"] = True
                fallback_kern, _, fallback_log, fallback_fills = (
                    _postprocess_bar_decode_entry(
                        fallback_entry,
                        tokenizer,
                        constraints_cpu,
                    )
                )
                fallback_blocks = (
                    _parse_fragment_text(fallback_kern)[2]
                    if fallback_kern else []
                )
                if fallback_kern and fallback_fills and fallback_blocks:
                    accepted_bars = 1
                    accepted_ids = _accepted_prefix_token_ids(
                        token_ids, tokenizer, 1,
                    )
                    attempt_repairs = fallback_log[
                        "tail_rest_completions"
                    ]
                    model_entry.update({
                        "measures": [cursor, cursor],
                        "accepted_bars": 1,
                        "completion_policy": "tail_rest_fallback",
                        "fallback_trigger": "decoder_completed_zero_bars",
                        "tail_rest_completion_bars": [0],
                        "tail_rest_completions": attempt_repairs,
                    })
                    fallback_log.update({
                        "record_type": "decode_attempt",
                        "artifact_name": artifact_name,
                        "scope_measures": [output_start, output_end - 1],
                        "reset_downbeat_bar": cursor,
                        "requested_measures": [
                            cursor, cursor + requested_bars - 1,
                        ],
                        "requested_bars": requested_bars,
                        "boundary_completed_bars": 0,
                        "accepted_bars": 1,
                        "attempt": model_entry["attempt"],
                        "stop_reason": decode_stats['stop_reason'],
                        "fallback_trigger": "decoder_completed_zero_bars",
                    })
                    attempt_log = fallback_log
                else:
                    fallback_error = fallback_log.get(
                        "fail",
                        "partial bar is not eligible for tail-rest completion",
                    )
                    model_entry.update({
                        "measures": [],
                        "accepted_bars": 0,
                        "fallback_attempted": True,
                        "tail_rest_fallback_error": fallback_error,
                    })
                    attempt_log = {
                        "record_type": "decode_attempt",
                        "piece": perf_id,
                        "chunk": output_chunk,
                        "artifact_name": artifact_name,
                        "scope_measures": [output_start, output_end - 1],
                        "attempt": model_entry["attempt"],
                        "reset_downbeat_bar": cursor,
                        "requested_measures": [
                            cursor, cursor + requested_bars - 1,
                        ],
                        "requested_bars": requested_bars,
                        "boundary_completed_bars": 0,
                        "accepted_bars": 0,
                        "stop_reason": decode_stats['stop_reason'],
                        "fallback_attempted": True,
                        "tail_rest_fallback_error": fallback_error,
                        "fail": "decoder completed zero bars",
                    }
            else:
                if boundary_bars > requested_bars:
                    raise RuntimeError(
                        f"{artifact_name} attempt {model_entry['attempt']}: "
                        f"completed {boundary_bars} bars with a limit of "
                        f"{requested_bars}"
                    )
                rejected_prefixes = []
                for candidate_bars in range(boundary_bars, 0, -1):
                    candidate_entry = dict(model_entry)
                    candidate_entry["measures"] = [
                        cursor,
                        cursor + candidate_bars - 1,
                    ]
                    candidate_entry["skip_trailing_bar_dedup"] = True
                    candidate_kern, _, candidate_log, _ = (
                        _postprocess_bar_decode_entry(
                            candidate_entry,
                            tokenizer,
                            constraints_cpu,
                        )
                    )
                    parsed_blocks = (
                        _parse_fragment_text(candidate_kern)[2]
                        if candidate_kern else []
                    )
                    try:
                        candidate_ids = _accepted_prefix_token_ids(
                            token_ids,
                            tokenizer,
                            candidate_bars,
                        )
                    except ValueError as error:
                        rejected_prefixes.append({
                            "bars": candidate_bars,
                            "reason": str(error),
                        })
                        continue
                    if candidate_kern and len(parsed_blocks) >= candidate_bars:
                        accepted_bars = candidate_bars
                        accepted_ids = candidate_ids
                        attempt_log = candidate_log
                        break
                    rejected_prefixes.append({
                        "bars": candidate_bars,
                        "reason": candidate_log.get(
                            "fail",
                            f"reconstruction returned {len(parsed_blocks)} bars",
                        ),
                    })
                if accepted_bars:
                    model_entry.update({
                        "measures": [
                            cursor,
                            cursor + accepted_bars - 1,
                        ],
                        "accepted_bars": accepted_bars,
                    })
                    attempt_log.update({
                        "record_type": "decode_attempt",
                        "artifact_name": artifact_name,
                        "scope_measures": [output_start, output_end - 1],
                        "reset_downbeat_bar": cursor,
                        "requested_measures": [
                            cursor, cursor + requested_bars - 1,
                        ],
                        "requested_bars": requested_bars,
                        "boundary_completed_bars": boundary_bars,
                        "accepted_bars": accepted_bars,
                        "attempt": model_entry["attempt"],
                        "stop_reason": decode_stats['stop_reason'],
                    })
                    if rejected_prefixes:
                        attempt_log["rejected_bar_prefixes"] = rejected_prefixes
                else:
                    model_entry.update({
                        "measures": [],
                        "accepted_bars": 0,
                    })
                    attempt_log = {
                        "record_type": "decode_attempt",
                        "piece": perf_id,
                        "chunk": output_chunk,
                        "artifact_name": artifact_name,
                        "scope_measures": [output_start, output_end - 1],
                        "attempt": model_entry["attempt"],
                        "reset_downbeat_bar": cursor,
                        "requested_measures": [
                            cursor, cursor + requested_bars - 1,
                        ],
                        "requested_bars": requested_bars,
                        "boundary_completed_bars": boundary_bars,
                        "accepted_bars": 0,
                        "stop_reason": decode_stats['stop_reason'],
                        "rejected_bar_prefixes": rejected_prefixes,
                        "fail": "no complete reconstructable bar prefix",
                    }

            attempts.append({
                "model_decode_entry": model_entry,
                "log_entry": attempt_log,
            })
            if accepted_bars == 0 and cursor > 0:
                retry_start = cursor - 1
                retry_requested = min(
                    max_bars_per_chunk,
                    output_end - retry_start,
                )
                retry_mel = _slice_with_margin(
                    mel,
                    downbeat_frames[retry_start],
                    chunk_frames,
                )
                retry_input = (
                    retry_mel if retry_mel.dim() == 3
                    else retry_mel.unsqueeze(0)
                )
                retry_token_ids, retry_log_probs, decode_stats = yield _DecodeRequest(
                    mel=retry_input,
                    stop_after_n_bars=retry_requested,
                    forced_schema=(
                        schema_timeline[retry_start:]
                        if schema_timeline is not None else None
                    ),
                    memory_moments_override=(
                        coordinate_intervention.moments(
                            manifest_item, downbeat_frames[retry_start],
                            chunk_frames, device,
                        ) if coordinate_intervention is not None else None
                    ),
                )
                retry_boundary = int(
                    decode_stats['boundary_completed_bars']
                )
                retry_entry = {
                    "record_type": "decode_attempt",
                    "piece": perf_id,
                    "chunk": output_chunk,
                    "artifact_name": artifact_name,
                    "scope_measures": [output_start, output_end - 1],
                    "attempt": attempt_index,
                    "attempt_kind": "previous_downbeat_retry",
                    "fallback_trigger": "first_bar_unreconstructable",
                    "reset_downbeat_bar": retry_start,
                    "target_reset_bar": cursor,
                    "discarded_prefix_bars": 1,
                    "requested_measures": [
                        retry_start,
                        retry_start + retry_requested - 1,
                    ],
                    "requested_bars": retry_requested,
                    "boundary_completed_bars": retry_boundary,
                    "stop_reason": decode_stats['stop_reason'],
                    "selection": "grammar_constrained_greedy",
                    "token_ids": list(retry_token_ids),
                    "tokens": [
                        tokenizer.id_to_token.get(token_id, "")
                        for token_id in retry_token_ids
                    ],
                    "log_probs": retry_log_probs,
                    "masked_argmax": decode_stats['masked_argmax'],
                    "seam_repairs": decode_stats['seam_repairs'],
                    "steps": decode_stats['steps'],
                }
                attempt_index += 1
                retry_raw_bars = 0
                retry_raw_ids: List[int] = []
                retry_post_log = None
                retry_rejected = []
                if retry_boundary >= 2:
                    for candidate_bars in range(retry_boundary, 1, -1):
                        candidate_entry = dict(retry_entry)
                        candidate_entry["measures"] = [
                            retry_start,
                            retry_start + candidate_bars - 1,
                        ]
                        candidate_entry["skip_trailing_bar_dedup"] = True
                        candidate_kern, _, candidate_log, _ = (
                            _postprocess_bar_decode_entry(
                                candidate_entry,
                                tokenizer,
                                constraints_cpu,
                            )
                        )
                        parsed_blocks = (
                            _parse_fragment_text(candidate_kern)[2]
                            if candidate_kern else []
                        )
                        try:
                            candidate_ids = _accepted_prefix_token_ids(
                                retry_token_ids,
                                tokenizer,
                                candidate_bars,
                            )
                        except ValueError as error:
                            retry_rejected.append({
                                "bars": candidate_bars,
                                "reason": str(error),
                            })
                            continue
                        if (candidate_kern
                                and len(parsed_blocks) >= candidate_bars):
                            retry_raw_bars = candidate_bars
                            retry_raw_ids = candidate_ids
                            retry_post_log = candidate_log
                            break
                        retry_rejected.append({
                            "bars": candidate_bars,
                            "reason": candidate_log.get(
                                "fail",
                                "retry prefix did not reconstruct",
                            ),
                        })

                if retry_raw_bars >= 2:
                    retry_blocks = _bar_token_blocks(
                        retry_raw_ids,
                        tokenizer,
                    )
                    target_blocks = retry_blocks[1:]
                    accepted_bars = min(
                        len(target_blocks),
                        output_end - cursor,
                    )
                    accepted_ids = [
                        token
                        for block in target_blocks[:accepted_bars]
                        for token in block
                    ]
                    retry_entry.update({
                        "measures": [
                            retry_start,
                            retry_start + retry_raw_bars - 1,
                        ],
                        "accepted_bars": retry_raw_bars,
                        "target_accepted_bars": accepted_bars,
                    })
                    retry_post_log.update({
                        "record_type": "decode_attempt",
                        "artifact_name": artifact_name,
                        "scope_measures": [output_start, output_end - 1],
                        "attempt": retry_entry["attempt"],
                        "attempt_kind": "previous_downbeat_retry",
                        "fallback_trigger": "first_bar_unreconstructable",
                        "reset_downbeat_bar": retry_start,
                        "target_reset_bar": cursor,
                        "discarded_prefix_bars": 1,
                        "requested_measures": [
                            retry_start,
                            retry_start + retry_requested - 1,
                        ],
                        "requested_bars": retry_requested,
                        "boundary_completed_bars": retry_boundary,
                        "accepted_bars": retry_raw_bars,
                        "target_accepted_bars": accepted_bars,
                        "stop_reason": decode_stats['stop_reason'],
                    })
                    if retry_rejected:
                        retry_post_log[
                            "rejected_bar_prefixes"
                        ] = retry_rejected
                    attempt_repairs = []
                    for repair in retry_post_log.get(
                            "tail_rest_completions", []):
                        repair_bar = int(repair["bar_index"])
                        if repair_bar == 0:
                            continue
                        stored = dict(repair)
                        stored["bar_index"] = repair_bar - 1
                        attempt_repairs.append(stored)
                    attempts.append({
                        "model_decode_entry": retry_entry,
                        "log_entry": retry_post_log,
                    })
                    model_entry = retry_entry
                else:
                    retry_entry.update({
                        "measures": [],
                        "accepted_bars": 0,
                        "target_accepted_bars": 0,
                    })
                    retry_post_log = {
                        "record_type": "decode_attempt",
                        "piece": perf_id,
                        "chunk": output_chunk,
                        "artifact_name": artifact_name,
                        "scope_measures": [output_start, output_end - 1],
                        "attempt": retry_entry["attempt"],
                        "attempt_kind": "previous_downbeat_retry",
                        "fallback_trigger": "first_bar_unreconstructable",
                        "reset_downbeat_bar": retry_start,
                        "target_reset_bar": cursor,
                        "discarded_prefix_bars": 1,
                        "requested_measures": [
                            retry_start,
                            retry_start + retry_requested - 1,
                        ],
                        "requested_bars": retry_requested,
                        "boundary_completed_bars": retry_boundary,
                        "accepted_bars": 0,
                        "target_accepted_bars": 0,
                        "stop_reason": decode_stats['stop_reason'],
                        "rejected_bar_prefixes": retry_rejected,
                        "fail": "previous-downbeat retry made no progress",
                    }
                    attempts.append({
                        "model_decode_entry": retry_entry,
                        "log_entry": retry_post_log,
                    })
            if accepted_bars == 0:
                scope_entry = {
                    "record_type": "output_scope",
                    "piece": perf_id,
                    "chunk": output_chunk,
                    "artifact_name": artifact_name,
                    "measures": [output_start, output_end - 1],
                    "status": "failed",
                }
                yield {
                    "piece": perf_id,
                    "chunk": output_chunk,
                    "artifact_name": artifact_name,
                    "measures": scope_entry["measures"],
                    "attempts": attempts,
                    "scope_entry": scope_entry,
                    "pred_kern": "",
                    "failed": True,
                    "had_fills": False,
                }
                scope_failed = True
                break

            blocks = _bar_token_blocks(accepted_ids, tokenizer)
            if len(blocks) != accepted_bars:
                raise RuntimeError(
                    f"{artifact_name} attempt {model_entry['attempt']}: split "
                    f"{len(blocks)} accepted blocks for {accepted_bars} bars"
                )
            for local_index, block in enumerate(blocks):
                bar_tokens[cursor + local_index] = block
            for repair in attempt_repairs:
                global_bar = cursor + int(repair["bar_index"])
                stored = dict(repair)
                stored["bar_index"] = global_bar
                bar_repairs[global_bar] = stored
            cursor += accepted_bars

        if scope_failed:
            continue

        output_token_ids = [
            token
            for bar_index in range(
                output_start, output_start + max_bars_per_chunk
            )
            for token in bar_tokens[bar_index]
        ]
        output_repairs = []
        for bar_index in range(
            output_start, output_start + max_bars_per_chunk
        ):
            repair = bar_repairs.get(bar_index)
            if repair is None:
                continue
            remapped = dict(repair)
            remapped["bar_index"] = bar_index - output_start
            output_repairs.append(remapped)
        output_entry = {
            "piece": perf_id,
            "chunk": output_chunk,
            "measures": [
                output_start,
                output_start + max_bars_per_chunk - 1,
            ],
            "token_ids": output_token_ids,
            "skip_trailing_bar_dedup": True,
        }
        if output_repairs:
            output_entry.update({
                "completion_policy": "tail_rest_fallback",
                "tail_rest_completion_bars": sorted({
                    int(repair["bar_index"])
                    for repair in output_repairs
                }),
                "tail_rest_completions": output_repairs,
            })
        pred_kern, _, output_log, output_fills = (
            _postprocess_bar_decode_entry(
                output_entry,
                tokenizer,
                constraints_cpu,
            )
        )
        parsed_blocks = (
            _parse_fragment_text(pred_kern)[2]
            if pred_kern else []
        )
        scope_entry = {
            "record_type": "output_scope",
            "piece": perf_id,
            "chunk": output_chunk,
            "artifact_name": artifact_name,
            "measures": [
                output_start,
                output_start + max_bars_per_chunk - 1,
            ],
            "status": "ready",
        }
        if not pred_kern or len(parsed_blocks) < max_bars_per_chunk:
            yield {
                "piece": perf_id,
                "chunk": output_chunk,
                "artifact_name": artifact_name,
                "measures": scope_entry["measures"],
                "attempts": attempts,
                "scope_entry": scope_entry,
                "output_log_entry": output_log,
                "pred_kern": "",
                "failed": True,
                "had_fills": bool(output_fills),
            }
            return
        output_log.update({
            "record_type": "five_bar_output",
            "artifact_name": artifact_name,
            "source_attempts": [
                attempt["model_decode_entry"]["attempt"]
                for attempt in attempts
                if attempt["model_decode_entry"].get("measures")
            ],
        })
        yield {
            "piece": perf_id,
            "chunk": output_chunk,
            "artifact_name": artifact_name,
            "measures": scope_entry["measures"],
            "attempts": attempts,
            "scope_entry": scope_entry,
            "output_log_entry": output_log,
            "pred_kern": pred_kern,
            "failed": False,
            "had_fills": bool(output_fills),
        }
def _retain_completed_decode_entries(
    path: Path, completed_ids: Set[str], requested_ids: Set[str],
) -> None:
    """Drop partial rows being retried, preserving recordings outside this run."""
    if not path.exists():
        return

    kept_lines: List[str] = []
    seen_entries: Set[Tuple[str, str, int, int]] = set()
    with open(path) as existing_log:
        for line in existing_log:
            try:
                entry = json.loads(line)
                if entry["piece"] not in requested_ids:
                    kept_lines.append(line)
                    continue
                key = (
                    entry["piece"],
                    str(entry.get("record_type", "attempt")),
                    int(entry["chunk"]),
                    int(entry.get("attempt", -1)),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if entry["piece"] in completed_ids and key not in seen_entries:
                kept_lines.append(json.dumps(entry) + "\n")
                seen_entries.add(key)

    cleaned_path = path.with_suffix(path.suffix + ".tmp")
    with open(cleaned_path, "w") as cleaned_log:
        cleaned_log.writelines(kept_lines)
    os.replace(cleaned_path, path)


def _postprocess_bar_decode_entry(
    entry: dict,
    tokenizer: KernTokenizer,
    constraints_cpu: dict,
) -> Tuple[str, int, dict, bool]:
    """Turn one preserved model decode into its auditable score fragment."""
    try:
        perf_id = str(entry["piece"])
        chunk_index = int(entry["chunk"])
        measure_start, measure_end = (int(value) for value in entry["measures"])
        decoded_token_ids = [int(value) for value in entry["token_ids"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed model decode entry: {error}") from error
    if measure_end < measure_start:
        raise ValueError(
            f"{perf_id} chunk {chunk_index}: invalid measure span "
            f"{measure_start}:{measure_end}"
        )

    bars_to_keep = measure_end - measure_start + 1
    bar_id = tokenizer.vocab["<bar>"]
    token_ids = truncate_token_ids_to_n_bars(
        decoded_token_ids, bar_id, bars_to_keep,
    )
    n_after_truncate = len(token_ids)
    if not entry.get("skip_trailing_bar_dedup", False):
        token_ids = dedup_trailing_bars(token_ids, bar_id)
    n_after_dedup = len(token_ids)

    # The replay describes the topology of the kept output, not every
    # transient prefix visited during decoding.
    DECODE_STATS["max_voice_width"] = 1
    tail_rest_completion_bars = {
        int(index)
        for index in entry.get("tail_rest_completion_bars", [])
    }
    allow_tail_rest_fallback = bool(tail_rest_completion_bars) or (
        entry.get("completion_policy") == "tail_rest_fallback"
    )
    if allow_tail_rest_fallback and not tail_rest_completion_bars:
        tail_rest_completion_bars.add(bars_to_keep - 1)
    tail_rest_completions: List[dict] = []

    log_entry = {
        "piece": perf_id,
        "chunk": chunk_index,
        "measures": [measure_start, measure_end],
        "masked_argmax": int(entry.get("masked_argmax", 0)),
        "seam_repairs": int(entry.get("seam_repairs", 0)),
        "steps": int(entry.get("steps", len(decoded_token_ids))),
        "decoded_tokens": len(decoded_token_ids),
        "tokens_after_truncate": n_after_truncate,
        "tokens_after_dedup": n_after_dedup,
        "tokens_after_repair": len(token_ids),
        "max_voice_width": DECODE_STATS["max_voice_width"],
        "completion_policy": (
            "tail_rest_fallback" if allow_tail_rest_fallback else "strict"
        ),
        "n_fills": 0,
    }

    release_normalizations: List[dict] = []
    tie_normalizations: List[dict] = []
    readback_mismatches: List[dict] = []
    try:
        pred_kern = reconstruct_kern(
            token_ids,
            tokenizer,
            release_normalizations=release_normalizations,
            tie_normalizations=tie_normalizations,
            readback_mismatches=readback_mismatches,
            tail_rest_completions=(
                tail_rest_completions
                if allow_tail_rest_fallback else None
            ),
            tail_rest_completion_bars=(
                tail_rest_completion_bars
                if allow_tail_rest_fallback else None
            ),
        )
    except ValueError as error:
        log_entry["fail"] = str(error)[:200]
        pred_kern = ""
    for normalization in release_normalizations:
        normalization["measure_index"] = (
            measure_start + normalization["bar_index"]
        )
    for normalization in tie_normalizations:
        normalization["measure_index"] = (
            measure_start + normalization["bar_index"]
        )
        for landing in normalization.get("candidate_landings", []):
            landing["measure_index"] = (
                measure_start + landing["bar_index"]
            )
    for mismatch in readback_mismatches:
        mismatch["measure_index"] = (
            measure_start + mismatch["bar_index"]
        )
    log_entry["n_release_normalizations"] = len(release_normalizations)
    log_entry["reconstructed_after_release_normalization"] = bool(
        release_normalizations and pred_kern
    )
    if release_normalizations:
        log_entry["release_normalizations"] = release_normalizations
    log_entry["n_tie_normalizations"] = len(tie_normalizations)
    log_entry["n_stripped_tie_markers"] = sum(
        int(normalization["stripped_markers"])
        for normalization in tie_normalizations
    )
    log_entry["reconstructed_after_tie_normalization"] = bool(
        tie_normalizations and pred_kern
    )
    if tie_normalizations:
        log_entry["tie_normalizations"] = tie_normalizations
    log_entry["n_readback_mismatches"] = len(readback_mismatches)
    if readback_mismatches:
        log_entry["readback_mismatches"] = readback_mismatches
    expected_tail_rest_completions = entry.get("tail_rest_completions")
    if (
        expected_tail_rest_completions is not None
        and expected_tail_rest_completions != tail_rest_completions
    ):
        raise ValueError(
            f"{perf_id} chunk {chunk_index}: replay tail-rest completion "
            "does not match the recorded repair"
        )
    if tail_rest_completions:
        log_entry["tail_rest_completions"] = tail_rest_completions
        log_entry["fills"] = tail_rest_completions
        log_entry["n_fills"] = len(tail_rest_completions)
    return (
        pred_kern,
        bars_to_keep,
        log_entry,
        bool(tail_rest_completions),
    )


def _iter_model_decode_entries(path: Path) -> Iterator[dict]:
    """Stream preserved model decodes without loading the full log in RAM."""
    with open(path) as source:
        for line_number, line in enumerate(source, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield entry


def run_replay_mode(args) -> None:
    """Rebuild direct five-bar artifacts from immutable decode attempts."""
    source_path = Path(args.replay_model_decode)
    if not source_path.is_file():
        raise FileNotFoundError(f"Model decode log not found: {source_path}")
    if args.output_dir is None:
        raise ValueError("Replay requires explicit --output-dir")

    kern_dir = Path(args.output_dir)
    if kern_dir.exists() and any(kern_dir.iterdir()):
        raise FileExistsError(
            f"Replay output must be new or empty: {kern_dir}"
        )
    kern_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = KernTokenizer()
    constraints_cpu = _build_decode_constraints(
        tokenizer, torch.device("cpu"),
    )
    attempts_by_scope: Dict[str, List[dict]] = {}
    scopes: List[dict] = []
    for entry in _iter_model_decode_entries(source_path):
        record_type = entry.get("record_type", "decode_attempt")
        if record_type == "decode_attempt":
            artifact_name = str(entry.get("artifact_name", ""))
            if not artifact_name:
                raise ValueError("decode attempt lacks artifact_name")
            attempts_by_scope.setdefault(artifact_name, []).append(entry)
        elif record_type == "output_scope":
            scopes.append(entry)
        else:
            raise ValueError(f"unknown model-decode record type: {record_type}")
    if not scopes:
        raise ValueError("model decode log contains no five-bar output scopes")

    decode_log_path = kern_dir / "_decode_log.jsonl"
    provenance_path = kern_dir / "_replay.json"
    n_filled = n_mismatch_chunks = n_mismatch_bars = 0
    n_acoustic = n_parse = 0
    n_ready = n_failed = total_attempts = 0
    with open(decode_log_path, "w") as decode_log:
        for scope in tqdm(scopes, desc="Replay five-bar outputs"):
            perf_id = str(scope["piece"])
            chunk_index = int(scope["chunk"])
            artifact_name = str(scope["artifact_name"])
            first, last = (int(value) for value in scope["measures"])
            if last - first + 1 != 5:
                raise ValueError(
                    f"{artifact_name}: output scope [{first}, {last}] "
                    "is not five bars"
                )
            attempts = sorted(
                attempts_by_scope.get(artifact_name, []),
                key=lambda entry: int(entry["attempt"]),
            )
            total_attempts += len(attempts)
            if scope.get("status") == "failed":
                n_failed += 1
                failure = attempts[-1] if attempts else {}
                decode_log.write(json.dumps({
                    "record_type": "five_bar_output",
                    "piece": perf_id,
                    "chunk": chunk_index,
                    "artifact_name": artifact_name,
                    "measures": [first, last],
                    "fail": failure.get(
                        "tail_rest_fallback_error",
                        "scope reconstruction failed",
                    ),
                }) + "\n")
                continue
            if not attempts:
                raise ValueError(f"{artifact_name}: no decode attempts")
            bar_tokens: Dict[int, List[int]] = {}
            repairs: Dict[int, dict] = {}
            for entry in attempts:
                accepted = int(entry.get("accepted_bars", 0))
                if accepted <= 0:
                    continue
                attempt_first, attempt_last = (
                    int(value) for value in entry["measures"]
                )
                if attempt_last - attempt_first + 1 != accepted:
                    raise ValueError(
                        f"{artifact_name} attempt {entry['attempt']}: "
                        f"measure span [{attempt_first}, {attempt_last}] "
                        f"does not hold {accepted} bars"
                    )
                accepted_ids = _accepted_prefix_token_ids(
                    [int(value) for value in entry["token_ids"]],
                    tokenizer,
                    accepted,
                )
                blocks = _bar_token_blocks(accepted_ids, tokenizer)
                if len(blocks) != accepted:
                    raise ValueError(
                        f"{artifact_name} attempt {entry['attempt']}: split "
                        f"{len(blocks)} blocks for {accepted} bars"
                    )
                discarded = int(entry.get("discarded_prefix_bars", 0))
                for local_index, block in enumerate(
                        blocks[discarded:], start=discarded):
                    bar_index = attempt_first + local_index
                    if bar_index in bar_tokens:
                        raise ValueError(
                            f"{artifact_name}: duplicate decoded bar "
                            f"{bar_index}"
                        )
                    bar_tokens[bar_index] = block
                for repair in entry.get("tail_rest_completions", []):
                    repair_index = int(repair["bar_index"])
                    if repair_index < discarded:
                        continue
                    bar_index = attempt_first + repair_index
                    stored = dict(repair)
                    stored["bar_index"] = bar_index
                    repairs[bar_index] = stored
            try:
                token_ids = [
                    token
                    for bar_index in range(first, last + 1)
                    for token in bar_tokens[bar_index]
                ]
            except KeyError as error:
                raise ValueError(
                    f"{artifact_name}: missing decoded bar {error.args[0]}"
                ) from None
            output_repairs = []
            for bar_index in range(first, last + 1):
                repair = repairs.get(bar_index)
                if repair is None:
                    continue
                remapped = dict(repair)
                remapped["bar_index"] = bar_index - first
                output_repairs.append(remapped)
            output_entry = {
                "piece": perf_id,
                "chunk": chunk_index,
                "measures": [first, last],
                "token_ids": token_ids,
                "skip_trailing_bar_dedup": True,
            }
            if output_repairs:
                output_entry.update({
                    "completion_policy": "tail_rest_fallback",
                    "tail_rest_completion_bars": sorted({
                        int(repair["bar_index"])
                        for repair in output_repairs
                    }),
                    "tail_rest_completions": output_repairs,
                })
            pred_kern, _, log_entry, had_fills = (
                _postprocess_bar_decode_entry(
                    output_entry,
                    tokenizer,
                    constraints_cpu,
                )
            )
            if not pred_kern:
                raise ValueError(f"{artifact_name}: replay reconstruction failed")
            log_entry.update({
                "record_type": "five_bar_output",
                "artifact_name": artifact_name,
            })
            decode_log.write(json.dumps(log_entry) + "\n")
            (kern_dir / f"{artifact_name}.krn").write_text(pred_kern)
            n_ready += 1
            n_filled += bool(had_fills)
            mismatches = log_entry.get("readback_mismatches", [])
            if mismatches:
                n_mismatch_chunks += 1
                n_mismatch_bars += len(mismatches)
                n_acoustic += any(
                    item["severity"] == "acoustic_change"
                    for item in mismatches
                )
                n_parse += any(
                    item["severity"] == "parse_error"
                    for item in mismatches
                )

    provenance = {
        "model_decode": str(source_path.resolve()),
        "decode_attempts": total_attempts,
        "five_bar_outputs": n_ready,
        "failed_scopes": n_failed,
        "tail_rest_fallback_chunks": n_filled,
        "readback_mismatch_chunks": n_mismatch_chunks,
        "readback_mismatch_bars": n_mismatch_bars,
        "readback_acoustic_change_chunks": n_acoustic,
        "readback_parse_error_chunks": n_parse,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info(
        "Replay done — %d five-bar kern from %d attempts → %s",
        n_ready,
        total_attempts,
        kern_dir,
    )
    logger.info("              chunks with rest fills: %d", n_filled)
    logger.info(
        "              readback diagnostics chunks/bars/acoustic/parse: "
        "%d/%d/%d/%d",
        n_mismatch_chunks,
        n_mismatch_bars,
        n_acoustic,
        n_parse,
    )
    logger.info("              model decode kept read-only: %s", source_path)
    logger.info("              replay provenance → %s", provenance_path)


def run_bar_mode(args) -> None:
    """Downbeat-guided inference with direct five-bar score outputs.

    Sliding window: feed chunk_frames (1280) to encoder, trust only the first
    trust_frames (640). Dynamic attempts fill one fixed five-bar token scope;
    only that complete scope is reconstructed and written.
    """
    kern_dir = Path(args.output_dir)
    kern_dir.mkdir(parents=True, exist_ok=True)

    schema_mode_path = kern_dir / "_schema_consensus_mode.json"
    schema_mode = {
        "schema_consensus": "off",
        "coordinate_intervention": getattr(args, "coordinate_intervention", "none"),
    }
    if schema_mode_path.exists():
        if json.loads(schema_mode_path.read_text()) != schema_mode:
            raise ValueError("Inference settings changed; use a fresh output directory")
    else:
        # Completed inventories alone do not identify how the schema was generated.
        if any(kern_dir.glob("*.krn")) or (kern_dir / "_model_decode.jsonl").exists():
            raise ValueError("Existing predictions lack inference settings; use a fresh output directory")
        schema_mode_path.write_text(json.dumps(schema_mode) + "\n")

    model, raw_cfg = load_model(args.checkpoint, args.config, args.device)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = KernTokenizer()

    chunk_cfg = raw_cfg.get("data", {}).get("chunking", {})
    chunk_frames = chunk_cfg.get("chunk_frames", 1280)
    trust_frames = chunk_cfg.get("overlap_frames", 640)

    with open(args.manifest) as f:
        manifest = json.load(f)
    coordinate_intervention = None
    if getattr(args, "coordinate_intervention", "none") != "none":
        coordinate_intervention = CoordinateIntervention(
            args.coordinate_intervention, model, manifest,
            beatthis_dir=getattr(args, "beatthis_dir", None),
        )
    metadata = {}
    if args.metadata:
        with open(args.metadata) as f:
            metadata = index_augmentation_metadata_by_render(json.load(f))
    grounding_by_recording: Dict[str, List[dict]] = {}
    if args.grounding:
        with open(args.grounding) as grounding_file:
            for line_number, line in enumerate(grounding_file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                recording_id = str(row.get("recording_id", ""))
                if not recording_id or not row.get("chunk_id"):
                    raise ValueError(
                        f"{args.grounding}:{line_number}: incomplete grounding row"
                    )
                start = int(row["start_bar_index"])
                end = int(row["end_bar_index_exclusive"])
                if end - start != args.n_bars:
                    raise ValueError(
                        f"{args.grounding}:{line_number}: expected "
                        f"{args.n_bars} bars, found [{start}, {end})"
                    )
                grounding_by_recording.setdefault(recording_id, []).append(row)

    manifest_dir = Path(args.manifest_dir)
    max_bars_per_chunk = args.n_bars
    if max_bars_per_chunk != 5:
        raise ValueError("bar mode emits fixed five-bar scopes")
    max_len = args.max_len

    start_idx = max(0, getattr(args, 'start_idx', 0) or 0)
    if start_idx:
        manifest = manifest[start_idx:]
    if args.max_samples:
        manifest = manifest[:args.max_samples]

    kern_ok = total_outputs = total_attempts = 0
    n_filled_chunks = n_failed_chunks = 0
    n_readback_mismatch_chunks = 0
    n_readback_mismatch_bars = 0
    n_readback_acoustic_chunks = 0
    n_readback_parse_error_chunks = 0
    n_resumed = 0
    constraints_cpu = _build_decode_constraints(tokenizer, torch.device('cpu'))
    decode_log_path = kern_dir / "_decode_log.jsonl"
    model_decode_path = kern_dir / "_model_decode.jsonl"

    def output_scopes_for_item(item: dict, n_measures: int) -> List[dict]:
        if not grounding_by_recording:
            return [
                {
                    "start": start,
                    "chunk": index,
                    "artifact_name": f"{item['id']}.{index}",
                }
                for index, start in enumerate(
                    range(0, n_measures - max_bars_per_chunk + 1,
                          max_bars_per_chunk)
                )
            ]
        recording_id = (
            f"{item.get('piece_id', '')}#{item.get('performance_id', '')}"
        )
        rows = grounding_by_recording.get(recording_id, [])
        return [
            {
                "start": int(row["start_bar_index"]),
                "chunk": int(row.get("position_index", index)),
                "artifact_name": str(row["chunk_id"]),
            }
            for index, row in enumerate(rows)
        ]

    completed_ids: Set[str] = set()
    if args.resume:
        for item in manifest:
            perf_id = item["id"]
            coordinate_source = metadata.get(perf_id, item)
            n_measures = len(coordinate_source.get("audio_measures", []))
            scopes = output_scopes_for_item(item, n_measures)
            if scopes and all(
                (kern_dir / f"{scope['artifact_name']}.krn").is_file()
                for scope in scopes
            ):
                completed_ids.add(perf_id)
        requested_ids = {item["id"] for item in manifest}
        _retain_completed_decode_entries(decode_log_path, completed_ids, requested_ids)
        _retain_completed_decode_entries(model_decode_path, completed_ids, requested_ids)
        logger.info(
            "Resume enabled: %d completed performances will be skipped",
            len(completed_ids),
        )
    if args.num_beams != 1:
        raise ValueError("dynamic downbeat inference currently requires --num-beams 1")

    decode_log = open(decode_log_path, "a")
    model_decode_log = open(model_decode_path, "a")

    for item in tqdm(manifest, desc="Inference (bar)"):
        perf_id = item["id"]
        if perf_id in completed_ids:
            n_resumed += 1
            continue
        coordinate_source = metadata.get(perf_id, item)
        audio_measures: List[dict] = coordinate_source.get("audio_measures", [])
        if not audio_measures:
            raise ValueError(
                f"No audio_measures for {perf_id} in metadata or manifest"
            )
        n_measures = len(audio_measures)
        output_scopes = output_scopes_for_item(item, n_measures)

        mel_path_abs = manifest_dir / item["mel_path"]
        if not mel_path_abs.exists():
            logger.warning(f"Mel missing: {mel_path_abs}, skipping")
            continue

        if str(mel_path_abs).endswith(".npy"):
            import numpy as np
            mel = torch.from_numpy(np.load(str(mel_path_abs), mmap_mode="r")).float()
        else:
            mel = torch.load(str(mel_path_abs), map_location="cpu", weights_only=True)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        downbeat_frames = [
            int(float(measure["start_sec"]) * MEL_FPS)
            for measure in audio_measures
        ]
        piece_outputs = piece_attempts = piece_failed_scopes = 0
        for result in iter_kern_piece_from_downbeats(
            perf_id=perf_id,
            model=model,
            mel=mel,
            downbeat_frames=downbeat_frames,
            tokenizer=tokenizer,
            constraints_cpu=constraints_cpu,
            chunk_frames=chunk_frames,
            trust_frames=trust_frames,
            max_bars_per_chunk=max_bars_per_chunk,
            max_len=max_len,
            num_beams=args.num_beams,
            device=args.device,
            output_scopes=output_scopes,
            coordinate_intervention=coordinate_intervention,
            manifest_item=item,
            inference_batch_size=args.inference_batch_size,
            encoder_batch_size=args.encoder_batch_size,
        ):
            for attempt in result["attempts"]:
                model_decode_log.write(json.dumps(
                    attempt["model_decode_entry"]
                ) + "\n")
                decode_log.write(json.dumps(
                    attempt["log_entry"]
                ) + "\n")
                total_attempts += 1
                piece_attempts += 1
            if result.get("scope_entry") is not None:
                model_decode_log.write(json.dumps(
                    result["scope_entry"]
                ) + "\n")
            model_decode_log.flush()
            decode_log.flush()
            total_outputs += 1

            if result["failed"]:
                n_failed_chunks += 1
                piece_failed_scopes += 1
                logger.warning(
                    "%s: five-bar scope %s could not be reconstructed",
                    perf_id,
                    result["artifact_name"],
                )
                continue
            output_log = result["output_log_entry"]
            decode_log.write(json.dumps(output_log) + "\n")
            decode_log.flush()
            if result.get("had_fills"):
                n_filled_chunks += 1

            readback_mismatches = output_log.get("readback_mismatches", [])
            if readback_mismatches:
                n_readback_mismatch_chunks += 1
                n_readback_mismatch_bars += len(readback_mismatches)
                if any(
                    mismatch["severity"] == "acoustic_change"
                    for mismatch in readback_mismatches
                ):
                    n_readback_acoustic_chunks += 1
                if any(
                    mismatch["severity"] == "parse_error"
                    for mismatch in readback_mismatches
                ):
                    n_readback_parse_error_chunks += 1

            kern_out = kern_dir / f"{result['artifact_name']}.krn"
            with open(kern_out, "w") as f:
                f.write(result["pred_kern"])
            kern_ok += 1
            piece_outputs += 1

        logger.info(
            "%s: %d measures → %d scopes: %d ready, %d missing "
            "(%d decode attempts)",
            perf_id,
            n_measures,
            len(output_scopes),
            piece_outputs,
            piece_failed_scopes,
            piece_attempts,
        )

    decode_log.close()
    model_decode_log.close()
    logger.info(
        "Done — five-bar kern: %d/%d from %d attempts → %s",
        kern_ok,
        total_outputs,
        total_attempts,
        kern_dir,
    )
    logger.info(f"       chunks with rest fills: {n_filled_chunks}, "
                f"unreconstructable: {n_failed_chunks}")
    logger.info(
        "       readback diagnostics chunks/bars/acoustic/parse: "
        "%d/%d/%d/%d",
        n_readback_mismatch_chunks,
        n_readback_mismatch_bars,
        n_readback_acoustic_chunks,
        n_readback_parse_error_chunks,
    )
    if args.resume:
        logger.info(f"       resumed completed performances: {n_resumed}")
    if coordinate_intervention is not None:
        (kern_dir / "_coordinate_intervention.json").write_text(json.dumps(
            {"mode": coordinate_intervention.mode, **coordinate_intervention.stats,
             "tracker_sources": coordinate_intervention.source_metadata},
        ) + "\n")
        logger.info(f"       coordinate intervention: {coordinate_intervention.stats}")
    logger.info(f"       decode log → {decode_log_path}")
    logger.info(f"       model decode → {model_decode_path}")
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        logger.info(
            "       GPU peak allocated/reserved: %.2f/%.2f GiB",
            torch.cuda.max_memory_allocated() / (1024 ** 3),
            torch.cuda.max_memory_reserved() / (1024 ** 3),
        )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for PianoModel")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint")
    mode.add_argument(
        "--replay-model-decode",
        help="Rebuild score artifacts from an existing _model_decode.jsonl",
    )
    parser.add_argument("--config", default="configs/piano_2gpu.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=None)

    parser.add_argument("--metadata", default=None,
                        help="Path to augmentation_metadata.json "
                             "(default: <manifest-dir>/augmentation_metadata.json)")
    parser.add_argument(
        "--grounding",
        default=None,
        help="Optional JSONL of explicit five-bar scopes and artifact names",
    )
    parser.add_argument("--n-bars", type=int, default=5,
                        help="Bars per output scope and max per decode attempt")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Beam search width (1 = greedy)")
    parser.add_argument("--inference-batch-size", type=int, default=4,
                        help="Number of independent greedy chunks decoded together")
    parser.add_argument("--encoder-batch-size", type=int, default=1,
                        help="Encoder micro-batch size used by batched inference")
    parser.add_argument("--resume", action="store_true",
                        help="Skip performances with complete five-bar outputs")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument(
        "--coordinate-intervention", default="none", choices=COORDINATE_INTERVENTIONS,
        help="Replace the audio-side metrical-position posterior at the decoder "
             "cross-attention: oracle (annotated), other-work (another work's "
             "annotation), zero (no contribution), beatthis (tracker events); bar mode only",
    )
    parser.add_argument("--beatthis-dir", default=None,
                        help="Directory of saved tracker beat/downbeat predictions")

    args = parser.parse_args()
    if args.inference_batch_size < 1 or args.encoder_batch_size < 1:
        parser.error("inference and encoder batch sizes must be positive")
    if args.coordinate_intervention == "beatthis" and not args.beatthis_dir:
        parser.error("--coordinate-intervention beatthis needs --beatthis-dir")

    if args.replay_model_decode:
        run_replay_mode(args)
        return

    # Resolve paths from config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.manifest_dir is None:
        args.manifest_dir = config["paths"]["manifest_dir"]
    if args.manifest is None:
        args.manifest = f"{args.manifest_dir}/test_manifest.json"
    if args.metadata is None:
        metadata_candidate = Path(args.manifest_dir) / "augmentation_metadata.json"
        args.metadata = str(metadata_candidate) if metadata_candidate.exists() else None
    if args.output_dir is None:
        args.output_dir = f"{args.manifest_dir}/test_kern_pred"
    if args.max_len is None:
        args.max_len = config.get("model", {}).get("max_seq_len", 2048)

    run_bar_mode(args)


if __name__ == "__main__":
    main()
