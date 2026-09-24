"""Convolution reverb for rendered audio, shared by the ASAP and Syn chains.

Dry renders sound like a close-mic studio capture while the target domain is
a hall, so rendered slots are convolved with a measured room response. Real
recordings never pass through here.
"""

from functools import lru_cache
from pathlib import Path

import hashlib

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

IR_DIR = Path(__file__).resolve().parents[2] / "data" / "ir"

# Great Hall (800 seats), Octagon (wettest), Jack Lyons (350 seats),
# Rymer (driest).
#
# Attribution required by the impulse responses' licences:
#
#   Room impulse responses from the C4DM RIR data set, (c) Centre for Digital
#   Music, Queen Mary University of London, licensed under CC BY-NC-SA 4.0.
#   R. Stewart and M. Sandler, "Database of omnidirectional and B-format room
#   impulse responses," in Proc. IEEE ICASSP, Dallas, TX, 2010, pp. 165-168.
#   (Great Hall, Octagon)
#
#   Room impulse responses from OpenAIR (openairlib.net), AudioLab,
#   University of York, licensed under CC BY 4.0.
#   (Jack Lyons Concert Hall, Arthur Sykes Rymer Auditorium)
#
# The C4DM non-commercial clause covers research use; it has to be revisited
# before any commercial release of models trained on renders using those two.
IR_POOL = (
    "c4dm_greathall",
    "c4dm_octagon",
    "openair_jack_lyons",
    "openair_rymer_auditorium",
)

MIX_RANGE = (0.20, 0.50)

# A piece's render family is four versions. valid renders only version 0 and
# inherits whatever that slot drew, so it is dry a quarter of the time like
# any other slot.
REVERB_SLOTS = 4


@lru_cache(maxsize=None)
def load_ir(name: str, rate: int) -> np.ndarray:
    """Mono impulse response at the render rate, peak-normalized."""
    path = IR_DIR / f"{name}.wav"
    data, ir_rate = sf.read(str(path), always_2d=True)
    ir = data.mean(axis=1).astype(np.float64)
    if ir_rate != rate:
        ir = resample_poly(ir, rate, ir_rate)
    peak = np.abs(ir).max()
    return ir / peak if peak > 0 else ir


def apply_reverb(
    audio: np.ndarray, rate: int, ir_name: str, mix: float
) -> np.ndarray:
    """Mix a convolved copy into the dry signal at an honest energy ratio.

    A raw convolution's wet level sits far below the dry signal, so the wet
    copy is brought to the dry RMS first; only then does `mix` mean the
    fraction of wet energy a listener hears. The convolution tail is kept, so
    the result is longer than the input by the IR's length.
    """
    dry = np.asarray(audio, dtype=np.float64)
    ir = load_ir(ir_name, rate)
    wet = fftconvolve(dry, ir)
    dry_rms = np.sqrt(np.mean(dry ** 2))
    wet_rms = np.sqrt(np.mean(wet[: len(dry)] ** 2))
    if wet_rms > 0 and dry_rms > 0:
        wet = wet * (dry_rms / wet_rms)
    padded = np.zeros_like(wet)
    padded[: len(dry)] = dry
    return (1.0 - mix) * padded + mix * wet


def draw_reverb_plan(stem: str, n_slots: int = REVERB_SLOTS):
    """One room per render slot of a piece, or None for the dry slot.

    Drawn from a stream of its own so that adding reverb does not shift the
    tempo-scaling draws that share the caller's global seed.
    """
    seed = int(hashlib.md5(f"{stem}|reverb".encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_slots)
    plan = [None] * n_slots
    for slot in order[1:]:  # the first slot drawn stays dry
        plan[int(slot)] = {
            "ir": IR_POOL[int(rng.integers(len(IR_POOL)))],
            "mix": float(rng.uniform(*MIX_RANGE)),
        }
    return plan
