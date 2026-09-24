"""
Frozen hFT foundation: input contract, loader, tap.

The mel params and construction dims are fixed by the published checkpoint, not
hyperparameters -- so they live as constants (asserted against parameter.json),
never in the training yaml where tuning them would silently feed the frozen ear
out-of-distribution input.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .hft_model import Decoder_SPEC2MIDI, Encoder_SPEC2MIDI, Model_SPEC2MIDI


@dataclass(frozen=True)
class HFTMelSpec:
    # Values upstream left at torchaudio defaults are pinned explicitly so the
    # contract survives torchaudio default changes. Source: amt.py + config.json.
    sample_rate: int = 16000
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 256
    n_mels: int = 256
    f_min: float = 0.0
    f_max: float = 8000.0          # upstream default None -> sample_rate / 2
    power: float = 2.0
    mel_norm: str = "slaney"       # amt.py passes norm='slaney'; torchaudio default is None
    mel_scale: str = "htk"
    pad_mode: str = "constant"
    log_offset: float = 1e-8
    per_sample_normalize: bool = False  # upstream feeds raw log-mel


@dataclass(frozen=True)
class HFTArch:
    n_margin: int = 32
    n_frame: int = 128
    n_bin: int = 256
    cnn_channel: int = 4
    cnn_kernel: int = 5
    hid_dim: int = 256
    n_layers: int = 3
    n_heads: int = 4
    pf_dim: int = 512
    dropout: float = 0.1
    n_note: int = 88
    n_velocity: int = 128


HFT_MEL = HFTMelSpec()
HFT_ARCH = HFTArch()

# Silence pad value for clip-boundary margins (upstream pads margins with this).
HFT_PAD_VALUE: float = math.log(HFT_MEL.log_offset)


def build_frozen_hft(state_dict_path, device="cuda", arch=HFT_ARCH, parameter_json=None, freeze=True):
    # device must be final at construction: self.device / scale_* are plain
    # attributes pinned here, and .to() does NOT relocate them.
    if parameter_json is not None:
        _assert_arch_matches(arch, parameter_json)

    enc = Encoder_SPEC2MIDI(
        n_margin=arch.n_margin, n_frame=arch.n_frame, n_bin=arch.n_bin,
        cnn_channel=arch.cnn_channel, cnn_kernel=arch.cnn_kernel,
        hid_dim=arch.hid_dim, n_layers=arch.n_layers, n_heads=arch.n_heads,
        pf_dim=arch.pf_dim, dropout=arch.dropout, device=device,
    )
    dec = Decoder_SPEC2MIDI(
        n_frame=arch.n_frame, n_bin=arch.n_bin, n_note=arch.n_note,
        n_velocity=arch.n_velocity, hid_dim=arch.hid_dim, n_layers=arch.n_layers,
        n_heads=arch.n_heads, pf_dim=arch.pf_dim, dropout=arch.dropout, device=device,
    )
    model = Model_SPEC2MIDI(enc, dec).to(device)
    _expected_missing = {
        'encoder_spec2midi._pos_freq_base',
        'decoder_spec2midi._pos_note_base',
        'decoder_spec2midi._pos_frame_base',
    }
    missing, unexpected = model.load_state_dict(
        torch.load(state_dict_path, map_location=device), strict=False,
    )
    surprise = set(missing) - _expected_missing
    if surprise or unexpected:
        raise RuntimeError(
            f"hFT state_dict mismatch: unexpected missing={surprise}, "
            f"unexpected extra={unexpected}"
        )
    model.eval()
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    model = torch.compile(model, mode="reduce-overhead")
    return model


def _assert_arch_matches(arch, parameter_json):
    p = json.loads(Path(parameter_json).read_text(encoding="utf-8"))
    tf, cnn = p["transformer"], p["cnn"]
    checks = {
        "hid_dim": (arch.hid_dim, tf["hid_dim"]),
        "pf_dim": (arch.pf_dim, tf["pf_dim"]),
        "n_heads": (arch.n_heads, tf["encoder"]["n_head"]),
        "enc_n_layers": (arch.n_layers, tf["encoder"]["n_layer"]),
        "dec_n_layers": (arch.n_layers, tf["decoder"]["n_layer"]),
        "cnn_channel": (arch.cnn_channel, cnn["channel"]),
        "cnn_kernel": (arch.cnn_kernel, cnn["kernel"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise ValueError(f"HFT_ARCH diverged from parameter.json: {bad}")


def tap_midi_time(model, input_spec, arch=HFT_ARCH):
    # midi_time is not returned by forward(), so capture it with a hook.
    # input_spec: [B, n_bin, n_margin + n_frame + n_margin]; returns [B, n_frame, n_note, hid].
    feat = {}
    handle = model.decoder_spec2midi.layers_time[-1].register_forward_hook(
        lambda _m, _i, out: feat.__setitem__("mt", out.detach())
    )
    try:
        with torch.no_grad():
            model(input_spec)
    finally:
        handle.remove()
    b = input_spec.shape[0]
    return feat["mt"].reshape(b, arch.n_note, arch.n_frame, arch.hid_dim).permute(0, 2, 1, 3).contiguous()
