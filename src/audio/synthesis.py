# Copyright 2024 Wei Zeng (National University of Singapore)
# Licensed under the Apache License, Version 2.0
# Original source: https://github.com/wei-zeng98/piano-a2s
#
# MIDI processing and audio synthesis utilities for Zeng et al. baseline reproduction.
# This code is used for apple-to-apple comparison only.

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from mido import MidiFile
from midi2audio import FluidSynth

from .reverb import IR_DIR, apply_reverb


AUDIO_TARGET_LUFS = -27.0
# Float samples carry no full-scale ceiling, so the fixed integrated loudness
# is reachable at every crest factor the corpus holds and no take needs a
# limiter or a gain of its own.
AUDIO_SUBTYPE = "FLOAT"


def audio_matches_contract(wav_path: str | Path) -> bool:
    """Whether an existing WAV is a product of the current audio contract.

    Existence alone does not say which contract wrote a file, so a render
    published under an earlier sample container is not reusable and has to be
    rebuilt from its MIDI, which is the performance and never changes here.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        return False
    try:
        return sf.info(str(wav_path)).subtype == AUDIO_SUBTYPE
    except Exception:
        return False


@lru_cache(maxsize=None)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    """Hash immutable file contents while avoiding repeated soundfont reads."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(path: Path) -> str:
    stat = path.stat()
    return _file_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def audio_render_fingerprint(
    midi_path: str | Path,
    soundfont_path: str | Path,
    reverb: dict | None,
) -> str:
    """Identify every input that determines a reusable synthesized WAV."""
    midi_path = Path(midi_path)
    soundfont_path = Path(soundfont_path)
    payload = {
        "midi_sha256": _content_sha256(midi_path),
        "reverb": reverb,
        "soundfont_sha256": _content_sha256(soundfont_path),
    }
    if reverb is not None:
        payload["reverb_ir_sha256"] = _content_sha256(
            IR_DIR / f"{reverb['ir']}.wav")
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MIDIProcess:
    """MIDI post-processing for Zeng et al. baseline.

    Handles:
    - Cutting initial blank
    - Cutting last pedal
    - Random tempo scaling for data augmentation
    """

    def __init__(self, midi_path: str, split: str = "train"):
        self.midi = MidiFile(midi_path)
        assert split in ["train", "valid", "test"]
        self.split = split

    def cut_last_pedal(self):
        for track in self.midi.tracks:
            if (
                track[-2].type == "control_change"
                and track[-2].channel == 0
                and track[-2].control == 64
                and track[-2].value == 0
            ):
                track[-2].time = 0

    def cut_initial_blank(self):
        total_time_before_first_note = 0
        found_first_note = False

        for track in self.midi.tracks:
            time_accumulated = 0
            for msg in track:
                if not found_first_note:
                    time_accumulated += msg.time
                    if (msg.type == "note_on" and msg.velocity > 0) or (
                        msg.type == "control_change" and msg.value > 0
                    ):
                        found_first_note = True
                        total_time_before_first_note = time_accumulated - msg.time
                        msg.time = 0
                else:
                    msg.time -= total_time_before_first_note
                    break

    # Message types whose delta times are scaled for tempo augmentation.
    # set_tempo / time_signature are NOT scaled (tempo map stays fixed;
    # moving the events in tick-space achieves the speed change).
    _SCALABLE_TYPES = frozenset([
        "note_on", "note_off", "control_change", "program_change",
        "marker",  # measure-boundary markers injected by the pipeline
    ])

    def apply_scaling(self, scaling: float):
        """Apply a specific tempo scaling factor (no RNG).

        Args:
            scaling: Tempo scaling factor to apply to MIDI tick times.

        Returns:
            Tuple of (scaling, original_length) for compatibility.
        """
        original_length = self.midi.length
        for track in self.midi.tracks:
            for msg in track:
                if msg.type in self._SCALABLE_TYPES:
                    msg.time = int(msg.time * scaling)
        return scaling, original_length

    def random_scaling(self, range=(0.85, 1.15)):
        """Apply random tempo scaling for data augmentation.

        Note: Removed Zeng's 4-12 second length constraint, which was designed
        for 5-bar chunks. For full-song processing, we apply tempo_range directly.
        """
        original_length = self.midi.length
        lower_bound = range[0]
        upper_bound = range[1]
        if self.split == "test" or self.split == "valid":
            if lower_bound > 1:
                scaling = lower_bound
            elif upper_bound < 1:
                scaling = upper_bound
            else:
                scaling = 1
        elif self.split == "train":
            scaling = np.random.uniform(lower_bound, upper_bound)
        for track in self.midi.tracks:
            for msg in track:
                if msg.type in self._SCALABLE_TYPES:
                    msg.time = int(msg.time * scaling)
        return scaling, original_length

    def save(self, path: str):
        try:
            self.midi.save(path)
        except Exception:
            print(f"Error in saving midi file {path}")

    def process(self, path: str, temp_path: str = "temp/temp.mid",
                tempo_range: tuple = (0.85, 1.15), scaling: float = None):
        """Full processing pipeline.

        Args:
            path: Output path for processed MIDI
            temp_path: Temporary file path for intermediate processing
            tempo_range: Tuple of (min_scale, max_scale) for tempo augmentation
            scaling: If provided, use this exact scaling factor instead of
                drawing from RNG. This avoids RNG double-consumption when
                the caller already drew the value.

        Returns:
            Tuple of (scaling, original_length, success):
            - scaling: The tempo scaling factor applied (or 1.0 if failed)
            - original_length: Original MIDI length in seconds
            - success: True if tempo scaling succeeded, False if failed (negative delta time)
        """
        self.cut_last_pedal()
        # NOTE: cut_initial_blank removed to preserve alignment between
        # Score-derived measure timing and audio. The small silence before
        # the first note is harmless for training.
        # Save to get correct length
        try:
            self.midi.save(temp_path)
            self.midi = MidiFile(temp_path)
            if scaling is not None:
                actual_scaling, original_length = self.apply_scaling(scaling)
            else:
                actual_scaling, original_length = self.random_scaling(range=tempo_range)
            if actual_scaling is not None:
                self.save(path)
            return actual_scaling, original_length, True
        except ValueError as e:
            if "negative" in str(e).lower():
                # MIDI has negative delta time - can't apply tempo scaling
                # Return failure flag so caller can fallback to original MIDI with tempo=1.0
                print(f"[TEMPO SCALING FAILED] {path} - negative delta time, will use original MIDI", flush=True)
                return 1.0, 0.0, False
            raise


def render_one_midi(
    fs: FluidSynth,
    midi_path: str,
    wav_path: str,
    reverb: dict = None,
):
    """Render MIDI to WAV: mono, optional hall, fixed integrated loudness.

    A single gain puts every render on the real-piano input level without
    changing its crest factor.  No compressor or limiter is applied.

    Args:
        fs: FluidSynth object with soundfont loaded
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        reverb: {"ir": name, "mix": float} to place the render in a room,
            or None for the dry slot
    """
    # Neither the raw render nor the rewrite may touch wav_path itself: a kill
    # between them would leave a well-formed but unfinished file that every
    # later run accepts as done.  Work on a sibling, publish with a rename.
    tmp_path = os.path.join(os.path.dirname(wav_path),
                            f".{os.path.basename(wav_path)}")
    try:
        fs.midi_to_audio(midi_path, tmp_path)
        data, rate = sf.read(tmp_path)
        if np.ndim(data) > 1:
            data = np.mean(data, axis=1)  # Convert to mono

        if reverb is not None:
            data = apply_reverb(data, rate, reverb["ir"], reverb["mix"])

        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        if not np.isfinite(loudness):
            raise ValueError(f"non-finite integrated loudness for {midi_path}")
        data = pyln.normalize.loudness(data, loudness, AUDIO_TARGET_LUFS)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"non-finite samples after the gain for {midi_path}")
        sf.write(tmp_path, data, rate, subtype=AUDIO_SUBTYPE)
        os.replace(tmp_path, wav_path)

    except Exception:
        print(f"Error rendering: {wav_path}")
        with open("errors.txt", "a") as f:
            f.write(wav_path + "\n")
        # Leave nothing behind: an unfinished render must not be mistaken
        # for output.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
