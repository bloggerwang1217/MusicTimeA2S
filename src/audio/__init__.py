"""Audio processing utilities."""

from .converter import convert_to_wav, load_audio
from .mel import (
    audio_to_mel,
    load_audio_to_mel,
    process_audio_file,
    duration_to_frames,
    frames_to_duration,
)
from .render_epr import (
    EPR_STYLES,
    EPRError,
    EPRRender,
    EPRTempoPoint,
    EPRTimeout,
    render_epr,
)
from .separator import VocalSeparator
from .synthesis import MIDIProcess, render_one_midi

__all__ = [
    # converter
    "convert_to_wav",
    "load_audio",
    # mel
    "audio_to_mel",
    "load_audio_to_mel",
    "process_audio_file",

    "duration_to_frames",
    "frames_to_duration",
    # render_epr
    "EPR_STYLES",
    "EPRError",
    "EPRRender",
    "EPRTempoPoint",
    "EPRTimeout",
    "render_epr",
    # separator
    "VocalSeparator",
    # synthesis
    "MIDIProcess",
    "render_one_midi",
]
