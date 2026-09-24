"""
Audio format conversion utilities.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """
    Check if ffmpeg is available in the system.

    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True, timeout=10
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = 44100,
    mono: bool = True,
) -> Path:
    """
    Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to input audio file (MP3, FLAC, etc.)
        output_path: Path for output WAV file
        sample_rate: Target sample rate (default: 44100 Hz)
        mono: Convert to mono if True (default: True)

    Returns:
        Path to the converted WAV file

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If ffmpeg is not available or conversion fails
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. "
            "Please install ffmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-y",  # Overwrite output
    ]

    if mono:
        cmd.extend(["-ac", "1"])

    cmd.append(str(output_path))

    try:
        result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        logger.info(f"Converted {input_path.name} -> {output_path.name}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg conversion failed: {e.stderr}")
        raise RuntimeError(f"Audio conversion failed: {e.stderr}")


def load_audio(
    path: Union[str, Path],
    sample_rate: int = 44100,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.

    Args:
        path: Path to audio file
        sample_rate: Target sample rate (None to use original)
        mono: Convert to mono if True

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
    logger.debug(f"Loaded audio: {path.name}, shape={audio.shape}, sr={sr}")

    return audio, sr


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sample_rate: int = 44100,
) -> Path:
    """
    Save audio array to file.

    Args:
        audio: Audio array (1D for mono, 2D for stereo with shape [channels, samples])
        path: Output file path
        sample_rate: Sample rate

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle stereo audio (channels, samples) -> (samples, channels) for soundfile
    if audio.ndim == 2:
        audio = audio.T

    sf.write(str(path), audio, sample_rate)
    logger.debug(f"Saved audio: {path.name}")

    return path
