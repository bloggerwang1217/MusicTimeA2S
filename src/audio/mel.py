"""
Mel Spectrogram Utilities
=========================

Stateless functions for converting audio to log-mel spectrograms.
All mel parameters are caller-supplied (no built-in defaults);
the frozen foundation spec lives in HFT_MEL.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def audio_to_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    mel_norm: Optional[str],
    log_offset: float,
    pad_mode: str,
    normalize: bool,
) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram.

    Returns:
        Log-mel spectrogram tensor of shape [1, n_mels, T]
        where T = num_samples / hop_length
    """
    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Ensure shape is [1, samples]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=mel_norm,
        pad_mode=pad_mode,
    )

    # Generate mel spectrogram: [1, n_mels, T]
    mel = mel_transform(waveform)

    mel = torch.log(mel + log_offset)

    # Per-sample normalization
    if normalize:
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    return mel


def load_audio_to_mel(
    audio_path: Union[str, Path],
    target_sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    mel_norm: Optional[str],
    log_offset: float,
    pad_mode: str,
    normalize: bool,
) -> torch.Tensor:
    """Load audio file and convert to log-mel spectrogram.

    Returns:
        Log-mel spectrogram tensor of shape [1, n_mels, T]
    """
    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)

    return audio_to_mel(
        waveform=waveform,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        mel_norm=mel_norm,
        log_offset=log_offset,
        pad_mode=pad_mode,
        normalize=normalize,
    )


def process_audio_file(
    audio_path: Union[str, Path],
    mel_path: Union[str, Path],
    target_sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    mel_norm: Optional[str],
    log_offset: float,
    pad_mode: str,
    normalize: bool,
    skip_existing: bool = True,
) -> Tuple[str, Optional[Tuple[int, ...]]]:
    """Process a single audio file and save mel spectrogram as .npy."""
    audio_path = Path(audio_path)
    mel_path = Path(mel_path)

    if not audio_path.exists():
        return "missing", None

    # A mel older than its audio was taken from a render that has since been
    # replaced, so existence alone does not make it reusable.
    if (skip_existing and mel_path.exists()
            and mel_path.stat().st_mtime_ns >= audio_path.stat().st_mtime_ns):
        return "skipped", None

    try:
        mel = load_audio_to_mel(
            audio_path=audio_path,
            target_sample_rate=target_sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            mel_norm=mel_norm,
            log_offset=log_offset,
            pad_mode=pad_mode,
            normalize=normalize,
        )

        # Ensure parent directory exists
        mel_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as numpy for mmap support (OS page cache, zero-copy slice).
        # Publish atomically: a kill mid-write would leave a short .npy that
        # skip_existing accepts and mmap then faults on.  np.save takes the
        # handle, not the path, so it cannot append a second .npy suffix.
        tmp_path = mel_path.with_name(f".{mel_path.name}")
        with open(tmp_path, "wb") as fh:
            np.save(fh, mel.numpy())
        os.replace(tmp_path, mel_path)

        return "generated", tuple(mel.shape)

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to process {audio_path}: {e}")
        return "failed", None


def _process_audio_task(args):
    key, audio_path, mel_path, spec, skip_existing = args
    status, shape = process_audio_file(
        audio_path=audio_path,
        mel_path=mel_path,
        target_sample_rate=spec.sample_rate,
        n_mels=spec.n_mels,
        n_fft=spec.n_fft,
        hop_length=spec.hop_length,
        f_min=spec.f_min,
        f_max=spec.f_max,
        mel_norm=spec.mel_norm,
        log_offset=spec.log_offset,
        pad_mode=spec.pad_mode,
        normalize=spec.per_sample_normalize,
        skip_existing=skip_existing,
    )
    return key, status, shape


def process_audio_batch(
    tasks: Sequence[Tuple[str, Union[str, Path], Union[str, Path]]],
    *,
    spec: Any,
    workers: int = 1,
    skip_existing: bool = True,
    description: str = "Generating mel",
) -> Dict[str, Tuple[str, Optional[Tuple[int, ...]]]]:
    """Run the shared mel transform for a caller-defined set of audio files."""
    keys = [key for key, _, _ in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("mel task keys must be unique")
    worker_tasks = [
        (key, str(audio_path), str(mel_path), spec, skip_existing)
        for key, audio_path, mel_path in tasks
    ]
    if workers <= 1:
        rows = (
            _process_audio_task(task)
            for task in tqdm(worker_tasks, desc=description)
        )
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        rows = executor.map(_process_audio_task, worker_tasks)
        rows = tqdm(rows, total=len(worker_tasks), desc=description)

    try:
        return {key: (status, shape) for key, status, shape in rows}
    finally:
        if workers > 1:
            executor.shutdown()


def duration_to_frames(duration_sec: float, sample_rate: int, hop_length: int) -> int:
    """Number of mel frames from audio duration."""
    return int(duration_sec * sample_rate / hop_length)


def frames_to_duration(n_frames: int, sample_rate: int, hop_length: int) -> float:
    """Audio duration from number of mel frames."""
    return n_frames * hop_length / sample_rate
