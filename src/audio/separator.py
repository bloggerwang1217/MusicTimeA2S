"""
Audio source separation using Demucs.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from ..utils.device import get_device, log_device_info

logger = logging.getLogger(__name__)


class VocalSeparator:
    """
    Separate vocals from music using Demucs htdemucs model.

    The htdemucs model separates audio into 4 stems:
    - drums (index 0)
    - bass (index 1)
    - other (index 2)
    - vocals (index 3)
    """

    STEM_INDICES = {"drums": 0, "bass": 1, "other": 2, "vocals": 3}

    def __init__(self, model_name: str = "htdemucs", device: Optional[str] = None):
        """
        Initialize the vocal separator.

        Args:
            model_name: Demucs model name (default: "htdemucs")
            device: Device to use ("cuda", "mps", "cpu") or None for auto-detect
        """
        self.model_name = model_name
        self.device = device or get_device()
        self._model = None

    def _load_model(self):
        """Lazy load the Demucs model."""
        if self._model is None:
            from demucs.pretrained import get_model

            logger.info(f"Loading {self.model_name} model...")
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            log_device_info()

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems.

        Args:
            audio: Audio array, shape (samples,) for mono or (2, samples) for stereo
            sample_rate: Sample rate of the audio

        Returns:
            Dictionary with stem names as keys and audio arrays as values
        """
        self._load_model()

        from demucs.apply import apply_model

        # Ensure stereo format (2, samples)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        # Convert to tensor: (batch=1, channels=2, samples)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        logger.info("Running source separation...")
        with torch.no_grad():
            sources = apply_model(self._model, audio_tensor, device=self.device)

        # Convert back to numpy
        # sources shape: (batch=1, stems=4, channels=2, samples)
        result = {}
        for stem_name, idx in self.STEM_INDICES.items():
            result[stem_name] = sources[0, idx].cpu().numpy()

        return result

    def extract_vocals(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        mono: bool = True,
    ) -> np.ndarray:
        """
        Extract only vocals from audio.

        Args:
            audio: Audio array
            sample_rate: Sample rate
            mono: Convert to mono if True

        Returns:
            Vocals audio array
        """
        stems = self.separate(audio, sample_rate)
        vocals = stems["vocals"]

        if mono and vocals.shape[0] == 2:
            vocals = np.mean(vocals, axis=0)

        return vocals

    def extract_accompaniment(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        mono: bool = True,
    ) -> np.ndarray:
        """
        Extract accompaniment (everything except vocals).

        Args:
            audio: Audio array
            sample_rate: Sample rate
            mono: Convert to mono if True

        Returns:
            Accompaniment audio array
        """
        stems = self.separate(audio, sample_rate)

        # Sum drums, bass, and other
        accompaniment = stems["drums"] + stems["bass"] + stems["other"]

        if mono and accompaniment.shape[0] == 2:
            accompaniment = np.mean(accompaniment, axis=0)

        return accompaniment
