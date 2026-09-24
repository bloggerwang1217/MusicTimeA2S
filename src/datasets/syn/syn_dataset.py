"""
Synthesized Piano Dataset for piano-base
=============================================

PyTorch Dataset for synthesized piano audio, supporting both
offline (pre-generated) and online (on-the-fly) audio synthesis.

Modes:
- offline: Reads pre-generated WAV files (for piano-base, ~60GB)
- online: Synthesizes audio on-the-fly (for tutti, when storage is limited)
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SynDataset(Dataset):
    """PyTorch Dataset for synthesized piano audio.

    Supports:
    - mode='offline': Read pre-generated WAV files (piano-base)
    - mode='online': On-the-fly synthesis (tutti)

    Directory structure expected (offline mode):
        data_dir/
        ├── kern/           # Cleaned kern files
        │   ├── piece1.krn
        │   └── piece2.krn
        ├── audio/          # Pre-rendered audio (offline mode)
        │   ├── piece1_v0~soundfont.wav
        │   └── piece2_v0~soundfont.wav
        └── midi/           # MIDI files (for online mode)
            ├── piece1.mid
            └── piece2.mid
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        mode: str = "offline",
        transform: Optional[Callable] = None,
        sample_rate: int = 44100,
        max_duration: float = 30.0,
        split_file: Optional[Path] = None,
    ):
        """Initialize SynDataset.

        Args:
            data_dir: Path to data directory containing kern/, audio/, midi/
            split: One of 'train', 'valid', 'test'
            mode: 'offline' (read pre-generated WAV) or 'online' (synthesize on-the-fly)
            transform: Optional transform to apply to audio tensor
            sample_rate: Target sample rate for audio
            max_duration: Maximum audio duration in seconds (for filtering)
            split_file: Path to split file (CSV with 'name' column)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_duration = max_duration

        # Validate mode
        if mode not in ["offline", "online"]:
            raise ValueError(f"mode must be 'offline' or 'online', got '{mode}'")

        # Load samples
        self.samples: List[Dict[str, Any]] = []
        self._load_samples(split_file)

        logger.info(f"SynDataset initialized: {len(self.samples)} samples ({split}, {mode})")

    def _load_samples(self, split_file: Optional[Path] = None) -> None:
        """Load sample list based on split."""
        kern_dir = self.data_dir / "kern"
        audio_dir = self.data_dir / "audio"

        if not kern_dir.exists():
            raise FileNotFoundError(f"Kern directory not found: {kern_dir}")

        # Load split file if provided
        split_names: Optional[set] = None
        if split_file and split_file.exists():
            import pandas as pd
            # "!" is the comment marker because split names may contain "#".
            df = pd.read_csv(split_file, comment="!")
            split_names = set(df["name"].tolist())

        # Collect samples
        if self.mode == "offline":
            # Match audio files to kern files
            if not audio_dir.exists():
                raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

            audio_files = sorted(audio_dir.glob("*.wav"))
            for audio_path in audio_files:
                # Parse filename: piece_v0~soundfont.wav
                stem = audio_path.stem
                if "~" in stem:
                    piece_part = stem.split("~")[0]
                else:
                    piece_part = stem

                # Remove version suffix (_v0, _v1, etc.)
                if "_v" in piece_part:
                    base_name = piece_part.rsplit("_v", 1)[0]
                else:
                    base_name = piece_part

                # Find corresponding kern file
                kern_path = kern_dir / f"{base_name}.krn"
                if not kern_path.exists():
                    logger.warning(f"Kern file not found for {audio_path}: {kern_path}")
                    continue

                # Check split membership
                if split_names is not None:
                    if not self._in_split(base_name, split_names):
                        continue

                self.samples.append({
                    "audio_path": audio_path,
                    "kern_path": kern_path,
                    "name": base_name,
                })

        else:  # online mode
            midi_dir = self.data_dir / "midi"
            if not midi_dir.exists():
                raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")

            kern_files = sorted(kern_dir.glob("*.krn"))
            for kern_path in kern_files:
                base_name = kern_path.stem
                midi_path = midi_dir / f"{base_name}.mid"

                if not midi_path.exists():
                    logger.warning(f"MIDI file not found for {kern_path}: {midi_path}")
                    continue

                # Check split membership
                if split_names is not None:
                    if not self._in_split(base_name, split_names):
                        continue

                self.samples.append({
                    "midi_path": midi_path,
                    "kern_path": kern_path,
                    "name": base_name,
                })

    def _in_split(self, name: str, split_names: set) -> bool:
        """Check if a sample belongs to the current split."""
        # Direct match
        if name in split_names:
            return True

        # Handle different naming conventions
        # e.g., "beethoven_piano_sonatas_sonata01-1" vs "beethoven#sonata01-1"
        for split_name in split_names:
            if "#" in split_name:
                prefix, piece = split_name.split("#", 1)
                prefix_map = {
                    "beethoven": "beethoven_piano_sonatas",
                    "haydn": "haydn_piano_sonatas",
                    "mozart": "mozart_piano_sonatas",
                    "chopin": "humdrum_chopin_first_editions",
                    "joplin": "joplin",
                    "scarlatti": "scarlatti_keyboard_sonatas",
                }
                if prefix in prefix_map:
                    expected = f"{prefix_map[prefix]}_{piece}"
                    if name == expected:
                        return True
            elif name.startswith("musesyn_"):
                if name[8:] == split_name:
                    return True

        return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict[str, Any]]:
        """Get a sample.

        Returns:
            Tuple of (audio_tensor, kern_content, metadata)
            - audio_tensor: [1, num_samples] float tensor
            - kern_content: Full kern file content as string
            - metadata: Dict with 'name', 'audio_path'/'midi_path', etc.
        """
        sample = self.samples[idx]

        # Load kern content
        with open(sample["kern_path"], "r", encoding="utf-8") as f:
            kern_content = f.read()

        # Load or synthesize audio
        if self.mode == "offline":
            data, sr = sf.read(sample["audio_path"], dtype="float32")
            audio = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
        else:
            audio, sr = self._synthesize_audio(sample["midi_path"])

        # Resample if necessary
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Apply transform if provided
        if self.transform is not None:
            audio = self.transform(audio)

        metadata = {
            "name": sample["name"],
            "kern_path": str(sample["kern_path"]),
        }
        if "audio_path" in sample:
            metadata["audio_path"] = str(sample["audio_path"])
        if "midi_path" in sample:
            metadata["midi_path"] = str(sample["midi_path"])

        return audio, kern_content, metadata

    def _synthesize_audio(self, midi_path: Path) -> Tuple[torch.Tensor, int]:
        """Synthesize audio from MIDI on-the-fly (online mode).

        This is a placeholder - full implementation requires FluidSynth setup.
        """
        raise NotImplementedError(
            "Online synthesis not yet implemented. "
            "Use mode='offline' with pre-generated audio files."
        )


def create_dataloader(
    data_dir: Path,
    split: str,
    batch_size: int = 1,
    mode: str = "offline",
    num_workers: int = 4,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for SynDataset.

    Args:
        data_dir: Path to data directory
        split: One of 'train', 'valid', 'test'
        batch_size: Batch size
        mode: 'offline' or 'online'
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for SynDataset

    Returns:
        DataLoader instance
    """
    dataset = SynDataset(
        data_dir=data_dir,
        split=split,
        mode=mode,
        **dataset_kwargs,
    )

    shuffle = split == "train"

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
