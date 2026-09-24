"""
Model configuration
========================

Base configuration dataclass for the audio-to-score models.
Inherited by the piano configuration.
"""

from dataclasses import dataclass


@dataclass
class A2SConfig:
    """Base configuration for the audio-to-score models.

    Contains all architectural hyperparameters.
    Subclasses can extend with domain-specific settings.
    """

    # === Attention ===
    d_model: int = 512
    n_heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1

    # === Decoder ===
    max_seq_len: int = 4096
    vocab_size: int = 512   # Will be set from tokenizer

    # === Training ===
    label_smoothing: float = 0.0

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
