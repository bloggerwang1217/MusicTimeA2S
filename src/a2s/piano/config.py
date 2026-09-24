"""
Piano configuration
========================

Configuration for the piano model.
Extends the base configuration with piano-specific settings.
"""

from dataclasses import dataclass
from typing import Optional

from ..config import A2SConfig


@dataclass
class PianoConfig(A2SConfig):
    """Configuration for the piano model.

    hFT front end, learned converter and an autoregressive Transformer decoder.
    """

    # Model name (for model selection in train.py)
    name: str = "piano-base"

    # Piano-specific defaults
    vocab_size: int = 512   # ~220 factorized tokens + padding

    # Token cross-entropy
    label_smoothing: float = 0.0

    # Chunking for long pieces
    chunk_frames: int = 1280
    overlap_frames: int = 640

    # Training hyperparameters (read from YAML; single source of truth)
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    lr_min_ratio: float = 0.1   # min_lr = learning_rate * lr_min_ratio (cosine decay endpoint)
    phase_lr_decay_start_step: Optional[int] = None
    phase_lr_decay_end_step: Optional[int] = None
    phase_lr_min: Optional[float] = None
    max_epochs: int = 100
    training_seed: int = 1234

    # Runtime / infrastructure (read from YAML; CLI args are only --config/--resume/--wandb/--debug/--sanity-check)
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_clip: float = 1.0
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 0   # 0 = disabled

    # Paths (defaults match legacy CLI defaults; override in YAML)
    manifest_dir: str = "data/experiments/syn"
    checkpoint_dir: str = "checkpoints/full"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PianoConfig":
        """Load config from YAML file."""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Extract model config
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        training_cfg = cfg.get("training", {})
        seed_cfg = cfg.get("seed", {})
        chunk_cfg = data_cfg.get("chunking", {})

        # Get defaults from a temporary instance
        defaults = cls()

        return cls(
            # Model name
            name=model_cfg.get("name", defaults.name),

            # Attention
            d_model=model_cfg.get("d_model", defaults.d_model),
            n_heads=model_cfg.get("n_heads", defaults.n_heads),
            ff_dim=model_cfg.get("ff_dim", defaults.ff_dim),
            dropout=model_cfg.get("dropout", defaults.dropout),

            label_smoothing=model_cfg.get("label_smoothing", defaults.label_smoothing),
            # Architecture
            max_seq_len=model_cfg.get("max_seq_len", defaults.max_seq_len),

            # Chunking
            chunk_frames=chunk_cfg.get("chunk_frames", defaults.chunk_frames),
            overlap_frames=chunk_cfg.get("overlap_frames", defaults.overlap_frames),

            # Training
            learning_rate=training_cfg.get("learning_rate", 1e-4),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            warmup_steps=training_cfg.get("warmup_steps", 1000),
            lr_min_ratio=training_cfg.get("lr_min_ratio", 0.1),
            phase_lr_decay_start_step=training_cfg.get("phase_lr_decay_start_step"),
            phase_lr_decay_end_step=training_cfg.get("phase_lr_decay_end_step"),
            phase_lr_min=training_cfg.get("phase_lr_min"),
            max_epochs=training_cfg.get("max_epochs", 100),
            training_seed=seed_cfg.get("training", 1234),

            # Runtime / infrastructure
            batch_size=training_cfg.get("batch_size", defaults.batch_size),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", defaults.gradient_accumulation_steps),
            gradient_clip=training_cfg.get("gradient_clip", defaults.gradient_clip),
            save_every_n_epochs=training_cfg.get("save_every_n_epochs", defaults.save_every_n_epochs),
            early_stopping_patience=training_cfg.get("early_stopping_patience", defaults.early_stopping_patience),

            # Paths
            manifest_dir=cfg.get("paths", {}).get("manifest_dir", defaults.manifest_dir),
            checkpoint_dir=cfg.get("paths", {}).get("checkpoint_dir", defaults.checkpoint_dir),
        )
