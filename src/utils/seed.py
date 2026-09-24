"""Seed utilities for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    This matches Zeng et al.'s seed setting for fair comparison.

    Args:
        seed: Random seed value

    Usage:
        # Data augmentation (use seed=0 to match Zeng)
        set_seed(0)

        # Training (use seed=1234 to match Zeng)
        set_seed(1234)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state_dict() -> dict:
    """Capture the RNG streams set_seed initializes, for exact resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state_dict(state: dict) -> None:
    """Restore streams captured by rng_state_dict.

    Raises on a CUDA topology mismatch so the caller can fall back loudly
    instead of resuming half-restored.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are normally loaded onto the training device, which also
    # relocates this CPU-generator state. PyTorch requires a CPU ByteTensor at
    # the restore boundary regardless of the model's device.
    torch_cpu_state = state["torch_cpu"].detach().to(
        device="cpu", dtype=torch.uint8,
    ).contiguous()
    torch.set_rng_state(torch_cpu_state)
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                f"checkpoint has RNG states for {len(cuda_states)} CUDA "
                f"devices, this process sees {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([
            cuda_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            for cuda_state in cuda_states
        ])


# Default seeds (matching Zeng et al.)
SEED_DATA_AUGMENTATION = 0
SEED_TRAINING = 1234
