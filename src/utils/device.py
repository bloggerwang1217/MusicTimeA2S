"""
Device detection and selection utilities.

Provides automatic GPU detection with priority: CUDA > MPS > CPU
"""

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def get_device() -> str:
    """
    Automatically select the best available device.

    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU

    Returns:
        Device string for PyTorch ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        Dictionary with device availability and details
    """
    info = {
        "selected": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
    }

    if info["cuda_available"] and info["cuda_device_count"] > 0:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    return info


def log_device_info() -> str:
    """
    Log device information and return selected device.

    Returns:
        Selected device string
    """
    device = get_device()

    if device == "cuda":
        logger.info(f"Using CUDA (NVIDIA GPU): {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        logger.info("Using MPS (Apple Silicon GPU)")
    else:
        logger.info("Using CPU (no GPU acceleration available)")

    return device
