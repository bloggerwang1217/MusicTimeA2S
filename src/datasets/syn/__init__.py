"""Synthetic dataset module for piano-base.

This module contains:
- syn_dataset.py: PyTorch Dataset class
- syn_manifest.py: Manifest builder for train/valid/test splits
- Metadata files: split definitions, failed kerns list
"""

from .syn_dataset import SynDataset
from .syn_manifest import create_manifest, load_manifest, get_manifest_stats

__all__ = [
    "SynDataset",
    "create_manifest",
    "load_manifest",
    "get_manifest_stats",
]
