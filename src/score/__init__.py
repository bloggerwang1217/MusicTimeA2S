"""Score processing utilities (MusicXML, Kern)."""

from .generate_score import kern_to_musicxml
from .clean_kern import (
    extract_visual_info,
    extract_visual_from_sequence,
    strip_cue_passages,
    strip_spines,
    fold_extra_spines,
    strip_articulation,
    strip_ties,
    clean_kern_token,
    clean_kern_sequence,
)
from .sanitize_piano_score import heal_cross_staff

__all__ = [
    "kern_to_musicxml",
    # Visual information extraction (for Visual Auxiliary Head)
    "extract_visual_info",
    "extract_visual_from_sequence",
    # Cue passage removal
    "strip_cue_passages",
    # Spine filtering
    "strip_spines",
    "fold_extra_spines",
    # Kern token cleaning (Phase 1: visual only)
    "clean_kern_token",
    "clean_kern_sequence",
    # Content stripping (Phase 1.5: standardize)
    "strip_articulation",
    "strip_ties",
    # Score sanitization
    "heal_cross_staff",
]
