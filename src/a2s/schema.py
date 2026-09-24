"""
Schema helpers — key/meter vocabulary, GT builders, bar masks.

Extracted from the old decoder.py; model-agnostic.
"""

from typing import List, Optional, Tuple
from functools import lru_cache
import re

import torch


# =============================================================================
# Schema vocabulary (key / meter class labels)
# =============================================================================

KEY_LABELS: List[str] = [
    "0",
    "1#", "2#", "3#", "4#", "5#", "6#", "7#",
    "1b", "2b", "3b", "4b", "5b", "6b", "7b",
]
N_KEYS: int = len(KEY_LABELS)  # 15

# 5/4 excluded: train has only 5 scattered 5/4 bars (no learnable signal)
METER_LABELS: List[str] = [
    "4/4", "3/4", "2/4", "2/2", "6/8", "3/8",
    "12/8", "6/4", "9/8", "9/16", "7/4",
    "12/16", "6/16", "12/32", "10/4", "17/16", "2/8", "4/2",
]
N_METERS: int = len(METER_LABELS)  # 18

_KEY_LABEL_TO_ID = {label: i for i, label in enumerate(KEY_LABELS)}
_METER_LABEL_TO_ID = {label: i for i, label in enumerate(METER_LABELS)}

_KEY_SIG_TO_LABEL = {
    "*k[]": "0",
    "*k[f#]": "1#", "*k[f#c#]": "2#", "*k[f#c#g#]": "3#",
    "*k[f#c#g#d#]": "4#", "*k[f#c#g#d#a#]": "5#",
    "*k[f#c#g#d#a#e#]": "6#", "*k[f#c#g#d#a#e#b#]": "7#",
    "*k[b-]": "1b", "*k[b-e-]": "2b", "*k[b-e-a-]": "3b",
    "*k[b-e-a-d-]": "4b", "*k[b-e-a-d-g-]": "5b",
    "*k[b-e-a-d-g-c-]": "6b", "*k[b-e-a-d-g-c-f-]": "7b",
}

_METER_RE = re.compile(r"^\*M(\d+/\d+)\b")
_KEY_RE = re.compile(r"^\*k\[[^\]]*\]")
_BAR_RE = re.compile(r"^=\d+")


@lru_cache(maxsize=8192)
def build_per_bar_schema_gt(
    chunk_kern: str,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Extract per-bar (key_id, meter_id) from a chunk's humdrum text.
    -100 for out-of-vocab meters/keys."""
    current_meter: Optional[str] = None
    current_key: Optional[str] = None
    key_ids: List[int] = []
    meter_ids: List[int] = []
    for raw in chunk_kern.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        first_field = line.split("\t", 1)[0]
        if first_field.startswith("*"):
            m = _METER_RE.match(first_field)
            if m:
                current_meter = m.group(1)
                continue
            if _KEY_RE.match(first_field):
                current_key = _KEY_SIG_TO_LABEL.get(first_field)
                continue
        if _BAR_RE.match(first_field):
            key_id = _KEY_LABEL_TO_ID.get(current_key, -100) if current_key else -100
            meter_id = _METER_LABEL_TO_ID.get(current_meter, -100) if current_meter else -100
            key_ids.append(key_id)
            meter_ids.append(meter_id)
    return tuple(key_ids), tuple(meter_ids)


@lru_cache(maxsize=8192)
def build_per_bar_key_gt(chunk_kern: str) -> Tuple[int, ...]:
    """Extract per-bar key IDs without constructing legacy meter labels."""
    current_key: Optional[str] = None
    key_ids: List[int] = []
    for raw in chunk_kern.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        first_field = line.split("\t", 1)[0]
        if first_field.startswith("*"):
            if _KEY_RE.match(first_field):
                current_key = _KEY_SIG_TO_LABEL.get(first_field)
            continue
        if _BAR_RE.match(first_field):
            key_ids.append(
                _KEY_LABEL_TO_ID.get(current_key, -100)
                if current_key else -100
            )
    return tuple(key_ids)


def build_batch_schema_gt(
    chunk_kern_list: List[Optional[str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B, N_max] padded per-bar GT; -100 for pad/OOV."""
    per_keys: List[Tuple[int, ...]] = []
    per_meters: List[Tuple[int, ...]] = []
    n_max = 0
    for kern in chunk_kern_list or []:
        if kern is None:
            per_keys.append(())
            per_meters.append(())
            continue
        ks, ms = build_per_bar_schema_gt(kern)
        per_keys.append(ks)
        per_meters.append(ms)
        n_max = max(n_max, len(ks))
    B = len(per_keys)
    if n_max == 0 or B == 0:
        return (
            torch.zeros(B, 0, dtype=torch.long, device=device),
            torch.zeros(B, 0, dtype=torch.long, device=device),
        )
    key_gt = torch.full((B, n_max), -100, dtype=torch.long, device=device)
    meter_gt = torch.full((B, n_max), -100, dtype=torch.long, device=device)
    for b, (ks, ms) in enumerate(zip(per_keys, per_meters)):
        if ks:
            key_gt[b, : len(ks)] = torch.tensor(ks, dtype=torch.long, device=device)
            meter_gt[b, : len(ms)] = torch.tensor(ms, dtype=torch.long, device=device)
    return key_gt, meter_gt


def build_batch_schema_gt_from_ids(
    key_ids_list: List[Optional[List[int]]],
    meter_ids_list: List[Optional[List[int]]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B, N_max] padded from pre-extracted per-sample id lists."""
    B = len(key_ids_list or [])
    n_max = 0
    for ks in key_ids_list or []:
        if ks:
            n_max = max(n_max, len(ks))
    if B == 0 or n_max == 0:
        return (
            torch.zeros(B, 0, dtype=torch.long, device=device),
            torch.zeros(B, 0, dtype=torch.long, device=device),
        )
    key_gt = torch.full((B, n_max), -100, dtype=torch.long, device=device)
    meter_gt = torch.full((B, n_max), -100, dtype=torch.long, device=device)
    for b in range(B):
        ks = key_ids_list[b] if key_ids_list else None
        ms = meter_ids_list[b] if meter_ids_list else None
        if ks:
            key_gt[b, : len(ks)] = torch.tensor(ks, dtype=torch.long, device=device)
        if ms:
            meter_gt[b, : len(ms)] = torch.tensor(ms, dtype=torch.long, device=device)
    return key_gt, meter_gt


def build_bar_mask(
    input_ids: torch.Tensor,
    bar_token_id: int,
    sos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Returns [B, S] bool: True at <bar> positions (excluding <sos> if given)."""
    mask = input_ids == bar_token_id
    if sos_token_id is not None:
        mask = mask & (input_ids != sos_token_id)
    return mask


def build_bar_token_index(
    input_ids: torch.Tensor,
    bar_token_id: int,
    pad_token_id: int = 0,
    sos_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map each token to its enclosing bar.

    Returns:
        bar_mask: [B, S] True at <bar> tokens
        bar_index: [B, S] int, -1 before first bar / on padding
        in_bar_mask: [B, S] bool, True for tokens belonging to a decoded bar
    """
    bar_mask = build_bar_mask(input_ids, bar_token_id, sos_token_id)
    valid = input_ids != pad_token_id
    bar_index = bar_mask.long().cumsum(dim=-1) - 1
    in_bar_mask = (bar_index >= 0) & valid
    bar_index = torch.where(in_bar_mask, bar_index, torch.full_like(bar_index, -1))
    return bar_mask, bar_index, in_bar_mask
