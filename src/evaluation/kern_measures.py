"""Split a whole-piece kern into per-measure blocks and per-measure state."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.score.kern_postprocess import _active_spine_path_rows

# The mel frame rate every whole-piece timeline is expressed in.
MEL_FPS = 62.5

_METER_RE = re.compile(r"^\*M(\d+)/(\d+)$")
_KEYSIG_RE = re.compile(r"^\*k\[[^\]]*\]$")


def parse_whole_piece(kern_text: str) -> Tuple[List[List[str]], List[dict]]:
    """Split a whole-piece kern into per-measure blocks + per-position state.

    blocks[p] = lines of measure p (barline row first); state[p] carries the key, meter,
    and active spine paths when measure p starts. Interpretation rows inside a block
    update the running state for later positions. Header rows before the
    first barline seed the state and are not kept as content.
    """
    blocks: List[List[str]] = []
    state: List[dict] = []
    current: Optional[List[str]] = None
    cur_key: Optional[str] = None
    cur_meter: Optional[str] = None

    for raw in kern_text.splitlines():
        if not raw:
            continue
        first = raw.split("\t", 1)[0].strip()
        if first.startswith("="):
            if current is not None:
                blocks.append(current)
            state.append({"key": cur_key, "meter": cur_meter})
            current = [raw]
            continue
        if first == "*-":
            continue
        if _KEYSIG_RE.match(first):
            cur_key = first
        else:
            m = _METER_RE.match(first)
            if m:
                cur_meter = first
        if current is not None:
            current.append(raw)
    if current is not None:
        blocks.append(current)
    lines = kern_text.splitlines(keepends=True)
    bar_positions = [
        i for i, line in enumerate(lines)
        if line.split("\t", 1)[0].strip().startswith("=")
    ]
    if bar_positions:
        for item, position in zip(state, bar_positions, strict=True):
            item["width"] = len(lines[position].rstrip("\r\n").split("\t"))
            item["spine_paths"] = [
                row.rstrip("\r\n") for row in _active_spine_path_rows(
                    lines, bar_positions[0], position,
                )
            ]
        closing = next(
            (
                line.rstrip("\r\n").split("\t") for line in reversed(lines)
                if line.strip() and all(
                    cell == "*-" for cell in line.rstrip("\r\n").split("\t")
                )
            ),
            None,
        )
        if closing is None:
            raise ValueError("Whole-piece kern has no complete spine terminator")
        state[-1]["closing_width"] = len(closing)
    return blocks, state
