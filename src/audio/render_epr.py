"""Render a score as an expressive performance (EPR) with VirtuosoNet.

Which style a version gets is the pipeline's decision; this module only knows
how to run the model, where its files land, and how to keep the model's clock
track out of the audio.

The timing function comes from the model's own piecewise-constant tempo output,
not from note onsets.  The audible clock grid is retained only long enough to
identify and remove its MIDI track.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mido

VIRTUOSONET_DIR = Path("external/virtuosoNet")
CHECKPOINT = Path("pretrained_weights/han_measnote_gru/checkpoint_best.pt")

# The 16 names the released weights were trained on (its README).
EPR_STYLES: Tuple[str, ...] = (
    "Bach", "Balakirev", "Beethoven", "Brahms", "Chopin", "Debussy", "Glinka",
    "Haydn", "Liszt", "Mozart", "Prokofiev", "Rachmaninoff", "Ravel",
    "Schubert", "Schumann", "Scriabin",
)

# The clock is written as an extra piano instrument, so it would be
# synthesized along with the performance.
_CLOCK_PITCH = 64
_CLOCK_VELOCITY = 64


class EPRError(RuntimeError):
    """The performance model did not produce a usable render."""


class EPRTimeout(EPRError):
    """It ran past its time limit and was killed."""


@dataclass(frozen=True)
class EPRTempoPoint:
    """One boundary of the performance model's tempo function."""

    quarter_offset: float
    seconds: float
    qpm: float


@dataclass(frozen=True)
class EPRRender:
    """One rendered performance and the clock that produced it."""

    midi_path: Path
    beat_seconds: Tuple[float, ...]
    tempo_points: Tuple[EPRTempoPoint, ...]


def output_midi_path(xml_path: Path, output_dir: Path, style: str) -> Path:
    """Where the model writes its performance (virtuoso/inference.py)."""
    return output_dir / f"{xml_path.parent.stem}_{xml_path.stem}_by_isgn_{style}.mid"


def read_beat_seconds(midi_path: Path) -> Tuple[float, ...]:
    beat_csv = midi_path.with_name(midi_path.name + "_beat.csv")
    if not beat_csv.exists():
        raise EPRError(f"no beat grid beside {midi_path.name}")
    with open(beat_csv, newline="", encoding="utf-8") as f:
        row = next(csv.reader(f))
    return tuple(float(value) for value in row if value)


def read_tempo_points(midi_path: Path) -> Tuple[EPRTempoPoint, ...]:
    """Read the exact score-position/time segments used for the render."""
    tempo_csv = midi_path.with_name(midi_path.name + "_tempo.csv")
    if not tempo_csv.exists():
        raise EPRError(f"no tempo function beside {midi_path.name}")

    with open(tempo_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    try:
        points = tuple(EPRTempoPoint(
            quarter_offset=float(row["quarter_offset"]),
            seconds=float(row["seconds"]),
            qpm=float(row["qpm"]),
        ) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise EPRError(f"malformed tempo function beside {midi_path.name}") from exc

    if not points:
        raise EPRError(f"empty tempo function beside {midi_path.name}")
    if any(point.qpm <= 0 for point in points):
        raise EPRError(f"non-positive tempo beside {midi_path.name}")
    if any(right.quarter_offset <= left.quarter_offset
           or right.seconds <= left.seconds
           for left, right in zip(points, points[1:])):
        raise EPRError(f"non-monotonic tempo function beside {midi_path.name}")
    return points


def is_clock_track(track, n_beats: int) -> bool:
    onsets = [msg for msg in track
              if msg.type == "note_on" and msg.velocity > 0]
    if len(onsets) != n_beats:
        return False
    return all(msg.note == _CLOCK_PITCH and msg.velocity == _CLOCK_VELOCITY
               for msg in onsets)


def strip_clock_track(midi_path: Path, n_beats: int) -> None:
    """Remove the clock instrument so only the performance is synthesized."""
    midi = mido.MidiFile(str(midi_path))
    clock_tracks = [t for t in midi.tracks if is_clock_track(t, n_beats)]
    if len(clock_tracks) != 1:
        raise EPRError(
            f"{midi_path.name}: expected one clock track, found "
            f"{len(clock_tracks)}")
    midi.tracks.remove(clock_tracks[0])
    midi.save(str(midi_path))


def render_epr(
    xml_path: Path,
    style: str,
    output_dir: Path,
    *,
    qpm_primo: Optional[float] = None,
    interval_in_16th: int = 4,
    virtuosonet_dir: Path = VIRTUOSONET_DIR,
    timeout_sec: Optional[float] = 1800.0,
) -> EPRRender:
    """Run one inference pass; return the performance and its beat grid.

    `qpm_primo` is the tempo the model performs at, so speed is decided before
    the performance exists instead of by stretching one that already does.
    """
    if style not in EPR_STYLES:
        raise ValueError(f"unknown style {style!r}")
    xml_path = xml_path.resolve()
    output_dir = output_dir.resolve()
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    command: List[str] = [
        sys.executable, "-m", "virtuoso",
        "--session_mode=inference",
        f"--checkpoint={CHECKPOINT}",
        f"--xml_path={xml_path}",
        f"--composer={style}",
        f"--output_path={output_dir}",
        "--tempo_clock=true",
        f"--clock_interval_in_16th={interval_in_16th}",
    ]
    if qpm_primo is not None:
        command.append(f"--qpm_primo={qpm_primo:.12g}")

    # No --yml_path: that branch recovers the input size from the model's
    # training data, which the released weights ship without.
    try:
        result = subprocess.run(
            command, cwd=virtuosonet_dir, capture_output=True, text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        raise EPRTimeout(
            f"{xml_path.name} ({style}): killed after {timeout_sec:.0f}s")
    if result.returncode != 0:
        raise EPRError(
            f"{xml_path.name} ({style}): exit {result.returncode}\n"
            f"{result.stderr[-2000:]}")

    midi_path = output_midi_path(xml_path, output_dir, style)
    if not midi_path.exists():
        raise EPRError(
            f"{xml_path.name} ({style}): reported success but "
            f"{midi_path.name} is missing")

    beat_seconds = read_beat_seconds(midi_path)
    tempo_points = read_tempo_points(midi_path)
    strip_clock_track(midi_path, len(beat_seconds))

    # The grid is in hand and the worm plot is a byproduct; leaving either
    # behind would put stray files in the corpus directory.
    midi_path.with_name(midi_path.name + "_beat.csv").unlink(missing_ok=True)
    midi_path.with_name(midi_path.name + "_tempo.csv").unlink(missing_ok=True)
    midi_path.with_suffix(".png").unlink(missing_ok=True)

    return EPRRender(
        midi_path=midi_path,
        beat_seconds=beat_seconds,
        tempo_points=tempo_points,
    )
