"""
MusicXML generation utilities.
"""

import logging
from pathlib import Path
from typing import Union

import converter21
import music21

# Register converter21 for Humdrum support
converter21.register()

logger = logging.getLogger(__name__)


def kern_to_musicxml(
    kern_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """
    Convert a Humdrum kern file to MusicXML.

    Uses converter21 to parse kern and music21 to write MusicXML.

    Args:
        kern_path: Path to input kern file
        output_path: Path to output MusicXML file

    Returns:
        Path to saved MusicXML file

    Example:
        >>> kern_to_musicxml("score.krn", "score.musicxml")
    """
    kern_path = Path(kern_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting kern to MusicXML: {kern_path}")

    # Parse kern file using converter21
    score = music21.converter.parse(str(kern_path))

    # Write to MusicXML
    score.write("musicxml", fp=str(output_path))

    logger.info(f"Saved MusicXML to: {output_path}")
    return output_path
