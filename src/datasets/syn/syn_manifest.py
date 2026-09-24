"""
Synthetic Dataset Manifest Builder
==================================

Creates train/valid/test manifest JSON files for synthetic datasets.
Manifests contain paths to mel spectrograms and ground truth kern files.

Usage:
    from src.datasets.syn.syn_manifest import create_manifest
    counts = create_manifest(data_dir, config_path)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.datasets.manifest import create_manifests_from_metadata

PROTOCOL_EXCLUDE_FILENAME = "protocol_exclude.txt"
_TEST_SOUNDFONTS = {
    Path(name).stem
    for name in json.loads(
        (Path(__file__).resolve().parents[2] / "audio/augmentation.json").read_text()
    )["soundfonts"]["test"]
}


def is_test_render(render_id: str) -> bool:
    # Existing render files remain evidence even when the evaluation timbre changes.
    return render_id.rsplit("~", 1)[-1] in _TEST_SOUNDFONTS


def select_test_renders(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in entries if is_test_render(
        entry.get("performance_id") or entry.get("recording_id") or entry["id"]
    )]


def load_protocol_exclude(metadata_dir: Optional[Union[str, Path]] = None) -> set:
    """Kern stems barred from every split (see protocol_exclude.txt for why).

    Lines starting with "!" are comments; "#" cannot serve as the marker
    because split entries such as "beethoven#sonata03-1" contain it.
    """
    if metadata_dir is None:
        metadata_dir = Path(__file__).resolve().parent
    path = Path(metadata_dir) / PROTOCOL_EXCLUDE_FILENAME
    if not path.exists():
        return set()

    stems = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("!") or line == "name":
            continue
        stems.add(line)
    return stems


def create_manifest(
    data_dir: Union[str, Path],
    metadata_filename: str = "augmentation_metadata.json",
    validate_files: bool = True,
    sample_validation_count: int = 5,
    metadata_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, int]:
    """Create train/valid/test manifests from augmentation metadata.

    Syn's protocol exclusion is resolved here before the dataset-neutral
    compiler sees the metadata.
    """
    with (Path(data_dir) / metadata_filename).open() as handle:
        metadata = json.load(handle)
    excluded_render_ids = {
        render["audio_key"]
        for entry in metadata.values() if entry["split"] == "test"
        for render in entry.get("renders", [])
        if not is_test_render(render["audio_key"])
    }
    return create_manifests_from_metadata(
        data_dir=data_dir,
        metadata_filename=metadata_filename,
        validate_files=validate_files,
        sample_validation_count=sample_validation_count,
        excluded_stems=load_protocol_exclude(metadata_dir),
        excluded_render_ids=excluded_render_ids,
    )


def load_manifest(
    manifest_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load a manifest JSON file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        List of manifest entries
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_manifest_stats(manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get statistics from a manifest.

    Args:
        manifest: List of manifest entries

    Returns:
        Dictionary with statistics:
        - count: number of entries
        - total_duration_sec: total duration in seconds
        - total_frames: total number of mel frames
        - avg_duration_sec: average duration per sample
    """
    if not manifest:
        return {
            "count": 0,
            "total_duration_sec": 0.0,
            "total_frames": 0,
            "avg_duration_sec": 0.0,
        }

    total_duration = sum(entry["duration_sec"] for entry in manifest)
    total_frames = sum(entry["n_frames"] for entry in manifest)
    return {
        "count": len(manifest),
        "total_duration_sec": total_duration,
        "total_frames": total_frames,
        "avg_duration_sec": total_duration / len(manifest),
    }
