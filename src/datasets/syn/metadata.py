"""Helpers for resolving synthetic-render augmentation metadata."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def index_augmentation_metadata_by_render(
    metadata: Mapping[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Index source-version metadata by both source and rendered-audio IDs."""
    by_render: Dict[str, Dict[str, Any]] = {}
    owner: Dict[str, str] = {}

    for source_id, entry in metadata.items():
        previous_owner = owner.get(source_id)
        if previous_owner is not None and previous_owner != source_id:
            raise ValueError(
                f"Augmentation source ID {source_id} collides with a render of "
                f"{previous_owner}"
            )
        by_render[source_id] = entry
        owner[source_id] = source_id

        for render in entry.get("renders", []):
            audio_key = render.get("audio_key")
            if not audio_key:
                continue
            previous_owner = owner.get(audio_key)
            if previous_owner is not None and previous_owner != source_id:
                raise ValueError(
                    f"Duplicate augmentation metadata for {audio_key}: "
                    f"{previous_owner} and {source_id}"
                )
            by_render[audio_key] = entry
            owner[audio_key] = source_id

    return by_render
