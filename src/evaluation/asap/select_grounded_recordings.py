"""Restrict a test manifest to the recordings that a grounding manifest covers.

    python -m src.evaluation.asap.select_grounded_recordings \
        --manifest test_manifest.json --grounding grounding.jsonl --output out.json

The Piano-A2S grounding only exists for the recordings its own pipeline could
process, and inference on the pre-segmented windows must read the same
recording set, in the manifest's original order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grounded = set()
    with args.grounding.open() as handle:
        for line in handle:
            if line.strip():
                grounded.add(json.loads(line)["recording_id"].replace("#", "__"))
    items = json.loads(args.manifest.read_text())
    selected = [item for item in items if item["id"] in grounded]
    missing = grounded - {item["id"] for item in selected}
    if missing:
        raise ValueError(f"Grounded recordings absent from the manifest: {sorted(missing)[:10]}")
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.output}: {len(selected)} of {len(items)} recordings")


if __name__ == "__main__":
    main()
