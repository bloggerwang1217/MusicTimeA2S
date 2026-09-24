"""Meter and key switch penalties from the switch rates of a training corpus.

Inference decides meter and key along non-overlapping five-bar scopes and
charges one scalar per change of state. That is the Viterbi path of a Markov
prior that stays with probability 1 - p and moves to each of the other K - 1
states with probability p / (K - 1), when the penalty is ln((1 - p)(K - 1) / p).
Here p is counted on the performed bar sequence of each training render, cut
into scopes exactly as inference cuts it, each scope carrying the meter and key
in effect at its first bar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.a2s.piano.schema_inference import scope_plan
from src.a2s.piano.tokenizer import KEY_TOKENS, METER_DEN_TOKENS, METER_NUM_TOKENS
from src.evaluation.kern_measures import _KEYSIG_RE, _METER_RE, parse_whole_piece

# The scorecard ranks every (numerator, denominator) token pair and every key token.
STATES = {"meter": len(METER_NUM_TOKENS) * len(METER_DEN_TOKENS), "key": len(KEY_TOKENS)}


def measure_states(kern_text: str) -> list[dict]:
    """Meter and key in effect in each measure.

    The parser records the state as each barline is read, so a `*M` or `*k[]`
    written just after a barline only appears one measure later; applying the
    interpretation rows that open each measure puts it back on its own measure.
    """
    blocks, state = parse_whole_piece(kern_text)
    effective = []
    for block, start in zip(blocks, state, strict=True):
        current = {"meter": start["meter"], "key": start["key"]}
        for raw in block[1:]:
            first = raw.split("\t", 1)[0].strip()
            if first.startswith("!"):
                continue
            if not first.startswith("*"):
                break
            if _KEYSIG_RE.match(first):
                current["key"] = first
            elif _METER_RE.match(first):
                current["meter"] = first
        effective.append(current)
    return effective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True,
                        help="Root that the manifest's kern_gt_path entries are relative to")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = json.loads(args.manifest.read_text())
    cache: dict[Path, list[dict]] = {}
    counts = {field: {"transitions": 0, "changes": 0, "renders_with_change": 0} for field in STATES}
    skipped = {"measure outside score": 0, "fewer than two scopes": 0, "state missing": 0}
    renders = 0
    for item in items:
        path = args.manifest_dir / item["kern_gt_path"]
        if path not in cache:
            cache[path] = measure_states(path.read_text())
        states = cache[path]
        bars = [measure["measure"] for measure in item["audio_measures"]]
        if not bars or max(bars) >= len(states):
            skipped["measure outside score"] += 1
            continue
        plan = scope_plan(item["id"], len(bars))
        if len(plan) < 2:
            skipped["fewer than two scopes"] += 1
            continue
        sequence = [states[bars[scope["start"]]] for scope in plan]
        if any(state[field] is None for state in sequence for field in STATES):
            skipped["state missing"] += 1
            continue
        renders += 1
        for field in STATES:
            values = [state[field] for state in sequence]
            changes = sum(a != b for a, b in zip(values, values[1:]))
            counts[field]["transitions"] += len(values) - 1
            counts[field]["changes"] += changes
            counts[field]["renders_with_change"] += changes > 0

    fields = {}
    for field, count in counts.items():
        rate = count["changes"] / count["transitions"]
        fields[field] = {
            **count,
            "states": STATES[field],
            "switch_rate": rate,
            "penalty": math.log((1 - rate) * (STATES[field] - 1) / rate),
        }
    report = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "scores": len(cache),
        "renders": renders,
        "skipped": skipped,
        "fields": fields,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "schema_switch_penalty.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"{renders} renders, {len(cache)} scores; skipped {skipped}",
        "",
        "| field | states K | scope pairs | changes | switch rate p | penalty ln((1-p)(K-1)/p) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for field, row in fields.items():
        lines.append(f"| {field} | {row['states']} | {row['transitions']} | {row['changes']} | "
                     f"{row['switch_rate']:.5f} | {row['penalty']:.3f} |")
    (args.out / "schema_switch_penalty.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
