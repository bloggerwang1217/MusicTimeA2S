"""
Strip optimizer/scheduler state from a checkpoint for fine-tuning.

Produces a weights-only checkpoint that, when loaded via --resume,
forces a fresh optimizer init (load_checkpoint falls back on KeyError).

Usage:
  poetry run python scripts/strip_checkpoint.py checkpoints/<arm>/best.pt
  poetry run python scripts/strip_checkpoint.py path/to/ckpt.pt --out path/to/out.pt
"""

import argparse
from pathlib import Path

import torch


def strip(src: Path, dst: Path) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    slim = {
        "model_state_dict": ckpt["model_state_dict"],
        "config": ckpt["config"],
        "epoch": 0,
        "global_step": 0,
        "best_valid_loss": float("inf"),
        "epochs_without_improvement": 0,
    }
    torch.save(slim, dst)
    print(f"{src} → {dst}")
    print(f"  kept : model_state_dict ({len(slim['model_state_dict'])} tensors), config")
    print(f"  dropped: optimizer_state_dict, scheduler_state_dict")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <src_stem>_weights_only.pt)")
    args = ap.parse_args()

    dst = args.out or args.src.with_name(args.src.stem + "_finetuned.pt")
    strip(args.src, dst)
