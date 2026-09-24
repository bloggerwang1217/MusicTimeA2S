"""Turn the hFT-Transformer release checkpoint into the state dict this repo loads.

The frozen front end is not redistributed here. Download the ISMIR 2023 release
of sony/hFT-Transformer and run this script on it:

    wget https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip
    poetry run python scripts/convert_hft_checkpoint.py checkpoint.zip

It writes checkpoints/hft/hft_maestro_v3_statedict.pt and checkpoints/hft/parameter.json.
The release pickles the whole nn.Module (model_016_003.pkl), which only loads
against the class names it was saved with; this resolves them to the vendored
copy in src/a2s/piano/foundation/hft_model.py and saves the plain state dict,
which is what configs/*.yaml point at (hft_state_dict / hft_parameter_json).
If checkpoints/manifest.json is present, the SHA-256 of both files is checked
against it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.a2s.piano.foundation import hft_model  # noqa: E402

PKL = "checkpoint/MAESTRO-V3/model_016_003.pkl"
PARAM = "checkpoint/MAESTRO-V3/parameter.json"
OUT_DIR = REPO / "checkpoints" / "hft"
OUT_PT = OUT_DIR / "hft_maestro_v3_statedict.pt"
OUT_JSON = OUT_DIR / "parameter.json"


class _Unpickler(pickle.Unpickler):
    """Resolve the release's `model.model_spec2midi.*` classes to the vendored module."""

    def find_class(self, module, name):
        if module == "model.model_spec2midi":
            return getattr(hft_model, name)
        if module == "torch.storage" and name == "_load_from_bytes":
            # The release was saved from a CUDA model; land its storages on CPU.
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_sha256(state: dict) -> str:
    """SHA-256 over the tensors themselves (name, dtype, shape, bytes), in name order."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path,
                        help="checkpoint.zip from the ismir2023 release, or the unzipped model_016_003.pkl")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if args.source.suffix == ".zip":
        with zipfile.ZipFile(args.source) as zf:
            pkl_bytes = zf.read(PKL)
            param_bytes = zf.read(PARAM)
    else:
        pkl_bytes = args.source.read_bytes()
        param_path = args.source.with_name("parameter.json")
        if not param_path.exists():
            sys.exit(f"parameter.json must sit next to {args.source}")
        param_bytes = param_path.read_bytes()

    model = _Unpickler(io.BytesIO(pkl_bytes)).load()
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = args.out_dir / OUT_PT.name
    out_json = args.out_dir / OUT_JSON.name
    torch.save(state, out_pt)
    out_json.write_bytes(param_bytes)
    print(f"{out_pt}: {len(state)} tensors, {sum(v.numel() for v in state.values()) / 1e6:.2f}M params")
    print(f"{out_json}: {json.loads(param_bytes)}")

    manifest = args.out_dir.parent / "manifest.json"
    if manifest.exists():
        entries = json.load(open(manifest))
        checks = (
            # The .pt container differs between torch versions; the tensors do not.
            ("hft/hft_maestro_v3_statedict.pt", "content_sha256", content_sha256(state)),
            ("hft/parameter.json", "sha256", sha256(out_json)),
        )
        for name, field, got in checks:
            want = entries.get(name, {}).get(field)
            status = "ok" if want == got else ("MISMATCH" if want else "not listed")
            print(f"{field} {status}: {name}")


if __name__ == "__main__":
    main()
