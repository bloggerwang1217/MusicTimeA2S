# MusicTime-A2S

End-to-end piano audio-to-score transcription with a learned musical-time
coordinate: a phase model estimates each audio frame's position within the
notated bar, the score decoder is conditioned on that coordinate, and the same
coordinate supplies bar boundaries so whole pieces are transcribed without an
external beat tracker.

## Setup

```bash
git clone https://github.com/bloggerwang1217/MusicTimeA2S.git
cd MusicTimeA2S
git submodule update --init external/MV2H external/virtuosoNet external/efficient-musicdiff
poetry install
bash scripts/download_piano_soundfonts.sh      # piano soundfonts for Syn rendering
bash scripts/download_virtuosonet_weights.sh   # expressive-performance rendering weights
```

Compile MV2H once (`external/MV2H`, see its README) so that `external/MV2H/bin`
exists; the evaluation scripts call it from there.

### Checkpoints

The paper's checkpoints are released separately and go under `checkpoints/`:

| File | Model | Training seed |
|---|---|---|
| `checkpoints/full_seed42.pt`, `full_seed91.pt`, `full_seed1217.pt` | Full model | 42, 91, 1217 |
| `checkpoints/without_coordinate_seed42.pt`, `_seed91.pt`, `_seed1217.pt` | Without the coordinate (no phase model) | 42, 91, 1217 |
| `checkpoints/without_fourierpe_seed42.pt` | Phase supervision only, no Fourier PE | 42 |
| `checkpoints/without_durationpe_seed42.pt` | Audio-side Fourier PE only | 42 |
| `checkpoints/without_audiope_seed42.pt` | Score-side Fourier PE only | 42 |
| `checkpoints/hft/hft_maestro_v3_statedict.pt`, `parameter.json` | Frozen hFT-Transformer front end (MAESTRO-V3 release) | – |

`checkpoints/manifest.json` lists the SHA-256 of every file.

### Machine-local paths

Only the stages that involve external data or the compared systems need them.
Training, inference and self-segmented whole-piece transcription of our own
model need none.

- `configs/baselines.yaml` holds the locations of the compared systems; fill it
  in once. `asap_root` is the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset)
  checkout; `piano_a2s_repo` (with `piano_a2s_hparams` and `piano_a2s_checkpoint`
  under it) the [Piano-A2S fork](https://github.com/bloggerwang1217/piano-a2s)
  checkout on `main` with its own `env.sh` filled in; `beatthis_fold_split` the
  ASAP 8-fold split file inside the [Beat This!](https://github.com/CPJKU/beat_this)
  checkout; the two `*_conda_env` entries name the conda environments those
  repositories run in. The same file lists the four baseline runs under `runs`
  (`beatthis_asap`, `beatthis_syn`, `oracle_asap`, `oracle_syn`), selected by
  `src/baselines/run_asap102_omr.sh SYSTEM DATASET`; the replication scripts
  read `asap_root` and `piano_a2s_repo` from it as well.
- ASAP data preparation and reference building take the checkout on the command
  line: `--asap-root /path/to/asap-dataset`.

## Data

- **Syn**: `src/datasets/syn/prepare_syn.py` builds the manifests from the
  MuseSyn and HumSyn scores; audio is rendered by `src/audio/render_epr.py`
  (virtuosoNet performance rendering, then the soundfonts). The splits are the
  fixed lists in `src/datasets/syn/`.
- **ASAP**: `poetry run python src/datasets/asap/prepare_asap.py --asap-root /path/to/asap-dataset --split test`
  builds the test manifest from the 102-recording ASAP test split of Liu et al.;
  the 74-recording subset outside the front end's training data is
  `src/datasets/asap/asap102_hft_clean_74_recordings.txt`.

## Training

```bash
CONFIG=configs/piano_2gpu.yaml bash src/a2s/piano/train_ddp.sh
```

One config per model variant is in `configs/`; the seed-91 and seed-1217 runs
use the same files with only the seed changed.

## Inference

```bash
CHECKPOINT=checkpoints/full_seed42.pt CONFIG=configs/piano_2gpu.yaml \
    bash src/a2s/piano/run_inference.sh                 # pre-segmented excerpts
CHECKPOINT=checkpoints/full_seed42.pt CONFIG=configs/piano_2gpu.yaml \
    bash src/a2s/piano/run_inference_self_segmented.sh  # whole pieces, self-segmented
```

## Reproducing the paper's numbers

Two entry points; each stage refuses to overwrite an existing output directory.

```bash
bash src/evaluation/replicate_scores.sh STAGE [ARGS]   # decode and score (see its header for the stage order)
bash src/analysis/replicate_tables.sh [preseg|selfseg|all|agreement]   # Table 1, Table 2, the scorer-agreement footnote
poetry run python -m src.analysis.plot_fourierpe_interaction            # Fig. 2
```

The Piano-A2S row of Table 1 is scored by the fork's own pipeline; run its
`evaluate_slurm.sh` with `MV2H_TIMEOUT=300 MV2H_KEEP_ZERO=1` as described in
the header of `replicate_scores.sh`.

## License

MIT; third-party components keep their own licenses (see `LICENSE`).
