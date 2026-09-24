"""Piano-A2S's own spectrogram pass over a built feature folder.

Runs inside the Piano-A2S environment from its repository root, fed on stdin
so its modules resolve exactly as in its own scripts:

    python - FEATURE_FOLDER < src/evaluation/syn/piano_a2s_spectrograms.py

The processor is constructed without __init__, which skips the ASAP folder
scan this pass does not read.
"""
import sys

from hyperpyyaml import load_hyperpyyaml
from datasets.asap import ProcessASAP

feature_folder = sys.argv[1]
with open('hparams/finetune.yaml') as handle:
    hparams = load_hyperpyyaml(handle, {})
hparams['feature_folder'] = feature_folder
process = ProcessASAP.__new__(ProcessASAP)
process.hparams = hparams
process.feature_folder = feature_folder
process._prepare_spectrograms()
