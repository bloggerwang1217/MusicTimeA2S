#!/bin/bash
# Piano-A2S on annotated downbeats: the gray Piano-A2S row of the whole-piece
# table. decode runs the baseline on segments cut at the annotated downbeats;
# score builds the reference pairs, then the whole-piece MV2H and the OMR-NED
# of that decode. replicate_tables.sh compares the result with the other rows.
#   bash src/baselines/run_oracle_pianoa2s.sh DATASET STAGE
#   DATASET  syn | asap;  STAGE  decode | score
#   The run is oracle_<DATASET> in configs/baselines.yaml.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DATASET="${1:?Usage: run_oracle_pianoa2s.sh DATASET STAGE}"
STAGE="${2:?stage}"
case "$DATASET" in
    syn|asap) RUN="oracle_$DATASET" ;;
    *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac

OUT=$(poetry run python -c 'import sys,yaml; print(yaml.safe_load(open("configs/baselines.yaml"))["runs"][sys.argv[1]]["asap102_output_root"])' "$RUN")
MAPPING="$OUT/reference/mapping.jsonl"
PAIRS="$OUT/score_pairs"
export WORKERS="${WORKERS:-4}"

if [[ "${DRY_RUN:-0}" == 1 && "$STAGE" != decode ]]; then
    echo "DRY_RUN is supported for decode; score consumes its completed outputs." >&2
    exit 2
fi

case "$STAGE" in
    decode)
        bash src/baselines/run_asap102_omr.sh oracle_pianoa2s "$DATASET"
        ;;
    score)
        poetry run python -m src.evaluation.build_omr_ned_pairs \
            --prediction-manifest "$OUT/prediction-manifest-blank-fill.jsonl" \
            --mapping "$MAPPING" --out-dir "$PAIRS"
        MAPPING="$MAPPING" ARMS="oracle=$PAIRS/pairs.jsonl" \
            OUTPUT_DIR="$OUT/whole_score" RECORDING_TIMEOUT= \
            bash src/evaluation/slurm_eval_whole_score.sh --local
        PAIRS="$PAIRS/pairs_ready.jsonl" OUTPUT_DIR="$OUT/omr_ned" \
            bash src/evaluation/slurm_omr_ned.sh --local
        poetry run python -m src.evaluation.summarize_omr_ned \
            --pairs "$PAIRS/pairs.jsonl" --results "$OUT/omr_ned/results.jsonl" \
            --out "$OUT/omr_summary.json"
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
