#!/bin/bash
# Decode and score everything the two result tables are computed from. Stages
# that write refuse an existing output directory; CONTINUE=1 resumes one that a
# killed run left behind.
#
# Usage (from the project root): bash src/evaluation/replicate_scores.sh STAGE [ARGS]
#   ARM      full | without_fourierpe | without_coordinate | without_durationpe |
#            without_audiope; full and without_coordinate also as ARM_seed91 /
#            ARM_seed1217 (a seed arm reads the base arm's config, only the
#            checkpoint differs)
#   DATASET  asap | syn | asap_midi19
#   CASCADE  beatthis_pianoa2s
#   BACKEND=slurm (default) submits arrays; BACKEND=local runs on this machine
#   with WORKERS processes.
#   METER_SWITCH_PENALTY / KEY_SWITCH_PENALTY  self-segmented switch penalties
#   (default 8.3 / 6.3); the self-segmented output names carry both values.
#   PRESEG_PROTOCOL  which pre-segmented windows and reference the preseg stages
#   use; required by them. piano_a2s: Piano-A2S's own five-bar windows and
#   reference MIDI, its MV2H files as its row (Table 1). native: our stride-five
#   windows and ground-truth score MIDI, Ours arms only (Fig. 2).
#   PRESEG_ROOT / GT_SCORE_MIDI_ROOT / PRED_DIR / OUTPUT_DIR select prepared
#   windows, references, and existing score outputs.
#   PIANO_A2S_ROOT / ASAP_ROOT  the Piano-A2S and asap-dataset checkouts; the
#   stages that need them read piano_a2s_repo / asap_root from
#   configs/baselines.yaml. PA2S_ASAP, PA2S_ASAP_TARGET and
#   PA2S_SYN name Piano-A2S's run directories directly.
#   BOUNDARIES  self-segmented bar boundaries: predicted (default, forward phase
#   crossings), annotated (annotated measure starts, the model's own phase) or
#   annotated_phase (annotated measure starts and annotated phase); the output
#   names carry it (forward400 / annotated / annotatedphase).
#
# Order of the whole reproduction:
#   preseg, once per protocol (PRESEG_PROTOCOL=piano_a2s for Table 1, native for Fig. 2)
#     prepare-preseg DATASET      the protocol's windows and reference MIDI.
#         piano_a2s: ASAP indexes the Piano-A2S run that its own repository
#         produced (chunks, inference, MV2H on $PA2S_ASAP); Syn drives that
#         repository through src/evaluation/syn/run_preseg.sh.
#         native: src/evaluation/prepare_preseg.py on our manifests.
#     decode-preseg DATASET ARM   the existing Ours inference entry on those windows
#     score-preseg DATASET ARM    MV2H of that arm against the protocol's reference
#     score-preseg DATASET piano_a2s   piano_a2s only: Piano-A2S's own MV2H files
#         (ASAP checks they exist; Syn runs its evaluate worker). The native
#         protocol has no Piano-A2S row.
#   selfseg
#     ASAP  decode-asap-selfseg full; decode-asap-selfseg without_fourierpe
#           bash src/baselines/run_asap102_omr.sh CASCADE asap    (writes the baseline OMR-NED)
#           score-selfseg asap full; score-selfseg asap without_fourierpe
#     Syn   decode-syn-selfseg
#           bash src/baselines/run_asap102_omr.sh CASCADE syn
#           score-selfseg syn full
#           cascade-omr-ned CASCADE
#     score-whole DATASET [ARM]   single-path MV2H on the complete scores
#   bash src/analysis/replicate_tables.sh    both tables from the finished scores
#
# Fixed inputs, read here and never written: the ASAP and Syn test manifests,
# the existing prediction scores and native MIDI, and the frozen selfseg
# reference windows of both test sets. The ASAP selfseg reference windows
# ($ASAP_WHOLE_GT_MIDI) are the output of
#   poetry run python -m src.evaluation.asap.eval_asap_native_gt \
#       --native-action build-reference --asap-root $ASAP_ROOT --gt-score-midi-root $ASAP_WHOLE_GT_MIDI
# and the Syn ones ($SYN_WHOLE_GT_MIDI) of
#   poetry run python -m src.evaluation.syn.build_native_gt_score_midi \
#       --manifest-dir $SYN --out-dir $SYN_WHOLE_GT_MIDI
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV POETRY_ACTIVE
SELF="$PROJECT_ROOT/src/evaluation/replicate_scores.sh"

ASAP="${ASAP_DATA:-data/experiments/asap102}"
SYN="${SYN_DATA:-data/experiments/syn}"
MIDI19=data/experiments/asap_midi_only_19
ASAP_WHOLE_GT_MIDI=$ASAP/whole_reference
SYN_WHOLE_GT_MIDI=$SYN/native_reference_syn_test_ydp
S74=src/datasets/asap/asap102_hft_clean_74_recordings.txt
# Piano-A2S protocol (Table 1): its ASAP run lives in its own workspace; the Syn
# run is driven by run_preseg.sh into $SYN_PRESEG_PA2S and the same workspace.
path_from_config() {  # KEY: its value in the baseline config; exits while it is a placeholder
    local config=configs/baselines.yaml value
    value=$(poetry run python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$config" "$1")
    if [[ -z "$value" || "$value" == /path/to/* ]]; then
        echo "$config: $1 is not filled in" >&2; exit 1
    fi
    printf '%s\n' "$value"
}
piano_a2s_root() { PIANO_A2S_ROOT="${PIANO_A2S_ROOT:-$(path_from_config piano_a2s_repo)}"; }
pa2s_runs() {  # sets PA2S_ASAP, PA2S_ASAP_TARGET, PA2S_SYN from the Piano-A2S workspace
    if [[ -z "${PA2S_ASAP:-}" || -z "${PA2S_ASAP_TARGET:-}" || -z "${PA2S_SYN:-}" ]]; then
        piano_a2s_root
    fi
    PA2S_ASAP="${PA2S_ASAP:-$PIANO_A2S_ROOT/workspace/1234/asap102_eval.epr_ft.gt/results}"
    PA2S_ASAP_TARGET="${PA2S_ASAP_TARGET:-$PIANO_A2S_ROOT/workspace/feature.asap102/test/target}"
    PA2S_SYN="${PA2S_SYN:-$PIANO_A2S_ROOT/workspace/1234/syn_test540_eval.epr_ft.gt/results}"
}
ZENG_GROUNDING=$ASAP/piano_a2s_gt_grounding.jsonl
ZENG_MANIFEST=$ASAP/test_manifest_zenggt21.json
SYN_PRESEG_PA2S="${SYN_PRESEG_PA2S:-$SYN/syn_table1_preseg}"
MV2H_BIN=$PROJECT_ROOT/external/MV2H/bin
BACKEND="${BACKEND:-slurm}"
export CHUNK_TIMEOUT=300 CHUNKS_PER_JOB="${CHUNKS_PER_JOB:-32}" MAX_CONCURRENT="${MAX_CONCURRENT:-16}" \
       WORKERS="${WORKERS:-16}"
stamp() { date '+%F %T'; }
METER_SWITCH_PENALTY="${METER_SWITCH_PENALTY:-8.3}"
KEY_SWITCH_PENALTY="${KEY_SWITCH_PENALTY:-6.3}"
BOUNDARIES="${BOUNDARIES:-predicted}"
case "$BOUNDARIES" in
    predicted)       BAR_BOUNDARIES=predicted; COORDINATE_INTERVENTION=none; BOUNDARY_TAG=forward400 ;;
    annotated)       BAR_BOUNDARIES=annotated; COORDINATE_INTERVENTION=none; BOUNDARY_TAG=annotated ;;
    annotated_phase) BAR_BOUNDARIES=annotated; COORDINATE_INTERVENTION=oracle; BOUNDARY_TAG=annotatedphase ;;
    *) echo "unknown BOUNDARIES: $BOUNDARIES" >&2; exit 1 ;;
esac
PROTO="${BOUNDARY_TAG}_meter${METER_SWITCH_PENALTY}_key${KEY_SWITCH_PENALTY}"

arm() {
    case "$1" in
        full)
            TAG=full_off
            CHECKPOINT=checkpoints/full_seed42.pt
            CONFIG=configs/piano_2gpu.yaml ;;
        without_fourierpe)
            TAG=without_fourierpe_off
            CHECKPOINT=checkpoints/without_fourierpe_seed42.pt
            CONFIG=configs/piano_2gpu_without_fourierpe.yaml ;;
        without_coordinate)
            TAG=without_coordinate_off
            CHECKPOINT=checkpoints/without_coordinate_seed42.pt
            CONFIG=configs/piano_2gpu_without_coordinate.yaml ;;
        # Seed replication of the two Table 1 arms (seed 42 is the unsuffixed one).
        # The seed runs were trained from the base config with only the seed
        # changed, so inference reads that config.
        without_coordinate_seed91)
            TAG=without_coordinate_seed91_off
            CHECKPOINT=checkpoints/without_coordinate_seed91.pt
            CONFIG=configs/piano_2gpu_without_coordinate.yaml ;;
        without_coordinate_seed1217)
            TAG=without_coordinate_seed1217_off
            CHECKPOINT=checkpoints/without_coordinate_seed1217.pt
            CONFIG=configs/piano_2gpu_without_coordinate.yaml ;;
        full_seed91)
            TAG=full_seed91_off
            CHECKPOINT=checkpoints/full_seed91.pt
            CONFIG=configs/piano_2gpu.yaml ;;
        full_seed1217)
            TAG=full_seed1217_off
            CHECKPOINT=checkpoints/full_seed1217.pt
            CONFIG=configs/piano_2gpu.yaml ;;
        # Audio-side Fourier PE only: the duration side is dropped from the
        # decoder's cross-attention terms.
        without_durationpe)
            TAG=without_durationpe_off
            CHECKPOINT=checkpoints/without_durationpe_seed42.pt
            CONFIG=configs/piano_2gpu_without_durationpe.yaml ;;
        # Score-side Fourier PE only: the audio side is dropped from the
        # decoder's cross-attention key and value terms.
        without_audiope)
            TAG=without_audiope_off
            CHECKPOINT=checkpoints/without_audiope_seed42.pt
            CONFIG=configs/piano_2gpu_without_audiope.yaml ;;
        *) echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

# How OMR-NED finds each test set's reference scores.
dataset() {
    case "$1" in
        asap)
            ASAP_ROOT="${ASAP_ROOT:-$(path_from_config asap_root)}"
            PAIR_SOURCE=(--asap-root "$ASAP_ROOT") ;;
        syn) PAIR_SOURCE=(--mapping "$SYN_WHOLE_GT_MIDI/mapping.jsonl") ;;
        asap_midi19) PAIR_SOURCE=(--mapping "$MIDI19/native_reference/mapping.jsonl") ;;
        *) echo "unknown dataset: $1" >&2; exit 1 ;;
    esac
}

cascade() {  # DATASET CASCADE
    case "$1:$2" in
        asap:beatthis_pianoa2s) SOURCE=data/experiments/beatthis_pianoa2s_asap102_holdout/prediction-manifest-blank-fill.jsonl ;;
        syn:beatthis_pianoa2s) SOURCE=data/experiments/beatthis_pianoa2s_syn_test_ydp/prediction-manifest-blank-fill.jsonl ;;
        asap_midi19:beatthis_pianoa2s) SOURCE=data/experiments/beatthis_pianoa2s_asap_midi19/prediction-manifest-blank-fill.jsonl ;;
        *) echo "unknown dataset or cascade: $1 $2" >&2; exit 1 ;;
    esac
}

fresh() {
    [[ "${DRY_RUN:-0}" == 1 ]] && return 0
    if [[ -e "$1" && "${CONTINUE:-0}" != 1 ]]; then
        echo "refusing to overwrite $1" >&2; exit 1
    fi
}

decode_selfseg() {  # MANIFEST_DIR OUTPUT_DIR
    fresh "$2"
    GPU="${GPU:-0}" CHECKPOINT="$CHECKPOINT" CONFIG="$CONFIG" \
    MANIFEST_DIR="$1" MANIFEST="${MANIFEST:-$1/test_manifest.json}" OUTPUT_DIR="$2" \
    PIECE_SCOPE=five_bar KEY_SWITCH_PENALTY="$KEY_SWITCH_PENALTY" METER_SWITCH_PENALTY="$METER_SWITCH_PENALTY" \
    BAR_BOUNDARIES="$BAR_BOUNDARIES" COORDINATE_INTERVENTION="$COORDINATE_INTERVENTION" \
        bash src/a2s/piano/run_inference_self_segmented.sh
}

preseg_dataset() {
    case "${PRESEG_PROTOCOL:-}" in
        piano_a2s|native) ;;
        *) echo "set PRESEG_PROTOCOL=piano_a2s (Table 1) or native (Fig. 2)" >&2; exit 1 ;;
    esac
    case "$1" in
        asap)
            DATA="$ASAP"
            GT_SCORE_MIDI="${GT_SCORE_MIDI_ROOT:-$ASAP/native_reference}"
            # Table 1 keeps to the recordings outside Piano-A2S's training data;
            # the figure uses every recording outside the front end's.
            if [[ "$PRESEG_PROTOCOL" == native ]]; then
                SELECTION=(--recording-list "${RECORDING_LIST:-$S74}")
                PRESEG="${PRESEG_ROOT:-$ASAP/fig2_preseg}"
            else
                SELECTION=(--recording-list "${RECORDING_LIST:-src/datasets/asap/asap102_preseg_unexposed_recordings.txt}")
                PRESEG="${PRESEG_ROOT:-$ASAP/table1_preseg}"
            fi ;;
        syn)
            DATA="$SYN"
            GT_SCORE_MIDI="${GT_SCORE_MIDI_ROOT:-$SYN/native_reference_syn_test_ydp_grid}"
            SELECTION=()
            [[ -z "${RECORDING_LIST:-}" ]] || SELECTION=(--recording-list "$RECORDING_LIST")
            PRESEG="${PRESEG_ROOT:-$SYN/table1_preseg}" ;;
        *) echo "unknown preseg dataset: $1" >&2; exit 1 ;;
    esac
}

syn_preseg() {  # STAGE: one stage of the Piano-A2S-protocol Syn driver
    PRESEG="$SYN_PRESEG_PA2S" bash src/evaluation/syn/run_preseg.sh "$@"
}

decode_preseg() {  # MANIFEST_DIR MANIFEST GROUNDING OUTPUT_DIR; reads CHECKPOINT/CONFIG from arm()
    fresh "$4"
    GPU="${GPU:-0}" CHECKPOINT="$CHECKPOINT" CONFIG="$CONFIG" \
    MANIFEST_DIR="$1" MANIFEST="$2" GROUNDING="$3" OUTPUT_DIR="$4" \
        bash src/a2s/piano/run_inference.sh
}

omr_ned() {  # OUT_DIR: musicdiff over the ready pairs, then the successful-only summary
    local summarize=(poetry run python -m src.evaluation.summarize_omr_ned --pairs "$1/pairs.jsonl"
                     --results "$1/omr_ned/results.jsonl" --out "$1/summary.json")
    if [[ "$BACKEND" == local ]]; then
        PAIRS="$1/pairs_ready.jsonl" OUTPUT_DIR="$1/omr_ned" bash src/evaluation/slurm_omr_ned.sh --local
        "${summarize[@]}"
    else
        PAIRS="$1/pairs_ready.jsonl" OUTPUT_DIR="$1/omr_ned" bash src/evaluation/slurm_omr_ned.sh
        echo "once the musicdiff array finishes: ${summarize[*]}"
    fi
}

whole_layout() {  # DATASET; reads TAG, so call arm() first
    # The headline arm keeps the unsuffixed output names it has always written.
    local suffix=""
    [[ "$TAG" == full_off ]] || suffix="_$TAG"
    case "$1" in
        asap)
            WHOLE_MAPPING=$ASAP_WHOLE_GT_MIDI/mapping.jsonl
            WHOLE_OURS=$ASAP/omr_ned_selfseg_${TAG}_${PROTO}/pairs.jsonl
            WHOLE_ROOT=$ASAP/whole_score_${PROTO}${suffix} ;;
        syn)
            WHOLE_MAPPING=$SYN_WHOLE_GT_MIDI/mapping.jsonl
            WHOLE_OURS=$SYN/omr_ned_syn_test_${TAG}_${PROTO}_ydp/pairs.jsonl
            WHOLE_ROOT=$SYN/whole_score_${PROTO}${suffix}_ydp ;;
        asap_midi19)
            WHOLE_MAPPING=$MIDI19/native_reference/mapping.jsonl
            WHOLE_OURS=$MIDI19/omr_ned_selfseg_${TAG}_${PROTO}/pairs.jsonl
            WHOLE_ROOT=$MIDI19/whole_score_${PROTO}${suffix} ;;
        *) echo "unknown dataset: $1" >&2; exit 1 ;;
    esac
}

stage="${1:?Usage: bash src/evaluation/replicate_scores.sh STAGE [ARGS]}"
shift
case "$stage" in
prepare-preseg)
    preseg_dataset "${1:?dataset}"
    if [[ "$PRESEG_PROTOCOL" == native ]]; then
        poetry run python -m src.evaluation.prepare_preseg \
            --manifest "${MANIFEST:-$DATA/test_manifest.json}" --gt-score-midi-root "$GT_SCORE_MIDI" \
            --output-dir "${OUTPUT_DIR:-$PRESEG/inputs}" "${SELECTION[@]}"
    elif [[ "$1" == asap ]]; then
        # The Piano-A2S ASAP run (its chunk builder, inference, and MV2H) is made
        # in its own repository; this freezes its targets and reference MIDI
        # into a grounding and keeps the recordings it covers.
        fresh "$ZENG_GROUNDING"
        piano_a2s_root
        pa2s_runs
        poetry run python -m src.evaluation.asap.build_piano_a2s_gt_grounding \
            --target-dir "$PA2S_ASAP_TARGET" --results-dir "$PA2S_ASAP" \
            --our-manifest "$ASAP/test_manifest.json" --piano-a2s-repo "$PIANO_A2S_ROOT" \
            --require-gt-midi --output "$ZENG_GROUNDING"
        poetry run python -m src.evaluation.asap.select_grounded_recordings \
            --manifest "$ASAP/test_manifest.json" --grounding "$ZENG_GROUNDING" --output "$ZENG_MANIFEST"
    else
        for step in layout features spectrograms pa2s-infer pa2s-mv2h grounding-decode grounding; do
            syn_preseg "$step"
        done
    fi ;;
decode-preseg)
    preseg_dataset "${1:?dataset}"
    arm "${2:?arm}"
    if [[ "$PRESEG_PROTOCOL" == native ]]; then
        decode_preseg "$DATA" "$PRESEG/inputs/manifest.json" "$PRESEG/inputs/grounding.jsonl" \
            "${OUTPUT_DIR:-$PRESEG/kern_$TAG}"
    elif [[ "$1" == asap ]]; then
        decode_preseg "$ASAP" "$ZENG_MANIFEST" "$ZENG_GROUNDING" \
            "${OUTPUT_DIR:-$ASAP/test_kern_pred_piano_a2s_bar5_asap102_zenggt21_$TAG}"
    else
        syn_preseg select-inputs decode
        decode_preseg "$SYN" "$SYN_PRESEG_PA2S/test_inputs/manifest_decode.json" \
            "$SYN_PRESEG_PA2S/test_inputs/grounding_decode.jsonl" "${OUTPUT_DIR:-$SYN_PRESEG_PA2S/kern_$TAG}"
    fi ;;
score-preseg)
    preseg_dataset "${1:?dataset}"
    local_flag=()
    [[ "$BACKEND" == local ]] && local_flag=(--local)
    if [[ "${2:?system}" == piano_a2s ]]; then
        if [[ "$PRESEG_PROTOCOL" != piano_a2s ]]; then
            echo "the native protocol scores only our arms; Piano-A2S is compared on its own windows and reference (PRESEG_PROTOCOL=piano_a2s)" >&2
            exit 1
        fi
        if [[ "$1" == asap ]]; then
            pa2s_runs
            [[ -d "$PA2S_ASAP/mv2h" ]] || { echo "no Piano-A2S MV2H files at $PA2S_ASAP/mv2h: run its evaluate_slurm.sh with MV2H_TIMEOUT=300 MV2H_KEEP_ZERO=1" >&2; exit 1; }
            echo "Piano-A2S row: $(ls "$PA2S_ASAP/mv2h" | wc -l) MV2H files at $PA2S_ASAP/mv2h"
        else
            syn_preseg pa2s-mv2h
        fi
        exit 0
    fi
    arm "$2"
    if [[ "$PRESEG_PROTOCOL" == native ]]; then
        out="${OUTPUT_DIR:-$PRESEG/eval_$TAG}"
        fresh "$out"
        PRED_DIR="${PRED_DIR:-$PRESEG/kern_$TAG}" GROUNDING="$PRESEG/inputs/grounding.jsonl" \
        GT_SCORE_MIDI_ROOT="$GT_SCORE_MIDI" OUTPUT_DIR="$out" \
            bash src/evaluation/slurm_eval_window.sh "${local_flag[@]}"
    elif [[ "$1" == asap ]]; then
        pa2s_runs
        out="${OUTPUT_DIR:-$ASAP/eval_asap102_zenggt_all21_${TAG}_t300}"
        fresh "$out"
        PRED_DIR="${PRED_DIR:-$ASAP/test_kern_pred_piano_a2s_bar5_asap102_zenggt21_$TAG}" \
        GROUNDING="$ZENG_GROUNDING" REFERENCE_ROOT="$PA2S_ASAP" OUTPUT_DIR="$out" \
            bash src/evaluation/slurm_eval_window_piano_a2s.sh "${local_flag[@]}"
    else
        syn_preseg select-inputs score
        pa2s_runs
        out="${OUTPUT_DIR:-$SYN_PRESEG_PA2S/eval_$TAG}"
        fresh "$out"
        PRED_DIR="${PRED_DIR:-$SYN_PRESEG_PA2S/kern_$TAG}" \
        GROUNDING="$SYN_PRESEG_PA2S/test_inputs/grounding_score.jsonl" REFERENCE_ROOT="$PA2S_SYN" OUTPUT_DIR="$out" \
            bash src/evaluation/slurm_eval_window_piano_a2s.sh "${local_flag[@]}"
    fi ;;
decode-asap-selfseg)
    arm "${1:?arm}"
    decode_selfseg "$ASAP" "${OUTPUT_DIR:-$ASAP/test_kern_pred_self_segmented_${TAG}_${PROTO}}" ;;
decode-asap-midi19-selfseg)
    arm full
    decode_selfseg "$MIDI19" "${OUTPUT_DIR:-$MIDI19/test_kern_pred_self_segmented_${TAG}_${PROTO}}" ;;
decode-syn-selfseg)
    arm "${1:-full}"
    decode_selfseg "$SYN" "${OUTPUT_DIR:-$SYN/test_kern_pred_self_segmented_${TAG}_${PROTO}_ydp}" ;;
score-selfseg)
    # Whole-piece OMR-NED of a self-segmented decode against the reference
    # scores. Whole-score MV2H reads the pairs this writes.
    dataset "${1:?dataset}"
    arm "${2:?arm}"
    if [[ "$1" == asap ]]; then
        pred=$ASAP/test_kern_pred_self_segmented_${TAG}_${PROTO}; omr=$ASAP/omr_ned_selfseg_${TAG}_${PROTO}
    elif [[ "$1" == asap_midi19 ]]; then
        pred=$MIDI19/test_kern_pred_self_segmented_${TAG}_${PROTO}
        omr=$MIDI19/omr_ned_selfseg_${TAG}_${PROTO}
    else
        pred=$SYN/test_kern_pred_self_segmented_${TAG}_${PROTO}_ydp; omr=$SYN/omr_ned_syn_test_${TAG}_${PROTO}_ydp
    fi
    omr="${OMR_DIR:-$omr}"
    fresh "$omr"
    mkdir -p "$omr"
    [[ -f "$omr/pairs_ready.jsonl" ]] || poetry run python -m src.evaluation.build_omr_ned_pairs \
        --pred-kern-dir "$pred" --out-dir "$omr" "${PAIR_SOURCE[@]}" --pre-clean none --workers 4
    omr_ned "$omr" ;;
cascade-omr-ned)
    # Syn only: on ASAP the baseline runner itself writes the cascade OMR-NED.
    cascade syn "${1:?cascade}"
    out="${OMR_DIR:-$SYN/omr_ned_syn_test_${1}_ydp}"
    fresh "$out"
    [[ -f "$out/pairs_ready.jsonl" ]] || poetry run python -m src.evaluation.build_omr_ned_pairs \
        --prediction-manifest "$SOURCE" --mapping "$SYN_WHOLE_GT_MIDI/mapping.jsonl" --out-dir "$out"
    omr_ned "$out" ;;
score-whole)
    scope="${1:?dataset}"
    arm "${2:-full}"
    whole_layout "$scope"
    mapping="${MAPPING:-$WHOLE_MAPPING}"
    out="${OUTPUT_DIR:-$WHOLE_ROOT}"
    pair_specs=("ours=$WHOLE_OURS")
    for system in beatthis_pianoa2s; do
        cascade "$scope" "$system"
        pairs="$out/pairs_$system"
        if [[ ! -f "$pairs/pairs.jsonl" ]]; then
            poetry run python -m src.evaluation.build_omr_ned_pairs \
                --prediction-manifest "$SOURCE" --mapping "$mapping" --out-dir "$pairs"
        fi
        pair_specs+=("bt=$pairs/pairs.jsonl")
    done
    mode=()
    [[ "$BACKEND" == local ]] && mode=(--local)
    MAPPING="$mapping" ARMS="${pair_specs[*]}" OUTPUT_DIR="$out" \
        bash src/evaluation/slurm_eval_whole_score.sh "${mode[@]}" ;;
*) echo "unknown stage: $stage" >&2; exit 1 ;;
esac
