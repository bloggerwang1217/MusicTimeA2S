#!/bin/bash
# Table 1 pools common scored windows; Table 2 averages whole recordings.
# Both use paired work-cluster intervals; OMR-NED also reports corpus totals.
#
# Usage (from the project root): bash src/analysis/replicate_tables.sh [preseg|selfseg|all|agreement]
#   preseg           Table 1 only; selfseg: Table 2 only; all: both
#   agreement        the Metrics footnote: single-path versus official MV2H on
#                    the Table 1 windows (WORKERS processes; slow on Syn)
#   TABLES           output directory, refused if it exists
#   PA2S_ORACLE_ASAP / PA2S_ORACLE_SYN  the Piano-A2S runs on annotated downbeats
#                    (src/baselines/run_oracle_pianoa2s.sh) for the gray Table 2 row
#   ASAP_PRESEG_CSV  Table 1 ASAP scores per arm (%s is the arm tag), written by
#                    replicate_scores.sh score-preseg under PRESEG_PROTOCOL=piano_a2s
#   SYN_PRESEG_PA2S  the Syn directory of that protocol
#   PIANO_A2S_ROOT   the Piano-A2S checkout whose workspace holds its MV2H files;
#                    preseg and agreement read it from piano_a2s_repo in
#                    configs/baselines.yaml unless PA2S_ASAP /
#                    PA2S_SYN name its result directories directly
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV POETRY_ACTIVE

ASAP=data/experiments/asap102
SYN=data/experiments/syn
SYN_REF=$SYN/native_reference_syn_test_ydp
# Table 1 is scored on Piano-A2S's own windows and reference MIDI; its row is
# its own MV2H files. Our stride-five windows serve Fig. 2 only.
path_from_config() {  # KEY: its value in the baseline config; exits while it is a placeholder
    local config=configs/baselines.yaml value
    value=$(poetry run python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$config" "$1")
    if [[ -z "$value" || "$value" == /path/to/* ]]; then
        echo "$config: $1 is not filled in" >&2; exit 1
    fi
    printf '%s\n' "$value"
}
piano_a2s_root() { PIANO_A2S_ROOT="${PIANO_A2S_ROOT:-$(path_from_config piano_a2s_repo)}"; }
pa2s_runs() {  # sets PA2S_ASAP and PA2S_SYN from the Piano-A2S workspace
    if [[ -z "${PA2S_ASAP:-}" || -z "${PA2S_SYN:-}" ]]; then
        piano_a2s_root
    fi
    PA2S_ASAP="${PA2S_ASAP:-$PIANO_A2S_ROOT/workspace/1234/asap102_eval.epr_ft.gt/results}"
    PA2S_SYN="${PA2S_SYN:-$PIANO_A2S_ROOT/workspace/1234/syn_test540_eval.epr_ft.gt/results}"
}
ASAP_PRESEG_CSV="${ASAP_PRESEG_CSV:-$ASAP/eval_asap102_zenggt_all21_%s_t300/eval_asap.csv}"
SYN_PRESEG_PA2S="${SYN_PRESEG_PA2S:-$SYN/syn_table1_preseg}"
# Table 2 ASAP: the ASAP-102 recordings outside the hFT front end's training data.
S74=src/datasets/asap/asap102_hft_clean_74_recordings.txt
PA2S_ORACLE_ASAP="${PA2S_ORACLE_ASAP:-data/experiments/oracle_downbeats_pianoa2s_asap74}"
PA2S_ORACLE_SYN="${PA2S_ORACLE_SYN:-data/experiments/oracle_downbeats_pianoa2s_syn_test_ydp}"
BT_OMR_ASAP="${BT_OMR_ASAP:-data/experiments/beatthis_pianoa2s_asap102_holdout/omr_ned_blank_fill/output.csv}"
BT_OMR_SYN="${BT_OMR_SYN:-$SYN/omr_ned_syn_test_beatthis_pianoa2s_ydp}"
TABLES="${TABLES:-data/experiments/paper_tables_native}"
stamp() { date '+%F %T'; }

pwb() { poetry run python -m src.analysis.paired_work_bootstrap --windows intersection "$@"; }

case "${1:-all}" in preseg|selfseg|all|agreement) ;; *) echo "expected preseg, selfseg, all or agreement" >&2; exit 1 ;; esac
[[ ! -e "$TABLES" ]] || { echo "refusing to overwrite $TABLES" >&2; exit 1; }
asap_csv() { printf "$ASAP_PRESEG_CSV" "$1"; }
syn_csv() { printf '%s/eval_%s/eval_asap.csv' "$SYN_PRESEG_PA2S" "$1"; }
table1_inputs() {  # DATASET: sets csv, pa2s, pa2s_results, windows
    pa2s_runs
    if [[ "$1" == asap ]]; then
        csv=asap_csv
        pa2s_results="$PA2S_ASAP"
        windows=(--grounding "$ASAP/piano_a2s_gt_grounding.jsonl"
                 --work-list src/datasets/asap/asap102_preseg_unexposed_pieces.txt)
    else
        csv=syn_csv
        pa2s_results="$PA2S_SYN"
        windows=(--grounding "$SYN_PRESEG_PA2S/test_inputs/grounding_score.jsonl"
                 --recording-list "$SYN_PRESEG_PA2S/test_inputs/recordings.txt"
                 --manifest "$SYN_PRESEG_PA2S/test_inputs/manifest_decode.json")
    fi
    pa2s="$pa2s_results/mv2h"
}
mkdir -p "$TABLES"
if [[ "${1:-all}" == agreement ]]; then
    # The Metrics footnote: the single-path scorer against the official
    # multi-path one, per window, on the Table 1 intersection. Each system's
    # pair is converted with the MV2H copy its official score came from.
    out="$TABLES/preseg_agreement"
    piano_a2s_root
    for dataset in asap syn; do
        table1_inputs "$dataset"
        echo "[$(stamp)] agreement, $dataset"
        equiv() { poetry run python -m src.analysis.preseg_equivalence "$1" --out "$out/$dataset" "${@:2}"; }
        equiv prepare "${windows[@]}" --piano-a2s-results "$pa2s_results" \
            --piano-a2s-mv2h-bin "$PIANO_A2S_ROOT/MV2H/bin" \
            --system "Piano-A2S=$pa2s" \
            --system "Ours seed42=$($csv full_off)" --system "Ours seed91=$($csv full_seed91_off)" \
            --system "Ours seed1217=$($csv full_seed1217_off)" \
            --system "w/o coordinate seed42=$($csv without_coordinate_off)" \
            --system "w/o coordinate seed91=$($csv without_coordinate_seed91_off)" \
            --system "w/o coordinate seed1217=$($csv without_coordinate_seed1217_off)"
        equiv run --workers "${WORKERS:-8}"
        equiv collect
    done
    poetry run python -m src.analysis.mv2h_agreement \
        --dataset "syn=$out/syn" --dataset "asap=$out/asap" \
        --row "Piano-A2S=Piano-A2S" --row "Ours=Ours seed42,Ours seed91,Ours seed1217" \
        --row "w/o coordinate=w/o coordinate seed42,w/o coordinate seed91,w/o coordinate seed1217" \
        --out "$out/agreement.json"
    echo "[$(stamp)] done: $out"
    exit 0
fi
[[ "${1:-all}" == selfseg ]] || for dataset in asap syn; do
    table1_inputs "$dataset"
    systems=(
        # Only the systems the table reports. A window enters the intersection
        # when all three, with every seed, have scored it. The two Ours rows
        # carry three training seeds each; Piano-A2S is one released checkpoint.
        "Piano-A2S=$pa2s"
        "w/o coordinate=$($csv without_coordinate_off),$($csv without_coordinate_seed91_off),$($csv without_coordinate_seed1217_off)"
        "Ours=$($csv full_off),$($csv full_seed91_off),$($csv full_seed1217_off)"
    )
    for candidate in 'ours|Ours' 'wo_coordinate|w/o coordinate'; do
        slug="${candidate%%|*}"; name="${candidate#*|}"
        args=()
        for system in "${systems[@]}"; do
            args+=(--system "$system")
            [[ "${system%%=*}" == "$name" ]] || args+=(--reference "${system%%=*}")
        done
        pwb "${windows[@]}" \
            --candidate "$name" --n-boot 10000 --seed 0 "${args[@]}" \
            --out "$TABLES/preseg_$dataset/$slug"
    done
done
[[ "${1:-all}" != preseg ]] || exit 0
# Whole-piece rows. Each Ours row is a mean over three training seeds under one
# boundary condition: the model's own bar boundaries, the annotated boundaries,
# then the annotated boundaries and coordinate. Piano-A2S is one released
# checkpoint, segmented by Beat This! or by the annotated downbeats. Every row
# is bootstrapped against every other row, so the marks (better than the
# cascade; better than all rows) come out of one grid per test set.
TAGS=(full_off full_seed91_off full_seed1217_off)   # training seeds 42, 91, 1217
whole_score_grid() {  # DATASET LIST MANIFEST NAME
    local dataset="$1" list="$2" manifest="$3" name="$4" grid="$TABLES/selfseg_$1"
    local root sfx oracle bt_omr pair row boundary proto tag suffix paths omrs candidate other
    local -A ws omr
    if [[ "$dataset" == syn ]]; then root=$SYN sfx=_ydp oracle=$PA2S_ORACLE_SYN bt_omr=$BT_OMR_SYN
    else root=$ASAP sfx="" oracle=$PA2S_ORACLE_ASAP bt_omr=$BT_OMR_ASAP; fi
    for pair in ours:forward400 ann:annotated annphase:annotatedphase; do
        row=${pair%%:*}; boundary=${pair#*:}
        proto="${boundary}_meter${METER_SWITCH_PENALTY:-8.3}_key${KEY_SWITCH_PENALTY:-6.3}"
        paths=""; omrs=""
        for tag in "${TAGS[@]}"; do
            suffix=""; [[ "$tag" == full_off ]] || suffix="_$tag"
            paths+="${paths:+,}$root/whole_score_${proto}${suffix}${sfx}/results.json"
            if [[ "$dataset" == syn ]]; then omrs+="${omrs:+,}$SYN/omr_ned_syn_test_${tag}_${proto}_ydp"
            else omrs+="${omrs:+,}$ASAP/omr_ned_selfseg_${tag}_${proto}"; fi
        done
        # The self-segmented decode labels its rows "ours" under every boundary condition.
        ws[$row]="$row:ours=$paths"; omr[$row]="$row=$omrs"
    done
    # score-whole scores the cascade beside the seed-42 decode, in the same results file.
    paths="${ws[ours]#*=}"; ws[bt]="bt=${paths%%,*}"; omr[bt]="bt=$bt_omr"
    ws[pa2s_oracle]="pa2s_oracle:oracle=$oracle/whole_score/results.json"
    omr[pa2s_oracle]="pa2s_oracle=$oracle/omr_summary.json"
    for candidate in pa2s_oracle ours ann annphase; do
        echo "[$(stamp)] whole-piece grid, $dataset: $candidate"
        args=()
        for other in bt pa2s_oracle ours ann annphase; do
            [[ "$other" == "$candidate" ]] || args+=(--reference "$other")
        done
        poetry run python -m src.analysis.whole_score_table \
            --system "${ws[bt]}" --system "${ws[pa2s_oracle]}" --system "${ws[ours]}" \
            --system "${ws[ann]}" --system "${ws[annphase]}" \
            --recording-list "$list" --candidate "$candidate" "${args[@]}" \
            --n-boot 10000 --seed 0 --out "$grid/ws_$candidate"
        args=()
        for other in bt pa2s_oracle ours ann annphase; do
            [[ "$other" == "$candidate" ]] || args+=(--reference "${omr[$other]}")
        done
        poetry run python -m src.analysis.paired_omr_ned --scores summary \
            --manifest "$manifest" --subset "$list" --candidate "${omr[$candidate]}" "${args[@]}" \
            --n-boot 10000 --seed 0 --out "$grid/omr_$candidate"
    done
    poetry run python -m src.analysis.selfseg_table --grid "$grid" --name "$name" --out "$grid/table"
}
whole_score_grid syn "$SYN_REF/recordings.txt" "$SYN_REF/manifest.json" "Syn test"
whole_score_grid asap "$S74" "$ASAP/test_manifest.json" "ASAP 74"
echo "[$(stamp)] done: $TABLES"
