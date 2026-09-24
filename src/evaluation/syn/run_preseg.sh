#!/bin/bash
# Pre-segmented Syn test comparison on the five-bar windows the Piano-A2S
# ASAP chunk builder produces: Piano-A2S (its own chain, ASAP fine-tuned
# checkpoint) and three of our arms decoded on the same annotated bars.
#
# Usage: bash src/evaluation/syn/run_preseg.sh STAGE [ARGS]
#   layout                     ASAP-shaped folder, piece list, recording TSV
#   features [local IDX...]    Piano-A2S chunk build (slurm array), waits and verifies
#   spectrograms               Piano-A2S VQT for every built target (local)
#   pa2s-infer                 Piano-A2S predictions (GPU, $PA2S_GPU)
#   pa2s-mv2h [local]          Piano-A2S reference MIDI + MV2H (slurm array), waits and verifies
#   grounding-decode           decode window manifest (needs only the built targets)
#   grounding                  scoring window manifest (windows with a reference MIDI)
#   decode ARM                 one of full_off, without_fourierpe_off, without_coordinate_off (GPU)
#   score ARM                  MV2H of that arm against the Piano-A2S reference, waits and verifies
#   verify-features | verify-pa2s-mv2h | verify-score ARM
#                              re-run a stage's completion check after its waiter died
#   select-inputs [decode|score] select the current test split without running jobs
# A stage starts only when the previous stage left its completion record in
# $PRESEG/stages/, so an unfinished stage is never read as a failed window.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$PROJECT_ROOT"
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV POETRY_ACTIVE
SELF="$PROJECT_ROOT/src/evaluation/syn/run_preseg.sh"
SYN=data/experiments/syn
TEST_MANIFEST=$SYN/test_manifest.json
REF=$SYN/native_reference_syn_test_ydp
PRESEG="${PRESEG:-$SYN/syn_table1_preseg}"
MV2H_TIMEOUT=300
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
# The cluster's MaxArraySize is 1001 (indices 0-1000).
MAX_ARRAY_TASKS=1000
LOGS="$PROJECT_ROOT/logs/syn_preseg"
LAYOUT="$(realpath -m "$PRESEG/piano_a2s_input")"
STAGES="$PRESEG/stages"
TEST_INPUTS="$PRESEG/test_inputs"
mkdir -p "$LOGS"
stamp() { date '+%F %T'; }
ceil_div() { echo $(( ($1 + $2 - 1) / $2 )); }

path_from_config() {  # KEY: its value in the baseline config; exits while it is a placeholder
    local config=configs/baselines.yaml value
    value=$(poetry run python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$config" "$1")
    if [[ -z "$value" || "$value" == /path/to/* ]]; then
        echo "$config: $1 is not filled in" >&2; exit 1
    fi
    printf '%s\n' "$value"
}
piano_a2s() {  # the Piano-A2S checkout and its run directories
    P="${PIANO_A2S_ROOT:-$(path_from_config piano_a2s_repo)}"
    PA2S_FEATURE="${PA2S_FEATURE:-$P/workspace/feature.syn_test540}"
    PA2S_OUTPUT="${PA2S_OUTPUT:-$P/workspace/1234/syn_test540_eval.epr_ft.gt}"
    PA2S_CHECKPOINT="$P/workspace/1234/finetune.epr/save"
}

pa2s_env() {
    # Runs a command inside the Piano-A2S conda environment from its repo root.
    # CONDA_PREFIX is unset above for poetry; a leftover CONDA_SHLVL would make
    # conda try to deactivate a prefix that no longer exists.
    env -u CONDA_SHLVL --chdir="$P" bash -c 'set -eo pipefail; source env.sh; source "$CONDA_SH"; conda activate "$CONDA_ENV"
        export PATH="$PROJECT_ROOT/humextra/bin:$PROJECT_ROOT/verovio/tools:$PATH"
        export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        '"$1"
}

arm_config() {
    case "$1" in
        full_off)
            CHECKPOINT=checkpoints/full_seed42.pt
            CONFIG=configs/piano_2gpu.yaml ;;
        without_fourierpe_off)
            CHECKPOINT=checkpoints/without_fourierpe_seed42.pt
            CONFIG=configs/piano_2gpu_without_fourierpe.yaml ;;
        without_coordinate_off)
            CHECKPOINT=checkpoints/without_coordinate_seed42.pt
            CONFIG=configs/piano_2gpu_without_coordinate.yaml ;;
        *) echo "unknown arm $1" >&2; exit 1 ;;
    esac
}

require_stage() {
    [ -s "$STAGES/$1.done" ] || { echo "stage $1 has no completion record in $STAGES" >&2; exit 1; }
}

prepare_test_inputs() {
    poetry run python -m src.evaluation.syn.preseg_stages select-inputs \
        "$TEST_MANIFEST" "$PRESEG" "$TEST_INPUTS" "$1"
}

require_test_layout() {
    poetry run python -m src.evaluation.syn.preseg_stages check-manifest \
        "$TEST_MANIFEST" "$LAYOUT/manifest.json"
}

require_test_chunks() {
    poetry run python -m src.evaluation.syn.preseg_stages check-chunks \
        "$TEST_MANIFEST" "$1"
}

mark_stage() {
    # $1 name, $2 one-line JSON with the counts the check established.
    mkdir -p "$STAGES"
    printf '%s\n' "$2" > "$STAGES/$1.done"
    echo "[$(stamp)] stage $1 complete: $2"
}

wait_jobs() {
    # Accounting is disabled on this cluster, so completion is judged from each
    # task's own output; this only waits for the jobs to leave the queue. A job
    # held on a dependency that can never be met stays queued forever.
    while true; do
        # One query per job: squeue rejects a whole list once any id has aged out.
        local queued="" id
        for id in "$@"; do
            queued+=$(squeue -h -j "$id" -o '%i %t %r' 2>/dev/null || true)$'\n'
        done
        queued=$(grep . <<<"$queued" || true)
        [ -z "$queued" ] && return 0
        if ! grep -qv 'DependencyNeverSatisfied' <<<"$queued"; then
            echo "jobs $* are held on a dependency that failed:" >&2
            head -5 <<<"$queued" >&2
            return 1
        fi
        sleep 60
    done
}

verify_features() {
    require_test_layout
    local job n missing
    job=$(cat "$PRESEG/jobs/features.jobid")
    n=$(grep -c . "$LAYOUT/test_perfs.tsv")
    missing=()
    for i in $(seq 0 $((n - 1))); do
        grep -qx "\[task $i\] done" "$LOGS/features_${job}_${i}.out" 2>/dev/null || missing+=("$i")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "feature tasks without a completion line (${#missing[@]}): ${missing[*]:0:20}" >&2
        exit 1
    fi
    mark_stage features "{\"recordings\": $n, \"targets\": $(ls "$PA2S_FEATURE/test/target" | wc -l), \"jobid\": $job}"
}

verify_pa2s_mv2h() {
    require_test_chunks "$PA2S_OUTPUT/results/test"
    poetry run python -m src.evaluation.syn.preseg_stages pa2s-mv2h \
        "$PA2S_OUTPUT/results" "$(cat "$PRESEG/jobs/pa2s_mv2h.tasks")" > "$PRESEG/jobs/pa2s_mv2h.verify.json"
    mark_stage pa2s-mv2h "$(cat "$PRESEG/jobs/pa2s_mv2h.verify.json")"
}

verify_score() {
    local arm="$1" out="$PRESEG/eval_$1"
    prepare_test_inputs score
    poetry run python -m src.evaluation.syn.preseg_stages score \
        "$out/eval_asap.csv" "$TEST_INPUTS/grounding_score.jsonl" > "$PRESEG/jobs/score_$arm.verify.json"
    mark_stage "score_$arm" "$(cat "$PRESEG/jobs/score_$arm.verify.json")"
}

stage="${1:?stage required}"; shift
# select-inputs, layout and decode run only our side.
case "$stage" in
features|features-task|verify-features|spectrograms|pa2s-infer|pa2s-mv2h|pa2s-mv2h-task|\
verify-pa2s-mv2h|grounding-decode|grounding|score|verify-score) piano_a2s ;;
esac
case "$stage" in
select-inputs) prepare_test_inputs "${1:-decode}" ;;
layout)
    poetry run python -m src.evaluation.syn.preseg_stages check-manifest \
        "$TEST_MANIFEST" "$REF/manifest.json"
    poetry run python -m src.evaluation.syn.build_piano_a2s_syn_layout \
        --manifest "$REF/manifest.json" --mapping "$REF/mapping.jsonl" \
        --audio-dir "$SYN/audio" --out "$LAYOUT"
    require_test_layout
    mark_stage layout "{\"recordings\": $(grep -c . "$LAYOUT/test_perfs.tsv")}"
    ;;
features)
    require_stage layout
    require_test_layout
    n=$(grep -c . "$LAYOUT/test_perfs.tsv")
    if [ "${1:-}" = local ]; then
        shift
        for idx in "$@"; do bash "$SELF" features-task "$idx"; done
        exit 0
    fi
    [ -e "$PA2S_FEATURE" ] && { echo "refusing to reuse $PA2S_FEATURE" >&2; exit 1; }
    [ "$n" -le "$MAX_ARRAY_TASKS" ] || { echo "$n recordings exceed the array limit" >&2; exit 1; }
    mkdir -p "$PRESEG/jobs"
    job=$(sbatch --parsable --job-name=syn_preseg_features --partition=compute \
        --array="0-$((n - 1))%$MAX_CONCURRENT" --cpus-per-task=2 --mem=8G \
        --time="${FEATURE_TIME:-06:00:00}" \
        --output="$LOGS/features_%A_%a.out" --error="$LOGS/features_%A_%a.err" \
        --wrap="bash $SELF features-task \$SLURM_ARRAY_TASK_ID")
    echo "$job" > "$PRESEG/jobs/features.jobid"
    echo "[$(stamp)] features array $job ($n tasks)"
    wait_jobs "$job"
    verify_features
    ;;
features-task)
    require_test_layout
    pa2s_env "python build_asap_one_perf.py --task-id $1 --tsv $LAYOUT/test_perfs.tsv \
        --test-list $LAYOUT/test_pieces.txt \
        --train-list data_processing/metadata/train_none.txt \
        --feature-folder $PA2S_FEATURE --workdir temp_workers/syn_preseg_features"
    ;;
verify-features) verify_features ;;
spectrograms)
    require_stage features
    require_test_chunks "$PA2S_FEATURE/test/target"
    pa2s_env "python - '$PA2S_FEATURE' < '$PROJECT_ROOT/src/evaluation/syn/piano_a2s_spectrograms.py'"
    targets=$(ls "$PA2S_FEATURE/test/target" | wc -l)
    spectrograms=$(ls "$PA2S_FEATURE/test/spectrogram" | wc -l)
    [ "$targets" -eq "$spectrograms" ] || { echo "targets $targets, spectrograms $spectrograms" >&2; exit 1; }
    mark_stage spectrograms "{\"spectrograms\": $spectrograms}"
    ;;
pa2s-infer)
    require_stage spectrograms
    require_test_chunks "$PA2S_FEATURE/test/target"
    require_test_chunks "$PA2S_FEATURE/test/spectrogram"
    [ -e "$PA2S_OUTPUT/results/test" ] && { echo "refusing to overwrite $PA2S_OUTPUT/results/test" >&2; exit 1; }
    pa2s_env "CUDA_VISIBLE_DEVICES=${PA2S_GPU:?set PA2S_GPU} python test_inference.py \
        --hparams hparams/finetune.yaml --checkpoint-dir $PA2S_CHECKPOINT \
        --feature-folder $PA2S_FEATURE --output-folder $PA2S_OUTPUT"
    predictions=$(ls "$PA2S_OUTPUT/results/test" | wc -l)
    spectrograms=$(ls "$PA2S_FEATURE/test/spectrogram" | wc -l)
    [ "$predictions" -eq "$spectrograms" ] || { echo "predictions $predictions, spectrograms $spectrograms" >&2; exit 1; }
    mark_stage pa2s-infer "{\"predictions\": $predictions}"
    ;;
pa2s-mv2h)
    require_stage pa2s-infer
    require_test_chunks "$PA2S_OUTPUT/results/test"
    n=$(ls "$PA2S_OUTPUT/results/test" | wc -l)
    tasks=$(ceil_div "$n" 32)
    [ "$tasks" -le "$MAX_ARRAY_TASKS" ] || tasks=$MAX_ARRAY_TASKS
    per_task=$(ceil_div "$n" "$tasks")
    mkdir -p "$PRESEG/jobs"
    echo "$tasks" > "$PRESEG/jobs/pa2s_mv2h.tasks"
    if [ "${1:-}" = local ]; then
        seq 0 $((tasks - 1)) | xargs -P "${WORKERS:-8}" -I{} bash "$SELF" pa2s-mv2h-task {} "$tasks"
    else
        # Budget every window at the MV2H timeout plus conversion time.
        minutes=$(( $(ceil_div $(( per_task * (MV2H_TIMEOUT + 15) )) 60) + 30 ))
        job=$(sbatch --parsable --job-name=syn_preseg_pa2s_mv2h --partition=compute \
            --array="0-$((tasks - 1))%$MAX_CONCURRENT" --cpus-per-task=1 --mem=4G \
            --time="$minutes" \
            --output="$LOGS/pa2s_mv2h_%A_%a.out" --error="$LOGS/pa2s_mv2h_%A_%a.err" \
            --wrap="bash $SELF pa2s-mv2h-task \$SLURM_ARRAY_TASK_ID $tasks")
        echo "$job" > "$PRESEG/jobs/pa2s_mv2h.jobid"
        echo "[$(stamp)] Piano-A2S MV2H array $job ($tasks tasks, $per_task windows, ${minutes} min each)"
        wait_jobs "$job"
    fi
    verify_pa2s_mv2h
    ;;
pa2s-mv2h-task)
    pa2s_env "SLURM_ARRAY_TASK_ID=$1 MV2H_TIMEOUT=$MV2H_TIMEOUT MV2H_KEEP_ZERO=1 python evaluate_worker.py \
        --output-folder $PA2S_OUTPUT --mv2h-bin MV2H/bin --split test --num-tasks $2"
    ;;
verify-pa2s-mv2h) verify_pa2s_mv2h ;;
grounding-decode|grounding)
    # The decode windows depend only on the built targets, so decoding need not
    # wait for Piano-A2S's MV2H; the scoring windows need its reference MIDI.
    if [ "$stage" = grounding-decode ]; then
        require_stage features
        kind=decode; extra=()
    else
        require_stage pa2s-mv2h; require_stage grounding_decode
        kind=score; extra=(--require-gt-midi)
    fi
    out="$PRESEG/grounding_$kind.jsonl"
    [ -e "$out" ] && { echo "refusing to overwrite $out" >&2; exit 1; }
    poetry run python -m src.evaluation.asap.build_piano_a2s_gt_grounding \
        --target-dir "$PA2S_FEATURE/test/target" --results-dir "$PA2S_OUTPUT/results" \
        --our-manifest "$LAYOUT/manifest.json" --piano-a2s-repo "$P" \
        --upbeat-recordings "$LAYOUT/upbeat_recordings.txt" "${extra[@]}" --output "$out"
    if [ "$kind" = score ]; then
        mark_stage grounding "{\"decode_windows\": $(grep -c . "$PRESEG/grounding_decode.jsonl"), \"score_windows\": $(grep -c . "$PRESEG/grounding_score.jsonl")}"
        exit 0
    fi
    poetry run python -m src.evaluation.syn.preseg_stages decode-manifest \
        "$LAYOUT/manifest.json" "$PRESEG/grounding_decode.jsonl" "$PRESEG/manifest_decode.json"
    mark_stage grounding_decode "{\"decode_windows\": $(grep -c . "$PRESEG/grounding_decode.jsonl")}"
    ;;
decode)
    arm="${1:?arm}"; arm_config "$arm"
    require_stage grounding_decode
    prepare_test_inputs decode
    mkdir -p "$PRESEG/jobs"
    GPU="${GPU:?set GPU}" CHECKPOINT="$CHECKPOINT" CONFIG="$CONFIG" \
    MANIFEST_DIR="$SYN" MANIFEST="$TEST_INPUTS/manifest_decode.json" GROUNDING="$TEST_INPUTS/grounding_decode.jsonl" \
    OUTPUT_DIR="$PRESEG/kern_$arm" LOG_DIR="$LOGS" \
    bash src/a2s/piano/run_inference.sh 2>&1 | tee "$PRESEG/jobs/decode_$arm.log"
    # run_inference.sh pipes through tee, so a crashed decoder can still exit
    # zero; its completion line is the record.
    grep -q "^Inference complete!" "$PRESEG/jobs/decode_$arm.log" \
        || { echo "decode log has no completion line" >&2; exit 1; }
    inventory=$(poetry run python -m src.evaluation.syn.preseg_stages decode-inventory \
        "$PRESEG/kern_$arm" "$TEST_INPUTS/grounding_decode.jsonl")
    mark_stage "decode_$arm" "$inventory"
    ;;
score)
    arm="${1:?arm}"
    require_stage "decode_$arm"; require_stage grounding
    prepare_test_inputs score
    out="$PRESEG/eval_$arm"
    [ -e "$out/eval_asap.csv" ] && { echo "refusing to overwrite $out" >&2; exit 1; }
    n=$(grep -c . "$TEST_INPUTS/grounding_score.jsonl")
    chunks=$(ceil_div "$n" "$MAX_ARRAY_TASKS")
    [ "$chunks" -ge 32 ] || chunks=32
    mkdir -p "$PRESEG/jobs"
    PRED_DIR="$PRESEG/kern_$arm" GROUNDING="$TEST_INPUTS/grounding_score.jsonl" \
    REFERENCE_ROOT="$PA2S_OUTPUT/results" OUTPUT_DIR="$out" \
    CHUNK_TIMEOUT=$MV2H_TIMEOUT CHUNKS_PER_JOB="$chunks" MAX_CONCURRENT="$MAX_CONCURRENT" \
    PROJECT_ROOT="$PROJECT_ROOT" bash src/evaluation/slurm_eval_window_piano_a2s.sh | tee "$PRESEG/jobs/score_$arm.submit.log"
    jobs=$(grep -oE 'submitted: job [0-9]+' "$PRESEG/jobs/score_$arm.submit.log" | awk '{print $3}' | tr '\n' ' ')
    [ "$(wc -w <<<"$jobs")" -eq 2 ] || { echo "could not read both array ids" >&2; exit 1; }
    echo "$jobs" > "$PRESEG/jobs/score_$arm.jobid"
    # shellcheck disable=SC2086
    wait_jobs $jobs
    verify_score "$arm"
    ;;
verify-score) verify_score "${1:?arm}" ;;
*)
    echo "unknown stage $stage" >&2; exit 1 ;;
esac
