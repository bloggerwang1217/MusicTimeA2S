#!/bin/bash
# Run one compared-system baseline named in configs/baselines.yaml.
#   bash src/baselines/run_asap102_omr.sh SYSTEM [DATASET] [extra args]
#   SYSTEM   beatthis_pianoa2s | oracle_pianoa2s;  DATASET  asap (default) | syn
#   RUN overrides the run name (SYSTEM and DATASET map to beatthis_asap,
#   beatthis_syn, oracle_asap, oracle_syn); DRY_RUN=1 passes --dry-run.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

SYSTEM="${1:-}"
DATASET="${2:-asap}"
[ "$#" -eq 0 ] || shift
[ "$#" -eq 0 ] || shift

case "$SYSTEM" in
    beatthis_pianoa2s) RUN="${RUN:-beatthis_$DATASET}"; ARGS=(--run "$RUN") ;;
    oracle_pianoa2s)   RUN="${RUN:-oracle_$DATASET}";   ARGS=(--run "$RUN" --decode-only) ;;
    *)
        echo "Usage: bash src/baselines/run_asap102_omr.sh {beatthis_pianoa2s|oracle_pianoa2s} [asap|syn]" >&2
        exit 2
        ;;
esac

if [ "${DRY_RUN:-0}" = "1" ]; then
    ARGS+=(--dry-run)
fi

poetry run python -m src.baselines.beatthis_pianoa2s.beatthis_pianoa2s_asap102 \
    "${ARGS[@]}" "$@"
