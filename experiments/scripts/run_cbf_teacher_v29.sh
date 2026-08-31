#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?set REPO to the committed v29 checkout}
PYTHON=${PYTHON:?set PYTHON to the MJLab Python interpreter}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to fixed pi0}
EXTERNAL_ROOT=${EXTERNAL_ROOT:?set EXTERNAL_ROOT outside the Git repository}
DEVICE=${DEVICE:-cuda:0}
RESULT_ROOT="$REPO/results/online/proximal_v29"
CONFIG=${CONFIG:-$RESULT_ROOT/config.json}

case "${1:-}" in
  freeze)
    shift
    "$PYTHON" "$REPO/experiments/scripts/freeze_cbf_teacher_v29.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --output "$CONFIG" "$@"
    ;;
  smoke)
    shift
    "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v29.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --config "$CONFIG" \
      --output-dir "$EXTERNAL_ROOT/smoke" \
      --device "$DEVICE" \
      --smoke "$@"
    ;;
  train)
    shift
    "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v29.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --config "$CONFIG" \
      --output-dir "$EXTERNAL_ROOT/training" \
      --device "$DEVICE" "$@"
    ;;
  audit)
    shift
    "$PYTHON" "$REPO/experiments/scripts/audit_cbf_teacher_v29.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --final-checkpoint "$EXTERNAL_ROOT/training/final_round_08.pt" \
      --training-summary "$EXTERNAL_ROOT/training/training_summary.json" \
      --config "$CONFIG" \
      --output-dir "$EXTERNAL_ROOT/final" \
      --device "$DEVICE" "$@"
    ;;
  package)
    shift
    "$PYTHON" "$REPO/experiments/scripts/package_cbf_teacher_v29.py" \
      --repo "$REPO" \
      --smoke-dir "$EXTERNAL_ROOT/smoke" \
      --training-dir "$EXTERNAL_ROOT/training" \
      --final-dir "$EXTERNAL_ROOT/final" \
      --output-dir "$RESULT_ROOT" "$@"
    ;;
  *)
    echo "usage: $0 {freeze|smoke|train|audit|package}" >&2
    exit 2
    ;;
esac
