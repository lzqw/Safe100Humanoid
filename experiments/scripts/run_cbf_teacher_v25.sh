#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?set REPO to the committed v25 checkout}
PYTHON=${PYTHON:?set PYTHON to the MJLab Python interpreter}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to frozen pi0}
DEVICE=${DEVICE:-cuda:0}
RESULT_ROOT="$REPO/results/online/proximal_v25"
EXTERNAL_ROOT=${EXTERNAL_ROOT:?set EXTERNAL_ROOT outside the Git repository}

case "${1:-}" in
  calibrate)
    shift
    "$PYTHON" "$REPO/experiments/scripts/calibrate_cbf_teacher_v25.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --precalibration-protocol "$RESULT_ROOT/precalibration_protocol.json" \
      --output-dir "$EXTERNAL_ROOT/calibration" \
      --device "$DEVICE" "$@"
    ;;
  train)
    shift
    "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v25.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --context "$RESULT_ROOT/calibration/context.json" \
      --protocol "$RESULT_ROOT/protocol.json" \
      --output-dir "$EXTERNAL_ROOT/training" \
      --device "$DEVICE" "$@"
    ;;
  audit)
    shift
    "$PYTHON" "$REPO/experiments/scripts/audit_cbf_teacher_v25.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --final-checkpoint "$EXTERNAL_ROOT/training/final_round_08.pt" \
      --training-summary "$EXTERNAL_ROOT/training/training_summary.json" \
      --context "$RESULT_ROOT/calibration/context.json" \
      --protocol "$RESULT_ROOT/protocol.json" \
      --output-dir "$EXTERNAL_ROOT/final" \
      --device "$DEVICE" "$@"
    ;;
  *)
    echo "usage: $0 {calibrate|train|audit} [additional arguments]" >&2
    exit 2
    ;;
esac
