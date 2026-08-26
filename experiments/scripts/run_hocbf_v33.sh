#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?set REPO to the committed v33 checkout}
PYTHON=${PYTHON:?set PYTHON to the MJLab Python interpreter}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to the fixed common base}
V31_FORMAL_ROOT=${V31_FORMAL_ROOT:?set V31_FORMAL_ROOT to v31 formal artifacts}
EXTERNAL_ROOT=${EXTERNAL_ROOT:?set EXTERNAL_ROOT outside Git}
DEVICE=${DEVICE:-cuda:0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
RESULT_ROOT="$REPO/results/online/hocbf_v33"
CONFIG="$RESULT_ROOT/config.json"
SMOKE="$EXTERNAL_ROOT/smoke_summary.json"
DEVELOPMENT="$EXTERNAL_ROOT/development"
SELECTED="$DEVELOPMENT/selected_hocbf.json"
FROZEN="$EXTERNAL_ROOT/frozen_policy_audit"
TRAINING="$EXTERNAL_ROOT/training"
FINAL="$EXTERNAL_ROOT/final"

COMMAND=${1:-}
if [[ -n "$COMMAND" ]]; then
  shift
fi

case "$COMMAND" in
  freeze)
    "$PYTHON" "$REPO/experiments/scripts/freeze_hocbf_v33.py" \
      --repo "$REPO" --base-checkpoint "$BASE_CHECKPOINT" \
      --v31-root "$V31_FORMAL_ROOT" --output "$CONFIG"
    ;;
  smoke)
    "$PYTHON" "$REPO/experiments/scripts/run_hocbf_v33.py" \
      --repo "$REPO" --config "$CONFIG" \
      --checkpoint "$V31_FORMAL_ROOT/F1/A2/round_08.pt" \
      --output "$SMOKE" --device "$DEVICE"
    ;;
  develop)
    "$PYTHON" "$REPO/experiments/scripts/develop_hocbf_v33.py" \
      --repo "$REPO" --config "$CONFIG" --v31-root "$V31_FORMAL_ROOT" \
      --output-root "$DEVELOPMENT" --device "$DEVICE" "$@"
    ;;
  frozen-audit)
    "$PYTHON" "$REPO/experiments/scripts/audit_hocbf_v33.py" \
      --repo "$REPO" --config "$CONFIG" --selected "$SELECTED" \
      --v31-root "$V31_FORMAL_ROOT" --output-root "$FROZEN" \
      --phase frozen --eval-batch-size "$EVAL_BATCH_SIZE" --device "$DEVICE" "$@"
    ;;
  train)
    for context in F1 F2 F3; do
      "$PYTHON" "$REPO/experiments/scripts/refine_hocbf_v33.py" \
        --repo "$REPO" --config "$CONFIG" --selected "$SELECTED" \
        --base-checkpoint "$BASE_CHECKPOINT" --context "$context" \
        --output-dir "$TRAINING/$context" --device "$DEVICE"
    done
    ;;
  resume-train)
    context=${1:?usage: resume-train F1|F2|F3}
    "$PYTHON" "$REPO/experiments/scripts/refine_hocbf_v33.py" \
      --repo "$REPO" --config "$CONFIG" --selected "$SELECTED" \
      --base-checkpoint "$BASE_CHECKPOINT" --context "$context" \
      --output-dir "$TRAINING/$context" --device "$DEVICE" --resume
    ;;
  final-audit)
    "$PYTHON" "$REPO/experiments/scripts/audit_hocbf_v33.py" \
      --repo "$REPO" --config "$CONFIG" --selected "$SELECTED" \
      --v31-root "$V31_FORMAL_ROOT" --base-checkpoint "$BASE_CHECKPOINT" \
      --training-root "$TRAINING" --output-root "$FINAL" --phase final \
      --eval-batch-size "$EVAL_BATCH_SIZE" --device "$DEVICE" "$@"
    ;;
  package)
    "$PYTHON" "$REPO/experiments/scripts/package_hocbf_v33.py" \
      --repo "$REPO" --config "$CONFIG" --smoke "$SMOKE" \
      --development-root "$DEVELOPMENT" --frozen-root "$FROZEN" \
      --training-root "$TRAINING" --final-root "$FINAL" --output-dir "$RESULT_ROOT"
    ;;
  *)
    echo "usage: $0 {freeze|smoke|develop [--resume]|frozen-audit [--resume]|train|resume-train CONTEXT|final-audit [--resume]|package}" >&2
    exit 2
    ;;
esac
