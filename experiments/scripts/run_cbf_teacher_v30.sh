#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?set REPO to the committed v30 checkout}
PYTHON=${PYTHON:?set PYTHON to the MJLab Python interpreter}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to fixed pi0}
EXTERNAL_ROOT=${EXTERNAL_ROOT:?set EXTERNAL_ROOT outside the Git repository}
DEVICE=${DEVICE:-cuda:0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
RESULT_ROOT="$REPO/results/online/proximal_v30"
PROTOCOL=${PROTOCOL:-$RESULT_ROOT/protocol.json}
SMOKE_ROOT="$EXTERNAL_ROOT/smoke"
DEVELOPMENT_ROOT="$EXTERNAL_ROOT/development"
DEVELOPMENT_AUDIT="$EXTERNAL_ROOT/development_audit"
FORMAL_ROOT="$EXTERNAL_ROOT/formal"
FORMAL_AUDIT="$EXTERNAL_ROOT/formal_audit"
MONITOR_ROOT="$EXTERNAL_ROOT/monitor"

selected_arm() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["formal"]["selected_teacher"]["arm"])' "$PROTOCOL"
}

COMMAND=${1:-}
if [[ -n "$COMMAND" ]]; then
  shift
fi

case "$COMMAND" in
  freeze-development)
    "$PYTHON" "$REPO/experiments/scripts/freeze_cbf_teacher_v30.py" \
      --stage development \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --external-root "$EXTERNAL_ROOT" \
      --output "$PROTOCOL"
    ;;
  smoke)
    "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v30.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --protocol "$PROTOCOL" \
      --output-dir "$SMOKE_ROOT" \
      --phase smoke \
      --arm A5 \
      --context DEV \
      --device "$DEVICE"
    ;;
  development)
    for arm in A0 A1 A2 A3 A4 A5; do
      "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v30.py" \
        --repo "$REPO" \
        --base-checkpoint "$BASE_CHECKPOINT" \
        --protocol "$PROTOCOL" \
        --output-dir "$DEVELOPMENT_ROOT/$arm" \
        --phase development \
        --arm "$arm" \
        --context DEV \
        --device "$DEVICE"
    done
    ;;
  audit-development)
    "$PYTHON" "$REPO/experiments/scripts/audit_cbf_teacher_v30_development.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --training-root "$DEVELOPMENT_ROOT" \
      --output-dir "$DEVELOPMENT_AUDIT" \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --device "$DEVICE" "$@"
    ;;
  freeze-formal)
    "$PYTHON" "$REPO/experiments/scripts/freeze_cbf_teacher_v30.py" \
      --stage formal \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --development-audit "$DEVELOPMENT_AUDIT/arm_summary.json" \
      --external-root "$EXTERNAL_ROOT" \
      --output "$PROTOCOL"
    ;;
  formal)
    teacher_arm=$(selected_arm)
    for context in F1 F2 F3; do
      for arm in A0 "$teacher_arm"; do
        "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v30.py" \
          --repo "$REPO" \
          --base-checkpoint "$BASE_CHECKPOINT" \
          --protocol "$PROTOCOL" \
          --output-dir "$FORMAL_ROOT/$context/$arm" \
          --phase formal \
          --arm "$arm" \
          --context "$context" \
          --device "$DEVICE"
      done
    done
    ;;
  audit-formal)
    "$PYTHON" "$REPO/experiments/scripts/audit_cbf_teacher_v30_formal.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --training-root "$FORMAL_ROOT" \
      --output-dir "$FORMAL_AUDIT" \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --device "$DEVICE" "$@"
    ;;
  monitor)
    "$PYTHON" "$REPO/experiments/scripts/monitor_cbf_teacher_v30.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --training-root "$FORMAL_ROOT" \
      --formal-audit "$FORMAL_AUDIT/combined_results.json" \
      --output-dir "$MONITOR_ROOT" \
      --device "$DEVICE" "$@"
    ;;
  package)
    "$PYTHON" "$REPO/experiments/scripts/package_cbf_teacher_v30.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --smoke-dir "$SMOKE_ROOT" \
      --development-training-root "$DEVELOPMENT_ROOT" \
      --development-audit-dir "$DEVELOPMENT_AUDIT" \
      --formal-training-root "$FORMAL_ROOT" \
      --formal-audit-dir "$FORMAL_AUDIT" \
      --monitor-dir "$MONITOR_ROOT" \
      --output-dir "$RESULT_ROOT"
    ;;
  *)
    echo "usage: $0 {freeze-development|smoke|development|audit-development|freeze-formal|formal|audit-formal|monitor|package}" >&2
    exit 2
    ;;
esac
