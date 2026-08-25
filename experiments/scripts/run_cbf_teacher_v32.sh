#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?set REPO to the committed v32 checkout}
PYTHON=${PYTHON:?set PYTHON to the MJLab Python interpreter}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to fixed pi0}
V31_FORMAL_ROOT=${V31_FORMAL_ROOT:?set V31_FORMAL_ROOT to v31 external formal training}
EXTERNAL_ROOT=${EXTERNAL_ROOT:?set EXTERNAL_ROOT outside the Git repository}
DEVICE=${DEVICE:-cuda:0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
RESULT_ROOT="$REPO/results/online/proximal_v32"
PROTOCOL=${PROTOCOL:-$RESULT_ROOT/protocol.json}
PREFLIGHT_ROOT="$EXTERNAL_ROOT/preflight"
TRAINING_ROOT="$EXTERNAL_ROOT/formal"
MONITOR_ROOT="$EXTERNAL_ROOT/monitor"
FORMAL_AUDIT_ROOT="$EXTERNAL_ROOT/formal_audit"

require_preflight() {
  "$PYTHON" - "$PREFLIGHT_ROOT/preflight_summary.json" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if not payload.get("passed") or not payload.get("formal_ready"):
    raise SystemExit("v32 functional preflight has not passed")
PY
}

run_continuation() {
  local context=$1
  local schedule=$2
  shift 2
  "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v32.py" \
    --repo "$REPO" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --v31-formal-root "$V31_FORMAL_ROOT" \
    --protocol "$PROTOCOL" \
    --output-dir "$TRAINING_ROOT/continuation/$context/$schedule" \
    --phase formal \
    --kind continuation \
    --context "$context" \
    --schedule "$schedule" \
    --device "$DEVICE" "$@"
}

run_mixed() {
  shift 0
  "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v32.py" \
    --repo "$REPO" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --v31-formal-root "$V31_FORMAL_ROOT" \
    --protocol "$PROTOCOL" \
    --output-dir "$TRAINING_ROOT/mixed/LongDecay" \
    --phase formal \
    --kind mixed \
    --context mixed \
    --schedule LongDecay \
    --device "$DEVICE" "$@"
}

COMMAND=${1:-}
if [[ -n "$COMMAND" ]]; then
  shift
fi

case "$COMMAND" in
  freeze)
    "$PYTHON" "$REPO/experiments/scripts/freeze_cbf_teacher_v32.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --v31-formal-root "$V31_FORMAL_ROOT" \
      --output "$PROTOCOL"
    ;;
  preflight)
    "$PYTHON" "$REPO/experiments/scripts/preflight_cbf_teacher_v32.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --v31-formal-root "$V31_FORMAL_ROOT" \
      --protocol "$PROTOCOL" \
      --output-dir "$PREFLIGHT_ROOT" \
      --device "$DEVICE"
    ;;
  formal)
    require_preflight
    for context in F1 F2 F3; do
      run_continuation "$context" LongConstant
      run_continuation "$context" LongDecay
    done
    run_mixed
    ;;
  resume-continuation)
    require_preflight
    context=${1:?usage: resume-continuation CONTEXT SCHEDULE}
    schedule=${2:?usage: resume-continuation CONTEXT SCHEDULE}
    run_continuation "$context" "$schedule" --resume
    ;;
  resume-mixed)
    require_preflight
    run_mixed --resume
    ;;
  monitor)
    require_preflight
    "$PYTHON" "$REPO/experiments/scripts/monitor_cbf_teacher_v32.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --v31-formal-root "$V31_FORMAL_ROOT" \
      --training-root "$TRAINING_ROOT" \
      --output-dir "$MONITOR_ROOT" \
      --device "$DEVICE" "$@"
    ;;
  audit-formal)
    require_preflight
    "$PYTHON" "$REPO/experiments/scripts/audit_cbf_teacher_v32_formal.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --preflight-summary "$PREFLIGHT_ROOT/preflight_summary.json" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --v31-formal-root "$V31_FORMAL_ROOT" \
      --training-root "$TRAINING_ROOT" \
      --output-dir "$FORMAL_AUDIT_ROOT" \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --device "$DEVICE" "$@"
    ;;
  package)
    "$PYTHON" "$REPO/experiments/scripts/package_cbf_teacher_v32.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --preflight-dir "$PREFLIGHT_ROOT" \
      --training-root "$TRAINING_ROOT" \
      --monitor-dir "$MONITOR_ROOT" \
      --formal-audit-dir "$FORMAL_AUDIT_ROOT" \
      --output-dir "$RESULT_ROOT"
    ;;
  *)
    echo "usage: $0 {freeze|preflight|formal|resume-continuation CONTEXT SCHEDULE|resume-mixed|monitor|audit-formal|package}" >&2
    exit 2
    ;;
esac
