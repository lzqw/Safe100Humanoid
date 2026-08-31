#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:?set REPO to the committed v31 checkout}
PYTHON=${PYTHON:?set PYTHON to the MJLab Python interpreter}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?set BASE_CHECKPOINT to fixed pi0}
EXTERNAL_ROOT=${EXTERNAL_ROOT:?set EXTERNAL_ROOT outside the Git repository}
DEVICE=${DEVICE:-cuda:0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
RESULT_ROOT="$REPO/results/online/proximal_v31"
PROTOCOL=${PROTOCOL:-$RESULT_ROOT/protocol.json}
PREFLIGHT_ROOT="$EXTERNAL_ROOT/preflight"
FORMAL_ROOT="$EXTERNAL_ROOT/formal"
FORMAL_AUDIT="$EXTERNAL_ROOT/formal_audit"
MONITOR_ROOT="$EXTERNAL_ROOT/monitor"

run_formal() {
  local context=$1
  local arm=$2
  shift 2
  "$PYTHON" "$REPO/experiments/scripts/refine_cbf_teacher_v31.py" \
    --repo "$REPO" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --protocol "$PROTOCOL" \
    --output-dir "$FORMAL_ROOT/$context/$arm" \
    --phase formal \
    --arm "$arm" \
    --context "$context" \
    --device "$DEVICE" "$@"
}

require_preflight() {
  "$PYTHON" - "$PREFLIGHT_ROOT/preflight_summary.json" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if not payload.get("passed") or not payload.get("formal_ready"):
    raise SystemExit("v31 single preflight has not passed")
PY
}

COMMAND=${1:-}
if [[ -n "$COMMAND" ]]; then
  shift
fi

case "$COMMAND" in
  freeze)
    "$PYTHON" "$REPO/experiments/scripts/freeze_cbf_teacher_v31.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --external-root "$EXTERNAL_ROOT" \
      --output "$PROTOCOL"
    ;;
  preflight)
    "$PYTHON" "$REPO/experiments/scripts/preflight_cbf_teacher_v31.py" \
      --repo "$REPO" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --protocol "$PROTOCOL" \
      --output-dir "$PREFLIGHT_ROOT" \
      --device "$DEVICE"
    ;;
  formal)
    require_preflight
    for context in F1 F2 F3; do
      for arm in A0 A1 A2; do
        run_formal "$context" "$arm"
      done
    done
    ;;
  resume-formal)
    require_preflight
    context=${1:?usage: resume-formal CONTEXT ARM}
    arm=${2:?usage: resume-formal CONTEXT ARM}
    run_formal "$context" "$arm" --resume
    ;;
  audit-formal)
    require_preflight
    "$PYTHON" "$REPO/experiments/scripts/audit_cbf_teacher_v31_formal.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --preflight-summary "$PREFLIGHT_ROOT/preflight_summary.json" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --training-root "$FORMAL_ROOT" \
      --output-dir "$FORMAL_AUDIT" \
      --eval-batch-size "$EVAL_BATCH_SIZE" \
      --device "$DEVICE" "$@"
    ;;
  monitor)
    "$PYTHON" "$REPO/experiments/scripts/monitor_cbf_teacher_v31.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --training-root "$FORMAL_ROOT" \
      --formal-audit "$FORMAL_AUDIT/combined_results.json" \
      --output-dir "$MONITOR_ROOT" \
      --device "$DEVICE" "$@"
    ;;
  package)
    "$PYTHON" "$REPO/experiments/scripts/package_cbf_teacher_v31.py" \
      --repo "$REPO" \
      --protocol "$PROTOCOL" \
      --preflight-dir "$PREFLIGHT_ROOT" \
      --formal-training-root "$FORMAL_ROOT" \
      --formal-audit-dir "$FORMAL_AUDIT" \
      --monitor-dir "$MONITOR_ROOT" \
      --output-dir "$RESULT_ROOT"
    ;;
  *)
    echo "usage: $0 {freeze|preflight|formal|resume-formal CONTEXT ARM|audit-formal|monitor|package}" >&2
    exit 2
    ;;
esac
