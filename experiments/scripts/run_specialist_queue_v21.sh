#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
PHASE="${1:-}"
case "$PHASE" in calibration|development|formal) ;; *) echo "usage: $0 calibration|development|formal" >&2; exit 2 ;; esac
DEFAULT_REPO="$ROOT/worktrees/v21_formal"
test "$PHASE" = calibration && DEFAULT_REPO="$ROOT/worktrees/v21_final_calibration_v2"
REPO="${SAFE100_V21_REPO:-$DEFAULT_REPO}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
PROTOCOL_COMMIT="${SAFE100_V21_PROTOCOL_COMMIT:?set SAFE100_V21_PROTOCOL_COMMIT}"
DEFAULT_PROTOCOL="$REPO/results/online/specialist_v21/protocol_formal.json"
test "$PHASE" = calibration && DEFAULT_PROTOCOL="$REPO/results/online/specialist_v21/protocol_precalibration_replacement_v2.json"
test "$PHASE" = development && DEFAULT_PROTOCOL="$REPO/results/online/specialist_v21/protocol_development.json"
PROTOCOL="${SAFE100_V21_PROTOCOL_FILE:-$DEFAULT_PROTOCOL}"
ARTIFACT_ROOT="${SAFE100_V21_ARTIFACT_ROOT:-$ROOT/artifacts/specialist_v21_replacement_v2}"
LOG_ROOT="${SAFE100_V21_LOG_ROOT:-$ROOT/logs/specialist_v21_replacement_v2}"
CONTEXT_ROOT="${SAFE100_V21_CONTEXT_ROOT:-$REPO/results/online/specialist_v21/contexts_replacement_v2}"
CALIBRATION_SUMMARY_ROOT="${SAFE100_V21_CALIBRATION_SUMMARY_ROOT:-$REPO/results/online/specialist_v21/calibration/replacement_v2}"
cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
mkdir -p "$LOG_ROOT"
QUEUE_LOG="$LOG_ROOT/queue_${PHASE}.log"
export SAFE100_V21_REPO="$REPO"
export SAFE100_V21_PROTOCOL_COMMIT="$PROTOCOL_COMMIT"
export SAFE100_V21_PROTOCOL_FILE="$PROTOCOL"
export SAFE100_V21_ARTIFACT_ROOT="$ARTIFACT_ROOT"
export SAFE100_V21_LOG_ROOT="$LOG_ROOT"
export SAFE100_V21_CONTEXT_ROOT="$CONTEXT_ROOT"
export SAFE100_V21_CALIBRATION_SUMMARY_ROOT="$CALIBRATION_SUMMARY_ROOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if test "$PHASE" = calibration; then
  for context_id in L_dev C_dev L1 L2 L3 L4 L5 C1 C2 C3 C4 C5; do
    printf '%s context=%s event=calibration_started\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$context_id" | tee -a "$QUEUE_LOG"
    bash experiments/scripts/run_specialist_calibration_v21.sh "$context_id" \
      2>&1 | tee -a "$QUEUE_LOG"
  done
elif test "$PHASE" = development; then
  for context_id in L_dev C_dev; do
    for beta in 0 1 4 16; do
      role=v21
      test "$beta" = 0 && role=control
      printf '%s context=%s beta=%s event=development_started\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$context_id" "$beta" | tee -a "$QUEUE_LOG"
      bash experiments/scripts/run_specialist_v21.sh "$context_id" "$role" "$beta" \
        2>&1 | tee -a "$QUEUE_LOG"
    done
  done
  "$PYTHON" experiments/scripts/select_development_beta_v21.py \
    --repo "$REPO" \
    --base-policy-checkpoint "$BASELINE" \
    --context-dir "$CONTEXT_ROOT" \
    --training-root "$ARTIFACT_ROOT/training" \
    --protocol-file "$PROTOCOL" \
    --protocol-commit "$PROTOCOL_COMMIT" \
    --output-dir "$REPO/results/online/specialist_v21/development" \
    --device cuda:0 \
    2>&1 | tee -a "$QUEUE_LOG"
else
  SELECTED_BETA="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["formal"]["selected_beta"])' "$PROTOCOL")"
  for context_id in L1 L2 L3 L4 L5 C1 C2 C3 C4 C5; do
    bash experiments/scripts/run_specialist_v21.sh "$context_id" control 0 \
      2>&1 | tee -a "$QUEUE_LOG"
    bash experiments/scripts/run_specialist_v21.sh "$context_id" v21 "$SELECTED_BETA" \
      2>&1 | tee -a "$QUEUE_LOG"
    for role in control v21; do
      beta="$SELECTED_BETA"
      test "$role" = control && beta=0
      label="beta_${beta//./p}"
      "$PYTHON" experiments/scripts/evaluate_learning_curve_v21.py \
        --repo "$REPO" \
        --context-id "$context_id" \
        --context "$CONTEXT_ROOT/$context_id.json" \
        --training-dir "$ARTIFACT_ROOT/training/$context_id/${role}_${label}" \
        --protocol-file "$PROTOCOL" \
        --protocol-commit "$PROTOCOL_COMMIT" \
        --output-dir "$ARTIFACT_ROOT/monitor/$context_id/$role" \
        --device cuda:0 \
        2>&1 | tee -a "$QUEUE_LOG"
    done
    "$PYTHON" experiments/scripts/audit_deployment_v21.py \
      --repo "$REPO" \
      --base-policy-checkpoint "$BASELINE" \
      --context-id "$context_id" \
      --context "$CONTEXT_ROOT/$context_id.json" \
      --control-training-dir "$ARTIFACT_ROOT/training/$context_id/control_beta_0" \
      --v21-training-dir "$ARTIFACT_ROOT/training/$context_id/v21_beta_${SELECTED_BETA//./p}" \
      --protocol-file "$PROTOCOL" \
      --protocol-commit "$PROTOCOL_COMMIT" \
      --output-dir "$ARTIFACT_ROOT/audit/$context_id" \
      --target-episodes 1024 \
      --d0-episodes 256 \
      --eval-batch-size 128 \
      --bootstrap-samples 10000 \
      --device cuda:0 \
      2>&1 | tee -a "$QUEUE_LOG"
  done
  "$PYTHON" experiments/scripts/aggregate_deployment_v21.py \
    --repo "$REPO" \
    --protocol-file "$PROTOCOL" \
    --protocol-commit "$PROTOCOL_COMMIT" \
    --audit-root "$ARTIFACT_ROOT/audit" \
    --monitor-root "$ARTIFACT_ROOT/monitor" \
    --output-dir "$REPO/results/online/specialist_v21/formal" \
    --bootstrap-samples 10000 \
    2>&1 | tee -a "$QUEUE_LOG"
  "$PYTHON" experiments/scripts/plot_deployment_v21.py \
    --repo "$REPO" \
    --formal-results "$REPO/results/online/specialist_v21/formal/formal_results.json" \
    --output-dir "$REPO/results/online/specialist_v21/figures" \
    2>&1 | tee -a "$QUEUE_LOG"
fi

printf '%s phase=%s event=queue_completed\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PHASE" | tee -a "$QUEUE_LOG"
