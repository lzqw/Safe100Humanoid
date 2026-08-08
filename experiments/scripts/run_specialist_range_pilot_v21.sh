#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_V21_REPO:-$ROOT/worktrees/v21_range_pilot}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
PROTOCOL_COMMIT="${SAFE100_V21_PROTOCOL_COMMIT:?set SAFE100_V21_PROTOCOL_COMMIT}"
PROTOCOL="${SAFE100_V21_PROTOCOL_FILE:-$REPO/results/online/specialist_v21/protocol_range_pilot_1.json}"
ARTIFACT_ROOT="${SAFE100_V21_ARTIFACT_ROOT:-$ROOT/artifacts/specialist_v21_range_pilot_1}"
LOG_ROOT="${SAFE100_V21_LOG_ROOT:-$ROOT/logs/specialist_v21_range_pilot_1}"

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
mkdir -p "$LOG_ROOT"
QUEUE_LOG="$LOG_ROOT/queue_range_pilot.log"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

for context_id in L_dev C_dev L1 L2 L3 L4 L5 C1 C2 C3 C4 C5; do
  output="$ARTIFACT_ROOT/$context_id"
  compact="$REPO/results/online/specialist_v21/range_pilot/pilot_1/$context_id"
  printf '%s context=%s event=range_pilot_started\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$context_id" | tee -a "$QUEUE_LOG"
  "$PYTHON" experiments/scripts/calibrate_deployment_contexts_v21.py \
    --repo "$REPO" \
    --base-policy-checkpoint "$BASELINE" \
    --protocol-file "$PROTOCOL" \
    --protocol-commit "$PROTOCOL_COMMIT" \
    --context-id "$context_id" \
    --output-dir "$output" \
    --context-output "$output/not_a_frozen_context.json" \
    --num-episodes 128 \
    --eval-batch-size 128 \
    --device cuda:0 \
    --exploratory \
    2>&1 | tee -a "$LOG_ROOT/range_pilot_${context_id}.log" \
    | tee -a "$QUEUE_LOG"
  mkdir -p "$compact"
  cp "$output/calibration_progress.json" "$compact/calibration_progress.json"
done

printf '%s phase=range_pilot event=queue_completed\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"
