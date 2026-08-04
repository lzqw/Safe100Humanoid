#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
OUTPUT="${SAFE100_CALIBRATION_OUTPUT_DIR:-$ROOT/artifacts/failure_focused_v15/context_calibration}"
CONTEXT="${SAFE100_DEPLOYMENT_CONTEXT:-$ROOT/artifacts/failure_focused_v15/frozen_dqh_medium_context.json}"
LOG="${SAFE100_CALIBRATION_LOG_PATH:-$ROOT/logs/failure_focused_v15/context_calibration.log}"

mkdir -p "$OUTPUT" "$(dirname "$CONTEXT")" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/calibrate_failure_focused_context_v15.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --output-dir "$OUTPUT" \
  --context-output "$CONTEXT" \
  --candidate-seeds 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 \
    1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 \
  --num-episodes 128 \
  --eval-batch-size 128 \
  --device cuda:0 \
  2>&1 | tee "$LOG"
