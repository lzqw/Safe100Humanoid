#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
CONTEXT="${SAFE100_DEPLOYMENT_CONTEXT:-$ROOT/artifacts/failure_focused_v15/frozen_dqh_medium_context.json}"
DEFAULT_CANDIDATE_TEMPLATE="$ROOT/artifacts/failure_focused_v15/train_seed{seed}/accepted_final.pt"
CANDIDATE_TEMPLATE="${SAFE100_CANDIDATE_TEMPLATE:-$DEFAULT_CANDIDATE_TEMPLATE}"
OUTPUT="${SAFE100_AUDIT_OUTPUT_DIR:-$ROOT/artifacts/failure_focused_v15/final_audit}"
LOG="${SAFE100_AUDIT_LOG_PATH:-$ROOT/logs/failure_focused_v15/final_audit.log}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/audit_failure_focused_v15.py \
  --repo "$REPO" \
  --baseline-checkpoint "$BASELINE" \
  --candidate-template "$CANDIDATE_TEMPLATE" \
  --deployment-context "$CONTEXT" \
  --training-seeds 42 142 242 \
  --output-dir "$OUTPUT" \
  --eval-batch-size 128 \
  --bootstrap-samples 10000 \
  --audit-seed 1500000 \
  --device cuda:0 \
  2>&1 | tee "$LOG"
