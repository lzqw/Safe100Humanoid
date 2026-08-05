#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
CONTEXT_DIR="${SAFE100_SPECIALIST_CONTEXT_DIR:-$ROOT/artifacts/specialist_v17/contexts}"
TRAINING_ROOT="${SAFE100_SPECIALIST_TRAINING_ROOT:-$ROOT/artifacts/specialist_v17/training}"
OUTPUT="${SAFE100_AUDIT_OUTPUT_DIR:-$ROOT/artifacts/specialist_v17/final_audit}"
LOG="${SAFE100_AUDIT_LOG_PATH:-$ROOT/logs/specialist_v17/final_audit.log}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/audit_specialists_v17.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --context-dir "$CONTEXT_DIR" \
  --training-root "$TRAINING_ROOT" \
  --output-dir "$OUTPUT" \
  --adaptation-seeds 42 142 242 \
  --eval-batch-size 128 \
  --diagonal-episodes 512 \
  --off-diagonal-episodes 256 \
  --d0-episodes 256 \
  --bootstrap-samples 10000 \
  --audit-seed 1900000 \
  --device cuda:0 \
  2>&1 | tee "$LOG"
