#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
CONTEXT_DIR="${SAFE100_SPECIALIST_CONTEXT_DIR:-$ROOT/artifacts/specialist_v17/contexts}"
TRAINING_ROOT="${SAFE100_SPECIALIST_TRAINING_ROOT:-$ROOT/artifacts/specialist_v17/training}"
TRAINING_MANIFEST="${SAFE100_SPECIALIST_TRAINING_MANIFEST:-$REPO/results/online/specialist_v17/formal/training_manifest.json}"
PROTOCOL="${SAFE100_DIAGONAL_V18_PROTOCOL_FILE:-$REPO/results/online/specialist_v18/protocol.json}"
OUTPUT="${SAFE100_DIAGONAL_V18_OUTPUT_DIR:-$ROOT/artifacts/specialist_v18/diagonal_audit}"
LOG="${SAFE100_DIAGONAL_V18_LOG_PATH:-$ROOT/logs/specialist_v18/diagonal_audit.log}"
: "${SAFE100_DIAGONAL_V18_PROTOCOL_COMMIT:?set SAFE100_DIAGONAL_V18_PROTOCOL_COMMIT to the frozen protocol commit}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/audit_specialists_diagonal_v18.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --context-dir "$CONTEXT_DIR" \
  --training-root "$TRAINING_ROOT" \
  --training-manifest "$TRAINING_MANIFEST" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$SAFE100_DIAGONAL_V18_PROTOCOL_COMMIT" \
  --output-dir "$OUTPUT" \
  --adaptation-seeds 42 142 242 \
  --eval-batch-size 128 \
  --target-episodes 512 \
  --d0-episodes 256 \
  --bootstrap-samples 10000 \
  --audit-seed 3100000 \
  --bootstrap-seed 4000000 \
  --device cuda:0 \
  2>&1 | tee "$LOG"
