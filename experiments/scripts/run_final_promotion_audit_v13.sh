#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
RUN_DIR="${SAFE100_RUN_DIR:?set SAFE100_RUN_DIR to one v13 arm output}"
OLD="${SAFE100_OLD_CHECKPOINT:?set SAFE100_OLD_CHECKPOINT to the accepted actor before the audited round}"
CANDIDATE="${SAFE100_CANDIDATE_CHECKPOINT:?set SAFE100_CANDIDATE_CHECKPOINT to candidate_round_NNN.pt}"
ROUND="${SAFE100_CANDIDATE_ROUND:?set SAFE100_CANDIDATE_ROUND to the one-based round}"
OUTPUT="${SAFE100_AUDIT_OUTPUT:-$RUN_DIR/final_audit_round_${ROUND}_512.json}"

cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

exec "$PYTHON" experiments/scripts/audit_task_first_candidate.py \
  --repo "$REPO" \
  --old-checkpoint "$OLD" \
  --candidate-checkpoint "$CANDIDATE" \
  --online-summary "$RUN_DIR/online_refinement_summary.json" \
  --baseline-eval "$RUN_DIR/baseline_ood_matrix.json" \
  --round "$ROUND" \
  --output "$OUTPUT" \
  --target-domain DQH \
  --retention-domain D0 \
  --neighbor-domain DQNH \
  --num-envs 64 \
  --repeats 8 \
  --seed 23000 \
  --device cuda:0 \
  --runtime-filter \
  --maximum-target-fall-rate 1.0
