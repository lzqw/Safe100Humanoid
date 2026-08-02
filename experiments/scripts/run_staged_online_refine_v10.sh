#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASE="${SAFE100_BASE_CHECKPOINT:-/home/carla/LZQW/SAFE100/Safe100Humanoid_publish/results/models/cbf/model_1500.pt}"
OUTPUT="${SAFE100_OUTPUT_ROOT:-$ROOT/artifacts/online_framework_v3/staged_dq_to_d4_v10}"
LOG="${SAFE100_LOG_PATH:-$ROOT/logs/online_framework_v3/staged_dq_to_d4_v10.log}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

exec "$PYTHON" experiments/scripts/staged_online_refine_stairs.py \
  --repo "$REPO" \
  --python "$PYTHON" \
  --base-checkpoint "$BASE" \
  --output-root "$OUTPUT" \
  --num-envs 32 \
  --rollout-steps 256 \
  --critic-burn-in-rounds 1 \
  --dq-rounds 5 \
  --d4-rounds 3 \
  --eval-num-envs 16 \
  --gate-repeats 3 \
  --actor-learning-rate 5e-6 \
  --minimum-dq-success 0.60 \
  --no-closed-loop-centering \
  --seed 42 \
  --device cuda:0 \
  2>&1 | tee "$LOG"
