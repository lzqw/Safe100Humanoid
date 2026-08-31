#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASE="${SAFE100_BASE_CHECKPOINT:-/home/carla/LZQW/SAFE100/Safe100Humanoid_publish/results/models/cbf/model_1500.pt}"
RESUME="${SAFE100_RESUME_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
SEED="${SAFE100_SEED:-42}"
OUTPUT="${SAFE100_OUTPUT_DIR:-$ROOT/artifacts/brief_ppo_v14/train_seed${SEED}}"
LOG="${SAFE100_LOG_PATH:-$ROOT/logs/brief_ppo_v14/train_seed${SEED}.log}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/brief_ppo_refine_stairs.py \
  --repo "$REPO" \
  --base-checkpoint "$BASE" \
  --resume-online-checkpoint "$RESUME" \
  --resume-hard-case-bank \
  --output-dir "$OUTPUT" \
  --train-domain DQH \
  --neighbor-domain DQNH \
  --seed "$SEED" \
  --num-envs 64 \
  --rollout-steps 1024 \
  --online-rounds 5 \
  --critic-burn-in-rounds 0 \
  --candidate-num-episodes 128 \
  --d0-check-num-episodes 128 \
  --final-eval-num-episodes 128 \
  --candidate-fractions 0.5 1.0 \
  --hard-case-fraction 0.20 \
  --hard-case-policy-weight 0.5 \
  --hard-case-pre-steps 10 \
  --hard-case-capacity 256 \
  --actor-learning-rate 2e-6 \
  --critic-learning-rate 1e-4 \
  --std-scale-from-base 0.35 \
  --fall-penalty-weight -200 \
  --device cuda:0 \
  --gate-device cuda:0 \
  2>&1 | tee "$LOG"
