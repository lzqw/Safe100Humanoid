#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASE="${SAFE100_BASE_CHECKPOINT:-/home/carla/LZQW/SAFE100/Safe100Humanoid_publish/results/models/cbf/model_1500.pt}"
RESUME="${SAFE100_RESUME_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
CONTEXT="${SAFE100_DEPLOYMENT_CONTEXT:-$ROOT/artifacts/failure_focused_v15/frozen_dqh_medium_context.json}"
CLASSIFICATION="${SAFE100_FAILURE_CLASSIFICATION:-$ROOT/artifacts/dominant_failure_v16/failure_classification/failure_classification.json}"
SEED="${SAFE100_SEED:-42}"
OUTPUT="${SAFE100_OUTPUT_DIR:-$ROOT/artifacts/dominant_failure_v16/train_seed${SEED}}"
LOG="${SAFE100_LOG_PATH:-$ROOT/logs/dominant_failure_v16/train_seed${SEED}.log}"

case "$SEED" in
  42|142|242) ;;
  *) echo "formal v16 seed must be 42, 142, or 242" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/brief_ppo_refine_stairs.py \
  --repo "$REPO" \
  --base-checkpoint "$BASE" \
  --resume-online-checkpoint "$RESUME" \
  --no-resume-hard-case-bank \
  --output-dir "$OUTPUT" \
  --failure-focused-v15 \
  --dominant-failure-type lateral_heading_drift \
  --failure-classification "$CLASSIFICATION" \
  --deployment-context "$CONTEXT" \
  --train-domain DQHMED \
  --neighbor-domain DQNHMED \
  --seed "$SEED" \
  --num-envs 64 \
  --rollout-steps 1024 \
  --online-rounds 5 \
  --critic-burn-in-rounds 0 \
  --candidate-num-episodes 128 \
  --d0-check-num-episodes 128 \
  --final-eval-num-episodes 128 \
  --candidate-fractions 0.5 1.0 1.5 \
  --hard-case-fraction 0.15625 \
  --hard-case-policy-weight 0.75 \
  --hard-case-pre-steps 10 \
  --hard-case-capacity 256 \
  --late-failure-minimum-steps 50 \
  --late-failure-maximum-steps 150 \
  --failure-discovery-max-rollouts 8 \
  --fall-redistribution-horizon 100 \
  --fall-redistribution-decay 0.97 \
  --fall-redistribution-amount 2.0 \
  --actor-learning-rate 5e-6 \
  --critic-learning-rate 1e-4 \
  --std-scale-from-base 0.35 \
  --fall-penalty-weight -100 \
  --device cuda:0 \
  --gate-device cuda:0 \
  2>&1 | tee "$LOG"
