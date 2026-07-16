#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$REPO/results/models/cbf/model_1500.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/results/online/refine_dq}"

cd "$REPO"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" MUJOCO_GL=egl \
python experiments/scripts/online_refine_stairs.py \
  --repo "$REPO" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --train-domain DQ \
  --neighbor-domain DQN \
  --baseline-domains D0 DQ DQN \
  --num-envs 8 \
  --rollout-steps 768 \
  --critic-burn-in-rounds 2 \
  --online-rounds 3 \
  --actor-learning-rate 5e-6 \
  --critic-learning-rate 1e-4 \
  --pre-intervention-weight 1.0 \
  --std-scale-from-base 0.25 \
  --eval-num-envs 32 \
  --eval-num-episodes 32 \
  --gate-device cuda:0 \
  --gate-repeats 2 \
  --seed 42
