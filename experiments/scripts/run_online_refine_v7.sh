#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/carla/LZQW/SAFE100/humanoid
REPO="$ROOT/third_party/unitree_rl_mjlab"
PYTHON="$ROOT/workspace/conda_env/bin/python"
BASE=/home/carla/LZQW/SAFE100/Safe100Humanoid_publish/results/models/cbf/model_1500.pt
RESUME="$ROOT/artifacts/online_framework_v2/online_dqh_offgate_5round_v7b/accepted_round_001.pt"
OUTPUT="$ROOT/artifacts/online_framework_v2/online_dqh_offgate_round5_v7c"
LOG="$ROOT/logs/online_framework_v2/online_dqh_offgate_round5_v7c.log"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"

export MUJOCO_GL=egl
exec "$PYTHON" experiments/scripts/online_refine_stairs.py \
  --repo . \
  --base-checkpoint "$BASE" \
  --resume-online-checkpoint "$RESUME" \
  --output-dir "$OUTPUT" \
  --num-envs 32 \
  --rollout-steps 512 \
  --critic-burn-in-rounds 1 \
  --online-rounds 1 \
  --eval-num-envs 16 \
  --eval-num-episodes 16 \
  --gate-repeats 3 \
  --train-domain DQH \
  --neighbor-domain DQNH \
  --baseline-domains D0 DQH DQNH \
  --train-runtime-filter on \
  --gate-runtime-filter off \
  --hard-case-fraction 0.25 \
  --hard-case-pre-steps 10 \
  --hard-case-capacity 256 \
  --actor-learning-rate 2e-6 \
  --critic-learning-rate 1e-4 \
  --pre-intervention-weight 0.20 \
  --target-intervention-per-riser 0.10 \
  --std-adaptation-rate 0.10 \
  --seed 42 \
  --device cuda:0 \
  --gate-device cuda:0 \
  2>&1 | tee "$LOG"
