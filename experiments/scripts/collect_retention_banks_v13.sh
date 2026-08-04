#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
CHECKPOINT="${SAFE100_RESUME_CHECKPOINT:-$ROOT/artifacts/online_framework_v2/online_dqh_offgate_round5_v7c/accepted_final.pt}"
OUTPUT="${SAFE100_RETENTION_BANK_DIR:-$ROOT/artifacts/retention_v13/banks}"
BANK_SIZE="${SAFE100_RETENTION_BANK_SIZE:-24000}"
NUM_ENVS="${SAFE100_RETENTION_BANK_NUM_ENVS:-64}"

mkdir -p "$OUTPUT"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/collect_retention_observation_bank.py \
  --repo "$REPO" \
  --task Unitree-G1-Stairs-Online-D0 \
  --domain D0 \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-kind online \
  --bank-size "$BANK_SIZE" \
  --num-envs "$NUM_ENVS" \
  --seed 17001 \
  --runtime-filter on \
  --device cuda:0 \
  --output-bank "$OUTPUT/D0_actor_observations.pt" \
  --output-manifest "$OUTPUT/D0_actor_observations.json"

"$PYTHON" experiments/scripts/collect_retention_observation_bank.py \
  --repo "$REPO" \
  --task Unitree-G1-Stairs-Online-DQNH \
  --domain DQNH \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-kind online \
  --bank-size "$BANK_SIZE" \
  --num-envs "$NUM_ENVS" \
  --seed 18001 \
  --runtime-filter on \
  --device cuda:0 \
  --output-bank "$OUTPUT/DQNH_actor_observations.pt" \
  --output-manifest "$OUTPUT/DQNH_actor_observations.json"
