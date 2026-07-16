#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

python scripts/train.py Unitree-G1-Stairs-Nominal \
  --env.scene.num-envs "${NUM_ENVS:-1024}" \
  --agent.max-iterations "${MAX_ITERATIONS:-1500}" \
  --agent.save-interval "${SAVE_INTERVAL:-500}" \
  --agent.seed "${SEED:-42}" \
  --agent.logger tensorboard \
  --agent.run-name "${RUN_NAME:-paper_nominal_curriculum_seed42_1024_1500}" \
  --enable-nan-guard True
