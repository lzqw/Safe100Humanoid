#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
MODE="${SAFE100_SPECIALIST_MODE:?set SAFE100_SPECIALIST_MODE to lateral, cbf, or balance}"
SEED="${SAFE100_SEED:?set SAFE100_SEED to 42, 142, or 242}"
CONTEXT="${SAFE100_SPECIALIST_CONTEXT:-$ROOT/artifacts/specialist_v17/contexts/${MODE}.json}"
OUTPUT="${SAFE100_OUTPUT_DIR:-$ROOT/artifacts/specialist_v17/training/${MODE}/seed${SEED}}"
LOG="${SAFE100_LOG_PATH:-$ROOT/logs/specialist_v17/train_${MODE}_seed${SEED}.log}"

case "$MODE" in
  lateral|cbf|balance) ;;
  *) echo "unknown specialist mode: $MODE" >&2; exit 2 ;;
esac
case "$SEED" in
  42|142|242) ;;
  *) echo "formal v17 seed must be 42, 142, or 242" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/refine_specialist_v17.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --deployment-context "$CONTEXT" \
  --mode "$MODE" \
  --output-dir "$OUTPUT" \
  --seed "$SEED" \
  --num-envs 64 \
  --rollout-steps 1024 \
  --online-rounds 5 \
  --candidate-num-episodes 128 \
  --candidate-eval-repeats 2 \
  --d0-check-num-episodes 128 \
  --final-eval-num-episodes 128 \
  --candidate-fractions 0.5 1.0 1.5 \
  --failure-start-fraction 0.15 \
  --success-start-fraction 0.15 \
  --failure-policy-weight 0.75 \
  --bank-capacity 256 \
  --success-pool-capacity 512 \
  --failure-discovery-max-rollouts 8 \
  --actor-learning-rate 5e-6 \
  --critic-learning-rate 1e-4 \
  --fall-penalty-weight -100 \
  --fall-redistribution-horizon 100 \
  --fall-redistribution-decay 0.97 \
  --fall-redistribution-amount 2.0 \
  --device cuda:0 \
  --gate-device cuda:0 \
  2>&1 | tee "$LOG"
