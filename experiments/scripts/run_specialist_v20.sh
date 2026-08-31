#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/worktrees/v20_formal}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
MODE="${SAFE100_SPECIALIST_MODE:?set SAFE100_SPECIALIST_MODE}"
SEED="${SAFE100_SEED:?set SAFE100_SEED}"
PROTOCOL_COMMIT="${SAFE100_V20_PROTOCOL_COMMIT:?set SAFE100_V20_PROTOCOL_COMMIT}"
CONTEXT="${SAFE100_SPECIALIST_CONTEXT:-$ROOT/artifacts/specialist_v20/contexts/${MODE}.json}"
PROTOCOL="${SAFE100_V20_PROTOCOL_FILE:-$REPO/results/online/specialist_v20/protocol.json}"
OUTPUT="${SAFE100_OUTPUT_DIR:-$ROOT/artifacts/specialist_v20/training/${MODE}/seed${SEED}}"
LOG="${SAFE100_LOG_PATH:-$ROOT/logs/specialist_v20/train_${MODE}_seed${SEED}.log}"

case "$MODE" in
  lateral|contact_stability) ;;
  *) echo "unknown v20 specialist mode: $MODE" >&2; exit 2 ;;
esac
case "$SEED" in
  73|173|273|373|473) ;;
  *) echo "formal v20 seed must be 73, 173, 273, 373, or 473" >&2; exit 2 ;;
esac

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
mkdir -p "$OUTPUT" "$(dirname "$LOG")"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/refine_specialist_v20.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --deployment-context "$CONTEXT" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$PROTOCOL_COMMIT" \
  --mode "$MODE" \
  --seed "$SEED" \
  --output-dir "$OUTPUT" \
  --num-envs 64 \
  --rollout-steps 1024 \
  --maximum-rounds 8 \
  --candidate-screen-episodes 64 \
  --candidate-confirm-episodes 128 \
  --d0-check-num-episodes 128 \
  --final-eval-num-episodes 128 \
  --candidate-fractions 0.5 1.0 1.5 \
  --failure-start-fraction 0.1875 \
  --success-start-fraction 0.1875 \
  --failure-policy-weight 1.0 \
  --success-policy-weight 1.25 \
  --bank-capacity 256 \
  --success-pool-capacity 1024 \
  --failure-discovery-max-rollouts 12 \
  --actor-learning-rate 5e-6 \
  --critic-learning-rate 1e-4 \
  --fall-penalty-weight -100 \
  --fall-redistribution-horizon 100 \
  --fall-redistribution-decay 0.97 \
  --fall-redistribution-amount 2.0 \
  --device cuda:0 \
  --gate-device cuda:0 \
  2>&1 | tee -a "$LOG"
