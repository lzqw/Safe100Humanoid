#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/worktrees/v19_formal}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
MODE="${SAFE100_SPECIALIST_MODE:?set SAFE100_SPECIALIST_MODE to lateral or contact_stability}"
PROTOCOL_COMMIT="${SAFE100_V19_PROTOCOL_COMMIT:?set SAFE100_V19_PROTOCOL_COMMIT to the frozen v19 commit}"
OUTPUT="${SAFE100_CALIBRATION_OUTPUT_DIR:-$ROOT/artifacts/specialist_v19/calibration/$MODE}"
CONTEXT="${SAFE100_SPECIALIST_CONTEXT:-$ROOT/artifacts/specialist_v19/contexts/${MODE}.json}"
LOG="${SAFE100_CALIBRATION_LOG_PATH:-$ROOT/logs/specialist_v19/calibration_${MODE}.log}"

case "$MODE" in
  lateral)
    EVALUATION_SEED_BASE=4700000
    CANDIDATES=(7106 7107 7108 7109 7110 7111)
    ;;
  contact_stability)
    EVALUATION_SEED_BASE=4710000
    CANDIDATES=(7217 7218 7219)
    ;;
  *) echo "unknown v19 specialist mode: $MODE" >&2; exit 2 ;;
esac

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
mkdir -p "$OUTPUT" "$(dirname "$CONTEXT")" "$(dirname "$LOG")"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/calibrate_specialist_contexts_v19.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --mode "$MODE" \
  --output-dir "$OUTPUT" \
  --context-output "$CONTEXT" \
  --candidate-seeds "${CANDIDATES[@]}" \
  --num-episodes 512 \
  --eval-batch-size 128 \
  --evaluation-seed-base "$EVALUATION_SEED_BASE" \
  --device cuda:0 \
  2>&1 | tee "$LOG"
