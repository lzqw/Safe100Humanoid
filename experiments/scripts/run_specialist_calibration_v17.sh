#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
MODE="${SAFE100_SPECIALIST_MODE:?set SAFE100_SPECIALIST_MODE to lateral, cbf, or balance}"
OUTPUT="${SAFE100_CALIBRATION_OUTPUT_DIR:-$ROOT/artifacts/specialist_v17/calibration/$MODE}"
CONTEXT="${SAFE100_SPECIALIST_CONTEXT:-$ROOT/artifacts/specialist_v17/contexts/${MODE}.json}"
LOG="${SAFE100_CALIBRATION_LOG_PATH:-$ROOT/logs/specialist_v17/calibration_${MODE}.log}"

case "$MODE" in
  lateral) EVALUATION_SEED_BASE=1831000; DEFAULT_CANDIDATE=2115 ;;
  cbf) EVALUATION_SEED_BASE=1832100; DEFAULT_CANDIDATE=2212 ;;
  balance) EVALUATION_SEED_BASE=1833000; DEFAULT_CANDIDATE=2314 ;;
  *) echo "unknown specialist mode: $MODE" >&2; exit 2 ;;
esac

read -r -a SEEDS <<< "${SAFE100_SPECIALIST_CANDIDATE_SEEDS:-$DEFAULT_CANDIDATE}"

mkdir -p "$OUTPUT" "$(dirname "$CONTEXT")" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/calibrate_specialist_contexts_v17.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --mode "$MODE" \
  --output-dir "$OUTPUT" \
  --context-output "$CONTEXT" \
  --candidate-seeds "${SEEDS[@]}" \
  --num-episodes 512 \
  --eval-batch-size 128 \
  --evaluation-seed-base "$EVALUATION_SEED_BASE" \
  --device cuda:0 \
  2>&1 | tee "$LOG"
