#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_V22_REPO:-$ROOT/worktrees/v22_effect_first}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
PROTOCOL_COMMIT="${SAFE100_V22_PROTOCOL_COMMIT:?set SAFE100_V22_PROTOCOL_COMMIT}"
PROTOCOL="${SAFE100_V22_PROTOCOL_FILE:?set SAFE100_V22_PROTOCOL_FILE}"
ARTIFACT_ROOT="${SAFE100_V22_ARTIFACT_ROOT:-$ROOT/artifacts/specialist_v22}"
LOG_ROOT="${SAFE100_V22_LOG_ROOT:-$ROOT/logs/specialist_v22}"
ACTION="${1:-}"
CONTEXT_ID="${2:-}"

case "$CONTEXT_ID" in
  L_effect) MODE="lateral" ;;
  C_effect) MODE="contact_stability" ;;
  *) echo "usage: $0 calibrate|train-test L_effect|C_effect" >&2; exit 2 ;;
esac
case "$ACTION" in calibrate|train-test) ;; *) echo "unknown v22 action: $ACTION" >&2; exit 2 ;; esac

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$ACTION" == "calibrate" ]]; then
  OUTPUT="${SAFE100_V22_OUTPUT_DIR:-$ARTIFACT_ROOT/calibration/$CONTEXT_ID}"
  CONTEXT="${SAFE100_V22_CONTEXT_OUTPUT:-$ARTIFACT_ROOT/contexts/$CONTEXT_ID.json}"
  LOG="${SAFE100_V22_LOG_PATH:-$LOG_ROOT/calibration_${CONTEXT_ID}.log}"
  mkdir -p "$OUTPUT" "$(dirname "$CONTEXT")" "$(dirname "$LOG")"
  "$PYTHON" experiments/scripts/calibrate_effect_first_v22.py \
    --repo "$REPO" \
    --base-policy-checkpoint "$BASELINE" \
    --protocol-file "$PROTOCOL" \
    --protocol-commit "$PROTOCOL_COMMIT" \
    --context-id "$CONTEXT_ID" \
    --output-dir "$OUTPUT" \
    --context-output "$CONTEXT" \
    --device cuda:0 \
    2>&1 | tee -a "$LOG"
  exit 0
fi

CONTEXT="${SAFE100_V22_CONTEXT:?set SAFE100_V22_CONTEXT}"
TRAINING="${SAFE100_V22_TRAINING_DIR:-$ARTIFACT_ROOT/training/$CONTEXT_ID}"
FINAL_TEST="${SAFE100_V22_FINAL_TEST_DIR:-$ARTIFACT_ROOT/final_test/$CONTEXT_ID}"
TRAIN_LOG="${SAFE100_V22_TRAIN_LOG:-$LOG_ROOT/train_${CONTEXT_ID}.log}"
TEST_LOG="${SAFE100_V22_TEST_LOG:-$LOG_ROOT/final_test_${CONTEXT_ID}.log}"
mkdir -p "$TRAINING" "$FINAL_TEST" "$(dirname "$TRAIN_LOG")" "$(dirname "$TEST_LOG")"
SEED="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["adaptation_seeds"][sys.argv[2]])' "$PROTOCOL" "$CONTEXT_ID")"

"$PYTHON" experiments/scripts/refine_effect_first_v22.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --deployment-context "$CONTEXT" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$PROTOCOL_COMMIT" \
  --context-id "$CONTEXT_ID" \
  --mode "$MODE" \
  --seed "$SEED" \
  --output-dir "$TRAINING" \
  --device cuda:0 \
  --gate-device cuda:0 \
  2>&1 | tee -a "$TRAIN_LOG"

"$PYTHON" experiments/scripts/test_effect_first_v22.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --context-id "$CONTEXT_ID" \
  --context "$CONTEXT" \
  --training-dir "$TRAINING" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$PROTOCOL_COMMIT" \
  --output-dir "$FINAL_TEST" \
  --device cuda:0 \
  2>&1 | tee -a "$TEST_LOG"
