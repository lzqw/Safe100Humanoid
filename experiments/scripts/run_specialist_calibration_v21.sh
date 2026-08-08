#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_V21_REPO:-$ROOT/worktrees/v21_final_calibration}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
PROTOCOL_COMMIT="${SAFE100_V21_PROTOCOL_COMMIT:?set SAFE100_V21_PROTOCOL_COMMIT}"
PROTOCOL="${SAFE100_V21_PROTOCOL_FILE:-$REPO/results/online/specialist_v21/protocol_precalibration_replacement.json}"
ARTIFACT_ROOT="${SAFE100_V21_ARTIFACT_ROOT:-$ROOT/artifacts/specialist_v21_replacement}"
LOG_ROOT="${SAFE100_V21_LOG_ROOT:-$ROOT/logs/specialist_v21_replacement}"
CONTEXT_ROOT="${SAFE100_V21_CONTEXT_ROOT:-$REPO/results/online/specialist_v21/contexts_replacement}"
CALIBRATION_SUMMARY_ROOT="${SAFE100_V21_CALIBRATION_SUMMARY_ROOT:-$REPO/results/online/specialist_v21/calibration/replacement}"
CONTEXT_ID="${1:-}"

case "$CONTEXT_ID" in
  L_dev|C_dev|L1|L2|L3|L4|L5|C1|C2|C3|C4|C5) ;;
  *) echo "usage: $0 L_dev|C_dev|L1|L2|L3|L4|L5|C1|C2|C3|C4|C5" >&2; exit 2 ;;
esac

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
OUTPUT="$ARTIFACT_ROOT/calibration/$CONTEXT_ID"
CONTEXT="$CONTEXT_ROOT/$CONTEXT_ID.json"
SUMMARY_ROOT="$CALIBRATION_SUMMARY_ROOT/$CONTEXT_ID"
LOG="$LOG_ROOT/calibration_${CONTEXT_ID}.log"
mkdir -p "$OUTPUT" "$(dirname "$CONTEXT")" "$SUMMARY_ROOT" "$(dirname "$LOG")"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" experiments/scripts/calibrate_deployment_contexts_v21.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$PROTOCOL_COMMIT" \
  --context-id "$CONTEXT_ID" \
  --output-dir "$OUTPUT" \
  --context-output "$CONTEXT" \
  --num-episodes 512 \
  --eval-batch-size 128 \
  --device cuda:0 \
  2>&1 | tee -a "$LOG"

cp "$OUTPUT/calibration_progress.json" "$SUMMARY_ROOT/calibration_progress.json"
cp "$OUTPUT/calibration_summary.json" "$SUMMARY_ROOT/calibration_summary.json"
