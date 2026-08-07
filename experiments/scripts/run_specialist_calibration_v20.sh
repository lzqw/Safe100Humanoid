#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/worktrees/v20_precalibration}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
MODE="${SAFE100_SPECIALIST_MODE:?set SAFE100_SPECIALIST_MODE}"
PROTOCOL_COMMIT="${SAFE100_V20_PRECALIBRATION_COMMIT:?set SAFE100_V20_PRECALIBRATION_COMMIT}"
PROTOCOL="${SAFE100_V20_PRECALIBRATION_PROTOCOL:-$REPO/results/online/specialist_v20/protocol_precalibration.json}"
OUTPUT="${SAFE100_CALIBRATION_OUTPUT_DIR:-$ROOT/artifacts/specialist_v20/calibration/$MODE}"
CONTEXT="${SAFE100_SPECIALIST_CONTEXT:-$ROOT/artifacts/specialist_v20/contexts/${MODE}.json}"
LOG="${SAFE100_CALIBRATION_LOG_PATH:-$ROOT/logs/specialist_v20/calibration_${MODE}.log}"

case "$MODE" in
  lateral|contact_stability) ;;
  *) echo "unknown v20 specialist mode: $MODE" >&2; exit 2 ;;
esac

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
mapfile -t CANDIDATES < <(
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(*p["calibration"][sys.argv[2]+"_candidate_seeds"], sep="\n")' \
    "$PROTOCOL" "$MODE"
)
EVALUATION_SEED_BASE="$(
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(p["calibration"][sys.argv[2]+"_evaluation_seed_base"])' \
    "$PROTOCOL" "$MODE"
)"
test "${#CANDIDATES[@]}" -gt 0
mkdir -p "$OUTPUT" "$(dirname "$CONTEXT")" "$(dirname "$LOG")"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/calibrate_specialist_contexts_v20.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$PROTOCOL_COMMIT" \
  --mode "$MODE" \
  --output-dir "$OUTPUT" \
  --context-output "$CONTEXT" \
  --candidate-seeds "${CANDIDATES[@]}" \
  --num-episodes 512 \
  --eval-batch-size 128 \
  --evaluation-seed-base "$EVALUATION_SEED_BASE" \
  --device cuda:0 \
  2>&1 | tee -a "$LOG"
