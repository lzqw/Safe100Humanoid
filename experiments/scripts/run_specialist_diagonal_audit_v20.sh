#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/worktrees/v20_formal}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
MODE="${SAFE100_SPECIALIST_MODE:?set SAFE100_SPECIALIST_MODE}"
CONTEXT="${SAFE100_SPECIALIST_CONTEXT:-$ROOT/artifacts/specialist_v20/contexts/${MODE}.json}"
TRAINING_ROOT="${SAFE100_SPECIALIST_TRAINING_ROOT:-$ROOT/artifacts/specialist_v20/training}"
PROTOCOL="${SAFE100_V20_PROTOCOL_FILE:-$REPO/results/online/specialist_v20/protocol.json}"
PROTOCOL_COMMIT="${SAFE100_V20_PROTOCOL_COMMIT:?set SAFE100_V20_PROTOCOL_COMMIT}"
OUTPUT="${SAFE100_V20_AUDIT_OUTPUT_DIR:-$ROOT/artifacts/specialist_v20/audit/${MODE}}"
LOG="${SAFE100_V20_AUDIT_LOG_PATH:-$ROOT/logs/specialist_v20/audit_${MODE}.log}"

case "$MODE" in
  lateral|contact_stability) ;;
  *) echo "unknown v20 specialist mode: $MODE" >&2; exit 2 ;;
esac

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
mkdir -p "$OUTPUT" "$(dirname "$LOG")"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/audit_specialist_diagonal_v20.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --mode "$MODE" \
  --context "$CONTEXT" \
  --training-root "$TRAINING_ROOT" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$PROTOCOL_COMMIT" \
  --output-dir "$OUTPUT" \
  --adaptation-seeds 73 173 273 373 473 \
  --eval-batch-size 128 \
  --target-episodes 512 \
  --d0-episodes 256 \
  --bootstrap-samples 10000 \
  --audit-seed 5500000 \
  --bootstrap-seed 6500000 \
  --device cuda:0 \
  2>&1 | tee "$LOG"

"$PYTHON" experiments/scripts/verify_specialist_audit_v20.py \
  --repo "$REPO" \
  --summary "$OUTPUT/final_audit_summary.json" \
  --paired-csv "$OUTPUT/paired_episode_metrics.csv" \
  --output "$OUTPUT/verification.json" \
  2>&1 | tee -a "$LOG"

"$PYTHON" experiments/scripts/collect_mechanism_telemetry_v20.py \
  --repo "$REPO" \
  --mode "$MODE" \
  --context "$CONTEXT" \
  --audit-dir "$OUTPUT" \
  --output-dir "$OUTPUT" \
  --device cuda:0 \
  2>&1 | tee -a "$LOG"
