#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_V21_REPO:-$ROOT/worktrees/v21_formal}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
PROTOCOL_COMMIT="${SAFE100_V21_PROTOCOL_COMMIT:?set SAFE100_V21_PROTOCOL_COMMIT}"
PROTOCOL="${SAFE100_V21_PROTOCOL_FILE:?set SAFE100_V21_PROTOCOL_FILE}"
ARTIFACT_ROOT="${SAFE100_V21_ARTIFACT_ROOT:-$ROOT/artifacts/specialist_v21}"
LOG_ROOT="${SAFE100_V21_LOG_ROOT:-$ROOT/logs/specialist_v21}"
CONTEXT_ID="${1:-}"
METHOD_ROLE="${2:-}"
BETA="${3:-}"

case "$CONTEXT_ID" in
  L_dev|L1|L2|L3|L4|L5) MODE="lateral" ;;
  C_dev|C1|C2|C3|C4|C5) MODE="contact_stability" ;;
  *) echo "unknown v21 context: $CONTEXT_ID" >&2; exit 2 ;;
esac
case "$METHOD_ROLE" in control|v21) ;; *) echo "method must be control or v21" >&2; exit 2 ;; esac
test -n "$BETA"

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
SEED="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["adaptation_seeds"][sys.argv[2]])' "$PROTOCOL" "$CONTEXT_ID")"
LABEL="beta_${BETA//./p}"
CONTEXT="$REPO/results/online/specialist_v21/contexts/$CONTEXT_ID.json"
OUTPUT="${SAFE100_OUTPUT_DIR:-$ARTIFACT_ROOT/training/$CONTEXT_ID/${METHOD_ROLE}_${LABEL}}"
LOG="${SAFE100_LOG_PATH:-$LOG_ROOT/train_${CONTEXT_ID}_${METHOD_ROLE}_${LABEL}.log}"
mkdir -p "$OUTPUT" "$(dirname "$LOG")"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
DEVELOPMENT_ARGS=()
case "$CONTEXT_ID" in L_dev|C_dev) DEVELOPMENT_ARGS=(--development) ;; esac

"$PYTHON" experiments/scripts/refine_deployment_v21.py \
  --repo "$REPO" \
  --base-policy-checkpoint "$BASELINE" \
  --deployment-context "$CONTEXT" \
  --protocol-file "$PROTOCOL" \
  --protocol-commit "$PROTOCOL_COMMIT" \
  --context-id "$CONTEXT_ID" \
  --mode "$MODE" \
  --method-role "$METHOD_ROLE" \
  --matched-success-preservation-beta "$BETA" \
  --seed "$SEED" \
  --output-dir "$OUTPUT" \
  --num-envs 64 \
  --rollout-steps 1024 \
  --maximum-rounds 8 \
  --candidate-screen-episodes 64 \
  --candidate-confirm-episodes 64 \
  --candidate-confirm-blocks 3 \
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
  "${DEVELOPMENT_ARGS[@]}" \
  2>&1 | tee -a "$LOG"
