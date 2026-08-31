#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASE="${SAFE100_BASE_CHECKPOINT:-/home/carla/LZQW/SAFE100/Safe100Humanoid_publish/results/models/cbf/model_1500.pt}"
RESUME="${SAFE100_RESUME_CHECKPOINT:-$ROOT/artifacts/online_framework_v2/online_dqh_offgate_round5_v7c/accepted_final.pt}"
BANK_DIR="${SAFE100_RETENTION_BANK_DIR:-$ROOT/artifacts/retention_v13/banks}"
ARM="${SAFE100_RETENTION_ARM:-B}"
BASELINE_REUSE="${SAFE100_BASELINE_EVAL_PATH:-}"
BASELINE_ARGS=()
if [[ -n "$BASELINE_REUSE" ]]; then
  BASELINE_ARGS=(--reuse-baseline-eval "$BASELINE_REUSE")
fi

case "$ARM" in
  A)
    ARM_SLUG="arm_a_v12_global_anchor"
    ANCHOR_ARGS=(
      --base-anchor-weight 0.01
      --d0-retention-anchor-weight 0.0
      --neighbor-retention-anchor-weight 0.0
    )
    ;;
  B)
    ARM_SLUG="arm_b_state_retention"
    ANCHOR_ARGS=(
      --base-anchor-weight 0.0
      --d0-retention-bank "$BANK_DIR/D0_actor_observations.pt"
      --neighbor-retention-bank "$BANK_DIR/DQNH_actor_observations.pt"
      --d0-retention-anchor-weight 0.02
      --neighbor-retention-anchor-weight 0.01
      --d0-retention-anchor-kl-budget 0.002
      --neighbor-retention-anchor-kl-budget 0.002
      --retention-anchor-adaptation-rate 10.0
      --maximum-retention-anchor-weight 0.20
      --retention-anchor-batch-size 4096
    )
    ;;
  *)
    echo "SAFE100_RETENTION_ARM must be A or B" >&2
    exit 2
    ;;
esac

OUTPUT="${SAFE100_OUTPUT_DIR:-$ROOT/artifacts/retention_v13/$ARM_SLUG}"
LOG="${SAFE100_LOG_PATH:-$ROOT/logs/retention_v13/$ARM_SLUG.log}"
mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$REPO"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

"$PYTHON" experiments/scripts/online_refine_stairs.py \
  --repo "$REPO" \
  --base-checkpoint "$BASE" \
  --resume-online-checkpoint "$RESUME" \
  --no-resume-hard-case-bank \
  --output-dir "$OUTPUT" \
  --num-envs 64 \
  --rollout-steps 1024 \
  --critic-burn-in-rounds 2 \
  --critic-burn-in-max-rounds 4 \
  --critic-min-explained-variance 0.45 \
  --online-rounds 3 \
  --eval-num-envs 16 \
  --eval-num-episodes 16 \
  --gate-repeats 3 \
  --candidate-fractions 0.25 0.5 1.0 1.5 \
  --candidate-screen-num-envs 64 \
  --candidate-screen-repeats 1 \
  --seed 42 \
  --actor-learning-rate 2e-6 \
  --critic-learning-rate 1e-4 \
  --std-scale-from-base 0.35 \
  "${ANCHOR_ARGS[@]}" \
  --task-first-constrained \
  --fall-multiplier-learning-rate 1.0 \
  --intervention-multiplier-learning-rate 0.10 \
  --maximum-cost-multiplier 20 \
  --intervention-budget-slack 1.05 \
  --hard-case-fraction 0.08 \
  --hard-case-policy-weight 0.0 \
  --neighbor-command-fraction 0.08 \
  --correction-distillation-weight 0.0 \
  --correction-success-horizon 100 \
  --risk-horizon 50 \
  --strong-intervention-fraction 0.5 \
  --risk-loss-coef 1.0 \
  --minimum-normal-complete-episodes 32 \
  --late-critic-risers 7 8 9 \
  --critic-min-samples-per-late-riser 64 \
  --critic-min-fall-events 2 \
  --risk-maximum-brier 0.25 \
  --risk-minimum-auc 0.55 \
  --minimum-pre-fall-cost-value-rise 0.0 \
  --maximum-target-fall-rate 1.0 \
  --train-runtime-filter on \
  --gate-runtime-filter on \
  --no-independence-audit \
  --no-adaptive-std \
  --train-domain DQH \
  --neighbor-domain DQNH \
  --baseline-domains D0 DQH DQNH \
  "${BASELINE_ARGS[@]}" \
  --device cuda:0 \
  --gate-device cuda:0 \
  2>&1 | tee "$LOG"
