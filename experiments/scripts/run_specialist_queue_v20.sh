#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
TRAINING_REPO="${SAFE100_V20_TRAINING_REPO:-$ROOT/worktrees/v20_formal}"
AUDIT_REPO="${SAFE100_V20_AUDIT_REPO:-${SAFE100_MJLAB_REPO:-$ROOT/worktrees/v20_audit_amendment}}"
MODE="${1:-${SAFE100_SPECIALIST_MODE:-}}"
PROTOCOL_COMMIT="${SAFE100_V20_PROTOCOL_COMMIT:?set SAFE100_V20_PROTOCOL_COMMIT}"
AUDIT_COMMIT="${SAFE100_V20_AUDIT_COMMIT:?set SAFE100_V20_AUDIT_COMMIT}"
QUEUE_LOG="${SAFE100_V20_QUEUE_LOG:-$ROOT/logs/specialist_v20/queue_${MODE}.log}"

case "$MODE" in
  lateral|contact_stability) ;;
  *) echo "usage: $0 lateral|contact_stability" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$QUEUE_LOG")"
cd "$TRAINING_REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
cd "$AUDIT_REPO"
test "$(git rev-parse HEAD)" = "$AUDIT_COMMIT"
git diff --quiet
git diff --cached --quiet

printf '%s mode=%s event=queue_started training_commit=%s audit_commit=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$PROTOCOL_COMMIT" \
  "$AUDIT_COMMIT" \
  | tee -a "$QUEUE_LOG"
for seed in 73 173 273 373 473; do
  printf '%s mode=%s seed=%s event=training_started\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$seed" \
    | tee -a "$QUEUE_LOG"
  SAFE100_SPECIALIST_MODE="$MODE" \
  SAFE100_SEED="$seed" \
  SAFE100_V20_PROTOCOL_COMMIT="$PROTOCOL_COMMIT" \
  SAFE100_MJLAB_REPO="$TRAINING_REPO" \
    bash "$TRAINING_REPO/experiments/scripts/run_specialist_v20.sh" \
    2>&1 | tee -a "$QUEUE_LOG"
  printf '%s mode=%s seed=%s event=training_completed\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" "$seed" \
    | tee -a "$QUEUE_LOG"
done

printf '%s mode=%s event=audit_started\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" \
  | tee -a "$QUEUE_LOG"
SAFE100_SPECIALIST_MODE="$MODE" \
SAFE100_V20_PROTOCOL_COMMIT="$PROTOCOL_COMMIT" \
SAFE100_V20_AUDIT_COMMIT="$AUDIT_COMMIT" \
SAFE100_V20_AUDIT_REPO="$AUDIT_REPO" \
  bash "$AUDIT_REPO/experiments/scripts/run_specialist_diagonal_audit_v20.sh" \
  2>&1 | tee -a "$QUEUE_LOG"
printf '%s mode=%s event=queue_completed\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODE" \
  | tee -a "$QUEUE_LOG"
