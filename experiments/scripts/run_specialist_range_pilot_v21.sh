#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_V21_REPO:-$ROOT/worktrees/v21_range_pilot}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
BASELINE="${SAFE100_BASELINE_CHECKPOINT:-$ROOT/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}"
PROTOCOL_COMMIT="${SAFE100_V21_PROTOCOL_COMMIT:?set SAFE100_V21_PROTOCOL_COMMIT}"
PROTOCOL="${SAFE100_V21_PROTOCOL_FILE:-$REPO/results/online/specialist_v21/protocol_range_pilot_4.json}"

mapfile -t PILOT_METADATA < <(
  "$PYTHON" -c '
import json
import sys

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
pilot = protocol["range_pilot"]
calibration = protocol["calibration"]
print(pilot["pilot_id"])
print(calibration["episodes_per_candidate"])
print(calibration["eval_batch_size"])
print(" ".join(pilot["contexts"]))
' "$PROTOCOL"
)
if [[ "${#PILOT_METADATA[@]}" -ne 4 ]]; then
  printf 'invalid v21 range-pilot metadata\n' >&2
  exit 1
fi
PILOT_ID="${PILOT_METADATA[0]}"
PILOT_EPISODES="${PILOT_METADATA[1]}"
PILOT_BATCH_SIZE="${PILOT_METADATA[2]}"
read -r -a CONTEXT_IDS <<< "${PILOT_METADATA[3]}"
if [[ "${#CONTEXT_IDS[@]}" -eq 0 ]]; then
  printf 'v21 range-pilot has no declared contexts\n' >&2
  exit 1
fi

ARTIFACT_ROOT="${SAFE100_V21_ARTIFACT_ROOT:-$ROOT/artifacts/specialist_v21_range_pilot_${PILOT_ID}}"
LOG_ROOT="${SAFE100_V21_LOG_ROOT:-$ROOT/logs/specialist_v21_range_pilot_${PILOT_ID}}"

cd "$REPO"
test "$(git rev-parse HEAD)" = "$PROTOCOL_COMMIT"
git diff --quiet
git diff --cached --quiet
mkdir -p "$LOG_ROOT"
QUEUE_LOG="$LOG_ROOT/queue_range_pilot.log"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

for context_id in "${CONTEXT_IDS[@]}"; do
  output="$ARTIFACT_ROOT/$context_id"
  compact="$REPO/results/online/specialist_v21/range_pilot/pilot_${PILOT_ID}/$context_id"
  printf '%s context=%s event=range_pilot_started\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$context_id" | tee -a "$QUEUE_LOG"
  "$PYTHON" experiments/scripts/calibrate_deployment_contexts_v21.py \
    --repo "$REPO" \
    --base-policy-checkpoint "$BASELINE" \
    --protocol-file "$PROTOCOL" \
    --protocol-commit "$PROTOCOL_COMMIT" \
    --context-id "$context_id" \
    --output-dir "$output" \
    --context-output "$output/not_a_frozen_context.json" \
    --num-episodes "$PILOT_EPISODES" \
    --eval-batch-size "$PILOT_BATCH_SIZE" \
    --device cuda:0 \
    --exploratory \
    2>&1 | tee -a "$LOG_ROOT/range_pilot_${context_id}.log" \
    | tee -a "$QUEUE_LOG"
  mkdir -p "$compact"
  cp "$output/calibration_progress.json" "$compact/calibration_progress.json"
done

printf '%s phase=range_pilot event=queue_completed\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$QUEUE_LOG"
