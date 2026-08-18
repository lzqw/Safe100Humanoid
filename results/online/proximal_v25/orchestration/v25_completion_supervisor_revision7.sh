#!/usr/bin/env bash
set -Eeuo pipefail

EXTERNAL_ROOT=/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v25_swing_teacher
ORCHESTRATION_ROOT="$EXTERNAL_ROOT/orchestration"
COMPLETION_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_complete_after_calibration_revision7.sh
TRAINING_ROOT="$EXTERNAL_ROOT/training"
CALIBRATION_ROOT="$EXTERNAL_ROOT/calibration"
CALIBRATION_LOCK="$ORCHESTRATION_ROOT/calibration_execution.lock"
QUEUE_PROCESS_PATTERN='^bash /home/carla/LZQW/SAFE100/humanoid/artifacts/v25_wait_and_calibrate_revision7\.sh$'
STATE_FILE="$ORCHESTRATION_ROOT/pipeline_state_revision7.json"
SUPERVISOR_STATE="$ORCHESTRATION_ROOT/supervisor_state_revision7.json"
SUPERVISOR_LOG="$ORCHESTRATION_ROOT/completion_supervisor_revision7.log"
MAX_RESTARTS=12
RETRY_SECONDS=300
EXPECTED_INITIAL_HEAD=1ac36e9c4483b9ca43428f2ced270609cf8c8bc2
EXPECTED_PROTOCOL_SHA=9786e15582989ab37a62561a9272fd5ae6eeafba26fad27bda8f931df7067167

mkdir -p "$ORCHESTRATION_ROOT"
exec 8>"$ORCHESTRATION_ROOT/completion_supervisor.lock"
if ! flock -n 8; then
  echo "another v25 completion supervisor already holds the lock" >&2
  exit 80
fi
exec >>"$SUPERVISOR_LOG" 2>&1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

pipeline_field() {
  local field=$1 default_value=$2
  python3 - "$STATE_FILE" "$field" "$default_value" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except (FileNotFoundError, json.JSONDecodeError, OSError):
    print(sys.argv[3])
else:
    print(payload.get(sys.argv[2], sys.argv[3]))
PY
}

write_supervisor_state() {
  local status=$1 restarts=$2 child_exit=$3
  python3 - "$SUPERVISOR_STATE" "$status" "$restarts" "$child_exit" "$$" \
    "$EXPECTED_INITIAL_HEAD" "$EXPECTED_PROTOCOL_SHA" <<'PY'
import json
import pathlib
import sys
import tempfile
from datetime import datetime

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "precalibration_revision": 7,
    "frozen_initial_git_commit": sys.argv[6],
    "precalibration_protocol_sha256": sys.argv[7],
    "status": sys.argv[2],
    "restart_count": int(sys.argv[3]),
    "last_child_exit_code": int(sys.argv[4]),
    "pid": int(sys.argv[5]),
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = pathlib.Path(handle.name)
temporary.replace(path)
PY
}

formal_adaptation_incomplete() {
  [[ -f "$TRAINING_ROOT/formal_execution_started.json" && \
     ! -f "$TRAINING_ROOT/formal_execution_completed.json" ]]
}

calibration_incomplete() {
  [[ -f "$CALIBRATION_ROOT/calibration_execution_started.json" && \
     ! -f "$CALIBRATION_ROOT/calibration_summary.json" ]]
}

calibration_queue_active() {
  if ! flock -n "$CALIBRATION_LOCK" -c true >/dev/null 2>&1; then
    return 0
  fi
  pgrep -f "$QUEUE_PROCESS_PATTERN" >/dev/null 2>&1
}

restart_count=0
write_supervisor_state running "$restart_count" 0
echo "$(timestamp) v25 completion supervisor started"

while true; do
  status=$(pipeline_field status missing)
  if [[ "$status" == complete || "$status" == complete_no_candidate || \
        "$status" == terminal_verification_failed ]]; then
    write_supervisor_state "$status" "$restart_count" 0
    echo "$(timestamp) pipeline already complete with status=$status"
    [[ "$status" == terminal_verification_failed ]] && exit 92
    exit 0
  fi
  if formal_adaptation_incomplete; then
    write_supervisor_state manual_audit_required "$restart_count" 90
    echo "$(timestamp) formal adaptation marker exists without completion; refusing automatic seed reuse"
    exit 90
  fi
  if calibration_incomplete && ! calibration_queue_active; then
    write_supervisor_state manual_audit_required "$restart_count" 93
    echo "$(timestamp) calibration marker exists without terminal summary and no queue owner is alive; refusing seed reuse"
    exit 93
  fi

  set +e
  "$COMPLETION_SCRIPT"
  child_exit=$?
  set -e
  status=$(pipeline_field status missing)
  phase=$(pipeline_field phase unknown)
  echo "$(timestamp) completion child exited code=$child_exit status=$status phase=$phase"

  if [[ "$status" == complete || "$status" == complete_no_candidate || \
        "$status" == terminal_verification_failed ]]; then
    write_supervisor_state "$status" "$restart_count" "$child_exit"
    [[ "$status" == terminal_verification_failed ]] && exit 92
    exit 0
  fi
  if formal_adaptation_incomplete; then
    write_supervisor_state manual_audit_required "$restart_count" "$child_exit"
    echo "$(timestamp) incomplete formal adaptation detected after child exit; stopping supervisor"
    exit 90
  fi
  if calibration_incomplete && ! calibration_queue_active; then
    write_supervisor_state manual_audit_required "$restart_count" "$child_exit"
    echo "$(timestamp) incomplete calibration detected after queue/child exit; stopping supervisor without seed reuse"
    exit 93
  fi

  restart_count=$((restart_count + 1))
  if (( restart_count > MAX_RESTARTS )); then
    write_supervisor_state retry_limit_reached "$restart_count" "$child_exit"
    echo "$(timestamp) retry limit reached; preserving failure for audit"
    exit 91
  fi
  write_supervisor_state retry_wait "$restart_count" "$child_exit"
  echo "$(timestamp) retrying idempotent completion in $RETRY_SECONDS seconds ($restart_count/$MAX_RESTARTS)"
  sleep "$RETRY_SECONDS"
done
