#!/usr/bin/env bash
set -Eeuo pipefail

EXTERNAL_ROOT=/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v25_swing_teacher
ORCHESTRATION_ROOT="$EXTERNAL_ROOT/orchestration"
SUPERVISOR_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_completion_supervisor_revision7.sh
SUPERVISOR_SESSION=safe100_v25_completion_supervisor_revision7
SUPERVISOR_STATE="$ORCHESTRATION_ROOT/supervisor_state_revision7.json"
COMPLETION_LOCK="$ORCHESTRATION_ROOT/completion_orchestrator.lock"
COMPLETION_PROCESS_PATTERN='^bash /home/carla/LZQW/SAFE100/humanoid/artifacts/v25_complete_after_calibration_revision7\.sh$'
TRAINING_ROOT="$EXTERNAL_ROOT/training"
CALIBRATION_ROOT="$EXTERNAL_ROOT/calibration"
CALIBRATION_LOCK="$ORCHESTRATION_ROOT/calibration_execution.lock"
QUEUE_PROCESS_PATTERN='^bash /home/carla/LZQW/SAFE100/humanoid/artifacts/v25_wait_and_calibrate_revision7\.sh$'
WATCHDOG_STATE="$ORCHESTRATION_ROOT/supervisor_watchdog_state_revision7.json"
WATCHDOG_LOG="$ORCHESTRATION_ROOT/supervisor_watchdog_revision7.log"
POLL_SECONDS=60
MAX_RESTARTS=12
EXPECTED_INITIAL_HEAD=1ac36e9c4483b9ca43428f2ced270609cf8c8bc2
EXPECTED_PROTOCOL_SHA=9786e15582989ab37a62561a9272fd5ae6eeafba26fad27bda8f931df7067167

mkdir -p "$ORCHESTRATION_ROOT"
exec 6>"$ORCHESTRATION_ROOT/supervisor_watchdog.lock"
if ! flock -n 6; then
  echo "another v25 supervisor watchdog already holds the lock" >&2
  exit 80
fi
exec >>"$WATCHDOG_LOG" 2>&1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

state_field() {
  local field=$1 default_value=$2
  python3 - "$SUPERVISOR_STATE" "$field" "$default_value" <<'PY'
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

write_state() {
  local status=$1 restart_count=$2
  python3 - "$WATCHDOG_STATE" "$status" "$restart_count" "$$" \
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
    "frozen_initial_git_commit": sys.argv[5],
    "precalibration_protocol_sha256": sys.argv[6],
    "status": sys.argv[2],
    "restart_count": int(sys.argv[3]),
    "pid": int(sys.argv[4]),
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

completion_active() {
  if ! flock -n "$COMPLETION_LOCK" -c true >/dev/null 2>&1; then
    return 0
  fi
  pgrep -f "$COMPLETION_PROCESS_PATTERN" >/dev/null 2>&1
}

restart_count=0
write_state running "$restart_count"
echo "$(timestamp) v25 supervisor watchdog started"

while true; do
  supervisor_status=$(state_field status missing)
  case "$supervisor_status" in
    complete|complete_no_candidate|terminal_verification_failed|manual_audit_required|retry_limit_reached)
      write_state "terminal:$supervisor_status" "$restart_count"
      echo "$(timestamp) terminal supervisor status=$supervisor_status; watchdog exiting"
      exit 0
      ;;
  esac

  if tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null; then
    write_state running "$restart_count"
  elif completion_active; then
    write_state orphan_completion_active "$restart_count"
    echo "$(timestamp) supervisor session absent but completion still owns its lock; waiting without duplication"
  elif calibration_incomplete && ! calibration_queue_active; then
    write_state manual_audit_required "$restart_count"
    echo "$(timestamp) calibration is incomplete and no queue owner is alive; refusing supervisor restart and seed reuse"
    exit 93
  elif formal_adaptation_incomplete; then
    # The marker is expected for the whole duration of a healthy eight-round
    # adaptation.  It becomes a manual-audit condition only after both the
    # supervisor session and its completion child are gone.  Checking it before
    # those liveness signals would make the watchdog exit as soon as training
    # starts and silently remove protection from the later audit/publication
    # phases.
    write_state manual_audit_required "$restart_count"
    echo "$(timestamp) formal adaptation is incomplete and no completion owner is alive; refusing supervisor restart"
    exit 90
  else
    restart_count=$((restart_count + 1))
    if (( restart_count > MAX_RESTARTS )); then
      write_state restart_limit_reached "$restart_count"
      echo "$(timestamp) supervisor restart limit reached"
      exit 91
    fi
    [[ -x "$SUPERVISOR_SCRIPT" ]] || {
      write_state supervisor_script_missing "$restart_count"
      echo "$(timestamp) supervisor script missing or not executable"
      exit 92
    }
    bash -n "$SUPERVISOR_SCRIPT"
    tmux new-session -d -s "$SUPERVISOR_SESSION" "$SUPERVISOR_SCRIPT"
    sleep 3
    if tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null; then
      write_state restarted "$restart_count"
      echo "$(timestamp) restarted and verified supervisor session ($restart_count/$MAX_RESTARTS)"
    else
      write_state restart_did_not_stay_alive "$restart_count"
      echo "$(timestamp) supervisor restart did not stay alive; will retry after lock-release interval"
    fi
  fi
  sleep "$POLL_SECONDS"
done
