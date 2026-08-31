#!/usr/bin/env bash
set -euo pipefail

GUARDIAN_MARKER=/home/carla/LZQW/SAFE100/GuardianFlowPaperResults/formal_5b5b47e
GUARDIAN_WATCHDOG_MARKER=/tmp/watch_guardian_paper_formal_5b5b47e.sh
GUARDIAN_RUN_SUPERVISOR=/home/carla/LZQW/SAFE100/GuardianFlowPaperAuditTools/supervise_guardian_flow_paper_run.sh
GUARDIAN_MASTER_LOG="$GUARDIAN_MARKER/suite_master.log"
GUARDIAN_CHECKPOINT_AUDIT="$GUARDIAN_MARKER/checkpoint_audit.json"
GUARDIAN_FORMAL_WATCHDOG_LOG="$GUARDIAN_MARKER/formal_watchdog.log"
GUARDIAN_COMPLETION_LOG="$GUARDIAN_MARKER/completion_watcher.stdout.log"
CVCI_ROOT=/home/carla/cvci_back
REPO=/home/carla/LZQW/SAFE100/humanoid/worktrees/v23_proximal
PYTHON=/home/carla/LZQW/SAFE100/humanoid/workspace/conda_env/bin/python
BASE_CHECKPOINT=/home/carla/LZQW/SAFE100/humanoid/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt
EXTERNAL_ROOT=/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v25_swing_teacher
ORCHESTRATION_ROOT="$EXTERNAL_ROOT/orchestration"
PRECALIBRATION_PROTOCOL=$REPO/results/online/proximal_v25/precalibration_protocol_revision7.json
EXPECTED_HEAD=1ac36e9c4483b9ca43428f2ced270609cf8c8bc2
EXPECTED_PROTOCOL_SHA=9786e15582989ab37a62561a9272fd5ae6eeafba26fad27bda8f931df7067167
EXPECTED_BASE_SHA=cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8
EXPECTED_GUARDIAN_RELEASE_RECEIPT_SHA=4ada999c2dd0f3b741f1a539d968042246c29093c76333d697c4cd97eb2f578c
MIN_AVAILABLE_KB=10485760
CLEAN_GPU_REQUIRED_SAMPLES=5
CLEAN_GPU_SAMPLE_SECONDS=60
MAX_IDLE_GPU_MEMORY_MIB=2048
GUARDIAN_TERMINAL_QUIET_SAMPLES=5
GUARDIAN_TERMINAL_SAMPLE_SECONDS=60
GUARDIAN_RELEASE_RECEIPT="$EXTERNAL_ROOT/orchestration/guardian_terminal_release_revision7.json"

mkdir -p "$EXTERNAL_ROOT"
mkdir -p "$ORCHESTRATION_ROOT"
exec 7>"$ORCHESTRATION_ROOT/calibration_execution.lock"
if ! flock -n 7; then
  echo "another v25 calibration queue or execution already holds the lock" >&2
  exit 80
fi
exec >>"$EXTERNAL_ROOT/calibration_queue_revision7.log" 2>&1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

guardian_active() {
  local arg command_file has_marker has_suite
  local -a arguments
  for command_file in /proc/[0-9]*/cmdline; do
    [[ -r "$command_file" ]] || continue
    arguments=()
    mapfile -d '' -t arguments <"$command_file" 2>/dev/null || continue
    has_marker=false
    has_suite=false
    for arg in "${arguments[@]}"; do
      [[ "$arg" == "$GUARDIAN_MARKER" ]] && has_marker=true
      [[ "${arg##*/}" == run_guardian_flow_dsac_paper_suite.py ]] && has_suite=true
    done
    if [[ "$has_marker" == true && "$has_suite" == true ]]; then
      return 0
    fi
  done
  return 1
}

guardian_watchdog_active() {
  local arg command_file
  local -a arguments
  for command_file in /proc/[0-9]*/cmdline; do
    [[ -r "$command_file" ]] || continue
    arguments=()
    mapfile -d '' -t arguments <"$command_file" 2>/dev/null || continue
    for arg in "${arguments[@]}"; do
      [[ "$arg" == "$GUARDIAN_WATCHDOG_MARKER" ]] && return 0
    done
  done
  return 1
}

guardian_complete() {
  python3 - "$GUARDIAN_MARKER/suite_summary.json" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
raise SystemExit(0 if payload.get("completed") is True else 1)
PY
}

guardian_release_receipt_valid() {
  local actual_receipt_sha
  [[ -f "$GUARDIAN_RELEASE_RECEIPT" ]] || return 1
  actual_receipt_sha=$(sha256sum "$GUARDIAN_RELEASE_RECEIPT" 2>/dev/null | awk '{print $1}')
  [[ "$actual_receipt_sha" == "$EXPECTED_GUARDIAN_RELEASE_RECEIPT_SHA" ]] || return 1
  python3 - "$GUARDIAN_RELEASE_RECEIPT" "$GUARDIAN_MASTER_LOG" \
    "$GUARDIAN_CHECKPOINT_AUDIT" "$GUARDIAN_FORMAL_WATCHDOG_LOG" \
    "$GUARDIAN_COMPLETION_LOG" <<'PY' >/dev/null 2>&1
import hashlib
import json
import pathlib
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt = pathlib.Path(sys.argv[1])
evidence = [pathlib.Path(value) for value in sys.argv[2:]]
payload = json.loads(receipt.read_text())
if (
    payload.get("schema_version") != 1
    or payload.get("release_reason") != "guardian_terminal_without_completed_summary"
    or payload.get("guardian_completed") is not False
    or payload.get("guardian_suite_complete") is not False
    or payload.get("quiet_sample_count") != 5
    or payload.get("quiet_sample_seconds") != 60
    or int(payload.get("quiet_window_end_epoch", 0))
    - int(payload.get("quiet_window_start_epoch", 0))
    < 240
    or payload.get("keyboard_interrupt_observed") is not True
):
    raise SystemExit(1)
recorded = payload.get("evidence", {})
for path in evidence:
    row = recorded.get(str(path))
    if not path.is_file() or not isinstance(row, dict):
        raise SystemExit(1)
    stat = path.stat()
    if (
        row.get("bytes") != stat.st_size
        or row.get("mtime_epoch") != int(stat.st_mtime)
        or row.get("sha256") != sha256(path)
    ):
        raise SystemExit(1)
PY
}

write_guardian_terminal_release_receipt() {
  python3 - "$GUARDIAN_RELEASE_RECEIPT" "$GUARDIAN_MASTER_LOG" \
    "$GUARDIAN_CHECKPOINT_AUDIT" "$GUARDIAN_FORMAL_WATCHDOG_LOG" \
    "$GUARDIAN_COMPLETION_LOG" "$GUARDIAN_TERMINAL_QUIET_SAMPLES" \
    "$GUARDIAN_TERMINAL_SAMPLE_SECONDS" "$@" <<'PY'
import hashlib
import json
import pathlib
import sys
import tempfile
from datetime import datetime

receipt = pathlib.Path(sys.argv[1])
evidence_paths = [pathlib.Path(value) for value in sys.argv[2:6]]
quiet_samples = int(sys.argv[6])
quiet_seconds = int(sys.argv[7])
observed_states = sys.argv[8:12]
quiet_start_path = receipt.with_name(".guardian_terminal_quiet_start_epoch")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if len(observed_states) != len(evidence_paths):
    raise RuntimeError("missing final Guardian quiet-sample evidence states")
evidence = {}
for path, observed_state in zip(evidence_paths, observed_states):
    if not path.is_file():
        raise RuntimeError(f"Guardian terminal evidence missing: {path}")
    stat = path.stat()
    current_state = f"{stat.st_size}:{int(stat.st_mtime)}:{sha256(path)}"
    if current_state != observed_state:
        raise RuntimeError(f"Guardian terminal evidence changed after final quiet sample: {path}")
    evidence[str(path)] = {
        "bytes": stat.st_size,
        "mtime_epoch": int(stat.st_mtime),
        "sha256": current_state.rsplit(":", 1)[1],
    }
master = evidence_paths[0].read_text(errors="replace")
checkpoint_audit = json.loads(evidence_paths[1].read_text())
if "KeyboardInterrupt" not in master:
    raise RuntimeError("Guardian master log lacks terminal KeyboardInterrupt evidence")
if checkpoint_audit.get("suite_complete") is not False:
    raise RuntimeError("Guardian checkpoint audit does not declare incomplete suite")
payload = {
    "schema_version": 1,
    "release_reason": "guardian_terminal_without_completed_summary",
    "guardian_completed": False,
    "guardian_suite_complete": False,
    "keyboard_interrupt_observed": True,
    "quiet_sample_count": quiet_samples,
    "quiet_sample_seconds": quiet_seconds,
    "quiet_window_start_epoch": int(quiet_start_path.read_text()),
    "quiet_window_end_epoch": int(datetime.now().timestamp()),
    "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "evidence": evidence,
}
rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
if receipt.exists():
    if receipt.read_text() != rendered:
        raise RuntimeError("refusing to overwrite a different Guardian release receipt")
else:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=receipt.parent, delete=False) as handle:
        handle.write(rendered)
        temporary = pathlib.Path(handle.name)
    temporary.replace(receipt)
print(json.dumps(payload, sort_keys=True))
PY
}

guardian_release_authorized() {
  guardian_complete || guardian_release_receipt_valid
}

establish_guardian_release() {
  local quiet_samples=0
  local -A evidence_state=()
  local digest file state
  local quiet_start_path="$ORCHESTRATION_ROOT/.guardian_terminal_quiet_start_epoch"
  while ! guardian_release_authorized; do
    if guardian_active || guardian_watchdog_active; then
      quiet_samples=0
      evidence_state=()
      : >"$quiet_start_path"
      echo "$(timestamp) waiting; Guardian suite or restart watchdog is active"
      sleep "$GUARDIAN_TERMINAL_SAMPLE_SECONDS"
      continue
    fi
    local evidence_ok=true
    for file in "$GUARDIAN_MASTER_LOG" "$GUARDIAN_CHECKPOINT_AUDIT" \
      "$GUARDIAN_FORMAL_WATCHDOG_LOG" "$GUARDIAN_COMPLETION_LOG"; do
      if [[ ! -f "$file" ]]; then
        evidence_ok=false
        break
      fi
      state=$(stat -c '%s:%Y' "$file" 2>/dev/null || true)
      if [[ -z "$state" ]]; then
        evidence_ok=false
        break
      fi
      digest=$(sha256sum "$file" 2>/dev/null | awk '{print $1}')
      if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
        evidence_ok=false
        break
      fi
      state="$state:$digest"
      if [[ -n "${evidence_state[$file]:-}" && "${evidence_state[$file]}" != "$state" ]]; then
        evidence_ok=false
        break
      fi
      evidence_state[$file]=$state
    done
    if [[ "$evidence_ok" != true ]] || \
       ! grep -q 'KeyboardInterrupt' "$GUARDIAN_MASTER_LOG" 2>/dev/null || \
       ! python3 - "$GUARDIAN_CHECKPOINT_AUDIT" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys
raise SystemExit(0 if json.loads(pathlib.Path(sys.argv[1]).read_text()).get("suite_complete") is False else 1)
PY
    then
      quiet_samples=0
      evidence_state=()
      : >"$quiet_start_path"
      echo "$(timestamp) waiting; Guardian is inactive but terminal evidence is incomplete or changing"
      sleep "$GUARDIAN_TERMINAL_SAMPLE_SECONDS"
      continue
    fi
    quiet_samples=$((quiet_samples + 1))
    if (( quiet_samples == 1 )); then
      date +%s >"$quiet_start_path"
    fi
    echo "$(timestamp) Guardian terminal quiet sample $quiet_samples/$GUARDIAN_TERMINAL_QUIET_SAMPLES (completed=false)"
    if (( quiet_samples >= GUARDIAN_TERMINAL_QUIET_SAMPLES )); then
      if guardian_active || guardian_watchdog_active || guardian_complete; then
        quiet_samples=0
        evidence_state=()
        : >"$quiet_start_path"
        continue
      fi
      write_guardian_terminal_release_receipt \
        "${evidence_state[$GUARDIAN_MASTER_LOG]}" \
        "${evidence_state[$GUARDIAN_CHECKPOINT_AUDIT]}" \
        "${evidence_state[$GUARDIAN_FORMAL_WATCHDOG_LOG]}" \
        "${evidence_state[$GUARDIAN_COMPLETION_LOG]}"
      guardian_release_receipt_valid || {
        echo "$(timestamp) abort: Guardian terminal release receipt failed immediate validation"
        exit 37
      }
      break
    fi
    sleep "$GUARDIAN_TERMINAL_SAMPLE_SECONDS"
  done
  if ! guardian_complete; then
    echo "$(timestamp) validated existing Guardian terminal release receipt"
  fi
}

foreign_compute_pids() {
  local compute_pid command_line
  while IFS= read -r compute_pid; do
    compute_pid=${compute_pid//[[:space:]]/}
    [[ -n "$compute_pid" && -r "/proc/$compute_pid/cmdline" ]] || continue
    command_line=$(tr '\0' ' ' <"/proc/$compute_pid/cmdline" 2>/dev/null) || continue
    if [[ "$command_line" == *"gnome-remote-desktop-daemon"* ]]; then
      continue
    fi
    printf '%s\n' "$compute_pid"
  done < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null
  )
}

foreign_gpu_scheduler_pids() {
  local arg command_file command_line has_guardian_marker has_guardian_supervisor
  local process_cwd scheduler_pid
  local -a arguments
  for command_file in /proc/[0-9]*/cmdline; do
    [[ -r "$command_file" ]] || continue
    command_line=$(tr '\0' ' ' <"$command_file" 2>/dev/null) || continue
    arguments=()
    mapfile -d '' -t arguments <"$command_file" 2>/dev/null || continue
    scheduler_pid=${command_file#/proc/}
    scheduler_pid=${scheduler_pid%/cmdline}
    process_cwd=$(readlink -f "/proc/$scheduler_pid/cwd" 2>/dev/null || true)
    has_guardian_marker=false
    has_guardian_supervisor=false
    for arg in "${arguments[@]}"; do
      [[ "$arg" == "$GUARDIAN_MARKER" ]] && has_guardian_marker=true
      [[ "$arg" == "$GUARDIAN_RUN_SUPERVISOR" ]] && has_guardian_supervisor=true
    done
    if [[ "$has_guardian_marker" == true && "$has_guardian_supervisor" == true ]]; then
      printf '%s\n' "$scheduler_pid"
      continue
    fi
    case "$process_cwd:$command_line" in
      "$CVCI_ROOT/"*:*"run_full_220.sh"*|\
      "$CVCI_ROOT/"*:*"finalize_after_run.sh"*|\
      "$CVCI_ROOT/"*:*"recover_0812_unresolved.sh"*)
        printf '%s\n' "$scheduler_pid"
        ;;
    esac
  done
}

gpu_memory_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | \
    awk 'NR == 1 {gsub(/[[:space:]]/, "", $0); print; exit}'
}

require_gpu_clear_now() {
  local memory_used_mib
  local -a active_compute_pids=()
  local -a active_scheduler_pids=()
  if ! guardian_release_authorized || guardian_active || guardian_watchdog_active; then
    echo "$(timestamp) abort: Guardian release changed during the final prelaunch check"
    return 1
  fi
  mapfile -t active_compute_pids < <(foreign_compute_pids)
  mapfile -t active_scheduler_pids < <(foreign_gpu_scheduler_pids)
  memory_used_mib=$(gpu_memory_used_mib)
  if (( ${#active_scheduler_pids[@]} > 0 )); then
    echo "$(timestamp) abort: foreign GPU scheduler appeared during final prelaunch check PIDs=${active_scheduler_pids[*]}"
    return 1
  fi
  if (( ${#active_compute_pids[@]} > 0 )); then
    echo "$(timestamp) abort: foreign GPU compute appeared during final prelaunch check PIDs=${active_compute_pids[*]}"
    return 1
  fi
  if [[ ! "$memory_used_mib" =~ ^[0-9]+$ ]] || \
     (( memory_used_mib > MAX_IDLE_GPU_MEMORY_MIB )); then
    echo "$(timestamp) abort: GPU memory changed during final prelaunch check used_mib=${memory_used_mib:-unknown} maximum_idle_mib=$MAX_IDLE_GPU_MEMORY_MIB"
    return 1
  fi
  echo "$(timestamp) final prelaunch GPU check passed memory_used_mib=$memory_used_mib"
}

require_storage_headroom() {
  local available_kb
  available_kb=$(df --output=avail "$EXTERNAL_ROOT" | tail -n 1 | tr -d ' ')
  if [[ ! "$available_kb" =~ ^[0-9]+$ ]] || (( available_kb < MIN_AVAILABLE_KB )); then
    echo "$(timestamp) abort: insufficient artifact storage available_kb=${available_kb:-unknown} required_kb=$MIN_AVAILABLE_KB"
    exit 36
  fi
  echo "$(timestamp) storage preflight available_kb=$available_kb required_kb=$MIN_AVAILABLE_KB"
}

echo "$(timestamp) v25 calibration queued behind Guardian suite $GUARDIAN_MARKER"
establish_guardian_release

echo "$(timestamp) Guardian release authorized; waiting for $CLEAN_GPU_REQUIRED_SAMPLES clean GPU samples"
quiet_samples=0
while (( quiet_samples < CLEAN_GPU_REQUIRED_SAMPLES )); do
  if ! guardian_release_authorized || guardian_active || guardian_watchdog_active; then
    quiet_samples=0
    echo "$(timestamp) waiting; Guardian release is invalid, active, or restart-watchdog-managed"
    sleep 60
    continue
  fi
  mapfile -t active_compute_pids < <(foreign_compute_pids)
  mapfile -t active_scheduler_pids < <(foreign_gpu_scheduler_pids)
  memory_used_mib=$(gpu_memory_used_mib)
  if (( ${#active_scheduler_pids[@]} > 0 )); then
    quiet_samples=0
    echo "$(timestamp) waiting; foreign GPU scheduler PIDs=${active_scheduler_pids[*]}"
  elif (( ${#active_compute_pids[@]} > 0 )); then
    quiet_samples=0
    echo "$(timestamp) waiting; non-desktop GPU compute PIDs=${active_compute_pids[*]}"
  elif [[ ! "$memory_used_mib" =~ ^[0-9]+$ ]] || \
       (( memory_used_mib > MAX_IDLE_GPU_MEMORY_MIB )); then
    quiet_samples=0
    echo "$(timestamp) waiting; GPU memory used MiB=${memory_used_mib:-unknown} maximum idle MiB=$MAX_IDLE_GPU_MEMORY_MIB"
  else
    quiet_samples=$((quiet_samples + 1))
    echo "$(timestamp) GPU clean sample $quiet_samples/$CLEAN_GPU_REQUIRED_SAMPLES memory_used_mib=$memory_used_mib"
  fi
  if (( quiet_samples < CLEAN_GPU_REQUIRED_SAMPLES )); then
    sleep "$CLEAN_GPU_SAMPLE_SECONDS"
  fi
done

require_storage_headroom
cd "$REPO"
actual_head=$(git rev-parse HEAD)
if [[ "$actual_head" != "$EXPECTED_HEAD" ]]; then
  echo "$(timestamp) abort: expected HEAD $EXPECTED_HEAD but found $actual_head"
  exit 32
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "$(timestamp) abort: v25 worktree is not clean"
  git status --short
  exit 33
fi
actual_base_sha=$(sha256sum "$BASE_CHECKPOINT" | awk '{print $1}')
if [[ "$actual_base_sha" != "$EXPECTED_BASE_SHA" ]]; then
  echo "$(timestamp) abort: frozen base-checkpoint SHA-256 mismatch"
  exit 34
fi
actual_protocol_sha=$(sha256sum "$PRECALIBRATION_PROTOCOL" | awk '{print $1}')
if [[ "$actual_protocol_sha" != "$EXPECTED_PROTOCOL_SHA" ]]; then
  echo "$(timestamp) abort: revision-7 precalibration protocol SHA-256 mismatch"
  exit 35
fi
require_gpu_clear_now || exit 38

echo "$(timestamp) starting prospectively frozen v25 revision-7 paired calibration"
set +e
export REPO PYTHON BASE_CHECKPOINT EXTERNAL_ROOT PRECALIBRATION_PROTOCOL
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=0 \
"$REPO/experiments/scripts/run_cbf_teacher_v25.sh" calibrate
return_code=$?
set -e
echo "$(timestamp) v25 paired calibration exited return_code=$return_code"
exit "$return_code"
