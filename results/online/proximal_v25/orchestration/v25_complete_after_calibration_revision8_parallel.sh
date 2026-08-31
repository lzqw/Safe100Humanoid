#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/carla/LZQW/SAFE100/humanoid/worktrees/v23_proximal
PYTHON=/home/carla/LZQW/SAFE100/humanoid/workspace/conda_env/bin/python
BASE_CHECKPOINT=/home/carla/LZQW/SAFE100/humanoid/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt
EXTERNAL_ROOT=/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v25_swing_teacher
RESULT_ROOT="$REPO/results/online/proximal_v25"
PRECALIBRATION_PROTOCOL="$RESULT_ROOT/precalibration_protocol_revision7.json"
CALIBRATION_ROOT="$EXTERNAL_ROOT/calibration"
TRAINING_ROOT="$EXTERNAL_ROOT/training"
FINAL_ROOT="$EXTERNAL_ROOT/final"
ORCHESTRATION_ROOT="$EXTERNAL_ROOT/orchestration"
RUNNER="$REPO/experiments/scripts/run_cbf_teacher_v25.sh"
QUEUE_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_wait_and_calibrate_revision7_guard2.sh
COMPLETION_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_complete_after_calibration_revision7_guard2.sh
SUPERVISOR_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_completion_supervisor_revision7_guard2.sh
WATCHDOG_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_supervisor_watchdog_revision7_guard2.sh
LEGACY_QUEUE_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_wait_and_calibrate_revision7.sh
LEGACY_COMPLETION_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_complete_after_calibration_revision7.sh
LEGACY_SUPERVISOR_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_completion_supervisor_revision7.sh
LEGACY_WATCHDOG_SCRIPT=/home/carla/LZQW/SAFE100/humanoid/artifacts/v25_supervisor_watchdog_revision7.sh
EXPECTED_INITIAL_HEAD=1ac36e9c4483b9ca43428f2ced270609cf8c8bc2
EXPECTED_PROTOCOL_SHA=9786e15582989ab37a62561a9272fd5ae6eeafba26fad27bda8f931df7067167
EXPECTED_BASE_SHA=cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8
EXPECTED_GUARDIAN_RELEASE_RECEIPT_SHA=4ada999c2dd0f3b741f1a539d968042246c29093c76333d697c4cd97eb2f578c
REMOTE=origin
REMOTE_BRANCH=feature/online-safe-refinement
PR_NUMBER=1
PR_COMMENT_ID=5262063724
GUARDIAN_MARKER=/home/carla/LZQW/SAFE100/GuardianFlowPaperResults/formal_5b5b47e
GUARDIAN_WATCHDOG_MARKER=/tmp/watch_guardian_paper_formal_5b5b47e.sh
GUARDIAN_RUN_SUPERVISOR=/home/carla/LZQW/SAFE100/GuardianFlowPaperAuditTools/supervise_guardian_flow_paper_run.sh
GUARDIAN_RELEASE_RECEIPT="$ORCHESTRATION_ROOT/guardian_terminal_release_revision7.json"
GUARD2_MIGRATION_RECEIPT="$ORCHESTRATION_ROOT/v25_revision7_guard2_migration_receipt.json"
CVCI_ROOT=/home/carla/cvci_back
GUARDIANFLOW_AUTOMATION_ROOT=/home/carla/LZQW/SAFE100/GuardianFlowPaperAutomation_99383a4
GUARDIANFLOW_REPOSITORY=/home/carla/LZQW/SAFE100/GuardianFlowPaperFormalRepo_99383a4
GUARDIANFLOW_PUBLISH_REPOSITORY=/home/carla/LZQW/SAFE100/GuardianFlowPaperPublishRepo
MIN_AVAILABLE_KB=10485760
CLEAN_GPU_REQUIRED_SAMPLES=5
CLEAN_GPU_SAMPLE_SECONDS=60
MAX_IDLE_GPU_MEMORY_MIB=2048
PARALLEL_MIN_FREE_GPU_MEMORY_MIB=3500
PARALLEL_RESOURCE_RECEIPT="$ORCHESTRATION_ROOT/v25_revision8_parallel_resource_receipt.json"
STATE_FILE="$ORCHESTRATION_ROOT/pipeline_state_revision7_guard2.json"

mkdir -p "$ORCHESTRATION_ROOT"
exec 9>"$ORCHESTRATION_ROOT/completion_orchestrator.lock"
if ! flock -n 9; then
  echo "another v25 completion orchestrator already holds the lock" >&2
  exit 80
fi
exec >>"$ORCHESTRATION_ROOT/completion_orchestrator_revision8_parallel.log" 2>&1

PIPELINE_STATUS=running
CURRENT_PHASE=waiting_for_calibration

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

write_state() {
  local exit_code=${1:-0}
  python3 - "$STATE_FILE" "$PIPELINE_STATUS" "$CURRENT_PHASE" "$exit_code" "$$" \
    "$EXPECTED_INITIAL_HEAD" "$EXPECTED_PROTOCOL_SHA" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
from datetime import datetime

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "precalibration_revision": 7,
    "orchestration_revision": "revision7_guard2",
    "frozen_initial_git_commit": sys.argv[6],
    "precalibration_protocol_sha256": sys.argv[7],
    "status": sys.argv[2],
    "phase": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "pid": int(sys.argv[5]),
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = pathlib.Path(handle.name)
temporary.replace(path)
PY
}

on_exit() {
  local exit_code=$?
  if (( exit_code != 0 )) && [[ "$PIPELINE_STATUS" == running ]]; then
    PIPELINE_STATUS=failed
  fi
  write_state "$exit_code" || true
  echo "$(timestamp) orchestrator exit status=$PIPELINE_STATUS phase=$CURRENT_PHASE code=$exit_code"
}
trap on_exit EXIT

fail() {
  echo "$(timestamp) ERROR: $*"
  PIPELINE_STATUS=failed
  exit 1
}

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  [[ -f "$1" ]] || fail "required file missing: $1"
}

require_storage_headroom() {
  local available_kb
  available_kb=$(df --output=avail "$EXTERNAL_ROOT" | tail -n 1 | tr -d ' ')
  [[ "$available_kb" =~ ^[0-9]+$ ]] || fail "could not determine artifact storage headroom"
  (( available_kb >= MIN_AVAILABLE_KB )) || fail \
    "insufficient artifact storage available_kb=$available_kb required_kb=$MIN_AVAILABLE_KB"
  echo "$(timestamp) storage preflight available_kb=$available_kb required_kb=$MIN_AVAILABLE_KB"
}

require_clean_repo() {
  local dirty
  dirty=$(git -C "$REPO" status --porcelain)
  [[ -z "$dirty" ]] || fail "repository is not clean before phase $CURRENT_PHASE: $dirty"
}

require_no_unrelated_repo_changes() {
  local unexpected
  unexpected=$(git -C "$REPO" status --porcelain --untracked-files=all | \
    awk '{path=$0; sub(/^.. /, "", path); if (path !~ /^results\/online\/proximal_v25\//) print $0}')
  [[ -z "$unexpected" ]] || fail \
    "repository has unrelated changes before phase $CURRENT_PHASE: $unexpected"
}

require_remote_tip_matches_head() {
  git -C "$REPO" fetch --quiet "$REMOTE" \
    "+refs/heads/$REMOTE_BRANCH:refs/remotes/$REMOTE/$REMOTE_BRANCH"
  local local_head remote_head
  local_head=$(git -C "$REPO" rev-parse HEAD)
  remote_head=$(git -C "$REPO" rev-parse "refs/remotes/$REMOTE/$REMOTE_BRANCH")
  [[ "$local_head" == "$remote_head" ]] || fail \
    "remote branch advanced or diverged (local=$local_head remote=$remote_head); refusing to overwrite"
}

push_current_head_if_ahead() {
  git -C "$REPO" fetch --quiet "$REMOTE" \
    "+refs/heads/$REMOTE_BRANCH:refs/remotes/$REMOTE/$REMOTE_BRANCH"
  local local_head remote_head
  local_head=$(git -C "$REPO" rev-parse HEAD)
  remote_head=$(git -C "$REPO" rev-parse "refs/remotes/$REMOTE/$REMOTE_BRANCH")
  if [[ "$local_head" == "$remote_head" ]]; then
    return
  fi
  if ! git -C "$REPO" merge-base --is-ancestor "$remote_head" "$local_head"; then
    fail "local and GitHub histories diverged; refusing to force-push"
  fi
  git -C "$REPO" push "$REMOTE" "HEAD:refs/heads/$REMOTE_BRANCH"
  echo "$(timestamp) recovered pending push of $local_head"
}

commit_and_push() {
  local message=$1
  shift
  git -C "$REPO" add -- "$@"
  local unexpected
  unexpected=$(git -C "$REPO" diff --cached --name-only | \
    awk '$0 !~ /^results\/online\/proximal_v25\// {print}')
  [[ -z "$unexpected" ]] || fail "refusing to commit paths outside v25 result package: $unexpected"
  if git -C "$REPO" diff --cached --quiet; then
    echo "$(timestamp) no new files to commit for: $message"
    push_current_head_if_ahead
    return
  fi
  require_remote_tip_matches_head
  git -C "$REPO" commit -m "$message"
  git -C "$REPO" push "$REMOTE" "HEAD:refs/heads/$REMOTE_BRANCH"
  local pushed_head remote_after_push
  pushed_head=$(git -C "$REPO" rev-parse HEAD)
  remote_after_push=$(git -C "$REPO" ls-remote "$REMOTE" \
    "refs/heads/$REMOTE_BRANCH" | awk 'NR == 1 {print $1}')
  [[ "$remote_after_push" == "$pushed_head" ]] || fail \
    "GitHub branch verification failed after push (local=$pushed_head remote=$remote_after_push)"
  echo "$(timestamp) pushed and verified $pushed_head: $message"
}

copy_exact() {
  local source=$1 destination=$2
  require_file "$source"
  if [[ -e "$destination" ]]; then
    [[ -f "$destination" ]] || fail "copy destination is not a regular file: $destination"
    [[ "$(sha256 "$source")" == "$(sha256 "$destination")" ]] || \
      fail "refusing to overwrite non-identical evidence: $destination"
    return
  fi
  mkdir -p "$(dirname "$destination")"
  install -m 0644 "$source" "$destination"
  [[ "$(sha256 "$source")" == "$(sha256 "$destination")" ]] || \
    fail "copy verification failed: $source -> $destination"
}

queue_active() {
  if ! flock -n "$ORCHESTRATION_ROOT/calibration_execution.lock" -c true \
    >/dev/null 2>&1; then
    return 0
  fi
  pgrep -f '^bash /home/carla/LZQW/SAFE100/humanoid/artifacts/v25_wait_and_calibrate_revision7_guard2\.sh$' \
    >/dev/null 2>&1
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
  python3 - "$GUARDIAN_RELEASE_RECEIPT" \
    "$GUARDIAN_MARKER/suite_master.log" \
    "$GUARDIAN_MARKER/checkpoint_audit.json" \
    "$GUARDIAN_MARKER/formal_watchdog.log" \
    "$GUARDIAN_MARKER/completion_watcher.stdout.log" <<'PY' >/dev/null 2>&1
import hashlib
import json
import pathlib
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
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
for value in sys.argv[2:]:
    path = pathlib.Path(value)
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

guardian_release_authorized() {
  guardian_complete || guardian_release_receipt_valid
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
  local has_guardianflow_controller
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
    has_guardianflow_controller=false
    for arg in "${arguments[@]}"; do
      [[ "$arg" == "$GUARDIAN_MARKER" ]] && has_guardian_marker=true
      [[ "$arg" == "$GUARDIAN_RUN_SUPERVISOR" ]] && has_guardian_supervisor=true
      case "$arg" in
        "$GUARDIANFLOW_AUTOMATION_ROOT/"*|\
        "$GUARDIANFLOW_REPOSITORY/scripts/supervise_"*|\
        "$GUARDIANFLOW_REPOSITORY/scripts/run_"*|\
        "$GUARDIANFLOW_REPOSITORY/scripts/train_"*|\
        "$GUARDIANFLOW_REPOSITORY/scripts/evaluate_"*|\
        "$GUARDIANFLOW_PUBLISH_REPOSITORY/scripts/supervise_guardian_flow_paper_completion.sh"|\
        "$GUARDIANFLOW_PUBLISH_REPOSITORY/scripts/watch_guardian_flow_paper_completion.py")
          has_guardianflow_controller=true
          ;;
      esac
    done
    if [[ "$has_guardian_marker" == true && "$has_guardian_supervisor" == true ]]; then
      printf '%s\n' "$scheduler_pid"
      continue
    fi
    if [[ "$has_guardianflow_controller" == true ]]; then
      printf '%s\n' "$scheduler_pid"
      continue
    fi
    case "$process_cwd:$command_line" in
      "$GUARDIANFLOW_REPOSITORY:"*:*"scripts/run_guardian_flow_dsac_paper_suite.py"*|\
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

gpu_memory_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | \
    awk 'NR == 1 {gsub(/[[:space:]]/, "", $0); print; exit}'
}

require_gpu_clear_now() {
  local memory_free_mib
  memory_free_mib=$(gpu_memory_free_mib)
  [[ "$memory_free_mib" =~ ^[0-9]+$ ]] || fail \
    "could not read free GPU memory during parallel prelaunch check for $CURRENT_PHASE"
  (( memory_free_mib >= PARALLEL_MIN_FREE_GPU_MEMORY_MIB )) || fail \
    "parallel GPU headroom changed before $CURRENT_PHASE: free_mib=$memory_free_mib required_free_mib=$PARALLEL_MIN_FREE_GPU_MEMORY_MIB"
  echo "$(timestamp) parallel prelaunch GPU check passed for $CURRENT_PHASE free_mib=$memory_free_mib required_free_mib=$PARALLEL_MIN_FREE_GPU_MEMORY_MIB"
}

wait_for_clean_gpu() {
  local memory_free_mib
  while true; do
    memory_free_mib=$(gpu_memory_free_mib)
    if [[ "$memory_free_mib" =~ ^[0-9]+$ ]] && \
       (( memory_free_mib >= PARALLEL_MIN_FREE_GPU_MEMORY_MIB )); then
      echo "$(timestamp) parallel GPU headroom available before $CURRENT_PHASE free_mib=$memory_free_mib required_free_mib=$PARALLEL_MIN_FREE_GPU_MEMORY_MIB"
      return 0
    fi
    echo "$(timestamp) waiting for parallel GPU headroom before $CURRENT_PHASE free_mib=${memory_free_mib:-unknown} required_free_mib=$PARALLEL_MIN_FREE_GPU_MEMORY_MIB"
    sleep 60
  done
}

write_parallel_resource_receipt() {
  local memory_used_mib memory_free_mib original_sha parallel_sha
  memory_used_mib=$(gpu_memory_used_mib)
  memory_free_mib=$(gpu_memory_free_mib)
  original_sha=$(sha256 "$COMPLETION_SCRIPT")
  parallel_sha=$(sha256 "$0")
  python3 - "$PARALLEL_RESOURCE_RECEIPT" "$memory_used_mib" "$memory_free_mib" \
    "$PARALLEL_MIN_FREE_GPU_MEMORY_MIB" "$original_sha" "$parallel_sha" \
    "$EXPECTED_INITIAL_HEAD" "$EXPECTED_PROTOCOL_SHA" "$EXPECTED_BASE_SHA" <<'PY'
import json
import pathlib
import sys
import tempfile
from datetime import datetime

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "orchestration_revision": "revision8_parallel_resource_only",
    "authorization": "user requested running v25 concurrently to reduce wall time",
    "scientific_protocol_changed": False,
    "parallel_resource_policy": {
        "foreign_gpu_workloads_allowed": True,
        "minimum_free_gpu_memory_mib_at_phase_start": int(sys.argv[4]),
        "clean_sample_count": 1,
    },
    "launch_snapshot": {
        "gpu_memory_used_mib": int(sys.argv[2]),
        "gpu_memory_free_mib": int(sys.argv[3]),
    },
    "provenance": {
        "original_revision7_guard2_script_sha256": sys.argv[5],
        "parallel_orchestrator_sha256": sys.argv[6],
        "frozen_git_head": sys.argv[7],
        "precalibration_protocol_sha256": sys.argv[8],
        "base_checkpoint_sha256": sys.argv[9],
    },
    "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
}
if path.exists():
    existing = json.loads(path.read_text())
    for key in ("schema_version", "orchestration_revision", "scientific_protocol_changed", "parallel_resource_policy", "provenance"):
        if existing.get(key) != payload.get(key):
            raise SystemExit(f"refusing mismatched existing parallel resource receipt: {key}")
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    temporary.replace(path)
PY
}

update_pr_comment() {
  local kind=$1
  python3 - "$ORCHESTRATION_ROOT/pr_comment.json" "$kind" "$RESULT_ROOT" \
    "$REMOTE_BRANCH" <<'PY'
import json
import pathlib
import sys

destination = pathlib.Path(sys.argv[1])
kind = sys.argv[2]
root = pathlib.Path(sys.argv[3])
branch = sys.argv[4]
base = (
    "https://github.com/lzqw/Safe100Humanoid/tree/"
    + branch
    + "/results/online/proximal_v25"
)
terminal_receipt = root / "orchestration/guardian_terminal_release_revision7.json"
if terminal_receipt.is_file():
    release_provenance = (
        "- Guardian release provenance: "
        f"[hash-bound terminal receipt]({base}/orchestration/guardian_terminal_release_revision7.json) "
        "(`guardian_completed=false`)"
    )
else:
    release_provenance = "- Guardian release provenance: normal completed-suite path"
if kind == "no_candidate":
    calibration = json.loads((root / "calibration/calibration_summary.json").read_text())
    attempts = calibration.get("attempts", [])
    body = "\n".join(
        [
            "## v25 revision-7 terminal calibration result",
            "",
            "The prospectively frozen, base-only paired calibration completed without a qualifying actuator-under-response candidate. Formal adaptation was therefore not started, and no grid, threshold, seed, or outcome-directed rerun was introduced.",
            "",
            f"- evaluated candidates: `{len(attempts)}`",
            "- adapted-policy evaluations: `0`",
            "- formal adaptation rounds: `0`",
            release_provenance,
            f"- compact evidence: [results/online/proximal_v25]({base})",
            "",
            "This is the protocol-defined terminal result for revision 7.",
        ]
    )
else:
    calibration = json.loads((root / "calibration/calibration_summary.json").read_text())
    training = json.loads((root / "training/training_summary.json").read_text())
    final = json.loads((root / "final/final_test.json").read_text())
    verification = json.loads((root / "final/verification.json").read_text())
    selected = calibration["selected_gate"]
    primary = final["primary_outcomes"]
    gate = final["development_gate"]
    body = "\n".join(
        [
            "## v25 revision-7 final result",
            "",
            "The prospectively frozen CBF-teacher experiment is complete: base-only first-qualifier calibration, exactly eight fixed rounds, and one fresh 512-pair four-condition audit.",
            "",
            f"- selected gain: `{calibration['selected_swing_underresponse_gain']}` (candidate `{calibration['selected_candidate_index']}`)",
            f"- calibration off/on success: `{selected['off_success_rate']:.3%}` / `{selected['on_success_rate']:.3%}`",
            f"- round-8 off success delta: `{primary['internalization_delta'] * 100:+.3f} pp`",
            f"- round-8 on success delta: `{primary['shielded_task_delta'] * 100:+.3f} pp`",
            f"- CBF interventions/riser relative reduction: `{primary['on_intervention_per_riser_relative_reduction']:.3%}`",
            f"- development gate: `{'PASS' if gate['passed'] else 'FAIL'}`",
            f"- independent verifier: `{'PASS' if verification['passed'] else 'FAIL'}` ({verification['check_count']} checks)",
            f"- hard rollbacks: `{training['hard_rollback_count']}`; performance rollbacks: `0`",
            release_provenance,
            f"- compact evidence and figures: [results/online/proximal_v25]({base})",
            "",
            "The fixed round-8 actor is reported unconditionally; there was no outcome-directed rerun or checkpoint selection.",
        ]
    )
destination.write_text(json.dumps({"body": body}, indent=2) + "\n")
PY
  if gh api --method PATCH \
    "repos/lzqw/Safe100Humanoid/issues/comments/$PR_COMMENT_ID" \
    --input "$ORCHESTRATION_ROOT/pr_comment.json" >/dev/null; then
    echo "$(timestamp) updated PR #$PR_NUMBER status comment"
  else
    echo "$(timestamp) WARNING: GitHub comment update failed; result commits were already pushed"
  fi
}

validate_final_resume_evidence() {
  "$PYTHON" - "$FINAL_ROOT" \
    "$RESULT_ROOT/protocol.json" \
    "$RESULT_ROOT/calibration/calibration_summary.json" \
    "$TRAINING_ROOT/training_summary.json" \
    "$BASE_CHECKPOINT" \
    "$TRAINING_ROOT/final_round_08.pt" \
    "$REPO/experiments/scripts/evaluate_cbf_teacher_v25.py" \
    "$REPO" <<'PY'
import csv
import hashlib
import json
import math
import pathlib
import sys

final_root = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
calibration_path = pathlib.Path(sys.argv[3])
training_path = pathlib.Path(sys.argv[4])
base_checkpoint = pathlib.Path(sys.argv[5])
final_checkpoint = pathlib.Path(sys.argv[6])
evaluator = pathlib.Path(sys.argv[7])
repo = pathlib.Path(sys.argv[8])
python = pathlib.Path(sys.executable)


def read_json(path):
    return json.loads(path.read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value):
    normalized = str(value).lower()
    if normalized not in ("true", "false"):
        raise RuntimeError(f"invalid final-audit CSV boolean {value!r}")
    return normalized == "true"


def finite_float(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {name} in final-audit evidence")
    return result


def close(actual, expected, *, tolerance=1.0e-12):
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def require_equal(actual, expected, message):
    if actual != expected:
        raise RuntimeError(f"{message}: {actual!r} != {expected!r}")


protocol = read_json(protocol_path)
calibration = read_json(calibration_path)
training = read_json(training_path)
started_path = final_root / "final_evaluation_started.json"
started = read_json(started_path)
evaluation = protocol["evaluation"]
conditions = tuple(evaluation["conditions"])
if conditions != ("pi0_off", "pi0_on", "pi8_on", "pi8_off"):
    raise RuntimeError("formal protocol contains an unexpected final condition order")
episodes = int(evaluation["episodes_per_condition"])
batch_size = int(evaluation["batch_size"])
if episodes < 1 or batch_size < 1 or episodes % batch_size:
    raise RuntimeError("formal final-audit batch schedule is invalid")
seed_base = int(evaluation["seed_base"])
seeds = tuple(seed_base + repeat for repeat in range(episodes // batch_size))
base_sha = file_sha256(base_checkpoint)
final_sha = file_sha256(final_checkpoint)
expected_started = {
    "protocol_id": protocol["protocol_id"],
    "protocol_sha256": file_sha256(protocol_path),
    "training_summary_sha256": file_sha256(training_path),
    "base_checkpoint_sha256": base_sha,
    "final_checkpoint_sha256": final_sha,
    "condition_order": list(conditions),
    "fresh_condition_count": episodes,
}
require_equal(started, expected_started, "final evaluation start marker changed")
require_equal(training.get("final_checkpoint_sha256"), final_sha, "round-8 checkpoint hash changed")

gain = float(calibration["selected_swing_underresponse_gain"])
policy_actor_hash = {
    "pi0": calibration["selected_actor_state_sha256"],
    "pi8": training["final_actor_sha256"],
}
checkpoint_sha = {"pi0": base_sha, "pi8": final_sha}
filter_mode = {
    "pi0_off": False,
    "pi0_on": True,
    "pi8_on": True,
    "pi8_off": False,
}
task = protocol["environment"]["task_id"]
environment_variant = protocol["environment"]["registered_variant"]
raw_root = final_root / "raw"
ledger_root = final_root / "execution_ledger"
expected_paths = {
    raw_root / condition / f"seed_{seed}" / name
    for condition in conditions
    for seed in seeds
    for name in ("summary.json", "episodes.csv")
}
expected_ledger_paths = {
    ledger_root / condition / f"seed_{seed}" / name
    for condition in conditions
    for seed in seeds
    for name in ("started.json", "completed.json")
}
if raw_root.exists():
    unexpected = sorted(
        str(path.relative_to(final_root))
        for path in raw_root.rglob("*")
        if path.is_file() and path not in expected_paths
    )
    if unexpected:
        raise RuntimeError(f"unexpected files in final-audit raw evidence: {unexpected}")
if ledger_root.exists():
    unexpected = sorted(
        str(path.relative_to(final_root))
        for path in ledger_root.rglob("*")
        if path.is_file() and path not in expected_ledger_paths
    )
    if unexpected:
        raise RuntimeError(f"unexpected files in final-audit execution ledger: {unexpected}")

completed = []
signatures = {}
pending_completion_ledgers = []
for condition in conditions:
    policy = condition.split("_", 1)[0]
    for seed in seeds:
        arm_root = raw_root / condition / f"seed_{seed}"
        summary_path = arm_root / "summary.json"
        csv_path = arm_root / "episodes.csv"
        ledger_dir = ledger_root / condition / f"seed_{seed}"
        started_ledger_path = ledger_dir / "started.json"
        completed_ledger_path = ledger_dir / "completed.json"
        command = [
            str(python),
            str(evaluator),
            "--repo",
            str(repo),
            "--checkpoint",
            str(base_checkpoint if policy == "pi0" else final_checkpoint),
            "--gain",
            str(gain),
            "--runtime-filter",
            "on" if filter_mode[condition] else "off",
            "--num-envs",
            str(batch_size),
            "--num-episodes",
            str(batch_size),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--output-json",
            str(summary_path),
            "--output-csv",
            str(csv_path),
        ]
        expected_started_ledger = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": file_sha256(protocol_path),
            "training_summary_sha256": file_sha256(training_path),
            "condition": condition,
            "seed": seed,
            "batch_size": batch_size,
            "checkpoint_sha256": checkpoint_sha[policy],
            "gain": gain,
            "runtime_filter": "on" if filter_mode[condition] else "off",
            "command_sha256": hashlib.sha256(
                json.dumps(command, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        if completed_ledger_path.exists() and not started_ledger_path.is_file():
            raise RuntimeError(
                f"completed final-audit ledger lacks start ledger: {condition} seed={seed}"
            )
        if started_ledger_path.is_file() and read_json(started_ledger_path) != expected_started_ledger:
            raise RuntimeError(
                f"final-audit start ledger changed: {condition} seed={seed}"
            )
        present = (summary_path.is_file(), csv_path.is_file())
        if present == (False, False):
            if started_ledger_path.exists() or completed_ledger_path.exists():
                raise RuntimeError(
                    "final-audit batch started without complete evidence; refusing seed reuse for "
                    f"{condition} seed={seed}"
                )
            continue
        if present != (True, True):
            raise RuntimeError(
                f"partial final-audit arm cannot be automatically resumed: {condition} seed={seed}"
            )
        if not started_ledger_path.is_file():
            raise RuntimeError(
                f"complete final-audit evidence lacks start ledger: {condition} seed={seed}"
            )
        expected_completed_ledger = {
            **expected_started_ledger,
            "summary_sha256": file_sha256(summary_path),
            "episodes_sha256": file_sha256(csv_path),
        }
        if completed_ledger_path.is_file():
            if read_json(completed_ledger_path) != expected_completed_ledger:
                raise RuntimeError(
                    f"completed final-audit evidence changed: {condition} seed={seed}"
                )
        else:
            pending_completion_ledgers.append((condition, seed))
        summary = read_json(summary_path)
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = tuple(reader.fieldnames or ())
        required_fields = {
            "evaluation_seed",
            "environment_id",
            "success",
            "fell",
            "timed_out",
            "failure_type",
            "toe_riser_kick",
            "toe_riser_kick_count",
            "toe_riser_overlap_fraction",
            "return",
            "steps",
            "max_riser",
            "completion_fraction",
            "intervention_count",
            "intervention_per_riser",
            "would_intervene_count",
            "would_intervene_per_riser",
            "mean_correction_norm",
            "mean_counterfactual_correction_norm",
            "teacher_reprojection_max_abs_error",
            "swing_selection_mismatch_count",
        }
        if not required_fields.issubset(fields):
            raise RuntimeError(f"final-audit arm has an incomplete CSV schema: {condition} seed={seed}")
        identities = [
            (int(row["evaluation_seed"]), int(row["environment_id"])) for row in rows
        ]
        expected_identities = [(seed, env_id) for env_id in range(batch_size)]
        require_equal(
            identities,
            expected_identities,
            f"final-audit arm identities changed for {condition} seed={seed}",
        )
        successes = [parse_bool(row["success"]) for row in rows]
        falls = [parse_bool(row["fell"]) for row in rows]
        timeouts = [parse_bool(row["timed_out"]) for row in rows]
        kicks = [parse_bool(row["toe_riser_kick"]) for row in rows]
        kick_counts = [int(row["toe_riser_kick_count"]) for row in rows]
        returns = [finite_float(row["return"], "return") for row in rows]
        steps = [int(row["steps"]) for row in rows]
        risers = [int(row["max_riser"]) for row in rows]
        interventions = [int(row["intervention_count"]) for row in rows]
        would_intervene = [int(row["would_intervene_count"]) for row in rows]
        corrections = [finite_float(row["mean_correction_norm"], "mean correction") for row in rows]
        counterfactual_corrections = [
            finite_float(row["mean_counterfactual_correction_norm"], "counterfactual correction")
            for row in rows
        ]
        reprojection = [
            finite_float(row["teacher_reprojection_max_abs_error"], "teacher reprojection")
            for row in rows
        ]
        mismatches = [int(row["swing_selection_mismatch_count"]) for row in rows]
        overlaps = [
            finite_float(row["toe_riser_overlap_fraction"], "toe/riser overlap")
            for row in rows
        ]
        if any(value < 1 for value in steps):
            raise RuntimeError("final-audit episode has a non-positive step count")
        if any(value < 0 for values in (kick_counts, risers, interventions, would_intervene, mismatches) for value in values):
            raise RuntimeError("final-audit episode has a negative count")
        if any(value < 0.0 for values in (corrections, counterfactual_corrections, reprojection) for value in values):
            raise RuntimeError("final-audit episode has a negative norm")
        if any(not 0.0 <= value <= 1.0 for value in overlaps):
            raise RuntimeError("final-audit overlap fraction is outside [0, 1]")
        for index, row in enumerate(rows):
            if kicks[index] != (kick_counts[index] > 0):
                raise RuntimeError("final-audit kick flag/count mismatch")
            expected_failure = (
                "success"
                if successes[index]
                else "toe_riser_under_clearance"
                if kicks[index]
                else "balance_or_other_fall"
                if falls[index]
                else "timeout_or_other_nonfall"
            )
            require_equal(row["failure_type"], expected_failure, "final-audit failure classification changed")
            if not close(row["intervention_per_riser"], interventions[index] / max(1, risers[index])):
                raise RuntimeError("final-audit intervention/riser row does not reconstruct")
            if not close(row["would_intervene_per_riser"], would_intervene[index] / max(1, risers[index])):
                raise RuntimeError("final-audit would-intervene/riser row does not reconstruct")
        if not filter_mode[condition] and (any(interventions) or any(value != 0.0 for value in corrections)):
            raise RuntimeError("CBF-off final arm claims an executed intervention")
        if any(value > 1.0e-6 for value in reprojection) or any(mismatches):
            raise RuntimeError("final-audit teacher/action routing invariant failed")

        n = len(rows)
        success_count = sum(successes)
        fall_count = sum(falls)
        timeout_count = sum(timeouts)
        failure_count = n - success_count
        toe_failure_count = sum((not success) and kick for success, kick in zip(successes, kicks, strict=True))
        total_risers = sum(risers)
        total_interventions = sum(interventions)
        total_would_intervene = sum(would_intervene)
        reconstructed = {
            "success_count": success_count,
            "success_rate": success_count / n,
            "fall_count": fall_count,
            "fall_rate": fall_count / n,
            "timeout_count": timeout_count,
            "timeout_rate": timeout_count / n,
            "failure_count": failure_count,
            "toe_riser_failure_count": toe_failure_count,
            "alignment_coverage": toe_failure_count / max(1, failure_count),
            "kick_episode_count": sum(kicks),
            "kick_rate": sum(kicks) / n,
            "mean_kick_count": sum(kick_counts) / n,
            "mean_return": sum(returns) / n,
            "mean_reached_riser": total_risers / n,
            "total_reached_risers": total_risers,
            "total_intervention_count": total_interventions,
            "total_would_intervene_count": total_would_intervene,
            "intervention_per_riser": total_interventions / max(1, total_risers),
            "would_intervene_per_riser": total_would_intervene / max(1, total_risers),
            "mean_correction_norm": sum(corrections) / n,
            "teacher_reprojection_max_abs_error": max(reprojection),
            "swing_selection_mismatch_count": sum(mismatches),
            "failure_type_counts": {
                failure_type: sum(row["failure_type"] == failure_type for row in rows)
                for failure_type in (
                    "toe_riser_under_clearance",
                    "balance_or_other_fall",
                    "timeout_or_other_nonfall",
                )
            },
        }
        for key, value in reconstructed.items():
            recorded = summary.get(key)
            if isinstance(value, float):
                if not isinstance(recorded, (int, float)) or not close(recorded, value):
                    raise RuntimeError(f"final-audit summary does not reconstruct {key}")
            else:
                require_equal(recorded, value, f"final-audit summary does not reconstruct {key}")
        expected_metadata = {
            "schema_version": 1,
            "task": task,
            "environment_variant": environment_variant,
            "checkpoint_sha256": checkpoint_sha[policy],
            "seed": seed,
            "num_envs": batch_size,
            "num_episodes": batch_size,
            "runtime_filter": filter_mode[condition],
            "deterministic_policy_mean": True,
            "one_initial_episode_per_env": True,
            "original_observation_interface": True,
            "actor_observation_dim": 405,
            "actor_state_sha256": policy_actor_hash[policy],
        }
        for key, value in expected_metadata.items():
            require_equal(summary.get(key), value, f"final-audit summary metadata changed for {key}")
        if not close(summary.get("swing_underresponse_gain"), gain):
            raise RuntimeError("final-audit summary uses the wrong gain")
        signature = summary.get("initial_state_signature")
        if not isinstance(signature, str) or len(signature) != 64 or any(
            character not in "0123456789abcdef" for character in signature
        ):
            raise RuntimeError("final-audit initial-state signature is malformed")
        previous_signature = signatures.setdefault(seed, signature)
        require_equal(signature, previous_signature, "completed final arms do not share initial states")
        completed.append(
            {
                "condition": condition,
                "seed": seed,
                "summary_sha256": file_sha256(summary_path),
                "episodes_sha256": file_sha256(csv_path),
            }
        )

if len(pending_completion_ledgers) > 1:
    raise RuntimeError(
        "more than one complete final-audit batch lacks its completion ledger: "
        f"{pending_completion_ledgers}"
    )

print(
    json.dumps(
        {
            "passed": True,
            "completed_arm_count": len(completed),
            "completed_episode_count": len(completed) * batch_size,
            "completed_arms": completed,
            "pending_completion_ledgers": pending_completion_ledgers,
            "partial_arms_accepted": 0,
            "simulator_episodes_run_by_validation": 0,
        },
        sort_keys=True,
    )
)
PY
}

run_final_audit_with_atomic_batch_ledger() {
  "$PYTHON" - "$FINAL_ROOT" \
    "$RESULT_ROOT/protocol.json" \
    "$RESULT_ROOT/calibration/calibration_summary.json" \
    "$TRAINING_ROOT/training_summary.json" \
    "$BASE_CHECKPOINT" \
    "$TRAINING_ROOT/final_round_08.pt" \
    "$REPO/experiments/scripts/evaluate_cbf_teacher_v25.py" \
    "$REPO/experiments/scripts/audit_cbf_teacher_v25.py" \
    "$REPO" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

final_root = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
calibration_path = pathlib.Path(sys.argv[3])
training_path = pathlib.Path(sys.argv[4])
base_checkpoint = pathlib.Path(sys.argv[5])
final_checkpoint = pathlib.Path(sys.argv[6])
evaluator = pathlib.Path(sys.argv[7])
auditor = pathlib.Path(sys.argv[8])
repo = pathlib.Path(sys.argv[9])
python = pathlib.Path(sys.executable)


def read_json(path):
    return json.loads(path.read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path, payload):
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise RuntimeError(f"refusing to overwrite different execution ledger: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered)
    temporary.replace(path)


protocol = read_json(protocol_path)
calibration = read_json(calibration_path)
training = read_json(training_path)
evaluation = protocol["evaluation"]
conditions = tuple(evaluation["conditions"])
if conditions != ("pi0_off", "pi0_on", "pi8_on", "pi8_off"):
    raise RuntimeError("formal protocol contains an unexpected final condition order")
episodes = int(evaluation["episodes_per_condition"])
batch_size = int(evaluation["batch_size"])
if episodes < 1 or batch_size < 1 or episodes % batch_size:
    raise RuntimeError("formal final-audit batch schedule is invalid")
seed_base = int(evaluation["seed_base"])
seeds = tuple(seed_base + repeat for repeat in range(episodes // batch_size))
gain = float(calibration["selected_swing_underresponse_gain"])
checkpoint = {"pi0": base_checkpoint, "pi8": final_checkpoint}
filter_mode = {
    "pi0_off": "off",
    "pi0_on": "on",
    "pi8_on": "on",
    "pi8_off": "off",
}

start_marker = final_root / "final_evaluation_started.json"
expected_start = {
    "protocol_id": protocol["protocol_id"],
    "protocol_sha256": file_sha256(protocol_path),
    "training_summary_sha256": file_sha256(training_path),
    "base_checkpoint_sha256": file_sha256(base_checkpoint),
    "final_checkpoint_sha256": file_sha256(final_checkpoint),
    "condition_order": list(conditions),
    "fresh_condition_count": episodes,
}
write_immutable(start_marker, expected_start)

ledger_root = final_root / "execution_ledger"
for condition in conditions:
    policy = condition.split("_", 1)[0]
    for seed in seeds:
        arm_root = final_root / "raw" / condition / f"seed_{seed}"
        summary_path = arm_root / "summary.json"
        episodes_path = arm_root / "episodes.csv"
        ledger_dir = ledger_root / condition / f"seed_{seed}"
        started_path = ledger_dir / "started.json"
        completed_path = ledger_dir / "completed.json"
        command = [
            str(python),
            str(evaluator),
            "--repo",
            str(repo),
            "--checkpoint",
            str(checkpoint[policy]),
            "--gain",
            str(gain),
            "--runtime-filter",
            filter_mode[condition],
            "--num-envs",
            str(batch_size),
            "--num-episodes",
            str(batch_size),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--output-json",
            str(summary_path),
            "--output-csv",
            str(episodes_path),
        ]
        command_sha = hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest()
        started = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": file_sha256(protocol_path),
            "training_summary_sha256": file_sha256(training_path),
            "condition": condition,
            "seed": seed,
            "batch_size": batch_size,
            "checkpoint_sha256": file_sha256(checkpoint[policy]),
            "gain": gain,
            "runtime_filter": filter_mode[condition],
            "command_sha256": command_sha,
        }
        if completed_path.exists():
            if not started_path.is_file():
                raise RuntimeError(f"completed batch lacks its start ledger: {condition} seed={seed}")
            if read_json(started_path) != started:
                raise RuntimeError(f"completed batch start ledger changed: {condition} seed={seed}")
            completed = read_json(completed_path)
            expected_completed = {
                **started,
                "summary_sha256": file_sha256(summary_path),
                "episodes_sha256": file_sha256(episodes_path),
            }
            if completed != expected_completed:
                raise RuntimeError(f"completed batch evidence changed: {condition} seed={seed}")
            continue
        if started_path.exists():
            raise RuntimeError(
                "final-audit batch has a start ledger without validated completion; "
                f"refusing seed reuse for {condition} seed={seed}"
            )
        if summary_path.exists() or episodes_path.exists() or arm_root.exists():
            raise RuntimeError(
                "final-audit evidence exists without an atomic start ledger; "
                f"refusing seed reuse for {condition} seed={seed}"
            )
        write_immutable(started_path, started)
        environment = dict(os.environ)
        environment.update(
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            PYTHONUNBUFFERED="1",
            CUDA_VISIBLE_DEVICES="0",
        )
        completed_process = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            check=False,
            text=True,
        )
        if completed_process.returncode:
            raise RuntimeError(
                f"final-audit batch failed for {condition} seed={seed}; "
                "start ledger preserved and automatic seed reuse forbidden"
            )
        if not summary_path.is_file() or not episodes_path.is_file():
            raise RuntimeError(
                f"final-audit batch produced incomplete evidence for {condition} seed={seed}"
            )
        # Validation and completion-record creation occur in the Bash caller
        # before another batch can launch.  Exit after exactly one new batch.
        print(
            json.dumps(
                {
                    "status": "one_batch_completed_pending_validation",
                    "condition": condition,
                    "seed": seed,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(75)

# The frozen auditor now finds all 16 raw arms complete and performs only its
# deterministic aggregation/final-report path.  Its resume checks still bind
# every summary to task, gain, checkpoint, seed, and arm configuration.
command = [
    str(python),
    str(auditor),
    "--repo",
    str(repo),
    "--base-checkpoint",
    str(base_checkpoint),
    "--final-checkpoint",
    str(final_checkpoint),
    "--training-summary",
    str(training_path),
    "--context",
    str(calibration_path.parent / "context.json"),
    "--protocol",
    str(protocol_path),
    "--output-dir",
    str(final_root),
    "--device",
    "cuda:0",
    "--resume",
]
completed = subprocess.run(command, cwd=repo, check=False)
raise SystemExit(completed.returncode)
PY
}

record_completed_final_batch() {
  "$PYTHON" - "$FINAL_ROOT" \
    "$RESULT_ROOT/protocol.json" \
    "$RESULT_ROOT/calibration/calibration_summary.json" \
    "$TRAINING_ROOT/training_summary.json" \
    "$BASE_CHECKPOINT" \
    "$TRAINING_ROOT/final_round_08.pt" \
    "$REPO/experiments/scripts/evaluate_cbf_teacher_v25.py" \
    "$REPO" <<'PY'
import hashlib
import json
import pathlib
import sys

final_root = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
calibration_path = pathlib.Path(sys.argv[3])
training_path = pathlib.Path(sys.argv[4])
base_checkpoint = pathlib.Path(sys.argv[5])
final_checkpoint = pathlib.Path(sys.argv[6])
evaluator = pathlib.Path(sys.argv[7])
repo = pathlib.Path(sys.argv[8])
python = pathlib.Path(sys.executable)


def read_json(path):
    return json.loads(path.read_text())


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


protocol = read_json(protocol_path)
calibration = read_json(calibration_path)
training = read_json(training_path)
evaluation = protocol["evaluation"]
conditions = tuple(evaluation["conditions"])
batch_size = int(evaluation["batch_size"])
episodes = int(evaluation["episodes_per_condition"])
seeds = tuple(
    int(evaluation["seed_base"]) + repeat for repeat in range(episodes // batch_size)
)
gain = float(calibration["selected_swing_underresponse_gain"])
checkpoint = {"pi0": base_checkpoint, "pi8": final_checkpoint}
filter_mode = {
    "pi0_off": "off",
    "pi0_on": "on",
    "pi8_on": "on",
    "pi8_off": "off",
}
pending = []
for condition in conditions:
    policy = condition.split("_", 1)[0]
    for seed in seeds:
        arm_root = final_root / "raw" / condition / f"seed_{seed}"
        ledger_dir = final_root / "execution_ledger" / condition / f"seed_{seed}"
        started_path = ledger_dir / "started.json"
        completed_path = ledger_dir / "completed.json"
        if completed_path.exists():
            continue
        if started_path.is_file():
            pending.append((condition, policy, seed, arm_root, started_path, completed_path))
if not pending:
    print(json.dumps({"recorded": False, "reason": "no_pending_batch"}, sort_keys=True))
    raise SystemExit(0)
if len(pending) != 1:
    raise RuntimeError(f"expected exactly one pending final batch after launch, found {len(pending)}")
condition, policy, seed, arm_root, started_path, completed_path = pending[0]
summary_path = arm_root / "summary.json"
episodes_path = arm_root / "episodes.csv"
if not summary_path.is_file() or not episodes_path.is_file():
    raise RuntimeError("pending final batch does not have complete output files")
started = read_json(started_path)
command = [
    str(python),
    str(evaluator),
    "--repo",
    str(repo),
    "--checkpoint",
    str(checkpoint[policy]),
    "--gain",
    str(gain),
    "--runtime-filter",
    filter_mode[condition],
    "--num-envs",
    str(batch_size),
    "--num-episodes",
    str(batch_size),
    "--seed",
    str(seed),
    "--device",
    "cuda:0",
    "--output-json",
    str(summary_path),
    "--output-csv",
    str(episodes_path),
]
command_sha = hashlib.sha256(
    json.dumps(command, separators=(",", ":")).encode()
).hexdigest()
expected_started = {
    "schema_version": 1,
    "protocol_id": protocol["protocol_id"],
    "protocol_sha256": file_sha256(protocol_path),
    "training_summary_sha256": file_sha256(training_path),
    "condition": condition,
    "seed": seed,
    "batch_size": batch_size,
    "checkpoint_sha256": file_sha256(checkpoint[policy]),
    "gain": gain,
    "runtime_filter": filter_mode[condition],
    "command_sha256": command_sha,
}
if started != expected_started:
    raise RuntimeError("pending final batch start ledger changed")
completed = {
    **started,
    "summary_sha256": file_sha256(summary_path),
    "episodes_sha256": file_sha256(episodes_path),
}
rendered = json.dumps(completed, indent=2, sort_keys=True) + "\n"
temporary = completed_path.with_name(f".{completed_path.name}.tmp")
temporary.write_text(rendered)
temporary.replace(completed_path)
print(json.dumps({"recorded": True, "condition": condition, "seed": seed}, sort_keys=True))
PY
}

build_external_manifest() {
  python3 - "$EXTERNAL_ROOT" "$RESULT_ROOT/external_artifact_manifest.json" \
    "$RESULT_ROOT/external_artifact_inventory.sha256" <<'PY'
import hashlib
import json
import pathlib
import sys

external_root = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
inventory_path = pathlib.Path(sys.argv[3])
entries = []
for phase in ("calibration", "training", "final"):
    phase_root = external_root / phase
    if not phase_root.is_dir():
        continue
    for path in sorted(item for item in phase_root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(
            {
                "path": str(path.relative_to(external_root)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
payload = {
    "schema_version": 1,
    "experiment": "proximal_v25_swing_teacher",
    "scope": "complete external calibration/training/final artifact store; orchestration logs excluded",
    "external_root": str(external_root),
    "file_count": len(entries),
    "total_bytes": sum(item["bytes"] for item in entries),
    "files": entries,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
inventory_path.write_text(
    "".join(f"{item['sha256']}  {item['path']}\n" for item in entries)
)
PY
}

build_calibration_pair_evidence() {
  python3 - "$CALIBRATION_ROOT" \
    "$PRECALIBRATION_PROTOCOL" \
    "$RESULT_ROOT/calibration/calibration_summary.json" \
    "$RESULT_ROOT/calibration/attempts.json" \
    "$RESULT_ROOT/calibration/all_evaluated_paired_episodes.csv" \
    "$RESULT_ROOT/calibration/calibration_evidence_verification.json" <<'PY'
import csv
import hashlib
import io
import json
import math
import pathlib
import sys

calibration_root = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
summary_path = pathlib.Path(sys.argv[3])
attempts_path = pathlib.Path(sys.argv[4])
paired_output = pathlib.Path(sys.argv[5])
verification_output = pathlib.Path(sys.argv[6])


def read_json(path):
    return json.loads(path.read_text())


def parse_bool(value):
    normalized = str(value).lower()
    if normalized not in ("true", "false"):
        raise RuntimeError(f"invalid calibration CSV boolean {value!r}")
    return normalized == "true"


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exact(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite different evidence: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields or not rows:
        raise RuntimeError(f"empty calibration episode table: {path}")
    return fields, rows


def identity(row):
    return int(row["evaluation_seed"]), int(row["environment_id"])


def reconstruct_gate(off_rows, on_rows, calibration):
    off_success = [parse_bool(row["success"]) for row in off_rows]
    on_success = [parse_bool(row["success"]) for row in on_rows]
    off_kick = [parse_bool(row["toe_riser_kick"]) for row in off_rows]
    paired_count = len(off_rows)
    off_success_count = sum(off_success)
    on_success_count = sum(on_success)
    off_failure_count = paired_count - off_success_count
    aligned = sum(
        (not success) and kick
        for success, kick in zip(off_success, off_kick, strict=True)
    )
    rescued = sum(
        (not off) and on
        for off, on in zip(off_success, on_success, strict=True)
    )
    off_success_rate = off_success_count / paired_count
    on_success_rate = on_success_count / paired_count
    alignment = aligned / max(1, off_failure_count)
    rescue = rescued / max(1, off_failure_count)
    off_bounds = calibration["off_success_bounds_inclusive"]
    on_bounds = calibration["on_success_bounds_inclusive"]
    conditions = {
        "alignment_coverage_at_least_80pct": (
            off_failure_count > 0
            and alignment >= calibration["alignment_coverage_minimum"]
        ),
        "shield_rescue_rate_at_least_60pct": (
            off_failure_count > 0
            and rescue >= calibration["shield_rescue_rate_minimum"]
        ),
        "off_success_in_40_to_65pct": (
            off_bounds[0] <= off_success_rate <= off_bounds[1]
        ),
        "on_success_in_80_to_95pct": (
            on_bounds[0] <= on_success_rate <= on_bounds[1]
        ),
    }
    return {
        "qualifies": all(conditions.values()),
        "conditions": conditions,
        "paired_count": paired_count,
        "off_success_count": off_success_count,
        "off_success_rate": off_success_rate,
        "on_success_count": on_success_count,
        "on_success_rate": on_success_rate,
        "off_failure_count": off_failure_count,
        "off_toe_riser_failure_count": aligned,
        "alignment_coverage": alignment,
        "rescued_count": rescued,
        "shield_rescue_rate": rescue,
    }


def assert_gate_matches(actual, expected, candidate_index):
    if set(actual) != set(expected):
        raise RuntimeError(
            f"candidate {candidate_index} gate fields differ: "
            f"{sorted(actual)} != {sorted(expected)}"
        )
    for key, value in actual.items():
        reference = expected[key]
        if isinstance(value, float):
            matches = isinstance(reference, (int, float)) and math.isclose(
                value, float(reference), rel_tol=0.0, abs_tol=1.0e-12
            )
        else:
            matches = value == reference
        if not matches:
            raise RuntimeError(
                f"candidate {candidate_index} gate mismatch for {key}: "
                f"{value!r} != {reference!r}"
            )


protocol = read_json(protocol_path)
summary = read_json(summary_path)
attempts = read_json(attempts_path)
if attempts != summary.get("attempts"):
    raise RuntimeError("attempts.json differs from calibration summary attempts")
if not isinstance(attempts, list) or not attempts:
    raise RuntimeError("calibration contains no completed candidate attempts")
if summary.get("candidate_count_evaluated") != len(attempts):
    raise RuntimeError("calibration candidate count differs from completed attempts")

calibration = protocol["calibration"]
grid = protocol["shift_family"]["candidate_grid"]
expected_indices = list(range(len(attempts)))
actual_indices = [int(item["candidate_index"]) for item in attempts]
if actual_indices != expected_indices:
    raise RuntimeError("calibration candidates are not a contiguous light-to-severe prefix")
if len(attempts) > len(grid):
    raise RuntimeError("calibration evaluated more candidates than the frozen grid")
for attempt, frozen in zip(attempts, grid, strict=False):
    if attempt["candidate_index"] != frozen["candidate_index"]:
        raise RuntimeError("calibration candidate index differs from frozen grid")
    if not math.isclose(
        float(attempt["swing_underresponse_gain"]),
        float(frozen["swing_underresponse_gain"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("calibration gain differs from frozen grid")
    if attempt["evaluation_seeds"] != frozen["evaluation_seeds"]:
        raise RuntimeError("calibration seeds differ from frozen grid")

status = summary.get("status")
qualifiers = [bool(item["qualifies"]) for item in attempts]
if status == "first_qualifying_candidate_frozen":
    if qualifiers[:-1] != [False] * (len(qualifiers) - 1) or not qualifiers[-1]:
        raise RuntimeError("selected calibration attempt is not the first qualifier")
    if summary.get("selected_candidate_index") != attempts[-1]["candidate_index"]:
        raise RuntimeError("selected candidate differs from first qualifying attempt")
    if not math.isclose(
        float(summary.get("selected_swing_underresponse_gain")),
        float(attempts[-1]["swing_underresponse_gain"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("selected gain differs from first qualifying attempt")
elif status == "no_candidate_qualified":
    if len(attempts) != len(grid) or any(qualifiers):
        raise RuntimeError("no-candidate result did not exhaust the frozen grid")
else:
    raise RuntimeError(f"unsupported calibration terminal status: {status!r}")

expected_batch_size = int(calibration["batch_size"])
expected_repeats = int(calibration["repeats_per_candidate"])
expected_pairs = int(calibration["episodes_per_candidate"])
if expected_batch_size * expected_repeats != expected_pairs:
    raise RuntimeError("frozen calibration batch schedule is internally inconsistent")

base_sha = protocol["base_checkpoint"]["sha256"]
merged_rows = []
raw_fields = None
candidate_verification = []
input_hash_rows = []
selected_expected_rows = []
for attempt in attempts:
    candidate_index = int(attempt["candidate_index"])
    gain = float(attempt["swing_underresponse_gain"])
    candidate_off = []
    candidate_on = []
    signatures = []
    actor_hash = None
    for seed in attempt["evaluation_seeds"]:
        arm_rows = {}
        arm_summaries = {}
        arm_fields = {}
        for arm in ("off", "on"):
            arm_root = (
                calibration_root
                / "candidates"
                / f"candidate_{candidate_index:02d}"
                / f"seed_{seed}"
                / arm
            )
            summary_file = arm_root / "summary.json"
            csv_file = arm_root / "episodes.csv"
            if not summary_file.is_file():
                raise FileNotFoundError(summary_file)
            arm_summary = read_json(summary_file)
            fields, rows = read_csv(csv_file)
            if len(rows) != expected_batch_size:
                raise RuntimeError(f"incomplete {arm} batch for candidate {candidate_index}")
            if arm_summary.get("seed") != seed:
                raise RuntimeError("raw calibration summary has the wrong seed")
            if arm_summary.get("num_envs") != expected_batch_size or arm_summary.get(
                "num_episodes"
            ) != expected_batch_size:
                raise RuntimeError("raw calibration summary has the wrong batch size")
            if arm_summary.get("runtime_filter") is not (arm == "on"):
                raise RuntimeError("raw calibration summary has the wrong filter arm")
            if not math.isclose(
                float(arm_summary.get("swing_underresponse_gain")),
                gain,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError("raw calibration summary has the wrong gain")
            if arm_summary.get("checkpoint_sha256") != base_sha:
                raise RuntimeError("raw calibration arm did not use frozen pi0")
            expected_identities = [(seed, value) for value in range(expected_batch_size)]
            rows.sort(key=identity)
            if [identity(row) for row in rows] != expected_identities:
                raise RuntimeError("raw calibration batch identities are incomplete")
            arm_summaries[arm] = arm_summary
            arm_fields[arm] = fields
            arm_rows[arm] = rows
            for source in (summary_file, csv_file):
                input_hash_rows.append(
                    f"{file_sha256(source)}  {source.relative_to(calibration_root)}"
                )
        if arm_fields["off"] != arm_fields["on"]:
            raise RuntimeError("off/on raw calibration CSV schemas differ")
        if raw_fields is None:
            raw_fields = arm_fields["off"]
        elif raw_fields != arm_fields["off"]:
            raise RuntimeError("raw calibration CSV schema changed between candidates")
        off_summary = arm_summaries["off"]
        on_summary = arm_summaries["on"]
        if off_summary.get("initial_state_signature") != on_summary.get(
            "initial_state_signature"
        ):
            raise RuntimeError("raw off/on calibration initial states differ")
        if off_summary.get("actor_state_sha256") != on_summary.get(
            "actor_state_sha256"
        ):
            raise RuntimeError("raw off/on calibration actors differ")
        if actor_hash is None:
            actor_hash = off_summary["actor_state_sha256"]
        elif actor_hash != off_summary["actor_state_sha256"]:
            raise RuntimeError("pi0 changed between calibration repeats")
        signatures.append(off_summary["initial_state_signature"])
        for off, on in zip(arm_rows["off"], arm_rows["on"], strict=True):
            if identity(off) != identity(on):
                raise RuntimeError("raw calibration off/on row identities differ")
            candidate_off.append(off)
            candidate_on.append(on)
            merged = {
                "candidate_index": str(candidate_index),
                "swing_underresponse_gain": format(gain, ".17g"),
                "evaluation_seed": off["evaluation_seed"],
                "environment_id": off["environment_id"],
            }
            for field in raw_fields:
                if field not in ("evaluation_seed", "environment_id"):
                    merged[f"off_{field}"] = off[field]
                    merged[f"on_{field}"] = on[field]
            merged_rows.append(merged)
    if len(candidate_off) != expected_pairs or len(candidate_on) != expected_pairs:
        raise RuntimeError("candidate does not contain the frozen number of pairs")
    if len(set(map(identity, candidate_off))) != expected_pairs:
        raise RuntimeError("candidate calibration identities are not unique")
    reconstructed = reconstruct_gate(candidate_off, candidate_on, calibration)
    expected_gate = {
        key: attempt[key]
        for key in (
            "qualifies",
            "conditions",
            "paired_count",
            "off_success_count",
            "off_success_rate",
            "on_success_count",
            "on_success_rate",
            "off_failure_count",
            "off_toe_riser_failure_count",
            "alignment_coverage",
            "rescued_count",
            "shield_rescue_rate",
        )
    }
    assert_gate_matches(reconstructed, expected_gate, candidate_index)
    if signatures != attempt["off_initial_state_signatures"] or signatures != attempt[
        "on_initial_state_signatures"
    ]:
        raise RuntimeError("attempt-level initial-state signatures differ from raw arms")
    candidate_verification.append(
        {
            "candidate_index": candidate_index,
            "swing_underresponse_gain": gain,
            "paired_count": expected_pairs,
            "actor_state_sha256": actor_hash,
            "gate": reconstructed,
            "gate_matches_summary": True,
            "paired_identities_complete_and_unique": True,
            "off_on_initial_state_signatures_match": True,
        }
    )
    if status == "first_qualifying_candidate_frozen" and candidate_index == int(
        summary["selected_candidate_index"]
    ):
        for off, on in zip(candidate_off, candidate_on, strict=True):
            selected_expected_rows.append(
                {
                    "evaluation_seed": off["evaluation_seed"],
                    "environment_id": off["environment_id"],
                    "off_success": off["success"],
                    "on_success": on["success"],
                    "off_fell": off["fell"],
                    "on_fell": on["fell"],
                    "off_toe_riser_kick": off["toe_riser_kick"],
                    "on_toe_riser_kick": on["toe_riser_kick"],
                    "off_failure_type": off["failure_type"],
                    "on_failure_type": on["failure_type"],
                    "off_max_riser": off["max_riser"],
                    "on_max_riser": on["max_riser"],
                    "off_would_intervene_count": off["would_intervene_count"],
                    "on_intervention_count": on["intervention_count"],
                    "off_would_intervene_per_riser": off[
                        "would_intervene_per_riser"
                    ],
                    "on_intervention_per_riser": on["intervention_per_riser"],
                }
            )

if raw_fields is None:
    raise RuntimeError("calibration evidence has no raw CSV schema")
output_fields = [
    "candidate_index",
    "swing_underresponse_gain",
    "evaluation_seed",
    "environment_id",
]
for field in raw_fields:
    if field not in ("evaluation_seed", "environment_id"):
        output_fields.extend((f"off_{field}", f"on_{field}"))
buffer = io.StringIO(newline="")
writer = csv.DictWriter(buffer, fieldnames=output_fields, lineterminator="\n")
writer.writeheader()
writer.writerows(merged_rows)
paired_payload = buffer.getvalue().encode()
write_exact(paired_output, paired_payload)

selected_matches = None
if status == "first_qualifying_candidate_frozen":
    selected_file = calibration_root / "selected_paired_episodes.csv"
    selected_fields, selected_rows = read_csv(selected_file)
    if selected_fields != list(selected_expected_rows[0]):
        raise RuntimeError("selected calibration paired CSV schema differs")
    if selected_rows != selected_expected_rows:
        raise RuntimeError("selected calibration paired CSV differs from raw arms")
    selected_matches = True

input_hash_rows.sort()
verification = {
    "schema_version": 1,
    "protocol_id": protocol["protocol_id"],
    "calibration_status": status,
    "candidate_count_evaluated": len(attempts),
    "all_evaluated_paired_episode_count": len(merged_rows),
    "expected_pairs_per_candidate": expected_pairs,
    "ordered_frozen_candidate_prefix": True,
    "first_qualifier_rule_verified": True,
    "all_candidate_gates_reconstructed": True,
    "all_raw_off_on_identities_verified": True,
    "selected_paired_csv_matches_raw_arms": selected_matches,
    "raw_input_file_count": len(input_hash_rows),
    "raw_input_inventory_sha256": sha256_bytes(
        ("\n".join(input_hash_rows) + "\n").encode()
    ),
    "all_evaluated_paired_csv": {
        "path": str(paired_output.name),
        "bytes": len(paired_payload),
        "sha256": sha256_bytes(paired_payload),
    },
    "candidates": candidate_verification,
    "passed": True,
}
verification_payload = (
    json.dumps(verification, indent=2, sort_keys=True) + "\n"
).encode()
write_exact(verification_output, verification_payload)
PY
}

copy_orchestration_sources() {
  copy_exact "$LEGACY_QUEUE_SCRIPT" "$RESULT_ROOT/orchestration/v25_wait_and_calibrate_revision7.sh"
  copy_exact "$LEGACY_COMPLETION_SCRIPT" "$RESULT_ROOT/orchestration/v25_complete_after_calibration_revision7.sh"
  copy_exact "$LEGACY_SUPERVISOR_SCRIPT" "$RESULT_ROOT/orchestration/v25_completion_supervisor_revision7.sh"
  copy_exact "$LEGACY_WATCHDOG_SCRIPT" "$RESULT_ROOT/orchestration/v25_supervisor_watchdog_revision7.sh"
  copy_exact "$QUEUE_SCRIPT" "$RESULT_ROOT/orchestration/v25_wait_and_calibrate_revision7_guard2.sh"
  copy_exact "$COMPLETION_SCRIPT" "$RESULT_ROOT/orchestration/v25_complete_after_calibration_revision7_guard2.sh"
  copy_exact "$SUPERVISOR_SCRIPT" "$RESULT_ROOT/orchestration/v25_completion_supervisor_revision7_guard2.sh"
  copy_exact "$WATCHDOG_SCRIPT" "$RESULT_ROOT/orchestration/v25_supervisor_watchdog_revision7_guard2.sh"
  copy_exact "$GUARD2_MIGRATION_RECEIPT" \
    "$RESULT_ROOT/orchestration/v25_revision7_guard2_migration_receipt.json"
  if guardian_complete; then
    :
  elif guardian_release_receipt_valid; then
    copy_exact "$GUARDIAN_RELEASE_RECEIPT" \
      "$RESULT_ROOT/orchestration/guardian_terminal_release_revision7.json"
  else
    fail "Guardian release provenance became invalid before result packaging"
  fi
  python3 - "$RESULT_ROOT/orchestration/execution_orchestration.json" \
    "$RESULT_ROOT/orchestration/v25_wait_and_calibrate_revision7.sh" \
    "$RESULT_ROOT/orchestration/v25_complete_after_calibration_revision7.sh" \
    "$RESULT_ROOT/orchestration/v25_completion_supervisor_revision7.sh" \
    "$RESULT_ROOT/orchestration/v25_supervisor_watchdog_revision7.sh" \
    "$RESULT_ROOT/orchestration/v25_wait_and_calibrate_revision7_guard2.sh" \
    "$RESULT_ROOT/orchestration/v25_complete_after_calibration_revision7_guard2.sh" \
    "$RESULT_ROOT/orchestration/v25_completion_supervisor_revision7_guard2.sh" \
    "$RESULT_ROOT/orchestration/v25_supervisor_watchdog_revision7_guard2.sh" \
    "$RESULT_ROOT/orchestration/v25_revision7_guard2_migration_receipt.json" \
    "$RESULT_ROOT/orchestration/guardian_terminal_release_revision7.json" \
    "$EXPECTED_INITIAL_HEAD" "$EXPECTED_PROTOCOL_SHA" <<'PY'
import hashlib
import json
import pathlib
import sys

destination = pathlib.Path(sys.argv[1])
files = []
for value in sys.argv[2:11]:
    path = pathlib.Path(value)
    files.append(
        {
            "file": str(path.relative_to(destination.parents[1])),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
terminal_receipt = pathlib.Path(sys.argv[11])
if terminal_receipt.is_file():
    files.append(
        {
            "file": str(terminal_receipt.relative_to(destination.parents[1])),
            "sha256": hashlib.sha256(terminal_receipt.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema_version": 1,
    "precalibration_revision": 7,
    "orchestration_revision": "revision7_guard2",
    "frozen_initial_git_commit": sys.argv[12],
    "precalibration_protocol_sha256": sys.argv[13],
    "role": "post-freeze execution orchestration only; not part of the bound algorithm or environment",
    "safety_properties": {
        "guardian_suite_and_restart_watchdog_must_both_exit": True,
        "guardian_completed_or_hash_bound_terminal_release_required_before_any_gpu_phase": True,
        "guardian_terminal_release_requires_five_60s_unchanged_evidence_samples": True,
        "guardian_terminal_release_receipt_sha256_is_runtime_pinned": True,
        "five_clean_gpu_samples_before_each_new_gpu_phase": True,
        "foreign_gpu_scheduler_must_exit_before_each_new_gpu_phase": True,
        "all_cvci_back_run_recovery_finalize_schedulers_must_exit": True,
        "guardianflow_gpu_restart_controllers_must_exit": True,
        "guard2_migration_occurred_before_any_v25_simulator_episode": True,
        "idle_gpu_memory_ceiling_mib": 2048,
        "all_calibration_pair_rows_reconstructed_before_publish": True,
        "interrupted_calibration_seed_never_automatically_reused": True,
        "never_force_push": True,
        "remote_branch_tip_verified_after_each_push": True,
        "formal_adaptation_seed_never_automatically_restarted": True,
        "final_audit_batch_has_atomic_start_and_completion_ledgers": True,
        "final_audit_can_resume_only_hash_bound_reconstructed_complete_arms": True,
        "supervisor_restarted_only_before_terminal_or_partial_adaptation_state": True,
    },
    "files": files,
}
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

build_final_readme() {
  python3 - "$RESULT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
calibration = json.loads((root / "calibration/calibration_summary.json").read_text())
protocol = json.loads((root / "protocol.json").read_text())
training = json.loads((root / "training/training_summary.json").read_text())
final = json.loads((root / "final/final_test.json").read_text())
verification = json.loads((root / "final/verification.json").read_text())
manifest = json.loads((root / "external_artifact_manifest.json").read_text())
terminal_receipt = root / "orchestration/guardian_terminal_release_revision7.json"
release_evidence_line = (
    "- [Guardian terminal release receipt](orchestration/guardian_terminal_release_revision7.json) (`guardian_completed=false`)\n"
    if terminal_receipt.is_file()
    else ""
)
selected = calibration["selected_gate"]
conditions = final["conditions"]
primary = final["primary_outcomes"]
gate = final["development_gate"]
round_lines = []
for row in training["rounds"]:
    metrics = row.get("metrics", {})
    round_lines.append(
        "| {round} | {status} | {success:.3%} | {teacher:.3%} | {kl:.8f} |".format(
            round=row["round"],
            status=row["status"],
            success=float(metrics.get("rollout_success_rate", 0.0)),
            teacher=float(metrics.get("teacher_transition_fraction", 0.0)),
            kl=float(metrics.get("moving_forward_kl", 0.0)),
        )
    )
condition_lines = []
for name in ("pi0_off", "pi0_on", "pi8_on", "pi8_off"):
    row = conditions[name]
    condition_lines.append(
        f"| `{name}` | {row['success_rate']:.3%} | {row['kick_rate']:.3%} | {row['intervention_per_riser']:.6f} |"
    )
failed = ", ".join(verification.get("failed_checks", [])) or "none"
verdict = "PASS" if gate["passed"] else "FAIL"
verify_verdict = "PASS" if verification["passed"] else "FAIL"
text = f"""# v25 CBF-Teacher Swing Under-Response — Final Result

## Outcome / 结论

The single prospectively frozen v25 experiment is complete. The unconditional
round-8 actor **{verdict}ED** the point-estimate development gate. The off-policy
success change was `{primary['internalization_delta'] * 100:+.3f} pp`, the shielded
success change was `{primary['shielded_task_delta'] * 100:+.3f} pp`, and runtime-CBF
interventions per reached riser changed by
`{primary['on_intervention_per_riser_relative_reduction']:+.3%}` (positive means a
reduction). The independent evidence verifier **{verify_verdict}ED**
`{verification['check_count']}` checks; failed checks: `{failed}`.

单次、预先冻结的 v25 实验已经完成。固定使用 round-8 actor，不按结果选择
checkpoint，也没有结果导向重跑。off success 变化
`{primary['internalization_delta'] * 100:+.3f} pp`，on success 变化
`{primary['shielded_task_delta'] * 100:+.3f} pp`，CBF 每个已到达台阶的干预相对变化
为 `{primary['on_intervention_per_riser_relative_reduction']:+.3%}`；开发门槛结果为
**{verdict}**，独立校验为 **{verify_verdict}**。

## Base-only first-qualifier calibration / 基础策略配对校准

Only the active swing-leg hip pitch, knee, and ankle pitch response was changed.
Terrain geometry, friction, command, controller, observations, reset process, and
CBF geometry remained fixed. Candidates were evaluated light-to-severe, stopping
at the first qualifier.

| Item | Result |
| --- | ---: |
| candidate / gain | `{calibration['selected_candidate_index']}` / `{calibration['selected_swing_underresponse_gain']}` |
| paired episodes | `{selected['paired_count']}` |
| pi0 CBF-off success | `{selected['off_success_count']}/{selected['paired_count']}` (`{selected['off_success_rate']:.3%}`) |
| pi0 CBF-on success | `{selected['on_success_count']}/{selected['paired_count']}` (`{selected['on_success_rate']:.3%}`) |
| aligned off failures | `{selected['off_toe_riser_failure_count']}/{selected['off_failure_count']}` (`{selected['alignment_coverage']:.3%}`) |
| shield-rescued off failures | `{selected['rescued_count']}/{selected['off_failure_count']}` (`{selected['shield_rescue_rate']:.3%}`) |

## Fixed eight-round update / 固定八轮更新

The actor loss was clipped PPO plus `0.5 * KL(pi_theta || pi_k)` plus
`0.1 * L_CBF_teacher`. PPO stored the raw sampled policy action while the runtime
environment executed the CBF-filtered safe action. One original-observation actor,
one privileged critic, and one on-policy `64 x 1024` batch per round were used.

| Round | Status | Rollout success | Teacher transitions | Moving forward KL |
| ---: | --- | ---: | ---: | ---: |
{chr(10).join(round_lines)}

Hard rollbacks: `{training['hard_rollback_count']}`. Performance rollbacks: `0`.
The external fixed round-8 checkpoint SHA-256 is
`{training['final_checkpoint_sha256']}`.

## One fresh four-condition audit / 唯一 fresh 四条件终评

All four arms used the same 512 fresh initial-condition identities and the
deterministic policy mean.

| Condition | Success | Toe/riser kick | Interventions/riser |
| --- | ---: | ---: | ---: |
{chr(10).join(condition_lines)}

Development-gate conditions:

- off success delta at least +5 pp: `{gate['conditions']['off_success_delta_at_least_five_pp']}`
- on success delta nonnegative: `{gate['conditions']['on_success_delta_nonnegative']}`
- off kick rate strictly decreases: `{gate['conditions']['off_policy_kick_rate_strictly_decreases']}`
- on interventions/riser decrease at least 20%: `{gate['conditions']['shield_interventions_per_riser_decrease_at_least_20pct']}`

## Integrity and evidence / 完整性与证据

- [revision-7 precalibration protocol](precalibration_protocol_revision7.json)
- [formal protocol](protocol.json)
- [calibration summary](calibration/calibration_summary.json)
- [all evaluated calibration pairs](calibration/all_evaluated_paired_episodes.csv)
- [independent calibration evidence reconstruction](calibration/calibration_evidence_verification.json)
- [512 calibration pairs](calibration/selected_paired_episodes.csv)
- [eight-round training summary](training/training_summary.json)
- [per-round metrics](training/round_metrics.csv)
- [authoritative final result](final/final_test.json)
- [512 four-condition episode rows](final/paired_episode_metrics.csv)
- [per-batch final execution ledgers](final/execution_ledger/)
- [independent verification](final/verification.json)
- [three figure categories](figures/figure_manifest.json)
- [external artifact manifest](external_artifact_manifest.json)
- [execution orchestration provenance](orchestration/execution_orchestration.json)
- [zero-episode resource-guard migration receipt](orchestration/v25_revision7_guard2_migration_receipt.json)
{release_evidence_line}- [compact-package hashes](SHA256SUMS)

The external manifest binds `{manifest['file_count']}` files totaling
`{manifest['total_bytes']}` bytes, including recovery checkpoints and raw isolated
evaluations. Those large files remain on the 4080 artifact store. The compact Git
package contains the complete metric-level evidence. Formal protocol SHA-256:
`{hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()}`.
"""
(root / "README.md").write_text(text)
PY
}

build_no_candidate_package() {
  copy_orchestration_sources
  build_external_manifest
  python3 - "$RESULT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "calibration/calibration_summary.json").read_text())
terminal_receipt = root / "orchestration/guardian_terminal_release_revision7.json"
release_evidence_line = (
    "- [Guardian terminal release receipt](orchestration/guardian_terminal_release_revision7.json) (`guardian_completed=false`)\n"
    if terminal_receipt.is_file()
    else ""
)
attempts = summary.get("attempts", [])
rows = []
for item in attempts:
    rows.append(
        f"| {item['candidate_index']} | {item['swing_underresponse_gain']} | "
        f"{item['off_success_rate']:.3%} | {item['on_success_rate']:.3%} | "
        f"{item['alignment_coverage']:.3%} | {item['shield_rescue_rate']:.3%} |"
    )
text = f"""# v25 Revision-7 — Terminal Calibration Result

The prospectively frozen base-only calibration evaluated all `{len(attempts)}`
ordered actuator-under-response candidates and found no first qualifier. Per the
frozen protocol, formal adaptation and final evaluation were not started. No grid,
threshold, seed, policy outcome, or candidate order was changed, and no
outcome-directed rerun was performed.

预先冻结的基础策略配对校准已经按从轻到重的顺序评估全部 `{len(attempts)}` 个候选，
没有候选同时进入对齐、可救援和成功率走廊。因此依据冻结协议停止：没有启动适应训练或
终评，也没有修改网格、阈值、seed 或根据结果重跑。

| Candidate | Gain | CBF-off success | CBF-on success | Alignment | Rescue |
| ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Evidence:

- [revision-7 precalibration protocol](precalibration_protocol_revision7.json)
- [terminal calibration summary](calibration/calibration_summary.json)
- [ordered attempts](calibration/attempts.json)
- [all evaluated calibration pairs](calibration/all_evaluated_paired_episodes.csv)
- [independent calibration evidence reconstruction](calibration/calibration_evidence_verification.json)
- [external artifact manifest](external_artifact_manifest.json)
- [execution orchestration provenance](orchestration/execution_orchestration.json)
- [zero-episode resource-guard migration receipt](orchestration/v25_revision7_guard2_migration_receipt.json)
{release_evidence_line}- [compact-package hashes](SHA256SUMS)
"""
(root / "README.md").write_text(text)
PY
  build_package_hashes
}

build_package_hashes() {
  python3 - "$RESULT_ROOT" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
lines = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.name == "SHA256SUMS":
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    lines.append(f"{digest.hexdigest()}  ./{path.relative_to(root)}\n")
(root / "SHA256SUMS").write_text("".join(lines))
PY
}

echo "$(timestamp) v25 completion orchestrator started"
write_parallel_resource_receipt
write_state 0

while queue_active; do
  latest=$(tail -n 1 "$EXTERNAL_ROOT/calibration_queue_revision7_guard2.log" 2>/dev/null || true)
  echo "$(timestamp) waiting for paired calibration queue: $latest"
  write_state 0
  sleep 300
done

exec 7>"$ORCHESTRATION_ROOT/calibration_execution.lock"
if ! flock -n 7; then
  fail "calibration execution lock remains held after queue exit"
fi

for _ in 1 2 3; do
  [[ -f "$CALIBRATION_ROOT/calibration_summary.json" ]] && break
  sleep 5
done
if [[ ! -f "$CALIBRATION_ROOT/calibration_summary.json" ]]; then
  CURRENT_PHASE=launch_unstarted_calibration_after_queue_exit
  write_state 0
  if [[ -f "$CALIBRATION_ROOT/calibration_execution_started.json" ]]; then
    fail "calibration started without a terminal summary; refusing to consume any calibration seed again"
  fi
  if [[ -d "$CALIBRATION_ROOT" ]] && \
     find "$CALIBRATION_ROOT" -type f -print -quit | grep -q .; then
    fail "calibration has files but no immutable start marker; refusing an ambiguous launch"
  fi
  require_file "$BASE_CHECKPOINT"
  require_file "$PRECALIBRATION_PROTOCOL"
  [[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_INITIAL_HEAD" ]] || fail \
    "refusing delayed calibration launch from a non-frozen repository HEAD"
  require_clean_repo
  [[ "$(sha256 "$BASE_CHECKPOINT")" == "$EXPECTED_BASE_SHA" ]] || fail \
    "base checkpoint hash changed before delayed calibration launch"
  [[ "$(sha256 "$PRECALIBRATION_PROTOCOL")" == "$EXPECTED_PROTOCOL_SHA" ]] || fail \
    "precalibration protocol hash changed before delayed calibration launch"
  export REPO PYTHON BASE_CHECKPOINT EXTERNAL_ROOT PRECALIBRATION_PROTOCOL
  # A queue can exit before simulation begins (for example, a transient
  # preflight failure).  With neither an immutable start marker nor any raw
  # file, one ordinary launch is still fresh.  Once the calibrator writes its
  # start marker, any interruption is terminal for automation: the fixed
  # identity schedule is never replayed under ``--resume``.
  require_storage_headroom
  wait_for_clean_gpu
  require_gpu_clear_now
  set +e
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  "$RUNNER" calibrate
  calibration_exit=$?
  set -e
  if [[ ! -f "$CALIBRATION_ROOT/calibration_summary.json" ]]; then
    fail "fresh calibration exited $calibration_exit without a terminal summary; start marker preserved and seed reuse refused"
  fi
fi
require_file "$CALIBRATION_ROOT/calibration_summary.json"

CURRENT_PHASE=package_calibration
write_state 0
require_file "$BASE_CHECKPOINT"
require_file "$PRECALIBRATION_PROTOCOL"
[[ "$(sha256 "$BASE_CHECKPOINT")" == "$EXPECTED_BASE_SHA" ]] || fail "base checkpoint hash changed"
[[ "$(sha256 "$PRECALIBRATION_PROTOCOL")" == "$EXPECTED_PROTOCOL_SHA" ]] || fail "precalibration protocol hash changed"
require_no_unrelated_repo_changes
if [[ ! -f "$RESULT_ROOT/calibration/calibration_summary.json" ]]; then
  [[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_INITIAL_HEAD" ]] || fail \
    "unexpected repository HEAD before calibration evidence commit"
fi
[[ -f "$RESULT_ROOT/calibration/calibration_summary.json" ]] || \
  copy_exact "$CALIBRATION_ROOT/calibration_summary.json" "$RESULT_ROOT/calibration/calibration_summary.json"
[[ -f "$RESULT_ROOT/calibration/attempts.json" ]] || \
  copy_exact "$CALIBRATION_ROOT/attempts.json" "$RESULT_ROOT/calibration/attempts.json"
[[ -f "$RESULT_ROOT/calibration/calibration_execution_started.json" ]] || \
  copy_exact "$CALIBRATION_ROOT/calibration_execution_started.json" \
    "$RESULT_ROOT/calibration/calibration_execution_started.json"

CALIBRATION_STATUS=$(python3 - "$CALIBRATION_ROOT/calibration_summary.json" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["status"])
PY
)

build_calibration_pair_evidence

if [[ "$CALIBRATION_STATUS" == no_candidate_qualified ]]; then
  CURRENT_PHASE=publish_no_candidate_result
  build_no_candidate_package
  commit_and_push "record terminal v25 calibration result" "$RESULT_ROOT"
  update_pr_comment no_candidate
  PIPELINE_STATUS=complete_no_candidate
  CURRENT_PHASE=complete
  exit 0
fi
[[ "$CALIBRATION_STATUS" == first_qualifying_candidate_frozen ]] || fail \
  "unexpected calibration status: $CALIBRATION_STATUS"

[[ -f "$RESULT_ROOT/calibration/context.json" ]] || \
  copy_exact "$CALIBRATION_ROOT/context.json" "$RESULT_ROOT/calibration/context.json"
[[ -f "$RESULT_ROOT/calibration/selected_paired_episodes.csv" ]] || \
  copy_exact "$CALIBRATION_ROOT/selected_paired_episodes.csv" \
    "$RESULT_ROOT/calibration/selected_paired_episodes.csv"
commit_and_push "record v25 paired calibration evidence" "$RESULT_ROOT/calibration"
require_no_unrelated_repo_changes

CURRENT_PHASE=freeze_formal_protocol
write_state 0
if [[ ! -f "$RESULT_ROOT/protocol.json" ]]; then
  "$PYTHON" "$REPO/experiments/scripts/freeze_cbf_teacher_v25_protocol.py" \
    --repo "$REPO" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --precalibration-protocol "$PRECALIBRATION_PROTOCOL" \
    --context "$RESULT_ROOT/calibration/context.json" \
    --calibration-started "$RESULT_ROOT/calibration/calibration_execution_started.json" \
    --calibration-summary "$RESULT_ROOT/calibration/calibration_summary.json" \
    --calibration-attempts "$RESULT_ROOT/calibration/attempts.json" \
    --calibration-paired-csv "$RESULT_ROOT/calibration/selected_paired_episodes.csv" \
    --calibration-all-paired-csv "$RESULT_ROOT/calibration/all_evaluated_paired_episodes.csv" \
    --calibration-evidence-verification "$RESULT_ROOT/calibration/calibration_evidence_verification.json" \
    --formal-output-dir "$TRAINING_ROOT" \
    --output "$RESULT_ROOT/protocol.json"
fi
commit_and_push "freeze v25 formal CBF-teacher protocol" "$RESULT_ROOT/protocol.json"
require_no_unrelated_repo_changes

CURRENT_PHASE=run_fixed_eight_rounds
write_state 0
if [[ ! -f "$TRAINING_ROOT/formal_execution_completed.json" ]]; then
  if [[ -f "$TRAINING_ROOT/formal_execution_started.json" ]]; then
    fail "formal adaptation previously started but is incomplete; refusing to consume the seed again"
  fi
  require_clean_repo
  require_storage_headroom
  wait_for_clean_gpu
  require_gpu_clear_now
  export REPO PYTHON BASE_CHECKPOINT EXTERNAL_ROOT PRECALIBRATION_PROTOCOL
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES=0 \
  "$RUNNER" train
fi
require_file "$TRAINING_ROOT/training_summary.json"
require_file "$TRAINING_ROOT/final_round_08.pt"

CURRENT_PHASE=run_four_condition_audit
write_state 0
if [[ ! -f "$FINAL_ROOT/final_test.json" ]]; then
  require_clean_repo
  require_storage_headroom
  wait_for_clean_gpu
  require_gpu_clear_now
  export REPO PYTHON BASE_CHECKPOINT EXTERNAL_ROOT PRECALIBRATION_PROTOCOL
  while [[ ! -f "$FINAL_ROOT/final_test.json" ]]; do
    if [[ -f "$FINAL_ROOT/final_evaluation_started.json" ]]; then
      validate_final_resume_evidence
      # A complete arm can exist between its atomic CSV/JSON rename and the
      # completion-ledger rename.  After raw evidence validation, reconcile at
      # most that one pending arm; no-pending is an idempotent no-op.  Partial
      # output or any mismatch remains fail-closed.
      record_completed_final_batch
      validate_final_resume_evidence
    fi
    require_gpu_clear_now
    set +e
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    run_final_audit_with_atomic_batch_ledger
    audit_exit=$?
    set -e
    if (( audit_exit == 75 )); then
      # The evaluator atomically wrote one 128-episode arm.  Independently
      # reconstruct it before recording the immutable completion ledger and
      # allowing the next frozen seed/condition batch to launch.
      validate_final_resume_evidence
      record_completed_final_batch
      validate_final_resume_evidence
      continue
    fi
    (( audit_exit == 0 )) || fail \
      "final audit exited $audit_exit; execution ledger preserved and automatic seed reuse refused"
  done
fi
require_file "$FINAL_ROOT/final_test.json"
require_file "$FINAL_ROOT/paired_episode_metrics.csv"

CURRENT_PHASE=build_compact_evidence
write_state 0
require_no_unrelated_repo_changes
for name in formal_execution_started.json formal_execution_completed.json round_metrics.json round_metrics.csv training_summary.json; do
  copy_exact "$TRAINING_ROOT/$name" "$RESULT_ROOT/training/$name"
done
for name in final_evaluation_started.json final_test.json paired_episode_metrics.csv; do
  copy_exact "$FINAL_ROOT/$name" "$RESULT_ROOT/final/$name"
done
while IFS= read -r ledger_file; do
  relative=${ledger_file#"$FINAL_ROOT/"}
  copy_exact "$ledger_file" "$RESULT_ROOT/final/$relative"
done < <(find "$FINAL_ROOT/execution_ledger" -type f -name '*.json' -print | sort)

set +e
"$PYTHON" "$REPO/experiments/scripts/verify_cbf_teacher_v25.py" \
  --repo "$REPO" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --final-checkpoint "$TRAINING_ROOT/final_round_08.pt" \
  --precalibration-protocol "$PRECALIBRATION_PROTOCOL" \
  --protocol "$RESULT_ROOT/protocol.json" \
  --context "$RESULT_ROOT/calibration/context.json" \
  --calibration-started "$RESULT_ROOT/calibration/calibration_execution_started.json" \
  --calibration-summary "$RESULT_ROOT/calibration/calibration_summary.json" \
  --calibration-attempts "$RESULT_ROOT/calibration/attempts.json" \
  --calibration-paired-csv "$RESULT_ROOT/calibration/selected_paired_episodes.csv" \
  --calibration-all-paired-csv "$RESULT_ROOT/calibration/all_evaluated_paired_episodes.csv" \
  --calibration-evidence-verification "$RESULT_ROOT/calibration/calibration_evidence_verification.json" \
  --training-summary "$RESULT_ROOT/training/training_summary.json" \
  --training-started "$RESULT_ROOT/training/formal_execution_started.json" \
  --training-completion "$RESULT_ROOT/training/formal_execution_completed.json" \
  --final-evaluation-started "$RESULT_ROOT/final/final_evaluation_started.json" \
  --final-test "$RESULT_ROOT/final/final_test.json" \
  --paired-csv "$RESULT_ROOT/final/paired_episode_metrics.csv" \
  --output "$RESULT_ROOT/final/verification.json"
VERIFY_EXIT=$?
set -e
echo "$(timestamp) independent verifier exit code=$VERIFY_EXIT (published verbatim)"
VERIFY_PASSED=$(python3 - "$RESULT_ROOT/final/verification.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
print("true" if json.loads(path.read_text()).get("passed") is True else "false")
PY
)
if (( VERIFY_EXIT == 0 )) && [[ "$VERIFY_PASSED" != true ]]; then
  fail "verifier returned success but verification.json is not passed"
fi

if [[ ! -f "$RESULT_ROOT/figures/figure_manifest.json" ]]; then
  MPLBACKEND=Agg "$PYTHON" "$REPO/experiments/scripts/plot_cbf_teacher_v25.py" \
    --training-summary "$RESULT_ROOT/training/training_summary.json" \
    --final-test "$RESULT_ROOT/final/final_test.json" \
    --output-dir "$RESULT_ROOT/figures"
fi
copy_orchestration_sources
build_external_manifest
build_final_readme
build_package_hashes

CURRENT_PHASE=publish_final_result
write_state 0
commit_and_push "publish complete v25 CBF-teacher result" "$RESULT_ROOT"
update_pr_comment final

if [[ "$VERIFY_PASSED" != true ]]; then
  PIPELINE_STATUS=terminal_verification_failed
  CURRENT_PHASE=complete
  echo "$(timestamp) v25 evidence published, but independent verification failed"
  exit 92
fi
PIPELINE_STATUS=complete
CURRENT_PHASE=complete
echo "$(timestamp) v25 formal pipeline and GitHub publication complete"
