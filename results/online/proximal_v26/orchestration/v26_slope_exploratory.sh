#!/usr/bin/env bash
set -euo pipefail

REPO=/home/carla/LZQW/SAFE100/humanoid/worktrees/v23_proximal
PYTHON=/home/carla/LZQW/SAFE100/humanoid/workspace/conda_env/bin/python
CHECKPOINT=/home/carla/LZQW/SAFE100/humanoid/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt
OUT=/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v26_higher_riser/slope_exploratory
HEIGHT=0.18
SEED=146261000
EPISODES=64
SLOPES=(0.50 0.65 0.80 1.00)

mkdir -p "$OUT"

wait_for_gpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( free >= 3200 )); then
      printf '{"event":"gpu_ready","free_vram_mib":%s}\n' "$free"
      return
    fi
    printf '{"event":"waiting_for_gpu","free_vram_mib":%s,"required_vram_mib":3200}\n' "$free"
    sleep 20
  done
}

run_arm() {
  local filter=$1
  local slope=$2
  local target=$3
  mkdir -p "$target"
  wait_for_gpu
  "$PYTHON" "$REPO/experiments/scripts/evaluate_cbf_teacher_v26.py" \
    --repo "$REPO" \
    --checkpoint "$CHECKPOINT" \
    --riser-height "$HEIGHT" \
    --clearance-slope "$slope" \
    --runtime-filter "$filter" \
    --num-envs "$EPISODES" \
    --num-episodes "$EPISODES" \
    --seed "$SEED" \
    --device cuda:0 \
    --output-json "$target/summary.json" \
    --output-csv "$target/episodes.csv"
}

if [[ ! -s "$OUT/off/episodes.csv" ]]; then
  run_arm off 0.0 "$OUT/off"
fi

for slope in "${SLOPES[@]}"; do
  target="$OUT/slope_${slope}"
  if [[ ! -s "$target/episodes.csv" ]]; then
    printf '{"event":"starting_on_arm","slope":%s}\n' "$slope"
    run_arm on "$slope" "$target"
  fi
done

"$PYTHON" - "$OUT" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
slopes = (0.50, 0.65, 0.80, 1.00)

def load(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def truth(value):
    return value.lower() == "true"

off = load(root / "off/episodes.csv")
attempts = []
for slope in slopes:
    on = load(root / f"slope_{slope:.2f}/episodes.csv")
    if len(off) != 64 or len(on) != 64:
        raise RuntimeError("incomplete v26 slope exploratory arm")
    off_success = [truth(row["success"]) for row in off]
    on_success = [truth(row["success"]) for row in on]
    off_kick = [truth(row["toe_riser_kick"]) for row in off]
    failures = sum(not value for value in off_success)
    aligned = sum(
        (not success) and kick
        for success, kick in zip(off_success, off_kick, strict=True)
    )
    rescued = sum(
        (not before) and after
        for before, after in zip(off_success, on_success, strict=True)
    )
    regressed = sum(
        before and (not after)
        for before, after in zip(off_success, on_success, strict=True)
    )
    attempt = {
        "development_only": True,
        "excluded_from_formal_v26": True,
        "riser_height_m": 0.18,
        "clearance_barrier_slope": slope,
        "evaluation_seed": 146261000,
        "paired_count": 64,
        "off_success_rate": sum(off_success) / 64,
        "on_success_rate": sum(on_success) / 64,
        "alignment_coverage": aligned / max(1, failures),
        "shield_rescue_rate": rescued / max(1, failures),
        "rescued_count": rescued,
        "regressed_count": regressed,
    }
    attempts.append(attempt)
    print(json.dumps(attempt, sort_keys=True), flush=True)

payload = {
    "schema_version": 1,
    "status": "complete",
    "development_only": True,
    "excluded_from_formal_v26": True,
    "shared_off_arm": True,
    "attempts": attempts,
}
temporary = root / ".slope_exploratory_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "slope_exploratory_summary.json")
PY
