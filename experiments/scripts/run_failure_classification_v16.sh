#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAFE100_HUMANOID_ROOT:-/home/carla/LZQW/SAFE100/humanoid}"
REPO="${SAFE100_MJLAB_REPO:-$ROOT/third_party/unitree_rl_mjlab}"
PYTHON="${SAFE100_PYTHON:-$ROOT/workspace/conda_env/bin/python}"
SOURCE="${SAFE100_V15_AUDIT_ROOT:-$ROOT/artifacts/failure_focused_v15/final_audit}"
OUTPUT="${SAFE100_CLASSIFICATION_OUTPUT_DIR:-$ROOT/artifacts/dominant_failure_v16/failure_classification}"

mkdir -p "$OUTPUT"
cd "$REPO"

"$PYTHON" experiments/scripts/classify_failure_modes_v16.py \
  --repo "$REPO" \
  --input-root "$SOURCE" \
  --source-audit-summary "$SOURCE/final_audit_summary.json" \
  --output-json "$OUTPUT/failure_classification.json" \
  --output-csv "$OUTPUT/classified_baseline_falls.csv" \
  --expected-episodes 1536
