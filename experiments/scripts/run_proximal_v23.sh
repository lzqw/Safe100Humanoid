#!/usr/bin/env bash
set -euo pipefail

v23_mode=${1:-}
v23_repo=${V23_REPO:-/home/carla/LZQW/SAFE100/humanoid/worktrees/v23_proximal}
v23_python=${V23_PYTHON:-/home/carla/LZQW/SAFE100/humanoid/workspace/conda_env/bin/python}
v23_base=${V23_BASE_CHECKPOINT:-/home/carla/LZQW/SAFE100/humanoid/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}
v23_context=${V23_CONTEXT:-${v23_repo}/results/online/specialist_v22/calibration/L_effect/context.json}
v23_protocol=${V23_PROTOCOL:-${v23_repo}/results/online/proximal_v23/protocol.json}
v23_artifacts=${V23_ARTIFACTS:-/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v23/lateral}

case "${v23_mode}" in
  train)
    "${v23_python}" "${v23_repo}/experiments/scripts/refine_proximal_v23.py" \
      --repo "${v23_repo}" \
      --base-checkpoint "${v23_base}" \
      --context "${v23_context}" \
      --protocol "${v23_protocol}" \
      --output-dir "${v23_artifacts}/training"
    ;;
  audit)
    "${v23_python}" "${v23_repo}/experiments/scripts/audit_proximal_v23.py" \
      --repo "${v23_repo}" \
      --base-checkpoint "${v23_base}" \
      --final-checkpoint "${v23_artifacts}/training/final_round_08.pt" \
      --training-summary "${v23_artifacts}/training/training_summary.json" \
      --context "${v23_context}" \
      --protocol "${v23_protocol}" \
      --output-dir "${v23_artifacts}/final"
    ;;
  plot)
    "${v23_python}" "${v23_repo}/experiments/scripts/plot_proximal_v23.py" \
      --training-summary "${v23_artifacts}/training/training_summary.json" \
      --final-test "${v23_artifacts}/final/final_test.json" \
      --output-dir "${v23_artifacts}/figures"
    ;;
  *)
    echo "usage: $0 {train|audit|plot}" >&2
    exit 2
    ;;
esac
