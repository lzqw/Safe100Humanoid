#!/usr/bin/env bash
set -euo pipefail

v24_mode=${1:-}
v24_repo=${V24_REPO:-/home/carla/LZQW/SAFE100/humanoid/worktrees/v23_proximal}
v24_python=${V24_PYTHON:-/home/carla/LZQW/SAFE100/humanoid/workspace/conda_env/bin/python}
v24_base=${V24_BASE_CHECKPOINT:-/home/carla/LZQW/SAFE100/humanoid/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}
v24_artifacts=${V24_ARTIFACTS:-/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v24/contact}
v24_precal_protocol=${V24_PRECAL_PROTOCOL:-${v24_repo}/results/online/proximal_v24/precalibration_protocol.json}
v24_context=${V24_CONTEXT:-${v24_repo}/results/online/proximal_v24/calibration/context.json}
v24_calibration_summary=${V24_CALIBRATION_SUMMARY:-${v24_repo}/results/online/proximal_v24/calibration/calibration_summary.json}
v24_protocol=${V24_PROTOCOL:-${v24_repo}/results/online/proximal_v24/protocol.json}
v24_protocol_commit=${V24_PROTOCOL_COMMIT:-$(git -C "${v24_repo}" rev-parse HEAD)}

case "${v24_mode}" in
  freeze-precal)
    "${v24_python}" "${v24_repo}/experiments/scripts/freeze_proximal_v24_precalibration.py" \
      --repo "${v24_repo}" \
      --base-checkpoint "${v24_base}" \
      --output "${v24_precal_protocol}"
    ;;
  calibrate)
    "${v24_python}" "${v24_repo}/experiments/scripts/calibrate_proximal_v24.py" \
      --repo "${v24_repo}" \
      --base-checkpoint "${v24_base}" \
      --precalibration-protocol "${v24_precal_protocol}" \
      --protocol-commit "${v24_protocol_commit}" \
      --output-dir "${v24_artifacts}/calibration" \
      --context-output "${v24_artifacts}/calibration/context.json"
    ;;
  freeze-formal)
    "${v24_python}" "${v24_repo}/experiments/scripts/freeze_proximal_v24_protocol.py" \
      --repo "${v24_repo}" \
      --base-checkpoint "${v24_base}" \
      --precalibration-protocol "${v24_precal_protocol}" \
      --context "${v24_context}" \
      --calibration-summary "${v24_calibration_summary}" \
      --formal-output-dir "${v24_artifacts}/training" \
      --output "${v24_protocol}"
    ;;
  train)
    "${v24_python}" "${v24_repo}/experiments/scripts/refine_proximal_v24.py" \
      --repo "${v24_repo}" \
      --base-checkpoint "${v24_base}" \
      --context "${v24_context}" \
      --protocol "${v24_protocol}" \
      --output-dir "${v24_artifacts}/training"
    ;;
  audit)
    "${v24_python}" "${v24_repo}/experiments/scripts/audit_proximal_v24.py" \
      --repo "${v24_repo}" \
      --base-checkpoint "${v24_base}" \
      --final-checkpoint "${v24_artifacts}/training/final_round_08.pt" \
      --training-summary "${v24_artifacts}/training/training_summary.json" \
      --context "${v24_context}" \
      --protocol "${v24_protocol}" \
      --output-dir "${v24_artifacts}/final"
    ;;
  plot)
    "${v24_python}" "${v24_repo}/experiments/scripts/plot_proximal_v24.py" \
      --training-summary "${v24_artifacts}/training/training_summary.json" \
      --final-test "${v24_artifacts}/final/final_test.json" \
      --output-dir "${v24_artifacts}/figures"
    ;;
  verify)
    "${v24_python}" "${v24_repo}/experiments/scripts/verify_proximal_v24.py" \
      --protocol "${v24_protocol}" \
      --context "${v24_context}" \
      --training-summary "${v24_artifacts}/training/training_summary.json" \
      --final-test "${v24_artifacts}/final/final_test.json" \
      --paired-csv "${v24_artifacts}/final/paired_episode_metrics.csv" \
      --v23-protocol "${v24_repo}/results/online/proximal_v23/protocol.json" \
      --v23-final-test "${v24_repo}/results/online/proximal_v23/final/final_test.json" \
      --output "${v24_artifacts}/final/verification.json"
    ;;
  combine)
    "${v24_python}" "${v24_repo}/experiments/scripts/build_proximal_completion.py" \
      --v23-final-test "${v24_repo}/results/online/proximal_v23/final/final_test.json" \
      --v24-final-test "${v24_artifacts}/final/final_test.json" \
      --output-dir "${v24_artifacts}/completion"
    ;;
  *)
    echo "usage: $0 {freeze-precal|calibrate|freeze-formal|train|audit|plot|verify|combine}" >&2
    exit 2
    ;;
esac
