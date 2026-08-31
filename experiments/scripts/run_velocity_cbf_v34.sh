#!/usr/bin/env bash
set -euo pipefail

REPO=${V34_REPO:-/home/carla/LZQW/SAFE100/humanoid/worktrees/v23_proximal}
PYTHON=${V34_PYTHON:-/home/carla/LZQW/SAFE100/humanoid/workspace/conda_env/bin/python}
BASE=${V34_BASE:-/home/carla/LZQW/SAFE100/humanoid/artifacts/retention_v13/arm_b_state_retention/accepted_final.pt}
V31=${V34_V31:-/home/carla/LZQW/SAFE100/humanoid/artifacts/proximal_v31/formal}
ARTIFACT=${V34_ARTIFACT:-/home/carla/LZQW/SAFE100/humanoid/artifacts/velocity_cbf_v34}
RESULT=${V34_RESULT:-${REPO}/results/online/velocity_cbf_v34}
DEVICE=${V34_DEVICE:-cuda:0}
PHASE=${1:-development}

CONFIG=${RESULT}/search_config.json
SMOKE=${ARTIFACT}/smoke_summary.json
SEARCH=${ARTIFACT}/search
TRAINING=${ARTIFACT}/training
SELECTION=${ARTIFACT}/selection
SELECTED=${RESULT}/search/selected_cbf.json
FINAL=${ARTIFACT}/final

case "${PHASE}" in
  freeze-search)
    "${PYTHON}" "${REPO}/experiments/scripts/freeze_velocity_cbf_v34.py" \
      --repo "${REPO}" --base-checkpoint "${BASE}" --v31-root "${V31}" \
      --output "${CONFIG}"
    ;;
  smoke)
    "${PYTHON}" "${REPO}/experiments/scripts/smoke_velocity_cbf_v34.py" \
      --repo "${REPO}" --search-config "${CONFIG}" \
      --checkpoint "${V31}/F1/A2/round_08.pt" --output "${SMOKE}" \
      --device "${DEVICE}"
    ;;
  search)
    resume=()
    if [[ -d "${SEARCH}" ]]; then resume=(--resume); fi
    "${PYTHON}" "${REPO}/experiments/scripts/optimize_velocity_cbf_v34.py" \
      --repo "${REPO}" --search-config "${CONFIG}" --v31-root "${V31}" \
      --output-root "${SEARCH}" --device "${DEVICE}" "${resume[@]}"
    ;;
  train)
    mapfile -t candidates < <("${PYTHON}" -c \
      'import json,sys; print(*[x["candidate"] for x in json.load(open(sys.argv[1]))["top2"]], sep="\n")' \
      "${SEARCH}/top2_candidates.json")
    for candidate in "${candidates[@]}"; do
      for context in F1 F2 F3; do
        directory=${TRAINING}/${candidate}/${context}
        if [[ -f "${directory}/execution_completed.json" ]]; then continue; fi
        resume=()
        if [[ -d "${directory}" ]]; then resume=(--resume); fi
        "${PYTHON}" "${REPO}/experiments/scripts/refine_velocity_cbf_v34.py" \
          --repo "${REPO}" --search-config "${CONFIG}" \
          --top2 "${SEARCH}/top2_candidates.json" --base-checkpoint "${BASE}" \
          --candidate "${candidate}" --context "${context}" \
          --output-dir "${directory}" --device "${DEVICE}" "${resume[@]}"
      done
    done
    ;;
  select)
    resume=()
    if [[ -d "${SELECTION}" ]]; then resume=(--resume); fi
    "${PYTHON}" "${REPO}/experiments/scripts/select_velocity_cbf_v34.py" \
      --repo "${REPO}" --search-config "${CONFIG}" \
      --top2 "${SEARCH}/top2_candidates.json" --training-root "${TRAINING}" \
      --output-root "${SELECTION}" --device "${DEVICE}" "${resume[@]}"
    ;;
  development)
    if [[ ! -f "${SMOKE}" ]]; then "$0" smoke; fi
    "$0" search
    "$0" train
    "$0" select
    ;;
  freeze-final)
    "${PYTHON}" "${REPO}/experiments/scripts/freeze_velocity_cbf_v34_final.py" \
      --repo "${REPO}" --search-config "${CONFIG}" \
      --development-selection "${SELECTION}/development_selection.json" \
      --top8-results "${SEARCH}/top8_results.csv" --output "${SELECTED}"
    ;;
  final)
    resume=()
    if [[ -f "${FINAL}/execution_started.json" ]]; then resume=(--resume); fi
    "${PYTHON}" "${REPO}/experiments/scripts/audit_velocity_cbf_v34.py" \
      --repo "${REPO}" --search-config "${CONFIG}" --selected "${SELECTED}" \
      --base-checkpoint "${BASE}" --v31-root "${V31}" --output-root "${FINAL}" \
      --device "${DEVICE}" "${resume[@]}"
    ;;
  package)
    "${PYTHON}" "${REPO}/experiments/scripts/package_velocity_cbf_v34.py" \
      --repo "${REPO}" --search-config "${CONFIG}" --selected "${SELECTED}" \
      --smoke "${SMOKE}" --search-root "${SEARCH}" --selection-root "${SELECTION}" \
      --training-root "${TRAINING}" --final-root "${FINAL}" --output-dir "${RESULT}"
    ;;
  *)
    echo "usage: $0 {freeze-search|smoke|search|train|select|development|freeze-final|final|package}" >&2
    exit 2
    ;;
esac
