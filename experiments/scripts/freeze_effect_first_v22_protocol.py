"""Build v22 protocols at precalibration and per-context adaptation boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from specialist_v22_protocol import (
  CALIBRATION_EPISODES,
  CALIBRATION_MINIMUM_FAILURES,
  CALIBRATION_MINIMUM_PURITY,
  CALIBRATION_SUCCESS_BOUNDS,
  CANDIDATE_CONFIRM_EPISODES,
  CANDIDATE_D0_EPISODES,
  CANDIDATE_FRACTIONS,
  CANDIDATE_SCREEN_EPISODES,
  CONTEXT_ADAPTATION_SEEDS,
  CONTEXT_CALIBRATION_CANDIDATE_SEEDS,
  CONTEXT_CALIBRATION_EVALUATION_SEEDS,
  CONTEXT_FINAL_D0_SEEDS,
  CONTEXT_FINAL_TARGET_SEEDS,
  CONTEXT_REPORT_BOOTSTRAP_SEEDS,
  CONTEXT_VALIDATION_SEEDS,
  CONTEXTS,
  DEVELOPMENT_MAXIMUM_FALL_DELTA,
  DEVELOPMENT_MINIMUM_SUCCESS_DELTA,
  D0_MINIMUM_SUCCESS_DELTA,
  DUAL_ROLLOUT_BATCHES,
  EVAL_BATCH_SIZE,
  FAILURE_DISCOVERY_MAX_ROLLOUTS,
  FINAL_D0_EPISODES,
  FINAL_TARGET_EPISODES,
  MODES,
  NORMAL_FAILURE_SUCCESS_SLOTS,
  NUM_ENVS,
  POLICY_METHOD,
  PROTOCOL_ID,
  REPORT_BOOTSTRAP_SAMPLES,
  ROLLOUT_STEPS,
  ROUNDS,
  VALIDATION_EPISODES,
  VALIDATION_MAXIMUM_FALL_DELTA,
  canonical_sha256,
  fresh_randomness_report,
)
from src.tasks.stairs_cbf.deployment_context import (
  V22_CONTEXT_SCHEMA_VERSION,
  V22_CONTEXT_SPECS,
  load_calibrated_v22_context,
)

SOURCE_FILES = (
  "docs/EFFECT_FIRST_V22_PROTOCOL.md",
  "src/tasks/velocity/mdp/observations.py",
  "src/tasks/stairs_cbf/actions.py",
  "src/tasks/stairs_cbf/command.py",
  "src/tasks/stairs_cbf/config.py",
  "src/tasks/stairs_cbf/deployment_context.py",
  "src/tasks/stairs_cbf/hard_cases.py",
  "src/tasks/stairs_cbf/mdp.py",
  "src/tasks/stairs_cbf/online.py",
  "experiments/scripts/evaluate_online_stairs.py",
  "experiments/scripts/online_refine_stairs.py",
  "experiments/scripts/refine_deployment_v21.py",
  "experiments/scripts/specialist_v21_protocol.py",
  "experiments/scripts/specialist_v22_protocol.py",
  "experiments/scripts/calibrate_effect_first_v22.py",
  "experiments/scripts/refine_effect_first_v22.py",
  "experiments/scripts/test_effect_first_v22.py",
  "experiments/scripts/plot_effect_first_v22.py",
  "experiments/scripts/freeze_effect_first_v22_protocol.py",
  "experiments/scripts/run_effect_first_v22.sh",
  "experiments/tests/test_specialist_v22.py",
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _git_output(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
  rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.exists() and path.read_text() != rendered:
    raise RuntimeError(f"refusing to overwrite a different v22 protocol: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(rendered)


def _source_hashes(repo: Path) -> dict[str, str]:
  missing = [relative for relative in SOURCE_FILES if not (repo / relative).is_file()]
  if missing:
    raise FileNotFoundError(f"v22 prospective source files are missing: {missing}")
  return {relative: _sha256(repo / relative) for relative in SOURCE_FILES}


def _verify_git_blob(repo: Path, path: Path, commit: str) -> dict[str, str]:
  relative = str(path.resolve().relative_to(repo))
  blob = subprocess.run(
    ["git", "show", f"{commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  digest = _sha256(path)
  if hashlib.sha256(blob).hexdigest() != digest:
    raise RuntimeError(f"v22 artifact differs from commit {commit}: {relative}")
  return {"file": relative, "sha256": digest, "git_commit": commit}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--stage", choices=("precalibration", "adaptation"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--base-checkpoint-reference", required=True)
  parser.add_argument("--base-checkpoint-sha256", required=True)
  parser.add_argument("--context-id", choices=CONTEXTS)
  parser.add_argument("--context", type=Path)
  parser.add_argument("--precalibration-protocol", type=Path)
  parser.add_argument("--precalibration-commit")
  parser.add_argument("--lateral-final-result", type=Path)
  parser.add_argument("--superseded-protocol", type=Path)
  parser.add_argument("--superseded-commit")
  parser.add_argument("--supersession-reason")
  parser.add_argument(
    "--superseded-before-any-base-evaluation", action="store_true"
  )
  return parser.parse_args()


def _common_payload(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  randomness = fresh_randomness_report(repo)
  if not randomness["passed"]:
    raise RuntimeError(f"v22 randomness collision: {randomness['collisions']}")
  specs = {context_id: V22_CONTEXT_SPECS[context_id] for context_id in CONTEXTS}
  return {
    "schema_version": 1,
    "context_schema_version": V22_CONTEXT_SCHEMA_VERSION,
    "protocol_id": PROTOCOL_ID,
    "policy_method": POLICY_METHOD,
    "experiment_class": "development/effect-first, not a formal generalization audit",
    "experiment_unit": "one frozen context, one pi0, one adaptation, one fresh paired test",
    "context_order": ["L_effect", "C_effect"],
    "context_specifications": specs,
    "context_specifications_sha256": canonical_sha256(specs),
    "adaptation_seeds": CONTEXT_ADAPTATION_SEEDS,
    "randomness_preflight": randomness,
    "calibration": {
      "base_policy_only": True,
      "adapted_policy_evaluations_used": False,
      "first_qualifying_candidate_is_frozen": True,
      "success_rate_bounds_inclusive": list(CALIBRATION_SUCCESS_BOUNDS),
      "minimum_failure_count": CALIBRATION_MINIMUM_FAILURES,
      "minimum_target_failure_fraction": CALIBRATION_MINIMUM_PURITY,
      "episodes_per_candidate": CALIBRATION_EPISODES,
      "eval_batch_size": EVAL_BATCH_SIZE,
      "candidate_seeds": {
        key: list(value)
        for key, value in CONTEXT_CALIBRATION_CANDIDATE_SEEDS.items()
      },
      "evaluation_seed_bases": CONTEXT_CALIBRATION_EVALUATION_SEEDS,
    },
    "training": {
      "matched_success_beta": 0.0,
      "control_or_parallel_comparison_branch": False,
      "one_actor": True,
      "one_privileged_critic": True,
      "new_risk_or_cost_head": False,
      "runtime_cbf": True,
      "ppo_stores_raw_policy_action": True,
      "fixed_round_budget": ROUNDS,
      "num_envs": NUM_ENVS,
      "rollout_steps_per_environment": ROLLOUT_STEPS,
      "dual_rollout_batches": DUAL_ROLLOUT_BATCHES,
      "normal_failure_success_slots": list(NORMAL_FAILURE_SUCCESS_SLOTS),
      "failure_discovery_max_rollouts": FAILURE_DISCOVERY_MAX_ROLLOUTS,
      "candidate_fractions": list(CANDIDATE_FRACTIONS),
      "screening_paired_episodes_per_candidate": CANDIDATE_SCREEN_EPISODES,
      "single_fresh_confirmation_paired_episodes": CANDIDATE_CONFIRM_EPISODES,
      "confirmation_success_delta_strictly_positive": True,
      "confirmation_maximum_fall_delta": 0.03,
      "confirmation_ci_or_block_gate": False,
      "candidate_d0_paired_episodes": CANDIDATE_D0_EPISODES,
      "candidate_d0_minimum_success_delta": D0_MINIMUM_SUCCESS_DELTA,
    },
    "validation_monitor": {
      "paired_conditions": VALIDATION_EPISODES,
      "seeds": CONTEXT_VALIDATION_SEEDS,
      "excluded_from_ppo_replay_failure_bank_and_candidate_screen": True,
      "evaluate_pi0_and_every_d0_safe_accepted_checkpoint": True,
      "selection_primary": "highest target success rate",
      "maximum_fall_increase_from_pi0": VALIDATION_MAXIMUM_FALL_DELTA,
      "tie_breaks": ["lower fall rate", "earlier round"],
      "final_deployment_uses_best_so_far_not_last": True,
    },
    "final_test": {
      "target_paired_episodes": FINAL_TARGET_EPISODES,
      "d0_paired_episodes": FINAL_D0_EPISODES,
      "eval_batch_size": EVAL_BATCH_SIZE,
      "target_seeds": CONTEXT_FINAL_TARGET_SEEDS,
      "d0_seeds": CONTEXT_FINAL_D0_SEEDS,
      "base_and_best_receive_same_initial_conditions": True,
      "repair_and_regression_reported": True,
      "report_only_bootstrap_samples": REPORT_BOOTSTRAP_SAMPLES,
      "report_only_bootstrap_seeds": CONTEXT_REPORT_BOOTSTRAP_SEEDS,
      "confidence_interval_is_gate": False,
    },
    "development_gate": {
      "minimum_target_success_delta": DEVELOPMENT_MINIMUM_SUCCESS_DELTA,
      "maximum_target_fall_delta": DEVELOPMENT_MAXIMUM_FALL_DELTA,
      "minimum_d0_success_delta": D0_MINIMUM_SUCCESS_DELTA,
    },
    "conditional_execution": {
      "lateral_runs_first": True,
      "contact_calibration_and_adaptation_require_lateral_final_gate": True,
      "if_lateral_fails_stop_and_preserve_negative_result": True,
      "additional_context_repeats_are_out_of_scope_until_both_modes_pass": True,
    },
    "figures": [
      "validation success versus accepted round",
      "base versus best final success and fall",
      "repair versus regression",
      "failure-specific same-rollout telemetry",
    ],
    "excluded": [
      "multi-context bootstrap",
      "off-diagonal evaluation",
      "macro gate",
      "control/v22 dual branches",
      "nonzero matched-success KL",
    ],
    "retry_policy": {
      "poor_outcome_may_be_rerun": False,
      "infrastructure_retry_only": True,
      "identical_context_seed_commit_checkpoint_required": True,
      "retry_provenance_required": True,
    },
    "sealed_inputs": {
      "source_creation_parent_commit": current_commit,
      "source_files": list(SOURCE_FILES),
      "source_file_sha256": _source_hashes(repo),
      "base_policy_checkpoint_reference": args.base_checkpoint_reference,
      "base_policy_checkpoint_sha256": args.base_checkpoint_sha256,
    },
  }


def _precalibration(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
  payload = _common_payload(repo, args)
  superseded = None
  supplied = (
    args.superseded_protocol is not None,
    args.superseded_commit is not None,
    args.supersession_reason is not None,
  )
  if len(set(supplied)) != 1:
    raise ValueError(
      "v22 superseded protocol, commit, and reason must be supplied together"
    )
  if args.superseded_protocol is not None:
    if not args.superseded_before_any_base_evaluation:
      raise ValueError("v22 supersession must attest that no base evaluation began")
    assert args.superseded_commit is not None
    superseded_path = args.superseded_protocol.resolve()
    prior = json.loads(superseded_path.read_text())
    binding = _verify_git_blob(repo, superseded_path, args.superseded_commit)
    if (
      prior.get("protocol_id") != PROTOCOL_ID
      or prior.get("protocol_revision") != 0
      or prior.get("status")
      != "prospectively_frozen_before_base_only_calibration"
      or prior.get("fresh_evidence_boundary", {}).get(
        "base_only_calibration_outcomes_seen"
      )
      is not False
    ):
      raise RuntimeError("invalid v22 superseded precalibration protocol")
    superseded = {
      "protocol": binding,
      "disposition": "superseded_before_first_base_policy_episode",
      "reason": args.supersession_reason,
      "calibration_process_started": False,
      "base_policy_episode_outcomes_observed": False,
      "adapted_policy_outcomes_observed": False,
      "external_calibration_artifacts_created": False,
      "waiting_queue_terminated_before_gpu_execution": True,
      "historical_v17_v21_results_modified": False,
    }
  payload.update(
    protocol_revision=0,
    precalibration_boundary_revision=1 if superseded is not None else 0,
    status="prospectively_frozen_before_base_only_calibration",
    fresh_evidence_boundary={
      "base_only_calibration_outcomes_seen": False,
      "adaptation_outcomes_seen": False,
      "final_test_outcomes_seen": False,
      "protocol_must_be_committed_before_calibration": True,
    },
  )
  if superseded is not None:
    payload["superseded_precalibration_boundary"] = superseded
  return payload


def _adaptation(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
  if not all(
    (
      args.context_id,
      args.context,
      args.precalibration_protocol,
      args.precalibration_commit,
    )
  ):
    raise ValueError("v22 adaptation freeze requires context and precalibration inputs")
  assert args.context_id is not None
  assert args.context is not None
  assert args.precalibration_protocol is not None
  assert args.precalibration_commit is not None
  payload = _common_payload(repo, args)
  context_path = args.context.resolve()
  context = load_calibrated_v22_context(context_path)
  if context["context_id"] != args.context_id:
    raise RuntimeError("v22 adaptation context ID differs")
  precal_path = args.precalibration_protocol.resolve()
  precal = json.loads(precal_path.read_text())
  precal_binding = _verify_git_blob(
    repo, precal_path, args.precalibration_commit
  )
  if (
    precal.get("protocol_id") != PROTOCOL_ID
    or precal.get("protocol_revision") != 0
    or precal.get("status")
    != "prospectively_frozen_before_base_only_calibration"
    or context["calibration"].get("prospective_protocol_file_sha256")
    != precal_binding["sha256"]
    or context["calibration"].get("prospective_protocol_git_commit")
    != args.precalibration_commit
  ):
    raise RuntimeError("v22 context is not bound to the frozen calibration protocol")
  context_relative = str(context_path.relative_to(repo))
  payload["sealed_inputs"]["contexts"] = {
    args.context_id: {
      "file": context_relative,
      "file_sha256": _sha256(context_path),
      "parameters_sha256": context["parameters_sha256"],
      "selected_calibration_seed": context["calibration"][
        "selected_candidate_seed"
      ],
      "selected_base_success_rate": context["calibration"]["attempts"][-1][
        "success_rate"
      ],
      "selected_failure_count": context["calibration"]["attempts"][-1][
        "failure_count"
      ],
      "selected_fall_count": context["calibration"]["attempts"][-1]["fall_count"],
      "selected_target_failure_fraction": context["calibration"]["attempts"][
        -1
      ]["target_failure_fraction"],
    }
  }
  payload["precalibration_protocol"] = precal_binding
  payload["protocol_revision"] = 1 if args.context_id == "L_effect" else 2
  payload["status"] = (
    f"prospectively_frozen_before_{args.context_id}_adaptation"
  )
  payload["fresh_evidence_boundary"] = {
    "current_context_adaptation_outcomes_seen": False,
    "current_context_final_test_outcomes_seen": False,
    "protocol_must_be_committed_before_adaptation": True,
  }
  payload["conditional_execution"]["lateral_final_gate_passed"] = False
  if args.context_id == "C_effect":
    if args.lateral_final_result is None:
      raise ValueError("contact freeze requires the passed lateral final result")
    lateral_path = args.lateral_final_result.resolve()
    lateral = json.loads(lateral_path.read_text())
    if (
      lateral.get("protocol_id") != PROTOCOL_ID
      or lateral.get("context_id") != "L_effect"
      or lateral.get("development_gate", {}).get("passed") is not True
    ):
      raise RuntimeError("contact cannot start because the lateral gate did not pass")
    payload["conditional_execution"]["lateral_final_gate_passed"] = True
    payload["conditional_execution"]["lateral_final_result"] = {
      "file": str(lateral_path.relative_to(repo)),
      "sha256": _sha256(lateral_path),
      "target_success_delta": lateral["development_gate"][
        "target_success_delta"
      ],
      "target_fall_delta": lateral["development_gate"]["target_fall_delta"],
      "d0_success_delta": lateral["development_gate"]["d0_success_delta"],
    }
  return payload


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("v22 freeze requires clean tracked files")
  payload = (
    _precalibration(repo, args)
    if args.stage == "precalibration"
    else _adaptation(repo, args)
  )
  _write_immutable(args.output.resolve(), payload)
  print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
