"""Build immutable v21 protocol revisions at the three evidence boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from specialist_v21_protocol import (
  BETA_GRID,
  CALIBRATION_BATCH_SIZE,
  CALIBRATION_EPISODES,
  CANDIDATE_D0_EPISODES,
  CANDIDATE_EVALUATION_EPISODES_PER_ROUND,
  CANDIDATE_FRACTIONS,
  CANDIDATE_SCREEN_EPISODES,
  CONFIRMATION_BLOCKS,
  CONFIRMATION_EPISODES_PER_BLOCK,
  CONTEXT_ADAPTATION_SEEDS,
  CONTEXT_CALIBRATION_CANDIDATE_SEEDS,
  CONTEXT_CALIBRATION_EVALUATION_SEEDS,
  CONTEXT_DEVELOPMENT_SELECTION_SEEDS,
  CONTEXT_FORMAL_AUDIT_SEEDS,
  CONTEXT_MONITOR_SEEDS,
  CONTEXTS,
  DEVELOPMENT_SELECTION_EPISODES,
  FORMAL_BOOTSTRAP_SAMPLES,
  FORMAL_BOOTSTRAP_SEED,
  FORMAL_D0_EPISODES,
  FORMAL_EVAL_BATCH_SIZE,
  FORMAL_MONITOR_EPISODES,
  FORMAL_ROUNDS,
  FORMAL_TARGET_EPISODES,
  POLICY_METHOD,
  PROTOCOL_ID,
  TELEMETRY_ENVIRONMENT_ID_PER_BATCH,
  V21_CONTEXT_SPECS,
  V21_DEVELOPMENT_CONTEXTS,
  V21_FORMAL_CONTEXTS,
  canonical_sha256,
  fresh_randomness_report,
)

SOURCE_FILES = (
  "docs/DEPLOYMENT_V21_PROTOCOL.md",
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
  "experiments/scripts/calibrate_deployment_contexts_v21.py",
  "experiments/scripts/refine_deployment_v21.py",
  "experiments/scripts/specialist_v21_protocol.py",
  "experiments/scripts/specialist_v21_tables.py",
  "experiments/scripts/select_development_beta_v21.py",
  "experiments/scripts/evaluate_learning_curve_v21.py",
  "experiments/scripts/audit_deployment_v21.py",
  "experiments/scripts/aggregate_deployment_v21.py",
  "experiments/scripts/plot_deployment_v21.py",
  "experiments/scripts/freeze_specialist_v21_protocol.py",
  "experiments/scripts/run_specialist_calibration_v21.sh",
  "experiments/scripts/run_specialist_v21.sh",
  "experiments/scripts/run_specialist_queue_v21.sh",
  "experiments/tests/test_specialist_v21.py",
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


def _git_blob(repo: Path, commit: str, relative: str) -> bytes:
  return subprocess.run(
    ["git", "show", f"{commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
  rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.exists() and path.read_text() != rendered:
    raise RuntimeError(f"refusing to overwrite a different v21 protocol: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(rendered)


def _relative(repo: Path, path: Path) -> str:
  try:
    return str(path.resolve().relative_to(repo))
  except ValueError as exc:
    raise ValueError(f"v21 artifact is outside the repository: {path}") from exc


def _verify_protocol_blob(
  repo: Path, path: Path, commit: str
) -> dict[str, str]:
  relative = _relative(repo, path)
  digest = _sha256(path)
  if hashlib.sha256(_git_blob(repo, commit, relative)).hexdigest() != digest:
    raise RuntimeError(f"v21 protocol differs from commit {commit}: {relative}")
  return {"file": relative, "sha256": digest, "git_commit": commit}


def _source_hashes(repo: Path) -> dict[str, str]:
  missing = [relative for relative in SOURCE_FILES if not (repo / relative).is_file()]
  if missing:
    raise FileNotFoundError(f"v21 prospective source files are missing: {missing}")
  return {relative: _sha256(repo / relative) for relative in SOURCE_FILES}


def _precalibration(args: argparse.Namespace) -> dict[str, Any]:
  repo = args.repo.resolve()
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("v21 precalibration freeze requires clean tracked files")
  randomness = fresh_randomness_report(repo)
  if not randomness["passed"]:
    raise RuntimeError(f"v21 randomness collision: {randomness['collisions']}")
  context_specs = {
    context_id: V21_CONTEXT_SPECS[context_id] for context_id in CONTEXTS
  }
  payload = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "protocol_revision": 0,
    "status": (
      "prospectively_frozen_before_base_only_calibration_and_development"
    ),
    "policy_method": POLICY_METHOD,
    "experiment_unit": (
      "one fixed deployment context plus pi0 plus exactly one adaptation "
      "plus fresh paired evaluation"
    ),
    "context_matrix": {
      "development_excluded_from_formal": list(V21_DEVELOPMENT_CONTEXTS),
      "formal": list(V21_FORMAL_CONTEXTS),
      "specifications": context_specs,
      "specifications_sha256": canonical_sha256(context_specs),
      "calibration_family_sweep": {
        "candidate_indices_in_order": list(range(8, 20)),
        "severity_increases_from_zero_to_one": True,
        "perturbation_direction_fixed_within_family": True,
        "normalized_action_bias_pattern_fixed_within_family": True,
        "low_fixed_disturbance_pulses_in_L1_and_L4_are_excitation_carriers": True,
        "adapted_policy_outcomes_used": False,
      },
    },
    "adaptation_seeds": CONTEXT_ADAPTATION_SEEDS,
    "randomness_preflight": randomness,
    "calibration": {
      "base_policy_only": True,
      "adapted_policy_evaluations_used": False,
      "first_qualifying_candidate_is_frozen": True,
      "success_rate_bounds_inclusive": [0.70, 0.85],
      "minimum_fall_count": 100,
      "lateral_minimum_target_failure_fraction": 0.80,
      "lateral_maximum_second_failure_fraction": 0.30,
      "contact_stability_minimum_target_failure_fraction": 0.75,
      "contact_stability_maximum_second_failure_fraction": 0.20,
      "episodes_per_candidate": CALIBRATION_EPISODES,
      "eval_batch_size": CALIBRATION_BATCH_SIZE,
      "candidate_seeds": {
        key: list(value)
        for key, value in CONTEXT_CALIBRATION_CANDIDATE_SEEDS.items()
      },
      "evaluation_seed_bases": CONTEXT_CALIBRATION_EVALUATION_SEEDS,
    },
    "development": {
      "contexts": list(V21_DEVELOPMENT_CONTEXTS),
      "excluded_from_formal_claims": True,
      "beta_grid": list(BETA_GRID),
      "beta_zero_is_v20_style_control": True,
      "selection_metric": "mean across L_dev/C_dev of repair_rate - regression_rate",
      "tie_breaks": [
        "higher worst-context score",
        "lower mean regression rate",
        "lower beta",
      ],
      "paired_evaluation_episodes_per_policy_per_context": (
        DEVELOPMENT_SELECTION_EPISODES
      ),
      "evaluation_seeds": CONTEXT_DEVELOPMENT_SELECTION_SEEDS,
      "formal_beta_frozen_after_development": True,
    },
    "training": {
      "fixed_round_budget": FORMAL_ROUNDS,
      "accepted_update_count_is_validity_gate": False,
      "zero_retained_updates_are_valid_and_final_equals_pi0": True,
      "normal_failure_success_start_slots": [40, 12, 12],
      "dual_rollouts_per_round": 2,
      "rollout_steps_per_environment": 1024,
      "candidate_fractions": list(CANDIDATE_FRACTIONS),
      "screening_paired_episodes_per_candidate": CANDIDATE_SCREEN_EPISODES,
      "confirmation_blocks": CONFIRMATION_BLOCKS,
      "confirmation_paired_episodes_per_block": (
        CONFIRMATION_EPISODES_PER_BLOCK
      ),
      "candidate_evaluation_episodes_per_round": (
        CANDIDATE_EVALUATION_EPISODES_PER_ROUND
      ),
      "confirmation_mean_success_delta_strictly_positive": True,
      "confirmation_minimum_positive_blocks": 2,
      "confirmation_maximum_mean_fall_delta": 0.03,
      "candidate_d0_paired_episodes": CANDIDATE_D0_EPISODES,
      "d0_retention_must_pass": True,
      "matched_success_actor_advantage_excluded_only_when_beta_positive": True,
      "critic_uses_all_transitions": True,
      "broad_d0_or_global_retention_bank": False,
    },
    "monitor": {
      "paired_conditions": FORMAL_MONITOR_EPISODES,
      "seeds": CONTEXT_MONITOR_SEEDS,
      "checkpoints": list(range(FORMAL_ROUNDS + 1)),
      "never_accessed_during_training": True,
      "candidate_selection_diagnostics_used_for_curve": False,
    },
    "formal": {
      "contexts": list(V21_FORMAL_CONTEXTS),
      "runs_per_context": {"control_beta_zero": 1, "v21_selected_beta": 1},
      "total_adaptation_runs": 20,
      "same_pi0_context_and_deployment_seed": True,
      "same_evaluation_randomness_for_base_control_v21": True,
      "target_paired_episodes_per_context": FORMAL_TARGET_EPISODES,
      "d0_paired_episodes_per_context": FORMAL_D0_EPISODES,
      "eval_batch_size": FORMAL_EVAL_BATCH_SIZE,
      "audit_seeds": CONTEXT_FORMAL_AUDIT_SEEDS,
      "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
      "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
      "statistical_unit": "deployment_context",
      "per_mode_gate": {
        "mean_success_delta_strictly_positive": True,
        "minimum_positive_contexts_out_of_five": 4,
        "maximum_mean_fall_delta": 0.03,
        "minimum_mean_d0_success_delta": -0.05,
      },
      "lcb95_positive_is_strong_evidence_not_validity_gate": True,
    },
    "mechanism_telemetry": {
      "captured_inline_during_actual_formal_rollouts": True,
      "environment_id_per_evaluation_batch": (
        TELEMETRY_ENVIRONMENT_ID_PER_BATCH
      ),
      "outcome_embedded_from_same_rollout": True,
      "post_audit_replay_for_mechanism_curves": False,
      "compact_curve_construction": {
        "normalized_episode_phase_bins": 101,
        "trace_level_interpolation_before_aggregation": True,
        "mean_and_interquartile_band_across_traces": True,
        "same_rollout_outcome_retained_per_trace": True,
      },
    },
    "analysis_plan": {
      "per_context_episode_interval": "paired episode bootstrap",
      "cross_context_interval": "bootstrap of five deployment-context point estimates",
      "cross_context_statistical_unit": "deployment_context",
      "formal_gate_applied_to": "v21 minus common base independently per mode",
      "control_gate_reported_descriptively": True,
      "v21_minus_control_reported": True,
      "selectivity_comparison": (
        "paired differences of base-referenced repair rate, regression rate, "
        "and repair-minus-regression across contexts"
      ),
      "monitor_curves": "pi0 through pi8 on never-accessed E_curve only",
      "candidate_selection_diagnostics_excluded_from_monitor_curves": True,
    },
    "retry_policy": {
      "poor_algorithm_outcome_may_be_rerun": False,
      "infrastructure_retry_only": True,
      "infrastructure_retry_requires_identical_context_seed_commit_and_checkpoint": True,
      "retry_provenance_required": True,
    },
    "sealed_inputs": {
      "source_commit": current_commit,
      "source_files": list(SOURCE_FILES),
      "source_file_sha256": _source_hashes(repo),
      "base_policy_checkpoint_reference": args.base_checkpoint_reference,
      "base_policy_checkpoint_sha256": args.base_checkpoint_sha256,
    },
    "fresh_evidence_boundary": {
      "base_only_calibration_outcomes_seen": False,
      "development_adaptation_outcomes_seen": False,
      "formal_adaptation_or_audit_outcomes_seen": False,
      "this_protocol_must_be_committed_before_calibration": True,
    },
  }
  return payload


def _development(args: argparse.Namespace) -> dict[str, Any]:
  repo = args.repo.resolve()
  pre_path = args.input_protocol.resolve()
  pre = json.loads(pre_path.read_text())
  pre_binding = _verify_protocol_blob(repo, pre_path, args.input_commit)
  if pre.get("protocol_id") != PROTOCOL_ID or pre.get("protocol_revision") != 0:
    raise RuntimeError("unexpected v21 precalibration protocol")
  current_hashes = _source_hashes(repo)
  if current_hashes != pre["sealed_inputs"]["source_file_sha256"]:
    raise RuntimeError("v21 source changed after precalibration freeze")
  from src.tasks.stairs_cbf.deployment_context import load_calibrated_v21_context

  contexts = {}
  for context_id in CONTEXTS:
    context_path = args.context_dir.resolve() / f"{context_id}.json"
    context = load_calibrated_v21_context(context_path)
    calibration = context["calibration"]
    if (
      context["context_id"] != context_id
      or calibration["prospective_protocol_git_commit"] != args.input_commit
      or calibration["prospective_protocol_file_sha256"] != pre_binding["sha256"]
      or calibration["adapted_policy_evaluations_used"] is not False
    ):
      raise RuntimeError(f"invalid v21 calibration boundary for {context_id}")
    selected = calibration["attempts"][-1]
    contexts[context_id] = {
      "file": _relative(repo, context_path),
      "file_sha256": _sha256(context_path),
      "parameters_sha256": context["parameters_sha256"],
      "context_family": context["context_family"],
      "formal_context": context["formal_context"],
      "selected_calibration_seed": calibration["selected_candidate_seed"],
      "selected_base_success_rate": selected["success_rate"],
      "selected_fall_count": selected["fall_count"],
      "selected_target_failure_fraction": selected["target_failure_fraction"],
      "selected_second_failure_fraction": selected["second_failure_fraction"],
    }
  payload = deepcopy(pre)
  payload["protocol_revision"] = 1
  payload["status"] = "prospectively_frozen_before_development_beta_selection"
  payload["precalibration_protocol"] = pre_binding
  payload["sealed_inputs"]["contexts"] = contexts
  payload["fresh_evidence_boundary"] = {
    "base_only_calibration_completed": True,
    "adapted_policy_used_for_context_selection": False,
    "development_adaptation_outcomes_seen": False,
    "formal_adaptation_or_audit_outcomes_seen": False,
    "this_protocol_must_be_committed_before_development_runs": True,
  }
  return payload


def _formal(args: argparse.Namespace) -> dict[str, Any]:
  repo = args.repo.resolve()
  development_path = args.input_protocol.resolve()
  development = json.loads(development_path.read_text())
  binding = _verify_protocol_blob(repo, development_path, args.input_commit)
  if (
    development.get("protocol_id") != PROTOCOL_ID
    or development.get("protocol_revision") != 1
  ):
    raise RuntimeError("unexpected v21 development protocol")
  if _source_hashes(repo) != development["sealed_inputs"]["source_file_sha256"]:
    raise RuntimeError("v21 source changed after development protocol freeze")
  selection_path = args.development_selection.resolve()
  selection = json.loads(selection_path.read_text())
  if (
    selection.get("protocol_id") != PROTOCOL_ID
    or selection.get("formal_context_outcomes_seen") is not False
    or selection.get("contexts") != list(V21_DEVELOPMENT_CONTEXTS)
  ):
    raise RuntimeError("invalid v21 development beta selection")
  selected_beta = float(selection["selection"]["selected_beta"])
  if selected_beta not in BETA_GRID:
    raise RuntimeError("development selected beta outside the frozen grid")
  payload = deepcopy(development)
  payload["protocol_revision"] = 2
  payload["status"] = "prospectively_frozen_before_formal_adaptation"
  payload["development_protocol"] = binding
  payload["development_selection"] = {
    "file": _relative(repo, selection_path),
    "file_sha256": _sha256(selection_path),
    "selection": selection["selection"],
  }
  payload["formal"]["selected_beta"] = selected_beta
  payload["formal"]["formal_beta_is_frozen"] = True
  payload["fresh_evidence_boundary"] = {
    "base_only_calibration_completed": True,
    "development_beta_selection_completed": True,
    "development_contexts_excluded_from_formal_claims": True,
    "formal_adaptation_or_audit_outcomes_seen": False,
    "this_protocol_must_be_committed_before_formal_adaptation": True,
  }
  return payload


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument(
    "--stage", choices=("precalibration", "development", "formal"), required=True
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--base-checkpoint-reference")
  parser.add_argument("--base-checkpoint-sha256")
  parser.add_argument("--input-protocol", type=Path)
  parser.add_argument("--input-commit")
  parser.add_argument("--context-dir", type=Path)
  parser.add_argument("--development-selection", type=Path)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  required = {
    "precalibration": (
      "base_checkpoint_reference",
      "base_checkpoint_sha256",
    ),
    "development": ("input_protocol", "input_commit", "context_dir"),
    "formal": (
      "input_protocol",
      "input_commit",
      "development_selection",
    ),
  }[args.stage]
  missing = [name for name in required if getattr(args, name) is None]
  if missing:
    raise ValueError(f"v21 {args.stage} freeze is missing arguments: {missing}")
  payload = {
    "precalibration": _precalibration,
    "development": _development,
    "formal": _formal,
  }[args.stage](args)
  _write_immutable(args.output.resolve(), payload)
  print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
