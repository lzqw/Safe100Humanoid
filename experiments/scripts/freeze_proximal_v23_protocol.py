"""Seal the prospective v23 boundary after implementation, before outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from proximal_v23_protocol import (
  ADAPTATION_SEED,
  BASE_CHECKPOINT_SHA256,
  CONTEXT_FILE_SHA256,
  CONTEXT_ID,
  CONTEXT_PARAMETERS_SHA256,
  EVAL_BATCH_SIZE,
  FINAL_D0_EPISODES,
  FINAL_D0_SEED,
  FINAL_TARGET_EPISODES,
  FINAL_TARGET_SEED,
  MAXIMUM_TARGET_FALL_DELTA,
  MINIMUM_D0_SUCCESS_DELTA,
  MINIMUM_TARGET_SUCCESS_DELTA,
  POLICY_METHOD,
  PROTOCOL_ID,
  REPORT_BOOTSTRAP_SAMPLES,
  REPORT_BOOTSTRAP_SEEDS,
  fresh_randomness_report,
  formal_algorithm_parameters,
)
from src.tasks.stairs_cbf.deployment_context import load_calibrated_v22_context


SOURCE_FILES = (
  "docs/CBF_PROXIMAL_V23_PROTOCOL.md",
  "src/tasks/stairs_cbf/actions.py",
  "src/tasks/stairs_cbf/command.py",
  "src/tasks/stairs_cbf/config.py",
  "src/tasks/stairs_cbf/deployment_context.py",
  "src/tasks/stairs_cbf/mdp.py",
  "src/tasks/stairs_cbf/online.py",
  "src/tasks/stairs_cbf/proximal.py",
  "src/tasks/stairs_cbf/proximal_context.py",
  "experiments/scripts/evaluate_proximal_v23.py",
  "experiments/scripts/proximal_v23_io.py",
  "experiments/scripts/proximal_v23_protocol.py",
  "experiments/scripts/refine_proximal_v23.py",
  "experiments/scripts/audit_proximal_v23.py",
  "experiments/scripts/plot_proximal_v23.py",
  "experiments/scripts/freeze_proximal_v23_protocol.py",
  "experiments/scripts/run_proximal_v23.sh",
  "experiments/tests/test_proximal_v23.py",
  "results/online/specialist_v22/calibration/L_effect/context.json",
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


def _verify_committed_sources(repo: Path, commit: str) -> dict[str, str]:
  hashes: dict[str, str] = {}
  for relative in SOURCE_FILES:
    path = repo / relative
    if not path.is_file():
      raise FileNotFoundError(path)
    content = path.read_bytes()
    blob = subprocess.run(
      ["git", "show", f"{commit}:{relative}"],
      cwd=repo,
      check=True,
      capture_output=True,
    ).stdout
    if blob != content:
      raise RuntimeError(f"prospective source is not committed at {commit}: {relative}")
    hashes[relative] = hashlib.sha256(content).hexdigest()
  return hashes


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
  rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.exists() and path.read_text() != rendered:
    raise RuntimeError(f"refusing to overwrite a different v23 protocol: {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(rendered)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--base-checkpoint-reference", required=True)
  parser.add_argument(
    "--base-checkpoint-sha256", default=BASE_CHECKPOINT_SHA256
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  repo = args.repo.resolve()
  context_path = args.context.resolve()
  output_path = args.output.resolve()
  commit = _git_output(repo, "rev-parse", "HEAD")
  if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("tracked worktree must be clean before freezing v23")
  source_hashes = _verify_committed_sources(repo, commit)
  context = load_calibrated_v22_context(context_path)
  selected = context["calibration"]["attempts"][-1]
  if (
    context.get("context_id") != CONTEXT_ID
    or _sha256(context_path) != CONTEXT_FILE_SHA256
    or context.get("parameters_sha256") != CONTEXT_PARAMETERS_SHA256
    or selected.get("candidate_seed") != 51011
    or selected.get("success_rate") != 0.6875
    or selected.get("failure_count") != 160
    or selected.get("target_failure_fraction") != 0.925
    or selected.get("base_policy_only") is not True
    or context["calibration"].get("adapted_policy_evaluations_used") is not False
  ):
    raise RuntimeError("reused lateral context no longer matches its base-only calibration")
  if args.base_checkpoint_sha256 != BASE_CHECKPOINT_SHA256:
    raise ValueError("base checkpoint hash differs from the preregistered checkpoint")
  randomness = fresh_randomness_report(repo)
  if not randomness["passed"]:
    raise RuntimeError(f"v23 randomness collision: {randomness['collisions']}")

  payload = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "policy_method": POLICY_METHOD,
    "experiment_class": "single frozen lateral context development test",
    "implementation_boundary": {
      "git_commit": commit,
      "source_files": source_hashes,
      "all_execution_sources_committed_before_outcomes": True,
    },
    "base_checkpoint": {
      "reference": args.base_checkpoint_reference,
      "sha256": args.base_checkpoint_sha256,
    },
    "context": {
      "context_id": CONTEXT_ID,
      "file": str(context_path.relative_to(repo)),
      "file_sha256": _sha256(context_path),
      "parameters_sha256": context["parameters_sha256"],
      "calibration_selected_candidate_seed": 51011,
      "calibration_base_success_rate": selected["success_rate"],
      "calibration_failure_count": selected["failure_count"],
      "calibration_lateral_purity": selected["target_failure_fraction"],
      "reused_without_reselection": True,
      "adapted_policy_outcomes_used_for_context": False,
    },
    "training": formal_algorithm_parameters(),
    "learning_semantics": {
      "actor_observation_dim": 405,
      "critic_observation_dim": 838,
      "single_actor": True,
      "single_privileged_critic": True,
      "runtime_cbf_executes_filtered_action": True,
      "ppo_stores_raw_sampled_action_and_behavior_log_probability": True,
      "moving_reference": "current round-start pi_k, refreshed every round",
      "forward_kl": "analytic diagonal Gaussian KL(pi_theta || pi_k)",
      "reference_stop_gradient": True,
      "reward": "base task + fall event + dual-CBF; specialist term absent",
      "ordinary_cbf_intervention_is_failure": False,
      "one_on_policy_batch_per_round": True,
      "normal_physical_initial_states_only": True,
    },
    "rollback": {
      "allowed_reasons": [
        "non-finite actor/critic/loss/gradient state",
        "moving forward KL above 0.01",
        "raw-action or behavior-Gaussian routing corruption",
        "actor or critic optimizer-state corruption",
      ],
      "restores": ["actor", "critic", "actor optimizer", "critic optimizer"],
      "performance_rollback_forbidden": True,
    },
    "excluded": {
      "specialist_reward": True,
      "failure_precursor_bank": True,
      "matched_success_bank": True,
      "state_restart": True,
      "grouped_advantages": True,
      "dual_rollout": True,
      "candidate_fractions": True,
      "candidate_screen_or_confirmation": True,
      "target_or_D0_training_gate": True,
      "validation_or_best_so_far_selection": True,
      "multi_context_or_off_diagonal_audit": True,
    },
    "final_policy": {
      "rule": "round 8 actor, independent of performance",
      "round_start_and_end_checkpoints_are_recovery_and_curve_only": True,
    },
    "evaluation": {
      "target_episodes": FINAL_TARGET_EPISODES,
      "D0_episodes": FINAL_D0_EPISODES,
      "batch_size": EVAL_BATCH_SIZE,
      "target_seed_start": FINAL_TARGET_SEED,
      "D0_seed_start": FINAL_D0_SEED,
      "base_and_final_conditions_paired": True,
      "report_bootstrap_samples": REPORT_BOOTSTRAP_SAMPLES,
      "report_bootstrap_seeds": REPORT_BOOTSTRAP_SEEDS,
      "confidence_intervals_are_gates": False,
      "repairs_and_regressions_reported": True,
    },
    "development_gate": {
      "minimum_target_success_delta": MINIMUM_TARGET_SUCCESS_DELTA,
      "maximum_target_fall_delta": MAXIMUM_TARGET_FALL_DELTA,
      "minimum_D0_success_delta": MINIMUM_D0_SUCCESS_DELTA,
      "point_estimates_only": True,
      "used_for_training_rollback_stopping_or_selection": False,
      "contact_context_eligible_only_after_pass": True,
    },
    "randomness_preflight": randomness,
    "fresh_execution_seeds": {
      "adaptation_seed": ADAPTATION_SEED,
      "final_target_seed_start": FINAL_TARGET_SEED,
      "final_D0_seed_start": FINAL_D0_SEED,
      "report_bootstrap_seeds": REPORT_BOOTSTRAP_SEEDS,
    },
    "prospective_execution": {
      "adapted_policy_outcomes_observed": False,
      "formal_adaptation_started": False,
      "fresh_adaptation_count_planned": 1,
    },
  }
  _write_immutable(output_path, payload)
  print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
