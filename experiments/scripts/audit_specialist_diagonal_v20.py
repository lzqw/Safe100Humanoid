"""Independent one-specialist formal paired audit for fixed-budget v20."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audit_specialists_diagonal_v19 import (
  _balanced_restart_strata,
  _binary_column,
  _git_output,
  _load_rows,
  _sha256,
  _tracked_worktree_is_clean,
  _validate_context,
)
from diagonal_audit_stats import (
  hierarchical_paired_scene_interval_v19,
  independent_diagonal_scene_gate_v19,
)
from online_refine_stairs import (
  _actor_state,
  _actor_state_sha256,
  _evaluate_state,
)
from specialist_v20_protocol import (
  FORMAL_ADAPTATION_SEEDS,
  FORMAL_AUDIT_SEED,
  FORMAL_BOOTSTRAP_SAMPLES,
  FORMAL_BOOTSTRAP_SEED,
  FORMAL_D0_EPISODES,
  FORMAL_ROUNDS,
  FORMAL_TARGET_EPISODES,
  POLICY_METHOD,
  PROTOCOL_ID,
  SPECIALIST_MODES,
)

PAIRED_FIELDS = [
  "specialist_mode",
  "evaluation_mode",
  "evaluation_role",
  "adaptation_seed",
  "pair_index",
  "evaluation_seed",
  "environment_id",
  "baseline_success",
  "final_success",
  "baseline_fell",
  "final_fell",
  "transition_class",
  "baseline_failure_type",
  "final_failure_type",
  "baseline_return",
  "final_return",
  "baseline_max_riser",
  "final_max_riser",
  "baseline_intervention_per_riser",
  "final_intervention_per_riser",
  "baseline_correction_mean",
  "final_correction_mean",
  "baseline_mean_abs_centerline_error",
  "final_mean_abs_centerline_error",
  "baseline_max_abs_heading_error",
  "final_max_abs_heading_error",
  "baseline_max_left_slip_speed",
  "final_max_left_slip_speed",
  "baseline_max_right_slip_speed",
  "final_max_right_slip_speed",
  "baseline_contact_mismatch_fraction",
  "final_contact_mismatch_fraction",
  "baseline_max_roll_signal",
  "final_max_roll_signal",
  "baseline_max_pitch_signal",
  "final_max_pitch_signal",
  "baseline_max_angular_velocity_signal",
  "final_max_angular_velocity_signal",
]
TRANSITION_CLASSES = (
  "failure_to_success",
  "success_to_failure",
  "success_to_success",
  "failure_to_failure",
)
AUDIT_AMENDMENT_RELATIVE = Path(
  "results/online/specialist_v20/audit_amendment.json"
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--mode", choices=SPECIALIST_MODES, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--audit-commit", required=True)
  parser.add_argument("--audit-amendment", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--adaptation-seeds",
    nargs="+",
    type=int,
    default=FORMAL_ADAPTATION_SEEDS,
  )
  parser.add_argument("--eval-batch-size", type=int, default=128)
  parser.add_argument(
    "--target-episodes", type=int, default=FORMAL_TARGET_EPISODES
  )
  parser.add_argument("--d0-episodes", type=int, default=FORMAL_D0_EPISODES)
  parser.add_argument(
    "--bootstrap-samples", type=int, default=FORMAL_BOOTSTRAP_SAMPLES
  )
  parser.add_argument("--audit-seed", type=int, default=FORMAL_AUDIT_SEED)
  parser.add_argument(
    "--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _validate_protocol(
  protocol: dict[str, Any], args: argparse.Namespace
) -> None:
  expected = {
    "protocol_id": PROTOCOL_ID,
    "protocol_revision": 1,
    "status": "prospectively_frozen_before_formal_adaptation",
    "policy_method": POLICY_METHOD,
    "specialist_modes": list(SPECIALIST_MODES),
    "adaptation_seeds": list(FORMAL_ADAPTATION_SEEDS),
  }
  mismatches = {
    key: {"actual": protocol.get(key), "required": expected_value}
    for key, expected_value in expected.items()
    if protocol.get(key) != expected_value
  }
  training = protocol.get("training", {})
  required_training = {
    "fixed_round_budget": FORMAL_ROUNDS,
    "early_termination_enabled": False,
    "minimum_accepted_updates": None,
    "rejection_patience": None,
    "accepted_update_count_is_validity_gate": False,
    "zero_to_eight_retained_updates_are_valid": True,
  }
  evaluation = protocol.get("evaluation", {})
  required_evaluation = {
    "audit_seed": FORMAL_AUDIT_SEED,
    "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
    "eval_batch_size": 128,
    "target_paired_episodes_per_adaptation_seed": (
      FORMAL_TARGET_EPISODES
    ),
    "d0_paired_episodes_per_adaptation_seed": FORMAL_D0_EPISODES,
    "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
    "specialists_audited_independently": True,
    "runtime_cbf": True,
    "off_diagonal_evaluation": False,
    "macro_average_computed": False,
    "filter_free_evaluation": False,
    "cbf_independence_gate": False,
  }
  for prefix, actual, required in (
    ("training", training, required_training),
    ("evaluation", evaluation, required_evaluation),
  ):
    for key, expected_value in required.items():
      if actual.get(key) != expected_value:
        mismatches[f"{prefix}.{key}"] = {
          "actual": actual.get(key),
          "required": expected_value,
        }
  if mismatches:
    raise ValueError(f"v20 protocol mismatch: {mismatches}")
  if args.smoke:
    return
  runtime = {
    "adaptation_seeds": list(args.adaptation_seeds),
    "audit_seed": args.audit_seed,
    "bootstrap_seed": args.bootstrap_seed,
    "eval_batch_size": args.eval_batch_size,
    "target_paired_episodes_per_adaptation_seed": args.target_episodes,
    "d0_paired_episodes_per_adaptation_seed": args.d0_episodes,
    "bootstrap_samples": args.bootstrap_samples,
  }
  required_runtime = {
    "adaptation_seeds": list(FORMAL_ADAPTATION_SEEDS),
    "audit_seed": FORMAL_AUDIT_SEED,
    "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
    "eval_batch_size": 128,
    "target_paired_episodes_per_adaptation_seed": (
      FORMAL_TARGET_EPISODES
    ),
    "d0_paired_episodes_per_adaptation_seed": FORMAL_D0_EPISODES,
    "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
  }
  runtime_mismatches = {
    key: {"actual": value, "required": required_runtime[key]}
    for key, value in runtime.items()
    if value != required_runtime[key]
  }
  if runtime_mismatches:
    raise ValueError(f"formal v20 audit runtime mismatch: {runtime_mismatches}")


def _git_blob(repo: Path, commit: str, relative: str) -> bytes:
  return subprocess.run(
    ["git", "show", f"{commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout


def _validate_audit_amendment(
  *,
  repo: Path,
  protocol_path: Path,
  protocol_commit: str,
  audit_commit: str,
  amendment_path: Path,
) -> dict[str, Any]:
  """Validate the post-training, pre-audit infrastructure amendment seal."""
  try:
    protocol_relative = str(protocol_path.relative_to(repo))
    amendment_relative = str(amendment_path.relative_to(repo))
  except ValueError as error:
    raise RuntimeError("v20 audit seal files must live inside the repo") from error
  if Path(amendment_relative) != AUDIT_AMENDMENT_RELATIVE:
    raise RuntimeError("v20 audit amendment path differs from the declared path")
  protocol_sha256 = _sha256(protocol_path)
  if hashlib.sha256(
    _git_blob(repo, protocol_commit, protocol_relative)
  ).hexdigest() != protocol_sha256:
    raise RuntimeError("v20 protocol differs from its frozen training commit")
  if _git_blob(repo, audit_commit, amendment_relative) != amendment_path.read_bytes():
    raise RuntimeError("v20 audit amendment differs from its committed blob")
  amendment = json.loads(amendment_path.read_text())
  expected = {
    "amendment_id": "safe100-specialist-v20-audit-infrastructure-amendment-1",
    "schema_version": 1,
    "status": "prospectively_frozen_before_first_formal_audit_episode_outcome",
  }
  failures = [
    name for name, value in expected.items() if amendment.get(name) != value
  ]
  training = amendment.get("training_protocol", {})
  if not (
    training.get("git_commit") == protocol_commit
    and training.get("file") == protocol_relative
    and training.get("sha256") == protocol_sha256
  ):
    failures.append("training_protocol")
  boundary = amendment.get("fresh_audit_evidence_boundary", {})
  if not (
    boundary.get("formal_audit_episode_outcomes_observed") is False
    and boundary.get("formal_audit_rows_written") == 0
    and boundary.get("training_artifacts_rerun_or_modified") is False
  ):
    failures.append("fresh_audit_evidence_boundary")
  fix = amendment.get("infrastructure_fix", {})
  if not (
    fix.get("scope") == "audit checkpoint loader configuration only"
    and fix.get("brief_ppo_refinement_checkpoint_semantics_enabled") is True
    and fix.get("actor_or_checkpoint_tensor_modified") is False
  ):
    failures.append("infrastructure_fix")
  unchanged = amendment.get("unchanged_formal_evaluation", {})
  required_unchanged = {
    "adaptation_seeds": list(FORMAL_ADAPTATION_SEEDS),
    "audit_seed": FORMAL_AUDIT_SEED,
    "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
    "bootstrap_samples": FORMAL_BOOTSTRAP_SAMPLES,
    "target_paired_episodes_per_adaptation_seed": FORMAL_TARGET_EPISODES,
    "d0_paired_episodes_per_adaptation_seed": FORMAL_D0_EPISODES,
    "runtime_cbf": True,
  }
  if any(unchanged.get(key) != value for key, value in required_unchanged.items()):
    failures.append("unchanged_formal_evaluation")
  source_checks: dict[str, bool] = {}
  for relative, expected_hash in amendment.get("source_file_sha256", {}).items():
    source_checks[relative] = (
      hashlib.sha256(_git_blob(repo, audit_commit, relative)).hexdigest()
      == expected_hash
      == _sha256(repo / relative)
    )
  if not source_checks or not all(source_checks.values()):
    failures.append("source_file_sha256")
  if failures:
    raise RuntimeError(f"v20 audit amendment seal failed: {sorted(set(failures))}")
  return {
    "path": str(amendment_path),
    "relative_path": amendment_relative,
    "sha256": _sha256(amendment_path),
    "git_commit": audit_commit,
    "source_file_sha256_checks": source_checks,
    "fresh_audit_evidence_boundary": boundary,
    "infrastructure_fix": fix,
  }


def _validate_training_artifacts(
  *,
  repo: Path,
  training_root: Path,
  mode: str,
  context: dict[str, Any],
  seeds: list[int],
  base_checkpoint_sha256: str,
  protocol_commit: str,
  protocol_sha256: str,
) -> tuple[dict[int, Path], dict[str, Any]]:
  checkpoints: dict[int, Path] = {}
  runs: dict[int, Any] = {}
  initial_actor_hashes: set[str] = set()
  source_manifests: set[str] = set()
  for seed in seeds:
    run_dir = training_root / mode / f"seed{seed}"
    summary_path = run_dir / "specialist_summary.json"
    checkpoint_path = run_dir / "accepted_final.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
      raise FileNotFoundError(f"missing v20 training artifact in {run_dir}")
    summary = json.loads(summary_path.read_text())
    reasons: list[str] = []
    if summary.get("method") != POLICY_METHOD:
      reasons.append("method identity differs")
    if summary.get("learning_core") != (
      "v19 Revision-4 Observable Failure-Conditioned Brief PPO"
    ):
      reasons.append("frozen v19-R4 learning-core identity differs")
    if summary.get("formal_protocol") is not True:
      reasons.append("run is not labeled formal")
    if summary.get("protocol_completed") is not True:
      reasons.append("fixed round budget did not complete")
    if summary.get("specialist_mode") != mode or summary.get("seed") != seed:
      reasons.append("mode or adaptation seed differs")
    if summary.get("runtime_cbf") is not True or summary.get(
      "raw_policy_action_for_ppo"
    ) is not True:
      reasons.append("runtime CBF/raw-action PPO invariant differs")
    if summary.get("independent_training_branch") is not True:
      reasons.append("training branch was not independent")
    if summary.get("base_policy_checkpoint_sha256") != base_checkpoint_sha256:
      reasons.append("base checkpoint hash differs")
    frozen_protocol = summary.get("frozen_protocol") or {}
    if not (
      frozen_protocol.get("git_commit") == protocol_commit
      and frozen_protocol.get("sha256") == protocol_sha256
      and frozen_protocol.get("tracked_worktree_and_index_clean") is True
      and all((frozen_protocol.get("checks") or {}).values())
    ):
      reasons.append("training protocol seal differs")
    if summary.get("deployment_context", {}).get(
      "parameters_sha256"
    ) != context["parameters_sha256"]:
      reasons.append("deployment context hash differs")
    expansion = summary.get("actor_observation_expansion", {})
    if not (
      expansion.get("legacy_width") == 405
      and expansion.get("expanded_width") == 410
      and expansion.get("new_feature_count") == 5
      and expansion.get("pre_adaptation_policy_exactly_preserved") is True
      and expansion.get("legacy_input_columns_frozen_during_adaptation")
      is True
      and expansion.get("legacy_first_layer_input_column_change_max_abs")
      == 0.0
      and expansion.get("new_input_columns_use_full_actor_learning_rate")
      is True
    ):
      reasons.append("observable input-adapter proof differs")
    structural = summary.get("structural_checks", {})
    if not (
      structural.get("actor_new_feature_count") == 5
      and math.isclose(
        structural.get(
          "actor_new_feature_learning_rate_multiplier", math.nan
        ),
        1.0,
      )
      and structural.get("freeze_legacy_actor_input_columns") is True
      and structural.get("actor_new_feature_optimizer_group_count") == 1
      and math.isclose(
        structural.get(
          "actor_new_feature_optimizer_learning_rate", math.nan
        ),
        5.0e-6,
      )
    ):
      reasons.append("full-rate observable adapter differs")
    if summary.get("integer_start_mixture_for_64_envs") != {
      "normal": 40,
      "failure": 12,
      "success": 12,
    }:
      reasons.append("40/12/12 replay allocation differs")
    if summary.get("failure_precursor_policy_weight") != 1.0 or summary.get(
      "success_counterexample_policy_weight"
    ) != 1.25:
      reasons.append("failure/success actor weights differ")
    selection = summary.get("candidate_selection", {})
    if not (
      selection.get("fractions") == [0.5, 1.0, 1.5]
      and selection.get("screening_paired_episodes_per_candidate") == 64
      and selection.get("independent_confirmation_paired_episodes") == 128
      and selection.get("candidate_episode_total") == 320
    ):
      reasons.append("two-stage candidate selection differs")
    protocol = summary.get("round_protocol", {})
    rounds = summary.get("rounds", [])
    accepted = protocol.get("accepted_updates", -1)
    if not (
      protocol.get("fixed_round_budget") == FORMAL_ROUNDS
      and protocol.get("actual_rounds") == FORMAL_ROUNDS
      and len(rounds) == FORMAL_ROUNDS
      and protocol.get("early_termination_enabled") is False
      and protocol.get("minimum_accepted_updates") is None
      and protocol.get("rejection_patience") is None
      and protocol.get("retained_update_count_is_protocol_gate") is False
      and protocol.get("zero_to_eight_retained_updates_are_valid") is True
      and 0 <= accepted <= FORMAL_ROUNDS
    ):
      reasons.append("fixed-budget round protocol differs")
    if [record.get("round") for record in rounds] != list(
      range(1, FORMAL_ROUNDS + 1)
    ):
      reasons.append("round indices are not exactly 1..8")
    if rounds and rounds[-1].get("accepted_update_count") != accepted:
      reasons.append("retained-update diagnostic count differs")
    for record in rounds:
      metrics = record.get("full_update_metrics", {})
      adapter = record.get("round_end_adapter", {})
      collectors = metrics.get("collector_metrics", [])
      if not (
        metrics.get("dual_rollout_batch_count") == 2
        and metrics.get("dual_rollout_gradient_cosine_is_gate") is False
        and metrics.get("v19_normal_advantage_count") == 81920.0
        and metrics.get("v19_failure_advantage_count") == 24576.0
        and metrics.get("v19_success_advantage_count") == 24576.0
        and len(record.get("dual_rollout_seeds", [])) == 2
        and len(set(record.get("dual_rollout_seeds", []))) == 2
      ):
        reasons.append(f"round {record.get('round')} dual PPO differs")
      if not (
        record.get("accepted_update_count_is_protocol_gate") is False
        and adapter.get(
          "legacy_input_column_change_from_initial_max_abs"
        )
        == 0.0
      ):
        reasons.append(f"round {record.get('round')} adapter proof differs")
      if not (
        len(collectors) == 2
        and all(
          item.get("matched_pair_sampling") is True
          and (item.get("matched_pair_audit") or {}).get(
            "exact_match_passed"
          )
          is True
          and (item.get("matched_pair_audit") or {}).get("pair_count") == 12
          and (item.get("matched_pair_audit") or {}).get(
            "maximum_marginal_imbalance"
          )
          == 0
          and _balanced_restart_strata(
            item.get("matched_pair_audit") or {}
          )
          for item in collectors
        )
      ):
        reasons.append(f"round {record.get('round')} exact replay differs")
      if not all(
        (item.get("bank_update_transaction") or {}).get("attempted") is True
        and (
          (item.get("bank_update_transaction") or {}).get(
            "usable_preflight"
          )
          or {}
        ).get("passed")
        is True
        for item in collectors
      ):
        reasons.append(
          f"round {record.get('round')} transactional restore differs"
        )
    if not (
      summary.get("bank_discovery_joint_balance_preflight", {}).get(
        "passed"
      )
      is True
      and summary.get("bank_joint_balance_preflight", {}).get("passed")
      is True
    ):
      reasons.append("final replay feasibility proof differs")
    table_counts = summary.get("training_tables", {})
    if table_counts != {
      "round_metrics_rows": 9,
      "candidate_metrics_rows": 24,
      "replay_metrics_rows": 16,
    }:
      reasons.append("training table row counts differ")
    for name in ("round_metrics.csv", "candidate_metrics.csv", "replay_metrics.csv"):
      if not (run_dir / name).is_file():
        reasons.append(f"missing training table {name}")
    source_manifest = summary.get("source_file_sha256")
    if not isinstance(source_manifest, dict) or not source_manifest:
      reasons.append("source manifest is missing")
    else:
      source_manifests.add(
        json.dumps(source_manifest, sort_keys=True, separators=(",", ":"))
      )
      for relative, expected_hash in source_manifest.items():
        path = repo / relative
        if not path.is_file() or _sha256(path) != expected_hash:
          reasons.append(f"training source differs: {relative}")
    if reasons:
      raise RuntimeError(f"v20 artifact invalid for {mode}/{seed}: {reasons}")
    initial_actor_hashes.add(summary["initial_actor_sha256"])
    checkpoints[seed] = checkpoint_path
    runs[seed] = {
      "summary_path": str(summary_path),
      "summary_sha256": _sha256(summary_path),
      "checkpoint_path": str(checkpoint_path),
      "checkpoint_sha256": _sha256(checkpoint_path),
      "initial_actor_sha256": summary["initial_actor_sha256"],
      "final_actor_sha256": summary["final_actor_sha256"],
      "round_count": len(rounds),
      "retained_update_count": accepted,
    }
  if len(initial_actor_hashes) != 1 or len(source_manifests) != 1:
    raise RuntimeError("v20 runs do not share one pi0 and source manifest")
  return checkpoints, {
    "same_base_policy_file_for_all_five_jobs": True,
    "same_initial_actor_for_all_five_jobs": True,
    "same_source_files_for_all_five_jobs": True,
    "initial_actor_sha256_values": sorted(initial_actor_hashes),
    "source_file_sha256": json.loads(next(iter(source_manifests))),
    "runs": runs,
  }


def _transition_class(old: dict[str, str], new: dict[str, str]) -> str:
  old_success = old["success"] == "True"
  new_success = new["success"] == "True"
  if not old_success and new_success:
    return "failure_to_success"
  if old_success and not new_success:
    return "success_to_failure"
  if old_success and new_success:
    return "success_to_success"
  return "failure_to_failure"


def _repair_summary(counts: dict[str, int]) -> dict[str, int]:
  normalized = {name: int(counts.get(name, 0)) for name in TRANSITION_CLASSES}
  repairs = normalized["failure_to_success"]
  regressions = normalized["success_to_failure"]
  return {
    **normalized,
    "repairs": repairs,
    "regressions": regressions,
    "net_success_change": repairs - regressions,
  }


def _pair_raw_rows(
  baseline: list[dict[str, str]], final: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
  """Pair by simulator replicate seed and initial environment identity."""
  key = lambda row: (int(row["evaluation_seed"]), int(row["environment_id"]))
  old = sorted(baseline, key=key)
  new = sorted(final, key=key)
  old_keys = [key(row) for row in old]
  new_keys = [key(row) for row in new]
  if len(old_keys) != len(set(old_keys)) or old_keys != new_keys:
    raise RuntimeError("v20 paired rows differ by seed/environment identity")
  return old, new


def _paired_binary_delta(
  baseline: list[dict[str, str]], final: list[dict[str, str]], field: str
):

  return _binary_column(final, field) - _binary_column(baseline, field)


def _paired_output_row(
  *,
  mode: str,
  role: str,
  seed: int,
  pair_index: int,
  baseline: dict[str, str],
  final: dict[str, str],
) -> dict[str, Any]:
  if (
    baseline["evaluation_seed"] != final["evaluation_seed"]
    or baseline["environment_id"] != final["environment_id"]
  ):
    raise RuntimeError("v20 paired output row identity differs")
  domain = mode if role == "target_diagonal_primary" else "D0"
  values: dict[str, Any] = {
    "specialist_mode": mode,
    "evaluation_mode": domain,
    "evaluation_role": role,
    "adaptation_seed": seed,
    "pair_index": pair_index,
    "evaluation_seed": int(baseline["evaluation_seed"]),
    "environment_id": int(baseline["environment_id"]),
    "baseline_success": int(baseline["success"] == "True"),
    "final_success": int(final["success"] == "True"),
    "baseline_fell": int(baseline["fell"] == "True"),
    "final_fell": int(final["fell"] == "True"),
    "transition_class": _transition_class(baseline, final),
    "baseline_failure_type": baseline["failure_type"],
    "final_failure_type": final["failure_type"],
  }
  source_fields = {
    "return": "return",
    "max_riser": "max_riser",
    "intervention_per_riser": "intervention_per_riser",
    "correction_mean": "correction_mean",
    "mean_abs_centerline_error": "mean_abs_centerline_error",
    "max_abs_heading_error": "max_abs_heading_error",
    "max_left_slip_speed": "maximum_left_contact_slip_speed",
    "max_right_slip_speed": "maximum_right_contact_slip_speed",
    "contact_mismatch_fraction": "contact_mismatch_fraction",
    "max_roll_signal": "maximum_roll_signal",
    "max_pitch_signal": "maximum_pitch_signal",
    "max_angular_velocity_signal": "maximum_angular_velocity_signal",
  }
  for output_name, source_name in source_fields.items():
    values[f"baseline_{output_name}"] = baseline[source_name]
    values[f"final_{output_name}"] = final[source_name]
  return values


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  _validate_protocol(protocol, args)
  if any(
    count % args.eval_batch_size
    for count in (args.target_episodes, args.d0_episodes)
  ):
    raise ValueError("v20 audit counts must divide into full paired batches")
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.audit_commit:
    raise RuntimeError("v20 audit HEAD differs from the frozen audit commit")
  tracked_clean = _tracked_worktree_is_clean(repo)
  if not args.smoke and not tracked_clean:
    raise RuntimeError("formal v20 audit requires a clean tracked worktree")
  audit_amendment = _validate_audit_amendment(
    repo=repo,
    protocol_path=protocol_path,
    protocol_commit=args.protocol_commit,
    audit_commit=current_commit,
    amendment_path=args.audit_amendment.resolve(),
  )

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import (
    configure_v19_observable_refinement_runner,
  )
  from src.tasks.stairs_cbf.deployment_context import (
    configure_v19_actor_interface,
    load_calibrated_v19_context,
  )

  mode = args.mode
  checkpoint = args.base_policy_checkpoint.resolve()
  context_path = args.context.resolve()
  training_root = args.training_root.resolve()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  sealed = protocol["sealed_inputs"]
  if _sha256(checkpoint) != sealed["base_policy_checkpoint_sha256"]:
    raise RuntimeError("base checkpoint differs from the v20 protocol")
  context = load_calibrated_v19_context(context_path)
  if context["specialist_mode"] != mode:
    raise RuntimeError("v20 audit context mode differs")
  declared_context = sealed["contexts"][mode]
  if not (
    _sha256(context_path) == declared_context["file_sha256"]
    and context["parameters_sha256"]
    == declared_context["parameters_sha256"]
    and context["calibration"]["selected_candidate_seed"]
    == declared_context["selected_calibration_seed"]
  ):
    raise RuntimeError("v20 audit context differs from its seal")
  calibration_checks = _validate_context(mode, context, protocol)
  seeds = list(args.adaptation_seeds)
  checkpoints, isolation = _validate_training_artifacts(
    repo=repo,
    training_root=training_root,
    mode=mode,
    context=context,
    seeds=seeds,
    base_checkpoint_sha256=_sha256(checkpoint),
    protocol_commit=args.protocol_commit,
    protocol_sha256=_sha256(protocol_path),
  )

  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  configure_v19_actor_interface(env_cfg, context)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.audit_seed
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  # The sealed base checkpoint predates the five-column observable append and
  # carries a legacy retention-actor payload. v20 is Brief PPO and never uses
  # that auxiliary state. Match the training loader semantics so only the
  # actor/critic are expanded; no checkpoint tensor or evaluated actor changes.
  alg_cfg = agent_cfg.algorithm
  alg_cfg.brief_ppo_refinement = True
  alg_cfg.failure_focused_refinement = True
  alg_cfg.observable_failure_conditioned_refinement = True
  alg_cfg.task_first_constrained = False
  alg_cfg.d0_retention_anchor_weight = 0.0
  alg_cfg.neighbor_retention_anchor_weight = 0.0
  alg_cfg.actor_new_feature_count = 5
  alg_cfg.actor_new_feature_learning_rate_multiplier = 1.0
  alg_cfg.freeze_legacy_actor_input_columns = True
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v20 audit task has no online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(
    str(checkpoint), map_location=args.device
  )
  if warm_start.get("pi0_exact_preservation_proof") is not True:
    raise RuntimeError("v20 audit pi0 expansion proof failed")
  base_actor = _actor_state(runner.alg.actor)
  if _actor_state_sha256(base_actor) != isolation[
    "initial_actor_sha256_values"
  ][0]:
    raise RuntimeError("v20 audit pi0 differs from training")

  paired_rows: list[dict[str, Any]] = []
  raw: dict[str, Any] = {}
  target_success_groups = []
  target_fall_groups = []
  d0_success_groups = []
  d0_fall_groups = []
  per_seed: dict[str, Any] = {}
  transition_counts = {name: 0 for name in TRANSITION_CLASSES}
  transition_counts_by_seed = {
    str(seed): {name: 0 for name in TRANSITION_CLASSES} for seed in seeds
  }
  mode_offset = 0 if mode == "lateral" else 10_000
  try:
    for seed_index, adaptation_seed in enumerate(seeds):
      runner.load_online_checkpoint(
        str(checkpoints[adaptation_seed]), map_location=args.device
      )
      final_actor = _actor_state(runner.alg.actor)
      expected_actor = isolation["runs"][adaptation_seed][
        "final_actor_sha256"
      ]
      if _actor_state_sha256(final_actor) != expected_actor:
        raise RuntimeError(f"loaded final actor differs for {mode}/{adaptation_seed}")
      raw[str(adaptation_seed)] = {}
      per_seed[str(adaptation_seed)] = {}
      for role in ("target_diagonal_primary", "d0_sanity"):
        target = role == "target_diagonal_primary"
        domain = "DQHMED" if target else "D0"
        episode_count = args.target_episodes if target else args.d0_episodes
        repeats = episode_count // args.eval_batch_size
        evaluation_seed = (
          args.audit_seed + mode_offset + 1_000 * seed_index
          if target
          else args.audit_seed
          + 90_000
          + mode_offset
          + 1_000 * seed_index
        )
        baseline_root = (
          output_dir
          / "raw"
          / "baseline"
          / f"seed{adaptation_seed}"
          / ("target" if target else "D0")
        )
        final_root = (
          output_dir
          / "raw"
          / "final"
          / f"seed{adaptation_seed}"
          / ("target" if target else "D0")
        )
        common = {
          "domains": (domain,),
          "num_envs": args.eval_batch_size,
          "num_episodes": args.eval_batch_size,
          "seed": evaluation_seed,
          "repeats": repeats,
          "device": args.device,
          "runtime_filter": True,
          "resume": True,
          "v19_context": context_path,
        }
        baseline_eval = _evaluate_state(
          runner,
          base_actor,
          artifact_dir=baseline_root,
          deployment_context=context_path if target else None,
          **common,
        )[domain]
        final_eval = _evaluate_state(
          runner,
          final_actor,
          artifact_dir=final_root,
          deployment_context=context_path if target else None,
          **common,
        )[domain]
        if baseline_eval["initial_state_signatures"] != final_eval[
          "initial_state_signatures"
        ]:
          raise RuntimeError(
            f"paired signatures differ for {mode}/{adaptation_seed}/{domain}"
          )
        baseline_rows = _load_rows(
          baseline_root,
          domain=domain,
          first_seed=evaluation_seed,
          repeats=repeats,
        )
        final_rows = _load_rows(
          final_root,
          domain=domain,
          first_seed=evaluation_seed,
          repeats=repeats,
        )
        if len(baseline_rows) != episode_count or len(final_rows) != episode_count:
          raise RuntimeError("fresh v20 audit raw row count differs")
        baseline_rows, final_rows = _pair_raw_rows(
          baseline_rows, final_rows
        )
        success_delta = _paired_binary_delta(
          baseline_rows, final_rows, "success"
        )
        fall_delta = _paired_binary_delta(
          baseline_rows, final_rows, "fell"
        )
        summary_key = "target" if target else "D0"
        per_seed[str(adaptation_seed)][summary_key] = {
          "baseline_success_rate": float(
            _binary_column(baseline_rows, "success").mean()
          ),
          "final_success_rate": float(
            _binary_column(final_rows, "success").mean()
          ),
          "paired_success_delta": float(success_delta.mean()),
          "baseline_fall_rate": float(
            _binary_column(baseline_rows, "fell").mean()
          ),
          "final_fall_rate": float(
            _binary_column(final_rows, "fell").mean()
          ),
          "paired_fall_delta": float(fall_delta.mean()),
        }
        if target:
          target_success_groups.append(success_delta)
          target_fall_groups.append(fall_delta)
        else:
          d0_success_groups.append(success_delta)
          d0_fall_groups.append(fall_delta)
        for pair_index, (old, new) in enumerate(
          zip(baseline_rows, final_rows, strict=True)
        ):
          row = _paired_output_row(
            mode=mode,
            role=role,
            seed=adaptation_seed,
            pair_index=pair_index,
            baseline=old,
            final=new,
          )
          paired_rows.append(row)
          if target:
            transition_counts[row["transition_class"]] += 1
            transition_counts_by_seed[str(adaptation_seed)][
              row["transition_class"]
            ] += 1
        raw[str(adaptation_seed)][role] = {
          "episode_count": episode_count,
          "evaluation_seed_start": evaluation_seed,
          "evaluation_seeds": baseline_eval["seeds"],
          "baseline": baseline_eval,
          "final": final_eval,
        }
  finally:
    env.close()

  paired_csv = output_dir / "paired_episode_metrics.csv"
  temporary_csv = output_dir / ".paired_episode_metrics.csv.tmp"
  with temporary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=PAIRED_FIELDS)
    writer.writeheader()
    writer.writerows(paired_rows)
  temporary_csv.replace(paired_csv)
  expected_rows = len(seeds) * (
    args.target_episodes + args.d0_episodes
  )
  if len(paired_rows) != expected_rows:
    raise RuntimeError("v20 paired CSV row count differs")

  offset = 0 if mode == "lateral" else 100
  intervals = {
    "target_success": hierarchical_paired_scene_interval_v19(
      target_success_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + offset,
    ),
    "target_fall": hierarchical_paired_scene_interval_v19(
      target_fall_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + offset + 1,
    ),
    "d0_success": hierarchical_paired_scene_interval_v19(
      d0_success_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + offset + 2,
    ),
    "d0_fall": hierarchical_paired_scene_interval_v19(
      d0_fall_groups,
      bootstrap_samples=args.bootstrap_samples,
      bootstrap_seed=args.bootstrap_seed + offset + 3,
    ),
  }
  per_seed_deltas = [
    per_seed[str(seed)]["target"]["paired_success_delta"] for seed in seeds
  ]
  gate = independent_diagonal_scene_gate_v19(
    diagonal_success_delta=intervals["target_success"][0],
    per_seed_success_deltas=per_seed_deltas,
    diagonal_fall_delta=intervals["target_fall"][0],
    d0_success_delta=intervals["d0_success"][0],
  )
  claim = {
    "question": f"Does the sealed {mode} v20 specialist improve its context?",
    "claim_passed": gate["passed"],
    "gate": gate,
    "strong_evidence_lcb95_positive": intervals["target_success"][1] > 0.0,
    "target": {
      "paired_success_delta_mean_lcb95_ucb95": intervals[
        "target_success"
      ],
      "paired_fall_delta_mean_lcb95_ucb95": intervals["target_fall"],
      "confidence_interval_is_report_only": True,
    },
    "D0_sanity": {
      "paired_success_delta_mean_lcb95_ucb95": intervals["d0_success"],
      "paired_fall_delta_mean_lcb95_ucb95": intervals["d0_fall"],
    },
    "per_adaptation_seed": per_seed,
  }
  repair_summary = _repair_summary(transition_counts)
  repair_summary["per_adaptation_seed"] = {
    seed: _repair_summary(counts)
    for seed, counts in transition_counts_by_seed.items()
  }
  result = {
    "protocol_id": PROTOCOL_ID,
    "analysis_version": (
      "v20 independent fixed-budget diagonal audit v2 "
      "(audit-infrastructure amendment 1)"
    ),
    "policy_method": POLICY_METHOD,
    "formal_protocol": not args.smoke,
    "evidence_role": "fresh paired audit never used by training gates",
    "specialist_mode": mode,
    "protocol_file": {
      "path": str(protocol_path),
      "sha256": _sha256(protocol_path),
      "git_commit": args.protocol_commit,
      "tracked_worktree_and_index_clean": tracked_clean,
    },
    "audit_implementation": audit_amendment,
    "audit_loader_configuration": {
      "brief_ppo_refinement": True,
      "failure_focused_refinement": True,
      "observable_failure_conditioned_refinement": True,
      "task_first_constrained": False,
      "legacy_constraint_payload_ignored": True,
      "actor_or_checkpoint_tensor_modified": False,
    },
    "runtime_cbf": True,
    "adaptation_seeds": seeds,
    "audit_seed": args.audit_seed,
    "bootstrap_seed": args.bootstrap_seed,
    "context": {
      "path": str(context_path),
      "file_sha256": _sha256(context_path),
      "parameters_sha256": context["parameters_sha256"],
      "calibration_checks": calibration_checks,
      "calibration": context["calibration"],
    },
    "training_isolation": isolation,
    "evaluation_protocol": {
      "target_paired_episodes_per_adaptation_seed": args.target_episodes,
      "d0_paired_episodes_per_adaptation_seed": args.d0_episodes,
      "off_diagonal_evaluation_performed": False,
      "macro_average_computed": False,
      "filter_free_evaluation_performed": False,
      "cbf_independence_gate_used": False,
      "joint_two_specialist_claim_defined": False,
      "confidence_interval_used_as_gate": False,
      "lcb95_positive_reported_as_strong_evidence_only": True,
      "paired_baseline_and_final_simulator_randomness": True,
    },
    "independent_claim": claim,
    "repairs_regressions": repair_summary,
    "paired_episode_metrics": {
      "path": str(paired_csv),
      "sha256": _sha256(paired_csv),
      "row_count": len(paired_rows),
      "schema": PAIRED_FIELDS,
    },
    "raw_evaluations": raw,
  }
  output_path = output_dir / "final_audit_summary.json"
  temporary = output_dir / ".final_audit_summary.json.tmp"
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary.replace(output_path)
  print(
    json.dumps(
      {
        "output_path": str(output_path),
        "paired_csv": result["paired_episode_metrics"],
        "independent_claim": claim,
        "repairs_regressions": result["repairs_regressions"],
      },
      indent=2,
      sort_keys=True,
    )
  )


if __name__ == "__main__":
  main()
