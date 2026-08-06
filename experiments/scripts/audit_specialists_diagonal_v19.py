"""Prospective two-diagonal audit for the ten sealed v19 specialist actors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

from diagonal_audit_stats import (
  hierarchical_paired_scene_interval_v19,
  independent_diagonal_scene_gate_v19,
)
from online_refine_stairs import _actor_state, _actor_state_sha256, _evaluate_state


PROTOCOL_ID = "safe100-observable-failure-conditioned-v19"
MODES = ("lateral", "contact_stability")
FORMAL_ADAPTATION_SEEDS = [43, 143, 243, 343, 443]
FORMAL_AUDIT_SEED = 5_100_000
FORMAL_BOOTSTRAP_SEED = 6_000_000


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _git_output(repo: Path, *arguments: str) -> str:
  return subprocess.run(
    ["git", *arguments],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _tracked_worktree_is_clean(repo: Path) -> bool:
  return (
    subprocess.run(["git", "diff", "--quiet"], cwd=repo, check=False).returncode
    == 0
    and subprocess.run(
      ["git", "diff", "--cached", "--quiet"], cwd=repo, check=False
    ).returncode
    == 0
  )


def _load_rows(
  root: Path, *, domain: str, first_seed: int, repeats: int
) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  for repeat in range(repeats):
    with (root / f"{domain}-seed{first_seed + repeat}.csv").open(
      newline=""
    ) as handle:
      rows.extend(csv.DictReader(handle))
  return rows


def _binary_column(rows: list[dict[str, str]], field: str) -> torch.Tensor:
  return torch.tensor([row[field] == "True" for row in rows], dtype=torch.float64)


def _paired_delta(
  baseline: list[dict[str, str]], final: list[dict[str, str]], field: str
) -> torch.Tensor:
  if len(baseline) != len(final):
    raise ValueError("paired audit row counts differ")
  for index, (old, new) in enumerate(zip(baseline, final, strict=True)):
    if old["episode"] != new["episode"]:
      raise ValueError(f"paired episode index differs at row {index}")
  return _binary_column(final, field) - _binary_column(baseline, field)


def _balanced_restart_strata(audit: dict[str, Any]) -> bool:
  marginals = audit.get("primary_marginal_counts") or {}
  expected = {
    "lateral": {
      "centerline_sign": {"-1", "1"},
      "heading_sign": {"-1", "1"},
      "riser_stage": {"early", "mid", "late"},
      "support_foot": {"0", "1"},
      "error_growth_bin": {"low", "high"},
    },
    "contact_stability": {
      "touchdown_foot": {"0", "1"},
      "slip_foot": {"0", "1"},
      "contact_timing": {"early", "delayed"},
      "support_foot": {"0", "1"},
    },
  }.get(audit.get("specialist_mode"))
  pair_count = audit.get("pair_count")
  return (
    bool(expected)
    and audit.get("quota_solver")
    in {"heuristic_multi_start", "exact_search_fallback"}
    and audit.get("maximum_marginal_imbalance") <= 1
    and set(marginals) == set(expected)
    and all(
      set(marginals[field]) == values
      and sum(marginals[field].values()) == pair_count
      and max(marginals[field].values()) - min(marginals[field].values()) <= 1
      for field, values in expected.items()
    )
  )


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--context-dir", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--adaptation-seeds", nargs="+", type=int, default=FORMAL_ADAPTATION_SEEDS
  )
  parser.add_argument("--eval-batch-size", type=int, default=128)
  parser.add_argument("--target-episodes", type=int, default=512)
  parser.add_argument("--d0-episodes", type=int, default=256)
  parser.add_argument("--bootstrap-samples", type=int, default=10000)
  parser.add_argument("--audit-seed", type=int, default=FORMAL_AUDIT_SEED)
  parser.add_argument("--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _validate_protocol(protocol: dict[str, Any], args: argparse.Namespace) -> None:
  expected = {
    "protocol_id": PROTOCOL_ID,
    "protocol_revision": 2,
    "status": "refrozen_after_revision1_base_only_calibration_failure_before_adaptation",
    "policy_method": "Observable Failure-Conditioned Brief PPO v19",
    "specialist_modes": list(MODES),
    "adaptation_seeds": FORMAL_ADAPTATION_SEEDS,
  }
  mismatches = {
    key: {"actual": protocol.get(key), "required": value}
    for key, value in expected.items()
    if protocol.get(key) != value
  }
  evaluation = protocol.get("evaluation", {})
  expected_evaluation = {
    "audit_seed": FORMAL_AUDIT_SEED,
    "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
    "eval_batch_size": 128,
    "target_paired_episodes_per_adaptation_seed": 512,
    "d0_paired_episodes_per_adaptation_seed": 256,
    "bootstrap_samples": 10000,
    "off_diagonal_evaluation": False,
    "macro_average_computed": False,
    "filter_free_evaluation": False,
    "cbf_independence_gate": False,
  }
  for key, value in expected_evaluation.items():
    if evaluation.get(key) != value:
      mismatches[f"evaluation.{key}"] = {
        "actual": evaluation.get(key),
        "required": value,
      }
  acceptance = protocol.get("independent_specialist_acceptance", {})
  expected_acceptance = {
    "mean_diagonal_success_delta_strictly_above": 0.0,
    "minimum_positive_adaptation_seed_count": 4,
    "maximum_diagonal_fall_increase": 0.03,
    "minimum_d0_success_delta": -0.05,
    "confidence_interval_is_a_gate": False,
    "lcb95_positive_is_strong_evidence_only": True,
    "joint_two_specialist_claim_defined": False,
  }
  for key, value in expected_acceptance.items():
    if acceptance.get(key) != value:
      mismatches[f"independent_specialist_acceptance.{key}"] = {
        "actual": acceptance.get(key),
        "required": value,
      }
  calibration = protocol.get("calibration", {})
  expected_calibration = {
    "adapted_policy_evaluations_used": False,
    "base_policy_only": True,
    "lateral_candidate_seeds": [
      7312,
      7313,
      7314,
      7315,
      7316,
      7317,
      7318,
      7319,
    ],
    "lateral_evaluation_seed_base": 4_720_000,
    "contact_stability_candidate_seeds": [7217, 7218, 7219],
    "contact_stability_evaluation_seed_base": 4_710_000,
    "episodes_per_candidate": 512,
    "eval_batch_size": 128,
    "minimum_fall_count": 100,
    "success_rate_bounds_inclusive": [0.70, 0.85],
    "first_qualifying_candidate_is_frozen": True,
    "lateral_minimum_target_failure_fraction": 0.80,
    "lateral_maximum_second_failure_fraction": 0.30,
    "contact_stability_minimum_target_failure_fraction": 0.75,
    "contact_stability_maximum_second_failure_fraction": 0.20,
  }
  for key, value in expected_calibration.items():
    if calibration.get(key) != value:
      mismatches[f"calibration.{key}"] = {
        "actual": calibration.get(key),
        "required": value,
      }
  if mismatches:
    raise ValueError(f"diagonal v19 protocol file mismatch: {mismatches}")
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
  required = {
    "adaptation_seeds": FORMAL_ADAPTATION_SEEDS,
    **{
      key: evaluation[key]
      for key in runtime
      if key != "adaptation_seeds"
    },
  }
  runtime_mismatches = {
    key: {"actual": value, "required": required[key]}
    for key, value in runtime.items()
    if value != required[key]
  }
  if runtime_mismatches:
    raise ValueError(f"formal diagonal v19 runtime mismatch: {runtime_mismatches}")


def _validate_context(
  mode: str, context: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
  calibration = context["calibration"]
  declared_candidates = protocol["calibration"][f"{mode}_candidate_seeds"]
  attempts = calibration.get("attempts", [])
  selected_seed = calibration.get("selected_candidate_seed")
  selected_index = declared_candidates.index(selected_seed)
  if [attempt["candidate_seed"] for attempt in attempts] != declared_candidates[
    : selected_index + 1
  ]:
    raise RuntimeError(f"{mode} calibration did not follow first-qualifying order")
  if any(attempt.get("qualifies") for attempt in attempts[:-1]) or not attempts[-1].get(
    "qualifies"
  ):
    raise RuntimeError(f"{mode} calibration did not select the first qualifier")
  selected = attempts[-1]
  minimum_purity = 0.80 if mode == "lateral" else 0.75
  maximum_second = 0.30 if mode == "lateral" else 0.20
  checks = {
    "base_policy_only": selected.get("base_policy_only") is True,
    "episodes_512": selected.get("num_episodes") == 512,
    "success_rate_70_to_85_percent": 0.70 <= selected["success_rate"] <= 0.85,
    "at_least_100_falls": selected["fall_count"] >= 100,
    "target_failure_purity": selected["target_failure_fraction"] >= minimum_purity,
    "second_failure_bound": selected["second_failure_fraction"] <= maximum_second,
    "selected_hash": selected["parameters_sha256"] == context["parameters_sha256"],
  }
  if not all(checks.values()):
    raise RuntimeError(f"frozen {mode} calibration no longer passes: {checks}")
  return checks


def _validate_training_artifacts(
  *,
  repo: Path,
  training_root: Path,
  contexts: dict[str, dict[str, Any]],
  seeds: list[int],
  base_checkpoint_sha256: str,
) -> tuple[dict[str, dict[int, Path]], dict[str, Any]]:
  checkpoints: dict[str, dict[int, Path]] = {mode: {} for mode in MODES}
  runs: dict[str, dict[int, Any]] = {mode: {} for mode in MODES}
  initial_actor_hashes: set[str] = set()
  source_manifests: set[str] = set()
  for mode in MODES:
    for seed in seeds:
      run_dir = training_root / mode / f"seed{seed}"
      summary_path = run_dir / "specialist_summary.json"
      checkpoint_path = run_dir / "accepted_final.pt"
      if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing v19 training artifact in {run_dir}")
      summary = json.loads(summary_path.read_text())
      reasons: list[str] = []
      if summary.get("method") != "Observable Failure-Conditioned Brief PPO v19":
        reasons.append("method identity differs")
      if summary.get("formal_protocol") is not True or summary.get(
        "protocol_completed"
      ) is not True:
        reasons.append("formal training protocol did not complete")
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
      if summary.get("deployment_context", {}).get(
        "parameters_sha256"
      ) != contexts[mode]["parameters_sha256"]:
        reasons.append("deployment context hash differs")
      expansion = summary.get("actor_observation_expansion", {})
      if not (
        expansion.get("legacy_width") == 405
        and expansion.get("v19_width") == 410
        and expansion.get("new_feature_count") == 5
        and expansion.get("pre_adaptation_policy_exactly_preserved") is True
        and expansion.get("legacy_input_columns_frozen_during_adaptation")
        is True
        and expansion.get("legacy_first_layer_input_column_change_max_abs")
        == 0.0
        and expansion.get("new_input_columns_use_full_actor_learning_rate")
        is True
        and expansion.get("new_first_layer_column_max_abs_after_adaptation", 0.0)
        > 0.0
      ):
        reasons.append("observable input-adapter proof differs")
      structural = summary.get("structural_checks", {})
      if not (
        structural.get("actor_new_feature_count") == 5
        and math.isclose(
          structural.get("actor_new_feature_learning_rate_multiplier", math.nan),
          1.0,
        )
        and structural.get("freeze_legacy_actor_input_columns") is True
        and structural.get("actor_new_feature_optimizer_group_count") == 1
        and math.isclose(
          structural.get("actor_new_feature_optimizer_learning_rate", math.nan),
          5.0e-6,
        )
        and structural.get(
          "actor_new_feature_optimizer_owns_first_layer_weight"
        )
        is True
      ):
        reasons.append("full-rate observable input adapter differs")
      rollout = summary.get("rollout_per_round", {})
      if rollout != {
        "batch_count": 2,
        "environments_per_batch": 64,
        "steps_per_environment": 1024,
        "same_frozen_behavior_policy": True,
        "different_rollout_seeds": True,
        "loss_and_gradient_combination": "arithmetic mean",
        "gradient_cosine_is_diagnostic_only": True,
      }:
        reasons.append("dual-rollout protocol differs")
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
        reasons.append("two-stage 320-episode candidate selection differs")
      round_protocol = summary.get("round_protocol", {})
      rounds = summary.get("rounds", [])
      if not (
        round_protocol.get("maximum_rounds") == 8
        and round_protocol.get("minimum_accepted_updates") == 3
        and round_protocol.get("rejection_patience_after_minimum") == 2
        and 3 <= round_protocol.get("accepted_updates", -1) <= len(rounds) <= 8
      ):
        reasons.append("round/accepted-update protocol differs")
      for record in rounds:
        metrics = record.get("full_update_metrics", {})
        input_adapter = record.get("input_adapter_update", {})
        if not (
          metrics.get("dual_rollout_batch_count") == 2
          and metrics.get("dual_rollout_gradient_cosine_is_gate") is False
          and metrics.get("v19_normal_advantage_count") == 81920.0
          and metrics.get("v19_failure_advantage_count") == 24576.0
          and metrics.get("v19_success_advantage_count") == 24576.0
          and len(record.get("dual_rollout_seeds", [])) == 2
          and len(set(record.get("dual_rollout_seeds", []))) == 2
        ):
          reasons.append(f"round {record.get('round')} dual PPO invariant differs")
        if not (
          input_adapter.get("legacy_input_column_change_max_abs") == 0.0
          and input_adapter.get("new_input_column_change_max_abs", 0.0) > 0.0
          and input_adapter.get("new_input_column_max_abs", 0.0) > 0.0
        ):
          reasons.append(
            f"round {record.get('round')} observable input update differs"
          )
        collector_metrics = metrics.get("collector_metrics", [])
        if not (
          len(collector_metrics) == 2
          and all(
            item.get("matched_pair_sampling") is True
            and (item.get("matched_pair_audit") or {}).get(
              "exact_match_passed"
            )
            is True
            and (item.get("matched_pair_audit") or {}).get("pair_count") == 12
            and _balanced_restart_strata(item.get("matched_pair_audit") or {})
            for item in collector_metrics
          )
        ):
          reasons.append(
            f"round {record.get('round')} balanced matched restart pairs differ"
          )
      failure = summary.get("failure_bank", {})
      success_pool = summary.get("success_pool", {})
      success = summary.get("success_counterexample_bank", {})
      if failure.get("outcome_counts") != {"failure": failure.get("size")}:
        reasons.append("failure bank outcome purity failed")
      if success_pool.get("outcome_counts") != {"success": success_pool.get("size")}:
        reasons.append("success pool outcome purity failed")
      if success.get("matched_entry_count") != success.get("size"):
        reasons.append("success counterexample matching is incomplete")
      if mode == "lateral":
        if not (
          set(failure.get("centerline_sign_counts", {})) == {"-1", "1"}
          and set(failure.get("heading_sign_counts", {})) == {"-1", "1"}
          and {"early", "mid", "late"}.issubset(
            failure.get("riser_stage_counts", {})
          )
          and set(failure.get("support_foot_counts", {})) == {"0", "1"}
          and {"low", "high"}.issubset(
            failure.get("error_growth_bin_counts", {})
          )
        ):
          reasons.append("lateral bank diversity is incomplete")
      elif not (
        set(failure.get("touchdown_foot_counts", {})) == {"0", "1"}
        and set(failure.get("slip_foot_counts", {})) == {"0", "1"}
        and set(failure.get("contact_timing_counts", {})) == {"early", "delayed"}
        and set(failure.get("support_foot_counts", {})) == {"0", "1"}
      ):
        reasons.append("contact bank diversity is incomplete")
      source_manifest = summary.get("source_file_sha256")
      if not isinstance(source_manifest, dict) or not source_manifest:
        reasons.append("source-file manifest is missing")
      else:
        source_manifests.add(
          json.dumps(source_manifest, sort_keys=True, separators=(",", ":"))
        )
        for relative, expected_hash in source_manifest.items():
          source_path = repo / relative
          if not source_path.is_file() or _sha256(source_path) != expected_hash:
            reasons.append(f"training source differs: {relative}")
      if reasons:
        raise RuntimeError(f"v19 artifact invariant failed for {mode}/{seed}: {reasons}")
      initial_actor_hashes.add(summary["initial_actor_sha256"])
      checkpoints[mode][seed] = checkpoint_path
      runs[mode][seed] = {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "initial_actor_sha256": summary["initial_actor_sha256"],
        "final_actor_sha256": summary["final_actor_sha256"],
        "round_count": len(rounds),
        "accepted_update_count": round_protocol["accepted_updates"],
      }
  if len(initial_actor_hashes) != 1 or len(source_manifests) != 1:
    raise RuntimeError("v19 runs did not share one expanded pi0 and source manifest")
  return checkpoints, {
    "same_base_policy_file_for_all_ten_jobs": True,
    "same_initial_actor_for_all_ten_jobs": True,
    "same_source_files_for_all_ten_jobs": True,
    "initial_actor_sha256_values": sorted(initial_actor_hashes),
    "source_file_sha256": json.loads(next(iter(source_manifests))),
    "runs": runs,
  }


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  _validate_protocol(protocol, args)
  if any(count % args.eval_batch_size for count in (args.target_episodes, args.d0_episodes)):
    raise ValueError("audit episode counts must divide into full paired batches")
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.protocol_commit:
    raise RuntimeError(
      f"audit HEAD {current_commit} differs from protocol commit {args.protocol_commit}"
    )
  tracked_clean = _tracked_worktree_is_clean(repo)
  if not args.smoke and not tracked_clean:
    raise RuntimeError("formal audit requires a clean tracked worktree and index")

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.config import configure_v19_observable_refinement_runner
  from src.tasks.stairs_cbf.deployment_context import (
    configure_v19_actor_interface,
    load_calibrated_v19_context,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  context_dir = args.context_dir.resolve()
  training_root = args.training_root.resolve()
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  if _sha256(checkpoint) != protocol["sealed_inputs"][
    "base_policy_checkpoint_sha256"
  ]:
    raise RuntimeError("base checkpoint hash differs from frozen v19 protocol")
  contexts: dict[str, dict[str, Any]] = {}
  context_paths: dict[str, Path] = {}
  calibration_checks: dict[str, Any] = {}
  for mode in MODES:
    path = context_dir / f"{mode}.json"
    context = load_calibrated_v19_context(path)
    if context["specialist_mode"] != mode:
      raise RuntimeError(f"{mode} context has another specialist mode")
    contexts[mode] = context
    context_paths[mode] = path
    calibration_checks[mode] = _validate_context(mode, context, protocol)

  checkpoints, isolation = _validate_training_artifacts(
    repo=repo,
    training_root=training_root,
    contexts=contexts,
    seeds=list(args.adaptation_seeds),
    base_checkpoint_sha256=_sha256(checkpoint),
  )

  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  configure_v19_actor_interface(env_cfg, contexts["lateral"])
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.audit_seed
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v19 audit task has no online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  if warm_start.get("pi0_exact_preservation_proof") is not True:
    raise RuntimeError("audit pi0 expansion proof failed")
  base_actor = _actor_state(runner.alg.actor)
  base_actor_sha256 = _actor_state_sha256(base_actor)
  if base_actor_sha256 != isolation["initial_actor_sha256_values"][0]:
    raise RuntimeError("audit expanded pi0 differs from every training run")

  seeds = list(args.adaptation_seeds)
  rows: dict[str, dict[int, dict[str, dict[str, list[dict[str, str]]]]]] = {
    mode: {} for mode in MODES
  }
  raw: dict[str, Any] = {mode: {} for mode in MODES}
  for mode_index, mode in enumerate(MODES):
    for seed_index, adaptation_seed in enumerate(seeds):
      runner.load_online_checkpoint(
        str(checkpoints[mode][adaptation_seed]), map_location=args.device
      )
      final_actor = _actor_state(runner.alg.actor)
      expected_actor = isolation["runs"][mode][adaptation_seed]["final_actor_sha256"]
      if _actor_state_sha256(final_actor) != expected_actor:
        raise RuntimeError(f"loaded final actor differs for {mode}/{adaptation_seed}")
      rows[mode][adaptation_seed] = {}
      raw[mode][str(adaptation_seed)] = {}
      for role in ("target_diagonal_primary", "d0_sanity"):
        target = role == "target_diagonal_primary"
        domain = "DQHMED" if target else "D0"
        episode_count = args.target_episodes if target else args.d0_episodes
        repeats = episode_count // args.eval_batch_size
        evaluation_seed = (
          args.audit_seed + 10_000 * mode_index + 1_000 * seed_index
          if target
          else args.audit_seed + 90_000 + 10_000 * mode_index + 1_000 * seed_index
        )
        context_path = context_paths[mode]
        baseline_root = (
          output_dir
          / "raw"
          / "baseline"
          / mode
          / f"seed{adaptation_seed}"
          / ("target" if target else "D0")
        )
        final_root = (
          output_dir
          / "raw"
          / mode
          / f"seed{adaptation_seed}"
          / ("target" if target else "D0")
          / "final"
        )
        common = dict(
          domains=(domain,),
          num_envs=args.eval_batch_size,
          num_episodes=args.eval_batch_size,
          seed=evaluation_seed,
          repeats=repeats,
          device=args.device,
          runtime_filter=True,
          resume=True,
          v19_context=context_path,
        )
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
          raise RuntimeError(f"paired signatures differ for {mode}/{adaptation_seed}/{domain}")
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
          raise RuntimeError("fresh v19 audit raw row count differs")
        rows[mode][adaptation_seed][role] = {
          "baseline": baseline_rows,
          "final": final_rows,
        }
        raw[mode][str(adaptation_seed)][role] = {
          "episode_count": episode_count,
          "evaluation_seed_start": evaluation_seed,
          "evaluation_seeds": baseline_eval["seeds"],
          "baseline": baseline_eval,
          "final": final_eval,
        }

  paired_csv = output_dir / "paired_episode_metrics.csv"
  fieldnames = [
    "specialist_mode",
    "evaluation_mode",
    "evaluation_role",
    "adaptation_seed",
    "pair_index",
    "baseline_success",
    "final_success",
    "baseline_fell",
    "final_fell",
    "baseline_failure_type",
    "final_failure_type",
  ]
  with paired_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for mode in MODES:
      for adaptation_seed in seeds:
        for role in ("target_diagonal_primary", "d0_sanity"):
          pair = rows[mode][adaptation_seed][role]
          for pair_index, (baseline, final) in enumerate(
            zip(pair["baseline"], pair["final"], strict=True)
          ):
            writer.writerow(
              {
                "specialist_mode": mode,
                "evaluation_mode": mode if role == "target_diagonal_primary" else "D0",
                "evaluation_role": role,
                "adaptation_seed": adaptation_seed,
                "pair_index": pair_index,
                "baseline_success": int(baseline["success"] == "True"),
                "final_success": int(final["success"] == "True"),
                "baseline_fell": int(baseline["fell"] == "True"),
                "final_fell": int(final["fell"] == "True"),
                "baseline_failure_type": baseline["failure_type"],
                "final_failure_type": final["failure_type"],
              }
            )

  claims: dict[str, Any] = {}
  for mode_index, mode in enumerate(MODES):
    per_seed: dict[str, Any] = {}
    target_success_groups: list[torch.Tensor] = []
    target_fall_groups: list[torch.Tensor] = []
    d0_success_groups: list[torch.Tensor] = []
    d0_fall_groups: list[torch.Tensor] = []
    for seed in seeds:
      target_pair = rows[mode][seed]["target_diagonal_primary"]
      d0_pair = rows[mode][seed]["d0_sanity"]
      target_success = _paired_delta(
        target_pair["baseline"], target_pair["final"], "success"
      )
      target_fall = _paired_delta(target_pair["baseline"], target_pair["final"], "fell")
      d0_success = _paired_delta(d0_pair["baseline"], d0_pair["final"], "success")
      d0_fall = _paired_delta(d0_pair["baseline"], d0_pair["final"], "fell")
      target_success_groups.append(target_success)
      target_fall_groups.append(target_fall)
      d0_success_groups.append(d0_success)
      d0_fall_groups.append(d0_fall)
      per_seed[str(seed)] = {
        "target": {
          "baseline_success_rate": float(_binary_column(target_pair["baseline"], "success").mean()),
          "final_success_rate": float(_binary_column(target_pair["final"], "success").mean()),
          "paired_success_delta": float(target_success.mean()),
          "baseline_fall_rate": float(_binary_column(target_pair["baseline"], "fell").mean()),
          "final_fall_rate": float(_binary_column(target_pair["final"], "fell").mean()),
          "paired_fall_delta": float(target_fall.mean()),
        },
        "D0": {
          "baseline_success_rate": float(_binary_column(d0_pair["baseline"], "success").mean()),
          "final_success_rate": float(_binary_column(d0_pair["final"], "success").mean()),
          "paired_success_delta": float(d0_success.mean()),
          "baseline_fall_rate": float(_binary_column(d0_pair["baseline"], "fell").mean()),
          "final_fall_rate": float(_binary_column(d0_pair["final"], "fell").mean()),
          "paired_fall_delta": float(d0_fall.mean()),
        },
      }
    intervals = {
      "target_success": hierarchical_paired_scene_interval_v19(
        target_success_groups,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 100 * mode_index,
      ),
      "target_fall": hierarchical_paired_scene_interval_v19(
        target_fall_groups,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 100 * mode_index + 1,
      ),
      "d0_success": hierarchical_paired_scene_interval_v19(
        d0_success_groups,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 100 * mode_index + 2,
      ),
      "d0_fall": hierarchical_paired_scene_interval_v19(
        d0_fall_groups,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 100 * mode_index + 3,
      ),
    }
    per_seed_deltas = [per_seed[str(seed)]["target"]["paired_success_delta"] for seed in seeds]
    gate = independent_diagonal_scene_gate_v19(
      diagonal_success_delta=intervals["target_success"][0],
      per_seed_success_deltas=per_seed_deltas,
      diagonal_fall_delta=intervals["target_fall"][0],
      d0_success_delta=intervals["d0_success"][0],
    )
    claims[mode] = {
      "question": f"Does the sealed {mode} v19 specialist improve its own frozen context?",
      "claim_passed": gate["passed"],
      "gate": gate,
      "strong_evidence_lcb95_positive": intervals["target_success"][1] > 0.0,
      "target": {
        "paired_success_delta_mean_lcb95_ucb95": intervals["target_success"],
        "paired_fall_delta_mean_lcb95_ucb95": intervals["target_fall"],
        "confidence_interval_is_report_only": True,
      },
      "D0_sanity": {
        "paired_success_delta_mean_lcb95_ucb95": intervals["d0_success"],
        "paired_fall_delta_mean_lcb95_ucb95": intervals["d0_fall"],
      },
      "per_adaptation_seed": per_seed,
    }

  expected_rows = len(MODES) * len(seeds) * (args.target_episodes + args.d0_episodes)
  with paired_csv.open(newline="") as handle:
    actual_rows = sum(1 for _ in handle) - 1
  if actual_rows != expected_rows:
    raise RuntimeError(f"paired CSV has {actual_rows} rows; expected {expected_rows}")
  passed = [mode for mode in MODES if claims[mode]["claim_passed"]]
  result = {
    "protocol_id": PROTOCOL_ID,
    "analysis_version": "v19 two-diagonal prospective audit",
    "policy_method": "Observable Failure-Conditioned Brief PPO v19",
    "formal_protocol": not args.smoke,
    "evidence_role": "fresh paired audit; never used by calibration, training, or candidate gates",
    "protocol_file": {
      "path": str(protocol_path),
      "sha256": _sha256(protocol_path),
      "git_commit": current_commit,
      "tracked_worktree_and_index_clean": tracked_clean,
    },
    "runtime_cbf": True,
    "adaptation_seeds": seeds,
    "audit_seed": args.audit_seed,
    "bootstrap_seed": args.bootstrap_seed,
    "contexts": {
      mode: {
        "path": str(context_paths[mode]),
        "file_sha256": _sha256(context_paths[mode]),
        "parameters_sha256": contexts[mode]["parameters_sha256"],
        "calibration_checks": calibration_checks[mode],
        "calibration": contexts[mode]["calibration"],
      }
      for mode in MODES
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
    "independent_claims": claims,
    "passed_specialists": passed,
    "failed_specialists": [mode for mode in MODES if mode not in passed],
    "joint_conclusion": {
      "defined": False,
      "all_specialists_required": False,
      "macro_gate_used": False,
      "note": "The lateral and contact-stability claims stand or fail independently.",
    },
    "paired_episode_metrics": {
      "path": str(paired_csv),
      "sha256": _sha256(paired_csv),
      "row_count": actual_rows,
    },
    "raw_evaluations": raw,
  }
  output_path = output_dir / "diagonal_audit_summary.json"
  temporary = output_dir / ".diagonal_audit_summary.json.tmp"
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary.replace(output_path)
  print(
    json.dumps(
      {
        "output_path": str(output_path),
        "paired_csv": result["paired_episode_metrics"],
        "independent_claims": {
          mode: {
            "claim_passed": claims[mode]["claim_passed"],
            "strong_evidence_lcb95_positive": claims[mode][
              "strong_evidence_lcb95_positive"
            ],
            "target": claims[mode]["target"],
            "gate": claims[mode]["gate"],
          }
          for mode in MODES
        },
        "joint_conclusion": result["joint_conclusion"],
      },
      indent=2,
      sort_keys=True,
    )
  )
  env.close()


if __name__ == "__main__":
  main()
