"""Publish compact formal filter-free results and representative models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ARMS = ("nominal_ft", "reward_only_ft", "filter_only_ft", "dual_safe_ft")
HARDWARE_PROXY_ARMS = ("filter_only_ft", "dual_safe_ft")
CONTEXTS = ("F1", "F2", "F3")
TRAINING_SEEDS = (201357000, 201357001, 201357002)
REPRESENTATIVE_MODEL_SEED = TRAINING_SEEDS[0]
ARM_LABELS = {
  "frozen": "Frozen",
  "nominal_ft": "Nominal FT",
  "reward_only_ft": "Reward-only FT",
  "filter_only_ft": "Filter-only FT",
  "dual_safe_ft": "Dual Safe-FT",
}
EXPECTED_ARM_FACTORS = {
  "nominal_ft": (False, False),
  "reward_only_ft": (False, True),
  "filter_only_ft": (True, False),
  "dual_safe_ft": (True, True),
}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument(
    "--result-relative-dir",
    type=Path,
    default=Path("results/online/paper_filter_free_ablation_v140"),
  )
  parser.add_argument("--branch", default="feature/online-safe-refinement")
  parser.add_argument("--push", action="store_true")
  return parser.parse_args()


def _git(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
  if not source.is_file():
    raise FileNotFoundError(source)
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(source, destination)


def _copy_named_summaries(source_root: Path, destination_root: Path) -> int:
  count = 0
  for source in sorted(source_root.rglob("summary.json")):
    _copy(source, destination_root / source.relative_to(source_root))
    count += 1
  return count


def _csv_fieldnames(path: Path) -> set[str]:
  with path.open(newline="") as handle:
    reader = csv.reader(handle)
    return set(next(reader))


def _validate_deployment_candidate(
  *,
  repo: Path,
  checkpoint: Path,
  checkpoint_hash: str,
  actor_hash: str,
  checkpoint_label: str,
  output: Path,
  report: Path,
  context: str,
  seed: int,
) -> dict[str, Any]:
  command = [
    sys.executable,
    str(repo / "experiments/scripts/validate_filter_free_deployment.py"),
    "--checkpoint",
    str(checkpoint),
    "--output",
    str(output),
    "--report",
    str(report),
    "--expected-checkpoint-sha256",
    checkpoint_hash,
    "--expected-actor-sha256",
    actor_hash,
    "--checkpoint-label",
    checkpoint_label,
    "--artifact-label",
    str(output.relative_to(repo)),
    "--context",
    context,
    "--training-seed",
    str(seed),
  ]
  subprocess.run(
    command,
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  )
  return json.loads(report.read_text())


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  training_root = args.training_root.resolve()
  if args.result_relative_dir.is_absolute() or ".." in args.result_relative_dir.parts:
    raise ValueError("result directory must be a safe repository-relative path")
  result_dir = repo / args.result_relative_dir
  if result_dir.exists():
    raise FileExistsError(result_dir)
  if _git(repo, "status", "--porcelain"):
    raise RuntimeError("publishing requires a clean worktree")
  if _git(repo, "branch", "--show-current") != args.branch:
    raise RuntimeError("publishing branch differs from the requested branch")

  evaluation_root = training_root / "evaluation_seed201357900"
  proxy_root = training_root / "hardware_proxy_seed201357901"
  final_root = training_root / "final_summary"
  required_status = (
    training_root / "training_progress.json",
    evaluation_root / "evaluation_progress.json",
    proxy_root / "proxy_progress.json",
  )
  for path in required_status:
    if not path.is_file() or json.loads(path.read_text()).get("status") != "complete":
      raise RuntimeError(f"formal stage is not complete: {path}")
  final_results_path = final_root / "final_results.json"
  if not final_results_path.is_file():
    raise FileNotFoundError(final_results_path)

  result_dir.mkdir(parents=True)
  for name in (
    "training_manifest.json",
    "training_progress.json",
  ):
    _copy(training_root / name, result_dir / "manifests" / name)
  for name in (
    "evaluation_manifest.json",
    "evaluation_progress.json",
    "paired_checkpoint_results.json",
  ):
    _copy(evaluation_root / name, result_dir / "evaluation" / name)
  for name in (
    "final_results.json",
    "main_table.csv",
    "learning_curves.csv",
    "training_safety.csv",
    "training_safety_curves.csv",
    "task_learning_curves.png",
    "task_learning_curves.svg",
    "training_safety_curves.png",
    "training_safety_curves.svg",
    "paired_statistics.json",
    "paired_statistics.csv",
    "factorial_contrasts.csv",
    "SUMMARY.md",
  ):
    _copy(final_root / name, result_dir / name)
  for name in (
    "proxy_manifest.json",
    "proxy_progress.json",
    "hardware_proxy_results.json",
    "hardware_proxy_main_table.csv",
    "proxy_policy_results.csv",
  ):
    _copy(proxy_root / name, result_dir / "hardware_proxy" / name)

  for arm in ARMS:
    for context in CONTEXTS:
      for seed in TRAINING_SEEDS:
        source = training_root / context / arm / f"seed_{seed}"
        destination = result_dir / "training" / context / arm / f"seed_{seed}"
        for name in ("training_summary.json", "round_metrics.json", "round_metrics.csv"):
          _copy(source / name, destination / name)

  evaluation_summary_count = _copy_named_summaries(
    evaluation_root,
    result_dir / "evaluation" / "checkpoint_summaries",
  )
  proxy_summary_count = _copy_named_summaries(
    proxy_root,
    result_dir / "hardware_proxy" / "policy_summaries",
  )
  if evaluation_summary_count != 222 or proxy_summary_count != 42:
    raise RuntimeError(
      "formal summary count differs: "
      f"paired={evaluation_summary_count}, proxy={proxy_summary_count}"
    )

  checkpoint_index: list[dict[str, Any]] = []
  deployment_index: list[dict[str, Any]] = []
  training_protocol_failures: list[str] = []
  for arm in ARMS:
    for context in CONTEXTS:
      for seed in TRAINING_SEEDS:
        source = (
          training_root / context / arm / f"seed_{seed}" / "round_04.pt"
        )
        summary = json.loads(
          (
            training_root
            / context
            / arm
            / f"seed_{seed}"
            / "training_summary.json"
          ).read_text()
        )
        metadata = summary.get("paper_filter_free_ablation_training") or {}
        expected_filter, expected_reward = EXPECTED_ARM_FACTORS[arm]
        protocol_checks = {
          "arm": metadata.get("arm") == arm,
          "context": metadata.get("context") == context,
          "seed": summary.get("seed") == seed,
          "frozen_v139": (
            metadata.get("expected_base_checkpoint_sha256")
            == "323f1e00b58d379b8746c0191a44272f2e1df134139050417c56e733cc484728"
          ),
          "four_rounds": metadata.get("rounds") == 4,
          "fixed_budget": (
            metadata.get("num_envs") == 128
            and metadata.get("rollout_steps") == 1024
          ),
          "training_std": metadata.get("training_action_std") == 0.05,
          "raw_action_ppo": (
            metadata.get("ppo_storage_action")
            == "raw_nominal_policy_action"
          ),
          "fixed_round_4": (
            metadata.get("primary_checkpoint_round") == 4
            and metadata.get("primary_checkpoint_rule") == "fixed_round_4"
            and metadata.get("checkpoint_rounds") == [0, 1, 2, 4]
          ),
          "matched_arm_filter": (
            metadata.get("runtime_filter_during_adaptation")
            is expected_filter
          ),
          "matched_arm_reward": (
            metadata.get("cbf_reward_during_adaptation") is expected_reward
          ),
          "cbf_off_primary": (
            metadata.get("final_deployment_filter") == "off"
            and metadata.get("primary_metric")
            == "post_adaptation_cbf_off_success_rate"
          ),
          "paired_512": (
            metadata.get("paired_evaluation_episodes_per_filter_condition")
            == 512
            and metadata.get("paired_evaluation_filter_conditions")
            == ["off", "on"]
          ),
          "no_checkpoint_selection": (
            metadata.get("checkpoint_candidate_selection") is False
            and metadata.get("candidate_selection_additional_evaluation_count")
            == 0
          ),
          "continuous_no_rollback": (
            metadata.get("training_trajectory")
            == "continuous_without_acceptance_or_rollback"
          ),
          "all_contract_checks": (
            bool(metadata.get("contract_checks"))
            and all(bool(value) for value in metadata["contract_checks"].values())
          ),
        }
        training_protocol_failures.extend(
          f"{arm}/{context}/{seed}:{name}"
          for name, passed in protocol_checks.items()
          if not passed
        )
        checkpoint_hash = _sha256(source)
        actor_hash = summary["final_actor_sha256"]
        published = arm == "dual_safe_ft" and seed == REPRESENTATIVE_MODEL_SEED
        destination = None
        if published:
          destination_path = (
            result_dir
            / "checkpoints"
            / f"dual_safe_ft_{context}_seed_{seed}_round_04.pt"
          )
          _copy(source, destination_path)
          destination = str(destination_path.relative_to(repo))
          onnx_path = (
            result_dir
            / "deployment"
            / f"dual_safe_ft_{context}_seed_{seed}_round_04.onnx"
          )
          report_path = onnx_path.with_suffix(".validation.json")
          deployment_report = _validate_deployment_candidate(
            repo=repo,
            checkpoint=source,
            checkpoint_hash=checkpoint_hash,
            actor_hash=actor_hash,
            checkpoint_label=destination,
            output=onnx_path,
            report=report_path,
            context=context,
            seed=seed,
          )
          deployment_index.append(
            {
              "context": context,
              "training_seed": seed,
              "checkpoint": destination,
              "onnx": str(onnx_path.relative_to(repo)),
              "validation_report": str(report_path.relative_to(repo)),
              "checkpoint_sha256": checkpoint_hash,
              "actor_sha256": actor_hash,
              "onnx_sha256": deployment_report["onnx_sha256"],
              "bridge_parity_passed": deployment_report["bridge_parity"][
                "passed"
              ],
              "pytorch_bridge_p95_ms": deployment_report["latency"][
                "pytorch_backend"
              ]["p95_ms"],
              "pytorch_bridge_deadline_passed": deployment_report["latency"][
                "pytorch_backend"
              ]["p95_within_policy_deadline"],
              "onnx_reference_p95_ms": deployment_report["latency"][
                "onnx_backend"
              ]["p95_ms"],
            }
          )
        checkpoint_index.append(
          {
            "arm": arm,
            "context": context,
            "training_seed": seed,
            "round": 4,
            "checkpoint_sha256": checkpoint_hash,
            "actor_sha256": actor_hash,
            "published_to_repository": published,
            "repository_path": destination,
            "selection_used": False,
          }
        )
  (result_dir / "checkpoint_index.json").write_text(
    json.dumps(checkpoint_index, indent=2, sort_keys=True) + "\n"
  )
  if len(deployment_index) != len(CONTEXTS):
    raise RuntimeError(
      f"deployment candidate count differs: {len(deployment_index)}"
    )
  (result_dir / "deployment_index.json").write_text(
    json.dumps(deployment_index, indent=2, sort_keys=True) + "\n"
  )

  final_results = json.loads(final_results_path.read_text())
  proxy_results = json.loads(
    (proxy_root / "hardware_proxy_results.json").read_text()
  )
  training_progress = json.loads(
    (training_root / "training_progress.json").read_text()
  )
  evaluation_progress = json.loads(
    (evaluation_root / "evaluation_progress.json").read_text()
  )
  proxy_progress = json.loads((proxy_root / "proxy_progress.json").read_text())
  paired_results = json.loads(
    (evaluation_root / "paired_checkpoint_results.json").read_text()
  )
  evaluation_manifest = json.loads(
    (evaluation_root / "evaluation_manifest.json").read_text()
  )
  required_paired_metrics = {
    "cbf_off_success_rate",
    "cbf_on_success_rate",
    "shield_gap",
    "cbf_off_fall_rate",
    "cbf_on_fall_rate",
    "cbf_off_mean_reached_riser",
    "cbf_on_mean_reached_riser",
    "cbf_off_mean_return",
    "cbf_on_mean_return",
    "cbf_off_mean_completion_time_s",
    "cbf_on_mean_completion_time_s",
    "cbf_off_nominal_violation_steps_per_riser",
    "cbf_on_nominal_violation_steps_per_riser",
    "cbf_off_would_intervene_fraction",
    "cbf_on_would_intervene_fraction",
    "cbf_off_mean_counterfactual_correction_norm",
    "cbf_on_mean_correction_norm",
    "cbf_off_unsafe_overlap_steps_per_riser",
    "cbf_on_unsafe_overlap_steps_per_riser",
    "cbf_off_toe_riser_kick_episode_rate",
    "cbf_on_toe_riser_kick_episode_rate",
  }
  required_task_curve_fields = {
    "arm",
    "context",
    "training_seed",
    "round",
    "online_transitions",
    "cbf_off_success_rate",
    "cbf_on_success_rate",
    "shield_gap",
    "cbf_off_nominal_violation_steps_per_riser",
    "cbf_off_would_intervene_fraction",
    "cbf_off_mean_counterfactual_correction_norm",
  }
  required_training_safety_fields = {
    "arm",
    "context",
    "training_seed",
    "training_falls",
    "training_shield_recoveries",
    "training_nominal_violation_count",
    "training_executed_violation_count",
    "training_minimum_nominal_barrier_margin",
    "training_minimum_executed_barrier_margin",
  }
  required_training_safety_curve_fields = {
    "arm",
    "context",
    "training_seed",
    "round",
    "online_transitions",
    "round_falls",
    "round_shield_recoveries",
    "round_nominal_violation_count",
    "round_executed_violation_count",
    "round_minimum_nominal_barrier_margin",
    "round_minimum_executed_barrier_margin",
  }
  structural_checks = {
    "training_matrix_complete_36_of_36": (
      training_progress.get("status") == "complete"
      and training_progress.get("completed_jobs") == 36
      and training_progress.get("job_count") == 36
    ),
    "paired_evaluation_complete_222_of_222": (
      evaluation_progress.get("status") == "complete"
      and evaluation_progress.get("completed_jobs") == 222
      and evaluation_progress.get("evaluation_job_count") == 222
    ),
    "paired_checkpoint_count_111": len(paired_results) == 111,
    "fixed_training_protocol_36_of_36": not training_protocol_failures,
    "paired_evaluation_protocol_512_on_off_rounds_0_1_2_4": (
      evaluation_manifest.get("episodes_per_filter_condition") == 512
      and evaluation_manifest.get("checkpoint_rounds") == [0, 1, 2, 4]
      and evaluation_manifest.get("checkpoint_spec_count") == 111
      and evaluation_manifest.get("shared_round_0_evaluation") is True
    ),
    "paired_metrics_complete": all(
      required_paired_metrics.issubset(row) for row in paired_results
    ),
    "task_learning_curve_metrics_complete": required_task_curve_fields.issubset(
      _csv_fieldnames(final_root / "learning_curves.csv")
    ),
    "hardware_proxy_complete_42_of_42": (
      proxy_progress.get("status") == "complete"
      and proxy_progress.get("completed_jobs") == 42
      and proxy_progress.get("job_count") == 42
    ),
    "main_table_contains_five_methods": (
      {row.get("arm") for row in final_results.get("main_table", [])}
      == {"frozen", *ARMS}
    ),
    "main_table_internalization_metrics_complete": all(
      {
        "mean_cbf_off_would_intervene_fraction",
        "mean_cbf_off_counterfactual_correction_norm",
        "mean_cbf_off_nominal_violation_steps_per_riser",
        "mean_shield_gap",
      }.issubset(row)
      for row in final_results.get("main_table", [])
    ),
    "task_curve_rows_144": final_results.get("curve_row_count") == 144,
    "training_safety_runs_36": (
      final_results.get("training_safety_run_count") == 36
    ),
    "training_safety_curve_rows_144": (
      final_results.get("training_safety_curve_row_count") == 144
    ),
    "training_safety_metrics_complete": required_training_safety_fields.issubset(
      _csv_fieldnames(final_root / "training_safety.csv")
    ),
    "training_safety_curve_metrics_complete": (
      required_training_safety_curve_fields.issubset(
        _csv_fieldnames(final_root / "training_safety_curves.csv")
      )
    ),
    "paired_statistics_four_adaptation_methods": (
      set(final_results.get("paired_report_statistics", {})) == set(ARMS)
    ),
    "factorial_contrasts_nine_task_and_safety_metrics": (
      final_results.get("factorial_contrasts_are_report_only") is True
      and len(final_results.get("factorial_contrasts", [])) == 9
      and {
        row.get("metric")
        for row in final_results.get("factorial_contrasts", [])
      }
      == {
        "mean_cbf_off_success_rate",
        "mean_cbf_off_nominal_violation_steps_per_riser",
        "mean_cbf_off_would_intervene_fraction",
        "mean_cbf_off_counterfactual_correction_norm",
        "mean_shield_gap",
        "training_falls_mean",
        "training_shield_recoveries_mean",
        "training_nominal_violation_fraction_mean",
        "training_executed_violation_fraction_mean",
      }
    ),
    "hardware_proxy_three_method_aggregate": (
      {row.get("arm") for row in proxy_results.get("aggregate", [])}
      == {"frozen", *HARDWARE_PROXY_ARMS}
    ),
    "checkpoint_index_36": len(checkpoint_index) == 36,
    "representative_fixed_round_4_models_3": sum(
      bool(row["published_to_repository"]) for row in checkpoint_index
    )
    == 3,
    "deployment_candidates_3": len(deployment_index) == 3,
    "deployment_bridge_parity_passed": all(
      bool(row["bridge_parity_passed"]) for row in deployment_index
    ),
  }
  failed_structural_checks = sorted(
    name for name, passed in structural_checks.items() if not passed
  )
  completion_audit = {
    "schema_version": 1,
    "scope": (
      "formal simulation adaptation, paired CBF-on/off evaluation, "
      "CBF-off hardware proxy, and offline ONNX/bridge validation"
    ),
    "physical_robot_experiment_executed": False,
    "physical_robot_claim_made": False,
    "structural_protocol_audit_passed": not failed_structural_checks,
    "structural_checks": structural_checks,
    "failed_structural_checks": failed_structural_checks,
    "training_protocol_failures": training_protocol_failures,
    "outcomes_not_used_as_completion_requirements": {
      "main_claim_supported": final_results["main_claim_supported"],
      "main_claim_checks": final_results["main_claim_checks"],
      "global_fall_and_violation_free_training_supported": final_results[
        "global_fall_and_violation_free_training_supported"
      ],
      "training_safety_checks": final_results["training_safety_checks"],
      "all_pytorch_bridge_p95_within_20ms": all(
        bool(row["pytorch_bridge_deadline_passed"])
        for row in deployment_index
      ),
    },
    "physical_stage_decision": (
      "operator-gated CBF-on adaptation and shadow checks may proceed; "
      "physical CBF-off still requires the documented independent safety gate"
      if final_results["main_claim_supported"]
      else "do not run physical CBF-off from this result; retain CBF-on and "
      "shadow-only evaluation because the simulation claim gate failed"
    ),
    "counts": {
      "training_jobs": training_progress.get("completed_jobs"),
      "evaluation_jobs": evaluation_progress.get("completed_jobs"),
      "paired_checkpoints": len(paired_results),
      "hardware_proxy_jobs": proxy_progress.get("completed_jobs"),
      "indexed_round_4_checkpoints": len(checkpoint_index),
      "published_representative_models": 3,
      "deployment_candidates": len(deployment_index),
    },
  }
  (result_dir / "COMPLETION_AUDIT.json").write_text(
    json.dumps(completion_audit, indent=2, sort_keys=True) + "\n"
  )
  if failed_structural_checks:
    raise RuntimeError(
      f"formal completion audit failed: {failed_structural_checks}"
    )
  table_lines = [
    "| Arm | Filter | CBF reward | F1 off | F2 off | F3 off | Mean off | Off Δ | Shield gap | Training falls |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in final_results["main_table"]:
    falls = (
      "—"
      if row["training_falls_mean"] is None
      else f"{float(row['training_falls_mean']):.1f}"
    )
    table_lines.append(
      "| {arm} | {filter} | {reward} | {f1:.3f} | {f2:.3f} | {f3:.3f} | "
      "{off:.3f} | {gain:+.3f} | {gap:+.3f} | {falls} |".format(
        arm=ARM_LABELS[str(row["arm"])],
        filter=row["training_filter"],
        reward=row["cbf_reward"],
        f1=float(row["f1_cbf_off_success_rate"]),
        f2=float(row["f2_cbf_off_success_rate"]),
        f3=float(row["f3_cbf_off_success_rate"]),
        off=float(row["mean_cbf_off_success_rate"]),
        gain=float(row["mean_off_improvement"]),
        gap=float(row["mean_shield_gap"]),
        falls=falls,
      )
    )
  proxy_lines = [
    "| Arm | Nominal-sim off | Hardware-proxy off | Absolute drop | Fall rate |",
    "|---|---:|---:|---:|---:|",
  ]
  for row in proxy_results["aggregate"]:
    proxy_lines.append(
      f"| {ARM_LABELS[str(row['arm'])]} | "
      f"{float(row['nominal_sim_cbf_off_success_rate']):.3f} | "
      f"{float(row['hardware_proxy_cbf_off_success_rate']):.3f} | "
      f"{float(row['absolute_success_rate_drop']):+.3f} | "
      f"{float(row['mean_fall_rate']):.3f} |"
    )
  deployment_lines = [
    "| Context | Bridge parity | PyTorch bridge p95 (ms) | 20 ms deadline |",
    "|---|---:|---:|---:|",
  ]
  for row in deployment_index:
    deployment_lines.append(
      f"| {row['context']} | {row['bridge_parity_passed']} | "
      f"{float(row['pytorch_bridge_p95_ms']):.3f} | "
      f"{row['pytorch_bridge_deadline_passed']} |"
    )
  if final_results["main_claim_supported"]:
    claim_interpretation = [
      "The preregistered simulation claim gate passed. This is simulation "
      "evidence only: physical CBF-off remains prohibited until the staged "
      "operator safety protocol and independent shadow gate are satisfied."
    ]
  else:
    claim_interpretation = [
      "The preregistered simulation claim gate failed. These artifacts must "
      "not be presented as evidence that runtime CBF can be removed on the "
      "physical robot. Any physical follow-up is restricted to CBF-on "
      "adaptation and shadow-only filter-free measurements."
    ]
  training_safety_interpretation = [
    "Global fall- and executed-violation-free simulation training supported: "
    f"**{final_results['global_fall_and_violation_free_training_supported']}**. "
    "A false result means the filtered methods improved the local barrier "
    "constraint but did not prove globally fall-free online adaptation."
  ]
  readme = [
    "# Filter-free deployment ablation (v140)",
    "",
    "Formal simulation ablation from the frozen v139 actor. Adaptation uses "
    "four rounds × 128 environments × 1024 steps, three preregistered seeds, "
    "and fixed round-4 publication. Primary deployment evaluation is "
    "deterministic CBF-off.",
    "",
    *table_lines,
    "",
    f"Main claim supported: **{final_results['main_claim_supported']}**.",
    "",
    *claim_interpretation,
    "",
    *training_safety_interpretation,
    "",
    "## Hardware proxy", 
    "",
    *proxy_lines,
    "",
    "The proxy combines actor sensor noise/encoder bias, 1- and 2-step action "
    "delay, 0.95 actuator gain, +1 cm stair-height estimate bias, ±1.5 cm "
    "tread perturbation, friction variation, and command delay. All proxy "
    "evaluations execute CBF-off.",
    "",
    "## Offline ONNX/bridge validation",
    "",
    *deployment_lines,
    "",
    "Each representative fixed round-4 Dual actor is exported to ONNX and "
    "checked on deterministic five-frame bridge inputs. Latency covers "
    "observation assembly, actor inference, and 12-to-29 target mapping on "
    "one CPU thread. ONNX ReferenceEvaluator latency is retained in the JSON "
    "reports as portability evidence, not as a production-runtime claim.",
    "",
    "## Contents",
    "",
    "- `final_results.json` and `main_table.csv`: primary result and claim checks",
    "- `learning_curves.csv`: rounds 0/1/2/4",
    "- `training_safety.csv`: per-run falls, violations, recoveries, and min h",
    "- `training_safety_curves.csv`: per-round safety metrics versus transitions",
    "- `task_learning_curves.*`: CBF-off success and shield-gap figures",
    "- `training_safety_curves.*`: executed violations and falls figures",
    "- `paired_statistics.*`: report-only paired-bootstrap intervals and repair/regression counts",
    "- `factorial_contrasts.csv`: report-only Filter, Reward, and interaction contrasts",
    "- `evaluation/checkpoint_summaries/`: all 222 paired-condition summaries",
    "- `hardware_proxy/`: 42 Frozen/Filter-only/Dual proxy summaries and aggregate table",
    "- `training/`: all 36 training summaries and round metrics",
    "- `checkpoints/`: fixed round-4 Dual Safe-FT models for F1/F2/F3, seed 201357000",
    "- `checkpoint_index.json`: hashes for all 36 fixed round-4 models",
    "- `deployment/`: three ONNX actors plus bridge parity/latency reports",
    "- `deployment_index.json`: deployment artifact hashes and compact checks",
    "- `COMPLETION_AUDIT.json`: one structural audit of all formal deliverables",
    "",
    "Episode CSVs and redundant checkpoints remain in the archived 4080 run; "
    "the repository contains the compact evidence needed to reproduce tables.",
  ]
  (result_dir / "README.md").write_text("\n".join(readme) + "\n")

  _git(repo, "add", str(args.result_relative_dir))
  _git(repo, "commit", "-m", "Publish formal filter-free ablation results")
  commit = _git(repo, "rev-parse", "HEAD")
  if args.push:
    _git(repo, "push", "origin", args.branch)
  print(
    json.dumps(
      {
        "result_dir": str(result_dir),
        "git_commit": commit,
        "pushed": args.push,
        "evaluation_summary_count": evaluation_summary_count,
        "proxy_summary_count": proxy_summary_count,
        "representative_model_count": 3,
        "deployment_candidate_count": len(deployment_index),
        "completion_audit_passed": completion_audit[
          "structural_protocol_audit_passed"
        ],
      },
      indent=2,
      sort_keys=True,
    )
  )


if __name__ == "__main__":
  main()
