"""Evaluate the frozen v21 beta grid on L_dev/C_dev and select beta by RR-RG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from audit_specialists_diagonal_v19 import _load_rows
from online_refine_stairs import _actor_state, _evaluate_state
from specialist_v21_protocol import (
  BETA_GRID,
  CONTEXT_ADAPTATION_SEEDS,
  CONTEXT_DEVELOPMENT_SELECTION_SEEDS,
  DEVELOPMENT_SELECTION_AMENDMENT_SOURCE_FILES,
  DEVELOPMENT_SELECTION_AMENDMENT_STAGE,
  DEVELOPMENT_SELECTION_EPISODES,
  FORMAL_EVAL_BATCH_SIZE,
  FORMAL_ROUNDS,
  PROTOCOL_ID,
  V21_DEVELOPMENT_CONTEXTS,
  configure_v21_policy_evaluation_algorithm,
  repair_regression_rates,
  select_development_beta,
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _beta_label(beta: float) -> str:
  return f"beta_{beta:g}".replace(".", "p")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--context-dir", type=Path, required=True)
  parser.add_argument("--training-root", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--implementation-amendment", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _git_output(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _validate_execution_amendment(
  *,
  args: argparse.Namespace,
  repo: Path,
  current_commit: str,
  protocol: dict,
  protocol_path: Path,
) -> dict[str, object] | None:
  if current_commit == args.protocol_commit:
    if args.implementation_amendment is not None:
      raise RuntimeError("v21 selection amendment supplied without source drift")
    return None
  if args.implementation_amendment is None:
    raise RuntimeError(
      "v21 selection implementation differs from the training freeze without "
      "a committed execution amendment"
    )
  amendment_path = args.implementation_amendment.resolve()
  amendment = json.loads(amendment_path.read_text())
  relative = amendment_path.relative_to(repo)
  committed = subprocess.run(
    ["git", "show", f"{current_commit}:{relative}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  changed = set(
    _git_output(repo, "diff", "--name-only", args.protocol_commit, current_commit)
    .splitlines()
  )
  source_files = set(protocol["sealed_inputs"]["source_files"])
  changed_sources = tuple(sorted(changed & source_files))
  expected_sources = tuple(sorted(DEVELOPMENT_SELECTION_AMENDMENT_SOURCE_FILES))
  failed_attempt = amendment.get("failed_attempt", {})
  repair_scope = amendment.get("repair_scope", {})
  retry_constraints = amendment.get("retry_constraints", {})
  checks = {
    "protocol_id": amendment.get("protocol_id") == PROTOCOL_ID,
    "stage": amendment.get("stage") == DEVELOPMENT_SELECTION_AMENDMENT_STAGE,
    "disposition": amendment.get("disposition")
    == "infrastructure_retry_with_identical_frozen_evaluation_design",
    "training_protocol_commit": amendment.get("training_protocol", {}).get(
      "git_commit"
    )
    == args.protocol_commit,
    "training_protocol_sha256": amendment.get("training_protocol", {}).get(
      "sha256"
    )
    == _sha256(protocol_path),
    "selection_source_sha256": failed_attempt.get("selection_source_sha256")
    == protocol["sealed_inputs"]["source_file_sha256"][
      "experiments/scripts/select_development_beta_v21.py"
    ],
    "selection_metrics_not_written": failed_attempt.get(
      "selection_metrics_written"
    )
    is False,
    "zero_actors_evaluated": failed_attempt.get("actors_evaluated") == 0,
    "runner_not_constructed": failed_attempt.get("runner_constructed") is False,
    "raw_evaluations_not_started": failed_attempt.get("raw_evaluations_started")
    is False,
    "selection_outcomes_not_seen": amendment.get("selection_outcomes_observed")
    is False,
    "formal_outcomes_not_seen": amendment.get("formal_outcomes_observed") is False,
    "development_training_completed": amendment.get(
      "development_training_completed_before_failure"
    )
    is True,
    "development_training_outcomes_disclosed": amendment.get(
      "development_training_outcomes_observed"
    )
    is True,
    "training_not_rerun": amendment.get("development_training_rerun") is False,
    "training_artifacts_not_mutated": amendment.get("training_artifacts_mutated")
    is False,
    "frozen_evaluation_design_unchanged": all(
      repair_scope.get(key) is False
      for key in (
        "adaptation_code_or_artifacts_changed",
        "beta_grid_changed",
        "candidate_gate_changed",
        "evaluation_contexts_or_seeds_changed",
        "evaluation_episode_count_changed",
        "formal_analysis_or_gate_changed",
        "selection_formula_or_tie_break_changed",
      )
    ),
    "training_artifacts_reused": retry_constraints.get(
      "development_training_artifacts_reused_without_mutation"
    )
    is True,
    "formal_contexts_remain_unseen": retry_constraints.get(
      "formal_contexts_remain_unseen"
    )
    is True,
    "paired_episode_count_unchanged": retry_constraints.get(
      "same_fresh_episodes_per_policy_per_context"
    )
    == DEVELOPMENT_SELECTION_EPISODES,
    "evaluation_seeds_unchanged": retry_constraints.get(
      "same_frozen_evaluation_seeds"
    )
    is True,
    "allowed_source_changes": tuple(
      sorted(amendment.get("allowed_source_changes", []))
    )
    == expected_sources,
    "actual_source_changes": changed_sources == expected_sources,
    "committed_amendment": hashlib.sha256(committed).hexdigest()
    == _sha256(amendment_path),
  }
  failed = [name for name, passed in checks.items() if not passed]
  if failed:
    raise RuntimeError(f"invalid v21 development execution amendment: {failed}")
  return {
    "file": str(relative),
    "sha256": _sha256(amendment_path),
    "training_protocol_commit": args.protocol_commit,
    "evaluation_implementation_commit": current_commit,
    "checks": checks,
  }


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if not args.smoke and subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"],
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip():
    raise RuntimeError("formal-sized v21 development selection requires clean files")
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  amendment_binding = _validate_execution_amendment(
    args=args,
    repo=repo,
    current_commit=current_commit,
    protocol=protocol,
    protocol_path=protocol_path,
  )
  frozen = subprocess.run(
    [
      "git",
      "show",
      f"{args.protocol_commit}:{protocol_path.relative_to(repo)}",
    ],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  if (
    hashlib.sha256(frozen).hexdigest() != _sha256(protocol_path)
    or protocol.get("protocol_id") != PROTOCOL_ID
    or protocol.get("protocol_revision") != 1
    or protocol.get("status")
    != "prospectively_frozen_before_development_beta_selection"
  ):
    raise RuntimeError("unexpected v21 development protocol")

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import configure_v19_observable_refinement_runner
  from src.tasks.stairs_cbf.deployment_context import (
    configure_v19_actor_interface,
    load_calibrated_v21_context,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  if _sha256(checkpoint) != protocol["sealed_inputs"][
    "base_policy_checkpoint_sha256"
  ]:
    raise RuntimeError("development selection base checkpoint differs")
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  episodes = 8 if args.smoke else DEVELOPMENT_SELECTION_EPISODES
  batch_size = 8 if args.smoke else FORMAL_EVAL_BATCH_SIZE
  repeats = episodes // batch_size
  selection_input: dict[float, dict[str, dict[str, float]]] = {
    beta: {} for beta in BETA_GRID
  }
  audit_rows = []
  policy_manifests: dict[str, dict[str, object]] = {}
  for context_id in V21_DEVELOPMENT_CONTEXTS:
    context_path = args.context_dir.resolve() / f"{context_id}.json"
    context = load_calibrated_v21_context(context_path)
    if context["context_id"] != context_id or context["formal_context"] is not False:
      raise RuntimeError("development selection received a formal/wrong context")
    declared = protocol["sealed_inputs"]["contexts"][context_id]
    if (
      _sha256(context_path) != declared["file_sha256"]
      or context["parameters_sha256"] != declared["parameters_sha256"]
    ):
      raise RuntimeError("development context differs from its seal")
    task = "Unitree-G1-Stairs-Online-DQH"
    env_cfg = load_env_cfg(task)
    configure_v19_actor_interface(env_cfg, context)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = CONTEXT_DEVELOPMENT_SELECTION_SEEDS[context_id]
    env_cfg.actions["joint_pos"].enabled = True
    agent_cfg = load_rl_cfg(task)
    configure_v19_observable_refinement_runner(agent_cfg)
    configure_v21_policy_evaluation_algorithm(
      agent_cfg.algorithm, matched_success_preservation_beta=0.0
    )
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task)
    if runner_cls is None:
      raise RuntimeError("v21 development task has no online runner")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
    runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
    actors = {"base": _actor_state(runner.alg.actor)}
    summaries: dict[str, dict] = {}
    for beta in BETA_GRID:
      training_dir = (
        args.training_root.resolve()
        / context_id
        / f"{'control' if beta == 0.0 else 'v21'}_{_beta_label(beta)}"
      )
      summary_path = training_dir / "specialist_summary.json"
      final_checkpoint = training_dir / "accepted_final.pt"
      summary = json.loads(summary_path.read_text())
      expected_role = "control" if beta == 0.0 else "v21"
      checks = {
        "context": summary.get("context_id") == context_id,
        "role": summary.get("method_role") == expected_role,
        "beta": summary.get("matched_success_preservation_beta") == beta,
        "seed": summary.get("seed") == CONTEXT_ADAPTATION_SEEDS[context_id],
        "rounds": len(summary.get("rounds", [])) == FORMAL_ROUNDS,
        "development": summary.get("development_run") is True,
        "protocol": summary.get("frozen_protocol", {}).get("git_commit")
        == args.protocol_commit,
      }
      if not all(checks.values()):
        raise RuntimeError(
          f"invalid development training {context_id}/{beta}: {checks}"
        )
      runner.load_online_checkpoint(str(final_checkpoint), map_location=args.device)
      actors[_beta_label(beta)] = _actor_state(runner.alg.actor)
      policy_manifests[f"{context_id}/{_beta_label(beta)}"] = {
        "summary": str(summary_path),
        "checkpoint": str(final_checkpoint),
        "checks": checks,
      }
    evaluation_seed = CONTEXT_DEVELOPMENT_SELECTION_SEEDS[context_id]
    rows_by_policy = {}
    for role, actor in actors.items():
      root = output_dir / "raw" / context_id / role
      summaries[role] = _evaluate_state(
        runner,
        actor,
        domains=("DQHMED",),
        num_envs=batch_size,
        num_episodes=batch_size,
        seed=evaluation_seed,
        repeats=repeats,
        device=args.device,
        runtime_filter=True,
        artifact_dir=root,
        resume=True,
        deployment_context=context_path,
        v19_context=context_path,
      )["DQHMED"]
      rows_by_policy[role] = _load_rows(
        root,
        domain="DQHMED",
        first_seed=evaluation_seed,
        repeats=repeats,
      )
    env.close()
    ordering_key = lambda row: (
      int(row["evaluation_seed"]),
      int(row["environment_id"]),
    )
    rows_by_policy = {
      role: sorted(rows, key=ordering_key)
      for role, rows in rows_by_policy.items()
    }
    base_rows = rows_by_policy["base"]
    base_keys = [
      (row["evaluation_seed"], row["environment_id"]) for row in base_rows
    ]
    baseline_success = [row["success"] == "True" for row in base_rows]
    for beta in BETA_GRID:
      label = _beta_label(beta)
      rows = rows_by_policy[label]
      if [
        (row["evaluation_seed"], row["environment_id"]) for row in rows
      ] != base_keys:
        raise RuntimeError("development beta evaluation is not paired")
      candidate_success = [row["success"] == "True" for row in rows]
      metrics = repair_regression_rates(baseline_success, candidate_success)
      selection_input[beta][context_id] = {
        "repair_rate": float(metrics["repair_rate"]),
        "regression_rate": float(metrics["regression_rate"]),
      }
      audit_rows.append(
        {
          "context_id": context_id,
          "beta": beta,
          "paired_episodes": episodes,
          "base_success_rate": summaries["base"]["success_rate"],
          "candidate_success_rate": summaries[label]["success_rate"],
          "success_delta": summaries[label]["success_rate"]
          - summaries["base"]["success_rate"],
          **metrics,
        }
      )
  selection = select_development_beta(selection_input)
  csv_path = output_dir / "development_beta_metrics.csv"
  with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
    writer.writeheader()
    writer.writerows(audit_rows)
  result = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "evidence_role": "development-only beta selection; excluded from formal claims",
    "contexts": list(V21_DEVELOPMENT_CONTEXTS),
    "formal_context_outcomes_seen": False,
    "development_training_protocol_commit": args.protocol_commit,
    "evaluation_implementation_commit": current_commit,
    "development_execution_amendment": amendment_binding,
    "paired_episodes_per_policy_per_context": episodes,
    "selection": selection,
    "metrics_csv": {
      "path": str(csv_path),
      "sha256": _sha256(csv_path),
      "row_count": len(audit_rows),
    },
    "training_manifests": policy_manifests,
  }
  output = output_dir / "development_beta_selection.json"
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
