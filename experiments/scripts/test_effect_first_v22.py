"""Fresh paired base-versus-best final test for one v22 development context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audit_deployment_v21 import _comparison_metrics
from audit_specialists_diagonal_v19 import _load_rows
from online_refine_stairs import _actor_state, _actor_state_sha256, _evaluate_state
from specialist_v22_protocol import (
  CONTEXT_ADAPTATION_SEEDS,
  CONTEXT_FINAL_D0_SEEDS,
  CONTEXT_FINAL_TARGET_SEEDS,
  CONTEXT_REPORT_BOOTSTRAP_SEEDS,
  CONTEXTS,
  EVAL_BATCH_SIZE,
  FINAL_D0_EPISODES,
  FINAL_TARGET_EPISODES,
  MODES,
  PROTOCOL_ID,
  REPORT_BOOTSTRAP_SAMPLES,
  ROUNDS,
  V22_CONTEXT_SCHEMA_VERSION,
  configure_v22_policy_evaluation_algorithm,
  development_success_gate,
)

POLICY_ROLES = ("base", "best")
PAIRED_FIELDS = [
  "context_id",
  "specialist_mode",
  "evaluation_role",
  "pair_index",
  "evaluation_seed",
  "environment_id",
  *[
    f"{role}_{field}"
    for role in POLICY_ROLES
    for field in ("success", "fell", "failure_type", "return", "max_riser")
  ],
]


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


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--context-id", choices=CONTEXTS, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--training-dir", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def _ordered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  return sorted(
    rows,
    key=lambda row: (int(row["evaluation_seed"]), int(row["environment_id"])),
  )


def _write_csv(
  path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.protocol_commit:
    raise RuntimeError("v22 final test HEAD differs from its frozen protocol")
  if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("v22 final test requires a clean tracked worktree")
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  relative_protocol = protocol_path.relative_to(repo)
  frozen_protocol = subprocess.run(
    ["git", "show", f"{current_commit}:{relative_protocol}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  protocol_sha256 = _sha256(protocol_path)
  if hashlib.sha256(frozen_protocol).hexdigest() != protocol_sha256:
    raise RuntimeError("v22 final-test protocol differs from its Git blob")
  context_id = args.context_id
  expected_status = f"prospectively_frozen_before_{context_id}_adaptation"
  if (
    protocol.get("protocol_id") != PROTOCOL_ID
    or protocol.get("context_schema_version") != V22_CONTEXT_SCHEMA_VERSION
    or protocol.get("status") != expected_status
    or protocol.get("final_test", {}).get("target_paired_episodes")
    != FINAL_TARGET_EPISODES
    or protocol.get("final_test", {}).get("d0_paired_episodes")
    != FINAL_D0_EPISODES
  ):
    raise RuntimeError("unexpected v22 final-test protocol")

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import src.tasks  # noqa: F401
  from src.tasks.stairs_cbf.config import configure_v19_observable_refinement_runner
  from src.tasks.stairs_cbf.deployment_context import (
    configure_v19_actor_interface,
    load_calibrated_v22_context,
  )

  context_path = args.context.resolve()
  context = load_calibrated_v22_context(context_path)
  mode = MODES[context_id]
  declared_context = protocol["sealed_inputs"]["contexts"][context_id]
  if not (
    context["context_id"] == context_id
    and context["specialist_mode"] == mode
    and _sha256(context_path) == declared_context["file_sha256"]
    and context["parameters_sha256"] == declared_context["parameters_sha256"]
  ):
    raise RuntimeError("v22 final-test context differs from its seal")
  checkpoint = args.base_policy_checkpoint.resolve()
  base_sha256 = _sha256(checkpoint)
  if base_sha256 != protocol["sealed_inputs"]["base_policy_checkpoint_sha256"]:
    raise RuntimeError("v22 final-test base checkpoint differs from its seal")

  training_dir = args.training_dir.resolve()
  training_summary_path = training_dir / "specialist_summary.json"
  best_checkpoint = training_dir / "best_so_far.pt"
  if not training_summary_path.is_file() or not best_checkpoint.is_file():
    raise FileNotFoundError("v22 training summary or best checkpoint is missing")
  training = json.loads(training_summary_path.read_text())
  checks = {
    "protocol": training.get("protocol_id") == PROTOCOL_ID,
    "context": training.get("context_id") == context_id,
    "mode": training.get("specialist_mode") == mode,
    "adaptation_seed": training.get("seed") == CONTEXT_ADAPTATION_SEEDS[context_id],
    "rounds": len(training.get("rounds", [])) == ROUNDS,
    "beta_zero": training.get("matched_success_preservation_beta") == 0.0,
    "single_branch": training.get("control_or_parallel_comparison_branch") is False,
    "final_unseen": training.get("final_test_accessed") is False,
    "protocol_commit": training.get("frozen_protocol", {}).get("git_commit")
    == current_commit,
    "protocol_sha256": training.get("frozen_protocol", {}).get("sha256")
    == protocol_sha256,
    "base": training.get("base_policy_checkpoint_sha256") == base_sha256,
    "best_hash": training.get("validation_monitor", {}).get(
      "best_checkpoint_sha256"
    )
    == _sha256(best_checkpoint),
  }
  failed = [name for name, passed in checks.items() if not passed]
  if failed:
    raise RuntimeError(f"invalid v22 training artifact: {failed}")

  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  configure_v19_actor_interface(env_cfg, context)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = CONTEXT_FINAL_TARGET_SEEDS[context_id]
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  configure_v22_policy_evaluation_algorithm(agent_cfg.algorithm)
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v22 final-test task has no online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  if warm_start.get("pi0_exact_preservation_proof") is not True:
    raise RuntimeError("v22 final-test pi0 expansion proof failed")
  actors = {"base": _actor_state(runner.alg.actor)}
  runner.load_online_checkpoint(str(best_checkpoint), map_location=args.device)
  actors["best"] = _actor_state(runner.alg.actor)
  if _actor_state_sha256(actors["base"]) != training["initial_actor_sha256"]:
    raise RuntimeError("v22 final-test pi0 differs from training")
  if _actor_state_sha256(actors["best"]) != training["validation_monitor"][
    "best_so_far"
  ]["actor_sha256"]:
    raise RuntimeError("v22 final-test actor differs from selected best checkpoint")

  output_dir = args.output_dir.resolve()
  if (output_dir / "final_test.json").exists():
    raise RuntimeError("refusing to rerun a completed v22 final test")
  output_dir.mkdir(parents=True, exist_ok=True)
  raw: dict[str, Any] = {}
  paired_rows: list[dict[str, Any]] = []
  telemetry_rows: list[dict[str, Any]] = []
  comparisons: dict[str, Any] = {}
  for evaluation_role, domain, episode_count, seed in (
    (
      "target",
      "DQHMED",
      FINAL_TARGET_EPISODES,
      CONTEXT_FINAL_TARGET_SEEDS[context_id],
    ),
    ("D0", "D0", FINAL_D0_EPISODES, CONTEXT_FINAL_D0_SEEDS[context_id]),
  ):
    repeats = episode_count // EVAL_BATCH_SIZE
    summaries: dict[str, Any] = {}
    rows_by_role: dict[str, list[dict[str, str]]] = {}
    for role in POLICY_ROLES:
      root = output_dir / "raw" / role / evaluation_role
      summaries[role] = _evaluate_state(
        runner,
        actors[role],
        domains=(domain,),
        num_envs=EVAL_BATCH_SIZE,
        num_episodes=EVAL_BATCH_SIZE,
        seed=seed,
        repeats=repeats,
        device=args.device,
        runtime_filter=True,
        artifact_dir=root,
        resume=True,
        deployment_context=context_path if evaluation_role == "target" else None,
        v19_context=context_path,
        telemetry_env_id=0 if evaluation_role == "target" else None,
      )[domain]
      rows_by_role[role] = _load_rows(
        root, domain=domain, first_seed=seed, repeats=repeats
      )
      if len(rows_by_role[role]) != episode_count:
        raise RuntimeError("v22 final-test raw episode count differs")
      if evaluation_role == "target":
        for telemetry_file in summaries[role]["inline_telemetry_files"]:
          with Path(telemetry_file).open(newline="") as handle:
            for row in csv.DictReader(handle):
              telemetry_rows.append(
                {
                  "context_id": context_id,
                  "specialist_mode": mode,
                  "policy_role": role,
                  **row,
                }
              )
    signatures = {
      role: summaries[role]["initial_state_signatures"] for role in POLICY_ROLES
    }
    if signatures["base"] != signatures["best"]:
      raise RuntimeError("v22 final-test paired initial-state signatures differ")
    ordered = {role: _ordered_rows(rows) for role, rows in rows_by_role.items()}
    keys = {
      role: [
        (int(row["evaluation_seed"]), int(row["environment_id"]))
        for row in ordered[role]
      ]
      for role in POLICY_ROLES
    }
    if keys["base"] != keys["best"] or len(keys["base"]) != len(set(keys["base"])):
      raise RuntimeError("v22 final-test paired episode identities differ")
    comparison = _comparison_metrics(
      ordered["base"],
      ordered["best"],
      bootstrap_samples=REPORT_BOOTSTRAP_SAMPLES,
      bootstrap_seed=CONTEXT_REPORT_BOOTSTRAP_SEEDS[context_id][evaluation_role],
    )
    comparisons[evaluation_role] = comparison
    raw[evaluation_role] = summaries
    for pair_index, (base_row, best_row) in enumerate(
      zip(ordered["base"], ordered["best"], strict=True)
    ):
      row: dict[str, Any] = {
        "context_id": context_id,
        "specialist_mode": mode,
        "evaluation_role": evaluation_role,
        "pair_index": pair_index,
        "evaluation_seed": base_row["evaluation_seed"],
        "environment_id": base_row["environment_id"],
      }
      for role, source in (("base", base_row), ("best", best_row)):
        for field in ("success", "fell", "failure_type", "return", "max_riser"):
          row[f"{role}_{field}"] = source[field]
      paired_rows.append(row)
  env.close()

  target = comparisons["target"]
  d0 = comparisons["D0"]
  gate = development_success_gate(
    target_success_delta=float(target["success_delta_mean_lcb95_ucb95"][0]),
    target_fall_delta=float(target["fall_delta_mean_lcb95_ucb95"][0]),
    d0_success_delta=float(d0["success_delta_mean_lcb95_ucb95"][0]),
  )
  paired_csv = output_dir / "paired_episode_metrics.csv"
  _write_csv(paired_csv, PAIRED_FIELDS, paired_rows)
  telemetry_csv = output_dir / "mechanism_telemetry.csv"
  if not telemetry_rows:
    raise RuntimeError("v22 final test produced no inline telemetry")
  _write_csv(telemetry_csv, list(telemetry_rows[0]), telemetry_rows)
  result = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "evidence_role": "effect-first fresh paired development test",
    "context_id": context_id,
    "specialist_mode": mode,
    "base_policy_checkpoint_sha256": base_sha256,
    "best_checkpoint_sha256": _sha256(best_checkpoint),
    "best_checkpoint_round": training["validation_monitor"]["best_so_far"][
      "round"
    ],
    "protocol": {
      "file": str(protocol_path),
      "sha256": protocol_sha256,
      "git_commit": current_commit,
    },
    "freshness": {
      "target_seed": CONTEXT_FINAL_TARGET_SEEDS[context_id],
      "d0_seed": CONTEXT_FINAL_D0_SEEDS[context_id],
      "target_paired_episodes": FINAL_TARGET_EPISODES,
      "d0_paired_episodes": FINAL_D0_EPISODES,
      "validation_or_candidate_conditions_reused": False,
      "same_conditions_for_base_and_best": True,
    },
    "four_primary_numbers": {
      "target_base_success_rate": target["old_success_rate"],
      "target_best_success_rate": target["new_success_rate"],
      "target_base_fall_rate": target["old_fall_rate"],
      "target_best_fall_rate": target["new_fall_rate"],
    },
    "comparisons": comparisons,
    "development_gate": gate,
    "confidence_intervals_are_report_only": True,
    "bootstrap_samples": REPORT_BOOTSTRAP_SAMPLES,
    "raw_evaluations": raw,
    "paired_episode_metrics": {
      "path": str(paired_csv),
      "sha256": _sha256(paired_csv),
      "row_count": len(paired_rows),
    },
    "mechanism_telemetry": {
      "path": str(telemetry_csv),
      "sha256": _sha256(telemetry_csv),
      "row_count": len(telemetry_rows),
      "same_rollout_outcome_bound": True,
    },
    "training_artifact_checks": checks,
  }
  output = output_dir / "final_test.json"
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
