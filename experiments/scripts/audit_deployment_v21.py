"""Fresh tri-policy paired audit with inline same-rollout mechanism telemetry."""

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

import numpy as np
from audit_specialists_diagonal_v19 import _binary_column, _load_rows
from online_refine_stairs import (
  _actor_state,
  _actor_state_sha256,
  _evaluate_state,
)
from specialist_v21_protocol import (
  CONTEXT_ADAPTATION_SEEDS,
  CONTEXT_FORMAL_AUDIT_SEEDS,
  FORMAL_BOOTSTRAP_SAMPLES,
  FORMAL_BOOTSTRAP_SEED,
  FORMAL_D0_EPISODES,
  FORMAL_EVAL_BATCH_SIZE,
  FORMAL_ROUNDS,
  FORMAL_TARGET_EPISODES,
  POLICY_METHOD,
  PROTOCOL_ID,
  TELEMETRY_ENVIRONMENT_ID_PER_BATCH,
  V21_FORMAL_CONTEXTS,
  repair_regression_rates,
)

POLICY_ROLES = ("base", "control", "v21")
COMPARISONS = (
  ("control_minus_base", "base", "control"),
  ("v21_minus_base", "base", "v21"),
  ("v21_minus_control", "control", "v21"),
)
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
  parser.add_argument("--context-id", choices=V21_FORMAL_CONTEXTS, required=True)
  parser.add_argument("--context", type=Path, required=True)
  parser.add_argument("--control-training-dir", type=Path, required=True)
  parser.add_argument("--v21-training-dir", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--target-episodes", type=int, default=FORMAL_TARGET_EPISODES
  )
  parser.add_argument("--d0-episodes", type=int, default=FORMAL_D0_EPISODES)
  parser.add_argument(
    "--eval-batch-size", type=int, default=FORMAL_EVAL_BATCH_SIZE
  )
  parser.add_argument(
    "--bootstrap-samples", type=int, default=FORMAL_BOOTSTRAP_SAMPLES
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--smoke", action="store_true")
  return parser.parse_args()


def _validate_training_run(
  training_dir: Path,
  *,
  context_id: str,
  method_role: str,
  expected_seed: int,
  expected_context_hash: str,
  expected_protocol_commit: str,
  expected_protocol_sha256: str,
  expected_base_sha256: str,
  selected_beta: float,
) -> tuple[Path, dict[str, Any]]:
  summary_path = training_dir / "specialist_summary.json"
  checkpoint = training_dir / "accepted_final.pt"
  if not summary_path.is_file() or not checkpoint.is_file():
    raise FileNotFoundError(f"incomplete v21 training directory: {training_dir}")
  summary = json.loads(summary_path.read_text())
  expected_beta = 0.0 if method_role == "control" else selected_beta
  checks = {
    "context_id": summary.get("context_id") == context_id,
    "method_role": summary.get("method_role") == method_role,
    "adaptation_seed": summary.get("seed") == expected_seed,
    "beta": summary.get("matched_success_preservation_beta") == expected_beta,
    "rounds": len(summary.get("rounds", [])) == FORMAL_ROUNDS,
    "protocol_completed": summary.get("protocol_completed") is True,
    "base_checkpoint": summary.get("base_policy_checkpoint_sha256")
    == expected_base_sha256,
    "context": summary.get("deployment_context", {}).get("parameters_sha256")
    == expected_context_hash,
    "protocol_commit": summary.get("frozen_protocol", {}).get("git_commit")
    == expected_protocol_commit,
    "protocol_sha256": summary.get("frozen_protocol", {}).get("sha256")
    == expected_protocol_sha256,
    "monitor_unseen": summary.get("learning_curve_protocol", {}).get(
      "monitor_set_accessed_during_training"
    )
    is False,
  }
  failed = [name for name, passed in checks.items() if not passed]
  if failed:
    raise RuntimeError(
      f"invalid {context_id}/{method_role} training artifact: {failed}"
    )
  return checkpoint, {"summary": str(summary_path), "checks": checks, **summary}


def _pair_policy_rows(
  rows_by_role: dict[str, list[dict[str, str]]]
) -> dict[str, list[dict[str, str]]]:
  def key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["evaluation_seed"]), int(row["environment_id"])

  ordered = {
    role: sorted(rows, key=key) for role, rows in rows_by_role.items()
  }
  keys = {role: [key(row) for row in rows] for role, rows in ordered.items()}
  reference = keys["base"]
  if len(reference) != len(set(reference)) or any(
    keys[role] != reference for role in POLICY_ROLES[1:]
  ):
    raise RuntimeError("v21 tri-policy rows differ by seed/environment identity")
  return ordered


def _paired_interval(
  values: np.ndarray, *, bootstrap_samples: int, seed: int
) -> list[float]:
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1 or not len(values):
    raise ValueError("paired interval requires a non-empty vector")
  rng = np.random.default_rng(seed)
  means = np.empty(bootstrap_samples, dtype=np.float64)
  chunk = 1000
  for start in range(0, bootstrap_samples, chunk):
    stop = min(start + chunk, bootstrap_samples)
    indices = rng.integers(0, len(values), size=(stop - start, len(values)))
    means[start:stop] = values[indices].mean(axis=1)
  return [
    float(values.mean()),
    float(np.quantile(means, 0.025)),
    float(np.quantile(means, 0.975)),
  ]


def _comparison_metrics(
  old: list[dict[str, str]],
  new: list[dict[str, str]],
  *,
  bootstrap_samples: int,
  bootstrap_seed: int,
) -> dict[str, Any]:
  old_success = _binary_column(old, "success")
  new_success = _binary_column(new, "success")
  old_fell = _binary_column(old, "fell")
  new_fell = _binary_column(new, "fell")
  success_delta = new_success - old_success
  fall_delta = new_fell - old_fell
  rates = repair_regression_rates(
    [bool(value) for value in old_success],
    [bool(value) for value in new_success],
  )
  return {
    "paired_episode_count": len(old),
    "old_success_rate": float(old_success.mean()),
    "new_success_rate": float(new_success.mean()),
    "success_delta_mean_lcb95_ucb95": _paired_interval(
      success_delta,
      bootstrap_samples=bootstrap_samples,
      seed=bootstrap_seed,
    ),
    "old_fall_rate": float(old_fell.mean()),
    "new_fall_rate": float(new_fell.mean()),
    "fall_delta_mean_lcb95_ucb95": _paired_interval(
      fall_delta,
      bootstrap_samples=bootstrap_samples,
      seed=bootstrap_seed + 1,
    ),
    "repairs_regressions": rates,
  }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  if any(
    count % args.eval_batch_size
    for count in (args.target_episodes, args.d0_episodes)
  ):
    raise ValueError("v21 audit episode counts must divide into full batches")
  if not args.smoke and (
    args.target_episodes != FORMAL_TARGET_EPISODES
    or args.d0_episodes != FORMAL_D0_EPISODES
    or args.eval_batch_size != FORMAL_EVAL_BATCH_SIZE
    or args.bootstrap_samples != FORMAL_BOOTSTRAP_SAMPLES
  ):
    raise ValueError("formal v21 audit size differs from its prospective freeze")
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.protocol_commit:
    raise RuntimeError("v21 audit HEAD differs from the formal protocol commit")
  if not args.smoke and _git_output(
    repo, "status", "--porcelain", "--untracked-files=no"
  ):
    raise RuntimeError("formal v21 audit requires a clean tracked worktree")
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  protocol_sha256 = _sha256(protocol_path)
  relative_protocol = protocol_path.relative_to(repo)
  frozen_protocol = subprocess.run(
    ["git", "show", f"{current_commit}:{relative_protocol}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  if hashlib.sha256(frozen_protocol).hexdigest() != protocol_sha256:
    raise RuntimeError("v21 formal protocol differs from its committed blob")
  if (
    protocol.get("protocol_id") != PROTOCOL_ID
    or protocol.get("protocol_revision") != 2
    or protocol.get("status")
    != "prospectively_frozen_before_formal_adaptation"
  ):
    raise RuntimeError("unexpected v21 formal protocol")

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

  context_id = args.context_id
  context_path = args.context.resolve()
  context = load_calibrated_v21_context(context_path)
  sealed = protocol["sealed_inputs"]
  declared_context = sealed["contexts"][context_id]
  if not (
    context["context_id"] == context_id
    and context["formal_context"] is True
    and _sha256(context_path) == declared_context["file_sha256"]
    and context["parameters_sha256"] == declared_context["parameters_sha256"]
  ):
    raise RuntimeError("v21 audit context differs from its formal seal")
  checkpoint = args.base_policy_checkpoint.resolve()
  base_checkpoint_sha256 = _sha256(checkpoint)
  if base_checkpoint_sha256 != sealed["base_policy_checkpoint_sha256"]:
    raise RuntimeError("v21 audit base checkpoint differs from its seal")
  selected_beta = float(protocol["formal"]["selected_beta"])
  adaptation_seed = CONTEXT_ADAPTATION_SEEDS[context_id]
  control_checkpoint, control_training = _validate_training_run(
    args.control_training_dir.resolve(),
    context_id=context_id,
    method_role="control",
    expected_seed=adaptation_seed,
    expected_context_hash=context["parameters_sha256"],
    expected_protocol_commit=current_commit,
    expected_protocol_sha256=protocol_sha256,
    expected_base_sha256=base_checkpoint_sha256,
    selected_beta=selected_beta,
  )
  v21_checkpoint, v21_training = _validate_training_run(
    args.v21_training_dir.resolve(),
    context_id=context_id,
    method_role="v21",
    expected_seed=adaptation_seed,
    expected_context_hash=context["parameters_sha256"],
    expected_protocol_commit=current_commit,
    expected_protocol_sha256=protocol_sha256,
    expected_base_sha256=base_checkpoint_sha256,
    selected_beta=selected_beta,
  )
  if control_training["initial_actor_sha256"] != v21_training[
    "initial_actor_sha256"
  ]:
    raise RuntimeError("control and v21 adaptations did not start from one pi0")

  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  configure_v19_actor_interface(env_cfg, context)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = CONTEXT_FORMAL_AUDIT_SEEDS[context_id]
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  configure_v19_observable_refinement_runner(agent_cfg)
  alg_cfg = agent_cfg.algorithm
  alg_cfg.actor_learning_rate = 5.0e-6
  alg_cfg.critic_learning_rate = 1.0e-4
  alg_cfg.actor_layer_multipliers = (0.10, 0.25, 0.50, 1.0)
  alg_cfg.log_std_learning_rate = 0.0
  alg_cfg.std_scale_from_base = 0.35
  alg_cfg.pre_intervention_weight = 0.0
  alg_cfg.intervention_advantage_weight = 0.0
  alg_cfg.base_anchor_weight = 0.0
  alg_cfg.d0_retention_anchor_weight = 0.0
  alg_cfg.neighbor_retention_anchor_weight = 0.0
  alg_cfg.safe_bc_weight = 0.0
  alg_cfg.correction_distillation_weight = 0.0
  alg_cfg.brief_ppo_refinement = True
  alg_cfg.failure_focused_refinement = True
  alg_cfg.observable_failure_conditioned_refinement = True
  alg_cfg.task_first_constrained = False
  alg_cfg.actor_new_feature_count = 5
  alg_cfg.actor_new_feature_learning_rate_multiplier = 1.0
  alg_cfg.freeze_legacy_actor_input_columns = True
  alg_cfg.hard_case_policy_weight = 1.0
  alg_cfg.success_counterexample_policy_weight = 1.25
  alg_cfg.matched_success_preservation_beta = 0.0
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("v21 audit task has no online runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  warm_start = runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  if warm_start.get("pi0_exact_preservation_proof") is not True:
    raise RuntimeError("v21 audit pi0 expansion proof failed")
  actors = {"base": _actor_state(runner.alg.actor)}
  for role, path in (("control", control_checkpoint), ("v21", v21_checkpoint)):
    runner.load_online_checkpoint(str(path), map_location=args.device)
    actors[role] = _actor_state(runner.alg.actor)
  if _actor_state_sha256(actors["base"]) != control_training[
    "initial_actor_sha256"
  ]:
    raise RuntimeError("v21 audit pi0 differs from training")

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  raw: dict[str, Any] = {}
  paired_rows: list[dict[str, Any]] = []
  telemetry_rows: list[dict[str, Any]] = []
  comparison_results: dict[str, Any] = {}
  audit_seed = CONTEXT_FORMAL_AUDIT_SEEDS[context_id]
  mode = context["specialist_mode"]
  for evaluation_index, (evaluation_role, domain, episode_count) in enumerate(
    (
      ("target", "DQHMED", args.target_episodes),
      ("D0", "D0", args.d0_episodes),
    )
  ):
    repeats = episode_count // args.eval_batch_size
    evaluation_seed = audit_seed + 100_000 * evaluation_index
    summaries: dict[str, Any] = {}
    rows_by_role: dict[str, list[dict[str, str]]] = {}
    for role in POLICY_ROLES:
      root = output_dir / "raw" / role / evaluation_role
      summaries[role] = _evaluate_state(
        runner,
        actors[role],
        domains=(domain,),
        num_envs=args.eval_batch_size,
        num_episodes=args.eval_batch_size,
        seed=evaluation_seed,
        repeats=repeats,
        device=args.device,
        runtime_filter=True,
        artifact_dir=root,
        resume=True,
        deployment_context=context_path if evaluation_role == "target" else None,
        v19_context=context_path,
        telemetry_env_id=(
          TELEMETRY_ENVIRONMENT_ID_PER_BATCH
          if evaluation_role == "target"
          else None
        ),
      )[domain]
      rows_by_role[role] = _load_rows(
        root, domain=domain, first_seed=evaluation_seed, repeats=repeats
      )
      if len(rows_by_role[role]) != episode_count:
        raise RuntimeError("v21 formal audit raw row count differs")
      if evaluation_role == "target":
        for telemetry_file in summaries[role]["inline_telemetry_files"]:
          path = Path(telemetry_file)
          with path.open(newline="") as handle:
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
    if any(signatures[role] != signatures["base"] for role in POLICY_ROLES[1:]):
      raise RuntimeError("v21 formal tri-policy initial-state signatures differ")
    ordered = _pair_policy_rows(rows_by_role)
    comparison_results[evaluation_role] = {}
    for comparison_index, (name, old_role, new_role) in enumerate(COMPARISONS):
      comparison_results[evaluation_role][name] = _comparison_metrics(
        ordered[old_role],
        ordered[new_role],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=(
          FORMAL_BOOTSTRAP_SEED
          + 100 * list(V21_FORMAL_CONTEXTS).index(context_id)
          + 10 * evaluation_index
          + 2 * comparison_index
        ),
      )
    for pair_index, triplet in enumerate(
      zip(*(ordered[role] for role in POLICY_ROLES), strict=True)
    ):
      row: dict[str, Any] = {
        "context_id": context_id,
        "specialist_mode": mode,
        "evaluation_role": evaluation_role,
        "pair_index": pair_index,
        "evaluation_seed": int(triplet[0]["evaluation_seed"]),
        "environment_id": int(triplet[0]["environment_id"]),
      }
      for role, source in zip(POLICY_ROLES, triplet, strict=True):
        row.update(
          {
            f"{role}_success": int(source["success"] == "True"),
            f"{role}_fell": int(source["fell"] == "True"),
            f"{role}_failure_type": source["failure_type"],
            f"{role}_return": source["return"],
            f"{role}_max_riser": source["max_riser"],
          }
        )
      paired_rows.append(row)
    raw[evaluation_role] = summaries

  env.close()

  paired_csv = output_dir / "paired_episode_metrics.csv"
  _write_csv(paired_csv, PAIRED_FIELDS, paired_rows)
  telemetry_csv = output_dir / "inline_mechanism_telemetry.csv"
  if not telemetry_rows:
    raise RuntimeError("v21 formal audit produced no inline telemetry")
  _write_csv(telemetry_csv, list(telemetry_rows[0]), telemetry_rows)
  result = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "policy_method": POLICY_METHOD,
    "evidence_role": "fresh formal tri-policy paired deployment audit",
    "context_id": context_id,
    "specialist_mode": mode,
    "adaptation_seed": adaptation_seed,
    "selected_beta": selected_beta,
    "same_deployment_seed_for_control_and_v21": True,
    "same_pi0_for_control_and_v21": True,
    "same_evaluation_randomness_for_all_three_policies": True,
    "protocol_file": {
      "path": str(protocol_path),
      "sha256": protocol_sha256,
      "git_commit": current_commit,
    },
    "context": {
      "path": str(context_path),
      "file_sha256": _sha256(context_path),
      "parameters_sha256": context["parameters_sha256"],
      "family": context["context_family"],
      "calibration": context["calibration"],
    },
    "training": {"control": control_training, "v21": v21_training},
    "evaluation_protocol": {
      "target_paired_episodes": args.target_episodes,
      "d0_paired_episodes": args.d0_episodes,
      "eval_batch_size": args.eval_batch_size,
      "bootstrap_samples": args.bootstrap_samples,
      "audit_seed": audit_seed,
      "inline_telemetry_environment_id_per_batch": (
        TELEMETRY_ENVIRONMENT_ID_PER_BATCH
      ),
      "inline_telemetry_was_captured_during_formal_rollouts": True,
      "post_audit_mechanism_replay_used": False,
    },
    "comparisons": comparison_results,
    "paired_episode_metrics": {
      "path": str(paired_csv),
      "sha256": _sha256(paired_csv),
      "row_count": len(paired_rows),
    },
    "inline_mechanism_telemetry": {
      "path": str(telemetry_csv),
      "sha256": _sha256(telemetry_csv),
      "row_count": len(telemetry_rows),
      "same_rollout_outcomes_embedded": True,
    },
    "raw_evaluations": raw,
  }
  output = output_dir / "formal_audit_summary.json"
  temporary = output_dir / ".formal_audit_summary.json.tmp"
  temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  temporary.replace(output)
  print(
    json.dumps(
      {
        "output": str(output),
        "context_id": context_id,
        "comparisons": comparison_results,
        "paired_rows": len(paired_rows),
        "telemetry_rows": len(telemetry_rows),
      },
      indent=2,
      sort_keys=True,
    )
  )


if __name__ == "__main__":
  main()
