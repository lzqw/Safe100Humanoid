"""Freeze the first randomly generated DQH-Medium context with 75--85% base SR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from online_refine_stairs import _actor_state, _evaluate_state


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
  serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  if path.exists():
    if path.read_text() != serialized:
      raise RuntimeError(f"refusing to overwrite a different frozen artifact: {path}")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(serialized)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--base-policy-checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--context-output", type=Path, required=True)
  parser.add_argument(
    "--candidate-seeds", nargs="+", type=int, default=tuple(range(1000, 1020))
  )
  parser.add_argument("--num-episodes", type=int, default=128)
  parser.add_argument("--eval-batch-size", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  candidate_seeds = list(args.candidate_seeds)
  if candidate_seeds != sorted(candidate_seeds):
    raise ValueError("calibration candidate seeds must be strictly ordered")
  if len(set(candidate_seeds)) != len(candidate_seeds):
    raise ValueError("calibration candidate seeds must be distinct")
  if args.eval_batch_size != args.num_episodes:
    raise ValueError("calibration requires one independent episode per environment")
  if args.num_episodes < 1:
    raise ValueError("calibration episode count must be positive")

  repo = args.repo.resolve()
  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.deployment_context import (
    FAILURE_FOCUSED_CALIBRATION_KIND,
    generate_failure_focused_context,
    validate_calibrated_deployment_context,
  )

  checkpoint = args.base_policy_checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  task = "Unitree-G1-Stairs-Online-DQH"
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = candidate_seeds[0]
  env_cfg.actions["joint_pos"].enabled = True
  agent_cfg = load_rl_cfg(task)
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("online refinement task has no custom runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  runner.load_online_checkpoint(str(checkpoint), map_location=args.device)
  base_actor = _actor_state(runner.alg.actor)

  attempts: list[dict[str, Any]] = []
  selected_payload: dict[str, Any] | None = None
  try:
    for candidate_seed in candidate_seeds:
      candidate_payload = generate_failure_focused_context(candidate_seed)
      candidate_dir = output_dir / f"candidate_seed{candidate_seed}"
      candidate_path = candidate_dir / "context.json"
      _write_immutable_json(candidate_path, candidate_payload)
      evaluation = _evaluate_state(
        runner,
        base_actor,
        domains=("DQHMED",),
        num_envs=args.eval_batch_size,
        num_episodes=args.num_episodes,
        seed=candidate_seed,
        device=args.device,
        runtime_filter=True,
        artifact_dir=candidate_dir / "evaluation",
        resume=True,
        deployment_context=candidate_path,
      )["DQHMED"]
      evaluated_hash = evaluation["replicates"][0]["deployment_context"][
        "parameters_sha256"
      ]
      if evaluated_hash != candidate_payload["parameters_sha256"]:
        raise RuntimeError("evaluation used a different deployment context")
      success_rate = float(evaluation["success_rate"])
      attempt = {
        "candidate_seed": candidate_seed,
        "evaluation_seed": candidate_seed,
        "parameters_sha256": candidate_payload["parameters_sha256"],
        "base_policy_only": True,
        "num_episodes": args.num_episodes,
        "success_rate": success_rate,
        "fall_rate": float(evaluation["fall_rate"]),
        "mean_reached_riser": float(evaluation["mean_reached_riser"]),
        "intervention_per_riser": float(evaluation["intervention_per_riser"]),
        "qualifies_by_success_rate_only": 0.75 <= success_rate <= 0.85,
      }
      attempts.append(attempt)
      progress = {
        "selection_rule": "first context with 0.75 <= base success rate <= 0.85",
        "candidate_seeds": candidate_seeds,
        "base_policy_checkpoint": str(checkpoint),
        "base_policy_checkpoint_sha256": _sha256(checkpoint),
        "attempts": attempts,
      }
      (output_dir / "calibration_progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n"
      )
      if attempt["qualifies_by_success_rate_only"]:
        selected_payload = candidate_payload
        break
  finally:
    env.close()

  if selected_payload is None:
    raise RuntimeError(
      "none of the declared calibration contexts produced 75--85% base success"
    )
  calibration = {
    "kind": FAILURE_FOCUSED_CALIBRATION_KIND,
    "selection_rule": "select the first context with 0.75 <= base success rate <= 0.85",
    "selection_metric_fields": ["success_rate"],
    "success_rate_bounds": [0.75, 0.85],
    "candidate_seeds": candidate_seeds,
    "attempts": attempts,
    "selected_candidate_seed": selected_payload["calibration_candidate_seed"],
    "selected_parameters_sha256": selected_payload["parameters_sha256"],
    "base_policy_checkpoint": str(checkpoint),
    "base_policy_checkpoint_sha256": _sha256(checkpoint),
    "adapted_policy_evaluations_used": False,
    "calibration_and_training_seeds_disjoint": True,
  }
  selected_payload["calibration"] = calibration
  selected_payload = validate_calibrated_deployment_context(selected_payload)
  context_output = args.context_output.resolve()
  _write_immutable_json(context_output, selected_payload)
  result = {
    "protocol": "Failure-Focused Brief PPO v15 context calibration",
    "selected": True,
    "frozen_context": str(context_output),
    "frozen_context_file_sha256": _sha256(context_output),
    "parameters_sha256": selected_payload["parameters_sha256"],
    "selected_candidate_seed": selected_payload["calibration_candidate_seed"],
    "selected_base_success_rate": attempts[-1]["success_rate"],
    "selection_used_only_base_success_rate": True,
    "adapted_policy_evaluations_used": False,
    "calibration": calibration,
  }
  summary_path = output_dir / "calibration_summary.json"
  summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
