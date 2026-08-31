"""Deterministic original-interface evaluator for CBF-Proximal PPO v23."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from proximal_v23_io import actor_state_sha256


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, required=True)
  parser.add_argument("--num-episodes", type=int, required=True)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--deployment-context", type=Path)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path, required=True)
  return parser.parse_args()


def _initial_state_signature(obs, base_env, action_term, command_term) -> str:
  signature = hashlib.sha256()
  tensors = [
    obs["actor"],
    base_env.scene["robot"].data.root_link_pos_w,
    base_env.scene["robot"].data.root_link_quat_w,
    base_env.scene["robot"].data.joint_pos,
    base_env.command_manager.get_command("twist"),
    getattr(
      command_term,
      "raw_command",
      base_env.command_manager.get_command("twist"),
    ),
    getattr(
      command_term,
      "delay_steps",
      torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device),
    ),
    getattr(
      command_term,
      "_delay_queue",
      torch.zeros(base_env.num_envs, 1, 3, device=base_env.device),
    ),
    getattr(
      action_term,
      "_deployment_action_queue",
      torch.zeros(
        base_env.num_envs,
        1,
        action_term.action_dim,
        device=base_env.device,
      ),
    ),
  ]
  for tensor in tensors:
    signature.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return signature.hexdigest()


def main() -> None:
  args = _parse_args()
  if args.num_envs != args.num_episodes or args.num_envs < 1:
    raise ValueError("v23 pairing requires one initial episode per environment")
  repo = args.repo.resolve()
  checkpoint = args.checkpoint.resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

  sys.path.insert(0, str(repo))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from src.tasks.stairs_cbf.deployment_context import load_calibrated_v22_context
  from src.tasks.stairs_cbf.proximal_context import apply_cbf_proximal_context

  cfg = load_env_cfg(args.task, play=True)
  context_metadata = None
  if args.deployment_context is not None:
    context = load_calibrated_v22_context(args.deployment_context.resolve())
    context_metadata = apply_cbf_proximal_context(cfg, context, role="target")
  elif args.task.endswith("DQHMED"):
    raise ValueError("the v23 target task requires its frozen lateral context")
  # D0 inherits the historical zero-weight placeholder.  Remove it as well so
  # every v23 runtime, not only the target context, has no specialist term.
  cfg.rewards.pop("specialist_failure_signal", None)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.actions["joint_pos"].enabled = True
  if "deployable_failure" in cfg.observations:
    raise RuntimeError("v23 evaluator contains the forbidden failure observation")
  if "specialist_failure_signal" in cfg.rewards:
    raise RuntimeError("v23 evaluator contains a specialist reward term")

  base_env = ManagerBasedRlEnv(cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  if runner_cls is None:
    raise RuntimeError("v23 evaluation task has no runner")
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=args.device)
  try:
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    if int(runner.alg.actor.obs_dim) != 405:
      raise RuntimeError("v23 evaluator actor is not the original 405-D policy")
    actor_hash = actor_state_sha256(runner.alg.actor.state_dict())
    policy = runner.get_inference_policy(args.device)
    obs, _ = env.reset()
    action_term = base_env.action_manager.get_term("joint_pos")
    command_term = base_env.command_manager.get_term("twist")
    initial_signature = _initial_state_signature(
      obs, base_env, action_term, command_term
    )
    num_risers = int(action_term._edge_x.shape[-1])
    active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
    returns = torch.zeros(args.num_envs, device=args.device)
    steps = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
    max_riser = torch.zeros_like(steps)
    intervention_count = torch.zeros_like(steps)
    correction_sum = torch.zeros(args.num_envs, device=args.device)
    completed: list[dict[str, Any]] = []
    maximum_steps = int(base_env.max_episode_length) + 2
    with torch.inference_mode():
      for _ in range(maximum_steps):
        actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        active_float = active.float()
        returns += rewards * active_float
        steps += active.long()
        root_x = base_env.scene["robot"].data.root_link_pos_w[:, 0:1]
        edge_x = action_term._edge_x[
          base_env.scene.terrain.terrain_levels,
          base_env.scene.terrain.terrain_types,
        ]
        current_riser = torch.sum(root_x >= edge_x, dim=1)
        max_riser = torch.maximum(max_riser, current_riser * active.long())
        intervention_count += action_term.intervened.long() * active.long()
        correction_sum += action_term.target_intervention_norm * active_float
        record_mask = dones.bool() & active
        if bool(record_mask.any()):
          fell = base_env.termination_manager.get_term("fell_over").bool()
          timed_out = base_env.termination_manager.get_term("time_out").bool()
          success = base_env.termination_manager.get_term("reached_top").bool()
          for env_id in record_mask.nonzero(as_tuple=False).flatten().tolist():
            reached = int(max_riser[env_id])
            episode_steps = max(1, int(steps[env_id]))
            completed.append(
              {
                "evaluation_seed": args.seed,
                "environment_id": env_id,
                "success": bool(success[env_id]),
                "fell": bool(fell[env_id]),
                "timed_out": bool(timed_out[env_id]),
                "return": float(returns[env_id]),
                "steps": episode_steps,
                "max_riser": reached,
                "completion_fraction": reached / num_risers,
                "intervention_count": int(intervention_count[env_id]),
                "intervention_per_riser": float(
                  intervention_count[env_id] / max(1, reached)
                ),
                "correction_mean": float(
                  correction_sum[env_id] / episode_steps
                ),
              }
            )
          active &= ~record_mask
          if not bool(active.any()):
            break
    if bool(active.any()) or len(completed) != args.num_episodes:
      raise RuntimeError("v23 evaluator did not finish every initial episode")
    completed.sort(key=lambda row: int(row["environment_id"]))
    summary = {
      "schema_version": 1,
      "task": args.task,
      "seed": args.seed,
      "num_envs": args.num_envs,
      "num_episodes": len(completed),
      "runtime_filter": True,
      "deterministic_policy_mean": True,
      "one_initial_episode_per_env": True,
      "original_observation_interface": True,
      "actor_observation_dim": 405,
      "actor_state_sha256": actor_hash,
      "initial_state_signature": initial_signature,
      "success_rate": sum(row["success"] for row in completed) / len(completed),
      "fall_rate": sum(row["fell"] for row in completed) / len(completed),
      "timeout_rate": sum(row["timed_out"] for row in completed) / len(completed),
      "mean_return": sum(row["return"] for row in completed) / len(completed),
      "intervention_per_riser": sum(
        row["intervention_per_riser"] for row in completed
      )
      / len(completed),
      "correction_mean": sum(row["correction_mean"] for row in completed)
      / len(completed),
      "deployment_context": context_metadata,
    }
  finally:
    env.close()

  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  with args.output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
    writer.writeheader()
    writer.writerows(completed)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
