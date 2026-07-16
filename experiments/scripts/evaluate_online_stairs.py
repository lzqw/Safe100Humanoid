"""Geometry-aware deterministic evaluation for online stair refinement."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def evaluate_policy(
  policy,
  *,
  task: str,
  num_envs: int,
  num_episodes: int,
  seed: int,
  device: str,
  runtime_filter: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  # Reset every RNG used by reset events and command generation so old and
  # candidate policies receive paired initial states and joystick traces.
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = seed
  cfg.actions["joint_pos"].enabled = runtime_filter
  base_env = ManagerBasedRlEnv(cfg, device=device)
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  # The wrapper does not forward reset kwargs; cfg.seed already fixes the
  # environment/randomization sequence for paired old/candidate evaluation.
  obs, _ = env.reset()
  policy.eval()
  action_term = base_env.action_manager.get_term("joint_pos")
  n_risers = action_term._edge_x.shape[-1]

  returns = torch.zeros(num_envs, device=device)
  steps = torch.zeros(num_envs, device=device)
  max_riser = torch.zeros(num_envs, dtype=torch.long, device=device)
  min_h = torch.full((num_envs,), torch.inf, device=device)
  interventions = torch.zeros(num_envs, device=device)
  corrections = torch.zeros(num_envs, device=device)
  correction_max = torch.zeros(num_envs, device=device)
  completed: list[dict[str, Any]] = []

  try:
    with torch.inference_mode():
      while len(completed) < num_episodes:
        actions = policy(obs)
        obs, reward, done, _ = env.step(actions)
        returns += reward
        steps += 1
        root_x = base_env.scene["robot"].data.root_link_pos_w[:, 0:1]
        risers = action_term._edge_x[
          base_env.scene.terrain.terrain_levels,
          base_env.scene.terrain.terrain_types,
        ]
        current_riser = torch.sum(root_x >= risers, dim=1)
        max_riser = torch.maximum(max_riser, current_riser)
        finite_h = torch.where(
          torch.isfinite(action_term.h),
          action_term.h,
          torch.full_like(action_term.h, torch.inf),
        )
        min_h = torch.minimum(min_h, finite_h)
        interventions += action_term.intervened.float()
        corrections += action_term.target_intervention_norm
        correction_max = torch.maximum(
          correction_max, action_term.target_intervention_norm
        )

        done_ids = done.nonzero(as_tuple=False).flatten()
        if len(done_ids) == 0:
          continue
        fell_all = base_env.termination_manager.get_term("fell_over")
        timeout_all = base_env.termination_manager.get_term("time_out")
        success_all = base_env.termination_manager.get_term("reached_top")
        for env_id in done_ids.tolist():
          reached = int(max_riser[env_id])
          completed.append(
            {
              "episode": len(completed),
              "success": bool(success_all[env_id]),
              "fell": bool(fell_all[env_id]),
              "timed_out": bool(timeout_all[env_id]),
              "return": float(returns[env_id]),
              "steps": int(steps[env_id]),
              "max_riser": reached,
              "completion_fraction": reached / n_risers,
              "minimum_cbf_h": (
                None if torch.isinf(min_h[env_id]) else float(min_h[env_id])
              ),
              "intervention_count": int(interventions[env_id]),
              "intervention_per_riser": float(
                interventions[env_id] / max(1, reached)
              ),
              "correction_mean": float(
                corrections[env_id] / max(1.0, float(steps[env_id]))
              ),
              "correction_max": float(correction_max[env_id]),
            }
          )
          if len(completed) >= num_episodes:
            break
        returns[done_ids] = 0.0
        steps[done_ids] = 0.0
        max_riser[done_ids] = 0
        min_h[done_ids] = torch.inf
        interventions[done_ids] = 0.0
        corrections[done_ids] = 0.0
        correction_max[done_ids] = 0.0
  finally:
    env.close()

  completed = completed[:num_episodes]
  survival = {}
  hazard = {}
  for k in range(1, n_risers + 1):
    reached_k = sum(int(row["max_riser"]) >= k for row in completed)
    reached_previous = sum(int(row["max_riser"]) >= k - 1 for row in completed)
    failed_at_k = sum(
      (not bool(row["success"])) and int(row["max_riser"]) == k - 1
      for row in completed
    )
    survival[str(k)] = reached_k / len(completed)
    hazard[str(k)] = failed_at_k / max(1, reached_previous)

  finite_h_values = [
    float(row["minimum_cbf_h"])
    for row in completed
    if row["minimum_cbf_h"] is not None
  ]
  summary = {
    "task": task,
    "seed": seed,
    "num_envs": num_envs,
    "num_episodes": len(completed),
    "num_risers": n_risers,
    "runtime_filter": runtime_filter,
    "success_rate": sum(bool(row["success"]) for row in completed) / len(completed),
    "fall_rate": sum(bool(row["fell"]) for row in completed) / len(completed),
    "timeout_rate": sum(bool(row["timed_out"]) for row in completed) / len(completed),
    "mean_reached_riser": sum(int(row["max_riser"]) for row in completed) / len(completed),
    "intervention_per_riser": sum(
      float(row["intervention_per_riser"]) for row in completed
    ) / len(completed),
    "correction_mean": sum(float(row["correction_mean"]) for row in completed) / len(completed),
    "minimum_cbf_h": min(finite_h_values) if finite_h_values else None,
    "survival_curve": survival,
    "conditional_failure_hazard": hazard,
  }
  return summary, completed


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--num-episodes", type=int, default=64)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--runtime-filter", choices=("on", "off"), default="on")
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo.resolve()))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(args.device)
  env.close()

  summary, episodes = evaluate_policy(
    policy,
    task=args.task,
    num_envs=args.num_envs,
    num_episodes=args.num_episodes,
    seed=args.seed,
    device=args.device,
    runtime_filter=args.runtime_filter == "on",
  )
  summary["checkpoint"] = str(args.checkpoint)
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  with args.output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
    writer.writeheader()
    writer.writerows(episodes)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
