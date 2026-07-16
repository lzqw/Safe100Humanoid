"""Deterministic batched evaluation for the G1 stair CBF experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--label", required=True)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--num-episodes", type=int, default=256)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--terrain-level", type=int, default=-1)
  parser.add_argument("--fixed-step-height", type=float)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo))

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = "cuda:0"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  if args.fixed_step_height is not None:
    terrain_cfg = env_cfg.scene.terrain
    if terrain_cfg is None or terrain_cfg.terrain_generator is None:
      raise RuntimeError("task has no generated terrain for --fixed-step-height")
    stairs_cfg = terrain_cfg.terrain_generator.sub_terrains["forward_stairs"]
    stairs_cfg.step_height_range = (
      args.fixed_step_height,
      args.fixed_step_height,
    )
    terrain_cfg.terrain_generator.num_rows = 1
    terrain_cfg.max_init_terrain_level = 0
  if args.terrain_level >= 0:
    if env_cfg.scene.terrain is None:
      raise RuntimeError("task has no terrain for --terrain-level")
    env_cfg.scene.terrain.max_init_terrain_level = args.terrain_level
  # Use the training horizon for a meaningful and bounded comparison while
  # keeping play-mode observation corruption and pushes disabled.
  env_cfg.episode_length_s = 20.0
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(args.task)
  base_env = ManagerBasedRlEnv(env_cfg, device=device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  obs, _ = env.reset()
  action_term = base_env.action_manager.get_term("joint_pos")
  robot = base_env.scene["robot"]

  n = args.num_envs
  episode_return = torch.zeros(n, device=device)
  episode_steps = torch.zeros(n, device=device)
  max_progress = torch.zeros(n, device=device)
  min_root_height = torch.full((n,), torch.inf, device=device)
  violation_integral = torch.zeros(n, device=device)
  violation_events = torch.zeros(n, device=device)
  geometrically_active_steps = torch.zeros(n, device=device)
  intervention_integral = torch.zeros(n, device=device)
  target_intervention_integral = torch.zeros(n, device=device)
  completed: list[dict[str, float | int | bool | str]] = []
  # Top platform starts at relative x=2.70 m. Allow 5 cm for root/site lag.
  success_progress = 2.65

  with torch.inference_mode():
    while len(completed) < args.num_episodes:
      progress_before = robot.data.root_link_pos_w[:, 0] - base_env.scene.env_origins[:, 0]
      height_before = robot.data.root_link_pos_w[:, 2] - base_env.scene.env_origins[:, 2]
      max_progress = torch.maximum(max_progress, progress_before)
      min_root_height = torch.minimum(min_root_height, height_before)
      action = policy(obs)
      obs, reward, done, _ = env.step(action)

      active = torch.isfinite(action_term.h)
      violation = torch.relu(-action_term.psi_nominal)
      episode_return += reward
      episode_steps += 1
      violation_integral += violation
      violation_events += (active & (action_term.psi_nominal < -1.0e-5)).float()
      geometrically_active_steps += active.float()
      intervention_integral += action_term.intervention_norm
      target_intervention_integral += action_term.target_intervention_norm

      done_ids = done.nonzero(as_tuple=False).flatten()
      if len(done_ids) == 0:
        continue
      fell = base_env.termination_manager.get_term("fell_over")[done_ids]
      timed_out = base_env.termination_manager.get_term("time_out")[done_ids]
      for env_id in done_ids.tolist():
        steps = max(float(episode_steps[env_id]), 1.0)
        completed.append(
          {
            "label": args.label,
            "task": args.task,
            "episode": len(completed),
            "return": float(episode_return[env_id]),
            "steps": int(episode_steps[env_id]),
            "max_progress_m": float(max_progress[env_id]),
            "min_root_height_m": float(min_root_height[env_id]),
            "success": bool(max_progress[env_id] >= success_progress),
            "fell": bool(fell[(done_ids == env_id).nonzero()[0, 0]]),
            "timed_out": bool(timed_out[(done_ids == env_id).nonzero()[0, 0]]),
            "cbf_violation_integral": float(violation_integral[env_id]),
            "cbf_violation_events": int(violation_events[env_id]),
            "cbf_active_fraction": float(geometrically_active_steps[env_id] / steps),
            "cbf_intervention_integral": float(intervention_integral[env_id]),
            "cbf_target_intervention_integral": float(
              target_intervention_integral[env_id]
            ),
          }
        )
        if len(completed) >= args.num_episodes:
          break
      episode_return[done_ids] = 0
      episode_steps[done_ids] = 0
      max_progress[done_ids] = 0
      min_root_height[done_ids] = torch.inf
      violation_integral[done_ids] = 0
      violation_events[done_ids] = 0
      geometrically_active_steps[done_ids] = 0
      intervention_integral[done_ids] = 0
      target_intervention_integral[done_ids] = 0

  env.close()
  completed = completed[: args.num_episodes]
  numeric_means = {}
  for key in (
    "return",
    "steps",
    "max_progress_m",
    "min_root_height_m",
    "cbf_violation_integral",
    "cbf_violation_events",
    "cbf_active_fraction",
    "cbf_intervention_integral",
    "cbf_target_intervention_integral",
  ):
    numeric_means[key] = sum(float(row[key]) for row in completed) / len(completed)
  result = {
    "label": args.label,
    "task": args.task,
    "checkpoint": str(args.checkpoint.resolve()),
    "seed": args.seed,
    "terrain_level": args.terrain_level,
    "fixed_step_height_m": args.fixed_step_height,
    "num_envs": args.num_envs,
    "num_episodes": len(completed),
    "success_threshold_progress_m": success_progress,
    "success_rate": sum(bool(row["success"]) for row in completed) / len(completed),
    "fall_rate": sum(bool(row["fell"]) for row in completed) / len(completed),
    "timeout_rate": sum(bool(row["timed_out"]) for row in completed) / len(completed),
    "means": numeric_means,
  }
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_csv.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  with args.output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
    writer.writeheader()
    writer.writerows(completed)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
