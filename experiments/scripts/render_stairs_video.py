"""Record a finite deterministic G1 stair-climbing rollout to MP4."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", default="Unitree-G1-Stairs-CBF")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--output-json", type=Path, required=True)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--step-height", type=float)
  parser.add_argument("--max-steps", type=int, default=1000)
  parser.add_argument("--width", type=int, default=854)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--runtime-filter", choices=("on", "off"), default="off")
  args = parser.parse_args()

  sys.path.insert(0, str(args.repo.resolve()))
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.wrappers import VideoRecorder

  device = "cuda:0"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = (
    args.max_steps * env_cfg.decimation * env_cfg.sim.mujoco.timestep
  )
  env_cfg.seed = args.seed
  terrain_cfg = env_cfg.scene.terrain
  if terrain_cfg is None or terrain_cfg.terrain_generator is None:
    raise RuntimeError("task has no generated staircase terrain")
  stairs_cfg = terrain_cfg.terrain_generator.sub_terrains["forward_stairs"]
  if args.step_height is not None:
    stairs_cfg.step_height_range = (args.step_height, args.step_height)
    if hasattr(stairs_cfg, "step_height_profile"):
      stairs_cfg.step_height_profile = None
  terrain_cfg.terrain_generator.num_rows = 1
  terrain_cfg.max_init_terrain_level = 0
  env_cfg.actions["joint_pos"].enabled = args.runtime_filter == "on"
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  env_cfg.viewer.distance = 4.0
  env_cfg.viewer.azimuth = 90.0
  env_cfg.viewer.elevation = -10.0

  base_env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
  task_slug = args.task.lower().replace("unitree-g1-stairs-", "").replace("-", "_")
  prefix = f"g1-stairs-{task_slug}-filter-{args.runtime_filter}-seed{args.seed}"
  recorded_env = VideoRecorder(
    base_env,
    video_folder=args.output_dir,
    step_trigger=lambda step: step == 0,
    video_length=args.max_steps,
    name_prefix=prefix,
  )
  agent_cfg = load_rl_cfg(args.task)
  env = RslRlVecEnvWrapper(recorded_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(args.checkpoint.resolve()),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  obs, _ = env.reset()
  robot = base_env.scene["robot"]
  action_term = base_env.action_manager.get_term("joint_pos")
  command_term = base_env.command_manager.get_term("twist")
  stair_half_width = float(getattr(command_term.cfg, "stair_half_width", 1.20))
  max_progress = 0.0
  min_root_height = float("inf")
  violation_integral = 0.0
  violation_events = 0
  intervention_integral = 0.0
  episode_return = 0.0
  fell = False
  timed_out = False
  steps = 0
  max_riser = 0
  reached_top = False
  centerline_error_integral = 0.0
  max_abs_centerline_error = 0.0
  max_abs_heading_error = 0.0
  min_root_edge_clearance = float("inf")
  min_foot_edge_clearance = float("inf")
  operator_correction_steps = 0

  try:
    with torch.inference_mode():
      for step in range(args.max_steps):
        progress = float(
          robot.data.root_link_pos_w[0, 0] - base_env.scene.env_origins[0, 0]
        )
        root_height = float(
          robot.data.root_link_pos_w[0, 2] - base_env.scene.env_origins[0, 2]
        )
        max_progress = max(max_progress, progress)
        min_root_height = min(min_root_height, root_height)
        action = policy(obs)
        obs, reward, done, _ = env.step(action)
        episode_return += float(reward[0])
        active = bool(torch.isfinite(action_term.h[0]))
        violation = max(-float(action_term.psi_nominal[0]), 0.0)
        violation_integral += violation
        violation_events += int(active and float(action_term.psi_nominal[0]) < -1.0e-5)
        intervention_integral += float(action_term.intervention_norm[0])
        centerline_error = abs(float(command_term.centerline_error[0]))
        heading_error = abs(float(command_term.heading_error[0]))
        centerline_error_integral += centerline_error
        max_abs_centerline_error = max(
          max_abs_centerline_error, centerline_error
        )
        max_abs_heading_error = max(max_abs_heading_error, heading_error)
        min_root_edge_clearance = min(
          min_root_edge_clearance, stair_half_width - centerline_error
        )
        patches = base_env.scene.terrain.flat_patches["stair_targets"]
        patches = patches[
          base_env.scene.terrain.terrain_levels,
          base_env.scene.terrain.terrain_types,
        ]
        center_y = patches[0, 0, 1]
        foot_y = robot.data.site_pos_w[0, action_term._site_local_ids, 1]
        foot_clearance = stair_half_width - float(
          torch.max(torch.abs(foot_y - center_y))
        )
        min_foot_edge_clearance = min(
          min_foot_edge_clearance, foot_clearance
        )
        operator_correction_steps += int(command_term.correction_active[0])
        if hasattr(action_term, "_edge_x"):
          terrain = base_env.scene.terrain
          assert terrain is not None
          risers = action_term._edge_x[
            terrain.terrain_levels, terrain.terrain_types
          ]
          current_riser = int(
            torch.sum(robot.data.root_link_pos_w[:, 0:1] >= risers, dim=1)[0]
          )
          max_riser = max(max_riser, current_riser)
        steps = step + 1
        if bool(done[0]):
          fell = bool(base_env.termination_manager.get_term("fell_over")[0])
          timed_out = bool(base_env.termination_manager.get_term("time_out")[0])
          if "reached_top" in base_env.termination_manager.active_terms:
            reached_top = bool(
              base_env.termination_manager.get_term("reached_top")[0]
            )
          break
  finally:
    env.close()

  video_path = args.output_dir / f"{prefix}-step-0.mp4"
  result = {
    "task": args.task,
    "checkpoint": str(args.checkpoint.resolve()),
    "video": str(video_path.resolve()),
    "seed": args.seed,
    "runtime_filter": args.runtime_filter,
    "step_height_m": args.step_height,
    "steps": steps,
    "return": episode_return,
    "max_progress_m": max_progress,
    "min_root_height_m": min_root_height,
    "max_riser": max_riser,
    "success": reached_top if "Online" in args.task else max_progress >= 2.65,
    "fell": fell,
    "timed_out": timed_out,
    "cbf_violation_events": violation_events,
    "cbf_violation_integral": violation_integral,
    "cbf_intervention_integral": intervention_integral,
    "mean_abs_centerline_error_m": (
      centerline_error_integral / max(1, steps)
    ),
    "max_abs_centerline_error_m": max_abs_centerline_error,
    "max_abs_heading_error_rad": max_abs_heading_error,
    "minimum_root_edge_clearance_m": min_root_edge_clearance,
    "minimum_foot_edge_clearance_m": min_foot_edge_clearance,
    "operator_correction_fraction": operator_correction_steps / max(1, steps),
    "side_edge_breach": (
      min_root_edge_clearance < 0.0 or min_foot_edge_clearance < 0.0
    ),
  }
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
