"""Headless integration smoke test for the G1 stair CBF task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--task", default="Unitree-G1-Stairs-CBF")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--expected-actions", type=int, default=12)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  sys.path.insert(0, str(args.repo))

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  obs, _ = env.reset(seed=args.seed)
  term = env.action_manager.get_term("joint_pos")
  action_dim = env.action_manager.total_action_dim
  robot = env.scene["robot"]

  # Put the left foot just before the first riser, mark it as the swing foot,
  # and construct a nominal joint velocity along +J_x. This creates a real
  # simulator/Jacobian CBF violation without waiting for an untrained policy to
  # walk to the staircase.
  terrain = env.scene.terrain
  assert terrain is not None
  level, terrain_type = terrain.terrain_levels, terrain.terrain_types
  first_edge = term._edge_x[level, terrain_type, 0]
  foot_pos = robot.data.site_pos_w[:, term._site_local_ids]
  desired_foot_x = first_edge - term.cfg.toe_margin - 0.02
  pose = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w], dim=1)
  pose[:, 0] += desired_foot_x - foot_pos[:, 0, 0]
  robot.write_root_link_pose_to_sim(pose)
  env.sim.forward()
  env.sim.sense()
  found = term._contact_sensor.data.found
  assert found is not None and found.shape[1] == 2
  found[:, 0] = 0
  found[:, 1] = 1
  air_time = term._contact_sensor.data.current_air_time
  if air_time is not None:
    air_time[:, 0] = 1.0
    air_time[:, 1] = 0.0
  foot_pos = robot.data.site_pos_w[:, term._site_local_ids]
  jac_x = term._foot_jacobians(foot_pos)[:, 0]
  q = robot.data.joint_pos[:, term.target_ids]
  qdot_attack = 50.0 * jac_x
  target_attack = q + env.step_dt * qdot_attack
  raw_attack = (target_attack - term.offset) / term.scale
  env.action_manager.process_action(raw_attack)
  adversarial = {
    "filter_enabled": bool(term.cfg.enabled),
    "active": bool(term.filter_active.all()),
    "nominal_margin_min": float(term.psi_nominal.min()),
    "filtered_margin_min": float(term.psi_filtered.min()),
    "intervention_norm_mean": float(term.intervention_norm.mean()),
  }
  obs, _ = env.reset(seed=args.seed)
  active_count = 0
  violation_count = 0
  filtered_min = float("inf")
  finite = True
  reward_std_samples = []
  try:
    for _ in range(args.steps):
      actions = 2.0 * torch.rand((args.num_envs, action_dim), device=env.device) - 1.0
      obs, reward, terminated, timeout, _ = env.step(actions)
      del terminated, timeout
      finite &= bool(torch.isfinite(reward).all())
      finite &= all(bool(torch.isfinite(v).all()) for v in obs.values())
      active = term.filter_active
      active_count += int(active.sum())
      violation_count += int((term.psi_nominal[active] < 0).sum())
      if active.any():
        filtered_min = min(filtered_min, float(term.psi_filtered[active].min()))
      reward_std_samples.append(float(reward.std(unbiased=False)))
  finally:
    env.close()

  result = {
    "task": args.task,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "action_dim": action_dim,
    "adversarial_filter_check": adversarial,
    "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
    "finite": finite,
    "cbf_active_count": active_count,
    "nominal_violation_count": violation_count,
    "filtered_margin_min": None if filtered_min == float("inf") else filtered_min,
    "reward_std_mean": sum(reward_std_samples) / len(reward_std_samples),
    "first_tread_top_z_by_level": term._edge_top_z[:, 0, 0].tolist(),
  }
  print(json.dumps(result, indent=2, sort_keys=True))
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  if action_dim != args.expected_actions or not finite:
    raise SystemExit(2)
  levels = result["first_tread_top_z_by_level"]
  if len(levels) > 1 and any(b <= a for a, b in zip(levels, levels[1:])):
    raise SystemExit(6)
  if adversarial["filter_enabled"]:
    if (
      not adversarial["active"]
      or adversarial["nominal_margin_min"] >= 0.0
      or adversarial["filtered_margin_min"] < -1.0e-4
    ):
      raise SystemExit(4)
  elif (
    adversarial["active"]
    or abs(
      adversarial["filtered_margin_min"] - adversarial["nominal_margin_min"]
    ) > 1.0e-5
  ):
    raise SystemExit(5)
  if active_count and filtered_min < -1.0e-4:
    raise SystemExit(3)


if __name__ == "__main__":
  main()
