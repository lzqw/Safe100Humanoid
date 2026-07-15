"""CBF rewards and diagnostics exposed to MJLab managers."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from .actions import StairCbfJointPositionAction
from .cbf_math import dual_cbf_reward


def _cbf_term(env: ManagerBasedRlEnv, action_name: str) -> StairCbfJointPositionAction:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, StairCbfJointPositionAction):
    raise TypeError(f"action {action_name!r} is not StairCbfJointPositionAction")
  return term


def cbf_violation(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  """Positive violation magnitude ``relu(-psi_nominal)``."""
  term = _cbf_term(env, action_name)
  value = torch.relu(-term.psi_nominal)
  env.extras["log"]["CBF/violation_mean"] = value.mean()
  env.extras["log"]["CBF/active_fraction"] = term.filter_active.float().mean()
  env.extras["log"]["CBF/filtered_margin_min"] = term.psi_filtered.min()
  return value


def cbf_intervention(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  term = _cbf_term(env, action_name)
  env.extras["log"]["CBF/intervention_norm_mean"] = term.intervention_norm.mean()
  return term.intervention_norm.square()


def cbf_dual_reward(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  sigma: float = 0.5,
) -> torch.Tensor:
  """Paper Eq. (23)/(27): violation plus bounded filter-imitation reward."""
  term = _cbf_term(env, action_name)
  active = torch.isfinite(term.h)
  value = dual_cbf_reward(
    term.psi_nominal, term.target_intervention_norm, active, sigma=sigma
  )
  env.extras["log"]["CBF/violation_mean"] = torch.relu(-term.psi_nominal).mean()
  env.extras["log"]["CBF/active_fraction"] = active.float().mean()
  env.extras["log"]["CBF/filtered_margin_min"] = term.psi_filtered.min()
  env.extras["log"]["CBF/intervention_norm_mean"] = term.intervention_norm.mean()
  env.extras["log"]["CBF/target_intervention_norm_mean"] = (
    term.target_intervention_norm.mean()
  )
  env.extras["log"]["CBF/dual_reward_mean"] = value.mean()
  return value


def stair_progress(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  robot = env.scene[asset_name]
  progress = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return progress.clamp_min(0.0)


def dont_wait(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_name: str = "robot",
  command_threshold: float = 0.2,
  minimum_forward_speed: float = 0.1,
) -> torch.Tensor:
  """Hiking-style dense penalty for standing while a forward target is active."""
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise RuntimeError(f"command {command_name!r} is unavailable")
  robot = env.scene[asset_name]
  commanded = command[:, 0] > command_threshold
  speed_deficit = torch.relu(
    minimum_forward_speed - robot.data.root_link_lin_vel_b[:, 0]
  ) / minimum_forward_speed
  return speed_deficit * commanded.float()


def stair_feet_clearance(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_name: str = "robot",
  default_height: float = 0.10,
  height_above_tread: float = 0.05,
) -> torch.Tensor:
  """Paper-described clearance reward with the next tread as reference."""
  term = _cbf_term(env, action_name)
  robot = env.scene[asset_name]
  foot_z = robot.data.site_pos_w[:, term._site_local_ids, 2]
  foot_vel_xy = robot.data.site_lin_vel_w[:, term._site_local_ids, :2]
  speed = torch.linalg.vector_norm(foot_vel_xy, dim=-1)
  target = env.scene.env_origins[:, 2:3].expand_as(foot_z) + default_height
  active = torch.isfinite(term.h) & (term.selected_foot >= 0)
  foot_ids = torch.arange(2, device=env.device).view(1, 2)
  selected = active.unsqueeze(1) & (term.selected_foot.unsqueeze(1) == foot_ids)
  stair_target = (term.selected_edge_top_z + height_above_tread).unsqueeze(1)
  target = torch.where(selected, stair_target, target)
  cost = torch.sum(torch.abs(foot_z - target) * speed, dim=1)
  env.extras["log"]["CBF/clearance_target_active_fraction"] = active.float().mean()
  return cost


def swing_foot_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  period: float = 0.6,
  stance_fraction: float = 0.56,
) -> torch.Tensor:
  """Penalize force on the gait-scheduled swing foot, as stated after Eq. (27)."""
  sensor = env.scene[sensor_name]
  if sensor.data.force is None:
    raise RuntimeError(f"contact sensor {sensor_name!r} does not expose force")
  phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.tensor((0.0, 0.5), device=env.device).view(1, 2)
  scheduled_swing = ((phase + offsets) % 1.0) >= stance_fraction
  force = torch.linalg.vector_norm(sensor.data.force, dim=-1)
  return torch.sum(force * scheduled_swing.float(), dim=1)
