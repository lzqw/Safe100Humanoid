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
