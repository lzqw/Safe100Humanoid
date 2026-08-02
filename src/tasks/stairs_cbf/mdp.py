"""CBF rewards and diagnostics exposed to MJLab managers."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from .actions import StairCbfJointPositionAction
from .cbf_math import dual_cbf_reward


def _terrain_risers(
  env: ManagerBasedRlEnv, patch_name: str = "stair_risers"
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None or patch_name not in terrain.flat_patches:
    raise RuntimeError(f"terrain has no riser metadata {patch_name!r}")
  return terrain.flat_patches[patch_name][
    terrain.terrain_levels, terrain.terrain_types
  ]


def stair_index(
  env: ManagerBasedRlEnv,
  asset_name: str = "robot",
  patch_name: str = "stair_risers",
) -> torch.Tensor:
  """Number of riser planes crossed by the pelvis/root."""
  robot = env.scene[asset_name]
  risers = _terrain_risers(env, patch_name)
  return torch.sum(
    robot.data.root_link_pos_w[:, 0:1] >= risers[..., 0], dim=1
  )


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
    term.psi_nominal,
    term.counterfactual_target_intervention_norm,
    active,
    sigma=sigma,
  )
  env.extras["log"]["CBF/violation_mean"] = torch.relu(-term.psi_nominal).mean()
  env.extras["log"]["CBF/geometric_active_fraction"] = (
    term.geometric_active.float().mean()
  )
  env.extras["log"]["CBF/intervention_fraction"] = term.intervened.float().mean()
  env.extras["log"]["CBF/would_intervene_fraction"] = (
    term.would_intervene.float().mean()
  )
  env.extras["log"]["CBF/filtered_margin_min"] = term.psi_filtered.min()
  env.extras["log"]["CBF/intervention_norm_mean"] = term.intervention_norm.mean()
  env.extras["log"]["CBF/target_intervention_norm_mean"] = (
    term.target_intervention_norm.mean()
  )
  env.extras["log"]["CBF/counterfactual_correction_mean"] = (
    term.counterfactual_target_intervention_norm.mean()
  )
  env.extras["log"]["CBF/dual_reward_mean"] = value.mean()
  # Per-environment transition data is consumed by OnlineSafePPO.  It lives
  # outside ``log`` so the scalar logger never reduces or serializes it.
  env.extras["cbf_intervened"] = term.intervened.detach().clone()
  env.extras["cbf_intervention_magnitude"] = (
    term.target_intervention_norm.detach().clone()
  )
  env.extras["cbf_would_intervene"] = term.would_intervene.detach().clone()
  env.extras["cbf_counterfactual_magnitude"] = (
    term.counterfactual_target_intervention_norm.detach().clone()
  )
  env.extras["cbf_nominal_target"] = term.nominal_target.detach().clone()
  env.extras["cbf_safe_target"] = term.safe_target.detach().clone()
  env.extras["cbf_safe_raw_action"] = term.safe_raw_action.detach().clone()
  env.extras["cbf_nominal_raw_action"] = term.nominal_raw_action.detach().clone()
  env.extras["cbf_executed_raw_action"] = (
    term.executed_raw_action.detach().clone()
  )
  env.extras["cbf_filter_enabled"] = torch.full(
    (env.num_envs,), term.cfg.enabled, dtype=torch.bool, device=env.device
  )
  env.extras["online_stair_index"] = stair_index(env).detach().clone()
  return value


class IncrementalStairProgress:
  """Stationary clipped delta-x reward shared by short and long stairs."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._env = env
    self._previous_x = env.scene["robot"].data.root_link_pos_w[:, 0].clone()
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self._previous_x[env_ids] = self._env.scene["robot"].data.root_link_pos_w[
      env_ids, 0
    ]
    self._initialized[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_name: str = "robot",
    maximum_forward_velocity: float = 0.8,
  ) -> torch.Tensor:
    current = env.scene[asset_name].data.root_link_pos_w[:, 0]
    delta = current - self._previous_x
    delta = torch.where(self._initialized, delta, torch.zeros_like(delta))
    self._previous_x[:] = current
    self._initialized[:] = True
    return torch.clamp(
      delta / (maximum_forward_velocity * env.step_dt), -1.0, 1.0
    )


class RiserCrossingReward:
  """Emit one event for each newly crossed riser, independent of horizon."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._env = env
    self._previous_index = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    current = stair_index(self._env)
    self._previous_index[env_ids] = current[env_ids]

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    current = stair_index(env)
    crossed = (current - self._previous_index).clamp_min(0).float()
    self._previous_index[:] = current
    # RewardManager scales rates by dt; divide here to keep a fixed event bonus.
    return crossed / env.step_dt


class TopCompletionReward:
  """A fixed terminal event bonus derived from actual stair metadata."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._env = env
    self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self._paid[env_ids] = False

  def __call__(
    self, env: ManagerBasedRlEnv, position_tolerance: float = 0.10
  ) -> torch.Tensor:
    reached = reached_stair_top(env, position_tolerance=position_tolerance)
    event = reached & ~self._paid
    self._paid |= event
    return event.float() / env.step_dt


def reached_stair_top(
  env: ManagerBasedRlEnv,
  asset_name: str = "robot",
  target_patch_name: str = "stair_targets",
  position_tolerance: float = 0.10,
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None or target_patch_name not in terrain.flat_patches:
    raise RuntimeError(f"terrain has no target metadata {target_patch_name!r}")
  patches = terrain.flat_patches[target_patch_name][
    terrain.terrain_levels, terrain.terrain_types
  ]
  robot = env.scene[asset_name]
  top = patches[:, -1]
  position = robot.data.root_link_pos_w
  crossed_all = stair_index(env) >= (patches.shape[1] - 1)
  on_top = (
    (position[:, 0] >= top[:, 0] - position_tolerance)
    & (position[:, 2] >= top[:, 2] + 0.45)
  )
  return crossed_all & on_top


def fall_termination(
  env: ManagerBasedRlEnv,
  termination_name: str = "fell_over",
) -> torch.Tensor:
  """Return only the fall termination indicator.

  Online stairs also terminate successfully at the top.  The generic
  ``is_terminated`` reward therefore cannot distinguish a fall from a success
  and was previously disabled, accidentally removing the explicit failure
  penalty from the online PPO signal.  Keeping this term action-independent
  and tied to the named termination preserves the successful-top bonus while
  restoring the same fall-event scale used by the base locomotion task.
  """
  fell = env.termination_manager.get_term(termination_name).float()
  env.extras["log"]["Online/fall_termination_fraction"] = fell.mean()
  env.extras["online_fell"] = fell.detach().bool().clone()
  return fell


def fixed_delay_queue_state(
  command_term,
  *,
  num_envs: int,
  command_dim: int,
  queue_length: int,
  device: str | torch.device,
) -> torch.Tensor:
  """Return a fixed-size newest-first command-delay queue for the critic."""
  if queue_length < 1:
    raise ValueError("queue_length must be positive")
  output = torch.zeros(
    num_envs, queue_length, command_dim, device=device
  )
  queue = getattr(command_term, "_delay_queue", None)
  if queue is None:
    return output
  if queue.ndim != 3 or queue.shape[0] != num_envs or queue.shape[2] != command_dim:
    raise ValueError(
      "command delay queue must have [num_envs, queue_steps, command_dim] shape"
    )
  copied = min(queue_length, queue.shape[1])
  output[:, :copied].copy_(queue[:, :copied])
  return output


def online_privileged_state(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  command_name: str = "twist",
  asset_name: str = "robot",
  delay_queue_length: int = 9,
) -> torch.Tensor:
  """Action-independent full state used only by the online value function.

  CBF quantities correspond to the *previously executed* action because
  observations are computed after the transition.  No quantity derived from
  the action currently being sampled is included.
  """
  term = _cbf_term(env, action_name)
  robot = env.scene[asset_name]
  command_term = env.command_manager.get_term(command_name)
  delivered = env.command_manager.get_command(command_name)
  raw = getattr(command_term, "raw_command", delivered)
  derivative = getattr(command_term, "command_derivative", torch.zeros_like(delivered))
  delay_steps = getattr(
    command_term,
    "delay_steps",
    torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
  ).float().unsqueeze(1)
  delay_queue = fixed_delay_queue_state(
    command_term,
    num_envs=env.num_envs,
    command_dim=delivered.shape[-1],
    queue_length=delay_queue_length,
    device=env.device,
  )

  risers = _terrain_risers(env)
  index = stair_index(env)
  n = risers.shape[1]
  previous_z = torch.cat(
    [env.scene.env_origins[:, 2:3], risers[:, :-1, 2]], dim=1
  )
  rises = risers[..., 2] - previous_z
  next_x = torch.cat([risers[:, 1:, 0], risers[:, -1:, 0]], dim=1)
  treads = next_x - risers[..., 0]
  lookahead = []
  batch = torch.arange(env.num_envs, device=env.device)
  for offset in range(3):
    sample_index = (index + offset).clamp_max(n - 1)
    lookahead.extend(
      [rises[batch, sample_index].unsqueeze(1), treads[batch, sample_index].unsqueeze(1)]
    )
  geometry = torch.cat(
    [
      (index.float() / n).unsqueeze(1),
      ((n - index).float() / n).unsqueeze(1),
      *lookahead,
    ],
    dim=1,
  )

  h = torch.where(torch.isfinite(term.h), term.h, torch.ones_like(term.h))
  cbf = torch.stack(
    [
      h.clamp(-1.0, 1.0),
      term.psi_nominal.clamp(-10.0, 10.0),
      term.psi_filtered.clamp(-10.0, 10.0),
      term.geometric_active.float(),
      term.intervened.float(),
      term.target_intervention_norm.clamp(0.0, 2.0),
      term.intervention_count / max(1, n),
    ],
    dim=1,
  )
  root_position = robot.data.root_link_pos_w - env.scene.env_origins
  feet_position = robot.data.site_pos_w[:, term._site_local_ids] - (
    env.scene.env_origins[:, None, :]
  )
  true_robot = torch.cat(
    [
      root_position,
      robot.data.root_link_quat_w,
      robot.data.root_link_lin_vel_b,
      robot.data.root_link_ang_vel_b,
      robot.data.joint_pos,
      robot.data.joint_vel,
      feet_position.flatten(1),
      robot.data.site_lin_vel_w[:, term._site_local_ids].flatten(1),
    ],
    dim=1,
  )
  episode = torch.stack(
    [
      env.episode_length_buf.float() / max(1, env.max_episode_length),
      index.float() / n,
      term.intervention_count / max(1, n),
    ],
    dim=1,
  )
  # Append the queue after the historical 111-D block.  This preserves the
  # exact prefix layout of existing 799-D online critics, allowing their first
  # layer and normalization statistics to be expanded with zero/identity
  # initialization rather than discarded.
  return torch.cat(
    [
      true_robot,
      geometry,
      raw,
      delivered,
      derivative,
      delay_steps,
      cbf,
      episode,
      delay_queue.flatten(1),
    ],
    dim=1,
  )


def stair_progress(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  """Legacy cumulative progress retained for the published baseline task only."""
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
