"""CBF rewards and diagnostics exposed to MJLab managers."""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv

from .actions import StairCbfJointPositionAction
from .cbf_math import (
  conditional_deployable_cbf_geometry,
  dual_cbf_reward,
  next_riser_clearance_reference,
  persistent_next_riser_geometry,
  sloped_toe_clearance_constraint,
)
from .edge_detection import select_active_riser


CBF_DEPLOYABLE_GEOMETRY_OBSERVATION_DIM = 5
CBF_DEPLOYABLE_CONDITIONAL_GEOMETRY_OBSERVATION_DIM = 16
CBF_DEPLOYABLE_PERSISTENT_GEOMETRY_OBSERVATION_DIM = 10


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


def cbf_deployable_geometry_observation(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_name: str = "robot",
) -> torch.Tensor:
  """Expose the five current-state coordinates used to activate the CBF.

  The values are available at deployment from foot forward kinematics, foot
  contact, and the same mapped stair edges used by the safety filter.  They do
  not contain a filtered action, intervention label, reward, or future state.
  Recomputing the geometry here avoids the one-transition delay that would be
  introduced by exposing fields populated by ``process_actions``.

  Columns are normalized horizontal clearance, normalized vertical clearance,
  normalized sloped barrier, swing-foot side (-1 left, +1 right), and the
  geometric-active flag.  Inactive clearance columns are zeroed so the
  zero-column actor warm start remains local to states in which the CBF itself
  can act.
  """
  term = _cbf_term(env, action_name)
  robot = env.scene[asset_name]
  found = term._contact_sensor.data.found
  if found is None:
    raise RuntimeError("deployable CBF geometry requires foot contact state")
  contact = found > 0
  if contact.ndim > 2:
    contact = contact.any(dim=tuple(range(2, contact.ndim)))
  if contact.ndim != 2 or contact.shape[1] != 2:
    raise RuntimeError("deployable CBF contact state must have shape [N, 2]")
  in_air = ~contact
  air_time = term._contact_sensor.data.current_air_time
  scores = (
    in_air.float()
    if air_time is None
    else torch.where(in_air, air_time, torch.full_like(air_time, -1.0))
  )
  selected_foot = scores.argmax(dim=1)
  has_swing = in_air.any(dim=1)
  batch = torch.arange(env.num_envs, device=env.device)
  foot_position = robot.data.site_pos_w[:, term._site_local_ids]
  selected_position = foot_position[batch, selected_foot]

  terrain = env.scene.terrain
  if terrain is None:
    raise RuntimeError("deployable CBF geometry requires stair metadata")
  edge_x = term._edge_x[terrain.terrain_levels, terrain.terrain_types]
  edge_top_z = term._edge_top_z[terrain.terrain_levels, terrain.terrain_types]
  _, horizontal, selected_top_z, edge_active = select_active_riser(
    selected_position[:, 0],
    selected_position[:, 2],
    edge_x,
    edge_top_z,
    toe_margin=term.cfg.toe_margin,
    top_clearance=term.cfg.top_clearance,
    activation_distance=term.cfg.activation_distance,
    recovery_distance=term.cfg.recovery_distance,
  )
  vertical = selected_position[:, 2] - selected_top_z - float(
    term.cfg.top_clearance
  )
  if term.cfg.clearance_barrier_slope > 0.0:
    # Only the barrier value is needed here; dummy Jacobians keep the
    # observation definition exactly tied to the filter's shared CBF helper.
    dummy = torch.zeros(env.num_envs, 1, device=env.device)
    barrier, _ = sloped_toe_clearance_constraint(
      horizontal,
      selected_position[:, 2],
      selected_top_z,
      dummy,
      dummy,
      top_clearance=term.cfg.top_clearance,
      slope=term.cfg.clearance_barrier_slope,
    )
  else:
    barrier = horizontal
  active = has_swing & edge_active
  distance_scale = max(
    float(term.cfg.activation_distance),
    float(term.cfg.recovery_distance),
    1.0e-6,
  )
  zeros = torch.zeros_like(horizontal)
  horizontal = torch.where(active, horizontal / distance_scale, zeros)
  vertical = torch.where(active, vertical / distance_scale, zeros)
  barrier = torch.where(active, barrier / distance_scale, zeros)
  swing_side = torch.where(
    active,
    selected_foot.to(horizontal.dtype).mul(2.0).sub(1.0),
    zeros,
  )
  observation = torch.stack(
    (
      horizontal.clamp(-1.5, 1.5),
      vertical.clamp(-1.5, 1.5),
      barrier.clamp(-2.0, 2.0),
      swing_side,
      active.to(horizontal.dtype),
    ),
    dim=1,
  )
  if observation.shape != (
    env.num_envs,
    CBF_DEPLOYABLE_GEOMETRY_OBSERVATION_DIM,
  ):
    raise RuntimeError("deployable CBF geometry has an unexpected shape")
  return observation


def cbf_deployable_conditional_geometry_observation(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_name: str = "robot",
) -> torch.Tensor:
  """Expose side/phase-conditioned deployable CBF geometry for v93."""
  geometry = cbf_deployable_geometry_observation(
    env,
    action_name=action_name,
    asset_name=asset_name,
  )
  observation = conditional_deployable_cbf_geometry(geometry)
  if observation.shape != (
    env.num_envs,
    CBF_DEPLOYABLE_CONDITIONAL_GEOMETRY_OBSERVATION_DIM,
  ):
    raise RuntimeError("conditional deployable CBF geometry has an unexpected shape")
  return observation


def cbf_deployable_persistent_geometry_observation(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_name: str = "robot",
) -> torch.Tensor:
  """Expose both feet's next-riser geometry early enough to plan toe lift."""
  term = _cbf_term(env, action_name)
  robot = env.scene[asset_name]
  found = term._contact_sensor.data.found
  if found is None:
    raise RuntimeError("persistent CBF geometry requires foot contact state")
  contact = found > 0
  if contact.ndim > 2:
    contact = contact.any(dim=tuple(range(2, contact.ndim)))
  if contact.shape != (env.num_envs, 2):
    raise RuntimeError("persistent CBF contact state must have shape [N, 2]")
  terrain = env.scene.terrain
  if terrain is None:
    raise RuntimeError("persistent CBF geometry requires stair metadata")
  edge_x = term._edge_x[terrain.terrain_levels, terrain.terrain_types]
  edge_top_z = term._edge_top_z[terrain.terrain_levels, terrain.terrain_types]
  foot_position = robot.data.site_pos_w[:, term._site_local_ids]
  lookahead = max(
    2.5 * float(term.cfg.activation_distance),
    float(term.cfg.activation_distance) + float(term.cfg.recovery_distance),
  )
  vertical_scale = max(float(term.cfg.activation_distance), 0.30)
  observation = persistent_next_riser_geometry(
    robot.data.root_link_pos_w[:, 0],
    foot_position[..., (0, 2)],
    contact,
    edge_x,
    edge_top_z,
    toe_margin=float(term.cfg.toe_margin),
    top_clearance=float(term.cfg.top_clearance),
    barrier_slope=float(term.cfg.clearance_barrier_slope),
    lookahead_distance=lookahead,
    horizontal_scale=lookahead,
    vertical_scale=vertical_scale,
  )
  if observation.shape != (
    env.num_envs,
    CBF_DEPLOYABLE_PERSISTENT_GEOMETRY_OBSERVATION_DIM,
  ):
    raise RuntimeError("persistent deployable CBF geometry has an unexpected shape")
  return observation


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
  margin_weight: float = 1.0,
  intervention_weight: float = 1.0,
  correction_space: str = "target",
) -> torch.Tensor:
  """Paper Eq. (23)/(27): violation plus filter-imitation reward.

  ``target`` retains the historical joint-target distance. ``raw_action``
  measures the correction in the policy's native action coordinates, matching
  the public CBF-RL navigation demo's action-to-filtered-action construction.
  """
  term = _cbf_term(env, action_name)
  active = torch.isfinite(term.h)
  if correction_space == "target":
    correction_norm = term.counterfactual_target_intervention_norm
  elif correction_space == "raw_action":
    correction_norm = torch.linalg.vector_norm(
      term.safe_raw_action - term.nominal_raw_action, dim=-1
    )
  elif correction_space == "foot_task":
    correction_norm = term.counterfactual_task_intervention_norm
  else:
    raise ValueError(
      "CBF dual correction_space must be 'target', 'raw_action', or "
      "'foot_task', got "
      f"{correction_space!r}"
    )
  value = dual_cbf_reward(
    term.psi_nominal,
    correction_norm,
    active,
    sigma=sigma,
    margin_weight=margin_weight,
    intervention_weight=intervention_weight,
  )
  margin_component = float(margin_weight) * torch.minimum(
    term.psi_nominal, torch.zeros_like(term.psi_nominal)
  )
  imitation_component = float(intervention_weight) * (
    torch.exp(-correction_norm.square() / sigma**2) - 1.0
  )
  active_float = active.to(term.psi_nominal.dtype)
  active_margin_component = margin_component * active_float
  active_imitation_component = imitation_component * active_float
  active_count = active_float.sum().clamp_min(1.0)
  env.extras["log"]["CBF/reward_margin_component_mean"] = (
    active_margin_component.sum() / active_count
  )
  env.extras["log"]["CBF/reward_imitation_component_mean"] = (
    active_imitation_component.sum() / active_count
  )
  env.extras["log"]["CBF/reward_correction_norm_mean"] = (
    (correction_norm * active_float).sum() / active_count
  )
  # Keep exact per-environment values outside the scalar logger so short
  # experiments can audit the numerical decomposition of Eq. (22)-(23).
  env.extras["cbf_reward_margin_component"] = (
    active_margin_component.detach().clone()
  )
  env.extras["cbf_reward_proximity_component"] = (
    active_imitation_component.detach().clone()
  )
  env.extras["cbf_reward_dual_component"] = value.detach().clone()
  env.extras["cbf_reward_active"] = active_float.detach().clone()
  env.extras["cbf_reward_correction_norm"] = (
    (correction_norm * active_float).detach().clone()
  )
  _record_online_cbf_telemetry(env, term, dual_reward=value)
  return value


def _record_online_cbf_telemetry(
  env: ManagerBasedRlEnv,
  term,
  *,
  dual_reward: torch.Tensor | None = None,
) -> None:
  """Emit per-transition CBF data independently of reward scalarization."""
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
  if dual_reward is not None:
    env.extras["log"]["CBF/dual_reward_mean"] = dual_reward.mean()
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
  env.extras["cbf_filter_enabled"] = term.runtime_filter_mask.detach().clone()
  env.extras["online_stair_index"] = stair_index(env).detach().clone()


def online_safety_telemetry(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  termination_name: str = "fell_over",
) -> torch.Tensor:
  """Record CBF/fall costs while contributing exactly zero task reward.

  MJLab skips zero-weight reward terms, so this term has unit configuration
  weight but returns an identically zero tensor.  It remains active when the
  fixed ``cbf_dual`` and ``fall_termination`` rewards are disabled by the
  task-first constrained algorithm.
  """
  term = _cbf_term(env, action_name)
  _record_online_cbf_telemetry(env, term)
  fell = env.termination_manager.get_term(termination_name).float()
  env.extras["log"]["Online/fall_termination_fraction"] = fell.mean()
  env.extras["online_fell"] = fell.detach().bool().clone()
  return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def _foot_contact_and_slip(
  env: ManagerBasedRlEnv,
  *,
  action_name: str = "joint_pos",
  asset_name: str = "robot",
  sensor_name: str = "feet_ground_contact",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return two deployable contact flags and tangential foot-speed magnitudes."""
  action = _cbf_term(env, action_name)
  robot = env.scene[asset_name]
  found = env.scene[sensor_name].data.found
  if found is None:
    raise RuntimeError("v19 deployable observation requires foot-contact state")
  if found.ndim == 3 and found.shape[-1] == 1:
    found = found.squeeze(-1)
  if found.ndim != 2 or found.shape[1] != 2:
    raise RuntimeError("v19 foot-contact state must have shape [num_envs, 2]")
  contact = found > 0
  tangential_velocity = robot.data.site_lin_vel_w[
    :, action._site_local_ids, :2
  ]
  # Swing-foot velocity is not slip.  Zero it here so every consumer—the
  # actor feature, recovery reward, failure classifier, and replay bank—uses
  # the support/contact-foot definition declared by v19.
  slip_speed = (
    torch.linalg.vector_norm(tangential_velocity, dim=-1) * contact.float()
  )
  return contact, slip_speed


def _phase_contact_mismatch(
  env: ManagerBasedRlEnv,
  contact: torch.Tensor,
  *,
  period: float = 0.6,
  stance_fraction: float = 0.56,
  phase_offset: float = 0.0,
) -> torch.Tensor:
  phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.tensor((0.0, 0.5), device=env.device).view(1, 2)
  expected_stance = ((phase + offsets + phase_offset) % 1.0) < stance_fraction
  return (expected_stance != contact).float().mean(dim=1)


def v19_deployable_failure_observation(
  env: ManagerBasedRlEnv,
  mode: str,
  command_name: str = "twist",
  action_name: str = "joint_pos",
  asset_name: str = "robot",
  sensor_name: str = "feet_ground_contact",
  centerline_rate_scale: float = 0.5,
  heading_rate_scale: float = 2.0,
  slip_speed_scale: float = 1.0,
  phase_offset: float = 0.0,
) -> torch.Tensor:
  """Five real-robot-obtainable failure coordinates appended to the actor."""
  if mode == "lateral":
    command = env.command_manager.get_term(command_name)
    centerline = getattr(
      command, "centerline_error", torch.zeros(env.num_envs, device=env.device)
    )
    heading = getattr(
      command, "heading_error", torch.zeros(env.num_envs, device=env.device)
    )
    centerline_rate = getattr(
      command,
      "centerline_error_rate",
      torch.zeros(env.num_envs, device=env.device),
    )
    heading_rate = getattr(
      command,
      "heading_error_rate",
      torch.zeros(env.num_envs, device=env.device),
    )
    stair_half_width = float(getattr(command.cfg, "stair_half_width", 1.20))
    return torch.stack(
      (
        (centerline / stair_half_width).clamp(-1.5, 1.5),
        torch.sin(heading),
        torch.cos(heading),
        (centerline_rate / centerline_rate_scale).clamp(-2.0, 2.0),
        (heading_rate / heading_rate_scale).clamp(-2.0, 2.0),
      ),
      dim=1,
    )
  if mode != "contact_stability":
    raise ValueError(f"unsupported v19 deployable observation mode: {mode!r}")
  contact, slip_speed = _foot_contact_and_slip(
    env,
    action_name=action_name,
    asset_name=asset_name,
    sensor_name=sensor_name,
  )
  mismatch = _phase_contact_mismatch(
    env, contact, phase_offset=phase_offset
  )
  return torch.cat(
    (
      contact.float(),
      (slip_speed / slip_speed_scale).clamp(0.0, 2.0),
      mismatch.unsqueeze(1),
    ),
    dim=1,
  )


class V19LateralRecoveryReward:
  """Potential progress in centerline/heading recovery plus one edge cost."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._env = env
    self._previous_potential = torch.zeros(env.num_envs, device=env.device)
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self._previous_potential[env_ids] = 0.0
    self._initialized[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    action_name: str = "joint_pos",
    asset_name: str = "robot",
    sensor_name: str = "feet_ground_contact",
    gamma: float = 0.99,
    centerline_coefficient: float = 0.60,
    heading_coefficient: float = 0.40,
    recovery_scale: float = 0.50,
    edge_penalty_scale: float = 0.20,
  ) -> torch.Tensor:
    components = specialist_failure_signal_components(
      env,
      command_name=command_name,
      action_name=action_name,
      asset_name=asset_name,
      sensor_name=sensor_name,
    )
    potential = -(
      centerline_coefficient * components["centerline"]
      + heading_coefficient * components["heading"]
    )
    progress = gamma * potential - self._previous_potential
    progress = torch.where(
      self._initialized, progress, torch.zeros_like(progress)
    )
    self._previous_potential[:] = potential
    self._initialized[:] = True
    reward = (
      recovery_scale * progress / env.step_dt
      - edge_penalty_scale * components["edge"]
    )
    env.extras["log"]["V19/lateral_recovery_progress"] = progress.mean()
    env.extras["log"]["V19/lateral_edge_fraction"] = components["edge"].mean()
    return reward


class V19ContactRecoveryReward:
  """Contact-window potential progress for slip, mismatch, and angular recovery."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._env = env
    self._previous_potential = torch.zeros(env.num_envs, device=env.device)
    self._previous_contact = torch.zeros(
      env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    self._contact_window_remaining = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
    self._initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self._previous_potential[env_ids] = 0.0
    self._previous_contact[env_ids] = False
    self._contact_window_remaining[env_ids] = 0
    self._initialized[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    asset_name: str = "robot",
    sensor_name: str = "feet_ground_contact",
    period: float = 0.6,
    stance_fraction: float = 0.56,
    phase_offset: float = 0.0,
    gamma: float = 0.99,
    slip_coefficient: float = 0.45,
    mismatch_coefficient: float = 0.35,
    angular_velocity_coefficient: float = 0.20,
    recovery_scale: float = 0.40,
    pre_touchdown_steps: int = 30,
    post_touchdown_steps: int = 45,
  ) -> torch.Tensor:
    contact, slip_speed = _foot_contact_and_slip(
      env,
      action_name=action_name,
      asset_name=asset_name,
      sensor_name=sensor_name,
    )
    mismatch = _phase_contact_mismatch(
      env,
      contact,
      period=period,
      stance_fraction=stance_fraction,
      phase_offset=phase_offset,
    )
    touchdown = contact & ~self._previous_contact & self._initialized.unsqueeze(1)
    touched = touchdown.any(dim=1)
    self._contact_window_remaining = torch.where(
      touched,
      torch.full_like(self._contact_window_remaining, post_touchdown_steps),
      (self._contact_window_remaining - 1).clamp_min(0),
    )
    phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.tensor((0.0, 0.5), device=env.device).view(1, 2)
    shifted_phase = (phase + offsets + phase_offset) % 1.0
    steps_until_scheduled_touchdown = (
      (1.0 - shifted_phase) % 1.0
    ) * period / env.step_dt
    pre_window = (steps_until_scheduled_touchdown <= pre_touchdown_steps).any(dim=1)
    contact_window = pre_window | (self._contact_window_remaining > 0) | touched
    robot = env.scene[asset_name]
    angular_velocity = (
      torch.linalg.vector_norm(robot.data.root_link_ang_vel_b, dim=1) / 4.0
    ).square().clamp(0.0, 1.0)
    contact_slip = torch.sum(slip_speed * contact.float(), dim=1).clamp(0.0, 2.0)
    potential = -(
      slip_coefficient * contact_slip
      + mismatch_coefficient * mismatch
      + angular_velocity_coefficient * angular_velocity
    )
    progress = gamma * potential - self._previous_potential
    progress = torch.where(
      self._initialized & contact_window,
      progress,
      torch.zeros_like(progress),
    )
    self._previous_potential[:] = potential
    self._previous_contact[:] = contact
    self._initialized[:] = True
    env.extras["v19_touchdown_left"] = touchdown[:, 0].detach().clone()
    env.extras["v19_touchdown_right"] = touchdown[:, 1].detach().clone()
    env.extras["log"]["V19/contact_window_fraction"] = contact_window.float().mean()
    env.extras["log"]["V19/contact_recovery_progress"] = progress.mean()
    return recovery_scale * progress / env.step_dt


def specialist_failure_signal_components(
  env: ManagerBasedRlEnv,
  *,
  command_name: str = "twist",
  action_name: str = "joint_pos",
  asset_name: str = "robot",
  sensor_name: str = "feet_ground_contact",
) -> dict[str, torch.Tensor]:
  """Return normalized sensor-derived components for all three specialists."""
  command = env.command_manager.get_term(command_name)
  action = _cbf_term(env, action_name)
  robot = env.scene[asset_name]
  centerline_error = getattr(
    command, "centerline_error", torch.zeros(env.num_envs, device=env.device)
  )
  heading_error = getattr(
    command, "heading_error", torch.zeros(env.num_envs, device=env.device)
  )
  centerline_rate = getattr(
    command,
    "centerline_error_rate",
    torch.zeros(env.num_envs, device=env.device),
  )
  heading_rate = getattr(
    command,
    "heading_error_rate",
    torch.zeros(env.num_envs, device=env.device),
  )
  stair_half_width = float(getattr(command.cfg, "stair_half_width", 1.20))
  centerline_threshold = (2.0 / 3.0) * stair_half_width
  centerline = (centerline_error.abs() / centerline_threshold).clamp(0.0, 1.0)
  heading = (heading_error.abs() / (math.pi / 2.0)).clamp(0.0, 1.0)
  terrain = env.scene.terrain
  if terrain is None:
    raise RuntimeError("specialist failure signal requires stair terrain")
  patches = terrain.flat_patches["stair_targets"][
    terrain.terrain_levels, terrain.terrain_types
  ]
  center_y = patches[:, 0, 1]
  foot_y = robot.data.site_pos_w[:, action._site_local_ids, 1]
  maximum_foot_error = torch.max(
    torch.abs(foot_y - center_y.unsqueeze(1)), dim=1
  ).values
  edge = (
    (centerline_error.abs() >= centerline_threshold)
    | (maximum_foot_error >= centerline_threshold)
  ).float()

  intervention = (action.target_intervention_norm / 0.5).clamp(0.0, 1.0)
  nominal_margin = (torch.relu(-action.psi_nominal) / 1.0).clamp(0.0, 1.0)
  gravity = robot.data.projected_gravity_b
  roll_angle = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
  pitch_angle = torch.atan2(
    -gravity[:, 0],
    torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square()),
  ).abs()
  roll = (roll_angle / 0.60).clamp(0.0, 1.0)
  pitch = (pitch_angle / 0.60).clamp(0.0, 1.0)
  angular_velocity = (
    torch.linalg.vector_norm(robot.data.root_link_ang_vel_b, dim=1) / 4.0
  ).square().clamp(0.0, 1.0)

  in_contact, slip_speed = _foot_contact_and_slip(
    env,
    action_name=action_name,
    asset_name=asset_name,
    sensor_name=sensor_name,
  )
  slip = (
    torch.sum(
      slip_speed.square() * in_contact.float(),
      dim=1,
    )
    / 1.0
  ).clamp(0.0, 1.0)
  contact_mismatch = _phase_contact_mismatch(
    env,
    in_contact,
    phase_offset=float(
      getattr(action.cfg, "deployment_contact_phase_offset", 0.0)
    ),
  )
  return {
    "centerline": centerline,
    "heading": heading,
    "edge": edge,
    "intervention": intervention,
    "nominal_margin": nominal_margin,
    "roll": roll,
    "pitch": pitch,
    "angular_velocity": angular_velocity,
    "slip": slip,
    "contact_mismatch": contact_mismatch,
    "centerline_signed": (centerline_error / stair_half_width).clamp(-1.5, 1.5),
    "heading_signed": (heading_error / (math.pi / 2.0)).clamp(-2.0, 2.0),
    "centerline_rate": (centerline_rate / 0.5).clamp(-2.0, 2.0),
    "heading_rate": (heading_rate / 2.0).clamp(-2.0, 2.0),
    "left_contact": in_contact[:, 0].float(),
    "right_contact": in_contact[:, 1].float(),
    "left_slip": slip_speed[:, 0].clamp(0.0, 2.0),
    "right_slip": slip_speed[:, 1].clamp(0.0, 2.0),
  }


def specialist_failure_signal(
  env: ManagerBasedRlEnv,
  mode: str,
  weights: dict[str, float],
  command_name: str = "twist",
  action_name: str = "joint_pos",
  asset_name: str = "robot",
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """One mode-conditioned cost inserted into the single scalar reward."""
  active = {
    "lateral": {"centerline", "heading", "edge"},
    "cbf": {"intervention", "nominal_margin"},
    "balance": {
      "roll",
      "pitch",
      "angular_velocity",
      "slip",
      "contact_mismatch",
    },
  }
  if mode not in active:
    raise ValueError(f"unsupported specialist failure-signal mode: {mode!r}")
  if set(weights) != set().union(*active.values()):
    raise ValueError("specialist failure-signal component set is incomplete")
  if any(float(weight) < 0.0 for weight in weights.values()):
    raise ValueError("specialist failure-signal weights must be non-negative")
  if not math.isclose(sum(float(weights[name]) for name in active[mode]), 1.0):
    raise ValueError("active specialist failure-signal weights must sum to one")
  if any(float(weights[name]) != 0.0 for name in set(weights) - active[mode]):
    raise ValueError("specialist failure signal contains cross-mode weight")
  components = specialist_failure_signal_components(
    env,
    command_name=command_name,
    action_name=action_name,
    asset_name=asset_name,
    sensor_name=sensor_name,
  )
  value = sum(float(weights[name]) * components[name] for name in active[mode])
  env.extras["log"][f"Specialist/{mode}_failure_signal_mean"] = value.mean()
  env.extras["specialist_failure_signal"] = value.detach().clone()
  for name in active[mode]:
    env.extras[f"specialist_{name}_component"] = components[name].detach().clone()
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
  safety_history = torch.cat(
    [
      term.barrier_derivative.clamp(-20.0, 20.0).unsqueeze(1),
      term.predicted_h.clamp(-1.0, 1.0).unsqueeze(1),
      term.h_history.clamp(-1.0, 1.0),
      term.correction_history.clamp(0.0, 2.0),
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
      safety_history,
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
  reference_mode: str = "cbf_active",
  lookahead_distance: float = 0.60,
) -> torch.Tensor:
  """Paper-described clearance reward with the next tread as reference."""
  term = _cbf_term(env, action_name)
  robot = env.scene[asset_name]
  foot_z = robot.data.site_pos_w[:, term._site_local_ids, 2]
  foot_vel_xy = robot.data.site_lin_vel_w[:, term._site_local_ids, :2]
  speed = torch.linalg.vector_norm(foot_vel_xy, dim=-1)
  if reference_mode == "cbf_active":
    target = env.scene.env_origins[:, 2:3].expand_as(foot_z) + default_height
    active = torch.isfinite(term.h) & (term.selected_foot >= 0)
    foot_ids = torch.arange(2, device=env.device).view(1, 2)
    selected = active.unsqueeze(1) & (term.selected_foot.unsqueeze(1) == foot_ids)
    stair_target = (term.selected_edge_top_z + height_above_tread).unsqueeze(1)
    target = torch.where(selected, stair_target, target)
  elif reference_mode == "next_riser":
    terrain = env.scene.terrain
    if terrain is None:
      raise RuntimeError("next-riser clearance requires stair terrain")
    edge_x = term._edge_x[terrain.terrain_levels, terrain.terrain_types]
    edge_top_z = term._edge_top_z[
      terrain.terrain_levels, terrain.terrain_types
    ]
    reference, active, index = next_riser_clearance_reference(
      robot.data.root_link_pos_w[:, 0],
      env.scene.env_origins[:, 2],
      edge_x,
      edge_top_z,
      default_height=default_height,
      height_above_tread=height_above_tread,
      lookahead_distance=lookahead_distance,
    )
    target = reference.unsqueeze(1).expand_as(foot_z)
    env.extras["log"]["CBF/clearance_reference_riser_mean"] = (
      index.float().mean()
    )
    env.extras["log"]["CBF/clearance_reference_height_mean"] = (
      reference.mean()
    )
  else:
    raise ValueError(
      "stair clearance reference_mode must be 'cbf_active' or 'next_riser', "
      f"got {reference_mode!r}"
    )
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
