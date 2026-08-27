"""Swing-foot CBF safety filter for a straight staircase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .cbf_math import project_halfspace, sloped_toe_clearance_constraint
from .edge_detection import (
  riser_edges_from_metadata,
  riser_edges_from_tread_patches,
  select_active_riser,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class StairCbfJointPositionActionCfg(JointPositionActionCfg):
  enabled: bool = True
  foot_site_names: tuple[str, str] = ("left_foot", "right_foot")
  contact_sensor_name: str = "feet_ground_contact"
  alpha: float = 10.0
  first_riser_offset: float = 0.60
  step_width: float = 0.35
  step_height: float = 0.13
  num_steps: int = 6
  activation_distance: float = 0.30
  toe_margin: float = 0.08
  top_clearance: float = 0.025
  clearance_barrier_slope: float = 0.0
  recovery_distance: float = 0.15
  patch_name: str = "stair_targets"
  riser_patch_name: str = "stair_risers"
  intervention_epsilon: float = 1.0e-5
  intervention_norm_scale: float = 0.05
  safety_history_length: int = 5
  safety_prediction_steps: int = 5
  deployment_action_gain: float = 1.0
  deployment_action_scale: tuple[float, ...] | dict[str, float] | None = None
  deployment_action_bias: tuple[float, ...] | None = None
  deployment_action_delay_steps: int = 0
  deployment_contact_delay_steps: int = 0
  deployment_contact_phase_offset: float = 0.0

  def build(self, env: "ManagerBasedRlEnv") -> "StairCbfJointPositionAction":
    return StairCbfJointPositionAction(self, env)


class StairCbfJointPositionAction(JointPositionAction):
  cfg: StairCbfJointPositionActionCfg

  def __init__(self, cfg: StairCbfJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self._joint_dof_ids = self._entity.indexing.joint_v_adr[self._target_ids]
    site_ids, _ = self._entity.find_sites(cfg.foot_site_names, preserve_order=True)
    if len(site_ids) != 2:
      raise RuntimeError(f"expected two foot sites, got {site_ids}")
    self._site_local_ids = torch.tensor(site_ids, device=self.device, dtype=torch.long)
    self._site_global_ids = self._entity.indexing.site_ids[self._site_local_ids]
    global_site_ids = [int(v) for v in self._site_global_ids.tolist()]
    self._body_ids = [
      int(self._env.sim.mj_model.site_bodyid[site_id]) for site_id in global_site_ids
    ]
    self._contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]
    terrain = env.scene.terrain
    if terrain is None or cfg.patch_name not in terrain.flat_patches:
      raise RuntimeError(f"terrain has no flat-patch set {cfg.patch_name!r}")
    if cfg.riser_patch_name in terrain.flat_patches:
      self._edge_x, self._edge_top_z = riser_edges_from_metadata(
        terrain.flat_patches[cfg.riser_patch_name], cfg.num_steps
      )
    else:
      # Backward compatibility for checkpoints/configs generated before exact
      # riser metadata was introduced.
      self._edge_x, self._edge_top_z = riser_edges_from_tread_patches(
        terrain.flat_patches[cfg.patch_name], cfg.step_width, cfg.num_steps
      )

    nworld = self.num_envs
    nv = self._env.sim.mj_model.nv
    self._jacp_wp = []
    self._jacr_wp = []
    self._points_wp = []
    self._bodies_wp = []
    self._jacp_torch = []
    self._points_torch = []
    with wp.ScopedDevice(self._env.sim.wp_device):
      for body_id in self._body_ids:
        jacp = wp.zeros((nworld, 3, nv), dtype=float)
        jacr = wp.zeros((nworld, 3, nv), dtype=float)
        points = wp.zeros(nworld, dtype=wp.vec3)
        bodies = wp.zeros(nworld, dtype=wp.int32)
        bodies.fill_(body_id)
        self._jacp_wp.append(jacp)
        self._jacr_wp.append(jacr)
        self._points_wp.append(points)
        self._bodies_wp.append(bodies)
        self._jacp_torch.append(wp.to_torch(jacp))
        self._points_torch.append(wp.to_torch(points).view(nworld, 3))

    shape = (self.num_envs,)
    self.h = torch.full(shape, torch.inf, device=self.device)
    self.psi_nominal = torch.zeros(shape, device=self.device)
    self.psi_filtered = torch.zeros(shape, device=self.device)
    self.filter_active = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.runtime_filter_mask = torch.full(
      shape, bool(cfg.enabled), dtype=torch.bool, device=self.device
    )
    self.geometric_active = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.intervened = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.would_intervene = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.intervention_norm = torch.zeros(shape, device=self.device)
    self.target_intervention_norm = torch.zeros(shape, device=self.device)
    self.counterfactual_intervention_norm = torch.zeros(shape, device=self.device)
    self.counterfactual_target_intervention_norm = torch.zeros(
      shape, device=self.device
    )
    # Paper Eq. (27) measures the filtered displacement in the reduced-order
    # swing-foot coordinates, not in the 12-D joint-action parameterization.
    self.counterfactual_task_intervention_norm = torch.zeros(
      shape, device=self.device
    )
    self.intervention_count = torch.zeros(shape, device=self.device)
    self.selected_edge_top_z = torch.zeros(shape, device=self.device)
    self.selected_foot = torch.full(shape, -1, dtype=torch.long, device=self.device)
    self.nominal_target = torch.zeros(
      self.num_envs, self.action_dim, device=self.device
    )
    self.safe_target = torch.zeros_like(self.nominal_target)
    self.nominal_raw_action = torch.zeros_like(self.nominal_target)
    self.safe_raw_action = torch.zeros_like(self.nominal_target)
    self.executed_raw_action = torch.zeros_like(self.nominal_target)
    # Optional shadow projection used by v35 to filter the deterministic policy
    # mean at the exact same pre-step state as the sampled rollout action.  It
    # never changes the action sent to the simulator.
    self.counterfactual_policy_action = torch.zeros_like(self.nominal_target)
    self.counterfactual_safe_policy_action = torch.zeros_like(self.nominal_target)
    self.counterfactual_policy_intervened = torch.zeros(
      shape, dtype=torch.bool, device=self.device
    )
    self.counterfactual_policy_correction_norm = torch.zeros(
      shape, device=self.device
    )
    self.counterfactual_policy_nominal_margin = torch.zeros(
      shape, device=self.device
    )
    self.counterfactual_policy_projection_valid = torch.zeros(
      shape, dtype=torch.bool, device=self.device
    )
    self._counterfactual_policy_action_staged = False
    if cfg.deployment_action_delay_steps < 0:
      raise ValueError("deployment action delay must be non-negative")
    if not 0.0 < cfg.deployment_action_gain <= 2.0:
      raise ValueError("deployment action gain must be in (0, 2]")
    if cfg.deployment_action_scale is None:
      scale = torch.ones(
        self.action_dim, device=self.device, dtype=self.nominal_target.dtype
      )
    elif isinstance(cfg.deployment_action_scale, dict):
      scale = torch.ones(
        self.action_dim, device=self.device, dtype=self.nominal_target.dtype
      )
      indices, _, values = resolve_matching_names_values(
        cfg.deployment_action_scale, self._target_names
      )
      scale[indices] = torch.tensor(
        values, device=self.device, dtype=self.nominal_target.dtype
      )
    else:
      if len(cfg.deployment_action_scale) != self.action_dim:
        raise ValueError(
          "deployment action scale must match the action dimension"
        )
      scale = torch.tensor(
        cfg.deployment_action_scale,
        device=self.device,
        dtype=self.nominal_target.dtype,
      )
    if not bool(((scale >= 0.5) & (scale <= 1.5)).all()):
      raise ValueError("deployment action scale must lie in [0.5, 1.5]")
    self._deployment_action_scale = scale
    bias = (
      (0.0,) * self.action_dim
      if cfg.deployment_action_bias is None
      else cfg.deployment_action_bias
    )
    if len(bias) != self.action_dim:
      raise ValueError(
        "deployment action bias must match action dimension: "
        f"{len(bias)} != {self.action_dim}"
      )
    self._deployment_action_bias = torch.tensor(
      bias, device=self.device, dtype=self.nominal_target.dtype
    )
    self._deployment_action_queue = torch.zeros(
      self.num_envs,
      cfg.deployment_action_delay_steps + 1,
      self.action_dim,
      device=self.device,
      dtype=self.nominal_target.dtype,
    )
    if cfg.deployment_contact_delay_steps < 0:
      raise ValueError("deployment contact delay must be non-negative")
    self._deployment_contact_queue = torch.zeros(
      self.num_envs,
      cfg.deployment_contact_delay_steps + 1,
      2,
      dtype=torch.bool,
      device=self.device,
    )
    self._deployment_contact_queue_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    if cfg.safety_history_length < 1 or cfg.safety_prediction_steps < 1:
      raise ValueError("CBF safety history and prediction horizon must be positive")
    if not 0.0 <= cfg.clearance_barrier_slope <= 2.0:
      raise ValueError("clearance barrier slope must lie in [0, 2]")
    self.barrier_derivative = torch.zeros(shape, device=self.device)
    self.predicted_h = torch.ones(shape, device=self.device)
    self.h_history = torch.ones(
      self.num_envs, cfg.safety_history_length, device=self.device
    )
    self.correction_history = torch.zeros_like(self.h_history)

  def set_runtime_filter_mask(self, enabled: torch.Tensor) -> None:
    """Select which vector environments execute the projected action."""
    if enabled.shape != self.runtime_filter_mask.shape or enabled.dtype != torch.bool:
      raise ValueError("runtime filter mask must be boolean with shape [num_envs]")
    if bool(enabled.any()) and not self.cfg.enabled:
      raise ValueError("cannot enable a per-environment filter in an off config")
    self.runtime_filter_mask.copy_(enabled)

  def stage_counterfactual_policy_action(self, actions: torch.Tensor) -> None:
    """Stage a shadow actor action for projection during the next env step."""
    if actions.shape != self.counterfactual_policy_action.shape:
      raise ValueError(
        "counterfactual policy action shape differs from the environment action "
        f"shape: {tuple(actions.shape)} != "
        f"{tuple(self.counterfactual_policy_action.shape)}"
      )
    if not bool(torch.isfinite(actions).all()):
      raise RuntimeError("counterfactual policy action contains non-finite values")
    if self.cfg.deployment_action_delay_steps:
      raise RuntimeError(
        "same-state counterfactual policy projection forbids action delay"
      )
    self.counterfactual_policy_action.copy_(actions.detach())
    self._counterfactual_policy_action_staged = True

  def _foot_xz_jacobians(self, foot_pos: torch.Tensor) -> torch.Tensor:
    jacobians = []
    with wp.ScopedDevice(self._env.sim.wp_device):
      for foot in range(2):
        self._points_torch[foot][:] = foot_pos[:, foot]
        mjwarp.jac(
          self._env.sim.wp_model,
          self._env.sim.wp_data,
          self._jacp_wp[foot],
          self._jacr_wp[foot],
          self._points_wp[foot],
          self._bodies_wp[foot],
        )
        jacobians.append(
          torch.stack(
            (
              self._jacp_torch[foot][:, 0, self._joint_dof_ids],
              self._jacp_torch[foot][:, 2, self._joint_dof_ids],
            ),
            dim=1,
          )
        )
    return torch.stack(jacobians, dim=1)

  def _foot_jacobians(self, foot_pos: torch.Tensor) -> torch.Tensor:
    """Return the historical x-axis Jacobian used by the default CBF."""
    return self._foot_xz_jacobians(foot_pos)[:, :, 0]

  def process_actions(self, actions: torch.Tensor) -> None:
    previous_h = self.h.clone()
    if self.cfg.deployment_action_delay_steps > 0:
      self._deployment_action_queue[:, 1:] = (
        self._deployment_action_queue[:, :-1].clone()
      )
    self._deployment_action_queue[:, 0] = actions
    delayed_actions = self._deployment_action_queue[
      :, self.cfg.deployment_action_delay_steps
    ]
    deployment_actions = (
      self.cfg.deployment_action_gain
      * self._deployment_action_scale
      * delayed_actions
      + self._deployment_action_bias
    )
    # The actor and PPO buffer retain ``actions``. Gain, bias, and delay are a
    # hidden plant-side transform; the runtime CBF filters the transformed
    # command that would otherwise reach the actuators.
    super().process_actions(deployment_actions)
    nominal_target = self._processed_actions.clone()
    self.nominal_target[:] = nominal_target
    self.nominal_raw_action[:] = deployment_actions
    q = self._entity.data.joint_pos[:, self._target_ids]
    qdot_nominal = (nominal_target - q) / self._env.step_dt
    foot_pos = self._entity.data.site_pos_w[:, self._site_local_ids]
    jac_xz = self._foot_xz_jacobians(foot_pos)
    jac_x = jac_xz[:, :, 0]
    jac_z = jac_xz[:, :, 1]

    found = self._contact_sensor.data.found
    if found is None:
      raise RuntimeError("CBF contact sensor must provide the 'found' field")
    contact = found > 0
    if contact.ndim > 2:
      contact = contact.any(dim=tuple(range(2, contact.ndim)))
    if contact.ndim != 2 or contact.shape[1] != 2:
      raise RuntimeError("CBF contact sensor must resolve to two feet")
    if self.cfg.deployment_contact_delay_steps:
      new_ids = ~self._deployment_contact_queue_initialized
      if bool(new_ids.any()):
        self._deployment_contact_queue[new_ids] = contact[new_ids].unsqueeze(1)
        self._deployment_contact_queue_initialized[new_ids] = True
      self._deployment_contact_queue[:, 1:] = (
        self._deployment_contact_queue[:, :-1].clone()
      )
      self._deployment_contact_queue[:, 0] = contact
      contact = self._deployment_contact_queue[
        :, self.cfg.deployment_contact_delay_steps
      ]
    in_air = ~contact
    air_time = self._contact_sensor.data.current_air_time
    if air_time is None:
      scores = in_air.float()
    else:
      scores = torch.where(in_air, air_time, torch.full_like(air_time, -1.0))
    foot_index = scores.argmax(dim=1)
    has_swing = in_air.any(dim=1)
    self.selected_foot[:] = torch.where(has_swing, foot_index, -1)
    batch = torch.arange(self.num_envs, device=self.device)
    selected_pos = foot_pos[batch, foot_index]
    selected_jac_x = jac_x[batch, foot_index]
    selected_jac_z = jac_z[batch, foot_index]

    terrain = self._env.scene.terrain
    assert terrain is not None
    edge_x = self._edge_x[terrain.terrain_levels, terrain.terrain_types]
    edge_top_z = self._edge_top_z[terrain.terrain_levels, terrain.terrain_types]
    _, horizontal_h, selected_top_z, edge_active = select_active_riser(
      selected_pos[:, 0],
      selected_pos[:, 2],
      edge_x,
      edge_top_z,
      toe_margin=self.cfg.toe_margin,
      top_clearance=self.cfg.top_clearance,
      activation_distance=self.cfg.activation_distance,
      recovery_distance=self.cfg.recovery_distance,
    )
    active = (
      has_swing
      & edge_active
    )
    if self.cfg.clearance_barrier_slope > 0.0:
      h, normal = sloped_toe_clearance_constraint(
        horizontal_h,
        selected_pos[:, 2],
        selected_top_z,
        selected_jac_x,
        selected_jac_z,
        top_clearance=self.cfg.top_clearance,
        slope=self.cfg.clearance_barrier_slope,
      )
    else:
      h = horizontal_h
      normal = -selected_jac_x
    rhs = -self.cfg.alpha * h
    projected_qdot, psi_nominal, psi_projected = project_halfspace(
      qdot_nominal, normal, rhs, active=active
    )
    projected_target = q + self._env.step_dt * projected_qdot
    projected_raw_action = (projected_target - self.offset) / self.scale
    counterfactual_qdot_norm = torch.linalg.vector_norm(
      projected_qdot - qdot_nominal, dim=1
    )
    counterfactual_target_norm = torch.linalg.vector_norm(
      projected_target - nominal_target, dim=1
    )
    task_velocity_correction = torch.stack(
      (
        torch.sum(selected_jac_x * (projected_qdot - qdot_nominal), dim=1),
        torch.sum(selected_jac_z * (projected_qdot - qdot_nominal), dim=1),
      ),
      dim=1,
    )
    counterfactual_task_norm = float(self._env.step_dt) * torch.linalg.vector_norm(
      task_velocity_correction, dim=1
    )
    would_intervene = active & (
      (psi_nominal < -self.cfg.intervention_epsilon)
      | (counterfactual_target_norm > self.cfg.intervention_epsilon)
    )
    self.counterfactual_policy_projection_valid.zero_()
    if self._counterfactual_policy_action_staged:
      shadow_deployment_action = (
        self.cfg.deployment_action_gain
        * self._deployment_action_scale
        * self.counterfactual_policy_action
        + self._deployment_action_bias
      )
      shadow_target = shadow_deployment_action * self.scale + self.offset
      shadow_qdot = (shadow_target - q) / self._env.step_dt
      shadow_projected_qdot, shadow_margin, _ = project_halfspace(
        shadow_qdot, normal, rhs, active=active
      )
      shadow_projected_target = q + self._env.step_dt * shadow_projected_qdot
      shadow_safe_deployment_action = (
        shadow_projected_target - self.offset
      ) / self.scale
      shadow_safe_policy_action = (
        shadow_safe_deployment_action - self._deployment_action_bias
      ) / (
        self.cfg.deployment_action_gain * self._deployment_action_scale
      )
      shadow_correction_norm = torch.linalg.vector_norm(
        shadow_safe_policy_action - self.counterfactual_policy_action,
        dim=1,
      )
      self.counterfactual_safe_policy_action.copy_(
        shadow_safe_policy_action.detach()
      )
      self.counterfactual_policy_intervened.copy_(
        active
        & (
          (shadow_margin < -self.cfg.intervention_epsilon)
          | (shadow_correction_norm > self.cfg.intervention_epsilon)
        )
      )
      self.counterfactual_policy_correction_norm.copy_(shadow_correction_norm)
      self.counterfactual_policy_nominal_margin.copy_(
        torch.where(active, shadow_margin, torch.zeros_like(shadow_margin))
      )
      self.counterfactual_policy_projection_valid.fill_(True)
      self._counterfactual_policy_action_staged = False
    else:
      self.counterfactual_policy_action.zero_()
      self.counterfactual_safe_policy_action.zero_()
      self.counterfactual_policy_intervened.zero_()
      self.counterfactual_policy_correction_norm.zero_()
      self.counterfactual_policy_nominal_margin.zero_()
    filter_enabled = self.runtime_filter_mask & bool(self.cfg.enabled)
    executed_qdot = torch.where(
      filter_enabled.unsqueeze(-1), projected_qdot, qdot_nominal
    )
    psi_executed = torch.where(filter_enabled, psi_projected, psi_nominal)

    # Match the upstream JointPositionAction behavior: do not add an extra
    # target clamp here. Clipping after projection could invalidate the CBF
    # half-space, while joint-limit handling already remains in the common
    # reward/actuator stack for both experiment arms.
    self._processed_actions[:] = q + self._env.step_dt * executed_qdot
    # ``safe_*`` is the counterfactual CBF projection even when execution is
    # unshielded.  This lets training observe what the filter would have done
    # without claiming that the action was actually changed.
    self.safe_target[:] = projected_target
    self.safe_raw_action[:] = projected_raw_action
    self.executed_raw_action[:] = torch.where(
      filter_enabled.unsqueeze(-1), projected_raw_action, deployment_actions
    )
    self.h[:] = torch.where(active, h, torch.full_like(h, torch.inf))
    self.selected_edge_top_z[:] = selected_top_z
    self.psi_nominal[:] = torch.where(active, psi_nominal, torch.zeros_like(psi_nominal))
    self.psi_filtered[:] = torch.where(
      active, psi_executed, torch.zeros_like(psi_executed)
    )
    self.geometric_active[:] = active
    self.filter_active[:] = active & filter_enabled
    self.intervention_norm[:] = torch.linalg.vector_norm(
      executed_qdot - qdot_nominal, dim=1
    )
    self.target_intervention_norm[:] = torch.linalg.vector_norm(
      self._processed_actions - nominal_target, dim=1
    )
    self.counterfactual_intervention_norm[:] = counterfactual_qdot_norm
    self.counterfactual_target_intervention_norm[:] = counterfactual_target_norm
    self.counterfactual_task_intervention_norm[:] = counterfactual_task_norm
    self.would_intervene[:] = would_intervene
    self.intervened[:] = filter_enabled & would_intervene
    self.intervention_count += self.intervened.float()
    current_h = torch.where(active, h, torch.ones_like(h)).clamp(-1.0, 1.0)
    derivative_valid = active & torch.isfinite(previous_h)
    self.barrier_derivative[:] = torch.where(
      derivative_valid,
      (current_h - previous_h.clamp(-1.0, 1.0)) / self._env.step_dt,
      torch.zeros_like(current_h),
    )
    self.predicted_h[:] = torch.where(
      active,
      (
        current_h
        + self.cfg.safety_prediction_steps
        * self._env.step_dt
        * self.barrier_derivative
      ).clamp(-1.0, 1.0),
      torch.ones_like(current_h),
    )
    self.h_history[:, 1:] = self.h_history[:, :-1].clone()
    self.h_history[:, 0] = current_h
    self.correction_history[:, 1:] = self.correction_history[:, :-1].clone()
    self.correction_history[:, 0] = self.target_intervention_norm

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self.h[env_ids] = torch.inf
    self.psi_nominal[env_ids] = 0.0
    self.psi_filtered[env_ids] = 0.0
    self.filter_active[env_ids] = False
    self.geometric_active[env_ids] = False
    self.intervened[env_ids] = False
    self.would_intervene[env_ids] = False
    self.intervention_norm[env_ids] = 0.0
    self.target_intervention_norm[env_ids] = 0.0
    self.counterfactual_intervention_norm[env_ids] = 0.0
    self.counterfactual_target_intervention_norm[env_ids] = 0.0
    self.counterfactual_task_intervention_norm[env_ids] = 0.0
    self.intervention_count[env_ids] = 0.0
    self.selected_edge_top_z[env_ids] = 0.0
    self.selected_foot[env_ids] = -1
    self.nominal_target[env_ids] = 0.0
    self.safe_target[env_ids] = 0.0
    self.nominal_raw_action[env_ids] = 0.0
    self.safe_raw_action[env_ids] = 0.0
    self.executed_raw_action[env_ids] = 0.0
    self.counterfactual_policy_action[env_ids] = 0.0
    self.counterfactual_safe_policy_action[env_ids] = 0.0
    self.counterfactual_policy_intervened[env_ids] = False
    self.counterfactual_policy_correction_norm[env_ids] = 0.0
    self.counterfactual_policy_nominal_margin[env_ids] = 0.0
    self.counterfactual_policy_projection_valid[env_ids] = False
    self._counterfactual_policy_action_staged = False
    self._deployment_action_queue[env_ids] = 0.0
    self._deployment_contact_queue[env_ids] = False
    self._deployment_contact_queue_initialized[env_ids] = False
    self.barrier_derivative[env_ids] = 0.0
    self.predicted_h[env_ids] = 1.0
    self.h_history[env_ids] = 1.0
    self.correction_history[env_ids] = 0.0
