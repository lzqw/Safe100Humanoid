"""Independent v33 acceleration HOCBF action and evaluation telemetry."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from typing import Any

import torch
from mjlab.envs.mdp.actions import JointPositionAction

from .cbf_math import sloped_toe_clearance_constraint
from .edge_detection import select_active_riser
from .hocbf_math import (
    estimate_hocbf_derivatives,
    hocbf_acceleration_rhs,
    project_task_consistent_hocbf,
)
from .teacher_v26 import (
    HigherRiserCbfAction,
    HigherRiserCbfActionCfg,
    v26_online_safety_telemetry,
)

HOCBF_MODE = "task_consistent_acceleration_hocbf"
CURRENT_CBF_MODE = "current_velocity_cbf"


def joint_diagonal_metric(
    target_names: tuple[str, ...] | list[str],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the fixed left/right-symmetric v33 joint correction metric."""
    values = []
    for name in target_names:
        if name.endswith(("_knee_joint", "_ankle_pitch_joint")):
            values.append(1.0)
        elif name.endswith("_hip_pitch_joint"):
            values.append(2.0)
        else:
            values.append(4.0)
    return torch.tensor(values, device=device, dtype=dtype)


def _quat_roll_pitch(
    quaternion_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return roll and pitch for batched scalar-first quaternions."""
    if quaternion_wxyz.shape[-1] != 4:
        raise ValueError("v33 root quaternion must end in four components")
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x.square() + y.square()),
    )
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def _start_timing(term) -> tuple[Any, Any] | None:
    if not term.cfg.measure_compute_time:
        return None
    if torch.device(term.device).type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end
    return time.perf_counter(), None


def _finish_timing(term, timing: tuple[Any, Any] | None) -> None:
    if timing is None:
        return
    start, end = timing
    if end is None:
        term._cpu_compute_seconds += time.perf_counter() - float(start)
        term._cpu_compute_calls += 1
    else:
        end.record()
        term._cuda_compute_events.append((start, end))


def _initialize_v33_telemetry(term) -> None:
    n = term.num_envs
    a = term.action_dim
    scalar = torch.zeros(n, device=term.device)
    vector = torch.zeros(n, a, device=term.device)
    term.hocbf_h_dot = scalar.clone()
    term.hocbf_instantaneous_drift = scalar.clone()
    term.hocbf_estimated_drift = scalar.clone()
    term.hocbf_rhs = scalar.clone()
    term.hocbf_projected_margin = scalar.clone()
    term.nominal_qddot_norm = scalar.clone()
    term.safe_qddot_norm = scalar.clone()
    term.qddot_correction_norm = scalar.clone()
    term.qddot_correction_jerk = scalar.clone()
    term.foot_forward_acceleration_deviation = scalar.clone()
    term.foot_vertical_acceleration_change = scalar.clone()
    term.nominal_safe_target_error = scalar.clone()
    term.intervention_duration_steps = torch.zeros(
        n, dtype=torch.long, device=term.device
    )
    term.selected_riser = torch.full((n,), -1, dtype=torch.long, device=term.device)
    term._previous_barrier = scalar.clone()
    term._previous_barrier_derivative = scalar.clone()
    term._previous_joint_velocity = vector.clone()
    term._previous_normal = vector.clone()
    term._previous_correction = vector.clone()
    term._previous_drift = scalar.clone()
    term._previous_selected_foot = torch.full(
        (n,), -1, dtype=torch.long, device=term.device
    )
    term._previous_selected_riser = torch.full(
        (n,), -1, dtype=torch.long, device=term.device
    )
    term._previous_active = torch.zeros(n, dtype=torch.bool, device=term.device)
    term._history_valid = torch.zeros(n, dtype=torch.bool, device=term.device)
    term._cuda_compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    term._cpu_compute_seconds = 0.0
    term._cpu_compute_calls = 0


def _reset_v33_telemetry(term, env_ids: torch.Tensor | slice | None) -> None:
    for name in (
        "hocbf_h_dot",
        "hocbf_instantaneous_drift",
        "hocbf_estimated_drift",
        "hocbf_rhs",
        "hocbf_projected_margin",
        "nominal_qddot_norm",
        "safe_qddot_norm",
        "qddot_correction_norm",
        "qddot_correction_jerk",
        "foot_forward_acceleration_deviation",
        "foot_vertical_acceleration_change",
        "nominal_safe_target_error",
        "_previous_barrier",
        "_previous_barrier_derivative",
        "_previous_joint_velocity",
        "_previous_normal",
        "_previous_correction",
        "_previous_drift",
    ):
        getattr(term, name)[env_ids] = 0.0
    term.intervention_duration_steps[env_ids] = 0
    term.selected_riser[env_ids] = -1
    term._previous_selected_foot[env_ids] = -1
    term._previous_selected_riser[env_ids] = -1
    term._previous_active[env_ids] = False
    term._history_valid[env_ids] = False


class _V33Timing:
    def mean_compute_time_ms(self) -> float | None:
        events = self._cuda_compute_events
        if events:
            torch.cuda.synchronize(device=self.device)
            return sum(float(start.elapsed_time(end)) for start, end in events) / len(
                events
            )
        if self._cpu_compute_calls:
            return 1_000.0 * self._cpu_compute_seconds / self._cpu_compute_calls
        return None


@dataclass(kw_only=True)
class InstrumentedCurrentCbfActionCfg(HigherRiserCbfActionCfg):
    """Behavior-identical CBF0 with additive v33 acceleration telemetry."""

    cbf_mode: str = CURRENT_CBF_MODE
    measure_compute_time: bool = True

    def build(self, env) -> InstrumentedCurrentCbfAction:
        return InstrumentedCurrentCbfAction(self, env)


class InstrumentedCurrentCbfAction(_V33Timing, HigherRiserCbfAction):
    """Retain the historical projection exactly and instrument its correction."""

    cfg: InstrumentedCurrentCbfActionCfg

    def __init__(self, cfg: InstrumentedCurrentCbfActionCfg, env) -> None:
        super().__init__(cfg, env)
        _initialize_v33_telemetry(self)

    def process_actions(self, actions: torch.Tensor) -> None:
        timing = _start_timing(self)
        q = self._entity.data.joint_pos[:, self._target_ids].clone()
        qdot = self._entity.data.joint_vel[:, self._target_ids].clone()
        super().process_actions(actions)
        dt = float(self._env.step_dt)
        nominal_velocity = (self.nominal_target - q) / dt
        safe_velocity = (self.safe_target - q) / dt
        nominal_acceleration = (nominal_velocity - qdot) / dt
        safe_acceleration = (safe_velocity - qdot) / dt
        correction = safe_acceleration - nominal_acceleration

        foot_pos = self._entity.data.site_pos_w[:, self._site_local_ids]
        batch = torch.arange(self.num_envs, device=self.device)
        foot_index = self.selected_foot.clamp_min(0)
        selected_pos = foot_pos[batch, foot_index]
        terrain = self._env.scene.terrain
        if terrain is None:
            raise RuntimeError("v33 current-CBF telemetry requires stair metadata")
        edge_x = self._edge_x[terrain.terrain_levels, terrain.terrain_types]
        edge_top_z = self._edge_top_z[terrain.terrain_levels, terrain.terrain_types]
        riser, _, _, edge_active = select_active_riser(
            selected_pos[:, 0],
            selected_pos[:, 2],
            edge_x,
            edge_top_z,
            toe_margin=self.cfg.toe_margin,
            top_clearance=self.cfg.top_clearance,
            activation_distance=self.cfg.activation_distance,
            recovery_distance=self.cfg.recovery_distance,
        )
        active = (self.selected_foot >= 0) & edge_active
        continuous = (
            self._history_valid
            & self._previous_active
            & active
            & (self._previous_selected_foot == self.selected_foot)
            & (self._previous_selected_riser == riser)
        )
        previous = torch.where(
            continuous.unsqueeze(-1),
            self._previous_correction,
            torch.zeros_like(self._previous_correction),
        )
        self.hocbf_h_dot[:] = self.barrier_derivative
        self.hocbf_instantaneous_drift.zero_()
        self.hocbf_estimated_drift.zero_()
        self.hocbf_rhs.zero_()
        self.hocbf_projected_margin[:] = self.psi_filtered
        self.nominal_qddot_norm[:] = torch.linalg.vector_norm(
            nominal_acceleration, dim=-1
        )
        self.safe_qddot_norm[:] = torch.linalg.vector_norm(safe_acceleration, dim=-1)
        self.qddot_correction_norm[:] = torch.linalg.vector_norm(correction, dim=-1)
        self.qddot_correction_jerk[:] = (
            torch.linalg.vector_norm(correction - previous, dim=-1) / dt
        )
        self.foot_forward_acceleration_deviation.zero_()
        self.foot_vertical_acceleration_change.zero_()
        nominal_safe = active & (self.psi_nominal >= 0.0)
        self.nominal_safe_target_error[:] = torch.where(
            nominal_safe,
            torch.amax(torch.abs(self.safe_target - self.nominal_target), dim=-1),
            torch.zeros_like(self.psi_nominal),
        )
        self.intervention_duration_steps[:] = torch.where(
            self.intervened,
            self.intervention_duration_steps + 1,
            torch.zeros_like(self.intervention_duration_steps),
        )
        self.selected_riser[:] = torch.where(active, riser, -1)
        self._previous_correction[:] = torch.where(
            active.unsqueeze(-1), correction, torch.zeros_like(correction)
        )
        self._previous_joint_velocity[:] = qdot
        self._previous_selected_foot[:] = self.selected_foot
        self._previous_selected_riser[:] = riser
        self._previous_active[:] = active
        self._history_valid[:] = True
        _finish_timing(self, timing)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if hasattr(self, "hocbf_h_dot"):
            _reset_v33_telemetry(self, env_ids)


@dataclass(kw_only=True)
class TaskConsistentHocbfActionCfg(HigherRiserCbfActionCfg):
    """Fixed configuration for the v33 acceleration-level task-metric HOCBF."""

    cbf_mode: str = HOCBF_MODE
    omega: float = 8.0
    zeta: float = 1.0
    forward_task_weight: float = 8.0
    correction_smoothness: float = 0.1
    drift_ema_previous: float = 0.8
    drift_clip: float = 20.0
    measure_compute_time: bool = True

    def build(self, env) -> TaskConsistentHocbfAction:
        return TaskConsistentHocbfAction(self, env)


class TaskConsistentHocbfAction(_V33Timing, HigherRiserCbfAction):
    """Acceleration-level HOCBF with a task-consistent closed-form GPU QP."""

    cfg: TaskConsistentHocbfActionCfg

    def __init__(self, cfg: TaskConsistentHocbfActionCfg, env) -> None:
        if cfg.cbf_mode != HOCBF_MODE:
            raise ValueError("v33 HOCBF action has an unexpected mode")
        if cfg.omega <= 0.0 or not math.isclose(cfg.zeta, 1.0, abs_tol=1.0e-12):
            raise ValueError("v33 requires positive omega and zeta exactly one")
        if cfg.forward_task_weight < 0.0 or cfg.correction_smoothness < 0.0:
            raise ValueError("v33 HOCBF weights must be non-negative")
        if not math.isclose(cfg.drift_ema_previous, 0.8, abs_tol=1.0e-12):
            raise ValueError("v33 drift EMA must be fixed at 0.8 previous")
        if not math.isclose(cfg.drift_clip, 20.0, abs_tol=1.0e-12):
            raise ValueError("v33 drift clip must be fixed at 20 m/s^2")
        if cfg.deployment_action_delay_steps != 0:
            raise ValueError("v33 requires zero action delay")
        if cfg.deployment_contact_delay_steps != 0:
            raise ValueError("v33 requires nominal contact sensing")
        if not math.isclose(cfg.deployment_action_gain, 1.0, abs_tol=1.0e-12):
            raise ValueError("v33 requires identity action gain")
        if cfg.deployment_action_scale is not None:
            raise ValueError("v33 requires identity action scale")
        if cfg.deployment_action_bias is not None and any(
            not math.isclose(value, 0.0, abs_tol=1.0e-12)
            for value in cfg.deployment_action_bias
        ):
            raise ValueError("v33 requires zero action bias")
        super().__init__(cfg, env)
        _initialize_v33_telemetry(self)
        self._diagonal_metric = joint_diagonal_metric(
            self._target_names,
            device=self.device,
            dtype=self.nominal_target.dtype,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        timing = _start_timing(self)
        self.toe_riser_kick.zero_()
        pre_filter_foot = self._current_swing_foot()
        self.pre_step_stair_index[:] = self._current_stair_index()
        self.pre_filter_selected_foot[:] = pre_filter_foot

        # The v33 deployment plant is identity.  Calling the upstream position
        # action directly preserves the raw-action PPO coordinates while
        # bypassing CBF0 rather than modifying or overwriting it.
        JointPositionAction.process_actions(self, actions)
        nominal_target = self._processed_actions.clone()
        self.nominal_target[:] = nominal_target
        self.nominal_raw_action[:] = actions
        q = self._entity.data.joint_pos[:, self._target_ids]
        qdot = self._entity.data.joint_vel[:, self._target_ids]
        dt = float(self._env.step_dt)
        nominal_velocity = (nominal_target - q) / dt
        nominal_acceleration = (nominal_velocity - qdot) / dt

        foot_pos = self._entity.data.site_pos_w[:, self._site_local_ids]
        jac_xz = self._foot_xz_jacobians(foot_pos)
        jac_x = jac_xz[:, :, 0]
        jac_z = jac_xz[:, :, 1]
        found = self._contact_sensor.data.found
        if found is None:
            raise RuntimeError("v33 HOCBF requires foot contact state")
        contact = found > 0
        if contact.ndim > 2:
            contact = contact.any(dim=tuple(range(2, contact.ndim)))
        if contact.ndim != 2 or contact.shape[1] != 2:
            raise RuntimeError("v33 contact state must have shape [N, 2]")
        in_air = ~contact
        air_time = self._contact_sensor.data.current_air_time
        scores = (
            in_air.float()
            if air_time is None
            else torch.where(in_air, air_time, torch.full_like(air_time, -1.0))
        )
        foot_index = scores.argmax(dim=1)
        has_swing = in_air.any(dim=1)
        self.selected_foot[:] = torch.where(has_swing, foot_index, -1)
        batch = torch.arange(self.num_envs, device=self.device)
        selected_pos = foot_pos[batch, foot_index]
        selected_jac_x = jac_x[batch, foot_index]
        selected_jac_z = jac_z[batch, foot_index]

        terrain = self._env.scene.terrain
        if terrain is None:
            raise RuntimeError("v33 HOCBF requires exact stair metadata")
        edge_x = self._edge_x[terrain.terrain_levels, terrain.terrain_types]
        edge_top_z = self._edge_top_z[terrain.terrain_levels, terrain.terrain_types]
        riser, horizontal_h, selected_top_z, edge_active = select_active_riser(
            selected_pos[:, 0],
            selected_pos[:, 2],
            edge_x,
            edge_top_z,
            toe_margin=self.cfg.toe_margin,
            top_clearance=self.cfg.top_clearance,
            activation_distance=self.cfg.activation_distance,
            recovery_distance=self.cfg.recovery_distance,
        )
        active = has_swing & edge_active
        h, normal = sloped_toe_clearance_constraint(
            horizontal_h,
            selected_pos[:, 2],
            selected_top_z,
            selected_jac_x,
            selected_jac_z,
            top_clearance=self.cfg.top_clearance,
            slope=self.cfg.clearance_barrier_slope,
        )
        continuous = (
            self._history_valid
            & self._previous_active
            & active
            & (self._previous_selected_foot == self.selected_foot)
            & (self._previous_selected_riser == riser)
        )
        h_dot, _, instantaneous_drift, drift = estimate_hocbf_derivatives(
            h,
            qdot,
            normal,
            self._previous_barrier,
            self._previous_barrier_derivative,
            self._previous_joint_velocity,
            self._previous_drift,
            continuous,
            control_dt=dt,
            drift_ema_previous=self.cfg.drift_ema_previous,
            drift_clip=self.cfg.drift_clip,
        )
        rhs = hocbf_acceleration_rhs(
            h,
            h_dot,
            drift,
            omega=self.cfg.omega,
            zeta=self.cfg.zeta,
        )
        previous_correction = torch.where(
            continuous.unsqueeze(-1),
            self._previous_correction,
            torch.zeros_like(self._previous_correction),
        )
        safe_acceleration, correction, nominal_margin, projected_margin = (
            project_task_consistent_hocbf(
                nominal_acceleration,
                normal,
                rhs,
                previous_correction,
                self._diagonal_metric,
                selected_jac_x,
                active,
                forward_weight=self.cfg.forward_task_weight,
                smoothness_weight=self.cfg.correction_smoothness,
            )
        )
        violated = active & (nominal_margin < 0.0)
        calculated_safe_target = q + dt * (qdot + dt * safe_acceleration)
        projected_target = torch.where(
            violated.unsqueeze(-1), calculated_safe_target, nominal_target
        )
        projected_raw_action = (projected_target - self.offset) / self.scale
        target_correction = projected_target - nominal_target
        would_intervene = violated & (
            torch.linalg.vector_norm(target_correction, dim=-1)
            > self.cfg.intervention_epsilon
        )
        executed_target = projected_target if self.cfg.enabled else nominal_target
        executed_acceleration = (
            safe_acceleration if self.cfg.enabled else nominal_acceleration
        )
        executed_margin = projected_margin if self.cfg.enabled else nominal_margin
        self._processed_actions[:] = executed_target
        self.safe_target[:] = projected_target
        self.safe_raw_action[:] = projected_raw_action
        self.executed_raw_action[:] = (
            projected_raw_action if self.cfg.enabled else actions
        )
        self.h[:] = torch.where(active, h, torch.full_like(h, torch.inf))
        self.selected_edge_top_z[:] = selected_top_z
        self.selected_riser[:] = torch.where(active, riser, -1)
        self.psi_nominal[:] = torch.where(
            active, nominal_margin, torch.zeros_like(nominal_margin)
        )
        self.psi_filtered[:] = torch.where(
            active, executed_margin, torch.zeros_like(executed_margin)
        )
        self.hocbf_projected_margin[:] = torch.where(
            active, projected_margin, torch.zeros_like(projected_margin)
        )
        self.geometric_active[:] = active
        self.filter_active[:] = active & self.cfg.enabled
        self.would_intervene[:] = would_intervene
        self.intervened[:] = self.cfg.enabled & would_intervene
        self.intervention_count += self.intervened.float()
        self.intervention_norm[:] = (
            torch.linalg.vector_norm(
                executed_acceleration - nominal_acceleration, dim=-1
            )
            * dt
        )
        self.target_intervention_norm[:] = torch.linalg.vector_norm(
            executed_target - nominal_target, dim=-1
        )
        self.counterfactual_intervention_norm[:] = (
            torch.linalg.vector_norm(correction, dim=-1) * dt
        )
        self.counterfactual_target_intervention_norm[:] = torch.linalg.vector_norm(
            target_correction, dim=-1
        )
        self.hocbf_h_dot[:] = torch.where(active, h_dot, torch.zeros_like(h_dot))
        self.hocbf_instantaneous_drift[:] = torch.where(
            active, instantaneous_drift, torch.zeros_like(instantaneous_drift)
        )
        self.hocbf_estimated_drift[:] = torch.where(
            active, drift, torch.zeros_like(drift)
        )
        self.hocbf_rhs[:] = torch.where(active, rhs, torch.zeros_like(rhs))
        self.nominal_qddot_norm[:] = torch.linalg.vector_norm(
            nominal_acceleration, dim=-1
        )
        self.safe_qddot_norm[:] = torch.linalg.vector_norm(safe_acceleration, dim=-1)
        self.qddot_correction_norm[:] = torch.linalg.vector_norm(correction, dim=-1)
        self.qddot_correction_jerk[:] = (
            torch.linalg.vector_norm(correction - previous_correction, dim=-1) / dt
        )
        self.foot_forward_acceleration_deviation[:] = torch.abs(
            torch.sum(selected_jac_x * correction, dim=-1)
        )
        self.foot_vertical_acceleration_change[:] = torch.sum(
            selected_jac_z * correction, dim=-1
        )
        self.nominal_safe_target_error[:] = torch.where(
            active & ~violated,
            torch.amax(torch.abs(projected_target - nominal_target), dim=-1),
            torch.zeros_like(nominal_margin),
        )
        self.intervention_duration_steps[:] = torch.where(
            self.intervened,
            self.intervention_duration_steps + 1,
            torch.zeros_like(self.intervention_duration_steps),
        )

        current_h = torch.where(active, h, torch.ones_like(h)).clamp(-1.0, 1.0)
        self.barrier_derivative[:] = self.hocbf_h_dot
        self.predicted_h[:] = torch.where(
            active,
            (
                current_h + self.cfg.safety_prediction_steps * dt * self.hocbf_h_dot
            ).clamp(-1.0, 1.0),
            torch.ones_like(current_h),
        )
        self.h_history[:, 1:] = self.h_history[:, :-1].clone()
        self.h_history[:, 0] = current_h
        self.correction_history[:, 1:] = self.correction_history[:, :-1].clone()
        self.correction_history[:, 0] = self.target_intervention_norm

        self._previous_barrier[:] = h
        self._previous_barrier_derivative[:] = h_dot
        self._previous_joint_velocity[:] = qdot
        self._previous_normal[:] = normal
        self._previous_correction[:] = torch.where(
            active.unsqueeze(-1), correction, torch.zeros_like(correction)
        )
        self._previous_drift[:] = drift
        self._previous_selected_foot[:] = self.selected_foot
        self._previous_selected_riser[:] = riser
        self._previous_active[:] = active
        self._history_valid[:] = True

        self.applied_plant_scale.fill_(1.0)
        self.swing_selection_matches[:] = self.selected_foot == pre_filter_foot
        self.teacher_policy_action[:] = self.safe_raw_action
        self.teacher_reprojected_action[:] = self.safe_raw_action
        self.teacher_reprojection_error.zero_()
        _finish_timing(self, timing)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if hasattr(self, "hocbf_h_dot"):
            _reset_v33_telemetry(self, env_ids)


def v33_online_safety_telemetry(
    env,
    action_name: str = "joint_pos",
    termination_name: str = "fell_over",
) -> torch.Tensor:
    """Append v33 HOCBF, contact, balance and support telemetry per transition."""
    zeros = v26_online_safety_telemetry(
        env, action_name=action_name, termination_name=termination_name
    )
    term = env.action_manager.get_term(action_name)
    if not isinstance(term, (InstrumentedCurrentCbfAction, TaskConsistentHocbfAction)):
        raise TypeError("v33 telemetry requires a v33-instrumented CBF action")
    sensor = env.scene[term.cfg.contact_sensor_name]
    force = sensor.data.force
    found = sensor.data.found
    if force is None or found is None:
        raise RuntimeError("v33 telemetry requires contact force and state")
    force_norm = torch.linalg.vector_norm(force, dim=-1)
    if force_norm.ndim > 2:
        force_norm = force_norm.amax(dim=tuple(range(2, force_norm.ndim)))
    contact = found > 0
    if contact.ndim > 2:
        contact = contact.any(dim=tuple(range(2, contact.ndim)))
    if force_norm.shape != (env.num_envs, 2) or contact.shape != force_norm.shape:
        raise RuntimeError("v33 foot contact telemetry must have shape [N, 2]")
    batch = torch.arange(env.num_envs, device=env.device)
    selected = term.selected_foot.clamp_min(0)
    selected_force = force_norm[batch, selected]
    toe_force = torch.where(
        term.toe_riser_overlap, selected_force, torch.zeros_like(selected_force)
    )
    foot_velocity = term._entity.data.site_lin_vel_w[:, term._site_local_ids, :2]
    slip = torch.linalg.vector_norm(foot_velocity, dim=-1)
    foot_ids = torch.arange(2, device=env.device).view(1, 2)
    support = contact & (foot_ids != term.selected_foot.unsqueeze(1))
    support_slip = torch.where(support, slip, torch.zeros_like(slip)).amax(dim=1)
    roll, pitch = _quat_roll_pitch(term._entity.data.root_link_quat_w)
    angular_velocity = torch.linalg.vector_norm(
        term._entity.data.root_link_ang_vel_b, dim=-1
    )
    payload = {
        "v33_h": term.h,
        "v33_h_dot": term.hocbf_h_dot,
        "v33_estimated_drift": term.hocbf_estimated_drift,
        "v33_nominal_hocbf_margin": term.psi_nominal,
        "v33_filtered_hocbf_margin": term.psi_filtered,
        "v33_projected_hocbf_margin": term.hocbf_projected_margin,
        "v33_nominal_qddot_norm": term.nominal_qddot_norm,
        "v33_safe_qddot_norm": term.safe_qddot_norm,
        "v33_qddot_correction_norm": term.qddot_correction_norm,
        "v33_qddot_correction_jerk": term.qddot_correction_jerk,
        "v33_foot_forward_acceleration_deviation": (
            term.foot_forward_acceleration_deviation
        ),
        "v33_foot_vertical_acceleration_change": (
            term.foot_vertical_acceleration_change
        ),
        "v33_intervention_duration_steps": term.intervention_duration_steps,
        "v33_toe_riser_contact_force": toe_force,
        "v33_toe_riser_contact_impulse_step": toe_force * float(env.step_dt),
        "v33_root_roll": roll,
        "v33_root_pitch": pitch,
        "v33_base_angular_velocity": angular_velocity,
        "v33_support_foot_slip": support_slip,
        "v33_selected_foot": term.selected_foot,
        "v33_selected_riser": term.selected_riser,
        "v33_nominal_safe_target_error": term.nominal_safe_target_error,
    }
    for name, value in payload.items():
        env.extras[name] = value.detach().clone()
    return zeros


def configure_v33_cbf(
    env_cfg,
    *,
    mode: str,
    runtime_filter: bool,
    omega: float | None = None,
    forward_task_weight: float | None = None,
    correction_smoothness: float | None = None,
    measure_compute_time: bool = True,
) -> dict[str, Any]:
    """Replace a v31 action config with one independent v33 CBF mode."""
    original = env_cfg.actions["joint_pos"]
    if not isinstance(original, HigherRiserCbfActionCfg):
        raise TypeError("v33 requires a configured v31 higher-riser action")
    kwargs = {
        field.name: getattr(original, field.name)
        for field in fields(original)
        if field.init
    }
    kwargs.update(
        enabled=bool(runtime_filter),
        measure_compute_time=bool(measure_compute_time),
    )
    if mode == CURRENT_CBF_MODE:
        kwargs.update(cbf_mode=CURRENT_CBF_MODE)
        configured = InstrumentedCurrentCbfActionCfg(**kwargs)
        parameters: dict[str, Any] = {
            "alpha": float(original.alpha),
            "omega": None,
            "forward_task_weight": None,
            "correction_smoothness": None,
        }
    elif mode == HOCBF_MODE:
        if (
            omega is None
            or forward_task_weight is None
            or correction_smoothness is None
        ):
            raise ValueError("v33 HOCBF mode requires omega, lambda_x and lambda_s")
        kwargs.update(
            cbf_mode=HOCBF_MODE,
            omega=float(omega),
            zeta=1.0,
            forward_task_weight=float(forward_task_weight),
            correction_smoothness=float(correction_smoothness),
            drift_ema_previous=0.8,
            drift_clip=20.0,
        )
        configured = TaskConsistentHocbfActionCfg(**kwargs)
        parameters = {
            "alpha": None,
            "omega": float(omega),
            "zeta": 1.0,
            "forward_task_weight": float(forward_task_weight),
            "correction_smoothness": float(correction_smoothness),
            "drift_ema_previous": 0.8,
            "drift_clip_m_per_s2": 20.0,
        }
    else:
        raise ValueError(f"unknown v33 CBF mode {mode!r}")
    env_cfg.actions["joint_pos"] = configured
    telemetry = env_cfg.rewards["online_safety_telemetry"]
    telemetry.func = v33_online_safety_telemetry
    telemetry.params = {
        "action_name": "joint_pos",
        "termination_name": "fell_over",
    }
    return {
        "mode": mode,
        "runtime_filter": bool(runtime_filter),
        "parameters": parameters,
        "current_CBF0_source_unchanged": True,
        "actor_observation_changed": False,
        "critic_observation_changed": False,
        "plant_action_transform": "identity",
    }
