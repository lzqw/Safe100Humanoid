"""Outcome-optimized task-metric velocity CBF action and telemetry for v34."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields
from typing import Any

import torch
from mjlab.envs.mdp.actions import JointPositionAction

from .cbf_math import sloped_toe_clearance_constraint
from .edge_detection import select_active_riser
from .teacher_v26 import (
    HigherRiserCbfAction,
    HigherRiserCbfActionCfg,
    v26_online_safety_telemetry,
)
from .velocity_cbf_math import project_task_metric_velocity_cbf

CURRENT_CBF_MODE = "current_velocity_cbf"
OPTIMIZED_CBF_MODE = "task_metric_velocity_cbf"


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


def _initialize_v34(term) -> None:
    n, a = term.num_envs, term.action_dim
    scalar = torch.zeros(n, device=term.device)
    vector = torch.zeros(n, a, device=term.device)
    term.velocity_correction_norm = scalar.clone()
    term.velocity_correction_jerk = scalar.clone()
    term.foot_forward_velocity_deviation = scalar.clone()
    term.foot_vertical_velocity_change = scalar.clone()
    term.nominal_safe_target_error = scalar.clone()
    term.intervention_duration_steps = torch.zeros(
        n, dtype=torch.long, device=term.device
    )
    term.selected_riser = torch.full((n,), -1, dtype=torch.long, device=term.device)
    term._v34_previous_safe_velocity = vector.clone()
    term._v34_previous_correction = vector.clone()
    term._v34_previous_selected_foot = torch.full(
        (n,), -1, dtype=torch.long, device=term.device
    )
    term._v34_previous_selected_riser = torch.full(
        (n,), -1, dtype=torch.long, device=term.device
    )
    term._v34_previous_active = torch.zeros(n, dtype=torch.bool, device=term.device)
    term._v34_history_valid = torch.zeros(n, dtype=torch.bool, device=term.device)
    term._cuda_compute_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    term._cpu_compute_seconds = 0.0
    term._cpu_compute_calls = 0


def _reset_v34(term, env_ids: torch.Tensor | slice | None) -> None:
    for name in (
        "velocity_correction_norm",
        "velocity_correction_jerk",
        "foot_forward_velocity_deviation",
        "foot_vertical_velocity_change",
        "nominal_safe_target_error",
        "_v34_previous_safe_velocity",
        "_v34_previous_correction",
    ):
        getattr(term, name)[env_ids] = 0.0
    term.intervention_duration_steps[env_ids] = 0
    term.selected_riser[env_ids] = -1
    term._v34_previous_selected_foot[env_ids] = -1
    term._v34_previous_selected_riser[env_ids] = -1
    term._v34_previous_active[env_ids] = False
    term._v34_history_valid[env_ids] = False


class _V34Timing:
    def mean_compute_time_ms(self) -> float | None:
        if self._cuda_compute_events:
            torch.cuda.synchronize(device=self.device)
            return sum(
                float(start.elapsed_time(end))
                for start, end in self._cuda_compute_events
            ) / len(self._cuda_compute_events)
        if self._cpu_compute_calls:
            return 1_000.0 * self._cpu_compute_seconds / self._cpu_compute_calls
        return None


@dataclass(kw_only=True)
class InstrumentedCurrentVelocityCbfActionCfg(HigherRiserCbfActionCfg):
    cbf_mode: str = CURRENT_CBF_MODE
    measure_compute_time: bool = True

    def build(self, env) -> InstrumentedCurrentVelocityCbfAction:
        return InstrumentedCurrentVelocityCbfAction(self, env)


class InstrumentedCurrentVelocityCbfAction(_V34Timing, HigherRiserCbfAction):
    """Behavior-identical historical CBF0 with v34-only instrumentation."""

    cfg: InstrumentedCurrentVelocityCbfActionCfg

    def __init__(self, cfg: InstrumentedCurrentVelocityCbfActionCfg, env) -> None:
        super().__init__(cfg, env)
        _initialize_v34(self)

    def process_actions(self, actions: torch.Tensor) -> None:
        timing = _start_timing(self)
        q = self._entity.data.joint_pos[:, self._target_ids].clone()
        super().process_actions(actions)
        dt = float(self._env.step_dt)
        nominal_velocity = (self.nominal_target - q) / dt
        safe_velocity = (self.safe_target - q) / dt
        counterfactual_correction = safe_velocity - nominal_velocity
        executed_correction = (
            counterfactual_correction
            if self.cfg.enabled
            else torch.zeros_like(counterfactual_correction)
        )
        foot_pos = self._entity.data.site_pos_w[:, self._site_local_ids]
        batch = torch.arange(self.num_envs, device=self.device)
        selected = self.selected_foot.clamp_min(0)
        selected_pos = foot_pos[batch, selected]
        terrain = self._env.scene.terrain
        if terrain is None:
            raise RuntimeError("v34 current CBF requires stair metadata")
        edge_x = self._edge_x[terrain.terrain_levels, terrain.terrain_types]
        edge_z = self._edge_top_z[terrain.terrain_levels, terrain.terrain_types]
        riser, _, _, edge_active = select_active_riser(
            selected_pos[:, 0],
            selected_pos[:, 2],
            edge_x,
            edge_z,
            toe_margin=self.cfg.toe_margin,
            top_clearance=self.cfg.top_clearance,
            activation_distance=self.cfg.activation_distance,
            recovery_distance=self.cfg.recovery_distance,
        )
        active = (self.selected_foot >= 0) & edge_active
        continuous = (
            self._v34_history_valid
            & self._v34_previous_active
            & active
            & (self._v34_previous_selected_foot == self.selected_foot)
            & (self._v34_previous_selected_riser == riser)
        )
        previous = torch.where(
            continuous.unsqueeze(-1),
            self._v34_previous_correction,
            torch.zeros_like(executed_correction),
        )
        self.velocity_correction_norm[:] = torch.linalg.vector_norm(
            executed_correction, dim=-1
        )
        self.velocity_correction_jerk[:] = (
            torch.linalg.vector_norm(executed_correction - previous, dim=-1) / dt
        )
        self.foot_forward_velocity_deviation.zero_()
        self.foot_vertical_velocity_change.zero_()
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
        self._v34_previous_safe_velocity[:] = safe_velocity
        self._v34_previous_correction[:] = executed_correction
        self._v34_previous_selected_foot[:] = self.selected_foot
        self._v34_previous_selected_riser[:] = riser
        self._v34_previous_active[:] = active
        self._v34_history_valid[:] = True
        _finish_timing(self, timing)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if hasattr(self, "velocity_correction_norm"):
            _reset_v34(self, env_ids)


@dataclass(kw_only=True)
class TaskMetricVelocityCbfActionCfg(HigherRiserCbfActionCfg):
    cbf_mode: str = OPTIMIZED_CBF_MODE
    swing_knee_weight: float = 1.0
    swing_ankle_pitch_weight: float = 1.0
    swing_hip_pitch_weight: float = 1.0
    stance_leg_weight: float = 1.0
    hip_roll_yaw_weight: float = 1.0
    other_joint_weight: float = 1.0
    forward_task_weight: float = 0.0
    correction_smoothness: float = 0.0
    measure_compute_time: bool = True

    def build(self, env) -> TaskMetricVelocityCbfAction:
        return TaskMetricVelocityCbfAction(self, env)


class TaskMetricVelocityCbfAction(_V34Timing, HigherRiserCbfAction):
    """Single-constraint task-metric velocity CBF with no acceleration model."""

    cfg: TaskMetricVelocityCbfActionCfg

    def __init__(self, cfg: TaskMetricVelocityCbfActionCfg, env) -> None:
        if cfg.cbf_mode != OPTIMIZED_CBF_MODE:
            raise ValueError("unexpected v34 CBF mode")
        for name in (
            "swing_knee_weight",
            "swing_ankle_pitch_weight",
            "swing_hip_pitch_weight",
            "stance_leg_weight",
            "hip_roll_yaw_weight",
            "other_joint_weight",
        ):
            if not math.isfinite(float(getattr(cfg, name))) or getattr(cfg, name) <= 0:
                raise ValueError(f"v34 {name} must be finite and positive")
        if cfg.forward_task_weight < 0.0 or cfg.correction_smoothness < 0.0:
            raise ValueError("v34 task weights must be non-negative")
        if cfg.deployment_action_delay_steps != 0:
            raise ValueError("v34 requires zero action delay")
        if cfg.deployment_contact_delay_steps != 0:
            raise ValueError("v34 requires nominal contact sensing")
        if not math.isclose(cfg.deployment_action_gain, 1.0, abs_tol=1.0e-12):
            raise ValueError("v34 requires identity action gain")
        if cfg.deployment_action_scale is not None:
            raise ValueError("v34 requires identity action scale")
        super().__init__(cfg, env)
        _initialize_v34(self)
        self._left_swing_metric = self._metric_for_swing("left")
        self._right_swing_metric = self._metric_for_swing("right")

    def _metric_for_swing(self, swing_side: str) -> torch.Tensor:
        values = []
        other_side = "right" if swing_side == "left" else "left"
        for name in self._target_names:
            if name.startswith(f"{other_side}_"):
                value = self.cfg.stance_leg_weight
            elif name.endswith("_knee_joint"):
                value = self.cfg.swing_knee_weight
            elif name.endswith("_ankle_pitch_joint"):
                value = self.cfg.swing_ankle_pitch_weight
            elif name.endswith("_hip_pitch_joint"):
                value = self.cfg.swing_hip_pitch_weight
            elif name.endswith(("_hip_roll_joint", "_hip_yaw_joint")):
                value = self.cfg.hip_roll_yaw_weight
            else:
                value = self.cfg.other_joint_weight
            values.append(float(value))
        return torch.tensor(values, device=self.device, dtype=self.nominal_target.dtype)

    def process_actions(self, actions: torch.Tensor) -> None:
        timing = _start_timing(self)
        self.toe_riser_kick.zero_()
        pre_filter_foot = self._current_swing_foot()
        self.pre_step_stair_index[:] = self._current_stair_index()
        self.pre_filter_selected_foot[:] = pre_filter_foot
        JointPositionAction.process_actions(self, actions)
        nominal_target = self._processed_actions.clone()
        self.nominal_target[:] = nominal_target
        self.nominal_raw_action[:] = actions
        q = self._entity.data.joint_pos[:, self._target_ids]
        dt = float(self._env.step_dt)
        nominal_velocity = (nominal_target - q) / dt

        foot_pos = self._entity.data.site_pos_w[:, self._site_local_ids]
        jac_xz = self._foot_xz_jacobians(foot_pos)
        jac_x, jac_z = jac_xz[:, :, 0], jac_xz[:, :, 1]
        found = self._contact_sensor.data.found
        if found is None:
            raise RuntimeError("v34 CBF requires foot contact state")
        contact = found > 0
        if contact.ndim > 2:
            contact = contact.any(dim=tuple(range(2, contact.ndim)))
        if contact.ndim != 2 or contact.shape[1] != 2:
            raise RuntimeError("v34 contact state must have shape [N, 2]")
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
        diagonal_metric = torch.where(
            (foot_index == 0).unsqueeze(-1),
            self._left_swing_metric,
            self._right_swing_metric,
        )

        terrain = self._env.scene.terrain
        if terrain is None:
            raise RuntimeError("v34 CBF requires exact stair metadata")
        edge_x = self._edge_x[terrain.terrain_levels, terrain.terrain_types]
        edge_z = self._edge_top_z[terrain.terrain_levels, terrain.terrain_types]
        riser, horizontal_h, selected_top_z, edge_active = select_active_riser(
            selected_pos[:, 0],
            selected_pos[:, 2],
            edge_x,
            edge_z,
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
        rhs = -float(self.cfg.alpha) * h
        continuous = (
            self._v34_history_valid
            & self._v34_previous_active
            & active
            & (self._v34_previous_selected_foot == self.selected_foot)
            & (self._v34_previous_selected_riser == riser)
        )
        safe_velocity, correction, nominal_margin, projected_margin = (
            project_task_metric_velocity_cbf(
                nominal_velocity,
                self._v34_previous_safe_velocity,
                normal,
                rhs,
                diagonal_metric,
                selected_jac_x,
                active,
                continuous,
                forward_weight=self.cfg.forward_task_weight,
                smoothness_weight=self.cfg.correction_smoothness,
            )
        )
        violated = active & (nominal_margin < 0.0)
        calculated_target = q + dt * safe_velocity
        projected_target = torch.where(
            violated.unsqueeze(-1), calculated_target, nominal_target
        )
        projected_raw = (projected_target - self.offset) / self.scale
        target_correction = projected_target - nominal_target
        would_intervene = active & (
            (nominal_margin < -self.cfg.intervention_epsilon)
            | (
                torch.linalg.vector_norm(target_correction, dim=-1)
                > self.cfg.intervention_epsilon
            )
        )
        executed_velocity = safe_velocity if self.cfg.enabled else nominal_velocity
        executed_target = projected_target if self.cfg.enabled else nominal_target
        executed_margin = projected_margin if self.cfg.enabled else nominal_margin
        executed_correction = (
            correction if self.cfg.enabled else torch.zeros_like(correction)
        )
        self._processed_actions[:] = executed_target
        self.safe_target[:] = projected_target
        self.safe_raw_action[:] = projected_raw
        self.executed_raw_action[:] = projected_raw if self.cfg.enabled else actions
        self.h[:] = torch.where(active, h, torch.full_like(h, torch.inf))
        self.selected_edge_top_z[:] = selected_top_z
        self.selected_riser[:] = torch.where(active, riser, -1)
        self.psi_nominal[:] = torch.where(
            active, nominal_margin, torch.zeros_like(nominal_margin)
        )
        self.psi_filtered[:] = torch.where(
            active, executed_margin, torch.zeros_like(executed_margin)
        )
        self.geometric_active[:] = active
        self.filter_active[:] = active & self.cfg.enabled
        self.would_intervene[:] = would_intervene
        self.intervened[:] = self.cfg.enabled & would_intervene
        self.intervention_count += self.intervened.float()
        self.intervention_norm[:] = torch.linalg.vector_norm(
            executed_velocity - nominal_velocity, dim=-1
        )
        self.target_intervention_norm[:] = torch.linalg.vector_norm(
            executed_target - nominal_target, dim=-1
        )
        self.counterfactual_intervention_norm[:] = torch.linalg.vector_norm(
            correction, dim=-1
        )
        self.counterfactual_target_intervention_norm[:] = torch.linalg.vector_norm(
            target_correction, dim=-1
        )
        previous_correction = torch.where(
            continuous.unsqueeze(-1),
            self._v34_previous_correction,
            torch.zeros_like(correction),
        )
        self.velocity_correction_norm[:] = torch.linalg.vector_norm(
            executed_correction, dim=-1
        )
        self.velocity_correction_jerk[:] = (
            torch.linalg.vector_norm(executed_correction - previous_correction, dim=-1)
            / dt
        )
        self.foot_forward_velocity_deviation[:] = torch.abs(
            torch.sum(selected_jac_x * executed_correction, dim=-1)
        )
        self.foot_vertical_velocity_change[:] = torch.sum(
            selected_jac_z * executed_correction, dim=-1
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
        previous_h = self.h_history[:, 0]
        derivative_valid = active & self._v34_previous_active
        self.barrier_derivative[:] = torch.where(
            derivative_valid,
            (current_h - previous_h) / dt,
            torch.zeros_like(current_h),
        )
        self.predicted_h[:] = torch.where(
            active,
            (
                current_h
                + self.cfg.safety_prediction_steps * dt * self.barrier_derivative
            ).clamp(-1.0, 1.0),
            torch.ones_like(current_h),
        )
        self.h_history[:, 1:] = self.h_history[:, :-1].clone()
        self.h_history[:, 0] = current_h
        self.correction_history[:, 1:] = self.correction_history[:, :-1].clone()
        self.correction_history[:, 0] = self.target_intervention_norm
        self._v34_previous_safe_velocity[:] = executed_velocity
        self._v34_previous_correction[:] = executed_correction
        self._v34_previous_selected_foot[:] = self.selected_foot
        self._v34_previous_selected_riser[:] = riser
        self._v34_previous_active[:] = active
        self._v34_history_valid[:] = True
        self.applied_plant_scale.fill_(1.0)
        self.swing_selection_matches[:] = self.selected_foot == pre_filter_foot
        self.teacher_policy_action[:] = self.safe_raw_action
        self.teacher_reprojected_action[:] = self.safe_raw_action
        self.teacher_reprojection_error.zero_()
        _finish_timing(self, timing)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if hasattr(self, "velocity_correction_norm"):
            _reset_v34(self, env_ids)


def _quat_roll_pitch(
    quaternion_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x.square() + y.square()),
    )
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def v34_online_safety_telemetry(
    env,
    action_name: str = "joint_pos",
    termination_name: str = "fell_over",
) -> torch.Tensor:
    zeros = v26_online_safety_telemetry(
        env, action_name=action_name, termination_name=termination_name
    )
    term = env.action_manager.get_term(action_name)
    if not isinstance(
        term,
        (InstrumentedCurrentVelocityCbfAction, TaskMetricVelocityCbfAction),
    ):
        raise TypeError("v34 telemetry requires a v34 velocity-CBF action")
    sensor = env.scene[term.cfg.contact_sensor_name]
    force, found = sensor.data.force, sensor.data.found
    if force is None or found is None:
        raise RuntimeError("v34 telemetry requires contact force and state")
    force_norm = torch.linalg.vector_norm(force, dim=-1)
    if force_norm.ndim > 2:
        force_norm = force_norm.amax(dim=tuple(range(2, force_norm.ndim)))
    contact = found > 0
    if contact.ndim > 2:
        contact = contact.any(dim=tuple(range(2, contact.ndim)))
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
    payload = {
        "v34_h": term.h,
        "v34_nominal_margin": term.psi_nominal,
        "v34_filtered_margin": term.psi_filtered,
        "v34_velocity_correction_norm": term.velocity_correction_norm,
        "v34_velocity_correction_jerk": term.velocity_correction_jerk,
        "v34_forward_velocity_deviation": term.foot_forward_velocity_deviation,
        "v34_vertical_velocity_change": term.foot_vertical_velocity_change,
        "v34_intervention_duration_steps": term.intervention_duration_steps,
        "v34_toe_riser_contact_force": toe_force,
        "v34_toe_riser_contact_impulse_step": toe_force * float(env.step_dt),
        "v34_root_roll": roll,
        "v34_root_pitch": pitch,
        "v34_base_angular_velocity": torch.linalg.vector_norm(
            term._entity.data.root_link_ang_vel_b, dim=-1
        ),
        "v34_support_foot_slip": support_slip,
        "v34_selected_foot": term.selected_foot,
        "v34_selected_riser": term.selected_riser,
        "v34_nominal_safe_target_error": term.nominal_safe_target_error,
    }
    for name, value in payload.items():
        env.extras[name] = value.detach().clone()
    return zeros


def configure_v34_cbf(
    env_cfg,
    *,
    mode: str,
    runtime_filter: bool,
    parameters: dict[str, Any] | None = None,
    measure_compute_time: bool = True,
) -> dict[str, Any]:
    original = env_cfg.actions["joint_pos"]
    if not isinstance(original, HigherRiserCbfActionCfg):
        raise TypeError("v34 requires a configured v31 action")
    kwargs = {
        field.name: getattr(original, field.name)
        for field in fields(original)
        if field.init
    }
    kwargs.update(
        enabled=bool(runtime_filter), measure_compute_time=bool(measure_compute_time)
    )
    if mode == CURRENT_CBF_MODE:
        kwargs.update(cbf_mode=CURRENT_CBF_MODE)
        configured = InstrumentedCurrentVelocityCbfActionCfg(**kwargs)
        actual = {
            "barrier_slope": float(original.clearance_barrier_slope),
            "alpha": float(original.alpha),
            "top_clearance": float(original.top_clearance),
            "toe_margin": float(original.toe_margin),
            "metric": "euclidean",
            "lambda_x": 0.0,
            "lambda_s": 0.0,
        }
    elif mode == OPTIMIZED_CBF_MODE:
        if parameters is None:
            raise ValueError("v34 optimized mode requires parameters")
        actual = dict(parameters)
        kwargs.update(
            cbf_mode=OPTIMIZED_CBF_MODE,
            clearance_barrier_slope=float(actual["barrier_slope"]),
            alpha=float(actual["alpha"]),
            top_clearance=float(actual["top_clearance"]),
            toe_margin=float(actual["toe_margin"]),
            swing_knee_weight=float(actual["swing_knee_weight"]),
            swing_ankle_pitch_weight=float(actual["swing_ankle_pitch_weight"]),
            swing_hip_pitch_weight=float(actual["swing_hip_pitch_weight"]),
            stance_leg_weight=float(actual["stance_leg_weight"]),
            hip_roll_yaw_weight=float(actual["hip_roll_yaw_weight"]),
            other_joint_weight=float(actual["other_joint_weight"]),
            forward_task_weight=float(actual["lambda_x"]),
            correction_smoothness=float(actual["lambda_s"]),
        )
        configured = TaskMetricVelocityCbfActionCfg(**kwargs)
    else:
        raise ValueError(f"unknown v34 CBF mode {mode!r}")
    env_cfg.actions["joint_pos"] = configured
    telemetry = env_cfg.rewards["online_safety_telemetry"]
    telemetry.func = v34_online_safety_telemetry
    telemetry.params = {
        "action_name": "joint_pos",
        "termination_name": "fell_over",
    }
    return {
        "mode": mode,
        "runtime_filter": bool(runtime_filter),
        "parameters": actual,
        "current_CBF0_source_unchanged": True,
        "actor_observation_changed": False,
        "critic_observation_changed": False,
        "plant_action_transform": "identity",
        "acceleration_or_drift_model": False,
    }
