"""Higher-riser deployment shift for the independent v26 follow-up.

The v25 result is intentionally left immutable.  This module reuses its
successful-rescue teacher and PPO data path, but replaces the actuator-gain
shift with one fixed geometric mismatch that the CBF observes exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from .actions import StairCbfJointPositionAction, StairCbfJointPositionActionCfg
from .teacher import (
    SwingUnderResponseCbfAction,
    audit_v25_fixed_deployment_config,
    v25_online_safety_telemetry,
)


@dataclass(kw_only=True)
class HigherRiserCbfActionCfg(StairCbfJointPositionActionCfg):
    """Nominal plant action path paired with exact higher-riser geometry."""

    def build(self, env) -> "HigherRiserCbfAction":
        return HigherRiserCbfAction(self, env)


class HigherRiserCbfAction(SwingUnderResponseCbfAction):
    """Reuse v25 telemetry while keeping the plant transform exactly nominal."""

    cfg: HigherRiserCbfActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        self.toe_riser_kick.zero_()
        selected_foot = self._current_swing_foot()
        self.pre_step_stair_index[:] = self._current_stair_index()
        self.pre_filter_selected_foot[:] = selected_foot

        # Call the historical filter directly: unlike v25, v26 has no hidden
        # actuator map, so its counterfactual safe raw action is already in the
        # Actor's coordinates and needs no inverse transform.
        StairCbfJointPositionAction.process_actions(self, actions)
        self.applied_plant_scale.fill_(1.0)
        self.swing_selection_matches[:] = self.selected_foot == selected_foot
        self.teacher_policy_action[:] = self.safe_raw_action
        self.teacher_reprojected_action[:] = self.safe_raw_action
        self.teacher_reprojection_error.zero_()


def v26_online_safety_telemetry(
    env,
    action_name: str = "joint_pos",
    termination_name: str = "fell_over",
) -> torch.Tensor:
    """Retain the frozen trainer keys and expose unambiguous v26 aliases."""
    zeros = v25_online_safety_telemetry(
        env, action_name=action_name, termination_name=termination_name
    )
    aliases = {
        "v26_teacher_policy_action": "v25_teacher_policy_action",
        "v26_applied_plant_scale": "v25_applied_plant_scale",
        "v26_pre_step_stair_index": "v25_pre_step_stair_index",
        "v26_teacher_reprojection_error": "v25_teacher_reprojection_error",
        "v26_swing_selection_matches": "v25_swing_selection_matches",
        "v26_toe_riser_kick": "v25_toe_riser_kick",
        "v26_toe_riser_overlap": "v25_toe_riser_overlap",
    }
    for destination, source in aliases.items():
        env.extras[destination] = env.extras[source]
    return zeros


def configure_v26_higher_riser(
    env_cfg,
    *,
    riser_height_m: float,
    runtime_filter: bool,
    clearance_barrier_slope: float = 0.0,
    recovery_distance_m: float = 0.15,
    filter_alpha: float = 10.0,
) -> dict[str, Any]:
    """Install one fixed higher-riser deployment without changing observations."""
    height = float(riser_height_m)
    if not math.isfinite(height) or not 0.13 < height <= 0.20:
        raise ValueError("v26 riser height must lie in (0.13, 0.20] m")
    slope = float(clearance_barrier_slope)
    if not math.isfinite(slope) or not 0.0 <= slope <= 2.0:
        raise ValueError("v26 clearance barrier slope must lie in [0, 2]")
    recovery_distance = float(recovery_distance_m)
    if not math.isfinite(recovery_distance) or not 0.0 <= recovery_distance <= 0.30:
        raise ValueError("v26 recovery distance must lie in [0, 0.30] m")
    alpha = float(filter_alpha)
    if not math.isfinite(alpha) or not 0.5 <= alpha <= 30.0:
        raise ValueError("v26 filter alpha must lie in [0.5, 30]")
    fixed_deployment = audit_v25_fixed_deployment_config(env_cfg)

    terrain = env_cfg.scene.terrain
    generator = None if terrain is None else terrain.terrain_generator
    if generator is None or set(generator.sub_terrains) != {"forward_stairs"}:
        raise RuntimeError("v26 requires the fixed single-profile DQ stair terrain")
    original_stairs = generator.sub_terrains["forward_stairs"]
    if original_stairs.step_height_profile is not None:
        raise RuntimeError("v26 requires a uniform baseline riser profile")
    generator.sub_terrains["forward_stairs"] = replace(
        original_stairs,
        step_height_range=(height, height),
        step_height_profile=None,
    )

    original = env_cfg.actions["joint_pos"]
    if not isinstance(original, StairCbfJointPositionActionCfg):
        raise TypeError("v26 requires the historical stair CBF action config")
    kwargs = {
        field.name: getattr(original, field.name)
        for field in fields(original)
        if field.init
    }
    kwargs.update(
        enabled=bool(runtime_filter),
        step_height=height,
        clearance_barrier_slope=slope,
        recovery_distance=recovery_distance,
        alpha=alpha,
        deployment_action_gain=1.0,
        deployment_action_scale=None,
        deployment_action_bias=None,
        deployment_action_delay_steps=0,
        deployment_contact_delay_steps=0,
        deployment_contact_phase_offset=0.0,
    )
    env_cfg.actions["joint_pos"] = HigherRiserCbfActionCfg(**kwargs)
    env_cfg.rewards.pop("specialist_failure_signal", None)
    telemetry = env_cfg.rewards["online_safety_telemetry"]
    telemetry.func = v26_online_safety_telemetry
    telemetry.params = {
        "action_name": "joint_pos",
        "termination_name": "fell_over",
    }
    return {
        "shift": "fixed_uniform_higher_riser",
        "riser_height_m": height,
        "baseline_riser_height_m": 0.13,
        "clearance_barrier_slope": slope,
        "recovery_distance_m": recovery_distance,
        "filter_alpha": alpha,
        "clearance_barrier": (
            "sloped_xz" if slope > 0.0 else "historical_horizontal_only"
        ),
        "runtime_filter": bool(runtime_filter),
        "terrain_geometry_changed": True,
        "friction_changed": False,
        "command_changed": False,
        "controller_changed": False,
        "plant_action_transform": "identity",
        "actor_observation_fields_added": 0,
        "cbf_geometry_exact": True,
        "fixed_deployment_environment": fixed_deployment,
    }
