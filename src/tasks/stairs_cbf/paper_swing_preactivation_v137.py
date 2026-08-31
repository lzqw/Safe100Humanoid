"""Paper-aligned swing-phase CBF preactivation for v137."""

from __future__ import annotations

import math
from typing import Any

from .actions import StairCbfJointPositionActionCfg
from .paper_scaled_continuation_v132 import PaperScaledContinuationV132PPO

METHOD_ID = "paper-cbf-dual-swing-preactivation-v137"


def configure_scheduled_swing_preactivation(env_cfg) -> dict[str, Any]:
  """Extend the contact CBF through scheduled toe-off without changing gait."""
  action = env_cfg.actions.get("joint_pos")
  if not isinstance(action, StairCbfJointPositionActionCfg):
    raise TypeError("v137 requires the configured stair CBF action")
  phase = env_cfg.observations["actor"].terms["phase"].params
  foot_gait = env_cfg.rewards["foot_gait"].params
  swing_force = env_cfg.rewards["swing_foot_force"].params
  period = float(phase["period"])
  stance_fraction = float(foot_gait["threshold"])
  timing_checks = {
    "actor_phase_matches_foot_gait": math.isclose(
      period, float(foot_gait["period"]), rel_tol=0.0, abs_tol=1.0e-12
    ),
    "actor_phase_matches_swing_force": math.isclose(
      period,
      float(swing_force.get("period", 0.6)),
      rel_tol=0.0,
      abs_tol=1.0e-12,
    ),
    "stance_fraction_matches_swing_force": math.isclose(
      stance_fraction,
      float(swing_force.get("stance_fraction", 0.56)),
      rel_tol=0.0,
      abs_tol=1.0e-12,
    ),
    "two_leg_half_cycle_offset": tuple(foot_gait["offset"]) == (0.0, 0.5),
  }
  failed = sorted(name for name, passed in timing_checks.items() if not passed)
  if failed:
    raise ValueError(f"v137 gait clocks differ: {failed}")
  action.scheduled_swing_preactivation = True
  action.gait_period = period
  action.gait_stance_fraction = stance_fraction
  return {
    "method_id": METHOD_ID,
    "selection": "physical_airborne_foot_else_scheduled_toe_off_foot",
    "physical_airborne_foot_has_priority": True,
    "preactivation_requires_both_feet_in_contact": True,
    "period_s": period,
    "stance_fraction": stance_fraction,
    "phase_offsets": list(foot_gait["offset"]),
    "actor_observation_changed": False,
    "reward_changed": False,
    "cbf_geometry_changed": False,
    "timing_checks": timing_checks,
  }


class PaperSwingPreactivationV137PPO(PaperScaledContinuationV132PPO):
  """Keep v132 PPO fixed while changing only the training-time CBF timing."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v132_scaled_paper_continuation": False,
        "v137_method_id": METHOD_ID,
        "v137_scheduled_swing_preactivation": True,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v132_scaled_paper_continuation": False,
        "v137_scheduled_swing_preactivation": True,
      }
    )
    return output
