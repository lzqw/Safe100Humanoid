"""Frozen hidden deployment contexts for failure-focused online refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import FlatPatchSamplingCfg


FAILURE_FOCUSED_CONTEXT_SCHEMA_VERSION = 1
FAILURE_FOCUSED_CONTEXT_KIND = "failure_focused_brief_ppo_v15"
FAILURE_FOCUSED_CALIBRATION_KIND = "base_success_first_qualifying_v1"
SPECIALIST_CONTEXT_SCHEMA_VERSION = 1
SPECIALIST_CONTEXT_KIND = "failure_mode_conditioned_brief_ppo_v17"
SPECIALIST_CALIBRATION_KIND = (
  "base_policy_single_dominant_failure_first_qualifying_v1"
)
SPECIALIST_MODES = ("lateral", "cbf", "balance")
SPECIALIST_FAILURE_TYPES = {
  "lateral": "lateral_heading_drift",
  "cbf": "non_lateral_high_cbf_demand",
  "balance": "non_lateral_balance_or_phase",
}
V19_CONTEXT_SCHEMA_VERSION = 1
V19_CONTEXT_KIND = "observable_failure_conditioned_brief_ppo_v19"
V19_CALIBRATION_KIND = (
  "base_policy_observable_failure_first_qualifying_v1"
)
V19_SPECIALIST_MODES = ("lateral", "contact_stability")
V19_SPECIALIST_FAILURE_TYPES = {
  "lateral": "lateral_heading_drift",
  "contact_stability": "contact_stability",
}
V21_CONTEXT_SCHEMA_VERSION = 1
V21_CONTEXT_KIND = "local_matched_success_preservation_v21"
V21_CALIBRATION_KIND = (
  "base_policy_context_family_first_qualifying_v21"
)
V21_CONTEXT_SPECS: dict[str, dict[str, Any]] = {
  "L_dev": {
    "mode": "lateral",
    "family": "excluded_development_mixed_lateral",
    "formal": False,
    "candidate_seed_prefix": 211,
    "direction": -1,
    "ranges": {
      "rise_profile_half_width": [0.0015, 0.0015],
      "tread_profile_half_width": [0.003, 0.003],
      "command_forward_scale": [1.00, 1.00],
      "command_delay_s": [0.14, 0.14],
      "command_low_pass_s": [0.12, 0.12],
      "action_gain": [0.995, 0.995],
      "action_bias_abs": [0.004, 0.004],
      "action_delay_steps": [0, 0],
      "encoder_bias_abs": [0.001, 0.001],
      "lateral_bias_abs": [0.030, 0.080],
      "yaw_bias_abs": [0.060, 0.160],
      "lateral_pulse_min": [0.040, 0.075],
      "lateral_pulse_max": [0.095, 0.170],
      "yaw_pulse_min": [0.090, 0.180],
      "yaw_pulse_max": [0.220, 0.400],
      "centerline_lateral_gain": [0.50, 0.50],
      "centerline_heading_gain": [0.85, 0.85],
      "centerline_max_lateral_velocity": [0.12, 0.12],
      "centerline_max_yaw_velocity": [0.32, 0.32],
    },
  },
  "C_dev": {
    "mode": "contact_stability",
    "family": "excluded_development_mixed_contact",
    "formal": False,
    "candidate_seed_prefix": 212,
    "direction": 1,
    "ranges": {
      "foot_friction": [0.52, 0.30],
      "contact_delay_steps": [1, 2],
      "phase_offset_abs": [0.02, 0.08],
      "response_asymmetry_abs": [0.010, 0.050],
      "action_gain": [1.0, 0.94],
      "action_bias_abs": [0.002, 0.025],
    },
  },
  "L1": {
    "mode": "lateral",
    "family": "command_delay_and_low_pass",
    "formal": True,
    "candidate_seed_prefix": 213,
    "direction": 1,
    "ranges": {
      "rise_profile_half_width": [0.0035, 0.0035],
      "tread_profile_half_width": [0.007, 0.007],
      "command_forward_scale": [1.00, 1.03],
      "command_delay_s": [0.14, 0.38],
      "command_low_pass_s": [0.12, 0.36],
      "action_gain": [0.99, 0.97],
      "action_bias_abs": [0.010, 0.016],
      "action_delay_steps": [1, 1],
      "encoder_bias_abs": [0.002, 0.007],
      "lateral_bias_abs": [0.030, 0.075],
      "yaw_bias_abs": [0.060, 0.150],
      "lateral_pulse_min": [0.045, 0.045],
      "lateral_pulse_max": [0.105, 0.105],
      "yaw_pulse_min": [0.100, 0.100],
      "yaw_pulse_max": [0.240, 0.240],
      "centerline_lateral_gain": [0.48, 0.38],
      "centerline_heading_gain": [0.82, 0.64],
    },
  },
  "L2": {
    "mode": "lateral",
    "family": "extended_yaw_bias_and_yaw_pulse",
    "formal": True,
    "candidate_seed_prefix": 214,
    "direction": -1,
    "ranges": {
      "rise_profile_half_width": [0.0, 0.0],
      "tread_profile_half_width": [0.0, 0.0],
      "command_forward_scale": [1.00, 1.00],
      "command_delay_s": [0.08, 0.08],
      "command_low_pass_s": [0.06, 0.06],
      "action_gain": [1.0, 1.0],
      "action_bias_abs": [0.0, 0.0],
      "action_delay_steps": [0, 0],
      "encoder_bias_abs": [0.0, 0.0],
      "yaw_bias_abs": [0.30, 0.50],
      "lateral_pulse_min": [0.035, 0.035],
      "lateral_pulse_max": [0.085, 0.085],
      "yaw_pulse_min": [0.34, 0.52],
      "yaw_pulse_max": [0.72, 1.08],
      "centerline_lateral_gain": [0.50, 0.50],
      "centerline_heading_gain": [0.86, 0.86],
    },
  },
  "L3": {
    "mode": "lateral",
    "family": "lateral_bias_and_lateral_pulse",
    "formal": True,
    "candidate_seed_prefix": 215,
    "direction": 1,
    "ranges": {
      "rise_profile_half_width": [0.0035, 0.0035],
      "tread_profile_half_width": [0.007, 0.007],
      "command_forward_scale": [1.00, 1.025],
      "command_delay_s": [0.20, 0.24],
      "command_low_pass_s": [0.18, 0.22],
      "action_gain": [0.99, 0.97],
      "action_bias_abs": [0.010, 0.016],
      "action_delay_steps": [1, 1],
      "encoder_bias_abs": [0.002, 0.007],
      "lateral_bias_abs": [0.05, 0.20],
      "lateral_pulse_min": [0.05, 0.14],
      "lateral_pulse_max": [0.12, 0.32],
      "yaw_pulse_min": [0.080, 0.080],
      "yaw_pulse_max": [0.180, 0.180],
      "centerline_lateral_gain": [0.50, 0.34],
      "centerline_heading_gain": [0.86, 0.58],
    },
  },
  "L4": {
    "mode": "lateral",
    "family": "weak_centerline_correction",
    "formal": True,
    "candidate_seed_prefix": 216,
    "direction": -1,
    "ranges": {
      "rise_profile_half_width": [0.0015, 0.0015],
      "tread_profile_half_width": [0.003, 0.003],
      "command_forward_scale": [1.00, 1.00],
      "command_delay_s": [0.14, 0.14],
      "command_low_pass_s": [0.12, 0.12],
      "action_gain": [0.995, 0.995],
      "action_bias_abs": [0.004, 0.004],
      "action_delay_steps": [0, 0],
      "encoder_bias_abs": [0.001, 0.001],
      "lateral_bias_abs": [0.030, 0.030],
      "yaw_bias_abs": [0.060, 0.060],
      "centerline_lateral_gain": [0.55, 0.20],
      "centerline_heading_gain": [0.95, 0.35],
      "centerline_max_lateral_velocity": [0.12, 0.07],
      "centerline_max_yaw_velocity": [0.32, 0.18],
      "lateral_pulse_min": [0.045, 0.045],
      "lateral_pulse_max": [0.105, 0.105],
      "yaw_pulse_min": [0.100, 0.100],
      "yaw_pulse_max": [0.240, 0.240],
    },
  },
  "L5": {
    "mode": "lateral",
    "family": "moderate_mixed_lateral_shift",
    "formal": True,
    "candidate_seed_prefix": 217,
    "direction": 1,
    "ranges": {
      "rise_profile_half_width": [0.0015, 0.0015],
      "tread_profile_half_width": [0.003, 0.003],
      "command_forward_scale": [1.00, 1.00],
      "command_delay_s": [0.08, 0.18],
      "command_low_pass_s": [0.06, 0.16],
      "action_gain": [0.998, 0.985],
      "action_bias_abs": [0.003, 0.008],
      "action_delay_steps": [0, 0],
      "encoder_bias_abs": [0.001, 0.004],
      "lateral_bias_abs": [0.015, 0.070],
      "yaw_bias_abs": [0.030, 0.140],
      "lateral_pulse_min": [0.025, 0.060],
      "lateral_pulse_max": [0.060, 0.150],
      "yaw_pulse_min": [0.060, 0.150],
      "yaw_pulse_max": [0.140, 0.340],
      "centerline_lateral_gain": [0.70, 0.45],
      "centerline_heading_gain": [1.20, 0.75],
      "centerline_max_lateral_velocity": [0.16, 0.10],
      "centerline_max_yaw_velocity": [0.42, 0.25],
    },
  },
  "C1": {
    "mode": "contact_stability",
    "family": "low_foot_friction",
    "formal": True,
    "candidate_seed_prefix": 218,
    "direction": -1,
    "ranges": {"foot_friction": [0.52, 0.25]},
  },
  "C2": {
    "mode": "contact_stability",
    "family": "left_right_action_asymmetry",
    "formal": True,
    "candidate_seed_prefix": 219,
    "direction": 1,
    "ranges": {"response_asymmetry_abs": [0.040, 0.200]},
  },
  "C3": {
    "mode": "contact_stability",
    "family": "action_gain_and_encoder_bias",
    "formal": True,
    "candidate_seed_prefix": 220,
    "direction": -1,
    "ranges": {
      "action_gain": [0.916, 0.900],
      "encoder_bias_abs": [0.0140, 0.0162],
    },
  },
  "C4": {
    "mode": "contact_stability",
    "family": "friction_and_command_dynamics_mismatch",
    "formal": True,
    "candidate_seed_prefix": 221,
    "direction": 1,
    "ranges": {
      "foot_friction": [0.55, 0.32],
      "command_forward_scale": [1.02, 1.20],
      "command_low_pass_s": [0.02, 0.16],
      "action_gain": [0.98, 0.82],
    },
  },
  "C5": {
    "mode": "contact_stability",
    "family": "moderate_mixed_contact_shift",
    "formal": True,
    "candidate_seed_prefix": 222,
    "direction": -1,
    "ranges": {
      "foot_friction": [0.54, 0.32],
      "contact_delay_steps": [1, 2],
      "phase_offset_abs": [0.02, 0.08],
      "response_asymmetry_abs": [0.010, 0.050],
      "action_gain": [1.0, 0.94],
      "action_bias_abs": [0.002, 0.025],
    },
  },
}
V21_DEVELOPMENT_CONTEXTS = ("L_dev", "C_dev")
V21_FORMAL_CONTEXTS = (
  "L1", "L2", "L3", "L4", "L5", "C1", "C2", "C3", "C4", "C5"
)
MEDIUM_TARGET_DOMAIN = "DQHMED"
MEDIUM_NEIGHBOR_DOMAIN = "DQNHMED"


def deployment_context_role_for_task(task: str) -> str | None:
  if task.endswith(f"-{MEDIUM_TARGET_DOMAIN}") or task == MEDIUM_TARGET_DOMAIN:
    return "target"
  if task.endswith(f"-{MEDIUM_NEIGHBOR_DOMAIN}") or task == MEDIUM_NEIGHBOR_DOMAIN:
    return "neighbor"
  return None


@dataclass(frozen=True)
class DeploymentContextParameters:
  """Environment-side parameters that are never appended to actor observations."""

  num_steps: int
  rise_profile: tuple[float, ...]
  tread_profile: tuple[float, ...]
  command_forward_scale: float
  command_delay_s: float
  command_low_pass_s: float
  action_gain: float
  action_bias: tuple[float, ...]
  action_delay_steps: int
  encoder_bias: float
  episode_length_s: float


@dataclass(frozen=True)
class SpecialistScenarioParameters:
  """Mode-specific environment and scalar-signal parameters frozen by hash."""

  disturbance_pulses_with_centering: bool
  lateral_command_bias: float
  yaw_command_bias: float
  lateral_pulse_min: float
  lateral_pulse_max: float
  yaw_pulse_min: float
  yaw_pulse_max: float
  pulse_interval_min_s: float
  pulse_interval_max_s: float
  pulse_duration_min_s: float
  pulse_duration_max_s: float
  centerline_lateral_gain: float
  centerline_heading_gain: float
  centerline_max_lateral_velocity: float
  centerline_max_yaw_velocity: float
  toe_margin: float
  foot_friction: float
  failure_signal_scale: float
  centerline_signal_weight: float
  heading_signal_weight: float
  edge_signal_weight: float
  intervention_signal_weight: float
  nominal_margin_signal_weight: float
  roll_signal_weight: float
  pitch_signal_weight: float
  angular_velocity_signal_weight: float
  slip_signal_weight: float
  contact_mismatch_signal_weight: float


@dataclass(frozen=True)
class V19ScenarioParameters:
  """Observable-failure scenario parameters frozen before v19 adaptation."""

  disturbance_pulses_with_centering: bool
  lateral_command_bias: float
  yaw_command_bias: float
  lateral_pulse_min: float
  lateral_pulse_max: float
  yaw_pulse_min: float
  yaw_pulse_max: float
  pulse_interval_min_s: float
  pulse_interval_max_s: float
  pulse_duration_min_s: float
  pulse_duration_max_s: float
  centerline_lateral_gain: float
  centerline_heading_gain: float
  centerline_max_lateral_velocity: float
  centerline_max_yaw_velocity: float
  toe_margin: float
  foot_friction: float
  contact_observation_delay_steps: int
  gait_phase_offset: float
  left_response_scale: float
  right_response_scale: float
  recovery_reward_scale: float
  edge_penalty_scale: float


def _canonical_json(value: Mapping[str, Any]) -> bytes:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def deployment_context_sha256(payload: Mapping[str, Any]) -> str:
  """Hash only the frozen parameters and schema, excluding calibration metrics."""
  if payload["kind"] == V21_CONTEXT_KIND:
    identity = {
      "kind": payload["kind"],
      "schema_version": payload["schema_version"],
      "calibration_candidate_seed": payload["calibration_candidate_seed"],
      "specialist_mode": payload["specialist_mode"],
      "context_id": payload["context_id"],
      "context_family": payload["context_family"],
      "formal_context": payload["formal_context"],
      "family_spec_sha256": payload["family_spec_sha256"],
      "target": payload["target"],
      "scenario": payload["scenario"],
    }
  elif payload["kind"] in (SPECIALIST_CONTEXT_KIND, V19_CONTEXT_KIND):
    identity = {
      "kind": payload["kind"],
      "schema_version": payload["schema_version"],
      "calibration_candidate_seed": payload["calibration_candidate_seed"],
      "specialist_mode": payload["specialist_mode"],
      "target": payload["target"],
      "scenario": payload["scenario"],
    }
  else:
    identity = {
      "kind": payload["kind"],
      "schema_version": payload["schema_version"],
      "calibration_candidate_seed": payload["calibration_candidate_seed"],
      "target": payload["target"],
      "neighbor": payload["neighbor"],
    }
  return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _validate_parameters(parameters: DeploymentContextParameters) -> None:
  if not 9 <= parameters.num_steps <= 24:
    raise ValueError("deployment context must contain 9--24 risers")
  if not (
    len(parameters.rise_profile)
    == len(parameters.tread_profile)
    == parameters.num_steps
  ):
    raise ValueError("deployment context geometry profiles do not match num_steps")
  finite = (
    *parameters.rise_profile,
    *parameters.tread_profile,
    parameters.command_forward_scale,
    parameters.command_delay_s,
    parameters.command_low_pass_s,
    parameters.action_gain,
    *parameters.action_bias,
    parameters.encoder_bias,
    parameters.episode_length_s,
  )
  if not all(math.isfinite(value) for value in finite):
    raise ValueError("deployment context contains a non-finite parameter")
  if not all(0.115 <= value <= 0.165 for value in parameters.rise_profile):
    raise ValueError("deployment rise profile is outside the supported range")
  if not all(0.28 <= value <= 0.39 for value in parameters.tread_profile):
    raise ValueError("deployment tread profile is outside the supported range")
  if not 0.8 <= parameters.command_forward_scale <= 1.2:
    raise ValueError("deployment command scale is outside [0.8, 1.2]")
  if not 0.0 <= parameters.command_delay_s <= 0.4:
    raise ValueError("deployment command delay is outside [0, 0.4] seconds")
  if not 0.0 <= parameters.command_low_pass_s <= 0.4:
    raise ValueError("deployment command filter is outside [0, 0.4] seconds")
  if not 0.8 <= parameters.action_gain <= 1.2:
    raise ValueError("deployment action gain is outside [0.8, 1.2]")
  if len(parameters.action_bias) != 12:
    raise ValueError("deployment action bias must match the 12-D actor action")
  if not all(abs(value) <= 0.08 for value in parameters.action_bias):
    raise ValueError("deployment action bias exceeds 0.08 raw-action units")
  if not 0 <= parameters.action_delay_steps <= 5:
    raise ValueError("deployment action delay is outside [0, 5] steps")
  if abs(parameters.encoder_bias) > 0.03:
    raise ValueError("deployment encoder bias exceeds 0.03 radians")
  if parameters.episode_length_s < 20.0:
    raise ValueError("deployment episode is too short for the staircase")


def _validate_specialist_scenario(
  mode: str, parameters: SpecialistScenarioParameters
) -> None:
  if mode not in SPECIALIST_MODES:
    raise ValueError(f"unsupported specialist mode: {mode!r}")
  finite = tuple(
    float(value)
    for name, value in asdict(parameters).items()
    if name != "disturbance_pulses_with_centering"
  )
  if not all(math.isfinite(value) for value in finite):
    raise ValueError("specialist scenario contains a non-finite parameter")
  if not -0.20 <= parameters.lateral_command_bias <= 0.20:
    raise ValueError("specialist lateral command bias is outside [-0.2, 0.2]")
  if not -0.50 <= parameters.yaw_command_bias <= 0.50:
    raise ValueError("specialist yaw command bias is outside [-0.5, 0.5]")
  ordered_nonnegative = (
    (parameters.lateral_pulse_min, parameters.lateral_pulse_max),
    (parameters.yaw_pulse_min, parameters.yaw_pulse_max),
    (parameters.pulse_interval_min_s, parameters.pulse_interval_max_s),
    (parameters.pulse_duration_min_s, parameters.pulse_duration_max_s),
  )
  if any(low < 0.0 or high < low for low, high in ordered_nonnegative):
    raise ValueError("specialist pulse ranges must be non-negative and ordered")
  if not 0.0 <= parameters.centerline_lateral_gain <= 2.0:
    raise ValueError("specialist lateral feedback gain is outside [0, 2]")
  if not 0.0 <= parameters.centerline_heading_gain <= 3.0:
    raise ValueError("specialist heading feedback gain is outside [0, 3]")
  if not 0.0 < parameters.centerline_max_lateral_velocity <= 0.5:
    raise ValueError("specialist lateral correction bound is invalid")
  if not 0.0 < parameters.centerline_max_yaw_velocity <= 1.0:
    raise ValueError("specialist yaw correction bound is invalid")
  if not 0.04 <= parameters.toe_margin <= 0.18:
    raise ValueError("specialist toe margin is outside [0.04, 0.18]")
  if not 0.25 <= parameters.foot_friction <= 1.20:
    raise ValueError("specialist foot friction is outside [0.25, 1.20]")
  if not 0.0 < parameters.failure_signal_scale <= 2.0:
    raise ValueError("specialist failure-signal scale is outside (0, 2]")
  signal_weights = (
    parameters.centerline_signal_weight,
    parameters.heading_signal_weight,
    parameters.edge_signal_weight,
    parameters.intervention_signal_weight,
    parameters.nominal_margin_signal_weight,
    parameters.roll_signal_weight,
    parameters.pitch_signal_weight,
    parameters.angular_velocity_signal_weight,
    parameters.slip_signal_weight,
    parameters.contact_mismatch_signal_weight,
  )
  if any(weight < 0.0 for weight in signal_weights):
    raise ValueError("specialist failure-signal weights must be non-negative")
  if not math.isclose(sum(signal_weights), 1.0, abs_tol=1.0e-9):
    raise ValueError("specialist failure-signal weights must sum to one")
  active_by_mode = {
    "lateral": signal_weights[:3],
    "cbf": signal_weights[3:5],
    "balance": signal_weights[5:],
  }
  if not math.isclose(sum(active_by_mode[mode]), 1.0, abs_tol=1.0e-9):
    raise ValueError("specialist signal has weight outside its declared mode")


def _validate_v19_scenario(
  mode: str, parameters: V19ScenarioParameters, *, v21: bool = False
) -> None:
  if mode not in V19_SPECIALIST_MODES:
    raise ValueError(f"unsupported v19 specialist mode: {mode!r}")
  finite = tuple(
    float(value)
    for name, value in asdict(parameters).items()
    if name not in (
      "disturbance_pulses_with_centering",
      "contact_observation_delay_steps",
    )
  )
  if not all(math.isfinite(value) for value in finite):
    raise ValueError("v19 scenario contains a non-finite parameter")
  if not -0.20 <= parameters.lateral_command_bias <= 0.20:
    raise ValueError("v19 lateral command bias is outside [-0.2, 0.2]")
  if not -0.50 <= parameters.yaw_command_bias <= 0.50:
    raise ValueError("v19 yaw command bias is outside [-0.5, 0.5]")
  for low, high in (
    (parameters.lateral_pulse_min, parameters.lateral_pulse_max),
    (parameters.yaw_pulse_min, parameters.yaw_pulse_max),
    (parameters.pulse_interval_min_s, parameters.pulse_interval_max_s),
    (parameters.pulse_duration_min_s, parameters.pulse_duration_max_s),
  ):
    if low < 0.0 or high < low:
      raise ValueError("v19 pulse ranges must be non-negative and ordered")
  if not 0.0 <= parameters.centerline_lateral_gain <= 2.0:
    raise ValueError("v19 centerline lateral gain is outside [0, 2]")
  if not 0.0 <= parameters.centerline_heading_gain <= 3.0:
    raise ValueError("v19 centerline heading gain is outside [0, 3]")
  if not 0.0 < parameters.centerline_max_lateral_velocity <= 0.5:
    raise ValueError("v19 lateral command bound is invalid")
  if not 0.0 < parameters.centerline_max_yaw_velocity <= 1.0:
    raise ValueError("v19 yaw command bound is invalid")
  if not 0.04 <= parameters.toe_margin <= 0.18:
    raise ValueError("v19 toe margin is outside [0.04, 0.18]")
  minimum_friction = 0.25 if v21 else 0.35
  if not minimum_friction <= parameters.foot_friction <= 1.20:
    raise ValueError(
      f"observable-context friction must stay in [{minimum_friction}, 1.20]"
    )
  if parameters.contact_observation_delay_steps not in (0, 1, 2):
    raise ValueError("v19 contact sensing delay must be 0--2 control steps")
  if abs(parameters.gait_phase_offset) > 0.08:
    raise ValueError("v19 gait/contact phase offset exceeds 0.08 cycles")
  maximum_response_shift = 0.20 if v21 else 0.04
  if not all(
    1.0 - maximum_response_shift <= value <= 1.0 + maximum_response_shift
    for value in (parameters.left_response_scale, parameters.right_response_scale)
  ):
    raise ValueError(
      "observable-context response scale exceeds its versioned bound"
    )
  if abs(parameters.left_response_scale - parameters.right_response_scale) > (
    2.0 * maximum_response_shift
  ):
    raise ValueError("observable-context response asymmetry is too large")
  if not 0.0 < parameters.recovery_reward_scale <= 1.0:
    raise ValueError("v19 recovery reward scale is invalid")
  if not 0.0 <= parameters.edge_penalty_scale <= 1.0:
    raise ValueError("v19 edge penalty scale is invalid")
  if mode == "lateral":
    if (
      parameters.contact_observation_delay_steps != 0
      or parameters.gait_phase_offset != 0.0
      or parameters.left_response_scale != 1.0
      or parameters.right_response_scale != 1.0
    ):
      raise ValueError("v19 lateral context contains contact-only perturbations")
  else:
    forbidden = (
      parameters.disturbance_pulses_with_centering
      or parameters.lateral_command_bias != 0.0
      or parameters.yaw_command_bias != 0.0
      or parameters.lateral_pulse_max != 0.0
      or parameters.yaw_pulse_max != 0.0
    )
    if forbidden:
      raise ValueError("v19 contact context contains a lateral/yaw disturbance")


def _rounded_profile(
  rng: random.Random,
  *,
  count: int,
  center: float,
  half_width: float,
) -> tuple[float, ...]:
  values = [center + rng.uniform(-half_width, half_width) for _ in range(count)]
  mean_error = sum(values) / count - center
  return tuple(round(value - mean_error, 6) for value in values)


def generate_failure_focused_context(candidate_seed: int) -> dict[str, Any]:
  """Generate one context without consulting any adapted-policy outcome."""
  rng = random.Random(candidate_seed)
  num_steps = rng.choice((10, 11, 12))
  rise_center = rng.uniform(0.134, 0.142)
  tread_center = rng.uniform(0.322, 0.346)
  action_bias = tuple(round(rng.uniform(-0.025, 0.025), 6) for _ in range(12))
  target = DeploymentContextParameters(
    num_steps=num_steps,
    rise_profile=_rounded_profile(
      rng, count=num_steps, center=rise_center, half_width=0.0045
    ),
    tread_profile=_rounded_profile(
      rng, count=num_steps, center=tread_center, half_width=0.009
    ),
    command_forward_scale=round(rng.uniform(0.94, 1.08), 6),
    command_delay_s=round(rng.uniform(0.10, 0.22), 6),
    command_low_pass_s=round(rng.uniform(0.09, 0.18), 6),
    action_gain=round(rng.uniform(0.94, 1.02), 6),
    action_bias=action_bias,
    action_delay_steps=rng.choice((0, 1, 2)),
    encoder_bias=round(rng.uniform(-0.012, 0.012), 6),
    episode_length_s=max(40.0, 3.4 * num_steps),
  )
  neighbor = replace(
    target,
    rise_profile=tuple(round(value + 0.002, 6) for value in target.rise_profile),
    tread_profile=tuple(
      round(value + 0.005, 6) for value in target.tread_profile
    ),
    command_delay_s=round(min(0.4, target.command_delay_s + 0.02), 6),
    command_low_pass_s=round(
      min(0.4, target.command_low_pass_s + 0.015), 6
    ),
    action_gain=round(max(0.8, target.action_gain - 0.01), 6),
    encoder_bias=round(max(-0.03, min(0.03, target.encoder_bias + 0.001)), 6),
  )
  _validate_parameters(target)
  _validate_parameters(neighbor)
  payload: dict[str, Any] = {
    "kind": FAILURE_FOCUSED_CONTEXT_KIND,
    "schema_version": FAILURE_FOCUSED_CONTEXT_SCHEMA_VERSION,
    "calibration_candidate_seed": int(candidate_seed),
    "target": asdict(target),
    "neighbor": asdict(neighbor),
  }
  payload["parameters_sha256"] = deployment_context_sha256(payload)
  return payload


def generate_specialist_context(mode: str, candidate_seed: int) -> dict[str, Any]:
  """Generate an ordered-difficulty specialist context without adapted outcomes."""
  if mode not in SPECIALIST_MODES:
    raise ValueError(f"unsupported specialist mode: {mode!r}")
  rng = random.Random(candidate_seed)
  difficulty = (candidate_seed % 100) / 19.0
  if not 0.0 <= difficulty <= 1.0:
    raise ValueError(
      "formal specialist candidate seeds must end in an index from 00 through 19"
    )
  num_steps = 11
  if mode == "cbf":
    # Keep whole-body response close to pi0 and sweep toe/riser demand
    # smoothly. A second discrete action-delay step created an all-fall cliff
    # in base-only exploration and did not isolate the intended CBF mode.
    rise_center = 0.140 + 0.014 * difficulty
    tread_center = 0.342 - 0.035 * difficulty
    forward_scale = 0.98 + 0.10 * difficulty
    action_gain = 1.0 - 0.035 * difficulty
    action_delay_steps = 1
    action_bias_width = 0.006 + 0.006 * difficulty
    command_delay_s = 0.07 + 0.04 * difficulty
    command_low_pass_s = 0.06 + 0.03 * difficulty
  elif mode == "balance":
    # Make toe/riser clearance deliberately easier than the CBF specialist,
    # then sweep contact friction and a smooth sagittal plant residual.  This
    # keeps falls attributable to balance/contact recovery instead of letting
    # the maximum-CBF classifier dominate on otherwise identical stairs.
    rise_center = 0.130 + 0.002 * difficulty
    tread_center = 0.360 - 0.005 * difficulty
    forward_scale = 0.94 + 0.025 * difficulty
    action_gain = 1.0 - 0.008 * difficulty
    action_delay_steps = 0
    action_bias_width = 0.008 + 0.012 * difficulty
    command_delay_s = 0.05 + 0.035 * difficulty
    command_low_pass_s = 0.04 + 0.025 * difficulty
  else:
    rise_center = 0.138
    tread_center = 0.338
    forward_scale = 0.99 + 0.04 * difficulty
    action_gain = 0.99
    action_delay_steps = 1
    action_bias_width = 0.012
    command_delay_s = 0.14 + 0.25 * difficulty
    command_low_pass_s = 0.12 + 0.24 * difficulty
  if mode == "balance":
    # Symmetric sagittal residuals stress pitch/contact recovery without
    # directly steering either foot toward a stair edge. Candidate parity
    # covers forward- and backward-biased plant response.
    direction = -1.0 if candidate_seed % 2 else 1.0
    amplitude = direction * (0.011 + 0.031 * difficulty)
    action_bias = tuple(
      round(value, 6)
      for value in (
        amplitude,
        0.0,
        0.0,
        -amplitude,
        0.5 * amplitude,
        0.0,
        amplitude,
        0.0,
        0.0,
        -amplitude,
        0.5 * amplitude,
        0.0,
      )
    )
  else:
    action_bias = tuple(
      round(rng.uniform(-action_bias_width, action_bias_width), 6)
      for _ in range(12)
    )
  target = DeploymentContextParameters(
    num_steps=num_steps,
    rise_profile=_rounded_profile(
      rng,
      count=num_steps,
      center=rise_center,
      half_width=0.0035 if mode != "cbf" else 0.0045,
    ),
    tread_profile=_rounded_profile(
      rng,
      count=num_steps,
      center=tread_center,
      half_width=0.007 if mode != "cbf" else 0.009,
    ),
    command_forward_scale=round(forward_scale, 6),
    command_delay_s=round(command_delay_s, 6),
    command_low_pass_s=round(command_low_pass_s, 6),
    action_gain=round(action_gain, 6),
    action_bias=action_bias,
    action_delay_steps=action_delay_steps,
    encoder_bias=round(rng.uniform(-0.008, 0.008), 6),
    episode_length_s=40.0,
  )
  zero_weights = {
    "centerline_signal_weight": 0.0,
    "heading_signal_weight": 0.0,
    "edge_signal_weight": 0.0,
    "intervention_signal_weight": 0.0,
    "nominal_margin_signal_weight": 0.0,
    "roll_signal_weight": 0.0,
    "pitch_signal_weight": 0.0,
    "angular_velocity_signal_weight": 0.0,
    "slip_signal_weight": 0.0,
    "contact_mismatch_signal_weight": 0.0,
  }
  if mode == "lateral":
    zero_weights.update(
      centerline_signal_weight=0.45,
      heading_signal_weight=0.35,
      edge_signal_weight=0.20,
    )
    direction = -1.0 if candidate_seed % 2 else 1.0
    scenario = SpecialistScenarioParameters(
      disturbance_pulses_with_centering=True,
      lateral_command_bias=round(direction * 0.080 * difficulty, 6),
      yaw_command_bias=round(-direction * 0.160 * difficulty, 6),
      lateral_pulse_min=round(0.035 + 0.050 * difficulty, 6),
      lateral_pulse_max=round(0.075 + 0.105 * difficulty, 6),
      yaw_pulse_min=round(0.080 + 0.115 * difficulty, 6),
      yaw_pulse_max=round(0.190 + 0.250 * difficulty, 6),
      pulse_interval_min_s=round(1.8 - 0.5 * difficulty, 6),
      pulse_interval_max_s=round(3.6 - 0.8 * difficulty, 6),
      pulse_duration_min_s=round(0.25 + 0.10 * difficulty, 6),
      pulse_duration_max_s=round(0.55 + 0.20 * difficulty, 6),
      centerline_lateral_gain=round(0.65 - 0.45 * difficulty, 6),
      centerline_heading_gain=round(1.10 - 0.70 * difficulty, 6),
      centerline_max_lateral_velocity=round(0.13 - 0.06 * difficulty, 6),
      centerline_max_yaw_velocity=round(0.34 - 0.12 * difficulty, 6),
      toe_margin=0.075,
      foot_friction=0.65,
      failure_signal_scale=0.50,
      **zero_weights,
    )
  elif mode == "cbf":
    zero_weights.update(
      intervention_signal_weight=0.60,
      nominal_margin_signal_weight=0.40,
    )
    scenario = SpecialistScenarioParameters(
      disturbance_pulses_with_centering=False,
      lateral_command_bias=0.0,
      yaw_command_bias=0.0,
      lateral_pulse_min=0.0,
      lateral_pulse_max=0.0,
      yaw_pulse_min=0.0,
      yaw_pulse_max=0.0,
      pulse_interval_min_s=3.0,
      pulse_interval_max_s=7.0,
      pulse_duration_min_s=0.2,
      pulse_duration_max_s=0.6,
      centerline_lateral_gain=1.60,
      centerline_heading_gain=2.60,
      centerline_max_lateral_velocity=0.30,
      centerline_max_yaw_velocity=0.80,
      toe_margin=round(0.085 + 0.070 * difficulty, 6),
      foot_friction=0.68,
      failure_signal_scale=0.35,
      **zero_weights,
    )
  else:
    zero_weights.update(
      roll_signal_weight=0.20,
      pitch_signal_weight=0.20,
      angular_velocity_signal_weight=0.20,
      slip_signal_weight=0.20,
      contact_mismatch_signal_weight=0.20,
    )
    scenario = SpecialistScenarioParameters(
      disturbance_pulses_with_centering=False,
      lateral_command_bias=0.0,
      yaw_command_bias=0.0,
      lateral_pulse_min=0.0,
      lateral_pulse_max=0.0,
      yaw_pulse_min=0.0,
      yaw_pulse_max=0.0,
      pulse_interval_min_s=3.0,
      pulse_interval_max_s=7.0,
      pulse_duration_min_s=0.2,
      pulse_duration_max_s=0.6,
      centerline_lateral_gain=2.00,
      centerline_heading_gain=3.00,
      centerline_max_lateral_velocity=0.50,
      centerline_max_yaw_velocity=1.00,
      toe_margin=0.040,
      foot_friction=round(0.52 - 0.26 * difficulty, 6),
      failure_signal_scale=0.35,
      **zero_weights,
    )
  _validate_parameters(target)
  _validate_specialist_scenario(mode, scenario)
  payload: dict[str, Any] = {
    "kind": SPECIALIST_CONTEXT_KIND,
    "schema_version": SPECIALIST_CONTEXT_SCHEMA_VERSION,
    "calibration_candidate_seed": int(candidate_seed),
    "specialist_mode": mode,
    "target": asdict(target),
    "scenario": asdict(scenario),
  }
  payload["parameters_sha256"] = deployment_context_sha256(payload)
  return payload


def generate_v19_specialist_context(
  mode: str, candidate_seed: int
) -> dict[str, Any]:
  """Generate a new-seed lateral or mechanism-pure contact-stability context."""
  if mode not in V19_SPECIALIST_MODES:
    raise ValueError(f"unsupported v19 specialist mode: {mode!r}")
  rng = random.Random(candidate_seed)
  difficulty = (candidate_seed % 100) / 19.0
  if not 0.0 <= difficulty <= 1.0:
    raise ValueError(
      "formal v19 candidate seeds must end in an index from 00 through 19"
    )
  num_steps = 11
  if mode == "lateral":
    rise_profile = _rounded_profile(
      rng, count=num_steps, center=0.138, half_width=0.0035
    )
    tread_profile = _rounded_profile(
      rng, count=num_steps, center=0.338, half_width=0.007
    )
    action_bias = tuple(round(rng.uniform(-0.012, 0.012), 6) for _ in range(12))
    target = DeploymentContextParameters(
      num_steps=num_steps,
      rise_profile=rise_profile,
      tread_profile=tread_profile,
      command_forward_scale=round(0.99 + 0.04 * difficulty, 6),
      command_delay_s=round(0.14 + 0.25 * difficulty, 6),
      command_low_pass_s=round(0.12 + 0.24 * difficulty, 6),
      action_gain=0.99,
      action_bias=action_bias,
      action_delay_steps=1,
      encoder_bias=round(rng.uniform(-0.008, 0.008), 6),
      episode_length_s=40.0,
    )
    direction = -1.0 if candidate_seed % 2 else 1.0
    scenario = V19ScenarioParameters(
      disturbance_pulses_with_centering=True,
      lateral_command_bias=round(direction * 0.080 * difficulty, 6),
      yaw_command_bias=round(-direction * 0.160 * difficulty, 6),
      lateral_pulse_min=round(0.035 + 0.050 * difficulty, 6),
      lateral_pulse_max=round(0.075 + 0.105 * difficulty, 6),
      yaw_pulse_min=round(0.080 + 0.115 * difficulty, 6),
      yaw_pulse_max=round(0.190 + 0.250 * difficulty, 6),
      pulse_interval_min_s=round(1.8 - 0.5 * difficulty, 6),
      pulse_interval_max_s=round(3.6 - 0.8 * difficulty, 6),
      pulse_duration_min_s=round(0.25 + 0.10 * difficulty, 6),
      pulse_duration_max_s=round(0.55 + 0.20 * difficulty, 6),
      centerline_lateral_gain=round(0.65 - 0.45 * difficulty, 6),
      centerline_heading_gain=round(1.10 - 0.70 * difficulty, 6),
      centerline_max_lateral_velocity=round(0.13 - 0.06 * difficulty, 6),
      centerline_max_yaw_velocity=round(0.34 - 0.12 * difficulty, 6),
      toe_margin=0.075,
      foot_friction=0.65,
      contact_observation_delay_steps=0,
      gait_phase_offset=0.0,
      left_response_scale=1.0,
      right_response_scale=1.0,
      recovery_reward_scale=0.50,
      edge_penalty_scale=0.20,
    )
  else:
    # No geometry, encoder, action-bias, action-delay, or command-latency
    # mixture is introduced here.  The only deployment shifts are moderate
    # friction, a short contact-sensor lag, a small phase offset, and a small
    # left/right response asymmetry.
    # Use a longer sequence of individually moderate steps to accumulate
    # contact-event exposure.  This reaches the calibration fall count without
    # resorting to very low friction or near-target hard geometry.
    num_steps = 24
    target = DeploymentContextParameters(
      num_steps=num_steps,
      rise_profile=(0.138,) * num_steps,
      tread_profile=(0.355,) * num_steps,
      command_forward_scale=1.0,
      command_delay_s=0.0,
      command_low_pass_s=0.0,
      action_gain=1.0,
      action_bias=(0.0,) * 12,
      action_delay_steps=0,
      encoder_bias=0.0,
      episode_length_s=40.0,
    )
    direction = -1.0 if candidate_seed % 2 else 1.0
    asymmetry = 0.008 + 0.031 * difficulty
    scenario = V19ScenarioParameters(
      disturbance_pulses_with_centering=False,
      lateral_command_bias=0.0,
      yaw_command_bias=0.0,
      lateral_pulse_min=0.0,
      lateral_pulse_max=0.0,
      yaw_pulse_min=0.0,
      yaw_pulse_max=0.0,
      pulse_interval_min_s=3.0,
      pulse_interval_max_s=7.0,
      pulse_duration_min_s=0.2,
      pulse_duration_max_s=0.6,
      centerline_lateral_gain=2.0,
      centerline_heading_gain=3.0,
      centerline_max_lateral_velocity=0.50,
      centerline_max_yaw_velocity=1.0,
      toe_margin=0.060,
      foot_friction=round(0.56 - 0.21 * difficulty, 6),
      contact_observation_delay_steps=1 if difficulty < 0.55 else 2,
      gait_phase_offset=round(direction * (0.020 + 0.060 * difficulty), 6),
      left_response_scale=round(1.0 + direction * asymmetry, 6),
      right_response_scale=round(1.0 - direction * asymmetry, 6),
      recovery_reward_scale=0.40,
      edge_penalty_scale=0.0,
    )
  _validate_parameters(target)
  _validate_v19_scenario(mode, scenario)
  payload: dict[str, Any] = {
    "kind": V19_CONTEXT_KIND,
    "schema_version": V19_CONTEXT_SCHEMA_VERSION,
    "calibration_candidate_seed": int(candidate_seed),
    "specialist_mode": mode,
    "target": asdict(target),
    "scenario": asdict(scenario),
  }
  payload["parameters_sha256"] = deployment_context_sha256(payload)
  return payload


def _v21_range_value(
  ranges: Mapping[str, Any], key: str, severity: float, default: float
) -> float:
  values = ranges.get(key)
  if values is None:
    return float(default)
  if not isinstance(values, list) or len(values) != 2:
    raise ValueError(f"v21 family range {key!r} must contain two endpoints")
  return float(values[0]) + severity * (float(values[1]) - float(values[0]))


def generate_v21_specialist_context(
  context_id: str, candidate_seed: int
) -> dict[str, Any]:
  """Generate one prospectively ranged v21 deployment-family candidate."""
  if context_id not in V21_CONTEXT_SPECS:
    raise ValueError(f"unsupported v21 context ID: {context_id!r}")
  spec = V21_CONTEXT_SPECS[context_id]
  candidate_index = candidate_seed % 100
  expected_prefix = int(spec["candidate_seed_prefix"])
  if candidate_seed // 100 != expected_prefix or not 8 <= candidate_index <= 19:
    raise ValueError(
      f"v21 {context_id} candidate seeds must be "
      f"{expected_prefix * 100 + 8}--{expected_prefix * 100 + 19}"
    )
  severity = (candidate_index - 8) / 11.0
  # Hold direction and the normalized action-bias pattern fixed within a
  # family. Candidate order then sweeps one severity axis instead of changing
  # the qualitative perturbation between calibration attempts.
  rng = random.Random(expected_prefix)
  ranges = spec["ranges"]
  mode = str(spec["mode"])
  direction = float(spec["direction"])
  if direction not in (-1.0, 1.0):
    raise ValueError(f"v21 {context_id} direction must be -1 or +1")

  if mode == "lateral":
    num_steps = 11
    tread = 0.338
    rise_center = _v21_range_value(ranges, "rise_center", severity, 0.138)
    tread_center = _v21_range_value(ranges, "tread_center", severity, tread)
    rise_half_width = _v21_range_value(
      ranges, "rise_profile_half_width", severity, 0.0
    )
    tread_half_width = _v21_range_value(
      ranges, "tread_profile_half_width", severity, 0.0
    )
    rise_profile = _rounded_profile(
      rng,
      count=num_steps,
      center=rise_center,
      half_width=rise_half_width,
    )
    tread_profile = _rounded_profile(
      rng,
      count=num_steps,
      center=tread_center,
      half_width=tread_half_width,
    )
    command_forward_scale = _v21_range_value(
      ranges, "command_forward_scale", severity, 1.0
    )
    command_delay = _v21_range_value(
      ranges, "command_delay_s", severity, 0.0
    )
    command_low_pass = _v21_range_value(
      ranges, "command_low_pass_s", severity, 0.0
    )
    action_gain = _v21_range_value(ranges, "action_gain", severity, 1.0)
    action_delay = int(
      round(
        _v21_range_value(ranges, "action_delay_steps", severity, 0.0)
      )
    )
    encoder_bias = direction * _v21_range_value(
      ranges, "encoder_bias_abs", severity, 0.0
    )
    action_bias_width = _v21_range_value(
      ranges, "action_bias_abs", severity, 0.0
    )
    action_bias = tuple(
      round(rng.uniform(-action_bias_width, action_bias_width), 6)
      for _ in range(12)
    )
    lateral_bias = direction * _v21_range_value(
      ranges, "lateral_bias_abs", severity, 0.0
    )
    yaw_bias = -direction * _v21_range_value(
      ranges, "yaw_bias_abs", severity, 0.0
    )
    lateral_pulse_max = _v21_range_value(
      ranges, "lateral_pulse_max", severity, 0.0
    )
    lateral_pulse_min = _v21_range_value(
      ranges,
      "lateral_pulse_min",
      severity,
      0.45 * lateral_pulse_max,
    )
    yaw_pulse_max = _v21_range_value(
      ranges, "yaw_pulse_max", severity, 0.0
    )
    yaw_pulse_min = _v21_range_value(
      ranges, "yaw_pulse_min", severity, 0.42 * yaw_pulse_max
    )
    pulses = lateral_pulse_max > 0.0 or yaw_pulse_max > 0.0
    scenario = V19ScenarioParameters(
      disturbance_pulses_with_centering=pulses,
      lateral_command_bias=round(lateral_bias, 6),
      yaw_command_bias=round(yaw_bias, 6),
      lateral_pulse_min=round(lateral_pulse_min, 6),
      lateral_pulse_max=round(lateral_pulse_max, 6),
      yaw_pulse_min=round(yaw_pulse_min, 6),
      yaw_pulse_max=round(yaw_pulse_max, 6),
      pulse_interval_min_s=round(1.8 - 0.5 * severity, 6),
      pulse_interval_max_s=round(3.6 - 0.8 * severity, 6),
      pulse_duration_min_s=round(0.25 + 0.10 * severity, 6),
      pulse_duration_max_s=round(0.55 + 0.20 * severity, 6),
      centerline_lateral_gain=round(
        _v21_range_value(
          ranges, "centerline_lateral_gain", severity, 0.80
        ),
        6,
      ),
      centerline_heading_gain=round(
        _v21_range_value(
          ranges, "centerline_heading_gain", severity, 1.40
        ),
        6,
      ),
      centerline_max_lateral_velocity=round(
        _v21_range_value(
          ranges, "centerline_max_lateral_velocity", severity, 0.16
        ),
        6,
      ),
      centerline_max_yaw_velocity=round(
        _v21_range_value(
          ranges, "centerline_max_yaw_velocity", severity, 0.45
        ),
        6,
      ),
      toe_margin=0.075,
      foot_friction=0.65,
      contact_observation_delay_steps=0,
      gait_phase_offset=0.0,
      left_response_scale=1.0,
      right_response_scale=1.0,
      recovery_reward_scale=0.50,
      edge_penalty_scale=0.20,
    )
  else:
    num_steps = 24
    tread = 0.355
    rise_center = _v21_range_value(ranges, "rise_center", severity, 0.138)
    tread_center = _v21_range_value(ranges, "tread_center", severity, tread)
    rise_half_width = _v21_range_value(
      ranges, "rise_profile_half_width", severity, 0.0
    )
    tread_half_width = _v21_range_value(
      ranges, "tread_profile_half_width", severity, 0.0
    )
    rise_profile = _rounded_profile(
      rng,
      count=num_steps,
      center=rise_center,
      half_width=rise_half_width,
    )
    tread_profile = _rounded_profile(
      rng,
      count=num_steps,
      center=tread_center,
      half_width=tread_half_width,
    )
    command_delay = _v21_range_value(
      ranges, "command_delay_s", severity, 0.0
    )
    command_low_pass = _v21_range_value(
      ranges, "command_low_pass_s", severity, 0.0
    )
    command_forward_scale = _v21_range_value(
      ranges, "command_forward_scale", severity, 1.0
    )
    action_gain = _v21_range_value(ranges, "action_gain", severity, 1.0)
    action_delay = int(
      round(
        _v21_range_value(ranges, "action_delay_steps", severity, 0.0)
      )
    )
    encoder_bias = direction * _v21_range_value(
      ranges, "encoder_bias_abs", severity, 0.0
    )
    action_bias_width = _v21_range_value(
      ranges, "action_bias_abs", severity, 0.0
    )
    action_bias = tuple(
      round(rng.uniform(-action_bias_width, action_bias_width), 6)
      for _ in range(12)
    )
    asymmetry = _v21_range_value(
      ranges, "response_asymmetry_abs", severity, 0.0
    )
    phase_offset = direction * _v21_range_value(
      ranges, "phase_offset_abs", severity, 0.0
    )
    delay_value = _v21_range_value(
      ranges, "contact_delay_steps", severity, 0.0
    )
    contact_delay = int(round(delay_value))
    scenario = V19ScenarioParameters(
      disturbance_pulses_with_centering=False,
      lateral_command_bias=0.0,
      yaw_command_bias=0.0,
      lateral_pulse_min=0.0,
      lateral_pulse_max=0.0,
      yaw_pulse_min=0.0,
      yaw_pulse_max=0.0,
      pulse_interval_min_s=3.0,
      pulse_interval_max_s=7.0,
      pulse_duration_min_s=0.2,
      pulse_duration_max_s=0.6,
      centerline_lateral_gain=2.0,
      centerline_heading_gain=3.0,
      centerline_max_lateral_velocity=0.50,
      centerline_max_yaw_velocity=1.0,
      toe_margin=0.060,
      foot_friction=round(
        _v21_range_value(ranges, "foot_friction", severity, 0.65), 6
      ),
      contact_observation_delay_steps=contact_delay,
      gait_phase_offset=round(phase_offset, 6),
      left_response_scale=round(1.0 + direction * asymmetry, 6),
      right_response_scale=round(1.0 - direction * asymmetry, 6),
      recovery_reward_scale=0.40,
      edge_penalty_scale=0.0,
    )

  target = DeploymentContextParameters(
    num_steps=num_steps,
    rise_profile=rise_profile,
    tread_profile=tread_profile,
    command_forward_scale=round(command_forward_scale, 6),
    command_delay_s=round(command_delay, 6),
    command_low_pass_s=round(command_low_pass, 6),
    action_gain=round(action_gain, 6),
    action_bias=action_bias,
    action_delay_steps=action_delay,
    encoder_bias=round(encoder_bias, 6),
    episode_length_s=40.0,
  )
  _validate_parameters(target)
  _validate_v19_scenario(mode, scenario, v21=True)
  family_spec_sha256 = hashlib.sha256(_canonical_json(spec)).hexdigest()
  payload: dict[str, Any] = {
    "kind": V21_CONTEXT_KIND,
    "schema_version": V21_CONTEXT_SCHEMA_VERSION,
    "calibration_candidate_seed": int(candidate_seed),
    "specialist_mode": mode,
    "context_id": context_id,
    "context_family": spec["family"],
    "formal_context": bool(spec["formal"]),
    "family_spec_sha256": family_spec_sha256,
    "candidate_severity": round(severity, 12),
    "target": asdict(target),
    "scenario": asdict(scenario),
  }
  payload["parameters_sha256"] = deployment_context_sha256(payload)
  return payload


def _validated_deployment_parameters(raw: Mapping[str, Any]) -> dict[str, Any]:
  parameters = DeploymentContextParameters(
    num_steps=int(raw["num_steps"]),
    rise_profile=tuple(float(value) for value in raw["rise_profile"]),
    tread_profile=tuple(float(value) for value in raw["tread_profile"]),
    command_forward_scale=float(raw["command_forward_scale"]),
    command_delay_s=float(raw["command_delay_s"]),
    command_low_pass_s=float(raw["command_low_pass_s"]),
    action_gain=float(raw["action_gain"]),
    action_bias=tuple(float(value) for value in raw["action_bias"]),
    action_delay_steps=int(raw["action_delay_steps"]),
    encoder_bias=float(raw["encoder_bias"]),
    episode_length_s=float(raw["episode_length_s"]),
  )
  _validate_parameters(parameters)
  return asdict(parameters)


def _validate_frozen_specialist_context(
  payload: Mapping[str, Any],
) -> dict[str, Any]:
  if payload.get("schema_version") != SPECIALIST_CONTEXT_SCHEMA_VERSION:
    raise ValueError("unexpected specialist context schema version")
  if not isinstance(payload.get("calibration_candidate_seed"), int):
    raise ValueError("specialist context candidate seed is missing")
  mode = str(payload.get("specialist_mode"))
  if mode not in SPECIALIST_MODES:
    raise ValueError("specialist context mode is missing or invalid")
  target = payload.get("target")
  scenario_raw = payload.get("scenario")
  if not isinstance(target, Mapping) or not isinstance(scenario_raw, Mapping):
    raise ValueError("specialist context target or scenario is missing")
  scenario = SpecialistScenarioParameters(
    **{
      field: (
        bool(scenario_raw[field])
        if field == "disturbance_pulses_with_centering"
        else float(scenario_raw[field])
      )
      for field in SpecialistScenarioParameters.__dataclass_fields__
    }
  )
  _validate_specialist_scenario(mode, scenario)
  output = dict(payload)
  output["target"] = _validated_deployment_parameters(target)
  output["scenario"] = asdict(scenario)
  expected = deployment_context_sha256(output)
  if payload.get("parameters_sha256") != expected:
    raise ValueError("specialist context parameter hash mismatch")
  output["parameters_sha256"] = expected
  return output


def _validate_frozen_v19_context(payload: Mapping[str, Any]) -> dict[str, Any]:
  if payload.get("schema_version") != V19_CONTEXT_SCHEMA_VERSION:
    raise ValueError("unexpected v19 context schema version")
  if not isinstance(payload.get("calibration_candidate_seed"), int):
    raise ValueError("v19 context candidate seed is missing")
  mode = str(payload.get("specialist_mode"))
  if mode not in V19_SPECIALIST_MODES:
    raise ValueError("v19 context mode is missing or invalid")
  target = payload.get("target")
  scenario_raw = payload.get("scenario")
  if not isinstance(target, Mapping) or not isinstance(scenario_raw, Mapping):
    raise ValueError("v19 context target or scenario is missing")
  scenario = V19ScenarioParameters(
    **{
      field: (
        bool(scenario_raw[field])
        if field == "disturbance_pulses_with_centering"
        else int(scenario_raw[field])
        if field == "contact_observation_delay_steps"
        else float(scenario_raw[field])
      )
      for field in V19ScenarioParameters.__dataclass_fields__
    }
  )
  _validate_v19_scenario(mode, scenario)
  output = dict(payload)
  output["target"] = _validated_deployment_parameters(target)
  output["scenario"] = asdict(scenario)
  expected = deployment_context_sha256(output)
  if payload.get("parameters_sha256") != expected:
    raise ValueError("v19 context parameter hash mismatch")
  output["parameters_sha256"] = expected
  return output


def _validate_frozen_v21_context(payload: Mapping[str, Any]) -> dict[str, Any]:
  if payload.get("schema_version") != V21_CONTEXT_SCHEMA_VERSION:
    raise ValueError("unexpected v21 context schema version")
  context_id = str(payload.get("context_id"))
  if context_id not in V21_CONTEXT_SPECS:
    raise ValueError("v21 context ID is missing or invalid")
  seed = payload.get("calibration_candidate_seed")
  if not isinstance(seed, int):
    raise ValueError("v21 context candidate seed is missing")
  target = payload.get("target")
  scenario_raw = payload.get("scenario")
  if not isinstance(target, Mapping) or not isinstance(scenario_raw, Mapping):
    raise ValueError("v21 context target or scenario is missing")
  scenario = V19ScenarioParameters(
    **{
      field: (
        bool(scenario_raw[field])
        if field == "disturbance_pulses_with_centering"
        else int(scenario_raw[field])
        if field == "contact_observation_delay_steps"
        else float(scenario_raw[field])
      )
      for field in V19ScenarioParameters.__dataclass_fields__
    }
  )
  mode = str(payload.get("specialist_mode"))
  _validate_v19_scenario(mode, scenario, v21=True)
  normalized = dict(payload)
  normalized["target"] = _validated_deployment_parameters(target)
  normalized["scenario"] = asdict(scenario)
  expected = generate_v21_specialist_context(context_id, seed)
  immutable_fields = (
    "kind",
    "schema_version",
    "calibration_candidate_seed",
    "specialist_mode",
    "context_id",
    "context_family",
    "formal_context",
    "family_spec_sha256",
    "candidate_severity",
    "target",
    "scenario",
    "parameters_sha256",
  )
  differing = [
    field for field in immutable_fields if normalized.get(field) != expected[field]
  ]
  if differing:
    raise ValueError(f"v21 context differs from its frozen family: {differing}")
  return {
    **normalized,
    **{field: expected[field] for field in immutable_fields},
  }


def validate_frozen_deployment_context(payload: Mapping[str, Any]) -> dict[str, Any]:
  if payload.get("kind") == V21_CONTEXT_KIND:
    return _validate_frozen_v21_context(payload)
  if payload.get("kind") == V19_CONTEXT_KIND:
    return _validate_frozen_v19_context(payload)
  if payload.get("kind") == SPECIALIST_CONTEXT_KIND:
    return _validate_frozen_specialist_context(payload)
  if payload.get("kind") != FAILURE_FOCUSED_CONTEXT_KIND:
    raise ValueError("unexpected deployment context kind")
  if payload.get("schema_version") != FAILURE_FOCUSED_CONTEXT_SCHEMA_VERSION:
    raise ValueError("unexpected deployment context schema version")
  if not isinstance(payload.get("calibration_candidate_seed"), int):
    raise ValueError("deployment context candidate seed is missing")
  output = dict(payload)
  for role in ("target", "neighbor"):
    raw = payload.get(role)
    if not isinstance(raw, Mapping):
      raise ValueError(f"deployment context role {role!r} is missing")
    output[role] = _validated_deployment_parameters(raw)
  expected = deployment_context_sha256(output)
  if payload.get("parameters_sha256") != expected:
    raise ValueError("deployment context parameter hash mismatch")
  output["parameters_sha256"] = expected
  return output


def load_frozen_deployment_context(path: str | Path) -> dict[str, Any]:
  context_path = Path(path).resolve()
  payload = json.loads(context_path.read_text())
  return validate_frozen_deployment_context(payload)


def validate_calibrated_deployment_context(
  payload: Mapping[str, Any],
) -> dict[str, Any]:
  """Require auditable base-policy-only selection of the first 75--85% context."""
  output = validate_frozen_deployment_context(payload)
  calibration = payload.get("calibration")
  if not isinstance(calibration, Mapping):
    raise ValueError("formal deployment context is missing calibration evidence")
  if calibration.get("kind") != FAILURE_FOCUSED_CALIBRATION_KIND:
    raise ValueError("unexpected deployment-context calibration kind")
  if calibration.get("selection_metric_fields") != ["success_rate"]:
    raise ValueError("context selection must use only base-policy success rate")
  if calibration.get("adapted_policy_evaluations_used") is not False:
    raise ValueError("context calibration must not use adapted-policy evaluations")
  bounds = calibration.get("success_rate_bounds")
  if bounds != [0.75, 0.85]:
    raise ValueError("formal context calibration bounds must be [0.75, 0.85]")
  candidate_seeds = calibration.get("candidate_seeds")
  attempts = calibration.get("attempts")
  if (
    not isinstance(candidate_seeds, list)
    or not candidate_seeds
    or not all(isinstance(seed, int) for seed in candidate_seeds)
  ):
    raise ValueError("calibration candidate seeds are missing or invalid")
  if not isinstance(attempts, list) or not attempts:
    raise ValueError("calibration attempts are missing")
  if len(attempts) > len(candidate_seeds):
    raise ValueError("calibration attempts exceed the declared candidate seeds")
  for index, attempt in enumerate(attempts):
    if not isinstance(attempt, Mapping):
      raise ValueError("calibration attempt must be a mapping")
    candidate_seed = attempt.get("candidate_seed")
    if candidate_seed != candidate_seeds[index]:
      raise ValueError("calibration did not inspect candidate seeds in order")
    if attempt.get("base_policy_only") is not True:
      raise ValueError("every calibration attempt must use only the base policy")
    if int(attempt.get("num_episodes", 0)) < 1:
      raise ValueError("calibration attempt has no evaluated episodes")
    success_rate = float(attempt.get("success_rate", float("nan")))
    if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
      raise ValueError("calibration success rate is missing or invalid")
    qualifies = 0.75 <= success_rate <= 0.85
    if index < len(attempts) - 1 and qualifies:
      raise ValueError("calibration skipped an earlier qualifying context")
    if index == len(attempts) - 1 and not qualifies:
      raise ValueError("selected calibration context is outside [0.75, 0.85]")
  selected = attempts[-1]
  if selected["candidate_seed"] != output["calibration_candidate_seed"]:
    raise ValueError("selected calibration seed differs from frozen context")
  if selected.get("parameters_sha256") != output["parameters_sha256"]:
    raise ValueError("selected calibration hash differs from frozen context")
  if calibration.get("selected_candidate_seed") != selected["candidate_seed"]:
    raise ValueError("calibration selected-candidate metadata is inconsistent")
  if calibration.get("selected_parameters_sha256") != output["parameters_sha256"]:
    raise ValueError("calibration selected hash is inconsistent")
  output["calibration"] = dict(calibration)
  return output


def load_calibrated_deployment_context(path: str | Path) -> dict[str, Any]:
  context_path = Path(path).resolve()
  payload = json.loads(context_path.read_text())
  return validate_calibrated_deployment_context(payload)


def validate_calibrated_specialist_context(
  payload: Mapping[str, Any],
) -> dict[str, Any]:
  """Require base-only selection under all single-dominant-failure gates."""
  output = _validate_frozen_specialist_context(payload)
  calibration = payload.get("calibration")
  if not isinstance(calibration, Mapping):
    raise ValueError("formal specialist context is missing calibration evidence")
  if calibration.get("kind") != SPECIALIST_CALIBRATION_KIND:
    raise ValueError("unexpected specialist calibration kind")
  if calibration.get("adapted_policy_evaluations_used") is not False:
    raise ValueError("specialist calibration used an adapted policy")
  if calibration.get("success_rate_bounds") != [0.70, 0.85]:
    raise ValueError("specialist success-rate bounds must be [0.70, 0.85]")
  if calibration.get("minimum_target_failure_fraction") != 0.60:
    raise ValueError("specialist target-failure fraction must be at least 0.60")
  if calibration.get("maximum_second_failure_fraction") != 0.30:
    raise ValueError("specialist second failure fraction must be at most 0.30")
  if calibration.get("minimum_fall_count") != 100:
    raise ValueError("specialist calibration must require at least 100 falls")
  if calibration.get("episodes_per_candidate", 0) < 512:
    raise ValueError("specialist calibration requires at least 512 episodes")
  candidate_seeds = calibration.get("candidate_seeds")
  attempts = calibration.get("attempts")
  if (
    not isinstance(candidate_seeds, list)
    or not candidate_seeds
    or candidate_seeds != sorted(set(candidate_seeds))
  ):
    raise ValueError("specialist calibration seeds are missing or unordered")
  if not isinstance(attempts, list) or not attempts:
    raise ValueError("specialist calibration attempts are missing")
  target_type = SPECIALIST_FAILURE_TYPES[output["specialist_mode"]]
  for index, attempt in enumerate(attempts):
    if attempt.get("candidate_seed") != candidate_seeds[index]:
      raise ValueError("specialist calibration did not inspect seeds in order")
    if attempt.get("base_policy_only") is not True:
      raise ValueError("specialist calibration attempt was not base-policy only")
    if int(attempt.get("num_episodes", 0)) < 512:
      raise ValueError("specialist calibration attempt has fewer than 512 episodes")
    qualifies = (
      0.70 <= float(attempt.get("success_rate", -1.0)) <= 0.85
      and int(attempt.get("fall_count", -1)) >= 100
      and float(attempt.get("target_failure_fraction", -1.0)) >= 0.60
      and float(attempt.get("second_failure_fraction", 2.0)) <= 0.30
      and attempt.get("target_failure_type") == target_type
    )
    if bool(attempt.get("qualifies")) is not qualifies:
      raise ValueError("specialist calibration qualification metadata is inconsistent")
    if index < len(attempts) - 1 and qualifies:
      raise ValueError("specialist calibration skipped an earlier qualifying context")
    if index == len(attempts) - 1 and not qualifies:
      raise ValueError("selected specialist context fails its calibration gates")
  selected = attempts[-1]
  if selected["candidate_seed"] != output["calibration_candidate_seed"]:
    raise ValueError("selected specialist seed differs from frozen context")
  if selected.get("parameters_sha256") != output["parameters_sha256"]:
    raise ValueError("selected specialist hash differs from frozen context")
  if calibration.get("selected_candidate_seed") != selected["candidate_seed"]:
    raise ValueError("specialist selected-seed metadata is inconsistent")
  if calibration.get("selected_parameters_sha256") != output["parameters_sha256"]:
    raise ValueError("specialist selected-hash metadata is inconsistent")
  output["calibration"] = dict(calibration)
  return output


def load_calibrated_specialist_context(path: str | Path) -> dict[str, Any]:
  context_path = Path(path).resolve()
  payload = json.loads(context_path.read_text())
  return validate_calibrated_specialist_context(payload)


def validate_calibrated_v19_context(
  payload: Mapping[str, Any],
) -> dict[str, Any]:
  """Validate base-only first-qualifying observable-context calibration."""
  v21 = payload.get("kind") == V21_CONTEXT_KIND
  output = (
    _validate_frozen_v21_context(payload)
    if v21
    else _validate_frozen_v19_context(payload)
  )
  calibration = payload.get("calibration")
  if not isinstance(calibration, Mapping):
    raise ValueError("formal v19 context is missing calibration evidence")
  expected_calibration_kind = (
    V21_CALIBRATION_KIND if v21 else V19_CALIBRATION_KIND
  )
  if calibration.get("kind") != expected_calibration_kind:
    raise ValueError("unexpected v19 calibration kind")
  if calibration.get("adapted_policy_evaluations_used") is not False:
    raise ValueError("v19 context calibration used an adapted policy")
  if calibration.get("success_rate_bounds") != [0.70, 0.85]:
    raise ValueError("v19 success-rate bounds must be [0.70, 0.85]")
  mode = output["specialist_mode"]
  required_purity = 0.80 if mode == "lateral" else 0.75
  required_second = 0.30 if mode == "lateral" else 0.20
  if calibration.get("minimum_target_failure_fraction") != required_purity:
    raise ValueError("v19 target-failure purity threshold is incorrect")
  if calibration.get("maximum_second_failure_fraction") != required_second:
    raise ValueError("v19 second-failure threshold is incorrect")
  if calibration.get("minimum_fall_count") != 100:
    raise ValueError("v19 calibration must require at least 100 falls")
  if int(calibration.get("episodes_per_candidate", 0)) < 512:
    raise ValueError("v19 calibration requires at least 512 episodes")
  candidate_seeds = calibration.get("candidate_seeds")
  attempts = calibration.get("attempts")
  if (
    not isinstance(candidate_seeds, list)
    or not candidate_seeds
    or candidate_seeds != sorted(set(candidate_seeds))
  ):
    raise ValueError("v19 calibration seeds are missing or unordered")
  if not isinstance(attempts, list) or not attempts:
    raise ValueError("v19 calibration attempts are missing")
  if len(attempts) > len(candidate_seeds):
    raise ValueError("v19 calibration attempts exceed candidate seeds")
  target_type = V19_SPECIALIST_FAILURE_TYPES[mode]
  for index, attempt in enumerate(attempts):
    if attempt.get("candidate_seed") != candidate_seeds[index]:
      raise ValueError("v19 calibration did not inspect seeds in order")
    if attempt.get("base_policy_only") is not True:
      raise ValueError("v19 calibration attempt was not base-policy only")
    if int(attempt.get("num_episodes", 0)) < 512:
      raise ValueError("v19 calibration attempt has fewer than 512 episodes")
    qualifies = (
      0.70 <= float(attempt.get("success_rate", -1.0)) <= 0.85
      and int(attempt.get("fall_count", -1)) >= 100
      and float(attempt.get("target_failure_fraction", -1.0)) >= required_purity
      and float(attempt.get("second_failure_fraction", 2.0)) <= required_second
      and attempt.get("target_failure_type") == target_type
    )
    if bool(attempt.get("qualifies")) is not qualifies:
      raise ValueError("v19 calibration qualification metadata is inconsistent")
    if index < len(attempts) - 1 and qualifies:
      raise ValueError("v19 calibration skipped an earlier qualifying context")
    if index == len(attempts) - 1 and not qualifies:
      raise ValueError("selected v19 context fails its calibration gates")
  selected = attempts[-1]
  if selected["candidate_seed"] != output["calibration_candidate_seed"]:
    raise ValueError("selected v19 seed differs from frozen context")
  if selected.get("parameters_sha256") != output["parameters_sha256"]:
    raise ValueError("selected v19 hash differs from frozen context")
  if calibration.get("selected_candidate_seed") != selected["candidate_seed"]:
    raise ValueError("v19 selected-seed metadata is inconsistent")
  if calibration.get("selected_parameters_sha256") != output["parameters_sha256"]:
    raise ValueError("v19 selected-hash metadata is inconsistent")
  output["calibration"] = dict(calibration)
  return output


def load_calibrated_v19_context(path: str | Path) -> dict[str, Any]:
  payload = json.loads(Path(path).resolve().read_text())
  return validate_calibrated_v19_context(payload)


def validate_calibrated_v21_context(
  payload: Mapping[str, Any],
) -> dict[str, Any]:
  if payload.get("kind") != V21_CONTEXT_KIND:
    raise ValueError("expected a v21 deployment context")
  return validate_calibrated_v19_context(payload)


def load_calibrated_v21_context(path: str | Path) -> dict[str, Any]:
  payload = json.loads(Path(path).resolve().read_text())
  return validate_calibrated_v21_context(payload)


def configure_v19_actor_interface(
  cfg,
  payload: Mapping[str, Any],
  *,
  target_contact_phase_shift: bool = False,
) -> dict[str, object]:
  """Install the v19 interface and, only on target, its gait-clock shift."""
  validated = validate_frozen_deployment_context(payload)
  if validated.get("kind") not in (V19_CONTEXT_KIND, V21_CONTEXT_KIND):
    raise ValueError("observable actor interface requires a v19/v21 context")
  mode = validated["specialist_mode"]
  scenario = V19ScenarioParameters(**validated["scenario"])
  from .config import configure_v19_observable_refinement_env

  effective_phase_offset = (
    scenario.gait_phase_offset
    if mode == "contact_stability" and target_contact_phase_shift
    else 0.0
  )
  effective_contact_delay = (
    scenario.contact_observation_delay_steps
    if mode == "contact_stability" and target_contact_phase_shift
    else 0
  )
  if effective_phase_offset:
    # The contact scenario represents a small desynchronization between the
    # deployable gait clock and physical contact.  Applying it to the existing
    # phase observation makes the shift observable to pi0 during base-only
    # calibration; the five appended columns remain zero-weight at warm start.
    # D0 calls this helper with the default False and therefore stays nominal.
    for group_name in ("actor", "critic"):
      phase_term = cfg.observations[group_name].terms["phase"]
      phase_term.params = dict(phase_term.params)
      phase_term.params["phase_offset"] = effective_phase_offset

  reward_parameters: dict[str, float | int] = {
    "recovery_scale": scenario.recovery_reward_scale
  }
  if mode == "lateral":
    reward_parameters["edge_penalty_scale"] = scenario.edge_penalty_scale
  else:
    reward_parameters.update(
      phase_offset=effective_phase_offset,
      pre_touchdown_steps=30,
      post_touchdown_steps=45,
    )
  metadata = configure_v19_observable_refinement_env(
    cfg,
    mode=mode,
    contact_observation_delay_steps=effective_contact_delay,
    phase_offset=effective_phase_offset,
    reward_parameters=reward_parameters,
  )
  metadata.update(
    target_contact_phase_shift_applied=bool(effective_phase_offset),
    actor_gait_phase_observation_offset=float(effective_phase_offset),
  )
  return metadata


def apply_frozen_deployment_context(
  cfg,
  payload: Mapping[str, Any],
  *,
  role: str,
) -> dict[str, Any]:
  """Apply a frozen context entirely on the environment side."""
  validated = validate_frozen_deployment_context(payload)
  legacy_specialist = validated.get("kind") == SPECIALIST_CONTEXT_KIND
  observable_specialist = validated.get("kind") in (
    V19_CONTEXT_KIND,
    V21_CONTEXT_KIND,
  )
  specialist = legacy_specialist or observable_specialist
  if role not in ({"target"} if specialist else {"target", "neighbor"}):
    raise ValueError("deployment context role must be 'target' or 'neighbor'")
  parameters = DeploymentContextParameters(**validated[role])
  terrain_generator = cfg.scene.terrain.terrain_generator
  if terrain_generator is None:
    raise ValueError("deployment context requires generated stair terrain")
  if len(terrain_generator.sub_terrains) != 1:
    raise ValueError("deployment context requires one template staircase")
  name, stairs = next(iter(terrain_generator.sub_terrains.items()))
  stairs.num_steps = parameters.num_steps
  stairs.step_height_range = (
    min(parameters.rise_profile),
    max(parameters.rise_profile),
  )
  stairs.step_width = sum(parameters.tread_profile) / parameters.num_steps
  stairs.step_height_profile = parameters.rise_profile
  stairs.step_width_profile = parameters.tread_profile
  stairs.flat_patch_sampling = {
    "stair_targets": FlatPatchSamplingCfg(
      num_patches=parameters.num_steps + 1,
      patch_radius=0.10,
      max_height_diff=0.02,
    ),
    "stair_risers": FlatPatchSamplingCfg(
      num_patches=parameters.num_steps,
      patch_radius=0.01,
      max_height_diff=1.0,
    ),
  }
  terrain_generator.sub_terrains = {name: stairs}
  terrain_generator.size = (
    1.35 + sum(parameters.tread_profile) + 1.5,
    terrain_generator.size[1],
  )

  command = cfg.commands["twist"]
  low, high = command.forward_velocity_range
  command.forward_velocity_range = (
    low * parameters.command_forward_scale,
    high * parameters.command_forward_scale,
  )
  command.command_delay_range_s = (
    parameters.command_delay_s,
    parameters.command_delay_s,
  )
  command.low_pass_time_constant_s = parameters.command_low_pass_s

  action = cfg.actions["joint_pos"]
  action.num_steps = parameters.num_steps
  action.step_width = stairs.step_width
  action.step_height = sum(parameters.rise_profile) / parameters.num_steps
  action.deployment_action_gain = parameters.action_gain
  action.deployment_action_bias = parameters.action_bias
  action.deployment_action_delay_steps = parameters.action_delay_steps

  scenario_metadata = None
  if observable_specialist:
    mode = validated["specialist_mode"]
    scenario = V19ScenarioParameters(**validated["scenario"])
    command.disturbance_pulses_with_centering = (
      scenario.disturbance_pulses_with_centering
    )
    command.fixed_lateral_bias = scenario.lateral_command_bias
    command.fixed_yaw_bias = scenario.yaw_command_bias
    command.lateral_pulse_abs_range = (
      scenario.lateral_pulse_min,
      scenario.lateral_pulse_max,
    )
    command.yaw_pulse_abs_range = (
      scenario.yaw_pulse_min,
      scenario.yaw_pulse_max,
    )
    command.pulse_interval_range_s = (
      scenario.pulse_interval_min_s,
      scenario.pulse_interval_max_s,
    )
    command.pulse_duration_range_s = (
      scenario.pulse_duration_min_s,
      scenario.pulse_duration_max_s,
    )
    command.centerline_lateral_gain = scenario.centerline_lateral_gain
    command.centerline_heading_gain = scenario.centerline_heading_gain
    command.centerline_max_lateral_velocity = (
      scenario.centerline_max_lateral_velocity
    )
    command.centerline_max_yaw_velocity = scenario.centerline_max_yaw_velocity
    action.toe_margin = scenario.toe_margin
    # Resolve by explicit joint name rather than assuming simulator/action
    # ordering.  Roll/yaw gains stay symmetric so this contact scene does not
    # manufacture a lateral specialist in disguise.
    action.deployment_action_scale = (
      {
        "left_hip_pitch_joint": scenario.left_response_scale,
        "left_knee_joint": scenario.left_response_scale,
        "left_ankle_pitch_joint": scenario.left_response_scale,
        "right_hip_pitch_joint": scenario.right_response_scale,
        "right_knee_joint": scenario.right_response_scale,
        "right_ankle_pitch_joint": scenario.right_response_scale,
      }
      if mode == "contact_stability"
      else None
    )
    action.deployment_contact_phase_offset = (
      scenario.gait_phase_offset if mode == "contact_stability" else 0.0
    )
    action.deployment_contact_delay_steps = (
      scenario.contact_observation_delay_steps
      if mode == "contact_stability"
      else 0
    )
    foot_geom_names = tuple(
      f"{side}_foot{index}_collision"
      for side in ("left", "right")
      for index in range(1, 8)
    )
    cfg.events["specialist_foot_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
        "operation": "abs",
        "ranges": (scenario.foot_friction, scenario.foot_friction),
        "shared_random": True,
      },
    )
    observable_metadata = configure_v19_actor_interface(
      cfg,
      validated,
      target_contact_phase_shift=True,
    )
    scenario_metadata = {
      "specialist_mode": mode,
      "target_failure_type": V19_SPECIALIST_FAILURE_TYPES[mode],
      "parameters": asdict(scenario),
      "observable_actor": observable_metadata,
    }

  if legacy_specialist:
    mode = validated["specialist_mode"]
    scenario = SpecialistScenarioParameters(**validated["scenario"])
    command.disturbance_pulses_with_centering = (
      scenario.disturbance_pulses_with_centering
    )
    command.fixed_lateral_bias = scenario.lateral_command_bias
    command.fixed_yaw_bias = scenario.yaw_command_bias
    command.lateral_pulse_abs_range = (
      scenario.lateral_pulse_min,
      scenario.lateral_pulse_max,
    )
    command.yaw_pulse_abs_range = (
      scenario.yaw_pulse_min,
      scenario.yaw_pulse_max,
    )
    command.pulse_interval_range_s = (
      scenario.pulse_interval_min_s,
      scenario.pulse_interval_max_s,
    )
    command.pulse_duration_range_s = (
      scenario.pulse_duration_min_s,
      scenario.pulse_duration_max_s,
    )
    command.centerline_lateral_gain = scenario.centerline_lateral_gain
    command.centerline_heading_gain = scenario.centerline_heading_gain
    command.centerline_max_lateral_velocity = (
      scenario.centerline_max_lateral_velocity
    )
    command.centerline_max_yaw_velocity = scenario.centerline_max_yaw_velocity
    action.toe_margin = scenario.toe_margin
    foot_geom_names = tuple(
      f"{side}_foot{index}_collision"
      for side in ("left", "right")
      for index in range(1, 8)
    )
    cfg.events["specialist_foot_friction"] = EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
        "operation": "abs",
        "ranges": (scenario.foot_friction, scenario.foot_friction),
        "shared_random": True,
      },
    )
    reward = cfg.rewards["specialist_failure_signal"]
    reward.weight = -scenario.failure_signal_scale
    reward.params = {
      "mode": mode,
      "command_name": "twist",
      "action_name": "joint_pos",
      "asset_name": "robot",
      "sensor_name": "feet_ground_contact",
      "weights": {
        "centerline": scenario.centerline_signal_weight,
        "heading": scenario.heading_signal_weight,
        "edge": scenario.edge_signal_weight,
        "intervention": scenario.intervention_signal_weight,
        "nominal_margin": scenario.nominal_margin_signal_weight,
        "roll": scenario.roll_signal_weight,
        "pitch": scenario.pitch_signal_weight,
        "angular_velocity": scenario.angular_velocity_signal_weight,
        "slip": scenario.slip_signal_weight,
        "contact_mismatch": scenario.contact_mismatch_signal_weight,
      },
    }
    scenario_metadata = {
      "specialist_mode": mode,
      "target_failure_type": SPECIALIST_FAILURE_TYPES[mode],
      "parameters": asdict(scenario),
    }

  cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (parameters.encoder_bias, parameters.encoder_bias),
    },
  )
  cfg.episode_length_s = parameters.episode_length_s
  cfg.scene.extent = max(cfg.scene.extent, 5.0)
  return {
    "role": role,
    "parameters_sha256": validated["parameters_sha256"],
    "parameters": asdict(parameters),
    "scenario": scenario_metadata,
    "actor_context_fields_added": 5 if observable_specialist else 0,
  }
