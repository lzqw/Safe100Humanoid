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


def _canonical_json(value: Mapping[str, Any]) -> bytes:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode("utf-8")


def deployment_context_sha256(payload: Mapping[str, Any]) -> str:
  """Hash only the frozen parameters and schema, excluding calibration metrics."""
  identity = {
    "kind": payload["kind"],
    "schema_version": payload["schema_version"],
    "calibration_candidate_seed": payload["calibration_candidate_seed"],
    "target": payload["target"],
    "neighbor": payload["neighbor"],
  }
  return hashlib.sha256(_canonical_json(identity)).hexdigest()


def _validate_parameters(parameters: DeploymentContextParameters) -> None:
  if not 9 <= parameters.num_steps <= 18:
    raise ValueError("deployment context must contain 9--18 risers")
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


def validate_frozen_deployment_context(payload: Mapping[str, Any]) -> dict[str, Any]:
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
    output[role] = asdict(parameters)
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


def apply_frozen_deployment_context(
  cfg,
  payload: Mapping[str, Any],
  *,
  role: str,
) -> dict[str, Any]:
  """Apply a frozen context entirely on the environment side."""
  validated = validate_frozen_deployment_context(payload)
  if role not in {"target", "neighbor"}:
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
    "actor_context_fields_added": 0,
  }
