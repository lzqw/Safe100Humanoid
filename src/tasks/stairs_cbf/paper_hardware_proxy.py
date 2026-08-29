"""Frozen CBF-off hardware-proxy perturbation bundle for final policies."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .paper_dual_v35 import configure_paper_training_domain_randomization

METHOD_ID = "paper-filter-free-hardware-proxy-v1"
ACTION_DELAY_STEPS = (1, 2)
ACTUATOR_GAIN = 0.95
STAIR_HEIGHT_ESTIMATE_BIAS_M = 0.01
TREAD_PERTURBATION_M = 0.015
COMMAND_DELAY_RANGE_S = (0.08, 0.20)
COMMAND_LOW_PASS_S = 0.10
RANDOMIZATION_STRENGTH = 0.50


def configure_hardware_proxy(env_cfg, *, action_delay_steps: int) -> dict[str, Any]:
  """Apply one frozen moderate hardware proxy without changing the actor."""
  if action_delay_steps not in ACTION_DELAY_STEPS:
    raise ValueError(
      f"hardware proxy action delay must be one of {ACTION_DELAY_STEPS}"
    )
  randomization = configure_paper_training_domain_randomization(
    env_cfg,
    "paper_static",
    strength=RANDOMIZATION_STRENGTH,
  )
  # The requested proxy varies sensing, contact, actuation, command delivery,
  # and stair geometry.  Base-COM randomization is deliberately excluded.
  env_cfg.events.pop("base_com", None)
  randomization["event_terms"] = ["encoder_bias", "foot_friction"]
  randomization.pop("base_com_operation", None)
  randomization.pop("base_com_ranges_m", None)

  command = env_cfg.commands["twist"]
  command.command_delay_range_s = COMMAND_DELAY_RANGE_S
  command.low_pass_time_constant_s = COMMAND_LOW_PASS_S

  action = env_cfg.actions["joint_pos"]
  # v25 deliberately freezes the learnable teacher's internal plant transform.
  # The evaluator applies the proxy-only gain and FIFO delay immediately before
  # ``env.step`` instead, leaving the frozen teacher configuration untouched.
  # A positive height-estimate bias is equivalent to raising the CBF's
  # estimated riser top while leaving the physical terrain unchanged.
  action.top_clearance += STAIR_HEIGHT_ESTIMATE_BIAS_M

  terrain_generator = env_cfg.scene.terrain.terrain_generator
  if terrain_generator is None or len(terrain_generator.sub_terrains) != 1:
    raise RuntimeError("hardware proxy requires one generated stair terrain")
  name, stairs = next(iter(terrain_generator.sub_terrains.items()))
  base_width = float(stairs.step_width)
  tread_profile = tuple(
    base_width + (TREAD_PERTURBATION_M if index % 2 == 0 else -TREAD_PERTURBATION_M)
    for index in range(int(stairs.num_steps))
  )
  stairs = replace(
    stairs,
    step_width_profile=tread_profile,
  )
  terrain_generator.sub_terrains = {name: stairs}
  terrain_generator.size = (
    1.35 + sum(tread_profile) + 1.5,
    terrain_generator.size[1],
  )
  return {
    "method_id": METHOD_ID,
    "cbf_off_evaluation_only": True,
    "actor_observation_corruption": True,
    "sensor_noise_source": "native_G1_actor_uniform_noise_at_half_strength",
    "imu_angular_velocity_noise_range": [-0.10, 0.10],
    "projected_gravity_noise_range": [-0.025, 0.025],
    "joint_position_noise_range": [-0.005, 0.005],
    "joint_velocity_noise_range": [-0.75, 0.75],
    "encoder_bias_range": randomization["encoder_bias_range"],
    "foot_friction_range": randomization["foot_friction_range"],
    "action_delay_steps": action_delay_steps,
    "action_delay_implementation": "evaluator_fifo_after_standard_action_clip_before_env_step",
    "actuator_gain": ACTUATOR_GAIN,
    "actuator_gain_implementation": "evaluator_scale_after_standard_action_clip_before_env_step",
    "stair_height_estimate_bias_m": STAIR_HEIGHT_ESTIMATE_BIAS_M,
    "stair_height_bias_implementation": "cbf_estimated_top_clearance_offset",
    "physical_tread_profile_m": list(tread_profile),
    "tread_perturbation_m": TREAD_PERTURBATION_M,
    "command_delay_range_s": list(COMMAND_DELAY_RANGE_S),
    "command_low_pass_time_constant_s": COMMAND_LOW_PASS_S,
    "base_com_randomization": False,
    "external_pushes": False,
  }
