"""Hiking-style flat-patch target sampling and position velocity command."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .teleop_math import centerline_feedback_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class StairTargetCommand(CommandTerm):
  cfg: "StairTargetCommandCfg"

  def __init__(self, cfg: "StairTargetCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    terrain = env.scene.terrain
    if terrain is None or cfg.patch_name not in terrain.flat_patches:
      raise RuntimeError(f"terrain has no flat-patch set {cfg.patch_name!r}")
    self._patches = terrain.flat_patches[cfg.patch_name]
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.metrics["target_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["forward_progress"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _env_patches(self, env_ids: torch.Tensor) -> torch.Tensor:
    terrain = self._env.scene.terrain
    assert terrain is not None
    return self._patches[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    patches = self._env_patches(env_ids)
    root_x = self.robot.data.root_link_pos_w[env_ids, 0]
    ahead = patches[..., 0] > root_x.unsqueeze(-1) + self.cfg.reached_radius
    patch_indices = torch.arange(
      patches.shape[1], device=self.device, dtype=torch.long
    ).expand_as(ahead)
    candidate = torch.where(
      ahead,
      patch_indices,
      torch.full_like(patch_indices, patches.shape[1]),
    )
    index = candidate.min(dim=1).values.clamp_max(patches.shape[1] - 1)
    index = (index + self.cfg.lookahead - 1).clamp_max(patches.shape[1] - 1)
    self.target_index[env_ids] = index
    self.target_w[env_ids] = patches[
      torch.arange(len(env_ids), device=self.device), index
    ]

  def _update_metrics(self) -> None:
    delta = self.target_w - self.robot.data.root_link_pos_w
    self.metrics["target_distance"] += torch.linalg.vector_norm(delta[:, :2], dim=1)
    self.metrics["forward_progress"] += self.robot.data.root_link_pos_w[:, 0]

  def _update_command(self) -> None:
    delta_w = self.target_w - self.robot.data.root_link_pos_w
    reached = torch.linalg.vector_norm(delta_w[:, :2], dim=1) < self.cfg.reached_radius
    terrain = self._env.scene.terrain
    assert terrain is not None
    last = self._patches.shape[2] - 1
    advance = reached & (self.target_index < last)
    if torch.any(advance):
      env_ids = advance.nonzero(as_tuple=False).flatten()
      self.target_index[env_ids] += 1
      patches = self._env_patches(env_ids)
      self.target_w[env_ids] = patches[
        torch.arange(len(env_ids), device=self.device), self.target_index[env_ids]
      ]
      delta_w = self.target_w - self.robot.data.root_link_pos_w

    delta_b = quat_apply_inverse(self.robot.data.root_link_quat_w, delta_w)
    heading_error = torch.atan2(delta_b[:, 1], delta_b[:, 0])
    self._command[:, 0] = torch.clamp(
      self.cfg.position_gain * delta_b[:, 0], min=0.0, max=self.cfg.max_forward_velocity
    )
    self._command[:, 1] = torch.clamp(
      self.cfg.position_gain * delta_b[:, 1],
      min=-self.cfg.max_lateral_velocity,
      max=self.cfg.max_lateral_velocity,
    )
    self._command[:, 2] = torch.clamp(
      self.cfg.heading_gain * heading_error,
      min=-self.cfg.max_yaw_velocity,
      max=self.cfg.max_yaw_velocity,
    )
    finished = reached & (self.target_index == last)
    self._command[finished] = 0.0


@dataclass(kw_only=True)
class StairTargetCommandCfg(CommandTermCfg):
  entity_name: str
  patch_name: str = "stair_targets"
  position_gain: float = 1.5
  heading_gain: float = 1.5
  max_forward_velocity: float = 0.8
  max_lateral_velocity: float = 0.20
  max_yaw_velocity: float = 0.8
  reached_radius: float = 0.22
  lookahead: int = 2

  def build(self, env: "ManagerBasedRlEnv") -> StairTargetCommand:
    return StairTargetCommand(self, env)


class JoystickVelocityCommand(CommandTerm):
  """Piecewise-constant teleoperation command with delivery imperfections.

  The actor only receives :attr:`command`, i.e. the delivered command.  Raw
  joystick state, delay state, and command derivative are exposed separately
  for the privileged critic and diagnostics.
  """

  cfg: "JoystickVelocityCommandCfg"

  def __init__(self, cfg: "JoystickVelocityCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    terrain = env.scene.terrain
    if terrain is None or cfg.patch_name not in terrain.flat_patches:
      raise RuntimeError(f"terrain has no flat-patch set {cfg.patch_name!r}")
    self._patches = terrain.flat_patches[cfg.patch_name]
    self.raw_command = torch.zeros(self.num_envs, 3, device=self.device)
    self.delivered_command = torch.zeros_like(self.raw_command)
    self.command_derivative = torch.zeros_like(self.raw_command)
    self.delay_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._max_delay_steps = max(
      0, math.ceil(cfg.command_delay_range_s[1] / env.step_dt)
    )
    self._delay_queue = torch.zeros(
      self.num_envs, self._max_delay_steps + 1, 3, device=self.device
    )
    self._next_pulse_steps = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._pulse_steps_left = torch.zeros_like(self._next_pulse_steps)
    self._pulse_value = torch.zeros_like(self.raw_command)
    self._release_countdown = torch.full_like(self._next_pulse_steps, -1)
    self._released = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.centerline_error = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.operator_correction = torch.zeros_like(self.raw_command)
    self.correction_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["delay_steps"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["abs_centerline_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["abs_heading_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["operator_correction_fraction"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.delivered_command

  def _env_patches(self, env_ids: torch.Tensor) -> torch.Tensor:
    terrain = self._env.scene.terrain
    assert terrain is not None
    return self._patches[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

  def _sample_integer_steps(
    self, env_ids: torch.Tensor, seconds_range: tuple[float, float]
  ) -> torch.Tensor:
    low = max(0, int(round(seconds_range[0] / self._env.step_dt)))
    high = max(low, int(round(seconds_range[1] / self._env.step_dt)))
    return torch.randint(
      low, high + 1, (len(env_ids),), device=self.device, dtype=torch.long
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    r = torch.empty(len(env_ids), device=self.device)
    self.raw_command[env_ids] = 0.0
    self.raw_command[env_ids, 0] = r.uniform_(*self.cfg.forward_velocity_range)
    self.delivered_command[env_ids] = 0.0
    self.command_derivative[env_ids] = 0.0
    delay = self._sample_integer_steps(env_ids, self.cfg.command_delay_range_s)
    self.delay_steps[env_ids] = delay.clamp_max(self._max_delay_steps)
    self._delay_queue[env_ids] = 0.0
    self._next_pulse_steps[env_ids] = self._sample_integer_steps(
      env_ids, self.cfg.pulse_interval_range_s
    )
    self._pulse_steps_left[env_ids] = 0
    self._pulse_value[env_ids] = 0.0
    self._release_countdown[env_ids] = -1
    self._released[env_ids] = False
    self.centerline_error[env_ids] = 0.0
    self.heading_error[env_ids] = 0.0
    self.operator_correction[env_ids] = 0.0
    self.correction_active[env_ids] = False

  def _update_metrics(self) -> None:
    max_steps = max(
      1.0, self.cfg.resampling_time_range[1] / self._env.step_dt
    )
    self.metrics["error_vel_xy"] += torch.linalg.vector_norm(
      self.delivered_command[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
      dim=1,
    ) / max_steps
    self.metrics["error_vel_yaw"] += torch.abs(
      self.delivered_command[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
    ) / max_steps
    self.metrics["delay_steps"] += self.delay_steps.float() / max_steps
    self.metrics["abs_centerline_error"] += (
      torch.abs(self.centerline_error) / max_steps
    )
    self.metrics["abs_heading_error"] += torch.abs(self.heading_error) / max_steps
    self.metrics["operator_correction_fraction"] += (
      self.correction_active.float() / max_steps
    )

  def _centerline_feedback(self) -> torch.Tensor:
    """Compute a visual-feedback-like joystick correction on the GPU.

    The flat-patch centers define the stair centerline.  A look-ahead point is
    transformed to the base frame, exactly as a human would turn toward a
    visually selected point farther up the staircase.  Only the resulting
    joystick command is exposed to the actor; centerline geometry remains in
    the command generator and diagnostics.
    """
    env_ids = torch.arange(self.num_envs, device=self.device)
    patches = self._env_patches(env_ids)
    root_position = self.robot.data.root_link_pos_w
    center_y = patches[:, 0, 1]
    self.centerline_error[:] = root_position[:, 1] - center_y

    lookahead_w = torch.zeros_like(root_position)
    lookahead_w[:, 0] = self.cfg.centerline_lookahead_distance
    lookahead_w[:, 1] = center_y - root_position[:, 1]
    lookahead_b = quat_apply_inverse(
      self.robot.data.root_link_quat_w, lookahead_w
    )
    self.heading_error[:] = torch.atan2(lookahead_b[:, 1], lookahead_b[:, 0])
    correction = centerline_feedback_command(
      self.raw_command[:, 0],
      lookahead_b[:, 1],
      self.heading_error,
      lateral_gain=self.cfg.centerline_lateral_gain,
      heading_gain=self.cfg.centerline_heading_gain,
      lateral_deadband=self.cfg.centerline_lateral_deadband,
      heading_deadband=self.cfg.centerline_heading_deadband,
      max_lateral_velocity=self.cfg.centerline_max_lateral_velocity,
      max_yaw_velocity=self.cfg.centerline_max_yaw_velocity,
    )
    self.operator_correction[:] = correction
    self.operator_correction[:, 0] = 0.0
    self.correction_active[:] = (
      torch.abs(correction[:, 1]) > self.cfg.correction_epsilon
    ) | (torch.abs(correction[:, 2]) > self.cfg.correction_epsilon)
    return correction

  def _update_pulses(self) -> None:
    active = self._pulse_steps_left > 0
    self._pulse_steps_left[active] -= 1
    inactive = ~active
    self._next_pulse_steps[inactive] -= 1
    start = inactive & (self._next_pulse_steps <= 0) & ~self._released
    if torch.any(start):
      env_ids = start.nonzero(as_tuple=False).flatten()
      signs = torch.where(
        torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0
      )
      amp = torch.empty(len(env_ids), device=self.device).uniform_(
        *self.cfg.lateral_pulse_abs_range
      )
      yaw = torch.empty(len(env_ids), device=self.device).uniform_(
        *self.cfg.yaw_pulse_abs_range
      )
      self._pulse_value[env_ids, 1] = signs * amp
      self._pulse_value[env_ids, 2] = -signs * yaw
      self._pulse_steps_left[env_ids] = self._sample_integer_steps(
        env_ids, self.cfg.pulse_duration_range_s
      ).clamp_min(1)
      self._next_pulse_steps[env_ids] = self._sample_integer_steps(
        env_ids, self.cfg.pulse_interval_range_s
      )
    ended = (self._pulse_steps_left <= 0) & ~start
    self._pulse_value[ended, 1:] = 0.0

  def _update_release(self) -> None:
    env_ids = torch.arange(self.num_envs, device=self.device)
    patches = self._env_patches(env_ids)
    top_x = patches[:, -1, 0]
    reached = self.robot.data.root_link_pos_w[:, 0] >= (
      top_x - self.cfg.release_position_tolerance
    )
    trigger = reached & (self._release_countdown < 0) & ~self._released
    if torch.any(trigger):
      trigger_ids = trigger.nonzero(as_tuple=False).flatten()
      self._release_countdown[trigger_ids] = self._sample_integer_steps(
        trigger_ids, self.cfg.release_delay_range_s
      )
    counting = self._release_countdown > 0
    self._release_countdown[counting] -= 1
    release_now = (self._release_countdown == 0) & ~self._released
    self._released[release_now] = True
    self.raw_command[self._released] = 0.0
    self._pulse_value[self._released] = 0.0

  def _update_command(self) -> None:
    if self.cfg.closed_loop_centering:
      correction = self._centerline_feedback()
      disturbance = torch.zeros_like(correction)
      if self.cfg.disturbance_pulses_with_centering:
        self._update_pulses()
        disturbance[:, 1:] = self._pulse_value[:, 1:]
      self.raw_command[:, 1] = (
        correction[:, 1]
        + disturbance[:, 1]
        + self.cfg.fixed_lateral_bias
      )
      self.raw_command[:, 2] = (
        correction[:, 2] + disturbance[:, 2] + self.cfg.fixed_yaw_bias
      )
      self.operator_correction[:, 1:] = self.raw_command[:, 1:]
      self.correction_active[:] = (
        torch.abs(self.raw_command[:, 1]) > self.cfg.correction_epsilon
      ) | (torch.abs(self.raw_command[:, 2]) > self.cfg.correction_epsilon)
    else:
      # Preserve the original open-loop teleoperation benchmark.  It injects
      # random stick pulses but deliberately has no knowledge of drift.
      self._update_pulses()
      self.raw_command[:, 1] = (
        self._pulse_value[:, 1] + self.cfg.fixed_lateral_bias
      )
      self.raw_command[:, 2] = self._pulse_value[:, 2] + self.cfg.fixed_yaw_bias
      self._centerline_feedback()
      self.operator_correction[:, 1:] = self.raw_command[:, 1:]
      self.correction_active[:] = (
        (self._pulse_steps_left > 0)
        | (abs(self.cfg.fixed_lateral_bias) > self.cfg.correction_epsilon)
        | (abs(self.cfg.fixed_yaw_bias) > self.cfg.correction_epsilon)
      )
    self._update_release()

    previous = self.delivered_command.clone()
    if self._max_delay_steps > 0:
      self._delay_queue[:, 1:] = self._delay_queue[:, :-1].clone()
    self._delay_queue[:, 0] = self.raw_command
    batch = torch.arange(self.num_envs, device=self.device)
    delayed = self._delay_queue[batch, self.delay_steps]
    if self.cfg.low_pass_time_constant_s > 0.0:
      retain = math.exp(
        -self._env.step_dt / self.cfg.low_pass_time_constant_s
      )
      self.delivered_command[:] = retain * previous + (1.0 - retain) * delayed
    else:
      self.delivered_command[:] = delayed
    self.command_derivative[:] = (
      self.delivered_command - previous
    ) / self._env.step_dt


@dataclass(kw_only=True)
class JoystickVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  patch_name: str = "stair_targets"
  forward_velocity_range: tuple[float, float] = (0.30, 0.50)
  lateral_pulse_abs_range: tuple[float, float] = (0.02, 0.08)
  yaw_pulse_abs_range: tuple[float, float] = (0.05, 0.20)
  pulse_interval_range_s: tuple[float, float] = (3.0, 7.0)
  pulse_duration_range_s: tuple[float, float] = (0.20, 0.60)
  command_delay_range_s: tuple[float, float] = (0.04, 0.16)
  low_pass_time_constant_s: float = 0.08
  release_delay_range_s: tuple[float, float] = (0.10, 0.40)
  release_position_tolerance: float = 0.15
  closed_loop_centering: bool = False
  disturbance_pulses_with_centering: bool = False
  fixed_lateral_bias: float = 0.0
  fixed_yaw_bias: float = 0.0
  stair_half_width: float = 1.20
  centerline_lookahead_distance: float = 1.00
  centerline_lateral_gain: float = 0.80
  centerline_heading_gain: float = 1.40
  centerline_lateral_deadband: float = 0.04
  centerline_heading_deadband: float = 0.03
  centerline_max_lateral_velocity: float = 0.16
  centerline_max_yaw_velocity: float = 0.45
  correction_epsilon: float = 1.0e-5

  def build(self, env: "ManagerBasedRlEnv") -> JoystickVelocityCommand:
    return JoystickVelocityCommand(self, env)
