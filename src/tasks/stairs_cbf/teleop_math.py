"""Pure-torch helpers for the simulated human stair operator."""

from __future__ import annotations

import torch


def signed_deadband(value: torch.Tensor, deadband: float) -> torch.Tensor:
  """Remove a symmetric deadband while preserving continuity and sign."""
  if deadband < 0.0:
    raise ValueError(f"deadband must be non-negative, got {deadband}")
  return torch.sign(value) * torch.relu(torch.abs(value) - deadband)


def centerline_feedback_command(
  forward_velocity: torch.Tensor,
  lateral_error_body: torch.Tensor,
  heading_error: torch.Tensor,
  *,
  lateral_gain: float,
  heading_gain: float,
  lateral_deadband: float,
  heading_deadband: float,
  max_lateral_velocity: float,
  max_yaw_velocity: float,
) -> torch.Tensor:
  """Return ``[vx, vy, wz]`` for a bounded centerline feedback operator.

  The caller supplies errors in the robot base frame.  Positive lateral error
  means that the stair centerline lies to the robot's left; positive heading
  error means that the look-ahead centerline target lies counter-clockwise.
  The forward stick is passed through unchanged while the simulated operator
  closes the lateral and heading loops.
  """
  if not (
    forward_velocity.shape == lateral_error_body.shape == heading_error.shape
  ):
    raise ValueError("forward velocity and feedback errors must have equal shapes")
  if lateral_gain < 0.0 or heading_gain < 0.0:
    raise ValueError("feedback gains must be non-negative")
  if max_lateral_velocity < 0.0 or max_yaw_velocity < 0.0:
    raise ValueError("feedback limits must be non-negative")

  lateral = lateral_gain * signed_deadband(
    lateral_error_body, lateral_deadband
  )
  yaw = heading_gain * signed_deadband(heading_error, heading_deadband)
  return torch.stack(
    (
      forward_velocity,
      torch.clamp(lateral, -max_lateral_velocity, max_lateral_velocity),
      torch.clamp(yaw, -max_yaw_velocity, max_yaw_velocity),
    ),
    dim=-1,
  )
