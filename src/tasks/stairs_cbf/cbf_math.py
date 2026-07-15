"""Pure-torch control-barrier helpers used by the stair action term."""

from __future__ import annotations

import torch


def dual_cbf_reward(
  nominal_margin: torch.Tensor,
  intervention_norm: torch.Tensor,
  active: torch.Tensor,
  sigma: float,
) -> torch.Tensor:
  """Bounded CBF-RL reward from paper Eq. (23)."""
  if sigma <= 0.0:
    raise ValueError(f"sigma must be positive, got {sigma}")
  violation_reward = torch.minimum(
    nominal_margin, torch.zeros_like(nominal_margin)
  )
  imitation_reward = torch.exp(-intervention_norm.square() / sigma**2) - 1.0
  return torch.where(active, violation_reward + imitation_reward, 0.0)


def project_halfspace(
  nominal: torch.Tensor,
  normal: torch.Tensor,
  rhs: torch.Tensor,
  active: torch.Tensor | None = None,
  eps: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Project batched vectors onto ``normal @ x >= rhs``.

  This is the closed-form Euclidean QP used by CBF-RL for a single affine
  barrier constraint. Inactive rows are returned unchanged.
  """
  if nominal.shape != normal.shape:
    raise ValueError(f"shape mismatch: nominal={nominal.shape}, normal={normal.shape}")
  if rhs.shape != nominal.shape[:-1]:
    raise ValueError(f"rhs must have shape {nominal.shape[:-1]}, got {rhs.shape}")

  margin = torch.sum(normal * nominal, dim=-1) - rhs
  violated = margin < 0.0
  if active is not None:
    violated &= active
  denominator = torch.sum(normal.square(), dim=-1).clamp_min(eps)
  multiplier = torch.where(violated, -margin / denominator, torch.zeros_like(margin))
  projected = nominal + multiplier.unsqueeze(-1) * normal
  projected_margin = torch.sum(normal * projected, dim=-1) - rhs
  return projected, margin, projected_margin


def stair_barrier(
  foot_x: torch.Tensor,
  edge_x: torch.Tensor,
  toe_margin: float,
) -> torch.Tensor:
  """Signed distance to a stair riser; positive is on the safe side."""
  return edge_x - (foot_x + toe_margin)


def next_riser(
  foot_x: torch.Tensor,
  origin_x: torch.Tensor,
  first_riser_offset: float,
  step_width: float,
  step_height: float,
  num_steps: int,
  origin_z: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return next forward riser index, x plane, top z and in-range mask."""
  relative = foot_x - origin_x - first_riser_offset
  # Subtract a tiny tolerance so a foot exactly on a representable riser plane
  # is not advanced to the following riser by float32 round-off.
  index = torch.ceil(relative / step_width - 1.0e-6).to(torch.long)
  index = index.clamp(0, num_steps - 1)
  edge_x = origin_x + first_riser_offset + index.to(foot_x.dtype) * step_width
  base_z = torch.zeros_like(foot_x) if origin_z is None else origin_z
  top_z = base_z + (index.to(foot_x.dtype) + 1.0) * step_height
  in_range = foot_x < origin_x + first_riser_offset + num_steps * step_width
  return index, edge_x, top_z, in_range
