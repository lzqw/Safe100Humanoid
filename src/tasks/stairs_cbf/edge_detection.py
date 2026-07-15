"""Vectorized riser-edge extraction for ordered stair tread patches.

Hiking in the Wild obtains sharp edges from terrain geometry. Our staircase is
procedural and axis-aligned, so the same safety representation can be recovered
exactly (without a CPU mesh walk) from each tread's flat-patch center and run.
"""

from __future__ import annotations

import torch


def riser_edges_from_tread_patches(
  patches: torch.Tensor, step_width: float, num_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return riser x planes and top heights from ordered tread centers."""
  if patches.shape[-1] != 3:
    raise ValueError(f"patches must end in xyz, got {patches.shape}")
  if patches.shape[-2] < num_steps:
    raise ValueError(f"need {num_steps} tread patches, got {patches.shape[-2]}")
  treads = patches[..., :num_steps, :]
  return treads[..., 0] - 0.5 * step_width, treads[..., 2]


def select_active_riser(
  foot_x: torch.Tensor,
  foot_z: torch.Tensor,
  edge_x: torch.Tensor,
  edge_top_z: torch.Tensor,
  *,
  toe_margin: float,
  top_clearance: float,
  activation_distance: float,
  recovery_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Choose the closest relevant riser for each foot/environment."""
  h_all = edge_x - (foot_x.unsqueeze(-1) + toe_margin)
  below_top = foot_z.unsqueeze(-1) < edge_top_z + top_clearance
  candidate = (
    below_top
    & (h_all < activation_distance)
    & (h_all > -recovery_distance)
  )
  scores = torch.where(candidate, h_all.abs(), torch.full_like(h_all, torch.inf))
  index = scores.argmin(dim=-1)
  active = candidate.any(dim=-1)
  selected_h = h_all.gather(-1, index.unsqueeze(-1)).squeeze(-1)
  selected_top = edge_top_z.gather(-1, index.unsqueeze(-1)).squeeze(-1)
  return index, selected_h, selected_top, active
