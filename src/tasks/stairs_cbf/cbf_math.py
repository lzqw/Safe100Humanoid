"""Pure-torch control-barrier helpers used by the stair action term."""

from __future__ import annotations

import torch


def conditional_deployable_cbf_geometry(
  geometry: torch.Tensor,
) -> torch.Tensor:
  """Split 5-D toe geometry by swing side and barrier phase.

  The four mutually exclusive blocks are left/unsafe, left/safe,
  right/unsafe, and right/safe. Each block contains horizontal clearance,
  vertical clearance, sloped barrier, and its binary mask.
  """
  if geometry.ndim < 2 or geometry.shape[-1] != 5:
    raise ValueError("conditional CBF geometry requires a final width of five")
  active = geometry[..., 4] > 0.5
  left = geometry[..., 3] < 0.0
  right = geometry[..., 3] > 0.0
  unsafe = geometry[..., 2] < 0.0
  coordinates = geometry[..., :3]
  blocks: list[torch.Tensor] = []
  for selected in (
    active & left & unsafe,
    active & left & ~unsafe,
    active & right & unsafe,
    active & right & ~unsafe,
  ):
    mask = selected.to(geometry.dtype).unsqueeze(-1)
    blocks.append(torch.cat((coordinates * mask, mask), dim=-1))
  return torch.cat(blocks, dim=-1)


def persistent_next_riser_geometry(
  root_x: torch.Tensor,
  foot_xz: torch.Tensor,
  contact: torch.Tensor,
  edge_x: torch.Tensor,
  edge_top_z: torch.Tensor,
  *,
  toe_margin: float,
  top_clearance: float,
  barrier_slope: float,
  lookahead_distance: float,
  horizontal_scale: float,
  vertical_scale: float,
) -> torch.Tensor:
  """Return per-foot next-riser geometry before and throughout swing.

  Each foot contributes normalized horizontal clearance, vertical clearance,
  sloped barrier, contact state, and a lookahead-valid flag. Unlike the narrow
  runtime-CBF activation window, the signal starts while both feet are still
  planted so the policy can plan lift before toe-off.
  """
  if root_x.ndim != 1 or foot_xz.shape != (len(root_x), 2, 2):
    raise ValueError("persistent geometry requires root [N] and feet [N,2,2]")
  if contact.shape != (len(root_x), 2) or contact.dtype != torch.bool:
    raise ValueError("persistent geometry contact must be boolean [N,2]")
  if edge_x.ndim != 2 or edge_x.shape != edge_top_z.shape:
    raise ValueError("persistent geometry edges must share shape [N,R]")
  if edge_x.shape[0] != len(root_x) or edge_x.shape[1] < 1:
    raise ValueError("persistent geometry requires at least one riser per env")
  if min(lookahead_distance, horizontal_scale, vertical_scale) <= 0.0:
    raise ValueError("persistent geometry scales and lookahead must be positive")

  root_distance = edge_x - root_x.unsqueeze(1)
  ahead = root_distance >= 0.0
  masked = torch.where(ahead, root_distance, torch.full_like(root_distance, torch.inf))
  index = masked.argmin(dim=1)
  selected_distance = masked.gather(1, index.unsqueeze(1)).squeeze(1)
  valid = torch.isfinite(selected_distance) & (
    selected_distance <= float(lookahead_distance)
  )
  selected_x = edge_x.gather(1, index.unsqueeze(1)).squeeze(1)
  selected_z = edge_top_z.gather(1, index.unsqueeze(1)).squeeze(1)
  horizontal = selected_x.unsqueeze(1) - foot_xz[..., 0] - float(toe_margin)
  vertical = foot_xz[..., 1] - selected_z.unsqueeze(1) - float(top_clearance)
  barrier = (
    vertical + float(barrier_slope) * horizontal
    if barrier_slope > 0.0
    else horizontal
  )
  valid_float = valid.to(foot_xz.dtype).unsqueeze(1).expand(-1, 2)
  features = torch.stack(
    (
      (horizontal / float(horizontal_scale)).clamp(-1.5, 1.5),
      (vertical / float(vertical_scale)).clamp(-1.5, 1.5),
      (barrier / float(vertical_scale)).clamp(-2.0, 2.0),
      contact.to(foot_xz.dtype),
      valid_float,
    ),
    dim=-1,
  )
  return (features * valid_float.unsqueeze(-1)).reshape(len(root_x), 10)


def dual_cbf_reward(
  nominal_margin: torch.Tensor,
  intervention_norm: torch.Tensor,
  active: torch.Tensor,
  sigma: float,
  *,
  margin_weight: float = 1.0,
  intervention_weight: float = 1.0,
) -> torch.Tensor:
  """CBF-RL reward from paper Eq. (23), with explicit term weights.

  The paper writes a single outer weight, while its public navigation demo
  scales the negative-margin and filter-imitation terms independently. The
  unit defaults preserve the historical Safe100Humanoid reward exactly.
  """
  if sigma <= 0.0:
    raise ValueError(f"sigma must be positive, got {sigma}")
  if margin_weight < 0.0 or intervention_weight < 0.0:
    raise ValueError("CBF-RL reward weights must be non-negative")
  violation_reward = float(margin_weight) * torch.minimum(
    nominal_margin, torch.zeros_like(nominal_margin)
  )
  imitation_reward = float(intervention_weight) * (
    torch.exp(-intervention_norm.square() / sigma**2) - 1.0
  )
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


def next_riser_clearance_reference(
  root_x: torch.Tensor,
  origin_z: torch.Tensor,
  edge_x: torch.Tensor,
  edge_top_z: torch.Tensor,
  *,
  default_height: float,
  height_above_tread: float,
  lookahead_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return the paper-style foot-clearance reference for the stair ahead.

  The reference follows the first riser in front of the robot instead of the
  short-lived CBF activation flag.  Once the robot passes the final riser, the
  top-platform height remains the reference.  This keeps the target stable for
  the entire swing rather than dropping it back to the flat-ground height as
  soon as the toe clears the riser.
  """
  if root_x.ndim != 1 or origin_z.shape != root_x.shape:
    raise ValueError("root_x and origin_z must share shape [N]")
  if edge_x.ndim != 2 or edge_x.shape != edge_top_z.shape:
    raise ValueError("riser edges and heights must share shape [N, R]")
  if edge_x.shape[0] != root_x.shape[0] or edge_x.shape[1] < 1:
    raise ValueError("riser metadata must contain at least one edge per env")
  if default_height <= 0.0 or height_above_tread <= 0.0:
    raise ValueError("clearance heights must be positive")
  if lookahead_distance <= 0.0:
    raise ValueError("clearance lookahead distance must be positive")

  distance = edge_x - root_x.unsqueeze(1)
  ahead = distance >= 0.0
  masked_distance = torch.where(
    ahead, distance, torch.full_like(distance, torch.inf)
  )
  index = masked_distance.argmin(dim=1)
  selected_distance = masked_distance.gather(1, index.unsqueeze(1)).squeeze(1)
  selected_top = edge_top_z.gather(1, index.unsqueeze(1)).squeeze(1)
  within_lookahead = torch.isfinite(selected_distance) & (
    selected_distance <= float(lookahead_distance)
  )
  beyond_final_riser = root_x > edge_x[:, -1]
  selected_top = torch.where(
    beyond_final_riser, edge_top_z[:, -1], selected_top
  )
  active = within_lookahead | beyond_final_riser
  flat_reference = origin_z + float(default_height)
  stair_reference = selected_top + float(height_above_tread)
  reference = torch.where(active, stair_reference, flat_reference)
  index = torch.where(
    beyond_final_riser,
    torch.full_like(index, edge_x.shape[1] - 1),
    index,
  )
  return reference, active, index


def sloped_toe_clearance_constraint(
  horizontal_margin: torch.Tensor,
  foot_z: torch.Tensor,
  edge_top_z: torch.Tensor,
  jac_x: torch.Tensor,
  jac_z: torch.Tensor,
  *,
  top_clearance: float,
  slope: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return a task-compatible toe/riser barrier and its joint-space normal.

  The safe boundary is a ramp in the x-z plane.  Far from the riser, the toe
  may remain low; at the riser plane it must clear ``edge_top_z`` plus the
  fixed margin.  Its derivative couples vertical lift and forward motion:

  ``h_dot = (J_z - slope * J_x) q_dot``.
  """
  if horizontal_margin.shape != foot_z.shape or foot_z.shape != edge_top_z.shape:
    raise ValueError("sloped-clearance margins must share shape [N]")
  if jac_x.shape != jac_z.shape or jac_x.shape[:-1] != foot_z.shape:
    raise ValueError("sloped-clearance Jacobians must share shape [N, A]")
  if not 0.0 < float(slope) <= 2.0:
    raise ValueError("sloped-clearance slope must lie in (0, 2]")
  vertical_margin = foot_z - edge_top_z - float(top_clearance)
  barrier = vertical_margin + float(slope) * horizontal_margin
  normal = jac_z - float(slope) * jac_x
  return barrier, normal


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
