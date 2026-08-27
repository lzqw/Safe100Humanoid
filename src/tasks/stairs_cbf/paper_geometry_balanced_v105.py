"""Persistent-geometry gradient balancing for full paper-dual PPO."""

from __future__ import annotations

from typing import Any

import torch

from .paper_full_batch_v72 import PaperFullBatchV72PPO

METHOD_ID = "persistent-geometry-gradient-balanced-paper-dual-v105"
PERSISTENT_GEOMETRY_FEATURE_COUNT = 10
TARGET_GEOMETRY_TO_LEGACY_GRADIENT_NORM_RATIO = 1.0
MAX_GEOMETRY_GRADIENT_SCALE = 32.0


def balance_persistent_geometry_gradient(
  gradient: torch.Tensor,
  *,
  feature_count: int = PERSISTENT_GEOMETRY_FEATURE_COUNT,
  target_ratio: float = TARGET_GEOMETRY_TO_LEGACY_GRADIENT_NORM_RATIO,
  max_scale: float = MAX_GEOMETRY_GRADIENT_SCALE,
) -> tuple[torch.Tensor, dict[str, float]]:
  """Raise the new input block to a bounded share of first-layer gradient."""
  if gradient.ndim != 2 or not 0 < feature_count < gradient.shape[1]:
    raise ValueError("v105 requires a 2-D first-layer gradient with a legacy prefix")
  legacy = gradient[:, :-feature_count]
  geometry = gradient[:, -feature_count:]
  legacy_norm = float(torch.linalg.vector_norm(legacy))
  geometry_norm = float(torch.linalg.vector_norm(geometry))
  if legacy_norm > 0.0 and geometry_norm > 0.0:
    requested_scale = target_ratio * legacy_norm / geometry_norm
    applied_scale = min(max_scale, max(1.0, requested_scale))
  else:
    requested_scale = 1.0
    applied_scale = 1.0
  balanced = gradient.clone()
  balanced[:, -feature_count:].mul_(applied_scale)
  scaled_geometry_norm = geometry_norm * applied_scale
  return balanced, {
    "geometry_gradient_legacy_block_norm": legacy_norm,
    "geometry_gradient_raw_block_norm": geometry_norm,
    "geometry_gradient_requested_scale": requested_scale,
    "geometry_gradient_applied_scale": applied_scale,
    "geometry_gradient_scaled_block_norm": scaled_geometry_norm,
    "geometry_gradient_scaled_to_legacy_ratio": (
      scaled_geometry_norm / legacy_norm if legacy_norm > 0.0 else 0.0
    ),
  }


class PaperGeometryBalancedV105PPO(PaperFullBatchV72PPO):
  """Train the full actor while preventing its 10-D geometry path starvation."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if int(self.actor.obs_dim) != 415:
      raise ValueError("v105 requires the 415-D persistent-geometry actor")
    first_layer = next(
      module for module in self.actor.mlp if isinstance(module, torch.nn.Linear)
    )
    if int(first_layer.in_features) != 415:
      raise ValueError("v105 first actor layer must have exactly 415 inputs")
    self._v105_gradient_metrics: dict[str, float] = {}
    self._v105_gradient_hook = first_layer.weight.register_hook(
      self._balance_first_layer_gradient
    )

  def _balance_first_layer_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
    balanced, metrics = balance_persistent_geometry_gradient(gradient)
    self._v105_gradient_metrics = metrics
    return balanced

  def update(self) -> dict[str, Any]:
    self._v105_gradient_metrics = {}
    result = super().update()
    if not self._v105_gradient_metrics:
      raise RuntimeError("v105 actor update did not observe a first-layer gradient")
    result.update(
      {
        "v105_method_id": METHOD_ID,
        "geometry_gradient_feature_count": PERSISTENT_GEOMETRY_FEATURE_COUNT,
        "geometry_gradient_target_ratio": (
          TARGET_GEOMETRY_TO_LEGACY_GRADIENT_NORM_RATIO
        ),
        "geometry_gradient_max_scale": MAX_GEOMETRY_GRADIENT_SCALE,
        **self._v105_gradient_metrics,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v105_geometry_gradient_feature_count": (
          PERSISTENT_GEOMETRY_FEATURE_COUNT
        ),
        "v105_geometry_gradient_target_ratio": (
          TARGET_GEOMETRY_TO_LEGACY_GRADIENT_NORM_RATIO
        ),
        "v105_geometry_gradient_max_scale": MAX_GEOMETRY_GRADIENT_SCALE,
      }
    )
    return output
