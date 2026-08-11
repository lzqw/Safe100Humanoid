"""Environment-only context adapter for the independent v23 path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .deployment_context import (
  V22_CONTEXT_KIND,
  apply_frozen_deployment_context,
  validate_frozen_deployment_context,
)


def apply_cbf_proximal_context(
  cfg,
  payload: Mapping[str, Any],
  *,
  role: str = "target",
) -> dict[str, Any]:
  """Retain v22 physics but remove its later learning-only adapter.

  The historical context applicator remains byte-for-byte unchanged.  On a
  fresh configuration we reuse its calibrated geometry, command disturbance,
  actuation and physical parameters, then remove the five-dimensional
  observation group and the specialist reward term before environment
  construction.  Consequently neither object exists at v23 runtime.
  """
  validated = validate_frozen_deployment_context(payload)
  if validated.get("kind") != V22_CONTEXT_KIND:
    raise ValueError("CBF-proximal refinement requires a calibrated v22 context")
  metadata = apply_frozen_deployment_context(
    cfg, validated, role=role
  )
  removed_observation = cfg.observations.pop("deployable_failure", None)
  removed_reward = cfg.rewards.pop("specialist_failure_signal", None)
  if removed_observation is None or removed_reward is None:
    raise RuntimeError("v22 context did not expose its removable learning adapter")
  if "deployable_failure" in cfg.observations:
    raise RuntimeError("CBF-proximal context retained a failure observation")
  if "specialist_failure_signal" in cfg.rewards:
    raise RuntimeError("CBF-proximal context retained specialist reward shaping")
  metadata["actor_context_fields_added"] = 0
  metadata["cbf_proximal_interface"] = {
    "original_observation_interface": True,
    "deployable_failure_group_absent": True,
    "specialist_reward_term_absent": True,
    "historical_adapter_removed_before_environment_construction": True,
  }
  return metadata
