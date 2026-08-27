"""Pure acceptance and step-size rules for v73 transactional CBF-RL."""

from __future__ import annotations

import math
from typing import Any

TARGET_MOVING_FORWARD_KL = 1.0e-5
MINIMUM_ACTOR_LEARNING_RATE = 1.0e-6
REJECTED_CANDIDATE_LEARNING_RATE_SCALE = 0.5


def rollout_candidate_decision(
  *,
  actor_sha256: str,
  success_count: int,
  episode_count: int,
  accepted_actor_sha256: str | None,
  accepted_success_rate: float | None,
) -> dict[str, Any]:
  """Accept the baseline/retry or a candidate that is not worse than anchor."""
  if not actor_sha256:
    raise ValueError("candidate actor SHA-256 must be non-empty")
  if episode_count <= 0 or not 0 <= success_count <= episode_count:
    raise ValueError("candidate rollout counts are invalid")
  success_rate = success_count / episode_count
  if accepted_actor_sha256 is None:
    if accepted_success_rate is not None:
      raise ValueError("an empty accepted actor requires an empty accepted rate")
    return {
      "accepted": True,
      "replace_anchor": True,
      "reason": "initial_baseline",
      "success_rate": success_rate,
      "improvement_percentage_points": None,
    }
  if accepted_success_rate is None or not 0.0 <= accepted_success_rate <= 1.0:
    raise ValueError("the accepted actor requires a finite success rate")
  if actor_sha256 == accepted_actor_sha256:
    return {
      "accepted": True,
      "replace_anchor": False,
      "reason": "accepted_actor_retry_after_rollback",
      "success_rate": success_rate,
      "improvement_percentage_points": 0.0,
    }
  improvement = 100.0 * (success_rate - accepted_success_rate)
  accepted = success_rate >= accepted_success_rate
  return {
    "accepted": accepted,
    "replace_anchor": accepted,
    "reason": (
      "candidate_noninferior_to_anchor"
      if accepted
      else "candidate_filter_off_regression"
    ),
    "success_rate": success_rate,
    "improvement_percentage_points": improvement,
  }


def adaptive_actor_learning_rate(
  current_learning_rate: float,
  moving_forward_kl: float,
  *,
  rejected: bool,
  target_kl: float = TARGET_MOVING_FORWARD_KL,
  minimum_learning_rate: float = MINIMUM_ACTOR_LEARNING_RATE,
) -> float:
  """Only shrink SGD, using KL geometry and an extra rejection penalty."""
  values = (
    current_learning_rate,
    moving_forward_kl,
    target_kl,
    minimum_learning_rate,
  )
  if not all(math.isfinite(value) for value in values):
    raise ValueError("transactional learning-rate inputs must be finite")
  if current_learning_rate <= 0.0 or moving_forward_kl < 0.0:
    raise ValueError("learning rate must be positive and KL non-negative")
  if target_kl <= 0.0 or minimum_learning_rate <= 0.0:
    raise ValueError("target KL and minimum learning rate must be positive")
  if moving_forward_kl == 0.0:
    kl_scale = 1.0
  else:
    kl_scale = min(1.0, max(0.5, math.sqrt(target_kl / moving_forward_kl)))
  rejection_scale = (
    REJECTED_CANDIDATE_LEARNING_RATE_SCALE if rejected else 1.0
  )
  return max(
    minimum_learning_rate,
    current_learning_rate * kl_scale * rejection_scale,
  )
