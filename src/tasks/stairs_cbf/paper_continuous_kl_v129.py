"""KL-controlled continuous PPO for paper-style safety-filtered CBF-RL."""

from __future__ import annotations

import math
from typing import Any

from .paper_early_start_v128 import PaperEarlyStartV128PPO

METHOD_ID = "paper-cbf-dual-continuous-kl-controlled-ppo-v129"
V79_CHECKPOINT_SHA256 = (
  "9a1316b281b4ed17f7ef45c54290d78de96cf8314de4a581957ee1d161edd317"
)
TARGET_FORWARD_KL = 6.0e-4
MINIMUM_ACTOR_LEARNING_RATE = 5.0e-7
MAXIMUM_ACTOR_LEARNING_RATE = 5.0e-6
MINIMUM_ROUND_SCALE = 0.5
MAXIMUM_ROUND_SCALE = 1.25


def continuous_ppo_kl_learning_rate(
  current_learning_rate: float,
  observed_forward_kl: float,
  *,
  target_forward_kl: float = TARGET_FORWARD_KL,
  minimum_learning_rate: float = MINIMUM_ACTOR_LEARNING_RATE,
  maximum_learning_rate: float = MAXIMUM_ACTOR_LEARNING_RATE,
  minimum_round_scale: float = MINIMUM_ROUND_SCALE,
  maximum_round_scale: float = MAXIMUM_ROUND_SCALE,
) -> tuple[float, dict[str, Any]]:
  """Update only the next round's LR while preserving the actor trajectory."""
  values = (
    current_learning_rate,
    observed_forward_kl,
    target_forward_kl,
    minimum_learning_rate,
    maximum_learning_rate,
    minimum_round_scale,
    maximum_round_scale,
  )
  if not all(math.isfinite(value) for value in values):
    raise ValueError("v129 KL controller inputs must be finite")
  if not (
    minimum_learning_rate > 0.0
    and minimum_learning_rate <= current_learning_rate <= maximum_learning_rate
    and target_forward_kl > 0.0
    and observed_forward_kl >= 0.0
    and 0.0 < minimum_round_scale <= 1.0
    and 1.0 <= maximum_round_scale
  ):
    raise ValueError("v129 KL controller inputs are outside their domains")

  raw_scale = math.sqrt(
    target_forward_kl / max(observed_forward_kl, 1.0e-12)
  )
  bounded_scale = min(
    maximum_round_scale, max(minimum_round_scale, raw_scale)
  )
  next_learning_rate = min(
    maximum_learning_rate,
    max(minimum_learning_rate, current_learning_rate * bounded_scale),
  )
  effective_scale = next_learning_rate / current_learning_rate
  return next_learning_rate, {
    "v129_kl_controller_enabled": True,
    "v129_kl_controller_target_forward_kl": target_forward_kl,
    "v129_kl_controller_observed_forward_kl": observed_forward_kl,
    "v129_kl_controller_raw_scale": raw_scale,
    "v129_kl_controller_bounded_scale": bounded_scale,
    "v129_kl_controller_effective_scale": effective_scale,
    "v129_kl_controller_learning_rate_before": current_learning_rate,
    "v129_kl_controller_learning_rate_after": next_learning_rate,
    "v129_kl_controller_minimum_learning_rate": minimum_learning_rate,
    "v129_kl_controller_maximum_learning_rate": maximum_learning_rate,
    "v129_kl_controller_changes_actor_state": False,
    "v129_kl_controller_uses_rollback": False,
  }


class PaperContinuousKlV129PPO(PaperEarlyStartV128PPO):
  """Identify continuous paper PPO whose round LR is KL-controlled."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v129_method_id": METHOD_ID,
        "v129_target_forward_kl": TARGET_FORWARD_KL,
        "v129_training_trajectory_continuous": True,
        "v129_transactional_rollback_disabled": True,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v129_target_forward_kl": TARGET_FORWARD_KL,
        "v129_training_trajectory_continuous": True,
        "v129_transactional_rollback_disabled": True,
      }
    )
    return output
