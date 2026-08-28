"""Low-noise continuation toward deterministic paper-style deployment."""

from __future__ import annotations

import math
from typing import Any

from .paper_continuous_kl_v129 import PaperContinuousKlV129PPO

METHOD_ID = "paper-cbf-dual-deterministic-aligned-low-noise-v131"
REFERENCE_TRAINING_ACTION_STD = 0.05
TRAINING_ACTION_STD = 0.03
INITIAL_ACTOR_LEARNING_RATE = 2.5e-6


def deterministic_alignment_diagnostics(
  training_action_std: float = TRAINING_ACTION_STD,
  *,
  reference_action_std: float = REFERENCE_TRAINING_ACTION_STD,
) -> dict[str, float]:
  """Quantify the exploration reduction relative to the v129 protocol."""
  if not (
    math.isfinite(training_action_std)
    and math.isfinite(reference_action_std)
    and 0.0 < training_action_std <= reference_action_std
  ):
    raise ValueError("v131 action standard deviations are outside their domains")
  standard_deviation_ratio = training_action_std / reference_action_std
  return {
    "v131_training_action_std": training_action_std,
    "v131_reference_action_std": reference_action_std,
    "v131_standard_deviation_ratio": standard_deviation_ratio,
    "v131_exploration_variance_ratio": standard_deviation_ratio**2,
    "v131_equal_kl_mean_shift_ratio": standard_deviation_ratio,
  }


class PaperDeterministicAlignedV131PPO(PaperContinuousKlV129PPO):
  """Keep the v129 PPO path while narrowing its frozen Gaussian sampler."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if not (
      math.isclose(
        self.minimum_std, TRAINING_ACTION_STD, rel_tol=0.0, abs_tol=1.0e-12
      )
      and math.isclose(
        self.maximum_std, TRAINING_ACTION_STD, rel_tol=0.0, abs_tol=1.0e-12
      )
    ):
      raise ValueError("v131 requires a frozen 0.03 training action std")

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v131_method_id": METHOD_ID,
        "v131_deterministic_aligned_low_noise": True,
        **deterministic_alignment_diagnostics(),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v131_deterministic_aligned_low_noise": True,
        **deterministic_alignment_diagnostics(),
      }
    )
    return output
