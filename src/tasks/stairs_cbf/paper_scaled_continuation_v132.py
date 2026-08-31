"""Larger-batch continuation of paper-style safety-filtered PPO."""

from __future__ import annotations

import math
from typing import Any

from .paper_continuous_kl_v129 import PaperContinuousKlV129PPO

METHOD_ID = "paper-cbf-dual-scaled-continuation-v132"
TRAINING_ACTION_STD = 0.05
INITIAL_ACTOR_LEARNING_RATE = 2.5e-6
REFERENCE_NUM_ENVS = 64
NUM_ENVS = 128
ROLLOUT_STEPS = 1024
ROUNDS = 8


def paper_scale_diagnostics(
  *,
  num_envs: int = NUM_ENVS,
  rollout_steps: int = ROLLOUT_STEPS,
  rounds: int = ROUNDS,
  reference_num_envs: int = REFERENCE_NUM_ENVS,
) -> dict[str, int | float]:
  """Record the exact bounded scale-up relative to the v129 run."""
  if min(num_envs, rollout_steps, rounds, reference_num_envs) < 1:
    raise ValueError("v132 scale inputs must be positive")
  transitions = num_envs * rollout_steps * rounds
  reference_transitions = reference_num_envs * rollout_steps * rounds
  return {
    "v132_num_envs": num_envs,
    "v132_rollout_steps": rollout_steps,
    "v132_rounds": rounds,
    "v132_transition_count": transitions,
    "v132_reference_transition_count": reference_transitions,
    "v132_parallel_scale_ratio": num_envs / reference_num_envs,
    "v132_transition_scale_ratio": transitions / reference_transitions,
  }


class PaperScaledContinuationV132PPO(PaperContinuousKlV129PPO):
  """Use unchanged v129 PPO dynamics under the larger rollout protocol."""

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
      raise ValueError("v132 requires the paper-aligned frozen 0.05 action std")

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v132_method_id": METHOD_ID,
        "v132_scaled_paper_continuation": True,
        **paper_scale_diagnostics(),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v132_scaled_paper_continuation": True,
        **paper_scale_diagnostics(),
      }
    )
    return output
