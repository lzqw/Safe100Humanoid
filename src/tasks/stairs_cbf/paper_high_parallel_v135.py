"""Higher-parallelism paper PPO from the fixed v129 continuation point."""

from __future__ import annotations

import math
from typing import Any

from .paper_continuous_kl_v129 import PaperContinuousKlV129PPO

METHOD_ID = "paper-cbf-dual-high-parallel-v135"
V129_SELECTED_CHECKPOINT_SHA256 = (
  "71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8"
)
TRAINING_ACTION_STD = 0.05
INITIAL_ACTOR_LEARNING_RATE = 2.5e-6
REFERENCE_NUM_ENVS = 64
NUM_ENVS = 192
ROLLOUT_STEPS = 1024
ROUNDS = 8


def high_parallel_scale_diagnostics(
  *,
  num_envs: int = NUM_ENVS,
  rollout_steps: int = ROLLOUT_STEPS,
  rounds: int = ROUNDS,
  reference_num_envs: int = REFERENCE_NUM_ENVS,
) -> dict[str, int | float]:
  """Record the exact synchronous scale relative to the v129 protocol."""
  if min(num_envs, rollout_steps, rounds, reference_num_envs) < 1:
    raise ValueError("v135 scale inputs must be positive")
  transition_count = num_envs * rollout_steps * rounds
  reference_transition_count = reference_num_envs * rollout_steps * rounds
  return {
    "v135_num_envs": num_envs,
    "v135_rollout_steps": rollout_steps,
    "v135_rounds": rounds,
    "v135_transition_count": transition_count,
    "v135_reference_transition_count": reference_transition_count,
    "v135_parallel_scale_ratio": num_envs / reference_num_envs,
    "v135_transition_scale_ratio": (
      transition_count / reference_transition_count
    ),
    "v135_sequential_update_count_changed_from_v129": False,
  }


class PaperHighParallelV135PPO(PaperContinuousKlV129PPO):
  """Use unchanged paper PPO dynamics with a larger synchronous batch."""

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
      raise ValueError("v135 requires the paper-aligned frozen 0.05 action std")

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v135_method_id": METHOD_ID,
        "v135_high_parallel_paper_ppo": True,
        **high_parallel_scale_diagnostics(),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v135_high_parallel_paper_ppo": True,
        **high_parallel_scale_diagnostics(),
      }
    )
    return output
