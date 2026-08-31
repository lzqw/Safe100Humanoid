"""Fully safety-filtered, direction-preserving PPO for paper-style CBF-RL."""

from __future__ import annotations

from typing import Any

from .paper_full_batch_v72 import PaperFullBatchV72PPO

METHOD_ID = "paper-cbf-dual-full-filter-full-batch-sgd-v75"


class PaperFullFilterV75PPO(PaperFullBatchV72PPO):
  """Identify the paper-faithful 100% filtered execution variant."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v75_method_id": METHOD_ID,
        "paper_training_execution_fully_safety_filtered": True,
        "paper_ppo_storage_uses_nominal_policy_action": True,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v75_training_execution_fully_safety_filtered": True,
        "v75_ppo_storage_uses_nominal_policy_action": True,
      }
    )
    return output
