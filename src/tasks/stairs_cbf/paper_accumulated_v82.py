"""Memory-bounded full-rollout actor gradient for paper-style CBF-RL."""

from __future__ import annotations

from typing import Any

from .paper_full_batch_v72 import PaperFullBatchV72PPO

METHOD_ID = "paper-cbf-dual-accumulated-full-batch-sgd-v82"


class PaperAccumulatedV82PPO(PaperFullBatchV72PPO):
  """Accumulate equal chunks before the single clipped actor SGD step."""

  accumulate_actor_microbatch_gradients = True

  def update(self) -> dict[str, Any]:
    result = super().update()
    if result["actor_optimizer_updates_completed"] != 1:
      raise RuntimeError("v82 must execute exactly one actor optimizer step")
    if result["actor_minibatches_completed"] != self.num_mini_batches:
      raise RuntimeError("v82 did not backpropagate every actor gradient chunk")
    result.update(
      {
        "v82_method_id": METHOD_ID,
        "actor_full_rollout_mean_gradient": True,
        "actor_full_batch_materialized_at_once": False,
        "actor_gradient_chunks_per_optimizer_step": self.num_mini_batches,
        "actor_optimizer_updates_per_round": 1,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v82_actor_full_rollout_mean_gradient": True,
        "v82_actor_full_batch_materialized_at_once": False,
        "v82_actor_gradient_chunks_per_optimizer_step": self.num_mini_batches,
        "v82_actor_optimizer_updates_per_round": 1,
      }
    )
    return output
