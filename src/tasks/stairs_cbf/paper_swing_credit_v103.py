"""Bounded full-swing temporal safety credit for paper-style CBF-RL."""

from __future__ import annotations

from typing import Any

from .paper_full_batch_v72 import PaperFullBatchV72PPO

METHOD_ID = "paper-cbf-dual-bounded-swing-credit-v103"


class PaperSwingCreditV103PPO(PaperFullBatchV72PPO):
  """Add one bounded 1 s look-back safety signal without frame accumulation."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(
      *args,
      allow_bounded_temporal_credit=True,
      **kwargs,
    )

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v103_method_id": METHOD_ID,
        "bounded_swing_credit": True,
        "swing_credit_aggregation": self.pre_intervention_aggregation,
        "swing_credit_horizon_steps": self.pre_intervention_horizon,
        "swing_credit_decay": self.pre_intervention_decay,
        "swing_credit_weight": self.pre_intervention_weight,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v103_bounded_swing_credit": True,
        "v103_swing_credit_aggregation": self.pre_intervention_aggregation,
        "v103_swing_credit_horizon_steps": self.pre_intervention_horizon,
        "v103_swing_credit_decay": self.pre_intervention_decay,
        "v103_swing_credit_weight": self.pre_intervention_weight,
      }
    )
    return output
