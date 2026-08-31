"""Conservative transactional outcome credit for persistent-geometry CBF-RL."""

from __future__ import annotations

from typing import Any

from .paper_outcome_geometry_v106 import PaperOutcomeGeometryV106PPO

METHOD_ID = "conservative-transactional-outcome-geometry-v107"
CONSERVATIVE_OUTCOME_ADVANTAGE_WEIGHT = 0.5


class PaperOutcomeTransactionalV107PPO(PaperOutcomeGeometryV106PPO):
  """Keep more task GAE while an outer aligned rollout rejects regressions."""

  outcome_method_id = METHOD_ID
  outcome_advantage_weight = CONSERVATIVE_OUTCOME_ADVANTAGE_WEIGHT

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v107_method_id": METHOD_ID,
        "v107_conservative_outcome_advantage_weight": (
          CONSERVATIVE_OUTCOME_ADVANTAGE_WEIGHT
        ),
        "v107_requires_transactional_rollout_acceptance": True,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v107_conservative_outcome_advantage_weight": (
          CONSERVATIVE_OUTCOME_ADVANTAGE_WEIGHT
        ),
        "v107_transactional_acceptance_scope": "aligned filter-off rollout",
      }
    )
    return output
