"""Deployment-majority mixed continuation of the v138 clearance policy."""

from __future__ import annotations

import math
from typing import Any

from .paper_clearance_margin_v138 import (
  CLEARANCE_MARGIN_M,
  PaperClearanceMarginV138PPO,
)

METHOD_ID = "paper-cbf-dual-clearance-mixed-v139"
V138_SELECTED_CHECKPOINT_SHA256 = (
  "7a3899c515d5afd93f79f4db251feab4cd59f003e7150711e506ef5850604c63"
)
FILTER_ON_FRACTION = 0.25
FILTER_OFF_FRACTION = 0.75
NUM_ENVS = 128
ROLLOUT_STEPS = 1024
ROUNDS = 4
INITIAL_ACTOR_LEARNING_RATE = 2.5e-6
TRAINING_ACTION_STD = 0.05


def mixed_clearance_diagnostics(
  filter_on_fraction: float = FILTER_ON_FRACTION,
) -> dict[str, Any]:
  """Describe the fixed v79-style mixture used to align v138 to deployment."""
  if not math.isclose(
    filter_on_fraction, FILTER_ON_FRACTION, rel_tol=0.0, abs_tol=1.0e-12
  ):
    raise ValueError("v139 requires exactly 25% filter-on execution")
  return {
    "method_id": METHOD_ID,
    "filter_on_fraction": FILTER_ON_FRACTION,
    "filter_off_fraction": FILTER_OFF_FRACTION,
    "filter_group_balanced_advantages": True,
    "checkpoint_selection_group": "filter_off",
    "clearance_margin_m": CLEARANCE_MARGIN_M,
    "clearance_margin_changed_from_v138": False,
    "cbf_changed_from_v138": False,
    "actor_observation_changed_from_v138": False,
  }


class PaperClearanceMixedV139PPO(PaperClearanceMarginV138PPO):
  """Keep v138 PPO/reward while identifying its mixed rollout distribution."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v138_paper_clearance_margin": False,
        "v139_method_id": METHOD_ID,
        "v139_paper_clearance_mixed": True,
        **mixed_clearance_diagnostics(),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v138_paper_clearance_margin": False,
        "v139_paper_clearance_mixed": True,
        **mixed_clearance_diagnostics(),
      }
    )
    return output
