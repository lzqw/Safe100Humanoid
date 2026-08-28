"""Second scaled stage of paper-style safety-filtered PPO."""

from __future__ import annotations

from typing import Any

from .paper_scaled_continuation_v132 import (
  PaperScaledContinuationV132PPO,
  paper_scale_diagnostics,
)

METHOD_ID = "paper-cbf-dual-scaled-stage-two-v133"
V132_SELECTED_CHECKPOINT_SHA256 = (
  "a7fdd4d07dc79f1f001b09ff3638bcc5de000c3804f369dcf13ed61ebd18bde3"
)
STAGE_INDEX = 2


def stage_two_scale_diagnostics() -> dict[str, int | float]:
  """Record the stage and cumulative scaled-continuation sample budget."""
  diagnostics = paper_scale_diagnostics()
  stage_transition_count = int(diagnostics["v132_transition_count"])
  return {
    "v133_stage_index": STAGE_INDEX,
    "v133_stage_transition_count": stage_transition_count,
    "v133_scaled_stage_cumulative_transition_count": (
      STAGE_INDEX * stage_transition_count
    ),
    "v133_objective_changed_from_v132": False,
    "v133_optimizer_protocol_changed_from_v132": False,
  }


class PaperScaledStageTwoV133PPO(PaperScaledContinuationV132PPO):
  """Identify a second unchanged 128-environment v132 continuation stage."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v133_method_id": METHOD_ID,
        "v133_scaled_stage_two": True,
        **stage_two_scale_diagnostics(),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v133_scaled_stage_two": True,
        **stage_two_scale_diagnostics(),
      }
    )
    return output
