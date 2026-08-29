"""Frozen controls and CBF-protected adaptation after the v139 initializer."""

from __future__ import annotations

import math
from typing import Any

from .paper_clearance_margin_v138 import (
  CLEARANCE_MARGIN_M,
  PaperClearanceMarginV138PPO,
)
from .paper_scaled_continuation_v132 import paper_scale_diagnostics

CONTROL_METHOD_ID = "paper-cbf-deployment-alignment-full-filter-control"
SAFE_ADAPTATION_METHOD_ID = "paper-cbf-protected-online-adaptation"
V138_SELECTED_CHECKPOINT_SHA256 = (
  "7a3899c515d5afd93f79f4db251feab4cd59f003e7150711e506ef5850604c63"
)
V139_SELECTED_CHECKPOINT_SHA256 = (
  "323f1e00b58d379b8746c0191a44272f2e1df134139050417c56e733cc484728"
)
NUM_ENVS = 128
ROLLOUT_STEPS = 1024
CONTROL_ROUNDS = 4
SAFE_ADAPTATION_ROUNDS = 2
INITIAL_ACTOR_LEARNING_RATE = 2.5e-6
TRAINING_ACTION_STD = 0.05
FULL_FILTER_FRACTION = 1.0
CONTROL_PRIMARY_CHECKPOINT_ROUND = 1


def deployment_pipeline_diagnostics(
  *,
  stage: str,
  context: str | None,
  filter_on_fraction: float = FULL_FILTER_FRACTION,
) -> dict[str, Any]:
  """Return the frozen protocol for the control or protected adaptation."""
  if stage not in ("full_filter_control", "safe_online_adaptation"):
    raise ValueError(f"unsupported deployment-pipeline stage {stage!r}")
  if context is not None and context not in ("F1", "F2", "F3"):
    raise ValueError(f"unsupported stair context {context!r}")
  if stage == "full_filter_control" and context != "F2":
    raise ValueError("the paired full-filter control is fixed to F2")
  if not math.isclose(
    filter_on_fraction,
    FULL_FILTER_FRACTION,
    rel_tol=0.0,
    abs_tol=1.0e-12,
  ):
    raise ValueError("real/sim safe adaptation requires 100% CBF execution")

  safe_adaptation = stage == "safe_online_adaptation"
  rounds = SAFE_ADAPTATION_ROUNDS if safe_adaptation else CONTROL_ROUNDS
  return {
    "method_id": (
      SAFE_ADAPTATION_METHOD_ID if safe_adaptation else CONTROL_METHOD_ID
    ),
    "stage": stage,
    "context": context,
    "expected_base_checkpoint_sha256": (
      V139_SELECTED_CHECKPOINT_SHA256
      if safe_adaptation
      else V138_SELECTED_CHECKPOINT_SHA256
    ),
    "filter_on_fraction": FULL_FILTER_FRACTION,
    "filter_off_fraction": 0.0,
    "all_executed_actions_cbf_filtered": True,
    "ppo_storage_action": "raw_nominal_policy_action",
    "clearance_margin_m": CLEARANCE_MARGIN_M,
    "rounds": rounds,
    "num_envs": NUM_ENVS,
    "rollout_steps": ROLLOUT_STEPS,
    "transition_count": NUM_ENVS * ROLLOUT_STEPS * rounds,
    "checkpoint_candidate_selection": False,
    "primary_checkpoint_rule": (
      "round_2_final_actor"
      if safe_adaptation
      else "round_1_actor_matched_to_v139_published_actor"
    ),
    "primary_checkpoint_round": (
      SAFE_ADAPTATION_ROUNDS
      if safe_adaptation
      else CONTROL_PRIMARY_CHECKPOINT_ROUND
    ),
    "actor_observation_changed_from_v139": False,
    "cbf_changed_from_v139": False,
  }


class PaperDeploymentAlignmentControlPPO(PaperClearanceMarginV138PPO):
  """Matched 100%-filter control for the v139 mixed-execution refinement."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v138_paper_clearance_margin": False,
        "deployment_alignment_control": True,
        **paper_scale_diagnostics(
          num_envs=NUM_ENVS,
          rollout_steps=ROLLOUT_STEPS,
          rounds=CONTROL_ROUNDS,
        ),
        **deployment_pipeline_diagnostics(
          stage="full_filter_control", context="F2"
        ),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": CONTROL_METHOD_ID,
        "v138_paper_clearance_margin": False,
        "deployment_alignment_control": True,
        **paper_scale_diagnostics(
          num_envs=NUM_ENVS,
          rollout_steps=ROLLOUT_STEPS,
          rounds=CONTROL_ROUNDS,
        ),
        **deployment_pipeline_diagnostics(
          stage="full_filter_control", context="F2"
        ),
      }
    )
    return output


class PaperSafeOnlineAdaptationPPO(PaperClearanceMarginV138PPO):
  """Two-round, fully CBF-protected PPO adaptation from frozen v139."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v138_paper_clearance_margin": False,
        "safe_online_adaptation": True,
        **paper_scale_diagnostics(
          num_envs=NUM_ENVS,
          rollout_steps=ROLLOUT_STEPS,
          rounds=SAFE_ADAPTATION_ROUNDS,
        ),
        **deployment_pipeline_diagnostics(
          stage="safe_online_adaptation", context=None
        ),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": SAFE_ADAPTATION_METHOD_ID,
        "v138_paper_clearance_margin": False,
        "safe_online_adaptation": True,
        **paper_scale_diagnostics(
          num_envs=NUM_ENVS,
          rollout_steps=ROLLOUT_STEPS,
          rounds=SAFE_ADAPTATION_ROUNDS,
        ),
        **deployment_pipeline_diagnostics(
          stage="safe_online_adaptation", context=None
        ),
      }
    )
    return output
