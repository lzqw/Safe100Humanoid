"""Paper-style stair-clearance margin continuation for v138."""

from __future__ import annotations

import math
from typing import Any

from .paper_scaled_continuation_v132 import (
  PaperScaledContinuationV132PPO,
  paper_scale_diagnostics,
)

METHOD_ID = "paper-cbf-dual-clearance-margin-v138"
V132_SELECTED_CHECKPOINT_SHA256 = (
  "a7fdd4d07dc79f1f001b09ff3638bcc5de000c3804f369dcf13ed61ebd18bde3"
)
REFERENCE_CLEARANCE_MARGIN_M = 0.05
CLEARANCE_MARGIN_M = 0.08
NUM_ENVS = 128
ROLLOUT_STEPS = 1024
ROUNDS = 4
INITIAL_ACTOR_LEARNING_RATE = 2.5e-6
TRAINING_ACTION_STD = 0.05


def configure_paper_clearance_margin(env_cfg) -> dict[str, Any]:
  """Raise only the paper-described next-stair clearance reference margin."""
  clearance = env_cfg.rewards.get("foot_clearance")
  if clearance is None:
    raise RuntimeError("v138 requires the stair foot-clearance reward")
  params = clearance.params
  if params.get("reference_mode") != "next_riser":
    raise ValueError("v138 requires the persistent next-riser reference")
  if not math.isclose(
    float(params.get("height_above_tread", -1.0)),
    REFERENCE_CLEARANCE_MARGIN_M,
    rel_tol=0.0,
    abs_tol=1.0e-12,
  ):
    raise ValueError("v138 reference clearance margin differs from 5 cm")
  clearance.params = {**params, "height_above_tread": CLEARANCE_MARGIN_M}
  return {
    "method_id": METHOD_ID,
    "reference_mode": "next_riser",
    "reference_clearance_margin_m": REFERENCE_CLEARANCE_MARGIN_M,
    "training_clearance_margin_m": CLEARANCE_MARGIN_M,
    "clearance_margin_increase_m": (
      CLEARANCE_MARGIN_M - REFERENCE_CLEARANCE_MARGIN_M
    ),
    "cbf_changed": False,
    "actor_observation_changed": False,
    "only_reward_parameter_changed": "foot_clearance.height_above_tread",
  }


class PaperClearanceMarginV138PPO(PaperScaledContinuationV132PPO):
  """Keep the v132 PPO path while identifying the clearance-margin stage."""

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v132_scaled_paper_continuation": False,
        "v138_method_id": METHOD_ID,
        "v138_paper_clearance_margin": True,
        "v138_clearance_margin_m": CLEARANCE_MARGIN_M,
        **paper_scale_diagnostics(
          num_envs=NUM_ENVS,
          rollout_steps=ROLLOUT_STEPS,
          rounds=ROUNDS,
        ),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v132_scaled_paper_continuation": False,
        "v138_paper_clearance_margin": True,
        "v138_clearance_margin_m": CLEARANCE_MARGIN_M,
        **paper_scale_diagnostics(
          num_envs=NUM_ENVS,
          rollout_steps=ROLLOUT_STEPS,
          rounds=ROUNDS,
        ),
      }
    )
    return output
