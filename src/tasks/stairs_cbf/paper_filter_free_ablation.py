"""Frozen filter-free deployment ablation starting from the v139 actor."""

from __future__ import annotations

import math
from typing import Any

from .paper_clearance_margin_v138 import (
  CLEARANCE_MARGIN_M,
  PaperClearanceMarginV138PPO,
)
from .paper_scaled_continuation_v132 import paper_scale_diagnostics

METHOD_ID = "paper-cbf-filter-free-deployment-ablation"
V139_SELECTED_CHECKPOINT_SHA256 = (
  "323f1e00b58d379b8746c0191a44272f2e1df134139050417c56e733cc484728"
)
CONTEXTS = ("F1", "F2", "F3")
FROZEN_ARM = "frozen"
ADAPTATION_ARMS = (
  "nominal_ft",
  "reward_only_ft",
  "filter_only_ft",
  "dual_safe_ft",
)
ALL_ARMS = (FROZEN_ARM, *ADAPTATION_ARMS)
TRAINING_SEEDS = (201357000, 201357001, 201357002)
EVALUATION_SEED = 201357900
NUM_ENVS = 128
ROLLOUT_STEPS = 1024
ROUNDS = 4
CHECKPOINT_ROUNDS = (0, 1, 2, 4)
PAIRED_EVALUATION_EPISODES = 512
INITIAL_ACTOR_LEARNING_RATE = 2.5e-6
TRAINING_ACTION_STD = 0.05

_ARM_VARIABLES = {
  "nominal_ft": (False, False),
  "reward_only_ft": (False, True),
  "filter_only_ft": (True, False),
  "dual_safe_ft": (True, True),
}


def arm_variables(arm: str) -> dict[str, bool]:
  """Return the two deliberately varied factors for one adaptation arm."""
  if arm not in _ARM_VARIABLES:
    raise ValueError(f"unsupported filter-free adaptation arm {arm!r}")
  runtime_filter, cbf_reward = _ARM_VARIABLES[arm]
  return {
    "runtime_filter_during_adaptation": runtime_filter,
    "cbf_reward_during_adaptation": cbf_reward,
  }


def filter_free_ablation_diagnostics(
  *,
  arm: str,
  context: str | None,
) -> dict[str, Any]:
  """Describe the prospective five-group filter-free deployment protocol."""
  if arm not in ALL_ARMS:
    raise ValueError(f"unsupported filter-free ablation arm {arm!r}")
  if context is not None and context not in CONTEXTS:
    raise ValueError(f"unsupported stair context {context!r}")

  frozen = arm == FROZEN_ARM
  variables = (
    {
      "runtime_filter_during_adaptation": None,
      "cbf_reward_during_adaptation": None,
    }
    if frozen
    else arm_variables(arm)
  )
  return {
    "method_id": METHOD_ID,
    "arm": arm,
    "context": context,
    "frozen_no_update": frozen,
    **variables,
    "expected_base_checkpoint_sha256": V139_SELECTED_CHECKPOINT_SHA256,
    "same_actor_critic_ppo_and_budget": True,
    "ppo_storage_action": "raw_nominal_policy_action",
    "clearance_margin_m": CLEARANCE_MARGIN_M,
    "rounds": 0 if frozen else ROUNDS,
    "num_envs": NUM_ENVS,
    "rollout_steps": ROLLOUT_STEPS,
    "transition_count": 0 if frozen else NUM_ENVS * ROLLOUT_STEPS * ROUNDS,
    "training_action_std": TRAINING_ACTION_STD,
    "checkpoint_rounds": [0] if frozen else list(CHECKPOINT_ROUNDS),
    "checkpoint_candidate_selection": False,
    "primary_checkpoint_rule": "frozen_round_0" if frozen else "fixed_round_4",
    "primary_checkpoint_round": 0 if frozen else ROUNDS,
    "paired_evaluation_episodes_per_filter_condition": (
      PAIRED_EVALUATION_EPISODES
    ),
    "paired_evaluation_filter_conditions": ["off", "on"],
    "paired_initial_conditions": True,
    "primary_metric": "post_adaptation_cbf_off_success_rate",
    "primary_improvement": "post_off_minus_pre_off_success_rate",
    "shield_gap": "cbf_on_success_rate_minus_cbf_off_success_rate",
    "final_deployment_filter": "off",
    "actor_observation_changed_from_v139": False,
    "cbf_changed_from_v139": False,
    "clearance_reward_changed_from_v139": False,
    "training_seeds": list(TRAINING_SEEDS),
    "evaluation_seed": EVALUATION_SEED,
  }


class _PaperFilterFreeAblationPPO(PaperClearanceMarginV138PPO):
  """Shared PPO implementation; subclasses identify the frozen arm only."""

  ARM = ""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.ARM not in ADAPTATION_ARMS:
      raise ValueError(f"invalid filter-free PPO arm {self.ARM!r}")
    if not (
      math.isclose(
        self.minimum_std,
        TRAINING_ACTION_STD,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      )
      and math.isclose(
        self.maximum_std,
        TRAINING_ACTION_STD,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      )
    ):
      raise ValueError("filter-free ablation requires frozen std 0.05")

  def _metadata(self) -> dict[str, Any]:
    return {
      "proximal_method_id": f"{METHOD_ID}-{self.ARM}",
      "v138_paper_clearance_margin": False,
      "paper_filter_free_ablation": True,
      **paper_scale_diagnostics(
        num_envs=NUM_ENVS,
        rollout_steps=ROLLOUT_STEPS,
        rounds=ROUNDS,
      ),
      **filter_free_ablation_diagnostics(arm=self.ARM, context=None),
    }

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(self._metadata())
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(self._metadata())
    return output


class PaperNominalFilterFreePPO(_PaperFilterFreeAblationPPO):
  ARM = "nominal_ft"


class PaperRewardOnlyFilterFreePPO(_PaperFilterFreeAblationPPO):
  ARM = "reward_only_ft"


class PaperFilterOnlyFilterFreePPO(_PaperFilterFreeAblationPPO):
  ARM = "filter_only_ft"


class PaperDualFilterFreePPO(_PaperFilterFreeAblationPPO):
  ARM = "dual_safe_ft"
