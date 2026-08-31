"""Safety-filter withdrawal consolidation after paper-style CBF-RL."""

from __future__ import annotations

import math
from typing import Any

import torch

from .paper_early_start_v128 import aligned_filtered_rollout_decision
from .teacher_v30 import CbfTeacherV30PPO

METHOD_ID = "paper-cbf-dual-shield-withdrawal-consolidation-v130"
V129_SELECTED_CHECKPOINT_SHA256 = (
  "71c5df2ed5bdfcca01eb6ee4302f0116bdca2482a2bd04486f5e239383a33ec8"
)
MINIMUM_FILTER_OFF_SELECTION_EPISODES = 64


def withdrawal_deployment_rollout_decision(
  *,
  candidate_round: int,
  runtime_filter_fraction: float,
  filter_off_success_count: int,
  filter_off_episode_count: int,
  filter_off_mean_reached_riser: float | None,
  incumbent_round: int | None,
  incumbent_success_count: int | None,
  incumbent_episode_count: int | None,
  incumbent_mean_reached_riser: float | None,
  minimum_episode_count: int = MINIMUM_FILTER_OFF_SELECTION_EPISODES,
) -> dict[str, Any]:
  """Select only sufficiently populated deployment-distribution rollouts."""
  if (
    candidate_round < 1
    or not math.isfinite(runtime_filter_fraction)
    or not 0.0 <= runtime_filter_fraction <= 1.0
    or minimum_episode_count < 1
    or filter_off_episode_count < 0
    or not 0 <= filter_off_success_count <= filter_off_episode_count
  ):
    raise ValueError("v130 withdrawal selection inputs are outside their domains")
  if filter_off_episode_count < minimum_episode_count:
    return {
      "eligible": False,
      "selected": False,
      "reason": "insufficient_filter_off_episodes",
      "candidate_rollout_round": candidate_round,
      "candidate_runtime_filter_fraction": runtime_filter_fraction,
      "candidate_filter_off_success_count": filter_off_success_count,
      "candidate_filter_off_episode_count": filter_off_episode_count,
      "minimum_filter_off_episode_count": minimum_episode_count,
      "selection_uses_training_rollout_only": True,
      "selection_changes_training_trajectory": False,
    }
  if filter_off_mean_reached_riser is None:
    raise ValueError("v130 eligible rollout must report filter-off stair progress")

  decision = aligned_filtered_rollout_decision(
    candidate_round=candidate_round,
    success_count=filter_off_success_count,
    episode_count=filter_off_episode_count,
    mean_reached_riser=float(filter_off_mean_reached_riser),
    incumbent_round=incumbent_round,
    incumbent_success_count=incumbent_success_count,
    incumbent_episode_count=incumbent_episode_count,
    incumbent_mean_reached_riser=incumbent_mean_reached_riser,
  )
  return {
    **decision,
    "eligible": True,
    "candidate_runtime_filter_fraction": runtime_filter_fraction,
    "candidate_filter_off_success_count": filter_off_success_count,
    "candidate_filter_off_episode_count": filter_off_episode_count,
    "minimum_filter_off_episode_count": minimum_episode_count,
    "selection_group": "filter_off",
  }


class PaperShieldWithdrawalV130PPO(CbfTeacherV30PPO):
  """Continuous standard PPO while the executed safety filter is withdrawn."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.teacher_mode != "none" or self.teacher_distillation_weight != 0.0:
      raise ValueError("v130 requires teacher-free A0 paper-dual PPO")
    if self.num_learning_epochs != 2 or self.num_mini_batches != 4:
      raise ValueError("v130 requires two PPO epochs and four minibatches")
    if self.moving_kl_beta != 0.0:
      raise ValueError("v130 requires no continuation KL loss")
    if not isinstance(self.actor_optimizer, torch.optim.Adam):
      raise ValueError("v130 requires the standard PPO Adam actor optimizer")

  def update(self) -> dict[str, Any]:
    result = super().update()
    if result["actor_optimizer_updates_completed"] != 8:
      raise RuntimeError("v130 did not complete all eight PPO updates")
    result.update(
      {
        "v130_method_id": METHOD_ID,
        "v130_filter_withdrawal_training": True,
        "v130_counterfactual_cbf_reward_retained": True,
        "v130_training_trajectory_continuous": True,
        "v130_transactional_rollback_disabled": True,
        "actor_optimizer_name": "adam",
        "actor_optimizer_updates_per_round": 8,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v130_filter_withdrawal_training": True,
        "v130_counterfactual_cbf_reward_retained": True,
        "v130_training_trajectory_continuous": True,
        "v130_transactional_rollback_disabled": True,
        "v130_actor_optimizer": "adam",
        "v130_actor_optimizer_updates_per_round": 8,
      }
    )
    return output
