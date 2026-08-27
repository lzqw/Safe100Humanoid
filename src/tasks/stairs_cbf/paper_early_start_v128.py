"""Continuous early-start PPO for paper-style safety-filtered CBF-RL."""

from __future__ import annotations

import math
from typing import Any

import torch

from .teacher_v30 import CbfTeacherV30PPO

METHOD_ID = "paper-cbf-dual-early-start-continuous-ppo-v128"


def aligned_filtered_rollout_decision(
  *,
  candidate_round: int,
  success_count: int,
  episode_count: int,
  mean_reached_riser: float,
  incumbent_round: int | None,
  incumbent_success_count: int | None,
  incumbent_episode_count: int | None,
  incumbent_mean_reached_riser: float | None,
) -> dict[str, Any]:
  """Select an already-observed filtered rollout without another evaluation.

  Success rate is the primary task metric.  Mean stair progress breaks an
  exact rate tie, and the later checkpoint wins a remaining exact tie.  This
  selection never changes the continuous optimization trajectory.
  """
  if candidate_round < 1 or episode_count < 1:
    raise ValueError("v128 aligned rollout requires a positive round and episodes")
  if not 0 <= success_count <= episode_count:
    raise ValueError("v128 aligned rollout success count is outside its domain")
  if not math.isfinite(mean_reached_riser):
    raise ValueError("v128 aligned rollout stair progress must be finite")

  candidate_rate = success_count / episode_count
  incumbent_rate = None
  if incumbent_round is None:
    if any(
      value is not None
      for value in (
        incumbent_success_count,
        incumbent_episode_count,
        incumbent_mean_reached_riser,
      )
    ):
      raise ValueError("v128 empty incumbent must not contain partial metrics")
    selected = True
    reason = "first_aligned_rollout"
  else:
    if (
      incumbent_success_count is None
      or incumbent_episode_count is None
      or incumbent_mean_reached_riser is None
      or incumbent_round < 1
      or incumbent_episode_count < 1
      or not 0 <= incumbent_success_count <= incumbent_episode_count
      or not math.isfinite(incumbent_mean_reached_riser)
    ):
      raise ValueError("v128 incumbent aligned rollout metrics are incomplete")
    incumbent_rate = incumbent_success_count / incumbent_episode_count
    candidate_key = (
      candidate_rate,
      mean_reached_riser,
      candidate_round,
    )
    incumbent_key = (
      incumbent_rate,
      incumbent_mean_reached_riser,
      incumbent_round,
    )
    selected = candidate_key > incumbent_key
    if candidate_rate > incumbent_rate:
      reason = "higher_success_rate"
    elif candidate_rate == incumbent_rate and (
      mean_reached_riser > incumbent_mean_reached_riser
    ):
      reason = "equal_rate_higher_stair_progress"
    elif candidate_key > incumbent_key:
      reason = "equal_performance_later_checkpoint"
    else:
      reason = "not_better_than_incumbent"

  return {
    "selected": selected,
    "reason": reason,
    "candidate_rollout_round": candidate_round,
    "candidate_success_count": success_count,
    "candidate_episode_count": episode_count,
    "candidate_success_rate": candidate_rate,
    "candidate_mean_reached_riser": mean_reached_riser,
    "incumbent_rollout_round_before": incumbent_round,
    "incumbent_success_rate_before": incumbent_rate,
    "selection_uses_training_rollout_only": True,
    "selection_changes_training_trajectory": False,
  }


class PaperEarlyStartV128PPO(CbfTeacherV30PPO):
  """Paper-core PPO: two clipped Adam epochs and no continuation KL anchor."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.teacher_mode != "none" or self.teacher_distillation_weight != 0.0:
      raise ValueError("v128 requires teacher-free A0 paper-dual PPO")
    if self.num_learning_epochs != 2 or self.num_mini_batches != 4:
      raise ValueError("v128 requires two PPO epochs and four minibatches")
    if self.moving_kl_beta != 0.0:
      raise ValueError("v128 removes the historical continuation KL anchor")
    if not isinstance(self.actor_optimizer, torch.optim.Adam):
      raise ValueError("v128 requires the standard PPO Adam actor optimizer")

  def update(self) -> dict[str, Any]:
    result = super().update()
    if result["actor_optimizer_updates_completed"] != 8:
      raise RuntimeError("v128 did not complete all eight continuous PPO updates")
    result.update(
      {
        "v128_method_id": METHOD_ID,
        "paper_training_execution_fully_safety_filtered": True,
        "paper_ppo_storage_uses_nominal_policy_action": True,
        "paper_continuous_training_without_transactional_rollback": True,
        "paper_continuation_kl_anchor_disabled": True,
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
        "v128_training_execution_fully_safety_filtered": True,
        "v128_ppo_storage_uses_nominal_policy_action": True,
        "v128_continuous_training_without_transactional_rollback": True,
        "v128_continuation_kl_anchor_disabled": True,
        "v128_actor_optimizer": "adam",
        "v128_actor_optimizer_updates_per_round": 8,
      }
    )
    return output
