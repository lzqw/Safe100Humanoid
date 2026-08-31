"""Success-conditioned imitation of actions executed through the CBF filter."""

from __future__ import annotations

import math
from typing import Any

import torch

from .paper_teacher_v35 import PaperMeanTeacherV35PPO
from .teacher_v30_math import (
  success_population_smooth_l1_loss,
  weighted_action_errors,
)

METHOD_ID = "paper-cbf-success-safe-action-imitation-v88"


class PaperSuccessImitationV88PPO(PaperMeanTeacherV35PPO):
  """Add a low-variance supervised signal from successful safe trajectories."""

  accumulate_actor_microbatch_gradients = True

  def __init__(
    self,
    *args,
    v88_success_imitation_weight: float = 0.5,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    requested_weight = float(v88_success_imitation_weight)
    if not math.isfinite(requested_weight) or not 0.0 < requested_weight <= 2.0:
      raise ValueError("v88 success-imitation weight must lie in (0, 2]")
    if self.num_learning_epochs != 1 or self.num_mini_batches < 2:
      raise ValueError("v88 requires one epoch and at least two gradient chunks")
    if any(
      (
        self.v35_failure_only_mean_teacher,
        self.v35_success_only_mean_teacher,
        self.v35_failure_focused_actor,
        self.v35_distill_only_actor,
        self.v35_split_filter_actor_objectives,
        self.v35_task_priority_gradient_surgery,
      )
    ) or self.v35_success_local_kl_beta != 0.0:
      raise ValueError("v88 success imitation is mutually exclusive with v35 gates")
    self.v88_success_imitation_weight = requested_weight
    self.teacher_distillation_weight = requested_weight

  def reset_proximal_optimizers(self) -> None:
    """Use one scalar SGD step after all actor gradient chunks."""
    super().reset_proximal_optimizers()
    actor_parameters = [
      parameter
      for parameter in self.actor.mlp.parameters()
      if parameter.requires_grad
    ]
    self.actor_optimizer = torch.optim.SGD(
      actor_parameters,
      lr=float(self.actor_learning_rate),
    )
    self.optimizer = self.actor_optimizer
    self.learning_rate = self.actor_learning_rate

  def _compute_teacher_labels(
    self, correction_norm: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    _, _, diagnostics = super()._compute_teacher_labels(correction_norm)
    eligible = self.v35_success_episode_transition.detach().clone()
    weights = eligible.to(correction_norm.dtype)
    zeros = torch.zeros_like(eligible)
    ones = torch.ones_like(eligible)
    return eligible, weights, {
      **diagnostics,
      "intervened": eligible,
      "crossed_within_horizon": zeros,
      "no_fall_within_horizon": ones,
      "no_recovery_takeover_within_horizon": ones,
      "no_emergency_termination_within_horizon": ones,
      "no_unsafe_termination_within_horizon": ones,
      "horizon_outcome_observed": ones,
      "terminal_observed_within_horizon": zeros,
      "magnitude_weight": weights,
    }

  def _teacher_loss(
    self,
    policy_mean: torch.Tensor,
    policy_std: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    weights: torch.Tensor,
  ) -> torch.Tensor:
    del policy_std, weights
    return success_population_smooth_l1_loss(
      policy_mean,
      target,
      eligible,
      beta=self.teacher_smooth_l1_beta,
    )

  def relabel_teacher_transitions(self) -> dict[str, Any]:
    metrics = super().relabel_teacher_transitions()
    target = self.teacher_policy_actions.detach()
    correction = (target - self.v30_reference_means).detach()
    correction_norm = torch.linalg.vector_norm(correction, dim=-1)
    self.v30_teacher_targets.copy_(target)
    self.v30_correction_vectors.copy_(correction)
    self.teacher_correction_norm.copy_(correction_norm)

    flat_eligible = self.teacher_eligible.flatten()
    flat_weights = self.teacher_weights.flatten()
    before_distance, before_per_action = weighted_action_errors(
      self.v30_reference_means.flatten(0, 1),
      target.flatten(0, 1),
      flat_eligible,
      flat_weights,
    )
    eligible_count = int(flat_eligible.sum())
    successful_intervention_count = int(
      (self.teacher_eligible & self.actual_cbf_intervened).sum()
    )
    metrics.update(
      {
        "v88_method_id": METHOD_ID,
        "v88_target_source": "executed_safe_sampled_action",
        "v88_eligibility": "complete_reached_top_episode_transitions",
        "v88_success_imitation_weight": self.v88_success_imitation_weight,
        "v88_loss_population_normalized": True,
        "v88_success_transition_count": eligible_count,
        "v88_success_transition_fraction": float(
          self.teacher_eligible.float().mean()
        ),
        "v88_success_intervention_transition_count": (
          successful_intervention_count
        ),
        "v88_success_intervention_fraction": (
          successful_intervention_count / eligible_count
          if eligible_count
          else 0.0
        ),
        "mean_residual_target_norm": float(
          correction_norm[self.teacher_eligible].mean()
        ) if eligible_count else 0.0,
        "mean_cbf_correction_norm": float(
          correction_norm[self.teacher_eligible].mean()
        ) if eligible_count else 0.0,
        "mean_policy_to_target_distance_before_update": float(before_distance),
        "per_action_teacher_error_before_update": [
          float(value) for value in before_per_action
        ],
        "teacher_target_shape": list(target.shape),
        "teacher_target_stop_gradient": not target.requires_grad,
      }
    )
    return metrics

  def update(self) -> dict[str, Any]:
    result = super().update()
    if result["actor_optimizer_updates_completed"] != 1:
      raise RuntimeError("v88 must execute exactly one actor optimizer step")
    result.update(
      {
        "v88_method_id": METHOD_ID,
        "v88_success_imitation_weight": self.v88_success_imitation_weight,
        "v88_actor_optimizer": "sgd",
        "v88_actor_optimizer_updates_per_round": 1,
        "v88_actor_gradient_chunks": self.num_mini_batches,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v88_success_imitation_weight": self.v88_success_imitation_weight,
        "v88_actor_optimizer": "sgd",
        "v88_actor_optimizer_updates_per_round": 1,
        "v88_actor_gradient_chunks": self.num_mini_batches,
      }
    )
    return output
