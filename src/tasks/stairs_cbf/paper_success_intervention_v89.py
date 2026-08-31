"""Low-noise success-conditioned imitation of deterministic CBF corrections."""

from __future__ import annotations

from typing import Any

import torch

from .paper_success_imitation_v88 import PaperSuccessImitationV88PPO
from .paper_teacher_v35 import PaperMeanTeacherV35PPO
from .teacher_v30_math import weighted_action_errors

METHOD_ID = "paper-cbf-success-intervention-safe-mean-imitation-v89"


class PaperSuccessInterventionV89PPO(PaperSuccessImitationV88PPO):
  """Clone only deterministic safe-mean corrections from successful episodes."""

  def _compute_teacher_labels(
    self, correction_norm: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    _, _, diagnostics = PaperMeanTeacherV35PPO._compute_teacher_labels(
      self, correction_norm
    )
    eligible = (
      self.v35_success_episode_transition & self.v35_mean_intervened
    ).detach().clone()
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

  def relabel_teacher_transitions(self) -> dict[str, Any]:
    metrics = super().relabel_teacher_transitions()
    target = self.v35_safe_policy_means.detach()
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
    metrics.update(
      {
        "v88_target_source": "same_state_deterministic_safe_policy_mean",
        "v88_eligibility": (
          "complete_reached_top_episode_and_deterministic_mean_intervention"
        ),
        "v88_success_transition_count": eligible_count,
        "v88_success_transition_fraction": float(
          self.teacher_eligible.float().mean()
        ),
        "v89_method_id": METHOD_ID,
        "v89_target_source": "same_state_deterministic_safe_policy_mean",
        "v89_eligibility": (
          "complete_reached_top_episode_and_deterministic_mean_intervention"
        ),
        "v89_success_intervention_transition_count": eligible_count,
        "v89_success_intervention_transition_fraction": float(
          self.teacher_eligible.float().mean()
        ),
        "v89_exploration_noise_target_removed": True,
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
    result.update(
      {
        "v89_method_id": METHOD_ID,
        "v89_target_source": "same_state_deterministic_safe_policy_mean",
        "v89_exploration_noise_target_removed": True,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v89_target_source": "same_state_deterministic_safe_policy_mean",
        "v89_exploration_noise_target_removed": True,
      }
    )
    return output
