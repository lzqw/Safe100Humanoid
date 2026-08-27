"""Bounded deterministic CBF residuals on successful intervention states."""

from __future__ import annotations

from typing import Any

import torch

from .paper_success_intervention_v89 import PaperSuccessInterventionV89PPO
from .teacher_v30_math import weighted_action_errors

METHOD_ID = "paper-cbf-success-intervention-bounded-residual-v90"


class PaperSuccessResidualV90PPO(PaperSuccessInterventionV89PPO):
  """Apply the A2 residual fraction instead of cloning the full safe mean."""

  def relabel_teacher_transitions(self) -> dict[str, Any]:
    metrics = super().relabel_teacher_transitions()
    full_correction = (
      self.v35_safe_policy_means - self.v35_policy_means
    ).detach()
    target = (
      self.v30_reference_means + float(self.teacher_eta) * full_correction
    ).detach()
    full_correction_norm = torch.linalg.vector_norm(full_correction, dim=-1)
    residual_norm = torch.linalg.vector_norm(
      target - self.v30_reference_means, dim=-1
    )
    self.v30_teacher_targets.copy_(target)
    self.v30_correction_vectors.copy_(full_correction)
    self.teacher_correction_norm.copy_(full_correction_norm)

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
        "v88_target_source": "bounded_deterministic_safe_mean_residual",
        "v89_target_source": "bounded_deterministic_safe_mean_residual",
        "v90_method_id": METHOD_ID,
        "v90_target_source": "bounded_deterministic_safe_mean_residual",
        "v90_residual_fraction": float(self.teacher_eta),
        "v90_full_safe_mean_cloning_disabled": True,
        "mean_residual_target_norm": float(
          residual_norm[self.teacher_eligible].mean()
        ) if eligible_count else 0.0,
        "mean_cbf_correction_norm": float(
          full_correction_norm[self.teacher_eligible].mean()
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
        "v90_method_id": METHOD_ID,
        "v90_target_source": "bounded_deterministic_safe_mean_residual",
        "v90_residual_fraction": float(self.teacher_eta),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v90_target_source": "bounded_deterministic_safe_mean_residual",
        "v90_residual_fraction": float(self.teacher_eta),
      }
    )
    return output
