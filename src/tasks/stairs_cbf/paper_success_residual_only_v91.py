"""Teacher-only actor update from successful bounded CBF residuals."""

from __future__ import annotations

from typing import Any

import torch

from .paper_success_residual_v90 import PaperSuccessResidualV90PPO

METHOD_ID = "paper-cbf-success-bounded-residual-only-actor-v91"


class PaperSuccessResidualOnlyV91PPO(PaperSuccessResidualV90PPO):
  """Remove noisy PPO/entropy gradients while keeping critic learning intact."""

  def _actor_ppo_transition_mask(self) -> torch.Tensor:
    return torch.zeros_like(self.teacher_eligible, dtype=torch.bool)

  def update(self) -> dict[str, Any]:
    result = super().update()
    if result["actor_ppo_transition_count"] != 0:
      raise RuntimeError("v91 actor must not receive PPO/entropy gradients")
    result.update(
      {
        "v91_method_id": METHOD_ID,
        "v91_actor_objective": "success_intervention_residual_only",
        "v91_actor_ppo_and_entropy_gradients_disabled": True,
        "v91_critic_task_and_cbf_learning_retained": True,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v91_actor_objective": "success_intervention_residual_only",
        "v91_actor_ppo_and_entropy_gradients_disabled": True,
        "v91_critic_task_and_cbf_learning_retained": True,
      }
    )
    return output
