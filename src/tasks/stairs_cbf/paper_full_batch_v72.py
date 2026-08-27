"""Direction-preserving full-batch actor update for paper-style CBF-RL."""

from __future__ import annotations

from typing import Any

import torch

from .teacher_v30 import CbfTeacherV30PPO

METHOD_ID = "paper-cbf-dual-full-batch-sgd-v72"


class PaperFullBatchV72PPO(CbfTeacherV30PPO):
  """Use one globally clipped SGD actor step instead of eight Adam steps."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.teacher_mode != "none" or self.teacher_distillation_weight != 0.0:
      raise ValueError("v72 full-batch SGD requires the teacher-free A0 arm")
    if self.num_learning_epochs != 1 or self.num_mini_batches != 1:
      raise ValueError("v72 requires exactly one full-batch actor update")
    if not 1.0e-6 <= float(self.actor_learning_rate) <= 1.0e-3:
      raise ValueError("v72 SGD learning rate must lie in [1e-6, 1e-3]")

  def reset_proximal_optimizers(self) -> None:
    """Keep the existing critic Adam but make the actor step scalar SGD."""
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

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v72_method_id": METHOD_ID,
        "actor_optimizer_name": "sgd",
        "actor_optimizer_preserves_global_gradient_direction": True,
        "actor_full_batch_update": True,
        "actor_optimizer_updates_per_round": 1,
        "actor_sgd_learning_rate": float(self.actor_learning_rate),
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v72_actor_optimizer": "sgd",
        "v72_actor_full_batch_update": True,
        "v72_actor_optimizer_updates_per_round": 1,
        "v72_actor_sgd_learning_rate": float(self.actor_learning_rate),
      }
    )
    return output
