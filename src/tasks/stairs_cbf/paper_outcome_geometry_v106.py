"""Full-episode outcome credit for persistent-geometry paper-dual PPO."""

from __future__ import annotations

from typing import Any

import torch

from .paper_dual_v35 import normalize_filter_group_advantages
from .paper_geometry_balanced_v105 import PaperGeometryBalancedV105PPO
from .teacher_v30_math import (
  disjoint_terminal_outcomes,
  episode_balanced_outcome_advantage,
)

METHOD_ID = "outcome-centered-persistent-geometry-paper-dual-v106"
OUTCOME_ADVANTAGE_WEIGHT = 1.0


class PaperOutcomeGeometryV106PPO(PaperGeometryBalancedV105PPO):
  """Blend unit-scale complete-episode outcome credit with paper-dual GAE."""

  outcome_method_id = METHOD_ID
  outcome_advantage_weight = OUTCOME_ADVANTAGE_WEIGHT

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    t = self.storage.num_transitions_per_env
    n = self.storage.num_envs
    self.v106_success_terminals = torch.zeros(
      t, n, dtype=torch.bool, device=self.device
    )

  def process_env_step(
    self,
    obs,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    step = self.storage.step
    if step < self.storage.num_transitions_per_env:
      reached_top = extras.get("v106_reached_top")
      if reached_top is None:
        raise RuntimeError("v106 exact reached-top telemetry is missing")
      self.v106_success_terminals[step].copy_(
        dones.bool() & reached_top.bool()
      )
    super().process_env_step(obs, rewards, dones, extras)

  def apply_outcome_centered_episode_advantage(
    self, filter_mask: torch.Tensor
  ) -> dict[str, float]:
    """Replace actor advantages with normalized GAE + normalized outcome credit."""
    failed_terminal, successful_terminal, joint_terminal = (
      disjoint_terminal_outcomes(
        self.storage.dones.squeeze(-1).bool(),
        self.fall_events.bool(),
        self.v106_success_terminals,
      )
    )
    outcome_credit, outcome_metrics = episode_balanced_outcome_advantage(
      self.teacher_episode_ids,
      successful_terminal,
      failed_terminal,
      filter_mask.to(self.device),
    )
    base_advantage = self.storage.advantages.squeeze(-1)
    balanced_base, base_metrics = normalize_filter_group_advantages(
      base_advantage, filter_mask.to(base_advantage.device)
    )
    outcome_weight = float(self.outcome_advantage_weight)
    combined = balanced_base + outcome_weight * outcome_credit
    balanced_combined, combined_metrics = normalize_filter_group_advantages(
      combined, filter_mask.to(combined.device)
    )
    self.storage.advantages.copy_(balanced_combined.unsqueeze(-1))

    applied = outcome_credit != 0.0
    if bool(applied.any()):
      base_values = balanced_base[applied]
      outcome_values = outcome_credit[applied]
      denominator = torch.linalg.vector_norm(base_values) * torch.linalg.vector_norm(
        outcome_values
      )
      cosine = float(
        torch.dot(base_values, outcome_values) / denominator.clamp_min(1.0e-12)
      )
    else:
      cosine = 0.0
    return {
      **outcome_metrics,
      **{f"outcome_base_{key}": value for key, value in base_metrics.items()},
      **combined_metrics,
      "outcome_method_id": self.outcome_method_id,
      "outcome_advantage_weight": outcome_weight,
      "outcome_joint_success_fall_terminal_count": float(joint_terminal.sum()),
      "outcome_credit_applied_transition_count": float(applied.sum()),
      "outcome_credit_applied_transition_fraction": float(applied.float().mean()),
      "outcome_credit_mean": float(outcome_credit.mean()),
      "outcome_credit_std": float(outcome_credit.std(unbiased=False)),
      "outcome_base_gae_credit_cosine": cosine,
      "outcome_combined_advantage_std_before_final_normalization": float(
        combined.std(unbiased=False)
      ),
      "outcome_combined_advantage_mean_after": float(balanced_combined.mean()),
      "outcome_combined_advantage_std_after": float(
        balanced_combined.std(unbiased=False)
      ),
    }

  def clear_cbf_rollout(self) -> None:
    super().clear_cbf_rollout()
    if hasattr(self, "v106_success_terminals"):
      self.v106_success_terminals.zero_()

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": self.outcome_method_id,
        "v106_outcome_advantage_weight": float(
          self.outcome_advantage_weight
        ),
        "v106_outcome_credit": (
          "episode-equal centered success/failure within each filter group"
        ),
        "v106_advantage_composition": "unit GAE + unit outcome, renormalized",
      }
    )
    return output
