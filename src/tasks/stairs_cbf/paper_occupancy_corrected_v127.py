"""State-value occupancy correction for paper-style filtered PPO rollouts.

The CBF-RL paper executes the safety-filtered action during training, while
the selected policy is evaluated without that filter.  This module keeps the
paper's filtered PPO credit, but estimates ``d_off(s) / d_on(s)`` from a
simultaneous nominal rollout cohort.  Only filtered transitions update the
actor; the critic continues to use every transition through the inherited
implementation.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .paper_accumulated_v82 import PaperAccumulatedV82PPO


METHOD_ID = "crossfit-state-value-occupancy-corrected-paper-dual-v127"


def crossfit_occupancy_corrected_advantages(
  features: torch.Tensor,
  advantages: torch.Tensor,
  filter_mask: torch.Tensor,
  *,
  folds: int = 2,
  density_ratio_limit: float = 4.0,
  variance_ridge: float = 1.0e-3,
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
  """Importance-weight filtered GAE toward the filter-off state occupancy.

  A diagonal shared-covariance discriminator is fitted out of fold by whole
  environment, so no transition receives a density ratio from a model that
  saw the same trajectory.  The density ratio is clipped and self-normalized.
  The filter-off cohort is used only to estimate occupancy: its actor
  advantages are exactly zero.  Scaling the retained group by its inverse
  population fraction preserves the full-population PPO gradient scale.
  """
  if features.ndim != 3 or advantages.ndim != 2:
    raise ValueError("v127 features/advantages must have [T,N,D]/[T,N] shape")
  if features.shape[:2] != advantages.shape or features.shape[-1] < 1:
    raise ValueError("v127 feature and advantage layouts differ")
  if filter_mask.shape != advantages.shape[1:] or filter_mask.dtype != torch.bool:
    raise ValueError("v127 filter mask must be boolean with shape [N]")
  if folds < 2 or density_ratio_limit <= 1.0 or variance_ridge <= 0.0:
    raise ValueError("v127 occupancy hyperparameters are outside their domains")
  if not bool(torch.isfinite(features).all() and torch.isfinite(advantages).all()):
    raise RuntimeError("v127 occupancy inputs contain non-finite values")

  on_ids = filter_mask.nonzero(as_tuple=False).flatten()
  off_ids = (~filter_mask).nonzero(as_tuple=False).flatten()
  if len(on_ids) < 2 * folds or len(off_ids) < 2 * folds:
    raise ValueError("v127 needs at least two environments per class and fold")

  logits = torch.empty_like(advantages)
  on_fold = torch.arange(len(on_ids), device=features.device) % folds
  off_fold = torch.arange(len(off_ids), device=features.device) % folds
  log_limit = math.log(float(density_ratio_limit))
  fold_separations: list[float] = []

  # A shared diagonal covariance keeps the fit deterministic and makes the
  # method cheap enough to run between rollout collection and one PPO step.
  for fold in range(folds):
    fit_on = on_ids[on_fold != fold]
    fit_off = off_ids[off_fold != fold]
    held_on = on_ids[on_fold == fold]
    held_off = off_ids[off_fold == fold]
    if not len(fit_on) or not len(fit_off) or not len(held_on) or not len(held_off):
      raise ValueError("v127 cross-fit fold contains an empty class")
    on_values = features[:, fit_on].flatten(0, 1).float()
    off_values = features[:, fit_off].flatten(0, 1).float()
    mean_on = on_values.mean(dim=0)
    mean_off = off_values.mean(dim=0)
    pooled_variance = 0.5 * (
      on_values.var(dim=0, unbiased=False)
      + off_values.var(dim=0, unbiased=False)
    )
    ridge = float(variance_ridge) * pooled_variance.mean().clamp_min(1.0)
    direction = (mean_off - mean_on) / (pooled_variance + ridge)
    midpoint = 0.5 * (mean_off + mean_on)

    held_on_logits = torch.einsum(
      "tnd,d->tn", features[:, held_on].float() - midpoint, direction
    )
    held_off_logits = torch.einsum(
      "tnd,d->tn", features[:, held_off].float() - midpoint, direction
    )
    logits[:, held_on] = held_on_logits.to(logits.dtype)
    logits[:, held_off] = held_off_logits.to(logits.dtype)
    fold_separations.append(float(held_off_logits.mean() - held_on_logits.mean()))

  on_logits = logits[:, filter_mask]
  off_logits = logits[:, ~filter_mask]
  true_positive_rate = float((off_logits > 0.0).float().mean())
  true_negative_rate = float((on_logits < 0.0).float().mean())
  balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
  separation = float(off_logits.mean() - on_logits.mean())
  correction_active = bool(separation > 0.0 and balanced_accuracy >= 0.52)

  if correction_active:
    weights = torch.exp(torch.clamp(on_logits, -log_limit, log_limit))
    weights = weights / weights.mean().clamp_min(1.0e-8)
  else:
    # Indistinguishable occupancies correctly reduce to uniform filtered PPO.
    weights = torch.ones_like(on_logits)

  on_advantages = advantages[:, filter_mask]
  normalized = (on_advantages - on_advantages.mean()) / (
    on_advantages.std(unbiased=False) + 1.0e-8
  )
  weighted = normalized * weights
  weighted = (weighted - weighted.mean()) / (
    weighted.std(unbiased=False) + 1.0e-8
  )
  population_scale = advantages.shape[1] / float(len(on_ids))
  output = torch.zeros_like(advantages)
  output[:, filter_mask] = weighted * population_scale

  effective_sample_size = float(weights.sum().square() / weights.square().sum())
  metrics: dict[str, float | int | bool | str] = {
    "v127_method_id": METHOD_ID,
    "occupancy_crossfit_folds": folds,
    "occupancy_feature_dim": features.shape[-1],
    "occupancy_filter_on_environment_count": len(on_ids),
    "occupancy_filter_off_environment_count": len(off_ids),
    "occupancy_classifier_balanced_accuracy": balanced_accuracy,
    "occupancy_classifier_true_positive_rate": true_positive_rate,
    "occupancy_classifier_true_negative_rate": true_negative_rate,
    "occupancy_heldout_logit_separation": separation,
    "occupancy_min_fold_separation": min(fold_separations),
    "occupancy_correction_active": correction_active,
    "occupancy_density_ratio_limit": float(density_ratio_limit),
    "occupancy_density_ratio_mean": float(weights.mean()),
    "occupancy_density_ratio_std": float(weights.std(unbiased=False)),
    "occupancy_density_ratio_min": float(weights.min()),
    "occupancy_density_ratio_max": float(weights.max()),
    "occupancy_density_ratio_effective_sample_fraction": (
      effective_sample_size / weights.numel()
    ),
    "occupancy_actor_filter_on_transition_fraction": float(filter_mask.float().mean()),
    "occupancy_actor_filter_off_advantage_max_abs": float(
      output[:, ~filter_mask].abs().max()
    ),
    "occupancy_actor_advantage_population_scale": population_scale,
    "occupancy_actor_filter_on_advantage_mean": float(output[:, filter_mask].mean()),
    "occupancy_actor_filter_on_advantage_std": float(
      output[:, filter_mask].std(unbiased=False)
    ),
    "occupancy_critic_uses_all_transitions": True,
  }
  return output, metrics


class PaperOccupancyCorrectedV127PPO(PaperAccumulatedV82PPO):
  """One paper-dual actor step corrected to the unshielded state occupancy."""

  occupancy_feature_chunk_size = 16_384

  def _state_value_features(self) -> torch.Tensor:
    observations = self.storage.observations
    flat_observations = observations.flatten(0, 1)
    chunks: list[torch.Tensor] = []
    actor_was_training = self.actor.training
    self.actor.eval()
    with torch.inference_mode():
      for start in range(0, len(flat_observations), self.occupancy_feature_chunk_size):
        batch = flat_observations[start : start + self.occupancy_feature_chunk_size]
        hidden = self.actor.get_latent(batch)
        for layer in list(self.actor.mlp.children())[:-1]:
          hidden = layer(hidden)
        chunks.append(hidden.detach())
    self.actor.train(actor_was_training)
    hidden_features = torch.cat(chunks).reshape(
      self.storage.num_transitions_per_env,
      self.storage.num_envs,
      -1,
    )
    values = self.storage.values.detach().to(hidden_features.dtype)
    return torch.cat((hidden_features, values), dim=-1)

  def apply_state_value_occupancy_correction(
    self, filter_mask: torch.Tensor
  ) -> dict[str, float | int | bool | str]:
    corrected, metrics = crossfit_occupancy_corrected_advantages(
      self._state_value_features(),
      self.storage.advantages.squeeze(-1),
      filter_mask.to(self.device),
    )
    self.storage.advantages.copy_(corrected.unsqueeze(-1))
    return metrics

  def update(self) -> dict[str, Any]:
    result = super().update()
    result.update(
      {
        "v127_method_id": METHOD_ID,
        "v127_filtered_actor_credit_only": True,
        "v127_critic_all_transition_credit": True,
      }
    )
    return result

  def save(self) -> dict[str, Any]:
    output = super().save()
    output.update(
      {
        "proximal_method_id": METHOD_ID,
        "v127_filtered_actor_credit_only": True,
        "v127_critic_all_transition_credit": True,
      }
    )
    return output
