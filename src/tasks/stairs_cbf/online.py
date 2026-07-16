"""Conservative on-policy refinement components for shielded deployment data."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from itertools import chain
from typing import Any

import torch
from rsl_rl.algorithms import PPO

from mjlab.rl import RslRlPpoAlgorithmCfg

from src.tasks.velocity.rl import VelocityOnPolicyRunner


@dataclass(frozen=True)
class CandidateGateThresholds:
  target_kl: float = 0.003
  maximum_clip_fraction: float = 0.30
  # The warm-started stair policy already uses near-bound actions on some
  # joints; reject only a substantially saturated candidate in v1.
  maximum_action_saturation: float = 0.50
  maximum_total_kl: float = 0.05
  maximum_intervention_ratio: float = 1.05
  d0_retention_tolerance: float = 0.02
  neighbor_success_tolerance: float = 0.02
  neighbor_fall_tolerance: float = 0.02


def backward_intervention_credit(
  magnitude: torch.Tensor,
  intervened: torch.Tensor,
  dones: torch.Tensor,
  *,
  horizon: int,
  decay: float,
  magnitude_scale: float,
) -> torch.Tensor:
  """Assign an intervention to preceding actions without crossing episode ends."""
  if magnitude.shape != intervened.shape or magnitude.shape != dones.shape:
    raise ValueError("magnitude, intervened, and dones must have identical [T, N] shapes")
  normalized = torch.clamp(magnitude / magnitude_scale, 0.0, 1.0)
  normalized = normalized * intervened.float()
  credit = torch.zeros_like(normalized)
  for event_step in range(normalized.shape[0]):
    alive = torch.ones(
      normalized.shape[1], dtype=torch.bool, device=normalized.device
    )
    for lag in range(min(horizon, event_step + 1)):
      source = event_step - lag
      if lag > 0:
        alive &= ~dones[source].bool()
      credit[source] += decay**lag * normalized[event_step] * alive.float()
  return credit


def candidate_gate(
  *,
  update_metrics: dict[str, float],
  old_eval: dict[str, dict[str, float]],
  candidate_eval: dict[str, dict[str, float]],
  base_d0_success: float,
  total_kl_from_base: float,
  parameters_finite: bool,
  thresholds: CandidateGateThresholds = CandidateGateThresholds(),
  target_domain: str = "D4",
  retention_domain: str = "D0",
  neighbor_domain: str = "D5",
) -> tuple[bool, list[str]]:
  """Apply the transactional D0/D4/D5 acceptance conditions."""
  reasons = candidate_precheck(
    update_metrics=update_metrics,
    total_kl_from_base=total_kl_from_base,
    parameters_finite=parameters_finite,
    thresholds=thresholds,
  )
  reasons = list(reasons)
  old_d4 = old_eval[target_domain]
  candidate_d4 = candidate_eval[target_domain]
  if candidate_d4["success_rate"] < old_d4["success_rate"]:
    reasons.append(f"{target_domain} success regressed")
  if candidate_d4["fall_rate"] > old_d4["fall_rate"]:
    reasons.append(f"{target_domain} fall rate increased")
  old_interventions = old_d4["intervention_per_riser"]
  candidate_interventions = candidate_d4["intervention_per_riser"]
  if candidate_interventions > thresholds.maximum_intervention_ratio * max(
    old_interventions, 1.0e-8
  ):
    reasons.append(f"{target_domain} intervention per riser increased")
  strictly_better = (
    candidate_d4["success_rate"] > old_d4["success_rate"]
    or candidate_d4["fall_rate"] < old_d4["fall_rate"]
    or candidate_interventions < old_interventions
  )
  if not strictly_better:
    reasons.append("target metrics show no strict improvement")
  if (
    candidate_eval[retention_domain]["success_rate"]
    < base_d0_success - thresholds.d0_retention_tolerance
  ):
    reasons.append(f"{retention_domain} retention bound violated")
  # D5 is mandatory evidence, even though the first version does not require
  # improvement there; a non-finite/missing result is rejected.
  if not torch.isfinite(
    torch.tensor(candidate_eval[neighbor_domain]["success_rate"])
  ):
    reasons.append(f"{neighbor_domain} evaluation missing or non-finite")
  else:
    old_neighbor = old_eval[neighbor_domain]
    candidate_neighbor = candidate_eval[neighbor_domain]
    if (
      candidate_neighbor["success_rate"]
      < old_neighbor["success_rate"] - thresholds.neighbor_success_tolerance
    ):
      reasons.append(f"{neighbor_domain} success regressed")
    if (
      candidate_neighbor["fall_rate"]
      > old_neighbor["fall_rate"] + thresholds.neighbor_fall_tolerance
    ):
      reasons.append(f"{neighbor_domain} fall rate increased")
  return len(reasons) == 0, reasons


def candidate_precheck(
  *,
  update_metrics: dict[str, float],
  total_kl_from_base: float,
  parameters_finite: bool,
  thresholds: CandidateGateThresholds = CandidateGateThresholds(),
) -> list[str]:
  """Cheap checks applied before launching paired simulator evaluation."""
  reasons: list[str] = []
  if not parameters_finite:
    reasons.append("non-finite model parameters")
  if update_metrics.get("mean_kl", float("inf")) > thresholds.target_kl:
    reasons.append("update KL exceeds target")
  if update_metrics.get("clip_fraction", float("inf")) > thresholds.maximum_clip_fraction:
    reasons.append("clip fraction exceeds limit")
  if (
    update_metrics.get("action_saturation_fraction", float("inf"))
    > thresholds.maximum_action_saturation
  ):
    reasons.append("action saturation exceeds limit")
  if total_kl_from_base > thresholds.maximum_total_kl:
    reasons.append("total KL from base exceeds limit")
  return reasons


def backtrack_actor_state(
  base_state: dict[str, torch.Tensor],
  candidate_state: dict[str, torch.Tensor],
  fraction: float,
) -> dict[str, torch.Tensor]:
  """Return a conservative point on one accepted PPO update direction.

  Only trainable actor MLP tensors are interpolated.  Frozen observation
  normalization and bounded distribution-variance state come from the
  candidate checkpoint unchanged.  This is a policy line search, not a new
  optimizer objective or a residual controller.
  """
  if not 0.0 <= fraction <= 1.0:
    raise ValueError(f"line-search fraction must be in [0, 1], got {fraction}")
  if base_state.keys() != candidate_state.keys():
    missing = sorted(base_state.keys() ^ candidate_state.keys())
    raise ValueError(f"actor state keys differ: {missing}")
  output = {key: value.detach().clone() for key, value in candidate_state.items()}
  for key, candidate in candidate_state.items():
    if not key.startswith("mlp."):
      continue
    base = base_state[key].to(device=candidate.device, dtype=candidate.dtype)
    if base.shape != candidate.shape:
      raise ValueError(
        f"actor state shape differs for {key}: {base.shape} != {candidate.shape}"
      )
    output[key] = torch.lerp(base, candidate, fraction)
  return output


@dataclass
class OnlineSafePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Small-step PPO defaults for a warm-started humanoid policy."""

  class_name: str = "src.tasks.stairs_cbf.online:OnlineSafePPO"
  actor_learning_rate: float = 1.0e-5
  critic_learning_rate: float = 1.0e-4
  actor_layer_multipliers: tuple[float, ...] = (0.10, 0.25, 0.50, 1.0)
  log_std_learning_rate: float = 0.0
  std_scale_from_base: float = 0.35
  minimum_std: float = 0.05
  maximum_std: float = 0.35
  pre_intervention_horizon: int = 10
  pre_intervention_decay: float = 0.8
  pre_intervention_weight: float = 0.20
  intervention_magnitude_scale: float = 0.05
  safe_bc_weight: float = 0.0


class OnlineSafePPO(PPO):
  """Single-clipped PPO with bounded exploration and CBF temporal credit."""

  def __init__(
    self,
    *args,
    actor_learning_rate: float = 1.0e-5,
    critic_learning_rate: float = 1.0e-4,
    actor_layer_multipliers: tuple[float, ...] = (0.10, 0.25, 0.50, 1.0),
    log_std_learning_rate: float = 0.0,
    std_scale_from_base: float = 0.35,
    minimum_std: float = 0.05,
    maximum_std: float = 0.35,
    pre_intervention_horizon: int = 10,
    pre_intervention_decay: float = 0.8,
    pre_intervention_weight: float = 0.20,
    intervention_magnitude_scale: float = 0.05,
    safe_bc_weight: float = 0.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.actor_learning_rate = actor_learning_rate
    self.critic_learning_rate = critic_learning_rate
    self.log_std_learning_rate = log_std_learning_rate
    self.actor_layer_multipliers = tuple(actor_layer_multipliers)
    self.std_scale_from_base = std_scale_from_base
    self.minimum_std = minimum_std
    self.maximum_std = maximum_std
    self.pre_intervention_horizon = pre_intervention_horizon
    self.pre_intervention_decay = pre_intervention_decay
    self.pre_intervention_weight = pre_intervention_weight
    self.intervention_magnitude_scale = intervention_magnitude_scale
    self.safe_bc_weight = safe_bc_weight
    self._critic_only = False
    self._std_initialized = False
    self._build_separate_optimizer()

    t = self.storage.num_transitions_per_env
    n = self.storage.num_envs
    action_dim = self.storage.actions.shape[-1]
    self.cbf_intervened = torch.zeros(t, n, dtype=torch.bool, device=self.device)
    self.cbf_magnitude = torch.zeros(t, n, device=self.device)
    self.nominal_targets = torch.zeros(t, n, action_dim, device=self.device)
    self.safe_targets = torch.zeros_like(self.nominal_targets)
    self.safe_raw_actions = torch.zeros_like(self.nominal_targets)
    self.fall_events = torch.zeros(t, n, dtype=torch.bool, device=self.device)
    self.pre_intervention_cost = torch.zeros(t, n, device=self.device)
    self.last_update_metrics: dict[str, float] = {}

  def _build_separate_optimizer(self) -> None:
    layers = [module for module in self.actor.mlp if isinstance(module, torch.nn.Linear)]
    if len(layers) != len(self.actor_layer_multipliers):
      raise ValueError(
        "actor_layer_multipliers must match actor Linear layers: "
        f"{len(self.actor_layer_multipliers)} != {len(layers)}"
      )
    groups: list[dict[str, Any]] = []
    for layer, multiplier in zip(layers, self.actor_layer_multipliers, strict=True):
      groups.append(
        {
          "params": list(layer.parameters()),
          "lr": self.actor_learning_rate * multiplier,
          "base_lr": self.actor_learning_rate * multiplier,
          "role": "actor",
        }
      )
    distribution_params = list(self.actor.distribution.parameters())
    if distribution_params:
      groups.append(
        {
          "params": distribution_params,
          "lr": self.log_std_learning_rate,
          "base_lr": self.log_std_learning_rate,
          "role": "std",
        }
      )
    groups.append(
      {
        "params": list(self.critic.parameters()),
        "lr": self.critic_learning_rate,
        "base_lr": self.critic_learning_rate,
        "role": "critic",
      }
    )
    # A fresh optimizer deliberately discards large-scale pretraining momentum.
    self.optimizer = torch.optim.Adam(groups)
    self.learning_rate = self.actor_learning_rate

  def reset_online_optimizer(self) -> None:
    """Discard momentum before a new accepted-policy refinement round."""
    self._build_separate_optimizer()

  def train_mode(self) -> None:
    super().train_mode()
    # The deployment actor uses the base observation scale throughout online
    # refinement.  The new full critic is allowed to calibrate its normalizer.
    if getattr(self.actor, "obs_normalization", False):
      self.actor.obs_normalizer.eval()

  def initialize_online_std(self) -> None:
    if self._std_initialized:
      return
    distribution = self.actor.distribution
    with torch.no_grad():
      if hasattr(distribution, "std_param"):
        distribution.std_param.mul_(self.std_scale_from_base).clamp_(
          self.minimum_std, self.maximum_std
        )
      elif hasattr(distribution, "log_std_param"):
        std = distribution.log_std_param.exp().mul_(self.std_scale_from_base)
        distribution.log_std_param.copy_(
          std.clamp(self.minimum_std, self.maximum_std).log()
        )
    self._std_initialized = True

  def clamp_online_std(self) -> None:
    distribution = self.actor.distribution
    with torch.no_grad():
      if hasattr(distribution, "std_param"):
        distribution.std_param.clamp_(self.minimum_std, self.maximum_std)
      elif hasattr(distribution, "log_std_param"):
        distribution.log_std_param.clamp_(
          torch.log(torch.tensor(self.minimum_std, device=self.device)),
          torch.log(torch.tensor(self.maximum_std, device=self.device)),
        )

  def set_critic_only(self, enabled: bool) -> None:
    self._critic_only = enabled
    for group in self.optimizer.param_groups:
      group["lr"] = 0.0 if enabled and group["role"] != "critic" else group["base_lr"]

  def scale_actor_learning_rate(self, factor: float) -> None:
    self.actor_learning_rate *= factor
    for group in self.optimizer.param_groups:
      if group["role"] == "actor":
        group["base_lr"] *= factor
        group["lr"] = 0.0 if self._critic_only else group["base_lr"]

  def scale_exploration_std(self, factor: float) -> None:
    distribution = self.actor.distribution
    with torch.no_grad():
      if hasattr(distribution, "std_param"):
        distribution.std_param.mul_(factor)
      elif hasattr(distribution, "log_std_param"):
        distribution.log_std_param.add_(torch.log(torch.tensor(factor)))
    self.clamp_online_std()

  def process_env_step(
    self,
    obs,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    step = self.storage.step
    if step < self.storage.num_transitions_per_env:
      intervened = extras.get("cbf_intervened")
      magnitude = extras.get("cbf_intervention_magnitude")
      nominal = extras.get("cbf_nominal_target")
      safe = extras.get("cbf_safe_target")
      safe_raw = extras.get("cbf_safe_raw_action")
      fell = extras.get("online_fell")
      if intervened is not None:
        self.cbf_intervened[step].copy_(intervened)
      if magnitude is not None:
        self.cbf_magnitude[step].copy_(magnitude)
      if nominal is not None:
        self.nominal_targets[step].copy_(nominal)
      if safe is not None:
        self.safe_targets[step].copy_(safe)
      if safe_raw is not None:
        self.safe_raw_actions[step].copy_(safe_raw)
      if fell is not None:
        self.fall_events[step].copy_(fell)
    super().process_env_step(obs, rewards, dones, extras)

  def relabel_pre_intervention_costs(self) -> dict[str, float]:
    """Back-propagate actual projection intensity without crossing resets."""
    t = self.storage.num_transitions_per_env
    credit = backward_intervention_credit(
      self.cbf_magnitude,
      self.cbf_intervened,
      self.storage.dones.squeeze(-1),
      horizon=self.pre_intervention_horizon,
      decay=self.pre_intervention_decay,
      magnitude_scale=self.intervention_magnitude_scale,
    )
    self.pre_intervention_cost.copy_(credit)
    self.storage.rewards -= self.pre_intervention_weight * credit.unsqueeze(-1)
    metrics = {
      "cbf_intervention_fraction": float(self.cbf_intervened.float().mean()),
      "cbf_correction_mean": float(self.cbf_magnitude.mean()),
      "pre_intervention_cost_mean": float(credit.mean()),
      "pre_intervention_cost_max": float(credit.max()),
      "fall_event_count": float(self.fall_events.sum()),
      "fall_event_fraction": float(self.fall_events.float().mean()),
    }
    self.last_update_metrics.update(metrics)
    return metrics

  def update(self) -> dict[str, float]:
    # Keep references: RolloutStorage.clear() only resets the cursor.
    observations = self.storage.observations.flatten(0, 1)
    actions = self.storage.actions.flatten(0, 1).clone()
    old_log_prob = self.storage.actions_log_prob.flatten(0, 1).clone()
    old_params = tuple(p.flatten(0, 1).clone() for p in self.storage.distribution_params)
    returns_before = self.storage.returns.flatten().clone()
    values_before = self.storage.values.flatten().clone()
    return_variance = torch.var(returns_before, unbiased=False)
    explained_variance = 1.0 - torch.var(
      returns_before - values_before, unbiased=False
    ) / return_variance.clamp_min(1.0e-8)
    centered_returns = returns_before - returns_before.mean()
    centered_values = values_before - values_before.mean()
    return_value_correlation = torch.sum(centered_returns * centered_values) / (
      torch.sqrt(torch.sum(centered_returns.square()) * torch.sum(centered_values.square()))
      .clamp_min(1.0e-8)
    )
    rollout_metrics = dict(self.last_update_metrics)
    losses = super().update()
    safe_bc = self.apply_safe_bc_auxiliary(
      observations=observations,
      learning_rate=self.actor_learning_rate * self.safe_bc_weight,
    )
    self.clamp_online_std()
    with torch.inference_mode():
      self.actor(observations, stochastic_output=True)
      new_log_prob = self.actor.get_output_log_prob(actions)
      ratio = torch.exp(new_log_prob - old_log_prob.squeeze(-1))
      new_params = self.actor.output_distribution_params
      kl = self.actor.get_kl_divergence(old_params, new_params).mean()
      clip_fraction = (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
      action_saturation = (self.actor.output_mean.abs() > 0.95).float().mean()
    diagnostics = {
      "mean_kl": float(kl),
      "clip_fraction": float(clip_fraction),
      "action_saturation_fraction": float(action_saturation),
      "actor_learning_rate": float(self.actor_learning_rate),
      "action_std_mean": float(self.actor.output_std.mean()),
      "safe_bc_loss": safe_bc["loss"],
      "safe_bc_weight": float(self.safe_bc_weight),
      "safe_bc_effective_learning_rate": safe_bc["learning_rate"],
      "safe_bc_gradient_norm": safe_bc["gradient_norm"],
      "explained_variance_before_update": float(explained_variance),
      "return_value_correlation_before_update": float(return_value_correlation),
    }
    diagnostics.update(rollout_metrics)
    losses.update(diagnostics)
    self.last_update_metrics = {}
    return losses

  def apply_safe_bc_auxiliary(
    self,
    *,
    observations=None,
    learning_rate: float,
  ) -> dict[str, float]:
    """Apply one explicitly scaled SGD step on true CBF interventions.

    A separate Adam step makes a nominal loss coefficient nearly ineffective
    on its first update because Adam normalizes the gradient magnitude.  This
    auxiliary deliberately uses a stateless layer-wise SGD micro-step, so a
    smaller coefficient produces a proportionally smaller policy change.  It
    does not update the Gaussian variance or critic and is never used as a
    policy-gradient objective.
    """
    result = {"loss": 0.0, "learning_rate": float(learning_rate), "gradient_norm": 0.0}
    intervention_mask = self.cbf_intervened.flatten()
    if learning_rate <= 0.0 or not torch.any(intervention_mask):
      return result
    if observations is None:
      observations = self.storage.observations.flatten(0, 1)
    predicted_mean = self.actor(observations)
    safe_actions = self.safe_raw_actions.flatten(0, 1)
    loss = torch.mean(
      (predicted_mean[intervention_mask] - safe_actions[intervention_mask].detach())
      ** 2
    )
    self.actor.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
      self.actor.parameters(), self.max_grad_norm
    )
    layers = [module for module in self.actor.mlp if isinstance(module, torch.nn.Linear)]
    with torch.no_grad():
      for layer, multiplier in zip(layers, self.actor_layer_multipliers, strict=True):
        for parameter in layer.parameters():
          if parameter.grad is not None:
            parameter.add_(
              parameter.grad,
              alpha=-learning_rate * multiplier,
            )
    self.actor.zero_grad(set_to_none=True)
    result.update(loss=float(loss), gradient_norm=float(gradient_norm))
    return result

  def clear_cbf_rollout(self) -> None:
    self.cbf_intervened.zero_()
    self.cbf_magnitude.zero_()
    self.nominal_targets.zero_()
    self.safe_targets.zero_()
    self.safe_raw_actions.zero_()
    self.fall_events.zero_()
    self.pre_intervention_cost.zero_()


class OnlineSafeRefinementRunner(VelocityOnPolicyRunner):
  """Runner helpers for base-policy warm start and transactional rollback."""

  alg: OnlineSafePPO

  def load_base_checkpoint(self, path: str, map_location: str | None = None) -> dict:
    """Warm-start actor and expand the old critic at its observation offset."""
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    self.alg.actor.load_state_dict(loaded["actor_state_dict"], strict=True)

    source = loaded["critic_state_dict"]
    target = self.alg.critic.state_dict()
    old_width = source["mlp.0.weight"].shape[1]
    group_widths = {
      name: int(self.env.get_observations()[name].shape[-1])
      for name in self.alg.critic.obs_groups
    }
    if "critic" not in group_widths or group_widths["critic"] != old_width:
      raise RuntimeError(
        "cannot locate legacy critic observation block: "
        f"old={old_width}, groups={group_widths}"
      )
    offset = sum(
      group_widths[name]
      for name in self.alg.critic.obs_groups[: self.alg.critic.obs_groups.index("critic")]
    )
    for key, value in source.items():
      if key == "mlp.0.weight":
        target[key].zero_()
        target[key][:, offset : offset + old_width].copy_(value)
      elif key.startswith("obs_normalizer._") and value.ndim == 2:
        if key.endswith("_var") or key.endswith("_std"):
          target[key].fill_(1.0)
        else:
          target[key].zero_()
        target[key][:, offset : offset + old_width].copy_(value)
      elif key in target and target[key].shape == value.shape:
        target[key].copy_(value)
    self.alg.critic.load_state_dict(target, strict=True)
    self.alg.initialize_online_std()
    self.current_learning_iteration = 0
    return {
      "source_iteration": int(loaded.get("iter", -1)),
      "legacy_critic_width": old_width,
      "expanded_critic_width": self.alg.critic.obs_dim,
      "legacy_critic_offset": offset,
    }

  def snapshot_candidate_state(self) -> dict[str, Any]:
    return {
      "actor": {k: v.detach().clone() for k, v in self.alg.actor.state_dict().items()},
      "critic": {k: v.detach().clone() for k, v in self.alg.critic.state_dict().items()},
      "optimizer": copy.deepcopy(self.alg.optimizer.state_dict()),
    }

  def restore_candidate_state(self, state: dict[str, Any]) -> None:
    # Normalization buffers may have most recently been updated during a
    # no-gradient rollout.  Restore all transactional state atomically.
    with torch.no_grad():
      self.alg.actor.load_state_dict(state["actor"], strict=True)
      self.alg.critic.load_state_dict(state["critic"], strict=True)
      self.alg.optimizer.load_state_dict(state["optimizer"])

  def reduce_after_rejection(self) -> None:
    self.alg.scale_actor_learning_rate(0.5)
    self.alg.scale_exploration_std(0.8)

  def parameters_are_finite(self) -> bool:
    return all(
      bool(torch.isfinite(p).all())
      for p in chain(self.alg.actor.parameters(), self.alg.critic.parameters())
    )
