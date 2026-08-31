"""Moving-reference KL PPO for CBF-shielded online refinement.

This module is the independent v23 learning path.  It deliberately reuses
only the environment-side CBF telemetry and raw-action audits from
``online.py``.  Specialist rewards, replay/state-restart banks, grouped
advantages, fixed-policy anchors, auxiliary critics, and candidate selection
are not part of this algorithm.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

import torch

from .online import (
  OnlineSafePPO,
  OnlineSafePpoAlgorithmCfg,
  OnlineSafeRefinementRunner,
  validate_behavior_distribution_params,
  validate_behavior_log_prob,
)


METHOD_ID = "cbf-proximal-online-policy-refinement-v23"


class ProximalHardRollback(RuntimeError):
  """A protocol-declared numerical/dataflow failure requiring rollback."""

  def __init__(self, reason: str, metrics: dict[str, Any] | None = None) -> None:
    super().__init__(reason)
    self.reason = reason
    self.metrics = dict(metrics or {})


def diagonal_gaussian_forward_kl(
  current_params: tuple[torch.Tensor, torch.Tensor],
  reference_params: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
  """Return analytic ``KL(current || reference)`` for diagonal Gaussians.

  The returned tensor is reduced over the action dimension only.  Callers
  retain the transition/minibatch dimension and choose their own reduction.
  Reference tensors are detached here as a second line of defense for the
  moving-reference stop-gradient invariant.
  """
  current_mean, current_std = current_params
  reference_mean, reference_std = (
    value.detach() for value in reference_params
  )
  if not (
    current_mean.shape
    == current_std.shape
    == reference_mean.shape
    == reference_std.shape
  ):
    raise ValueError("forward-KL Gaussian parameter shapes must match")
  if not bool(
    torch.isfinite(current_mean).all()
    and torch.isfinite(current_std).all()
    and torch.isfinite(reference_mean).all()
    and torch.isfinite(reference_std).all()
  ):
    raise ProximalHardRollback("non-finite Gaussian parameter in moving KL")
  if bool((current_std <= 0.0).any() or (reference_std <= 0.0).any()):
    raise ProximalHardRollback("non-positive Gaussian standard deviation")
  variance_ratio = current_std.square() / reference_std.square()
  mean_term = (current_mean - reference_mean).square() / reference_std.square()
  per_dimension = (
    torch.log(reference_std / current_std)
    + 0.5 * (variance_ratio + mean_term - 1.0)
  )
  return per_dimension.sum(dim=-1)


def optimizer_state_is_finite(optimizer: torch.optim.Optimizer) -> bool:
  """Audit all tensor-valued optimizer moments without mutating them."""
  return all(
    bool(torch.isfinite(value).all())
    for state in optimizer.state.values()
    for value in state.values()
    if isinstance(value, torch.Tensor)
  )


def module_state_is_finite(module: torch.nn.Module) -> bool:
  """Audit parameters and floating-point buffers in one module."""
  tensors = list(module.parameters()) + [
    value for value in module.buffers() if value.is_floating_point()
  ]
  return all(bool(torch.isfinite(value).all()) for value in tensors)


@dataclass
class CbfProximalPpoAlgorithmCfg(OnlineSafePpoAlgorithmCfg):
  """Frozen defaults for the v23 moving-reference online PPO core."""

  class_name: str = "src.tasks.stairs_cbf.proximal:CbfProximalPPO"
  actor_learning_rate: float = 5.0e-6
  critic_learning_rate: float = 1.0e-4
  actor_layer_multipliers: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
  actor_new_feature_count: int = 0
  freeze_legacy_actor_input_columns: bool = False
  log_std_learning_rate: float = 0.0
  std_scale_from_base: float = 0.35
  minimum_std: float = 0.05
  maximum_std: float = 0.25
  pre_intervention_weight: float = 0.0
  intervention_advantage_weight: float = 0.0
  base_anchor_weight: float = 0.0
  d0_retention_anchor_weight: float = 0.0
  neighbor_retention_anchor_weight: float = 0.0
  safe_bc_weight: float = 0.0
  use_counterfactual_cbf_credit: bool = False
  task_first_constrained: bool = False
  brief_ppo_refinement: bool = False
  failure_focused_refinement: bool = False
  observable_failure_conditioned_refinement: bool = False
  hard_case_policy_weight: float = 0.0
  success_counterexample_policy_weight: float = 1.0
  matched_success_preservation_beta: float = 0.0
  correction_distillation_weight: float = 0.0
  moving_kl_beta: float = 0.5
  hard_kl_ceiling: float = 0.01
  critic_learning_epochs: int = 2
  freeze_log_std: bool = True


class CbfProximalPPO(OnlineSafePPO):
  """Clipped PPO plus forward KL to the current round-start policy."""

  def __init__(
    self,
    *args,
    moving_kl_beta: float = 0.5,
    hard_kl_ceiling: float = 0.01,
    critic_learning_epochs: int = 2,
    freeze_log_std: bool = True,
    allow_bounded_temporal_credit: bool = False,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.moving_kl_beta = float(moving_kl_beta)
    self.hard_kl_ceiling = float(hard_kl_ceiling)
    self.critic_learning_epochs = int(critic_learning_epochs)
    self.freeze_log_std = bool(freeze_log_std)
    self.allow_bounded_temporal_credit = bool(allow_bounded_temporal_credit)
    self.round_reference_actor: torch.nn.Module | None = None
    self.round_reference_index = 0

    scalars = torch.tensor(
      [self.moving_kl_beta, self.hard_kl_ceiling], dtype=torch.float64
    )
    if not bool(torch.isfinite(scalars).all()):
      raise ValueError("proximal KL coefficients must be finite")
    if self.moving_kl_beta < 0.0:
      raise ValueError("moving KL beta must be non-negative")
    if self.desired_kl is None or not 0.0 < self.desired_kl < self.hard_kl_ceiling:
      raise ValueError("target KL must be positive and below the hard ceiling")
    if self.num_learning_epochs not in (1, 2):
      raise ValueError("proximal actor update supports at most two epochs")
    if self.critic_learning_epochs != 2:
      raise ValueError("proximal critic requires exactly two epochs")
    if not self.freeze_log_std:
      raise ValueError("v23 requires a frozen Gaussian log standard deviation")
    if self.normalize_advantage_per_mini_batch:
      raise ValueError("v23 requires whole-rollout advantage normalization")
    if self.schedule != "fixed":
      raise ValueError("v23 requires a fixed learning-rate schedule")
    if tuple(self.actor_layer_multipliers) != (1.0, 1.0, 1.0, 1.0):
      raise ValueError("v23 uses one uniform actor learning rate")
    if self.allow_bounded_temporal_credit:
      valid_temporal_credit = (
        self.pre_intervention_aggregation == "max"
        and self.pre_intervention_horizon == 50
        and math.isclose(
          self.pre_intervention_decay, 0.95, rel_tol=0.0, abs_tol=1.0e-12
        )
        and math.isclose(
          self.pre_intervention_weight, 0.01, rel_tol=0.0, abs_tol=1.0e-12
        )
      )
      if not valid_temporal_credit:
        raise ValueError("v103 bounded temporal credit configuration differs")
    disabled = {
      "actor_new_feature_count": self.actor_new_feature_count,
      "freeze_legacy_actor_input_columns": self.freeze_legacy_actor_input_columns,
      "log_std_learning_rate": self.log_std_learning_rate,
      "pre_intervention_weight": (
        0.0
        if self.allow_bounded_temporal_credit
        else self.pre_intervention_weight
      ),
      "intervention_advantage_weight": self.intervention_advantage_weight,
      "base_anchor_weight": self.base_anchor_weight,
      "d0_retention_anchor_weight": self.d0_retention_anchor_weight,
      "neighbor_retention_anchor_weight": self.neighbor_retention_anchor_weight,
      "safe_bc_weight": self.safe_bc_weight,
      "use_counterfactual_cbf_credit": self.use_counterfactual_cbf_credit,
      "task_first_constrained": self.task_first_constrained,
      "brief_ppo_refinement": self.brief_ppo_refinement,
      "failure_focused_refinement": self.failure_focused_refinement,
      "observable_failure_conditioned_refinement": (
        self.observable_failure_conditioned_refinement
      ),
      "hard_case_policy_weight": self.hard_case_policy_weight,
      "matched_success_preservation_beta": self.matched_success_preservation_beta,
      "correction_distillation_weight": self.correction_distillation_weight,
    }
    enabled = [name for name, value in disabled.items() if bool(value)]
    if enabled:
      raise ValueError(
        "v23 forbids specialist/anchor/auxiliary dependencies: "
        + ", ".join(enabled)
      )
    if self.fall_critic is not None or self.intervention_critic is not None:
      raise ValueError("v23 has exactly one task critic")
    if self.risk_head is not None:
      raise ValueError("v23 has no risk head")

    for parameter in self.actor.distribution.parameters():
      parameter.requires_grad_(False)
    self.base_actor_reference = None
    self.retention_actor_reference = None
    self.retention_anchor_banks = {}
    self.reset_proximal_optimizers()

  def reset_proximal_optimizers(self) -> None:
    """Create separate Adam states for actor and critic."""
    actor_parameters = [
      parameter for parameter in self.actor.mlp.parameters() if parameter.requires_grad
    ]
    critic_parameters = [
      parameter for parameter in self.critic.parameters() if parameter.requires_grad
    ]
    if not actor_parameters or not critic_parameters:
      raise RuntimeError("proximal actor and critic must both be trainable")
    self.actor_optimizer = torch.optim.Adam(
      actor_parameters, lr=self.actor_learning_rate
    )
    self.critic_optimizer = torch.optim.Adam(
      critic_parameters, lr=self.critic_learning_rate
    )
    # rsl_rl checkpoint/evaluator compatibility.  Learning in this class uses
    # the two explicit optimizers above.
    self.optimizer = self.actor_optimizer
    self.learning_rate = self.actor_learning_rate

  def freeze_round_reference(self) -> None:
    """Replace the reference with an exact frozen copy of current ``pi_k``."""
    distribution = getattr(self.actor, "distribution", None)
    cached = getattr(distribution, "_distribution", None)
    if hasattr(distribution, "_distribution"):
      distribution._distribution = None
    try:
      reference = copy.deepcopy(self.actor)
    finally:
      if hasattr(distribution, "_distribution"):
        distribution._distribution = cached
    reference.eval()
    for parameter in reference.parameters():
      parameter.requires_grad_(False)
    self.round_reference_actor = reference
    self.round_reference_index += 1

  def _raise_if_corrupt(self, stage: str) -> None:
    if not module_state_is_finite(self.actor):
      raise ProximalHardRollback(f"non-finite actor state after {stage}")
    if not module_state_is_finite(self.critic):
      raise ProximalHardRollback(f"non-finite critic state after {stage}")
    if not optimizer_state_is_finite(self.actor_optimizer):
      raise ProximalHardRollback(f"actor optimizer corruption after {stage}")
    if not optimizer_state_is_finite(self.critic_optimizer):
      raise ProximalHardRollback(f"critic optimizer corruption after {stage}")

  @staticmethod
  def _finite_gradients(parameters) -> bool:
    return all(
      parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
      for parameter in parameters
    )

  def _whole_batch_policy_metrics(
    self,
    observations,
    actions: torch.Tensor,
    old_log_prob: torch.Tensor,
    reference_params: tuple[torch.Tensor, torch.Tensor],
  ) -> dict[str, float]:
    with torch.inference_mode():
      self.actor(observations, stochastic_output=True)
      current_params = tuple(self.actor.output_distribution_params)
      current_log_prob = self.actor.get_output_log_prob(actions)
      log_ratio = current_log_prob - old_log_prob
      ratio = torch.exp(log_ratio)
      moving_kl = diagonal_gaussian_forward_kl(
        current_params, reference_params
      ).mean()
      behavior_approx_kl = (-log_ratio).mean()
      behavior_approx_kl_nonnegative = (ratio - 1.0 - log_ratio).mean()
      clip_fraction = (
        torch.abs(ratio - 1.0) > self.clip_param
      ).float().mean()
    values = {
      "moving_forward_kl": float(moving_kl),
      "behavior_approx_kl": float(behavior_approx_kl),
      "behavior_approx_kl_nonnegative": float(
        behavior_approx_kl_nonnegative
      ),
      "clip_fraction": float(clip_fraction),
    }
    if not all(math.isfinite(value) for value in values.values()):
      raise ProximalHardRollback("non-finite whole-batch policy diagnostic")
    return values

  def update(self) -> dict[str, Any]:
    """Run one transactional v23 actor update and one standard value fit."""
    if self.round_reference_actor is None:
      raise RuntimeError("round-start moving reference was not frozen")
    if self.storage.step != self.storage.num_transitions_per_env:
      raise RuntimeError("proximal PPO requires one complete on-policy rollout")
    if self.rnd or self.symmetry or self.actor.is_recurrent or self.critic.is_recurrent:
      raise RuntimeError("v23 supports feed-forward PPO without RND/symmetry")

    observations = self.storage.observations.flatten(0, 1)
    actions = self.storage.actions.flatten(0, 1).clone()
    old_log_prob = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1).clone()
    reference_params = tuple(
      value.flatten(0, 1).clone().detach()
      for value in self.storage.distribution_params
    )
    advantages = self.storage.advantages.flatten().detach()
    if not bool(torch.isfinite(advantages).all()):
      raise ProximalHardRollback("non-finite whole-batch advantages")

    with torch.inference_mode():
      self.round_reference_actor(observations, stochastic_output=True)
      frozen_params = tuple(
        value.detach()
        for value in self.round_reference_actor.output_distribution_params
      )
      frozen_log_prob = self.round_reference_actor.get_output_log_prob(actions)
      reference_param_error = validate_behavior_distribution_params(
        reference_params, frozen_params
      )
      reference_log_prob_error = validate_behavior_log_prob(
        old_log_prob, frozen_log_prob
      )
      self.actor(observations, stochastic_output=True)
      current_params_before = tuple(self.actor.output_distribution_params)
      current_log_prob_before = self.actor.get_output_log_prob(actions)
      current_param_error = validate_behavior_distribution_params(
        reference_params, current_params_before
      )
      current_log_prob_error = validate_behavior_log_prob(
        old_log_prob, current_log_prob_before
      )

    actor_loss_total = 0.0
    ppo_loss_total = 0.0
    moving_kl_loss_total = 0.0
    entropy_total = 0.0
    actor_updates = 0
    actor_epochs_completed = 0
    actor_gradient_norm_max = 0.0
    epoch_moving_kl: list[float] = []
    target_kl_early_stopped = False

    for epoch in range(self.num_learning_epochs):
      for batch in self.storage.mini_batch_generator(self.num_mini_batches, 1):
        self.actor(batch.observations, stochastic_output=True)
        new_log_prob = self.actor.get_output_log_prob(batch.actions)
        current_params = tuple(self.actor.output_distribution_params)
        ratio = torch.exp(
          new_log_prob - batch.old_actions_log_prob.squeeze(-1)
        )
        advantage = batch.advantages.squeeze(-1)
        surrogate = -advantage * ratio
        surrogate_clipped = -advantage * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        ppo_loss = torch.maximum(surrogate, surrogate_clipped).mean()
        moving_kl_loss = diagonal_gaussian_forward_kl(
          current_params,
          tuple(value.detach() for value in batch.old_distribution_params),
        ).mean()
        entropy = self.actor.output_entropy.mean()
        actor_loss = (
          ppo_loss
          + self.moving_kl_beta * moving_kl_loss
          - self.entropy_coef * entropy
        )
        if not bool(torch.isfinite(actor_loss)):
          raise ProximalHardRollback("non-finite actor loss")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_parameters = list(self.actor.mlp.parameters())
        if not self._finite_gradients(actor_parameters):
          raise ProximalHardRollback("non-finite actor gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
          actor_parameters, self.max_grad_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
          raise ProximalHardRollback("non-finite actor gradient norm")
        self.actor_optimizer.step()
        self._raise_if_corrupt(f"actor epoch {epoch + 1}")
        actor_updates += 1
        actor_loss_total += float(actor_loss.detach())
        ppo_loss_total += float(ppo_loss.detach())
        moving_kl_loss_total += float(moving_kl_loss.detach())
        entropy_total += float(entropy.detach())
        actor_gradient_norm_max = max(
          actor_gradient_norm_max, float(gradient_norm)
        )

      actor_epochs_completed += 1
      epoch_metrics = self._whole_batch_policy_metrics(
        observations, actions, old_log_prob, reference_params
      )
      epoch_kl = epoch_metrics["moving_forward_kl"]
      epoch_moving_kl.append(epoch_kl)
      if epoch_kl > self.hard_kl_ceiling:
        raise ProximalHardRollback(
          "moving forward KL exceeded hard ceiling",
          {
            "moving_forward_kl": epoch_kl,
            "hard_kl_ceiling": self.hard_kl_ceiling,
            "actor_epochs_completed": actor_epochs_completed,
          },
        )
      if epoch_kl > float(self.desired_kl):
        target_kl_early_stopped = epoch + 1 < self.num_learning_epochs
        break

    critic_loss_total = 0.0
    critic_updates = 0
    critic_gradient_norm_max = 0.0
    for epoch in range(self.critic_learning_epochs):
      for batch in self.storage.mini_batch_generator(self.num_mini_batches, 1):
        values = self.critic(batch.observations)
        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - batch.returns).square()
          clipped_losses = (value_clipped - batch.returns).square()
          value_loss = torch.maximum(value_losses, clipped_losses).mean()
        else:
          value_loss = (values - batch.returns).square().mean()
        critic_loss = self.value_loss_coef * value_loss
        if not bool(torch.isfinite(critic_loss)):
          raise ProximalHardRollback("non-finite value loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_parameters = list(self.critic.parameters())
        if not self._finite_gradients(critic_parameters):
          raise ProximalHardRollback("non-finite critic gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
          critic_parameters, self.max_grad_norm
        )
        if not bool(torch.isfinite(gradient_norm)):
          raise ProximalHardRollback("non-finite critic gradient norm")
        self.critic_optimizer.step()
        self._raise_if_corrupt(f"critic epoch {epoch + 1}")
        critic_updates += 1
        critic_loss_total += float(value_loss.detach())
        critic_gradient_norm_max = max(
          critic_gradient_norm_max, float(gradient_norm)
        )

    if actor_updates < 1 or critic_updates < 1:
      raise ProximalHardRollback("proximal update completed no optimizer steps")
    self.clamp_online_std()
    final_policy = self._whole_batch_policy_metrics(
      observations, actions, old_log_prob, reference_params
    )
    if final_policy["moving_forward_kl"] > self.hard_kl_ceiling:
      raise ProximalHardRollback(
        "moving forward KL exceeded hard ceiling after value fit",
        final_policy,
      )
    self._raise_if_corrupt("complete proximal update")

    rollout_metrics = dict(self.last_update_metrics)
    result: dict[str, Any] = {
      "actor_loss": actor_loss_total / actor_updates,
      "surrogate": ppo_loss_total / actor_updates,
      "moving_forward_kl_loss": moving_kl_loss_total / actor_updates,
      "value": critic_loss_total / critic_updates,
      "entropy": entropy_total / actor_updates,
      "moving_kl_beta": self.moving_kl_beta,
      "target_kl": float(self.desired_kl),
      "hard_kl_ceiling": self.hard_kl_ceiling,
      "actor_epochs_completed": actor_epochs_completed,
      "critic_epochs_completed": self.critic_learning_epochs,
      "actor_minibatches_completed": actor_updates,
      "critic_minibatches_completed": critic_updates,
      "target_kl_early_stopped": target_kl_early_stopped,
      "epoch_moving_forward_kl": epoch_moving_kl,
      "actor_gradient_norm_pre_clip_max": actor_gradient_norm_max,
      "critic_gradient_norm_pre_clip_max": critic_gradient_norm_max,
      "behavior_reference_distribution_param_max_abs_error": (
        reference_param_error
      ),
      "behavior_reference_log_prob_max_abs_error": reference_log_prob_error,
      "behavior_current_distribution_param_max_abs_error": current_param_error,
      "behavior_current_log_prob_max_abs_error": current_log_prob_error,
      "whole_batch_advantage_mean": float(advantages.mean()),
      "whole_batch_advantage_std": float(advantages.std()),
      "round_reference_index": self.round_reference_index,
      "freeze_log_std": self.freeze_log_std,
      "action_std_mean": float(self.actor.output_std.mean()),
      **final_policy,
      **rollout_metrics,
    }
    # Compatibility alias is explicitly the analytic forward moving KL, not
    # the historical fixed-base or behavior reverse KL.
    result["mean_kl"] = result["moving_forward_kl"]
    self.storage.clear()
    self.clear_cbf_rollout()
    self.last_update_metrics = {}
    return result

  def save(self) -> dict[str, Any]:
    """Save only the single actor/critic and the two recovery optimizers."""
    output: dict[str, Any] = {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      # Kept for actor-only compatibility with the standard runner loader.
      "optimizer_state_dict": self.actor_optimizer.state_dict(),
      "proximal_actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
      "proximal_critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
      "proximal_method_id": METHOD_ID,
      "proximal_round_reference_index": self.round_reference_index,
    }
    if self.round_reference_actor is not None:
      output["proximal_round_reference_state_dict"] = (
        self.round_reference_actor.state_dict()
      )
    return output


class CbfProximalRefinementRunner(OnlineSafeRefinementRunner):
  """Warm-start and atomic recovery helpers for the independent v23 path."""

  alg: CbfProximalPPO

  def load_initial_checkpoint(
    self, path: str, map_location: str | None = None
  ) -> dict[str, Any]:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    actor_state = loaded["actor_state_dict"]
    critic_state = loaded["critic_state_dict"]
    target_actor = self.alg.actor.state_dict()
    target_critic = self.alg.critic.state_dict()
    actor_mismatches = {
      key: (tuple(value.shape), tuple(target_actor[key].shape))
      for key, value in actor_state.items()
      if key not in target_actor or value.shape != target_actor[key].shape
    }
    critic_mismatches = {
      key: (tuple(value.shape), tuple(target_critic[key].shape))
      for key, value in critic_state.items()
      if key not in target_critic or value.shape != target_critic[key].shape
    }
    if set(actor_state) != set(target_actor) or actor_mismatches:
      raise RuntimeError(
        f"v23 requires the exact original actor interface: {actor_mismatches}"
      )
    if set(critic_state) != set(target_critic) or critic_mismatches:
      raise RuntimeError(
        f"v23 requires the exact original privileged critic: {critic_mismatches}"
      )
    self.alg.actor.load_state_dict(actor_state, strict=True)
    self.alg.critic.load_state_dict(critic_state, strict=True)
    self.alg._std_initialized = False
    self.alg.initialize_online_std()
    for parameter in self.alg.actor.distribution.parameters():
      parameter.requires_grad_(False)
    self.alg.reset_proximal_optimizers()
    self.current_learning_iteration = 0
    return {
      "source_iteration": int(loaded.get("iter", -1)),
      "actor_observation_dim": int(self.alg.actor.obs_dim),
      "critic_observation_dim": int(self.alg.critic.obs_dim),
      "actor_layout": "exact-original-interface",
      "critic_layout": "exact-original-privileged-interface",
      "source_optimizer_discarded": True,
      "source_auxiliary_heads_ignored": True,
    }

  def snapshot_proximal_state(self) -> dict[str, Any]:
    return {
      "actor": copy.deepcopy(self.alg.actor.state_dict()),
      "critic": copy.deepcopy(self.alg.critic.state_dict()),
      "actor_optimizer": copy.deepcopy(self.alg.actor_optimizer.state_dict()),
      "critic_optimizer": copy.deepcopy(self.alg.critic_optimizer.state_dict()),
      "round_reference": (
        None
        if self.alg.round_reference_actor is None
        else copy.deepcopy(self.alg.round_reference_actor.state_dict())
      ),
      "round_reference_index": self.alg.round_reference_index,
    }

  def restore_proximal_state(self, state: dict[str, Any]) -> None:
    with torch.no_grad():
      self.alg.actor.load_state_dict(state["actor"], strict=True)
      self.alg.critic.load_state_dict(state["critic"], strict=True)
      self.alg.actor_optimizer.load_state_dict(state["actor_optimizer"])
      self.alg.critic_optimizer.load_state_dict(state["critic_optimizer"])
      reference_state = state["round_reference"]
      if reference_state is None:
        self.alg.round_reference_actor = None
      else:
        self.alg.freeze_round_reference()
        assert self.alg.round_reference_actor is not None
        self.alg.round_reference_actor.load_state_dict(
          reference_state, strict=True
        )
      self.alg.round_reference_index = int(state["round_reference_index"])
    self.alg._raise_if_corrupt("transaction restore")

  def load_recovery_checkpoint(
    self, path: str, map_location: str | None = None
  ) -> dict[str, Any]:
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if loaded.get("proximal_method_id") != METHOD_ID:
      raise ValueError("recovery checkpoint is not a v23 proximal checkpoint")
    self.alg.actor.load_state_dict(loaded["actor_state_dict"], strict=True)
    self.alg.critic.load_state_dict(loaded["critic_state_dict"], strict=True)
    self.alg.actor_optimizer.load_state_dict(
      loaded["proximal_actor_optimizer_state_dict"]
    )
    self.alg.critic_optimizer.load_state_dict(
      loaded["proximal_critic_optimizer_state_dict"]
    )
    reference = loaded.get("proximal_round_reference_state_dict")
    if reference is not None:
      self.alg.freeze_round_reference()
      assert self.alg.round_reference_actor is not None
      self.alg.round_reference_actor.load_state_dict(reference, strict=True)
    self.alg.round_reference_index = int(
      loaded.get("proximal_round_reference_index", 0)
    )
    self.alg._std_initialized = True
    self.alg._raise_if_corrupt("recovery checkpoint load")
    return {
      "source_iteration": int(loaded.get("iter", -1)),
      "recovered_actor_optimizer": True,
      "recovered_critic_optimizer": True,
      "recovered_round_reference": reference is not None,
    }
