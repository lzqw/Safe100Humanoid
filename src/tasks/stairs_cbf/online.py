"""Conservative on-policy refinement components for shielded deployment data."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from itertools import chain
import math
from typing import Any

import torch
from rsl_rl.algorithms import PPO

from mjlab.rl import RslRlPpoAlgorithmCfg

from src.tasks.velocity.rl import VelocityOnPolicyRunner


# A 12-D diagonal-Gaussian log probability is a float32 reduction.  Repeating
# the identical reduction over a larger flattened GPU rollout can differ by a
# few ULPs even though every distribution parameter and sampled action is
# unchanged.  Keep this far below PPO-scale changes while avoiding false
# failures observed at 7.15e-5 on a 32 x 512 rollout.
BEHAVIOR_LOG_PROB_ATOL = 2.0e-4
BEHAVIOR_DISTRIBUTION_PARAM_ATOL = 1.0e-5


def validate_behavior_log_prob(
  stored: torch.Tensor,
  recomputed: torch.Tensor,
  *,
  tolerance: float = BEHAVIOR_LOG_PROB_ATOL,
) -> float:
  """Return max behavior log-prob error or reject a real action mismatch."""
  if stored.shape != recomputed.shape:
    raise ValueError("stored and recomputed log probabilities must have equal shape")
  if tolerance < 0.0:
    raise ValueError("behavior log-prob tolerance must be non-negative")
  if not bool(torch.isfinite(stored).all() and torch.isfinite(recomputed).all()):
    raise RuntimeError("behavior log probability contains non-finite values")
  maximum_error = float(torch.max(torch.abs(recomputed - stored)))
  if maximum_error > tolerance:
    raise RuntimeError(
      "stored behavior log probability is inconsistent with a_policy: "
      f"{maximum_error}"
    )
  return maximum_error


def validate_behavior_distribution_params(
  stored: tuple[torch.Tensor, ...],
  recomputed: tuple[torch.Tensor, ...],
  *,
  tolerance: float = BEHAVIOR_DISTRIBUTION_PARAM_ATOL,
) -> float:
  """Prove the behavior Gaussian itself is unchanged before tolerating a reduction ULP."""
  if len(stored) != len(recomputed) or not stored:
    raise ValueError("stored and recomputed distribution parameter sets must align")
  maximum_error = 0.0
  for old, new in zip(stored, recomputed, strict=True):
    if old.shape != new.shape:
      raise ValueError("stored and recomputed distribution parameters must align")
    if not bool(torch.isfinite(old).all() and torch.isfinite(new).all()):
      raise RuntimeError("behavior distribution parameter contains non-finite values")
    maximum_error = max(maximum_error, float(torch.max(torch.abs(new - old))))
  if maximum_error > tolerance:
    raise RuntimeError(
      "stored behavior distribution is inconsistent with policy: "
      f"{maximum_error}"
    )
  return maximum_error


@dataclass(frozen=True)
class CandidateGateThresholds:
  target_kl: float = 0.003
  maximum_clip_fraction: float = 0.30
  # The warm-started stair policy already uses near-bound actions on some
  # joints; reject only a substantially saturated candidate in v1.
  maximum_action_saturation: float = 0.50
  maximum_total_kl: float = 0.05
  maximum_intervention_ratio: float = 1.05
  maximum_target_fall_rate: float = 0.0
  d0_retention_tolerance: float = 0.02
  neighbor_success_tolerance: float = 0.02
  neighbor_fall_tolerance: float = 0.02
  minimum_safe_improvement: float = 0.0
  maximum_actor_gradient_norm_pre_clip: float = 100.0
  maximum_critic_gradient_norm_pre_clip: float = 100.0


@dataclass(frozen=True)
class SafeImprovementScoreWeights:
  """Dimensionless target-domain score used after hard safety constraints."""

  success: float = 1.0
  episode_return: float = 0.02
  fall: float = 2.0
  intervention_per_riser: float = 0.05
  policy_drift: float = 1.0


def safe_improvement_score(
  result: dict[str, Any],
  *,
  total_kl_from_base: float,
  weights: SafeImprovementScoreWeights = SafeImprovementScoreWeights(),
) -> dict[str, float]:
  """Combine task quality, falls, CBF reliance, and mean-policy drift."""
  components = {
    "success": weights.success * float(result["success_rate"]),
    "episode_return": weights.episode_return * float(result.get("mean_return", 0.0)),
    "fall": -weights.fall * float(result["fall_rate"]),
    "intervention_per_riser": -weights.intervention_per_riser
    * safety_demand_per_riser(result),
    "policy_drift": -weights.policy_drift * float(total_kl_from_base),
  }
  if not bool(torch.isfinite(torch.tensor(list(components.values()))).all()):
    raise ValueError("safe improvement score contains non-finite components")
  return {**components, "total": sum(components.values())}


@dataclass(frozen=True)
class CbfIndependenceThresholds:
  """Deployment criteria for removing the runtime toe-riser filter."""

  maximum_off_success_drop: float = 0.02
  maximum_off_fall_increase: float = 0.02
  maximum_intervention_per_riser: float = 0.10
  maximum_correction_mean: float = 5.0e-4


def cbf_independence_gate(
  *,
  filter_on_eval: dict[str, float],
  filter_off_eval: dict[str, float],
  thresholds: CbfIndependenceThresholds = CbfIndependenceThresholds(),
) -> tuple[bool, list[str]]:
  """Require equivalent unshielded deployment and near-zero shield demand."""
  reasons: list[str] = []
  required = (
    "success_rate",
    "fall_rate",
    "intervention_per_riser",
    "correction_mean",
  )
  for name, result in (("filter_on", filter_on_eval), ("filter_off", filter_off_eval)):
    for key in required:
      value = torch.tensor(float(result.get(key, float("nan"))))
      if not torch.isfinite(value):
        reasons.append(f"{name} {key} missing or non-finite")
  if reasons:
    return False, reasons
  if (
    filter_off_eval["success_rate"]
    < filter_on_eval["success_rate"] - thresholds.maximum_off_success_drop
  ):
    reasons.append("CBF-off success gap exceeds limit")
  if (
    filter_off_eval["fall_rate"]
    > filter_on_eval["fall_rate"] + thresholds.maximum_off_fall_increase
  ):
    reasons.append("CBF-off fall gap exceeds limit")
  if (
    filter_on_eval["intervention_per_riser"]
    > thresholds.maximum_intervention_per_riser
  ):
    reasons.append("runtime intervention per riser exceeds near-zero limit")
  if filter_on_eval["correction_mean"] > thresholds.maximum_correction_mean:
    reasons.append("runtime correction mean exceeds near-zero limit")
  return len(reasons) == 0, reasons


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


def critic_calibration_by_riser(
  values: torch.Tensor,
  returns: torch.Tensor,
  stair_indices: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
  """Summarize warm-started value calibration at each reached stair index."""
  if values.shape != returns.shape or values.shape != stair_indices.shape:
    raise ValueError("values, returns, and stair indices must share [T, N] shape")
  if values.ndim != 2:
    raise ValueError("critic calibration inputs must be two-dimensional")
  if not bool(torch.isfinite(values).all() and torch.isfinite(returns).all()):
    raise RuntimeError("critic calibration contains non-finite values")
  output: dict[str, dict[str, float | int]] = {}
  for index in torch.unique(stair_indices).sort().values.tolist():
    mask = stair_indices == index
    count = int(mask.sum())
    if count == 0:
      continue
    selected_values = values[mask]
    selected_returns = returns[mask]
    error = selected_values - selected_returns
    output[str(int(index))] = {
      "count": count,
      "value_mean": float(selected_values.mean()),
      "return_mean": float(selected_returns.mean()),
      "bias": float(error.mean()),
      "rmse": float(torch.sqrt(torch.mean(error.square()))),
    }
  return output


def pre_event_value_delta(
  values: torch.Tensor,
  events: torch.Tensor,
  dones: torch.Tensor,
  *,
  horizon: int,
) -> tuple[int, float | None]:
  """Measure value change from ``horizon`` steps before a safety event.

  Events whose look-back interval crosses an episode boundary are excluded.
  A negative mean indicates that the critic value declined before the event.
  """
  if values.shape != events.shape or values.shape != dones.shape:
    raise ValueError("values, events, and dones must share [T, N] shape")
  if values.ndim != 2 or horizon < 1:
    raise ValueError("pre-event diagnostics require [T, N] inputs and positive horizon")
  deltas = []
  for step in range(horizon, values.shape[0]):
    valid = events[step].bool()
    valid &= ~dones[step - horizon : step].bool().any(dim=0)
    if torch.any(valid):
      deltas.append(values[step, valid] - values[step - horizon, valid])
  if not deltas:
    return 0, None
  concatenated = torch.cat(deltas)
  return int(concatenated.numel()), float(concatenated.mean())


def rollout_action_dataflow_metrics(
  policy_actions: torch.Tensor,
  stored_actions: torch.Tensor,
  nominal_actions: torch.Tensor,
  safe_actions: torch.Tensor,
  executed_actions: torch.Tensor,
  filter_enabled: torch.Tensor,
  *,
  tolerance: float = 1.0e-6,
) -> dict[str, float]:
  """Audit the policy/nominal/safe/executed action paths of one rollout.

  PPO must retain the sampled policy action and its behavior log probability.
  Wrapper clipping and the CBF are environment-side transforms: the former
  produces ``nominal_actions`` and the latter produces ``safe_actions``.  The
  executed path must select safe actions only when the runtime filter is on.
  """
  action_tensors = (
    policy_actions,
    stored_actions,
    nominal_actions,
    safe_actions,
    executed_actions,
  )
  if any(tensor.shape != policy_actions.shape for tensor in action_tensors):
    raise ValueError("all rollout action tensors must have identical [T, N, A] shape")
  if filter_enabled.shape != policy_actions.shape[:-1]:
    raise ValueError("filter_enabled must have [T, N] shape")
  if tolerance < 0.0:
    raise ValueError("tolerance must be non-negative")

  routed_reference = torch.where(
    filter_enabled.unsqueeze(-1), safe_actions, nominal_actions
  )
  policy_storage_error = torch.max(torch.abs(policy_actions - stored_actions))
  executed_routing_error = torch.max(
    torch.abs(executed_actions - routed_reference)
  )
  policy_clipped = torch.abs(policy_actions - nominal_actions) > tolerance
  safe_changed = torch.linalg.vector_norm(
    safe_actions - nominal_actions, dim=-1
  ) > tolerance
  executed_changed = torch.linalg.vector_norm(
    executed_actions - nominal_actions, dim=-1
  ) > tolerance
  return {
    "policy_storage_max_abs_error": float(policy_storage_error),
    "executed_action_routing_max_abs_error": float(executed_routing_error),
    "policy_to_nominal_clip_fraction": float(policy_clipped.float().mean()),
    "counterfactual_safe_action_fraction": float(safe_changed.float().mean()),
    "executed_action_change_fraction": float(executed_changed.float().mean()),
    "runtime_filter_enabled_fraction": float(filter_enabled.float().mean()),
  }


def cbf_corrected_mean_target(
  policy_mean: torch.Tensor,
  nominal_raw_action: torch.Tensor,
  safe_raw_action: torch.Tensor,
) -> torch.Tensor:
  """Move the mean only by the CBF projection, cancelling sampled noise."""
  if not (
    policy_mean.shape == nominal_raw_action.shape == safe_raw_action.shape
  ):
    raise ValueError("mean, nominal raw action, and safe raw action shapes differ")
  correction = safe_raw_action - nominal_raw_action
  return policy_mean.detach() + correction.detach()


def candidate_gate(
  *,
  update_metrics: dict[str, float],
  old_eval: dict[str, dict[str, float]],
  candidate_eval: dict[str, dict[str, float]],
  base_d0_success: float,
  old_total_kl_from_base: float = 0.0,
  total_kl_from_base: float,
  parameters_finite: bool,
  thresholds: CandidateGateThresholds = CandidateGateThresholds(),
  target_domain: str = "D4",
  retention_domain: str = "D0",
  neighbor_domain: str = "D5",
  score_weights: SafeImprovementScoreWeights = SafeImprovementScoreWeights(),
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
  for domain in (retention_domain, target_domain, neighbor_domain):
    old_signature = old_eval[domain].get("initial_state_signatures")
    candidate_signature = candidate_eval[domain].get("initial_state_signatures")
    if old_signature is None or candidate_signature is None:
      reasons.append(f"{domain} paired initial-state signature missing")
    elif old_signature != candidate_signature:
      reasons.append(f"{domain} paired initial-state signature differs")
  intervals = candidate_gate_intervals(
    old_eval=old_eval,
    candidate_eval=candidate_eval,
    thresholds=thresholds,
    target_domain=target_domain,
    neighbor_domain=neighbor_domain,
    old_total_kl_from_base=old_total_kl_from_base,
    total_kl_from_base=total_kl_from_base,
    score_weights=score_weights,
  )
  success_delta = intervals["target_success_delta_95"]
  fall_delta = intervals["target_fall_delta_95"]
  intervention_ratio_delta = intervals["target_intervention_ratio_delta_95"]
  intervention_delta = intervals["target_intervention_delta_95"]
  if success_delta[2] < 0.0:
    reasons.append(f"{target_domain} success regressed")
  if fall_delta[1] > 0.0:
    reasons.append(f"{target_domain} fall rate increased")
  if candidate_d4["fall_rate"] > thresholds.maximum_target_fall_rate:
    reasons.append(f"{target_domain} candidate fall rate exceeds safety limit")
  if intervention_ratio_delta[1] > 0.0:
    reasons.append(f"{target_domain} intervention per riser increased")
  strictly_better = (
    success_delta[1] > 0.0
    or fall_delta[2] < 0.0
    or intervention_delta[2] < 0.0
  )
  if not strictly_better:
    reasons.append("target metrics show no strict improvement")
  safe_score_delta = intervals["target_safe_improvement_score_delta_95"]
  if safe_score_delta[0] <= thresholds.minimum_safe_improvement:
    reasons.append("target safe improvement score did not increase")
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
    neighbor_success_delta = intervals["neighbor_success_tolerance_delta_95"]
    neighbor_fall_delta = intervals["neighbor_fall_tolerance_delta_95"]
    if neighbor_success_delta[2] < 0.0:
      reasons.append(f"{neighbor_domain} success regressed")
    if neighbor_fall_delta[1] > 0.0:
      reasons.append(f"{neighbor_domain} fall rate increased")
  return len(reasons) == 0, reasons


def candidate_gate_intervals(
  *,
  old_eval: dict[str, dict[str, Any]],
  candidate_eval: dict[str, dict[str, Any]],
  thresholds: CandidateGateThresholds = CandidateGateThresholds(),
  target_domain: str = "D4",
  neighbor_domain: str = "D5",
  old_total_kl_from_base: float = 0.0,
  total_kl_from_base: float = 0.0,
  score_weights: SafeImprovementScoreWeights = SafeImprovementScoreWeights(),
) -> dict[str, tuple[float, float, float]]:
  """Expose the paired gate statistics used for an acceptance decision.

  Every tuple is ``(mean, lower_95, upper_95)``.  The ratio statistic is
  ``candidate_demand - maximum_intervention_ratio * old_demand``; therefore
  a strictly positive lower bound is evidence that shield demand grew beyond
  the allowed ratio.
  """
  old_target = old_eval[target_domain]
  candidate_target = candidate_eval[target_domain]
  old_neighbor = old_eval[neighbor_domain]
  candidate_neighbor = candidate_eval[neighbor_domain]
  return {
    "target_success_delta_95": paired_metric_delta_interval(
      old_target, candidate_target, "success_rate"
    ),
    "target_fall_delta_95": paired_metric_delta_interval(
      old_target, candidate_target, "fall_rate"
    ),
    "target_intervention_delta_95": paired_value_delta_interval(
      old_target,
      candidate_target,
      old_value=safety_demand_per_riser,
      candidate_value=safety_demand_per_riser,
    ),
    "target_intervention_ratio_delta_95": paired_value_delta_interval(
      old_target,
      candidate_target,
      old_value=lambda result: thresholds.maximum_intervention_ratio
      * safety_demand_per_riser(result),
      candidate_value=safety_demand_per_riser,
    ),
    "neighbor_success_tolerance_delta_95": paired_value_delta_interval(
      old_neighbor,
      candidate_neighbor,
      old_value=lambda result: float(result["success_rate"])
      - thresholds.neighbor_success_tolerance,
      candidate_value=lambda result: float(result["success_rate"]),
    ),
    "neighbor_fall_tolerance_delta_95": paired_value_delta_interval(
      old_neighbor,
      candidate_neighbor,
      old_value=lambda result: float(result["fall_rate"])
      + thresholds.neighbor_fall_tolerance,
      candidate_value=lambda result: float(result["fall_rate"]),
    ),
    "target_safe_improvement_score_delta_95": paired_value_delta_interval(
      old_target,
      candidate_target,
      old_value=lambda result: safe_improvement_score(
        result,
        total_kl_from_base=old_total_kl_from_base,
        weights=score_weights,
      )["total"],
      candidate_value=lambda result: safe_improvement_score(
        result,
        total_kl_from_base=total_kl_from_base,
        weights=score_weights,
      )["total"],
    ),
  }


def paired_value_delta_interval(
  old_result: dict[str, Any],
  candidate_result: dict[str, Any],
  *,
  old_value,
  candidate_value,
  method: str = "bootstrap",
  bootstrap_samples: int = 10000,
  bootstrap_seed: int = 0,
  z_value: float = 1.96,
) -> tuple[float, float, float]:
  """Return mean and a deterministic 95% interval of paired deltas.

  ``bootstrap`` resamples the *paired deltas*, never old/candidate results
  independently.  The fixed local generator makes gate decisions reproducible
  without changing the simulator RNG stream.  ``normal`` remains available
  for reproducing earlier diagnostic runs.
  """
  old_replicates = old_result.get("replicates")
  candidate_replicates = candidate_result.get("replicates")
  if old_replicates is None or candidate_replicates is None:
    deltas = [candidate_value(candidate_result) - old_value(old_result)]
  else:
    if len(old_replicates) != len(candidate_replicates) or not old_replicates:
      raise ValueError("paired evaluation replicate counts must match and be non-zero")
    deltas = [
      candidate_value(candidate) - old_value(old)
      for old, candidate in zip(
        old_replicates, candidate_replicates, strict=True
      )
    ]
  values = torch.tensor(deltas, dtype=torch.float64)
  mean = float(values.mean())
  if len(deltas) == 1:
    return mean, mean, mean
  if method == "normal":
    half_width = z_value * float(values.std(unbiased=True)) / math.sqrt(len(deltas))
    return mean, mean - half_width, mean + half_width
  if method != "bootstrap":
    raise ValueError(f"unknown paired interval method {method!r}")
  if bootstrap_samples < 100:
    raise ValueError("bootstrap_samples must be at least 100")
  generator = torch.Generator(device="cpu")
  generator.manual_seed(bootstrap_seed)
  indices = torch.randint(
    len(deltas),
    (bootstrap_samples, len(deltas)),
    generator=generator,
  )
  means = values[indices].mean(dim=1)
  lower, upper = torch.quantile(
    means, torch.tensor([0.025, 0.975], dtype=means.dtype)
  )
  return mean, float(lower), float(upper)


def paired_metric_delta_interval(
  old_result: dict[str, Any],
  candidate_result: dict[str, Any],
  metric: str,
  *,
  method: str = "bootstrap",
  bootstrap_samples: int = 10000,
  bootstrap_seed: int = 0,
  z_value: float = 1.96,
) -> tuple[float, float, float]:
  return paired_value_delta_interval(
    old_result,
    candidate_result,
    old_value=lambda result: float(result[metric]),
    candidate_value=lambda result: float(result[metric]),
    method=method,
    bootstrap_samples=bootstrap_samples,
    bootstrap_seed=bootstrap_seed,
    z_value=z_value,
  )


def adaptive_cbf_std_factor(
  intervention_per_riser: float,
  *,
  target_intervention_per_riser: float = 0.10,
  adaptation_rate: float = 0.10,
  minimum_factor: float = 0.80,
  maximum_factor: float = 1.05,
  fall_count: float = 0.0,
) -> float:
  """Compute the next-round exploration multiplier from actual shield demand."""
  values = torch.tensor(
    [
      intervention_per_riser,
      target_intervention_per_riser,
      adaptation_rate,
      minimum_factor,
      maximum_factor,
      fall_count,
    ],
    dtype=torch.float64,
  )
  if not torch.isfinite(values).all():
    raise ValueError("adaptive std inputs must be finite")
  if intervention_per_riser < 0.0 or target_intervention_per_riser < 0.0:
    raise ValueError("intervention rates must be non-negative")
  if adaptation_rate < 0.0 or not 0.0 < minimum_factor <= maximum_factor:
    raise ValueError("invalid adaptive std bounds/rate")
  factor = math.exp(
    adaptation_rate
    * (target_intervention_per_riser - intervention_per_riser)
  )
  factor = min(max(factor, minimum_factor), maximum_factor)
  if fall_count > 0.0:
    factor = min(factor, minimum_factor)
  return float(factor)


def safety_demand_per_riser(result: dict[str, Any]) -> float:
  """Select actual filter use or counterfactual demand by deployment mode."""
  if result.get("runtime_filter") is False:
    counterfactual = result.get("would_intervene_per_riser")
    if counterfactual is not None and torch.isfinite(
      torch.tensor(float(counterfactual))
    ):
      return float(counterfactual)
  return float(result["intervention_per_riser"])


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
  for key, limit, label in (
    (
      "actor_gradient_norm_pre_clip_max",
      thresholds.maximum_actor_gradient_norm_pre_clip,
      "actor gradient norm exceeds limit",
    ),
    (
      "critic_gradient_norm_pre_clip_max",
      thresholds.maximum_critic_gradient_norm_pre_clip,
      "critic gradient norm exceeds limit",
    ),
  ):
    if key in update_metrics:
      value = float(update_metrics[key])
      if not math.isfinite(value) or value > limit:
        reasons.append(label)
  if "value" in update_metrics and not math.isfinite(float(update_metrics["value"])):
    reasons.append("value loss is non-finite")
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
  actor_learning_rate: float = 5.0e-6
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
  base_anchor_weight: float = 0.01
  intervention_advantage_weight: float = 0.075
  safe_bc_weight: float = 0.0
  use_counterfactual_cbf_credit: bool = False


class OnlineSafePPO(PPO):
  """Single-clipped PPO with bounded exploration and CBF temporal credit."""

  def __init__(
    self,
    *args,
    actor_learning_rate: float = 5.0e-6,
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
    base_anchor_weight: float = 0.01,
    intervention_advantage_weight: float = 0.075,
    safe_bc_weight: float = 0.0,
    use_counterfactual_cbf_credit: bool = False,
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
    self.base_anchor_weight = base_anchor_weight
    self.intervention_advantage_weight = intervention_advantage_weight
    self.safe_bc_weight = safe_bc_weight
    self.use_counterfactual_cbf_credit = use_counterfactual_cbf_credit
    self._critic_only = False
    self._std_initialized = False
    self.base_actor_reference = None
    self._build_separate_optimizer()

    t = self.storage.num_transitions_per_env
    n = self.storage.num_envs
    action_dim = self.storage.actions.shape[-1]
    self.cbf_intervened = torch.zeros(t, n, dtype=torch.bool, device=self.device)
    self.cbf_magnitude = torch.zeros(t, n, device=self.device)
    self.actual_cbf_intervened = torch.zeros_like(self.cbf_intervened)
    self.actual_cbf_magnitude = torch.zeros_like(self.cbf_magnitude)
    self.nominal_targets = torch.zeros(t, n, action_dim, device=self.device)
    self.safe_targets = torch.zeros_like(self.nominal_targets)
    self.safe_raw_actions = torch.zeros_like(self.nominal_targets)
    self.nominal_raw_actions = torch.zeros_like(self.nominal_targets)
    self.executed_raw_actions = torch.zeros_like(self.nominal_targets)
    self.policy_actions = torch.zeros_like(self.nominal_targets)
    self.filter_enabled = torch.zeros(t, n, dtype=torch.bool, device=self.device)
    self.fall_events = torch.zeros(t, n, dtype=torch.bool, device=self.device)
    self.stair_indices = torch.zeros(t, n, dtype=torch.long, device=self.device)
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

  def set_base_actor_reference(
    self, base_state: dict[str, torch.Tensor] | None = None
  ) -> None:
    """Freeze the original mean policy used by the online KL anchor.

    When refinement resumes, ``self.actor`` contains the latest accepted
    policy.  In that case only the trainable MLP tensors are replaced from the
    original checkpoint; the frozen observation normalizer and bounded online
    standard deviation retain the exact deployment representation.
    """
    reference = copy.deepcopy(self.actor)
    if base_state is not None:
      state = reference.state_dict()
      with torch.no_grad():
        for key, value in base_state.items():
          if not key.startswith("mlp."):
            continue
          if key not in state or state[key].shape != value.shape:
            raise ValueError(f"base actor reference tensor is incompatible: {key}")
          state[key].copy_(
            value.to(device=state[key].device, dtype=state[key].dtype)
          )
      reference.load_state_dict(state, strict=True)
    reference.eval()
    for parameter in reference.parameters():
      parameter.requires_grad_(False)
    self.base_actor_reference = reference

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

  def shape_intervention_advantages(self) -> dict[str, float]:
    """Apply an immediate policy-only penalty for CBF correction magnitude.

    Returns/value targets remain those of the task, dual reward, and temporal
    pre-intervention credit.  This term is applied *after* normalized GAE, so
    it changes only the clipped policy surrogate and is not weakened by a
    second temporal propagation through the value target.
    """
    if self.intervention_magnitude_scale <= 0.0:
      raise ValueError("intervention magnitude scale must be positive")
    before = self.storage.advantages.clone()
    normalized = torch.clamp(
      self.cbf_magnitude / self.intervention_magnitude_scale, 0.0, 1.0
    ) * self.cbf_intervened.float()
    penalty = self.intervention_advantage_weight * normalized
    self.storage.advantages.sub_(penalty.unsqueeze(-1))
    return {
      "intervention_advantage_weight": float(
        self.intervention_advantage_weight
      ),
      "intervention_advantage_penalty_mean": float(penalty.mean()),
      "advantage_mean_before_intervention_shaping": float(before.mean()),
      "advantage_mean_after_intervention_shaping": float(
        self.storage.advantages.mean()
      ),
    }

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
      actual_intervened = intervened
      actual_magnitude = magnitude
      if self.use_counterfactual_cbf_credit:
        intervened = extras.get("cbf_would_intervene", intervened)
        magnitude = extras.get("cbf_counterfactual_magnitude", magnitude)
      nominal = extras.get("cbf_nominal_target")
      safe = extras.get("cbf_safe_target")
      safe_raw = extras.get("cbf_safe_raw_action")
      nominal_raw = extras.get("cbf_nominal_raw_action")
      executed_raw = extras.get("cbf_executed_raw_action")
      filter_enabled = extras.get("cbf_filter_enabled")
      fell = extras.get("online_fell")
      stair_index = extras.get("online_stair_index")
      if self.transition.actions is not None:
        self.policy_actions[step].copy_(self.transition.actions)
      if intervened is not None:
        self.cbf_intervened[step].copy_(intervened)
      if magnitude is not None:
        self.cbf_magnitude[step].copy_(magnitude)
      if actual_intervened is not None:
        self.actual_cbf_intervened[step].copy_(actual_intervened)
      if actual_magnitude is not None:
        self.actual_cbf_magnitude[step].copy_(actual_magnitude)
      if nominal is not None:
        self.nominal_targets[step].copy_(nominal)
      if safe is not None:
        self.safe_targets[step].copy_(safe)
      if safe_raw is not None:
        self.safe_raw_actions[step].copy_(safe_raw)
      if nominal_raw is not None:
        self.nominal_raw_actions[step].copy_(nominal_raw)
      if executed_raw is not None:
        self.executed_raw_actions[step].copy_(executed_raw)
      if filter_enabled is not None:
        self.filter_enabled[step].copy_(filter_enabled)
      if fell is not None:
        self.fall_events[step].copy_(fell)
      if stair_index is not None:
        self.stair_indices[step].copy_(stair_index)
    super().process_env_step(obs, rewards, dones, extras)

  def relabel_pre_intervention_costs(self) -> dict[str, float]:
    """Back-propagate actual or explicit counterfactual CBF demand."""
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
    action_metrics = rollout_action_dataflow_metrics(
      self.policy_actions,
      self.storage.actions,
      self.nominal_raw_actions,
      self.safe_raw_actions,
      self.executed_raw_actions,
      self.filter_enabled,
    )
    if action_metrics["policy_storage_max_abs_error"] > 1.0e-6:
      raise RuntimeError(
        "PPO storage no longer contains the sampled policy actions: "
        f"{action_metrics['policy_storage_max_abs_error']}"
      )
    if action_metrics["executed_action_routing_max_abs_error"] > 1.0e-5:
      raise RuntimeError(
        "runtime action does not match the configured CBF route: "
        f"{action_metrics['executed_action_routing_max_abs_error']}"
      )
    metrics = {
      "cbf_intervention_fraction": float(
        self.actual_cbf_intervened.float().mean()
      ),
      "cbf_correction_mean": float(self.actual_cbf_magnitude.mean()),
      "cbf_credit_event_fraction": float(self.cbf_intervened.float().mean()),
      "cbf_credit_correction_mean": float(self.cbf_magnitude.mean()),
      "cbf_credit_is_counterfactual": float(self.use_counterfactual_cbf_credit),
      "pre_intervention_cost_mean": float(credit.mean()),
      "pre_intervention_cost_max": float(credit.max()),
      "fall_event_count": float(self.fall_events.sum()),
      "fall_event_fraction": float(self.fall_events.float().mean()),
    }
    riser_delta = (self.stair_indices[1:] - self.stair_indices[:-1]).clamp_min(0)
    valid_transition = ~self.storage.dones[:-1].squeeze(-1).bool()
    riser_crossings = (riser_delta * valid_transition.long()).sum()
    metrics["riser_crossing_count"] = float(riser_crossings)
    metrics["cbf_intervention_per_riser"] = float(
      self.actual_cbf_intervened.sum() / riser_crossings.clamp_min(1)
    )
    metrics.update(action_metrics)
    self.last_update_metrics.update(metrics)
    return metrics

  def update(self) -> dict[str, Any]:
    """Run one exact single-clipped PPO update with a base-policy KL anchor."""
    if self.rnd or self.symmetry:
      raise RuntimeError("online safe PPO does not support RND or symmetry losses")
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise RuntimeError("online safe PPO currently requires feed-forward models")
    if self.schedule != "fixed":
      raise RuntimeError("online safe PPO requires a fixed learning-rate schedule")
    if self.base_anchor_weight > 0.0 and self.base_actor_reference is None:
      raise RuntimeError("base actor reference must be frozen before online PPO")

    # Keep references: RolloutStorage.clear() only resets the cursor.
    observations = self.storage.observations.flatten(0, 1)
    actions = self.storage.actions.flatten(0, 1).clone()
    old_log_prob = self.storage.actions_log_prob.flatten(0, 1).clone()
    old_params = tuple(
      parameter.flatten(0, 1).clone()
      for parameter in self.storage.distribution_params
    )
    with torch.inference_mode():
      self.actor(observations, stochastic_output=True)
      recomputed_old_log_prob = self.actor.get_output_log_prob(actions)
      old_distribution_param_max_error = validate_behavior_distribution_params(
        old_params, tuple(self.actor.output_distribution_params)
      )
      old_log_prob_max_error = validate_behavior_log_prob(
        old_log_prob.squeeze(-1), recomputed_old_log_prob
      )
      if self.base_actor_reference is None:
        anchor_kl_before = torch.tensor(0.0, device=self.device)
      else:
        current_params = tuple(self.actor.output_distribution_params)
        self.base_actor_reference(observations, stochastic_output=True)
        base_params = tuple(self.base_actor_reference.output_distribution_params)
        anchor_kl_before = self.actor.get_kl_divergence(
          current_params, base_params
        ).mean()

    returns_matrix = self.storage.returns.squeeze(-1).clone()
    values_matrix = self.storage.values.squeeze(-1).clone()
    dones_matrix = self.storage.dones.squeeze(-1).bool()
    returns_before = returns_matrix.flatten()
    values_before = values_matrix.flatten()
    return_variance = torch.var(returns_before, unbiased=False)
    explained_variance = 1.0 - torch.var(
      returns_before - values_before, unbiased=False
    ) / return_variance.clamp_min(1.0e-8)
    centered_returns = returns_before - returns_before.mean()
    centered_values = values_before - values_before.mean()
    return_value_correlation = torch.sum(centered_returns * centered_values) / (
      torch.sqrt(
        torch.sum(centered_returns.square())
        * torch.sum(centered_values.square())
      ).clamp_min(1.0e-8)
    )
    value_calibration = critic_calibration_by_riser(
      values_matrix, returns_matrix, self.stair_indices
    )
    intervention_events = self.actual_cbf_intervened.clone()
    intervention_events[1:] &= ~self.actual_cbf_intervened[:-1]
    intervention_count, intervention_value_delta = pre_event_value_delta(
      values_matrix,
      intervention_events,
      dones_matrix,
      horizon=self.pre_intervention_horizon,
    )
    fall_count, fall_value_delta = pre_event_value_delta(
      values_matrix,
      self.fall_events,
      dones_matrix,
      horizon=self.pre_intervention_horizon,
    )
    rollout_metrics = dict(self.last_update_metrics)

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_anchor_kl = 0.0
    maximum_actor_gradient_norm = 0.0
    maximum_critic_gradient_norm = 0.0
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1.0e-8)

      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      current_params = tuple(self.actor.output_distribution_params)
      entropy = self.actor.output_entropy

      ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
      surrogate = -batch.advantages.squeeze(-1) * ratio
      surrogate_clipped = -batch.advantages.squeeze(-1) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).square()
        value_losses_clipped = (value_clipped - batch.returns).square()
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).square().mean()

      if self.base_actor_reference is None or self._critic_only:
        anchor_kl = torch.zeros((), device=self.device)
      else:
        with torch.no_grad():
          self.base_actor_reference(batch.observations, stochastic_output=True)
          base_params = tuple(
            parameter.detach()
            for parameter in self.base_actor_reference.output_distribution_params
          )
        anchor_kl = self.actor.get_kl_divergence(
          current_params, base_params
        ).mean()

      if self._critic_only:
        loss = self.value_loss_coef * value_loss
      else:
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
          + self.base_anchor_weight * anchor_kl
        )
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
        self.actor.parameters(), self.max_grad_norm
      )
      critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
        self.critic.parameters(), self.max_grad_norm
      )
      self.optimizer.step()

      mean_value_loss += float(value_loss)
      mean_surrogate_loss += float(surrogate_loss)
      mean_entropy += float(entropy.mean())
      mean_anchor_kl += float(anchor_kl)
      maximum_actor_gradient_norm = max(
        maximum_actor_gradient_norm, float(actor_gradient_norm)
      )
      maximum_critic_gradient_norm = max(
        maximum_critic_gradient_norm, float(critic_gradient_norm)
      )

    num_updates = self.num_learning_epochs * self.num_mini_batches
    losses: dict[str, Any] = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "base_anchor_kl_loss": mean_anchor_kl / num_updates,
    }
    self.storage.clear()

    safe_bc = self.apply_safe_bc_auxiliary(
      observations=observations,
      learning_rate=(
        0.0
        if self._critic_only
        else self.actor_learning_rate * self.safe_bc_weight
      ),
    )
    self.clamp_online_std()
    with torch.inference_mode():
      self.actor(observations, stochastic_output=True)
      new_log_prob = self.actor.get_output_log_prob(actions)
      ratio = torch.exp(new_log_prob - old_log_prob.squeeze(-1))
      new_params = tuple(self.actor.output_distribution_params)
      kl = self.actor.get_kl_divergence(old_params, new_params).mean()
      clip_fraction = (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
      action_saturation = (self.actor.output_mean.abs() > 0.95).float().mean()
      if self.base_actor_reference is None:
        anchor_kl_after = torch.tensor(0.0, device=self.device)
      else:
        self.base_actor_reference(observations, stochastic_output=True)
        base_params = tuple(self.base_actor_reference.output_distribution_params)
        anchor_kl_after = self.actor.get_kl_divergence(
          new_params, base_params
        ).mean()
    diagnostics: dict[str, Any] = {
      "mean_kl": float(kl),
      "clip_fraction": float(clip_fraction),
      "action_saturation_fraction": float(action_saturation),
      "actor_learning_rate": float(self.actor_learning_rate),
      "action_std_mean": float(self.actor.output_std.mean()),
      "base_anchor_weight": float(self.base_anchor_weight),
      "base_anchor_kl_before_update": float(anchor_kl_before),
      "base_anchor_kl_after_update": float(anchor_kl_after),
      "actor_gradient_norm_pre_clip_max": maximum_actor_gradient_norm,
      "critic_gradient_norm_pre_clip_max": maximum_critic_gradient_norm,
      "safe_bc_loss": safe_bc["loss"],
      "safe_bc_weight": float(self.safe_bc_weight),
      "safe_bc_effective_learning_rate": safe_bc["learning_rate"],
      "safe_bc_gradient_norm": safe_bc["gradient_norm"],
      "explained_variance_before_update": float(explained_variance),
      "return_value_correlation_before_update": float(return_value_correlation),
      "critic_calibration_by_riser": value_calibration,
      "pre_intervention_value_event_count": intervention_count,
      "pre_intervention_value_delta": intervention_value_delta,
      "pre_fall_value_event_count": fall_count,
      "pre_fall_value_delta": fall_value_delta,
      "policy_old_log_prob_max_abs_error": old_log_prob_max_error,
      "policy_old_distribution_param_max_abs_error": (
        old_distribution_param_max_error
      ),
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
    # Safe-BC is only defined on actions that were actually projected and
    # executed.  Counterfactual events from filter-off simulation remain valid
    # reward credit but must not be mislabeled as real interventions here.
    intervention_mask = self.actual_cbf_intervened.flatten()
    if learning_rate <= 0.0 or not torch.any(intervention_mask):
      return result
    if observations is None:
      observations = self.storage.observations.flatten(0, 1)
    predicted_mean = self.actor(observations)
    safe_actions = self.safe_raw_actions.flatten(0, 1)
    nominal_actions = self.nominal_raw_actions.flatten(0, 1)
    target_mean = cbf_corrected_mean_target(
      predicted_mean,
      nominal_actions,
      safe_actions,
    )
    loss = torch.mean(
      (predicted_mean[intervention_mask] - target_mean[intervention_mask]) ** 2
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
    self.actual_cbf_intervened.zero_()
    self.actual_cbf_magnitude.zero_()
    self.nominal_targets.zero_()
    self.safe_targets.zero_()
    self.safe_raw_actions.zero_()
    self.nominal_raw_actions.zero_()
    self.executed_raw_actions.zero_()
    self.policy_actions.zero_()
    self.filter_enabled.zero_()
    self.fall_events.zero_()
    self.stair_indices.zero_()
    self.pre_intervention_cost.zero_()


class OnlineSafeRefinementRunner(VelocityOnPolicyRunner):
  """Runner helpers for base-policy warm start and transactional rollback."""

  alg: OnlineSafePPO

  def _load_critic_with_expansion(
    self,
    source: dict[str, torch.Tensor],
    *,
    source_is_base_critic: bool,
  ) -> dict[str, int | str]:
    """Load a critic while preserving old observation-column semantics."""
    target = self.alg.critic.state_dict()
    source_width = int(source["mlp.0.weight"].shape[1])
    target_width = int(target["mlp.0.weight"].shape[1])
    group_widths = {
      name: int(self.env.get_observations()[name].shape[-1])
      for name in self.alg.critic.obs_groups
    }
    if source_width == target_width:
      self.alg.critic.load_state_dict(source, strict=True)
      return {
        "source_critic_width": source_width,
        "expanded_critic_width": target_width,
        "source_critic_offset": 0,
        "critic_layout": "exact",
      }

    if source_is_base_critic:
      if "critic" not in group_widths or group_widths["critic"] != source_width:
        raise RuntimeError(
          "cannot locate legacy base critic observation block: "
          f"source={source_width}, groups={group_widths}"
        )
      offset = sum(
        group_widths[name]
        for name in self.alg.critic.obs_groups[
          : self.alg.critic.obs_groups.index("critic")
        ]
      )
      layout = "legacy_critic_group"
    else:
      # The delay queue was appended after the previous 799-D online state, so
      # every old feature retains its original column.  Only this exact prefix
      # expansion is supported; arbitrary shape mismatches fail loudly.
      if source_width >= target_width:
        raise RuntimeError(
          f"online critic cannot contract {source_width} inputs to {target_width}"
        )
      offset = 0
      layout = "online_prefix"

    for key, value in source.items():
      if key == "mlp.0.weight":
        target[key].zero_()
        target[key][:, offset : offset + source_width].copy_(value)
      elif key.startswith("obs_normalizer._") and value.ndim == 2:
        if key.endswith("_var") or key.endswith("_std"):
          target[key].fill_(1.0)
        else:
          target[key].zero_()
        target[key][:, offset : offset + source_width].copy_(value)
      elif key in target and target[key].shape == value.shape:
        target[key].copy_(value)
    self.alg.critic.load_state_dict(target, strict=True)
    return {
      "source_critic_width": source_width,
      "expanded_critic_width": target_width,
      "source_critic_offset": offset,
      "critic_layout": layout,
    }

  def load_base_checkpoint(self, path: str, map_location: str | None = None) -> dict:
    """Warm-start actor and expand the old critic at its observation offset."""
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    self.alg.actor.load_state_dict(loaded["actor_state_dict"], strict=True)
    expansion = self._load_critic_with_expansion(
      loaded["critic_state_dict"], source_is_base_critic=True
    )
    self.alg.initialize_online_std()
    self.alg.set_base_actor_reference()
    self.current_learning_iteration = 0
    return expansion | {
      "source_iteration": int(loaded.get("iter", -1)),
      "legacy_critic_width": expansion["source_critic_width"],
      "legacy_critic_offset": expansion["source_critic_offset"],
    }

  def load_online_checkpoint(
    self, path: str, map_location: str | None = None
  ) -> dict[str, Any]:
    """Resume an accepted online actor and expand an older full critic."""
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    self.alg.actor.load_state_dict(loaded["actor_state_dict"], strict=True)
    expansion = self._load_critic_with_expansion(
      loaded["critic_state_dict"], source_is_base_critic=False
    )
    # A new optimizer is mandatory after an input expansion and is also the
    # intended online protocol after every accepted/backtracked checkpoint.
    self.alg.reset_online_optimizer()
    self.alg._std_initialized = True
    self.current_learning_iteration = int(loaded.get("iter", 0))
    return expansion | {
      "source_iteration": int(loaded.get("iter", -1)),
      "optimizer_reset": True,
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
