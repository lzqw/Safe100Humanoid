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

from src.tasks.stairs_cbf.retention import (
  cyclic_retention_batch,
  increase_anchor_weight_on_budget_violation,
  validate_retention_observation_bank,
)
from src.tasks.velocity.rl import VelocityOnPolicyRunner


# A 12-D diagonal-Gaussian log probability is a float32 reduction.  Repeating
# the identical reduction over a larger flattened GPU rollout can differ by a
# few ULPs even though every distribution parameter and sampled action is
# unchanged.  Keep this far below PPO-scale changes while avoiding false
# failures observed at 2.57e-4 on a 64 x 1024 rollout.  Distribution
# parameters remain guarded independently at 1e-5, and the rollout audit
# still requires bitwise-identical stored policy actions.
BEHAVIOR_LOG_PROB_ATOL = 5.0e-4
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
  d0_fall_tolerance: float = 0.02
  neighbor_success_tolerance: float = 0.02
  neighbor_fall_tolerance: float = 0.02
  minimum_safe_improvement: float = 0.0
  maximum_actor_gradient_norm_pre_clip: float = 100.0
  maximum_critic_gradient_norm_pre_clip: float = 100.0
  require_task_improvement: bool = False


@dataclass(frozen=True)
class SafeImprovementScoreWeights:
  """Dimensionless target-domain score used after hard safety constraints."""

  success: float = 1.0
  episode_return: float = 0.02
  fall: float = 2.0
  intervention_per_riser: float = 0.05
  policy_drift: float = 1.0


@dataclass(frozen=True)
class BriefPpoGateThresholds:
  """Point-estimate gates used during brief PPO, not final paper evidence."""

  maximum_kl: float = 0.01
  maximum_target_fall_increase: float = 0.03
  d0_success_tolerance: float = 0.05


@dataclass(frozen=True)
class FailureFocusedGateThresholds:
  """Wide point gates for v15; final confidence gates are evaluated separately."""

  maximum_kl: float = 0.01
  maximum_target_fall_increase: float = 0.03
  maximum_cbf_demand_ratio: float = 1.25
  d0_success_tolerance: float = 0.05


@dataclass(frozen=True)
class SpecialistGateThresholds:
  """Frozen v17 target-only point gates; final evidence is audited later."""

  maximum_kl: float = 0.01
  maximum_target_fall_increase: float = 0.03
  d0_success_tolerance: float = 0.05


def brief_dual_reward_weight(round_index: int) -> float:
  """Task-first scalar-reward schedule from the v14 protocol."""
  if round_index < 1:
    raise ValueError("online round index must be positive")
  return 0.0 if round_index <= 2 else 0.02


def brief_target_score(result: dict[str, Any]) -> dict[str, float]:
  """Return ``SR - FR - 0.01 * CBF interventions/riser`` and components."""
  components = {
    "success": float(result.get("success_rate", float("nan"))),
    "fall": -float(result.get("fall_rate", float("nan"))),
    "intervention_per_riser": -0.01 * safety_demand_per_riser(result),
  }
  if not all(math.isfinite(value) for value in components.values()):
    raise ValueError("brief target score contains missing or non-finite values")
  return {**components, "total": sum(components.values())}


def brief_candidate_precheck(
  *,
  update_metrics: dict[str, Any],
  parameters_finite: bool,
  thresholds: BriefPpoGateThresholds = BriefPpoGateThresholds(),
) -> list[str]:
  """Apply only the numerical-health checks required before target evaluation."""
  reasons: list[str] = []
  if not parameters_finite:
    reasons.append("non-finite model parameters")
  kl = float(update_metrics.get("mean_kl", float("nan")))
  if not math.isfinite(kl):
    reasons.append("update KL is missing or non-finite")
  elif kl >= thresholds.maximum_kl:
    reasons.append("update KL is not below 0.01")
  return reasons


def brief_candidate_gate(
  *,
  update_metrics: dict[str, Any],
  old_eval: dict[str, Any],
  candidate_eval: dict[str, Any],
  parameters_finite: bool,
  thresholds: BriefPpoGateThresholds = BriefPpoGateThresholds(),
) -> tuple[bool, list[str], dict[str, dict[str, float]]]:
  """Accept a finite small-KL update with better target point estimate.

  Initial-state signatures are checked only to prove the advertised paired
  protocol.  No confidence interval, source-domain, or neighboring-domain
  performance condition is applied here.
  """
  reasons = brief_candidate_precheck(
    update_metrics=update_metrics,
    parameters_finite=parameters_finite,
    thresholds=thresholds,
  )
  old_signature = old_eval.get("initial_state_signatures")
  candidate_signature = candidate_eval.get("initial_state_signatures")
  if old_signature is None or candidate_signature is None:
    reasons.append("paired target initial-state signature missing")
  elif old_signature != candidate_signature:
    reasons.append("paired target initial-state signature differs")
  try:
    old_score = brief_target_score(old_eval)
    candidate_score = brief_target_score(candidate_eval)
  except (KeyError, TypeError, ValueError):
    reasons.append("target evaluation is missing or non-finite")
    old_score = {"total": float("nan")}
    candidate_score = {"total": float("nan")}
  if math.isfinite(old_score["total"]) and math.isfinite(
    candidate_score["total"]
  ):
    if candidate_score["total"] <= old_score["total"]:
      reasons.append("target point-estimate score did not improve")
    old_fall = float(old_eval["fall_rate"])
    candidate_fall = float(candidate_eval["fall_rate"])
    if candidate_fall > old_fall + thresholds.maximum_target_fall_increase:
      reasons.append("target fall rate increased by more than 3 percentage points")
  return len(reasons) == 0, reasons, {
    "old": old_score,
    "candidate": candidate_score,
  }


def brief_d0_retention_gate(
  *,
  baseline_eval: dict[str, Any],
  candidate_eval: dict[str, Any],
  thresholds: BriefPpoGateThresholds = BriefPpoGateThresholds(),
) -> tuple[bool, list[str]]:
  """Apply the periodic source check: candidate D0 SR >= baseline SR - 5 pp."""
  reasons: list[str] = []
  baseline_success = float(baseline_eval.get("success_rate", float("nan")))
  candidate_success = float(candidate_eval.get("success_rate", float("nan")))
  if not math.isfinite(baseline_success) or not math.isfinite(candidate_success):
    reasons.append("D0 success rate is missing or non-finite")
  elif candidate_success < baseline_success - thresholds.d0_success_tolerance:
    reasons.append("D0 success is more than 5 percentage points below baseline")
  baseline_signature = baseline_eval.get("initial_state_signatures")
  candidate_signature = candidate_eval.get("initial_state_signatures")
  if baseline_signature is None or candidate_signature is None:
    reasons.append("paired D0 initial-state signature missing")
  elif baseline_signature != candidate_signature:
    reasons.append("paired D0 initial-state signature differs")
  return len(reasons) == 0, reasons


def failure_focused_target_score(result: dict[str, Any]) -> dict[str, float]:
  """Return the v15 training score ``success - fall`` and its components."""
  components = {
    "success": float(result.get("success_rate", float("nan"))),
    "fall": -float(result.get("fall_rate", float("nan"))),
  }
  if not all(math.isfinite(value) for value in components.values()):
    raise ValueError("failure-focused target score is missing or non-finite")
  return {**components, "total": sum(components.values())}


def failure_focused_candidate_precheck(
  *,
  update_metrics: dict[str, Any],
  parameters_finite: bool,
  thresholds: FailureFocusedGateThresholds = FailureFocusedGateThresholds(),
) -> list[str]:
  reasons: list[str] = []
  if not parameters_finite:
    reasons.append("non-finite model parameters")
  kl = float(update_metrics.get("mean_kl", float("nan")))
  if not math.isfinite(kl):
    reasons.append("update KL is missing or non-finite")
  elif kl >= thresholds.maximum_kl:
    reasons.append("update KL is not below 0.01")
  return reasons


def failure_focused_candidate_gate(
  *,
  update_metrics: dict[str, Any],
  old_eval: dict[str, Any],
  candidate_eval: dict[str, Any],
  parameters_finite: bool,
  thresholds: FailureFocusedGateThresholds = FailureFocusedGateThresholds(),
) -> tuple[bool, list[str], dict[str, dict[str, float]]]:
  """Apply only v15 target score, fall, KL/finite, and catastrophic-CBF gates."""
  reasons = failure_focused_candidate_precheck(
    update_metrics=update_metrics,
    parameters_finite=parameters_finite,
    thresholds=thresholds,
  )
  old_signature = old_eval.get("initial_state_signatures")
  candidate_signature = candidate_eval.get("initial_state_signatures")
  if old_signature is None or candidate_signature is None:
    reasons.append("paired target initial-state signature missing")
  elif old_signature != candidate_signature:
    reasons.append("paired target initial-state signature differs")
  try:
    old_score = failure_focused_target_score(old_eval)
    candidate_score = failure_focused_target_score(candidate_eval)
    old_fall = float(old_eval["fall_rate"])
    candidate_fall = float(candidate_eval["fall_rate"])
    old_demand = safety_demand_per_riser(old_eval)
    candidate_demand = safety_demand_per_riser(candidate_eval)
  except (KeyError, TypeError, ValueError):
    reasons.append("target evaluation is missing or non-finite")
    old_score = {"total": float("nan")}
    candidate_score = {"total": float("nan")}
  else:
    if candidate_score["total"] <= old_score["total"]:
      reasons.append("target point-estimate score did not improve")
    if candidate_fall > old_fall + thresholds.maximum_target_fall_increase:
      reasons.append("target fall rate increased by more than 3 percentage points")
    if candidate_demand > thresholds.maximum_cbf_demand_ratio * old_demand:
      reasons.append("target CBF demand increased by more than 25 percent")
  return len(reasons) == 0, reasons, {
    "old": old_score,
    "candidate": candidate_score,
  }


def specialist_target_score(result: dict[str, Any]) -> dict[str, float]:
  """Return the v17 target score ``success rate - fall rate``."""
  return failure_focused_target_score(result)


def specialist_candidate_precheck(
  *,
  update_metrics: dict[str, Any],
  parameters_finite: bool,
  thresholds: SpecialistGateThresholds = SpecialistGateThresholds(),
) -> list[str]:
  """Check only finite parameters and the declared hard KL bound."""
  reasons: list[str] = []
  if not parameters_finite:
    reasons.append("non-finite model parameters")
  kl = float(update_metrics.get("mean_kl", float("nan")))
  if not math.isfinite(kl):
    reasons.append("update KL is missing or non-finite")
  elif kl >= thresholds.maximum_kl:
    reasons.append("update KL is not below 0.01")
  return reasons


def specialist_candidate_gate(
  *,
  update_metrics: dict[str, Any],
  old_eval: dict[str, Any],
  candidate_eval: dict[str, Any],
  parameters_finite: bool,
  thresholds: SpecialistGateThresholds = SpecialistGateThresholds(),
) -> tuple[bool, list[str], dict[str, dict[str, float]]]:
  """Apply exactly the v17 diagonal score, fall, KL, and pairing gates."""
  reasons = specialist_candidate_precheck(
    update_metrics=update_metrics,
    parameters_finite=parameters_finite,
    thresholds=thresholds,
  )
  old_signature = old_eval.get("initial_state_signatures")
  candidate_signature = candidate_eval.get("initial_state_signatures")
  if old_signature is None or candidate_signature is None:
    reasons.append("paired target initial-state signature missing")
  elif old_signature != candidate_signature:
    reasons.append("paired target initial-state signature differs")
  try:
    old_score = specialist_target_score(old_eval)
    candidate_score = specialist_target_score(candidate_eval)
    old_fall = float(old_eval["fall_rate"])
    candidate_fall = float(candidate_eval["fall_rate"])
  except (KeyError, TypeError, ValueError):
    reasons.append("target evaluation is missing or non-finite")
    old_score = {"total": float("nan")}
    candidate_score = {"total": float("nan")}
  else:
    if candidate_score["total"] <= old_score["total"]:
      reasons.append("target point-estimate score did not improve")
    if candidate_fall > old_fall + thresholds.maximum_target_fall_increase:
      reasons.append("target fall rate increased by more than 3 percentage points")
  return len(reasons) == 0, reasons, {
    "old": old_score,
    "candidate": candidate_score,
  }


def specialist_d0_retention_gate(
  *,
  baseline_eval: dict[str, Any],
  candidate_eval: dict[str, Any],
  thresholds: SpecialistGateThresholds = SpecialistGateThresholds(),
) -> tuple[bool, list[str]]:
  """Check only D0 success against the common base policy every two rounds."""
  reasons: list[str] = []
  baseline_success = float(baseline_eval.get("success_rate", float("nan")))
  candidate_success = float(candidate_eval.get("success_rate", float("nan")))
  if not math.isfinite(baseline_success) or not math.isfinite(candidate_success):
    reasons.append("D0 success rate is missing or non-finite")
  elif candidate_success < baseline_success - thresholds.d0_success_tolerance:
    reasons.append("D0 success is more than 5 percentage points below baseline")
  baseline_signature = baseline_eval.get("initial_state_signatures")
  candidate_signature = candidate_eval.get("initial_state_signatures")
  if baseline_signature is None or candidate_signature is None:
    reasons.append("paired D0 initial-state signature missing")
  elif baseline_signature != candidate_signature:
    reasons.append("paired D0 initial-state signature differs")
  return len(reasons) == 0, reasons


def redistributed_fall_credit(
  fall_events: torch.Tensor,
  dones: torch.Tensor,
  *,
  horizon: int = 100,
  decay: float = 0.97,
  amount_per_fall: float = 2.0,
) -> torch.Tensor:
  """Redistribute a fixed fall amount backward without crossing episodes.

  The returned tensor is a positive penalty magnitude. For every fall event,
  exactly ``amount_per_fall`` is assigned to at most ``horizon`` transitions,
  ending at the terminal transition. The caller subtracts it from the single
  scalar reward stream.
  """
  if fall_events.shape != dones.shape or fall_events.ndim != 2:
    raise ValueError("fall events and dones must share [T, N] shape")
  if horizon < 1:
    raise ValueError("fall redistribution horizon must be positive")
  if not 0.0 < decay <= 1.0:
    raise ValueError("fall redistribution decay must be in (0, 1]")
  if not math.isfinite(amount_per_fall) or amount_per_fall < 0.0:
    raise ValueError("fall redistribution amount must be finite and non-negative")
  credit = torch.zeros_like(fall_events, dtype=torch.float32)
  episode_start = torch.zeros(
    fall_events.shape[1], dtype=torch.long, device=fall_events.device
  )
  for step in range(fall_events.shape[0]):
    fall_ids = fall_events[step].bool().nonzero(as_tuple=False).flatten()
    for env_id in fall_ids.tolist():
      start = max(int(episode_start[env_id]), step - horizon + 1)
      indices = torch.arange(step, start - 1, -1, device=fall_events.device)
      powers = torch.arange(len(indices), device=fall_events.device)
      weights = decay ** powers.float()
      weights = amount_per_fall * weights / weights.sum()
      credit[indices, env_id] += weights
    done_ids = dones[step].bool().nonzero(as_tuple=False).flatten()
    if len(done_ids) > 0:
      episode_start[done_ids] = step + 1
  expected = amount_per_fall * float(fall_events.float().sum())
  if not math.isclose(
    float(credit.sum()), expected, rel_tol=1.0e-5, abs_tol=1.0e-5
  ):
    raise RuntimeError("redistributed fall penalty does not preserve episode total")
  return credit


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


def generalized_cost_advantage(
  costs: torch.Tensor,
  values: torch.Tensor,
  last_values: torch.Tensor,
  dones: torch.Tensor,
  *,
  gamma: float,
  lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute an undiscounted-sign cost GAE without crossing episode ends.

  All rollout tensors use ``[T, N]`` layout.  Timeout bootstraps, when
  required, must already have been added to ``costs`` in the same way that
  :class:`rsl_rl.algorithms.PPO` augments task rewards.
  """
  if not (costs.shape == values.shape == dones.shape):
    raise ValueError("costs, values, and dones must share [T, N] shape")
  if last_values.shape != costs.shape[1:]:
    raise ValueError("last cost values must have [N] shape")
  if costs.ndim != 2:
    raise ValueError("cost GAE inputs must be two-dimensional")
  scalars = torch.tensor([gamma, lam], dtype=torch.float64)
  if not bool(torch.isfinite(scalars).all()) or not 0.0 <= gamma <= 1.0:
    raise ValueError("cost GAE gamma must be finite and in [0, 1]")
  if not 0.0 <= lam <= 1.0:
    raise ValueError("cost GAE lambda must be finite and in [0, 1]")
  if not bool(
    torch.isfinite(costs).all()
    and torch.isfinite(values).all()
    and torch.isfinite(last_values).all()
  ):
    raise RuntimeError("cost GAE inputs contain non-finite values")

  advantages = torch.zeros_like(costs)
  advantage = torch.zeros_like(last_values)
  for step in reversed(range(costs.shape[0])):
    next_values = last_values if step == costs.shape[0] - 1 else values[step + 1]
    not_terminal = 1.0 - dones[step].float()
    delta = costs[step] + gamma * not_terminal * next_values - values[step]
    advantage = delta + gamma * lam * not_terminal * advantage
    advantages[step] = advantage
  return advantages, advantages + values


def projected_lagrange_update(
  multiplier: float,
  observed_cost: float,
  cost_budget: float,
  *,
  learning_rate: float,
  maximum: float,
) -> float:
  """Take one projected dual-ascent step for a non-negative cost multiplier."""
  values = torch.tensor(
    [multiplier, observed_cost, cost_budget, learning_rate, maximum],
    dtype=torch.float64,
  )
  if not bool(torch.isfinite(values).all()):
    raise ValueError("Lagrange update inputs must be finite")
  if multiplier < 0.0 or observed_cost < 0.0 or cost_budget < 0.0:
    raise ValueError("multipliers, observed costs, and budgets must be non-negative")
  if learning_rate < 0.0 or maximum <= 0.0:
    raise ValueError("dual learning rate/bound must be non-negative/positive")
  return float(
    min(maximum, max(0.0, multiplier + learning_rate * (observed_cost - cost_budget)))
  )


def future_event_labels(
  events: torch.Tensor,
  dones: torch.Tensor,
  *,
  horizon: int,
) -> torch.Tensor:
  """Label whether an event occurs now or soon without crossing a reset."""
  if events.shape != dones.shape or events.ndim != 2:
    raise ValueError("events and dones must share two-dimensional [T, N] shape")
  if horizon < 1:
    raise ValueError("future-event horizon must be positive")
  labels = events.bool().clone()
  for start in range(events.shape[0]):
    alive = torch.ones(events.shape[1], dtype=torch.bool, device=events.device)
    stop = min(events.shape[0], start + horizon + 1)
    for step in range(start + 1, stop):
      alive &= ~dones[step - 1].bool()
      labels[start] |= alive & events[step].bool()
  return labels


def success_gated_correction_mask(
  intervened: torch.Tensor,
  stair_indices: torch.Tensor,
  task_advantages: torch.Tensor,
  dones: torch.Tensor,
  fall_events: torch.Tensor,
  *,
  horizon: int,
) -> torch.Tensor:
  """Select CBF corrections followed by local progress or positive task value.

  A correction is eligible when it was actually executed, did not coincide
  with a fall, and either the task-only advantage is positive or the robot
  crosses a later riser within ``horizon`` steps before an episode boundary.
  This prevents the safety filter from becoming an unconditional behavior-
  cloning teacher.
  """
  shapes = {
    intervened.shape,
    stair_indices.shape,
    task_advantages.shape,
    dones.shape,
    fall_events.shape,
  }
  if len(shapes) != 1 or intervened.ndim != 2:
    raise ValueError("success-gate inputs must share two-dimensional [T, N] shape")
  if horizon < 1:
    raise ValueError("correction success horizon must be positive")
  progressed = torch.zeros_like(intervened, dtype=torch.bool)
  for start in range(stair_indices.shape[0]):
    alive = torch.ones(
      stair_indices.shape[1], dtype=torch.bool, device=stair_indices.device
    )
    stop = min(stair_indices.shape[0], start + horizon + 1)
    for step in range(start + 1, stop):
      alive &= ~dones[step - 1].bool()
      progressed[start] |= alive & (stair_indices[step] > stair_indices[start])
  return (
    intervened.bool()
    & ~fall_events.bool()
    & (progressed | (task_advantages > 0.0))
  )


def binary_risk_metrics(
  logits: torch.Tensor,
  labels: torch.Tensor,
) -> dict[str, float | int | None]:
  """Return dependency-free short-horizon risk calibration diagnostics."""
  if logits.shape != labels.shape:
    raise ValueError("risk logits and labels must have identical shape")
  flat_logits = logits.flatten()
  flat_labels = labels.flatten().bool()
  if not bool(torch.isfinite(flat_logits).all()):
    raise RuntimeError("risk logits contain non-finite values")
  probabilities = torch.sigmoid(flat_logits)
  targets = flat_labels.float()
  positives = int(flat_labels.sum())
  negatives = int((~flat_labels).sum())
  brier = float(torch.mean((probabilities - targets).square()))
  predicted = probabilities >= 0.5
  true_positive = int((predicted & flat_labels).sum())
  precision = true_positive / max(1, int(predicted.sum()))
  recall = true_positive / max(1, positives)
  auc: float | None = None
  if positives > 0 and negatives > 0:
    # Mann--Whitney form with an explicit half credit for tied scores.  This
    # keeps an untrained constant-logit head at AUC 0.5 instead of making the
    # result depend on the arbitrary ordering returned by argsort.
    _, inverse = torch.unique(probabilities, sorted=True, return_inverse=True)
    group_count = int(inverse.max()) + 1
    positive_by_score = torch.zeros(group_count, device=logits.device).scatter_add_(
      0, inverse, targets
    )
    negative_by_score = torch.zeros(group_count, device=logits.device).scatter_add_(
      0, inverse, 1.0 - targets
    )
    negative_below = torch.cumsum(negative_by_score, dim=0) - negative_by_score
    concordant = torch.sum(
      positive_by_score * (negative_below + 0.5 * negative_by_score)
    )
    auc = float(concordant / (positives * negatives))
  return {
    "count": int(flat_labels.numel()),
    "positive_count": positives,
    "positive_fraction": positives / max(1, int(flat_labels.numel())),
    "brier": brier,
    "precision_at_0_5": precision,
    "recall_at_0_5": recall,
    "auc": auc,
  }


def critic_readiness_reasons(
  diagnostics: dict[str, Any],
  *,
  late_risers: tuple[int, ...],
  minimum_samples_per_riser: int,
  minimum_fall_events: int,
  maximum_risk_brier: float,
  minimum_risk_auc: float = 0.0,
  minimum_pre_fall_cost_rise: float | None = None,
) -> list[str]:
  """Check local late-stair/failure evidence rather than global EV alone."""
  reasons: list[str] = []
  calibration = diagnostics.get("critic_calibration_by_riser", {})
  for riser in late_risers:
    count = int(calibration.get(str(riser), {}).get("count", 0))
    if count < minimum_samples_per_riser:
      reasons.append(
        f"riser {riser} critic coverage {count} < {minimum_samples_per_riser}"
      )
  fall_events = int(diagnostics.get("pre_fall_value_event_count", 0))
  if fall_events < minimum_fall_events:
    reasons.append(
      f"pre-fall critic events {fall_events} < {minimum_fall_events}"
    )
  risk = diagnostics.get("risk_prediction_after_update", {})
  brier = risk.get("brier")
  if brier is None or not math.isfinite(float(brier)):
    reasons.append("short-horizon risk Brier score missing or non-finite")
  elif float(brier) > maximum_risk_brier:
    reasons.append(
      f"short-horizon risk Brier {float(brier):.4f} > {maximum_risk_brier:.4f}"
    )
  auc = risk.get("auc")
  if minimum_risk_auc > 0.0:
    if auc is None or not math.isfinite(float(auc)):
      reasons.append("short-horizon risk AUC missing or non-finite")
    elif float(auc) < minimum_risk_auc:
      reasons.append(
        f"short-horizon risk AUC {float(auc):.4f} < {minimum_risk_auc:.4f}"
      )
  if minimum_pre_fall_cost_rise is not None:
    if not math.isfinite(minimum_pre_fall_cost_rise):
      raise ValueError("minimum pre-fall cost rise must be finite when enabled")
    cost_rise = diagnostics.get("pre_fall_cost_value_delta_after_update")
    if cost_rise is None or not math.isfinite(float(cost_rise)):
      reasons.append("pre-fall cost-value delta missing or non-finite")
    elif float(cost_rise) < minimum_pre_fall_cost_rise:
      reasons.append(
        "pre-fall cost value did not rise enough: "
        f"{float(cost_rise):.4f} < {minimum_pre_fall_cost_rise:.4f}"
      )
  return reasons


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
    retention_domain=retention_domain,
    neighbor_domain=neighbor_domain,
    old_total_kl_from_base=old_total_kl_from_base,
    total_kl_from_base=total_kl_from_base,
    score_weights=score_weights,
  )
  success_delta = intervals["target_success_delta_95"]
  fall_delta = intervals["target_fall_delta_95"]
  intervention_ratio_delta = intervals["target_intervention_ratio_delta_95"]
  intervention_delta = intervals["target_intervention_delta_95"]
  return_delta = intervals["target_return_delta_95"]
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
  task_better = (
    success_delta[1] > 0.0
    or fall_delta[2] < 0.0
    or return_delta[1] > 0.0
  )
  if thresholds.require_task_improvement and not task_better:
    reasons.append("target task metrics show no strict improvement")
  safe_score_delta = intervals["target_safe_improvement_score_delta_95"]
  if safe_score_delta[0] <= thresholds.minimum_safe_improvement:
    reasons.append("target safe improvement score did not increase")
  if (
    candidate_eval[retention_domain]["success_rate"]
    < base_d0_success - thresholds.d0_retention_tolerance
  ):
    reasons.append(f"{retention_domain} retention bound violated")
  retention_success_delta = intervals[
    "retention_success_tolerance_delta_95"
  ]
  retention_fall_delta = intervals["retention_fall_tolerance_delta_95"]
  if retention_success_delta[2] < 0.0:
    reasons.append(f"{retention_domain} paired success regressed")
  if retention_fall_delta[1] > 0.0:
    reasons.append(f"{retention_domain} fall rate increased")
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
  retention_domain: str = "D0",
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
  old_retention = old_eval.get(retention_domain)
  candidate_retention = candidate_eval.get(retention_domain)
  old_neighbor = old_eval[neighbor_domain]
  candidate_neighbor = candidate_eval[neighbor_domain]
  intervals = {
    "target_success_delta_95": paired_metric_delta_interval(
      old_target, candidate_target, "success_rate"
    ),
    "target_fall_delta_95": paired_metric_delta_interval(
      old_target, candidate_target, "fall_rate"
    ),
    "target_return_delta_95": paired_value_delta_interval(
      old_target,
      candidate_target,
      old_value=lambda result: float(result.get("mean_return", 0.0)),
      candidate_value=lambda result: float(result.get("mean_return", 0.0)),
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
  if old_retention is not None and candidate_retention is not None:
    intervals.update(
      {
        "retention_success_tolerance_delta_95": paired_value_delta_interval(
          old_retention,
          candidate_retention,
          old_value=lambda result: float(result["success_rate"])
          - thresholds.d0_retention_tolerance,
          candidate_value=lambda result: float(result["success_rate"]),
        ),
        "retention_fall_tolerance_delta_95": paired_value_delta_interval(
          old_retention,
          candidate_retention,
          old_value=lambda result: float(result["fall_rate"])
          + thresholds.d0_fall_tolerance,
          candidate_value=lambda result: float(result["fall_rate"]),
        ),
      }
    )
  return intervals


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


def hierarchical_specialist_macro_interval(
  scene_seed_deltas: list[list[torch.Tensor]],
  *,
  bootstrap_samples: int = 10000,
  bootstrap_seed: int = 0,
) -> tuple[float, float, float]:
  """Resample three scenes, adaptation seeds, then paired episode deltas."""
  if len(scene_seed_deltas) != 3 or any(
    len(seed_groups) != 3 for seed_groups in scene_seed_deltas
  ):
    raise ValueError("macro bootstrap requires three scenes and three seeds each")
  lengths = {
    int(group.numel())
    for seed_groups in scene_seed_deltas
    for group in seed_groups
  }
  if len(lengths) != 1 or 0 in lengths:
    raise ValueError("macro bootstrap episode groups must have one non-zero size")
  if bootstrap_samples < 1000:
    raise ValueError("formal macro bootstrap requires at least 1000 samples")
  values = torch.stack(
    [torch.stack(seed_groups) for seed_groups in scene_seed_deltas]
  ).to(dtype=torch.float64, device="cpu")
  if not bool(torch.isfinite(values).all()):
    raise ValueError("macro bootstrap deltas contain non-finite values")
  scene_count, seed_count, episode_count = values.shape
  generator = torch.Generator(device="cpu")
  generator.manual_seed(bootstrap_seed)
  means: list[torch.Tensor] = []
  chunk_size = 100
  for start in range(0, bootstrap_samples, chunk_size):
    count = min(chunk_size, bootstrap_samples - start)
    sampled_scene = torch.randint(
      scene_count, (count, scene_count), generator=generator
    )
    sampled_seed = torch.randint(
      seed_count, (count, scene_count, seed_count), generator=generator
    )
    samples = torch.empty(
      (count, scene_count, seed_count, episode_count), dtype=torch.float64
    )
    for draw in range(count):
      for scene_slot in range(scene_count):
        source_scene = int(sampled_scene[draw, scene_slot])
        for seed_slot in range(seed_count):
          source_seed = int(sampled_seed[draw, scene_slot, seed_slot])
          episode_ids = torch.randint(
            episode_count, (episode_count,), generator=generator
          )
          samples[draw, scene_slot, seed_slot] = values[
            source_scene, source_seed, episode_ids
          ]
    means.append(samples.mean(dim=(1, 2, 3)))
  bootstrap_means = torch.cat(means)
  lower, upper = torch.quantile(
    bootstrap_means, torch.tensor((0.025, 0.975), dtype=torch.float64)
  )
  point = torch.stack(
    [torch.cat(seed_groups).to(torch.float64).mean() for seed_groups in scene_seed_deltas]
  ).mean()
  return float(point), float(lower), float(upper)


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
  """Return an interpolation/extrapolation on one PPO update direction.

  Only trainable actor MLP tensors are interpolated.  Frozen observation
  normalization and bounded distribution-variance state come from the
  candidate checkpoint unchanged.  This is a policy line search, not a new
  optimizer objective or a residual controller.  Fractions through 1.0 are
  backtracks; 1.5 is the bounded extrapolation proposed for candidate-family
  screening.
  """
  if not 0.0 <= fraction <= 1.5:
    raise ValueError(f"line-search fraction must be in [0, 1.5], got {fraction}")
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
  d0_retention_anchor_weight: float = 0.0
  neighbor_retention_anchor_weight: float = 0.0
  d0_retention_anchor_kl_budget: float = 0.002
  neighbor_retention_anchor_kl_budget: float = 0.002
  retention_anchor_adaptation_rate: float = 10.0
  maximum_retention_anchor_weight: float = 0.20
  retention_anchor_batch_size: int = 4096
  intervention_advantage_weight: float = 0.075
  safe_bc_weight: float = 0.0
  use_counterfactual_cbf_credit: bool = False
  task_first_constrained: bool = False
  brief_ppo_refinement: bool = False
  failure_focused_refinement: bool = False
  kl_early_stopping: bool = False
  fall_redistribution_horizon: int = 100
  fall_redistribution_decay: float = 0.97
  fall_redistribution_amount: float = 2.0
  initial_fall_multiplier: float = 0.0
  initial_intervention_multiplier: float = 0.0
  fall_multiplier_learning_rate: float = 1.0
  intervention_multiplier_learning_rate: float = 0.10
  maximum_cost_multiplier: float = 20.0
  fall_cost_budget: float = 0.0
  intervention_cost_budget: float = 0.0
  hard_case_policy_weight: float = 0.0
  correction_distillation_weight: float = 0.0
  correction_success_horizon: int = 100
  risk_horizon: int = 50
  strong_intervention_fraction: float = 0.5
  risk_loss_coef: float = 1.0


class OnlineSafePPO(PPO):
  """Single-clipped PPO with optional task-first constrained safety costs."""

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
    d0_retention_anchor_weight: float = 0.0,
    neighbor_retention_anchor_weight: float = 0.0,
    d0_retention_anchor_kl_budget: float = 0.002,
    neighbor_retention_anchor_kl_budget: float = 0.002,
    retention_anchor_adaptation_rate: float = 10.0,
    maximum_retention_anchor_weight: float = 0.20,
    retention_anchor_batch_size: int = 4096,
    intervention_advantage_weight: float = 0.075,
    safe_bc_weight: float = 0.0,
    use_counterfactual_cbf_credit: bool = False,
    task_first_constrained: bool = False,
    brief_ppo_refinement: bool = False,
    failure_focused_refinement: bool = False,
    kl_early_stopping: bool = False,
    fall_redistribution_horizon: int = 100,
    fall_redistribution_decay: float = 0.97,
    fall_redistribution_amount: float = 2.0,
    initial_fall_multiplier: float = 0.0,
    initial_intervention_multiplier: float = 0.0,
    fall_multiplier_learning_rate: float = 1.0,
    intervention_multiplier_learning_rate: float = 0.10,
    maximum_cost_multiplier: float = 20.0,
    fall_cost_budget: float = 0.0,
    intervention_cost_budget: float = 0.0,
    hard_case_policy_weight: float = 0.0,
    correction_distillation_weight: float = 0.0,
    correction_success_horizon: int = 100,
    risk_horizon: int = 50,
    strong_intervention_fraction: float = 0.5,
    risk_loss_coef: float = 1.0,
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
    self.d0_retention_anchor_weight = d0_retention_anchor_weight
    self.neighbor_retention_anchor_weight = neighbor_retention_anchor_weight
    self.d0_retention_anchor_kl_budget = d0_retention_anchor_kl_budget
    self.neighbor_retention_anchor_kl_budget = (
      neighbor_retention_anchor_kl_budget
    )
    self.retention_anchor_adaptation_rate = retention_anchor_adaptation_rate
    self.maximum_retention_anchor_weight = maximum_retention_anchor_weight
    self.retention_anchor_batch_size = retention_anchor_batch_size
    self.intervention_advantage_weight = intervention_advantage_weight
    self.safe_bc_weight = safe_bc_weight
    self.use_counterfactual_cbf_credit = use_counterfactual_cbf_credit
    self.task_first_constrained = task_first_constrained
    self.brief_ppo_refinement = brief_ppo_refinement
    self.failure_focused_refinement = failure_focused_refinement
    self.kl_early_stopping = kl_early_stopping
    self.fall_redistribution_horizon = fall_redistribution_horizon
    self.fall_redistribution_decay = fall_redistribution_decay
    self.fall_redistribution_amount = fall_redistribution_amount
    self.fall_multiplier = float(initial_fall_multiplier)
    self.intervention_multiplier = float(initial_intervention_multiplier)
    self.fall_multiplier_learning_rate = fall_multiplier_learning_rate
    self.intervention_multiplier_learning_rate = (
      intervention_multiplier_learning_rate
    )
    self.maximum_cost_multiplier = maximum_cost_multiplier
    self.fall_cost_budget = fall_cost_budget
    self.intervention_cost_budget = intervention_cost_budget
    self.hard_case_policy_weight = hard_case_policy_weight
    self.correction_distillation_weight = correction_distillation_weight
    self.correction_success_horizon = correction_success_horizon
    self.risk_horizon = risk_horizon
    self.strong_intervention_fraction = strong_intervention_fraction
    self.risk_loss_coef = risk_loss_coef
    constraint_scalars = torch.tensor(
      [
        self.fall_multiplier,
        self.intervention_multiplier,
        self.fall_multiplier_learning_rate,
        self.intervention_multiplier_learning_rate,
        self.maximum_cost_multiplier,
        self.fall_cost_budget,
        self.intervention_cost_budget,
        self.hard_case_policy_weight,
        self.correction_distillation_weight,
        float(self.correction_success_horizon),
        float(self.risk_horizon),
        self.strong_intervention_fraction,
        self.risk_loss_coef,
        self.base_anchor_weight,
        self.d0_retention_anchor_weight,
        self.neighbor_retention_anchor_weight,
        self.d0_retention_anchor_kl_budget,
        self.neighbor_retention_anchor_kl_budget,
        self.retention_anchor_adaptation_rate,
        self.maximum_retention_anchor_weight,
      ],
      dtype=torch.float64,
    )
    if not bool(torch.isfinite(constraint_scalars).all()):
      raise ValueError("task-first constrained PPO parameters must be finite")
    if self.fall_multiplier < 0.0 or self.intervention_multiplier < 0.0:
      raise ValueError("cost multipliers must be non-negative")
    if self.maximum_cost_multiplier <= 0.0:
      raise ValueError("maximum cost multiplier must be positive")
    if not 0.0 <= self.hard_case_policy_weight <= 1.0:
      raise ValueError("hard-case policy weight must be in [0, 1]")
    if self.correction_success_horizon < 1 or self.risk_horizon < 1:
      raise ValueError("correction/risk horizons must be positive")
    if not 0.0 < self.strong_intervention_fraction <= 1.0:
      raise ValueError("strong intervention fraction must be in (0, 1]")
    if self.base_anchor_weight < 0.0:
      raise ValueError("base anchor weight must be non-negative")
    if (
      self.d0_retention_anchor_weight < 0.0
      or self.neighbor_retention_anchor_weight < 0.0
    ):
      raise ValueError("retention anchor weights must be non-negative")
    if (
      self.d0_retention_anchor_kl_budget < 0.0
      or self.neighbor_retention_anchor_kl_budget < 0.0
    ):
      raise ValueError("retention anchor KL budgets must be non-negative")
    if self.retention_anchor_adaptation_rate < 0.0:
      raise ValueError("retention anchor adaptation rate must be non-negative")
    if self.maximum_retention_anchor_weight <= 0.0:
      raise ValueError("maximum retention anchor weight must be positive")
    if max(
      self.d0_retention_anchor_weight,
      self.neighbor_retention_anchor_weight,
    ) > self.maximum_retention_anchor_weight:
      raise ValueError("initial retention anchor weight exceeds its maximum")
    if self.retention_anchor_batch_size < 1:
      raise ValueError("retention anchor batch size must be positive")
    if self.task_first_constrained and self.brief_ppo_refinement:
      raise ValueError("brief PPO cannot enable task-first constrained heads")
    if self.failure_focused_refinement and not self.brief_ppo_refinement:
      raise ValueError("failure-focused refinement must use brief PPO")
    if self.fall_redistribution_horizon < 1:
      raise ValueError("fall redistribution horizon must be positive")
    if not 0.0 < self.fall_redistribution_decay <= 1.0:
      raise ValueError("fall redistribution decay must be in (0, 1]")
    if self.fall_redistribution_amount < 0.0 or not math.isfinite(
      self.fall_redistribution_amount
    ):
      raise ValueError("fall redistribution amount must be finite and non-negative")
    if self.failure_focused_refinement and (
      self.fall_redistribution_horizon != 100
      or not math.isclose(
        self.fall_redistribution_decay, 0.97, rel_tol=0.0, abs_tol=1.0e-12
      )
      or not math.isclose(
        self.fall_redistribution_amount, 2.0, rel_tol=0.0, abs_tol=1.0e-12
      )
    ):
      raise ValueError(
        "failure-focused brief PPO requires fall redistribution (100, 0.97, 2.0)"
      )
    if self.kl_early_stopping and (
      self.desired_kl is None or self.desired_kl <= 0.0
    ):
      raise ValueError("KL early stopping requires a positive desired_kl")
    if self.brief_ppo_refinement:
      disabled_terms = {
        "pre_intervention_weight": self.pre_intervention_weight,
        "intervention_advantage_weight": self.intervention_advantage_weight,
        "base_anchor_weight": self.base_anchor_weight,
        "d0_retention_anchor_weight": self.d0_retention_anchor_weight,
        "neighbor_retention_anchor_weight": self.neighbor_retention_anchor_weight,
        "safe_bc_weight": self.safe_bc_weight,
        "correction_distillation_weight": self.correction_distillation_weight,
        "log_std_learning_rate": self.log_std_learning_rate,
      }
      enabled = [name for name, value in disabled_terms.items() if value != 0.0]
      if enabled:
        raise ValueError(
          "brief PPO requires all auxiliary policy losses/anchors disabled: "
          + ", ".join(enabled)
        )
      if self.num_learning_epochs != 1:
        raise ValueError("brief PPO requires exactly one learning epoch")
      if not math.isclose(self.clip_param, 0.05, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("brief PPO requires PPO clip 0.05")
      required_target_kl = 0.003 if self.failure_focused_refinement else 0.005
      if not math.isclose(
        float(self.desired_kl or 0.0),
        required_target_kl,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      ):
        raise ValueError(
          f"brief PPO mode requires target KL {required_target_kl}"
        )
      if not self.kl_early_stopping:
        raise ValueError("brief PPO requires target-KL early stopping")
      required_actor_lr = 5.0e-6 if self.failure_focused_refinement else 2.0e-6
      if not math.isclose(
        self.actor_learning_rate,
        required_actor_lr,
        rel_tol=0.0,
        abs_tol=1.0e-15,
      ):
        raise ValueError(
          f"brief PPO mode requires actor learning rate {required_actor_lr}"
        )
      if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        for value in self.actor_layer_multipliers
      ):
        raise ValueError("brief PPO requires the full actor at the configured LR")
      if not 0.25 <= self.std_scale_from_base <= 0.50:
        raise ValueError("brief PPO exploration std scale must be in [0.25, 0.50]")
      if self.failure_focused_refinement and not math.isclose(
        self.std_scale_from_base, 0.35, rel_tol=0.0, abs_tol=1.0e-12
      ):
        raise ValueError("failure-focused brief PPO requires base std scale 0.35")
      if self.failure_focused_refinement and not math.isclose(
        self.critic_learning_rate, 1.0e-4, rel_tol=0.0, abs_tol=1.0e-15
      ):
        raise ValueError("failure-focused brief PPO requires critic LR 1e-4")
      required_hard_weight = 0.75 if self.failure_focused_refinement else 0.5
      if not math.isclose(
        self.hard_case_policy_weight,
        required_hard_weight,
        rel_tol=0.0,
        abs_tol=1.0e-12,
      ):
        raise ValueError(
          f"brief PPO mode requires hard-case actor weight {required_hard_weight}"
        )

    # The task critic retains the pretrained value representation.  Cost and
    # short-horizon risk heads are distinct modules so safety cannot silently
    # reshape the task-value target through fixed reward scalarization.
    self.fall_critic = (
      copy.deepcopy(self.critic) if self.task_first_constrained else None
    )
    self.intervention_critic = (
      copy.deepcopy(self.critic) if self.task_first_constrained else None
    )
    self.risk_head = (
      copy.deepcopy(self.critic) if self.task_first_constrained else None
    )
    self._critic_only = False
    self._std_initialized = False
    self.base_actor_reference = None
    self.retention_actor_reference = None
    self.retention_anchor_banks: dict[str, torch.Tensor] = {}
    self.retention_anchor_bank_metadata: dict[str, dict[str, Any]] = {}
    self.retention_anchor_cursors = {"d0": 0, "neighbor": 0}
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
    self.fall_costs = torch.zeros(t, n, device=self.device)
    self.intervention_costs = torch.zeros(t, n, device=self.device)
    self.fall_cost_values = torch.zeros(t, n, device=self.device)
    self.intervention_cost_values = torch.zeros(t, n, device=self.device)
    self.fall_cost_returns = torch.zeros(t, n, device=self.device)
    self.intervention_cost_returns = torch.zeros(t, n, device=self.device)
    self.fall_cost_advantages = torch.zeros(t, n, device=self.device)
    self.intervention_cost_advantages = torch.zeros(t, n, device=self.device)
    self.task_advantages = torch.zeros(t, n, device=self.device)
    self.risk_labels = torch.zeros(t, n, dtype=torch.bool, device=self.device)
    self.successful_correction = torch.zeros(
      t, n, dtype=torch.bool, device=self.device
    )
    self.hard_case_transitions = torch.zeros(
      t, n, dtype=torch.bool, device=self.device
    )
    self.timeout_events = torch.zeros(t, n, dtype=torch.bool, device=self.device)
    self._pending_fall_values: torch.Tensor | None = None
    self._pending_intervention_values: torch.Tensor | None = None
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
    if self.task_first_constrained:
      assert self.fall_critic is not None
      assert self.intervention_critic is not None
      assert self.risk_head is not None
      for role, module in (
        ("fall_critic", self.fall_critic),
        ("intervention_critic", self.intervention_critic),
        ("risk_head", self.risk_head),
      ):
        groups.append(
          {
            "params": list(module.parameters()),
            "lr": self.critic_learning_rate,
            "base_lr": self.critic_learning_rate,
            "role": role,
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
    for module in (self.fall_critic, self.intervention_critic, self.risk_head):
      if module is not None:
        module.train()

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

  def _set_retention_actor_reference(
    self, state: dict[str, torch.Tensor] | None = None
  ) -> None:
    """Freeze the deployed pre-adaptation actor used by both fixed banks."""
    if self.retention_actor_reference is None:
      # GaussianDistribution caches the latest Normal, whose mean may be an
      # inference-mode view. It is transient and absent from state_dict; do
      # not ask deepcopy to traverse it during checkpoint restoration.
      distribution = getattr(self.actor, "distribution", None)
      has_cached_distribution = hasattr(distribution, "_distribution")
      cached_distribution = (
        distribution._distribution if has_cached_distribution else None
      )
      if has_cached_distribution:
        distribution._distribution = None
      try:
        reference = copy.deepcopy(self.actor)
      finally:
        if has_cached_distribution:
          distribution._distribution = cached_distribution
    else:
      reference = self.retention_actor_reference
      distribution = getattr(reference, "distribution", None)
      if hasattr(distribution, "_distribution"):
        distribution._distribution = None
    if state is not None:
      reference.load_state_dict(state, strict=True)
    reference.eval()
    for parameter in reference.parameters():
      parameter.requires_grad_(False)
    self.retention_actor_reference = reference

  @staticmethod
  def _retention_bank_identity(metadata: dict[str, Any]) -> tuple[Any, ...]:
    return (
      metadata.get("domain"),
      metadata.get("size"),
      metadata.get("actor_observation_dim"),
      metadata.get("checkpoint_sha256"),
      metadata.get("observation_sha256"),
    )

  def set_retention_anchor_banks(
    self,
    *,
    d0_payload: dict[str, Any] | None,
    neighbor_payload: dict[str, Any] | None,
    neighbor_domain: str = "DQNH",
  ) -> dict[str, dict[str, Any]]:
    """Install fixed actor-only D0/neighbor banks on host memory.

    The reference actor is captured once from the deployed policy before any
    actor update. A resumed v13 checkpoint restores that reference separately,
    while the bank checksums prevent silently swapping the fixed state sets.
    """
    actor_groups = tuple(getattr(self.actor, "obs_groups", ()))
    if actor_groups != ("actor",):
      raise RuntimeError(
        "retention anchors require an actor that consumes only the actor group"
      )
    expected_actor_dim = int(getattr(self.actor, "obs_dim", -1))
    if expected_actor_dim < 1:
      raise RuntimeError("actor observation dimension is unavailable")
    payloads = {"d0": d0_payload, "neighbor": neighbor_payload}
    expected_domains = {"d0": "D0", "neighbor": neighbor_domain}
    weights = {
      "d0": self.d0_retention_anchor_weight,
      "neighbor": self.neighbor_retention_anchor_weight,
    }
    installed: dict[str, torch.Tensor] = {}
    installed_metadata: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
      if payload is None:
        if weights[name] > 0.0:
          raise ValueError(f"{name} retention anchor weight requires a bank")
        continue
      observations, metadata = validate_retention_observation_bank(
        payload,
        expected_actor_dim=expected_actor_dim,
        expected_domain=expected_domains[name],
      )
      previous = self.retention_anchor_bank_metadata.get(name)
      if previous is not None and self._retention_bank_identity(
        previous
      ) != self._retention_bank_identity(metadata):
        raise ValueError(f"{name} retention bank differs from resumed checkpoint")
      installed[name] = observations.detach().cpu().contiguous()
      installed_metadata[name] = metadata
    if installed_metadata:
      checkpoint_hashes = {
        metadata["checkpoint_sha256"] for metadata in installed_metadata.values()
      }
      if len(checkpoint_hashes) != 1 or "" in checkpoint_hashes:
        raise ValueError("retention banks must share one non-empty policy checkpoint")
      if self.retention_actor_reference is None:
        self._set_retention_actor_reference()
    self.retention_anchor_banks = installed
    self.retention_anchor_bank_metadata = installed_metadata
    for name, observations in installed.items():
      self.retention_anchor_cursors[name] %= observations.shape[0]
    return copy.deepcopy(installed_metadata)

  def _retention_anchor_loss(self, name: str) -> torch.Tensor:
    if self.retention_actor_reference is None:
      raise RuntimeError("retention actor reference is not frozen")
    observations = self.retention_anchor_banks.get(name)
    if observations is None:
      raise RuntimeError(f"{name} retention anchor bank is not installed")
    batch, cursor = cyclic_retention_batch(
      observations,
      cursor=self.retention_anchor_cursors[name],
      batch_size=self.retention_anchor_batch_size,
    )
    self.retention_anchor_cursors[name] = cursor
    actor_observations = {"actor": batch.to(self.device)}
    actor_device = torch.device(self.device)
    rng_devices = [actor_device] if actor_device.type == "cuda" else []
    # MLPModel currently exposes distribution parameters only after a
    # stochastic forward, which also samples an unused action. Restore both
    # CPU/CUDA RNG afterward so the anchor cannot perturb future rollouts or
    # PPO minibatch permutations independently of its gradient.
    with torch.random.fork_rng(devices=rng_devices):
      self.actor(actor_observations, stochastic_output=True)
      current_params = tuple(self.actor.output_distribution_params)
      with torch.no_grad():
        self.retention_actor_reference(
          actor_observations, stochastic_output=True
        )
        reference_params = tuple(
          parameter.detach()
          for parameter in (
            self.retention_actor_reference.output_distribution_params
          )
        )
    return self.actor.get_kl_divergence(
      current_params, reference_params
    ).mean()

  def _full_retention_anchor_kl(self, name: str) -> float:
    if self.retention_actor_reference is None:
      raise RuntimeError("retention actor reference is not frozen")
    observations = self.retention_anchor_banks.get(name)
    if observations is None:
      raise RuntimeError(f"{name} retention anchor bank is not installed")
    total = 0.0
    diagnostic_batch_size = max(1, self.retention_anchor_batch_size)
    actor_device = torch.device(self.device)
    rng_devices = [actor_device] if actor_device.type == "cuda" else []
    with torch.inference_mode():
      for start in range(0, observations.shape[0], diagnostic_batch_size):
        actor_observations = {
          "actor": observations[
            start : start + diagnostic_batch_size
          ].to(self.device)
        }
        with torch.random.fork_rng(devices=rng_devices):
          self.actor(actor_observations, stochastic_output=True)
          current_params = tuple(self.actor.output_distribution_params)
          self.retention_actor_reference(
            actor_observations, stochastic_output=True
          )
          reference_params = tuple(
            self.retention_actor_reference.output_distribution_params
          )
        kl = self.actor.get_kl_divergence(current_params, reference_params)
        total += float(kl.sum())
    return total / observations.shape[0]

  def retention_anchor_kl_metrics(self) -> dict[str, float]:
    """Evaluate exact mean KL on every installed fixed bank."""
    metrics: dict[str, float] = {}
    for name in ("d0", "neighbor"):
      if name in self.retention_anchor_banks:
        metrics[f"{name}_retention_anchor_kl"] = (
          self._full_retention_anchor_kl(name)
        )
    return metrics

  def adapt_retention_anchor_weights(
    self,
    *,
    d0_kl: float | None = None,
    neighbor_kl: float | None = None,
  ) -> dict[str, float]:
    """Increase independent anchor weights after selecting a candidate."""
    observed = {"d0": d0_kl, "neighbor": neighbor_kl}
    budgets = {
      "d0": self.d0_retention_anchor_kl_budget,
      "neighbor": self.neighbor_retention_anchor_kl_budget,
    }
    attributes = {
      "d0": "d0_retention_anchor_weight",
      "neighbor": "neighbor_retention_anchor_weight",
    }
    metrics: dict[str, float] = {}
    for name in ("d0", "neighbor"):
      if name not in self.retention_anchor_banks:
        continue
      kl = observed[name]
      if kl is None:
        kl = self._full_retention_anchor_kl(name)
      before = float(getattr(self, attributes[name]))
      after = increase_anchor_weight_on_budget_violation(
        before,
        float(kl),
        budgets[name],
        learning_rate=self.retention_anchor_adaptation_rate,
        maximum=self.maximum_retention_anchor_weight,
      )
      setattr(self, attributes[name], after)
      metrics.update(
        {
          f"{name}_retention_anchor_kl": float(kl),
          f"{name}_retention_anchor_kl_budget": float(budgets[name]),
          f"{name}_retention_anchor_weight_before_adaptation": before,
          f"{name}_retention_anchor_weight_after_adaptation": after,
        }
      )
    return metrics

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
    critic_roles = {"critic", "fall_critic", "intervention_critic", "risk_head"}
    for group in self.optimizer.param_groups:
      group["lr"] = (
        0.0
        if enabled and group["role"] not in critic_roles
        else group["base_lr"]
      )

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

  def initialize_task_first_heads_from_critic(self) -> None:
    """Warm-start auxiliary representations and zero their scalar outputs."""
    if not self.task_first_constrained:
      return
    assert self.fall_critic is not None
    assert self.intervention_critic is not None
    assert self.risk_head is not None
    source = self.critic.state_dict()
    for module in (self.fall_critic, self.intervention_critic, self.risk_head):
      module.load_state_dict(source, strict=True)
      linear_layers = [
        layer for layer in module.mlp if isinstance(layer, torch.nn.Linear)
      ]
      if not linear_layers:
        raise RuntimeError("task-first head has no linear output layer")
      with torch.no_grad():
        linear_layers[-1].weight.zero_()
        if linear_layers[-1].bias is not None:
          linear_layers[-1].bias.zero_()
    self.reset_online_optimizer()

  def set_cost_budgets(
    self,
    *,
    fall_rate: float,
    intervention_per_riser: float,
    intervention_slack: float = 1.05,
  ) -> None:
    """Anchor safety budgets to the evaluated accepted/base policy."""
    values = torch.tensor(
      [fall_rate, intervention_per_riser, intervention_slack], dtype=torch.float64
    )
    if not bool(torch.isfinite(values).all()):
      raise ValueError("cost budget inputs must be finite")
    if fall_rate < 0.0 or intervention_per_riser < 0.0:
      raise ValueError("cost budgets cannot be based on negative rates")
    if intervention_slack < 1.0:
      raise ValueError("intervention budget slack must be at least one")
    self.fall_cost_budget = float(fall_rate)
    self.intervention_cost_budget = float(
      intervention_slack * intervention_per_riser
    )

  def constraint_state_dict(self) -> dict[str, Any]:
    if self.brief_ppo_refinement:
      # The brief checkpoint deliberately carries no auxiliary critic,
      # multiplier, policy anchor, or retention-bank state.
      return {
        "task_first_constrained": False,
        "brief_ppo_refinement": True,
        "failure_focused_refinement": self.failure_focused_refinement,
        "fall_redistribution": {
          "horizon": self.fall_redistribution_horizon,
          "decay": self.fall_redistribution_decay,
          "amount": self.fall_redistribution_amount,
        },
      }
    output: dict[str, Any] = {
      "task_first_constrained": self.task_first_constrained,
      "fall_multiplier": self.fall_multiplier,
      "intervention_multiplier": self.intervention_multiplier,
      "fall_cost_budget": self.fall_cost_budget,
      "intervention_cost_budget": self.intervention_cost_budget,
      "retention_anchor_state": {
        "d0_weight": self.d0_retention_anchor_weight,
        "neighbor_weight": self.neighbor_retention_anchor_weight,
        "d0_kl_budget": self.d0_retention_anchor_kl_budget,
        "neighbor_kl_budget": self.neighbor_retention_anchor_kl_budget,
        "adaptation_rate": self.retention_anchor_adaptation_rate,
        "maximum_weight": self.maximum_retention_anchor_weight,
        "batch_size": self.retention_anchor_batch_size,
        "cursors": dict(self.retention_anchor_cursors),
        "bank_metadata": copy.deepcopy(self.retention_anchor_bank_metadata),
      },
    }
    if self.retention_actor_reference is not None:
      output["retention_actor_reference_state_dict"] = (
        self.retention_actor_reference.state_dict()
      )
    if self.task_first_constrained:
      assert self.fall_critic is not None
      assert self.intervention_critic is not None
      assert self.risk_head is not None
      output.update(
        fall_critic_state_dict=self.fall_critic.state_dict(),
        intervention_critic_state_dict=self.intervention_critic.state_dict(),
        risk_head_state_dict=self.risk_head.state_dict(),
      )
    return output

  def load_constraint_state_dict(self, payload: dict[str, Any]) -> bool:
    """Load v12 heads if present; return false for a legacy v11 checkpoint."""
    if self.brief_ppo_refinement:
      # A v14 run may warm-start from a v12/v13 accepted actor and task critic,
      # but none of the old online constraint state is part of the new method.
      return False
    retention = payload.get("retention_anchor_state")
    if isinstance(retention, dict):
      self.d0_retention_anchor_weight = float(
        retention.get("d0_weight", self.d0_retention_anchor_weight)
      )
      self.neighbor_retention_anchor_weight = float(
        retention.get("neighbor_weight", self.neighbor_retention_anchor_weight)
      )
      self.d0_retention_anchor_kl_budget = float(
        retention.get("d0_kl_budget", self.d0_retention_anchor_kl_budget)
      )
      self.neighbor_retention_anchor_kl_budget = float(
        retention.get(
          "neighbor_kl_budget", self.neighbor_retention_anchor_kl_budget
        )
      )
      self.retention_anchor_adaptation_rate = float(
        retention.get(
          "adaptation_rate", self.retention_anchor_adaptation_rate
        )
      )
      self.maximum_retention_anchor_weight = float(
        retention.get("maximum_weight", self.maximum_retention_anchor_weight)
      )
      self.retention_anchor_batch_size = int(
        retention.get("batch_size", self.retention_anchor_batch_size)
      )
      cursors = retention.get("cursors", {})
      if isinstance(cursors, dict):
        for name in ("d0", "neighbor"):
          cursor = int(cursors.get(name, self.retention_anchor_cursors[name]))
          if cursor < 0:
            raise ValueError("retention anchor cursor cannot be negative")
          self.retention_anchor_cursors[name] = cursor
      metadata = retention.get("bank_metadata", {})
      if not isinstance(metadata, dict):
        raise ValueError("retention anchor bank metadata must be a dictionary")
      self.retention_anchor_bank_metadata = copy.deepcopy(metadata)
      validation_values = torch.tensor(
        [
          self.d0_retention_anchor_weight,
          self.neighbor_retention_anchor_weight,
          self.d0_retention_anchor_kl_budget,
          self.neighbor_retention_anchor_kl_budget,
          self.retention_anchor_adaptation_rate,
          self.maximum_retention_anchor_weight,
        ],
        dtype=torch.float64,
      )
      if not bool(torch.isfinite(validation_values).all()):
        raise ValueError("checkpoint retention anchor state is non-finite")
      if bool((validation_values[:5] < 0.0).any()):
        raise ValueError("checkpoint retention anchor state is negative")
      if (
        self.maximum_retention_anchor_weight <= 0.0
        or max(
          self.d0_retention_anchor_weight,
          self.neighbor_retention_anchor_weight,
        ) > self.maximum_retention_anchor_weight
        or self.retention_anchor_batch_size < 1
      ):
        raise ValueError("checkpoint retention anchor bounds are inconsistent")
    reference_state = payload.get("retention_actor_reference_state_dict")
    if reference_state is not None:
      if not isinstance(reference_state, dict):
        raise ValueError("retention actor reference state must be a dictionary")
      self._set_retention_actor_reference(reference_state)
    if not self.task_first_constrained:
      return False
    required = (
      "fall_critic_state_dict",
      "intervention_critic_state_dict",
      "risk_head_state_dict",
    )
    if any(key not in payload for key in required):
      self.initialize_task_first_heads_from_critic()
      return False
    assert self.fall_critic is not None
    assert self.intervention_critic is not None
    assert self.risk_head is not None
    self.fall_critic.load_state_dict(payload["fall_critic_state_dict"], strict=True)
    self.intervention_critic.load_state_dict(
      payload["intervention_critic_state_dict"], strict=True
    )
    self.risk_head.load_state_dict(payload["risk_head_state_dict"], strict=True)
    self.fall_multiplier = float(payload.get("fall_multiplier", self.fall_multiplier))
    self.intervention_multiplier = float(
      payload.get("intervention_multiplier", self.intervention_multiplier)
    )
    self.fall_cost_budget = float(
      payload.get("fall_cost_budget", self.fall_cost_budget)
    )
    self.intervention_cost_budget = float(
      payload.get("intervention_cost_budget", self.intervention_cost_budget)
    )
    return True

  def save(self) -> dict[str, Any]:
    saved = super().save()
    saved.update(self.constraint_state_dict())
    return saved

  def act(self, obs) -> torch.Tensor:
    actions = super().act(obs)
    if self.task_first_constrained:
      assert self.fall_critic is not None
      assert self.intervention_critic is not None
      with torch.no_grad():
        self._pending_fall_values = self.fall_critic(obs).detach().squeeze(-1)
        self._pending_intervention_values = (
          self.intervention_critic(obs).detach().squeeze(-1)
        )
    return actions

  def update_cost_multipliers(self) -> dict[str, float]:
    """Update both dual variables from rollout-level safety rates."""
    if not self.task_first_constrained:
      return {}
    observed_fall = float(
      self.last_update_metrics.get(
        "normal_start_fall_rate",
        self.last_update_metrics["completed_episode_fall_rate"],
      )
    )
    observed_intervention = float(
      self.last_update_metrics["cbf_intervention_per_riser"]
    )
    before_fall = self.fall_multiplier
    before_intervention = self.intervention_multiplier
    self.fall_multiplier = projected_lagrange_update(
      self.fall_multiplier,
      observed_fall,
      self.fall_cost_budget,
      learning_rate=self.fall_multiplier_learning_rate,
      maximum=self.maximum_cost_multiplier,
    )
    self.intervention_multiplier = projected_lagrange_update(
      self.intervention_multiplier,
      observed_intervention,
      self.intervention_cost_budget,
      learning_rate=self.intervention_multiplier_learning_rate,
      maximum=self.maximum_cost_multiplier,
    )
    metrics = {
      "fall_multiplier_before": before_fall,
      "fall_multiplier_after": self.fall_multiplier,
      "fall_cost_budget": self.fall_cost_budget,
      "fall_cost_observed": observed_fall,
      "intervention_multiplier_before": before_intervention,
      "intervention_multiplier_after": self.intervention_multiplier,
      "intervention_cost_budget": self.intervention_cost_budget,
      "intervention_cost_observed": observed_intervention,
    }
    self.last_update_metrics.update(metrics)
    return metrics

  def prepare_constrained_advantages(self) -> dict[str, float]:
    """Build ``A_R - lambda_F A_F - lambda_I A_I`` for one PPO surrogate."""
    if self.brief_ppo_refinement:
      return self.prepare_brief_advantages()
    if not self.task_first_constrained:
      return self.shape_intervention_advantages()
    task = self.task_advantages

    def normalized(values: torch.Tensor) -> torch.Tensor:
      return (values - values.mean()) / (values.std() + 1.0e-8)

    fall = normalized(self.fall_cost_advantages)
    intervention = normalized(self.intervention_cost_advantages)
    constrained = (
      task
      - self.fall_multiplier * fall
      - self.intervention_multiplier * intervention
    )
    sample_weights = torch.where(
      self.hard_case_transitions,
      torch.full_like(constrained, self.hard_case_policy_weight),
      torch.ones_like(constrained),
    )
    constrained = constrained * sample_weights
    self.storage.advantages.copy_(constrained.unsqueeze(-1))
    metrics = {
      "task_advantage_mean": float(task.mean()),
      "fall_cost_advantage_mean": float(self.fall_cost_advantages.mean()),
      "intervention_cost_advantage_mean": float(
        self.intervention_cost_advantages.mean()
      ),
      "constrained_advantage_mean": float(constrained.mean()),
      "constrained_advantage_std": float(constrained.std()),
      "fall_multiplier": self.fall_multiplier,
      "intervention_multiplier": self.intervention_multiplier,
      "hard_case_policy_weight": self.hard_case_policy_weight,
      "hard_case_transition_fraction": float(
        self.hard_case_transitions.float().mean()
      ),
    }
    self.last_update_metrics.update(metrics)
    return metrics

  def redistribute_failure_focused_fall_penalty(self) -> dict[str, float]:
    """Move half of each v15 fall penalty into its preceding scalar rewards."""
    if not self.failure_focused_refinement:
      return {
        "fall_redistribution_enabled": 0.0,
        "fall_redistribution_event_count": 0.0,
        "fall_redistribution_total": 0.0,
      }
    credit = redistributed_fall_credit(
      self.fall_events,
      self.storage.dones.squeeze(-1),
      horizon=self.fall_redistribution_horizon,
      decay=self.fall_redistribution_decay,
      amount_per_fall=self.fall_redistribution_amount,
    )
    self.storage.rewards.sub_(credit.unsqueeze(-1))
    metrics = {
      "fall_redistribution_enabled": 1.0,
      "fall_redistribution_event_count": float(self.fall_events.sum()),
      "fall_redistribution_total": float(credit.sum()),
      "fall_redistribution_per_event": (
        float(credit.sum() / self.fall_events.sum().clamp_min(1))
      ),
      "fall_redistribution_horizon": float(self.fall_redistribution_horizon),
      "fall_redistribution_decay": self.fall_redistribution_decay,
    }
    self.last_update_metrics.update(metrics)
    return metrics

  def prepare_brief_advantages(self) -> dict[str, float]:
    """Weight hard-case samples in the same single-reward PPO surrogate."""
    before = self.storage.advantages.squeeze(-1).clone()
    sample_weights = torch.where(
      self.hard_case_transitions,
      torch.full_like(before, self.hard_case_policy_weight),
      torch.ones_like(before),
    )
    weighted = before * sample_weights
    self.storage.advantages.copy_(weighted.unsqueeze(-1))
    metrics = {
      "brief_single_reward_advantage": 1.0,
      "brief_advantage_mean_before_weighting": float(before.mean()),
      "brief_advantage_mean_after_weighting": float(weighted.mean()),
      "hard_case_policy_weight": self.hard_case_policy_weight,
      "hard_case_transition_fraction": float(
        self.hard_case_transitions.float().mean()
      ),
    }
    self.last_update_metrics.update(metrics)
    return metrics

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
      hard_case_transition = extras.get("online_hard_case_transition")
      timeouts = extras.get("time_outs")
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
      if hard_case_transition is not None:
        self.hard_case_transitions[step].copy_(hard_case_transition)
      if timeouts is not None:
        self.timeout_events[step].copy_(timeouts.bool())
      if self.task_first_constrained:
        if self._pending_fall_values is None or self._pending_intervention_values is None:
          raise RuntimeError("task-first cost values were not computed before env.step")
        self.fall_cost_values[step].copy_(self._pending_fall_values)
        self.intervention_cost_values[step].copy_(
          self._pending_intervention_values
        )
        if fell is not None:
          self.fall_costs[step].copy_(fell.float())
        if intervened is not None and magnitude is not None:
          normalized_cost = torch.clamp(
            magnitude / self.intervention_magnitude_scale, 0.0, 1.0
          ) * intervened.float()
          self.intervention_costs[step].copy_(normalized_cost)
        if timeouts is not None:
          timeout_float = timeouts.float()
          self.fall_costs[step].add_(
            self.gamma * self.fall_cost_values[step] * timeout_float
          )
          self.intervention_costs[step].add_(
            self.gamma * self.intervention_cost_values[step] * timeout_float
          )
    super().process_env_step(obs, rewards, dones, extras)
    if self.task_first_constrained:
      assert self.fall_critic is not None
      assert self.intervention_critic is not None
      assert self.risk_head is not None
      self.fall_critic.update_normalization(obs)
      self.intervention_critic.update_normalization(obs)
      self.risk_head.update_normalization(obs)
      self._pending_fall_values = None
      self._pending_intervention_values = None

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
    if not self.task_first_constrained:
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
      "task_reward_excludes_fixed_safety_shaping": float(
        self.task_first_constrained
      ),
      "intervention_cost_mean": float(self.intervention_costs.mean()),
    }
    riser_delta = (self.stair_indices[1:] - self.stair_indices[:-1]).clamp_min(0)
    valid_transition = ~self.storage.dones[:-1].squeeze(-1).bool()
    riser_crossings = (riser_delta * valid_transition.long()).sum()
    metrics["riser_crossing_count"] = float(riser_crossings)
    metrics["cbf_intervention_per_riser"] = float(
      self.actual_cbf_intervened.sum() / riser_crossings.clamp_min(1)
    )
    metrics["intervention_cost_per_riser"] = float(
      self.intervention_costs.sum() / riser_crossings.clamp_min(1)
    )
    completed = self.storage.dones.squeeze(-1).bool() & ~self.timeout_events
    completed_count = completed.sum().clamp_min(1)
    metrics["completed_episode_count"] = float(completed.sum())
    metrics["completed_episode_fall_rate"] = float(
      (completed & self.fall_events).sum() / completed_count
    )
    metrics.update(action_metrics)
    self.last_update_metrics.update(metrics)
    return metrics

  def compute_returns(self, obs) -> None:
    """Compute task returns plus independent fall/intervention cost returns."""
    super().compute_returns(obs)
    if not self.task_first_constrained:
      return
    assert self.fall_critic is not None
    assert self.intervention_critic is not None
    with torch.no_grad():
      last_fall_values = self.fall_critic(obs).detach().squeeze(-1)
      last_intervention_values = (
        self.intervention_critic(obs).detach().squeeze(-1)
      )
    dones = self.storage.dones.squeeze(-1).bool()
    fall_advantages, fall_returns = generalized_cost_advantage(
      self.fall_costs,
      self.fall_cost_values,
      last_fall_values,
      dones,
      gamma=self.gamma,
      lam=self.lam,
    )
    intervention_advantages, intervention_returns = generalized_cost_advantage(
      self.intervention_costs,
      self.intervention_cost_values,
      last_intervention_values,
      dones,
      gamma=self.gamma,
      lam=self.lam,
    )
    self.fall_cost_advantages.copy_(fall_advantages)
    self.intervention_cost_advantages.copy_(intervention_advantages)
    self.fall_cost_returns.copy_(fall_returns)
    self.intervention_cost_returns.copy_(intervention_returns)
    self.task_advantages.copy_(self.storage.advantages.squeeze(-1))
    strong_intervention = self.cbf_intervened & (
      self.cbf_magnitude
      >= self.strong_intervention_fraction * self.intervention_magnitude_scale
    )
    self.risk_labels.copy_(
      future_event_labels(
        self.fall_events | strong_intervention,
        dones,
        horizon=self.risk_horizon,
      )
    )
    self.successful_correction.copy_(
      success_gated_correction_mask(
        self.actual_cbf_intervened,
        self.stair_indices,
        self.task_advantages,
        dones,
        self.fall_events,
        horizon=self.correction_success_horizon,
      )
    )

  def _train_task_first_heads(self, observations) -> dict[str, Any]:
    if not self.task_first_constrained:
      return {}
    assert self.fall_critic is not None
    assert self.intervention_critic is not None
    assert self.risk_head is not None
    flat_observations = observations
    fall_targets = self.fall_cost_returns.flatten().detach()
    intervention_targets = self.intervention_cost_returns.flatten().detach()
    risk_targets = self.risk_labels.flatten().float().detach()
    batch_size = flat_observations.shape[0]
    mini_batch_size = batch_size // self.num_mini_batches
    if mini_batch_size < 1:
      raise RuntimeError("task-first rollout is smaller than the minibatch count")
    usable = mini_batch_size * self.num_mini_batches
    indices = torch.randperm(batch_size, device=self.device)[:usable]
    positive_count = risk_targets.sum()
    negative_count = risk_targets.numel() - positive_count
    positive_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 20.0)

    fall_loss_total = 0.0
    intervention_loss_total = 0.0
    risk_loss_total = 0.0
    maximum_gradient_norm = 0.0
    for index in range(self.num_mini_batches):
      batch_indices = indices[
        index * mini_batch_size : (index + 1) * mini_batch_size
      ]
      batch_observations = flat_observations[batch_indices]
      fall_predictions = self.fall_critic(batch_observations).squeeze(-1)
      intervention_predictions = self.intervention_critic(
        batch_observations
      ).squeeze(-1)
      risk_logits = self.risk_head(batch_observations).squeeze(-1)
      fall_loss = torch.mean(
        (fall_predictions - fall_targets[batch_indices]).square()
      )
      intervention_loss = torch.mean(
        (intervention_predictions - intervention_targets[batch_indices]).square()
      )
      risk_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        risk_logits,
        risk_targets[batch_indices],
        pos_weight=positive_weight,
      )
      loss = (
        self.value_loss_coef * (fall_loss + intervention_loss)
        + self.risk_loss_coef * risk_loss
      )
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      gradient_norm = torch.nn.utils.clip_grad_norm_(
        chain(
          self.fall_critic.parameters(),
          self.intervention_critic.parameters(),
          self.risk_head.parameters(),
        ),
        self.max_grad_norm,
      )
      self.optimizer.step()
      fall_loss_total += float(fall_loss)
      intervention_loss_total += float(intervention_loss)
      risk_loss_total += float(risk_loss)
      maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))

    with torch.inference_mode():
      fall_predictions = self.fall_critic(flat_observations).squeeze(-1)
      intervention_predictions = self.intervention_critic(
        flat_observations
      ).squeeze(-1)
      risk_logits = self.risk_head(flat_observations).squeeze(-1)
    rollout_shape = self.fall_events.shape
    dones = self.storage.dones.squeeze(-1).bool()
    fall_value_event_count, fall_value_delta = pre_event_value_delta(
      fall_predictions.reshape(rollout_shape),
      self.fall_events,
      dones,
      horizon=self.pre_intervention_horizon,
    )
    intervention_events = self.actual_cbf_intervened.clone()
    intervention_events[1:] &= ~self.actual_cbf_intervened[:-1]
    intervention_value_event_count, intervention_value_delta = (
      pre_event_value_delta(
        intervention_predictions.reshape(rollout_shape),
        intervention_events,
        dones,
        horizon=self.pre_intervention_horizon,
      )
    )
    updates = float(self.num_mini_batches)
    return {
      "fall_value_loss": fall_loss_total / updates,
      "intervention_value_loss": intervention_loss_total / updates,
      "risk_prediction_loss": risk_loss_total / updates,
      "cost_critic_gradient_norm_pre_clip_max": maximum_gradient_norm,
      "fall_cost_explained_variance_after_update": float(
        1.0
        - torch.var(fall_targets - fall_predictions, unbiased=False)
        / torch.var(fall_targets, unbiased=False).clamp_min(1.0e-8)
      ),
      "intervention_cost_explained_variance_after_update": float(
        1.0
        - torch.var(
          intervention_targets - intervention_predictions, unbiased=False
        )
        / torch.var(intervention_targets, unbiased=False).clamp_min(1.0e-8)
      ),
      "risk_prediction_after_update": binary_risk_metrics(
        risk_logits, self.risk_labels.flatten()
      ),
      "pre_fall_cost_value_event_count_after_update": fall_value_event_count,
      "pre_fall_cost_value_delta_after_update": fall_value_delta,
      "pre_intervention_cost_value_event_count_after_update": (
        intervention_value_event_count
      ),
      "pre_intervention_cost_value_delta_after_update": (
        intervention_value_delta
      ),
    }

  def update(self) -> dict[str, Any]:
    """Run one exact single-clipped PPO update with optional legacy anchors."""
    if self.rnd or self.symmetry:
      raise RuntimeError("online safe PPO does not support RND or symmetry losses")
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise RuntimeError("online safe PPO currently requires feed-forward models")
    if self.schedule != "fixed":
      raise RuntimeError("online safe PPO requires a fixed learning-rate schedule")
    if self.base_anchor_weight > 0.0 and self.base_actor_reference is None:
      raise RuntimeError("base actor reference must be frozen before online PPO")
    retention_weights = {
      "d0": self.d0_retention_anchor_weight,
      "neighbor": self.neighbor_retention_anchor_weight,
    }
    for name, weight in retention_weights.items():
      if weight > 0.0 and name not in self.retention_anchor_banks:
        raise RuntimeError(f"{name} retention anchor bank must be installed")
    if any(weight > 0.0 for weight in retention_weights.values()) and (
      self.retention_actor_reference is None
    ):
      raise RuntimeError("retention actor reference must be frozen before PPO")

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
      retention_kl_before = self.retention_anchor_kl_metrics()
      if self.task_first_constrained:
        assert self.risk_head is not None
        risk_before = binary_risk_metrics(
          self.risk_head(observations).squeeze(-1),
          self.risk_labels.flatten(),
        )
      else:
        risk_before = {}

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
    mean_d0_retention_anchor_kl = 0.0
    mean_neighbor_retention_anchor_kl = 0.0
    maximum_actor_gradient_norm = 0.0
    maximum_critic_gradient_norm = 0.0
    completed_updates = 0
    attempted_batches = 0
    kl_early_stopped = False
    maximum_preupdate_kl = 0.0
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
      attempted_batches += 1
      if self.kl_early_stopping and not self._critic_only:
        with torch.inference_mode():
          preupdate_kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, current_params
          ).mean()
        if not bool(torch.isfinite(preupdate_kl)):
          raise RuntimeError("pre-update PPO KL is non-finite")
        maximum_preupdate_kl = max(maximum_preupdate_kl, float(preupdate_kl))
        if completed_updates > 0 and preupdate_kl > float(self.desired_kl):
          kl_early_stopped = True
          break

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

      d0_retention_anchor_kl = torch.zeros((), device=self.device)
      neighbor_retention_anchor_kl = torch.zeros((), device=self.device)
      if not self._critic_only:
        if self.d0_retention_anchor_weight > 0.0:
          d0_retention_anchor_kl = self._retention_anchor_loss("d0")
        if self.neighbor_retention_anchor_weight > 0.0:
          neighbor_retention_anchor_kl = self._retention_anchor_loss(
            "neighbor"
          )

      if self._critic_only:
        loss = self.value_loss_coef * value_loss
      else:
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy.mean()
          + self.base_anchor_weight * anchor_kl
          + self.d0_retention_anchor_weight * d0_retention_anchor_kl
          + self.neighbor_retention_anchor_weight
          * neighbor_retention_anchor_kl
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
      completed_updates += 1

      mean_value_loss += float(value_loss)
      mean_surrogate_loss += float(surrogate_loss)
      mean_entropy += float(entropy.mean())
      mean_anchor_kl += float(anchor_kl)
      mean_d0_retention_anchor_kl += float(d0_retention_anchor_kl)
      mean_neighbor_retention_anchor_kl += float(
        neighbor_retention_anchor_kl
      )
      maximum_actor_gradient_norm = max(
        maximum_actor_gradient_norm, float(actor_gradient_norm)
      )
      maximum_critic_gradient_norm = max(
        maximum_critic_gradient_norm, float(critic_gradient_norm)
      )

    if completed_updates < 1:
      raise RuntimeError("PPO update completed no minibatches")
    num_updates = completed_updates
    losses: dict[str, Any] = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "base_anchor_kl_loss": mean_anchor_kl / num_updates,
      "d0_retention_anchor_kl_loss": (
        mean_d0_retention_anchor_kl / num_updates
      ),
      "neighbor_retention_anchor_kl_loss": (
        mean_neighbor_retention_anchor_kl / num_updates
      ),
      "ppo_minibatches_attempted": attempted_batches,
      "ppo_minibatches_completed": completed_updates,
      "target_kl_early_stopped": kl_early_stopped,
      "maximum_preupdate_minibatch_kl": maximum_preupdate_kl,
    }
    task_first_heads = self._train_task_first_heads(observations)
    safe_bc = self.apply_safe_bc_auxiliary(
      observations=observations,
      learning_rate=(
        0.0
        if self._critic_only
        else self.actor_learning_rate
        * (
          self.correction_distillation_weight
          if self.task_first_constrained
          else self.safe_bc_weight
        )
      ),
    )
    self.storage.clear()
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
      retention_kl_after = self.retention_anchor_kl_metrics()
    diagnostics: dict[str, Any] = {
      "mean_kl": float(kl),
      "clip_fraction": float(clip_fraction),
      "action_saturation_fraction": float(action_saturation),
      "actor_learning_rate": float(self.actor_learning_rate),
      "action_std_mean": float(self.actor.output_std.mean()),
      "base_anchor_weight": float(self.base_anchor_weight),
      "base_anchor_kl_before_update": float(anchor_kl_before),
      "base_anchor_kl_after_update": float(anchor_kl_after),
      "d0_retention_anchor_weight": float(
        self.d0_retention_anchor_weight
      ),
      "neighbor_retention_anchor_weight": float(
        self.neighbor_retention_anchor_weight
      ),
      "d0_retention_anchor_kl_budget": float(
        self.d0_retention_anchor_kl_budget
      ),
      "neighbor_retention_anchor_kl_budget": float(
        self.neighbor_retention_anchor_kl_budget
      ),
      "retention_anchor_adaptation_rate": float(
        self.retention_anchor_adaptation_rate
      ),
      "maximum_retention_anchor_weight": float(
        self.maximum_retention_anchor_weight
      ),
      "retention_anchor_batch_size": self.retention_anchor_batch_size,
      "actor_gradient_norm_pre_clip_max": maximum_actor_gradient_norm,
      "critic_gradient_norm_pre_clip_max": maximum_critic_gradient_norm,
      "safe_bc_loss": safe_bc["loss"],
      "safe_bc_weight": float(self.safe_bc_weight),
      "safe_bc_effective_learning_rate": safe_bc["learning_rate"],
      "safe_bc_gradient_norm": safe_bc["gradient_norm"],
      "correction_distillation_success_count": safe_bc["eligible_count"],
      "correction_distillation_success_fraction": safe_bc[
        "eligible_fraction"
      ],
      "correction_distillation_weight": float(
        self.correction_distillation_weight
      ),
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
      "risk_prediction_before_update": risk_before,
    }
    for name, value in retention_kl_before.items():
      diagnostics[f"{name}_before_update"] = value
    for name, value in retention_kl_after.items():
      diagnostics[f"{name}_after_update"] = value
    diagnostics.update(task_first_heads)
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
    result = {
      "loss": 0.0,
      "learning_rate": float(learning_rate),
      "gradient_norm": 0.0,
      "eligible_count": 0,
      "eligible_fraction": 0.0,
    }
    # Safe-BC is only defined on actions that were actually projected and
    # executed.  Counterfactual events from filter-off simulation remain valid
    # reward credit but must not be mislabeled as real interventions here.
    intervention_mask = (
      self.successful_correction.flatten()
      if self.task_first_constrained
      else self.actual_cbf_intervened.flatten()
    )
    result["eligible_count"] = int(intervention_mask.sum())
    result["eligible_fraction"] = float(intervention_mask.float().mean())
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
    self.fall_costs.zero_()
    self.intervention_costs.zero_()
    self.fall_cost_values.zero_()
    self.intervention_cost_values.zero_()
    self.fall_cost_returns.zero_()
    self.intervention_cost_returns.zero_()
    self.fall_cost_advantages.zero_()
    self.intervention_cost_advantages.zero_()
    self.task_advantages.zero_()
    self.risk_labels.zero_()
    self.successful_correction.zero_()
    self.hard_case_transitions.zero_()
    self.timeout_events.zero_()
    self._pending_fall_values = None
    self._pending_intervention_values = None


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
    self.alg.initialize_task_first_heads_from_critic()
    self.alg.initialize_online_std()
    if not self.alg.brief_ppo_refinement:
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
    loaded_task_first_heads = self.alg.load_constraint_state_dict(loaded)
    # Standalone evaluators/smokes must never reach an anchored update without
    # a frozen reference.  The full refinement entrypoint replaces this
    # temporary accepted-policy reference with the original pretrained actor.
    if not self.alg.brief_ppo_refinement:
      self.alg.set_base_actor_reference()
    # A new optimizer is mandatory after an input expansion and is also the
    # intended online protocol after every accepted/backtracked checkpoint.
    self.alg.reset_online_optimizer()
    self.alg._std_initialized = True
    self.current_learning_iteration = int(loaded.get("iter", 0))
    return expansion | {
      "source_iteration": int(loaded.get("iter", -1)),
      "optimizer_reset": True,
      "loaded_task_first_heads": loaded_task_first_heads,
    }

  def snapshot_candidate_state(self) -> dict[str, Any]:
    state = {
      "actor": {k: v.detach().clone() for k, v in self.alg.actor.state_dict().items()},
      "critic": {k: v.detach().clone() for k, v in self.alg.critic.state_dict().items()},
      "optimizer": copy.deepcopy(self.alg.optimizer.state_dict()),
      "constraint": copy.deepcopy(self.alg.constraint_state_dict()),
    }
    return state

  def restore_candidate_state(self, state: dict[str, Any]) -> None:
    # Normalization buffers may have most recently been updated during a
    # no-gradient rollout.  Restore all transactional state atomically.
    with torch.no_grad():
      self.alg.actor.load_state_dict(state["actor"], strict=True)
      self.alg.critic.load_state_dict(state["critic"], strict=True)
      self.alg.load_constraint_state_dict(state["constraint"])
      self.alg.optimizer.load_state_dict(state["optimizer"])

  def reduce_after_rejection(self) -> None:
    """Discard rejected-direction momentum without permanently shrinking LR."""
    self.alg.reset_online_optimizer()

  def parameters_are_finite(self) -> bool:
    modules = [self.alg.actor, self.alg.critic]
    modules.extend(
      module
      for module in (
        self.alg.fall_critic,
        self.alg.intervention_critic,
        self.alg.risk_head,
      )
      if module is not None
    )
    return all(
      bool(torch.isfinite(parameter).all())
      for module in modules
      for parameter in module.parameters()
    )
