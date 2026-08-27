"""Deterministic-policy CBF teacher for the paper-aligned v35 study.

The rollout still executes the CBF-filtered stochastic PPO action.  In
parallel, the action term projects the frozen round-reference policy mean at
the identical pre-step state.  The auxiliary target therefore teaches only a
correction that the deployable deterministic actor itself requires, rather
than conditioning its mean on exploration noise that happened to trigger the
runtime filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .teacher_v26 import HigherRiserCbfAction, v26_online_safety_telemetry
from .teacher_v30 import CbfTeacherV30PPO, CbfTeacherV30PpoAlgorithmCfg
from .teacher_v30_math import (
  intervention_teacher_weights,
  weighted_action_errors,
)

METHOD_ID = "deterministic-mean-counterfactual-cbf-teacher-v35"


def v35_mean_teacher_telemetry(
  env,
  action_name: str = "joint_pos",
  termination_name: str = "fell_over",
) -> torch.Tensor:
  """Expose the shadow policy-mean projection before auto-reset."""
  zeros = v26_online_safety_telemetry(
    env, action_name=action_name, termination_name=termination_name
  )
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, HigherRiserCbfAction):
    raise TypeError("v35 mean teacher requires HigherRiserCbfAction")
  if not bool(term.counterfactual_policy_projection_valid.all()):
    missing = int((~term.counterfactual_policy_projection_valid).sum())
    raise RuntimeError(
      f"v35 deterministic policy projection missing on {missing} environments"
    )
  env.extras["v35_policy_mean_action"] = (
    term.counterfactual_policy_action.detach().clone()
  )
  env.extras["v35_policy_mean_safe_action"] = (
    term.counterfactual_safe_policy_action.detach().clone()
  )
  env.extras["v35_policy_mean_intervened"] = (
    term.counterfactual_policy_intervened.detach().clone()
  )
  env.extras["v35_policy_mean_correction_norm"] = (
    term.counterfactual_policy_correction_norm.detach().clone()
  )
  env.extras["v35_policy_mean_nominal_margin"] = (
    term.counterfactual_policy_nominal_margin.detach().clone()
  )
  return zeros


def configure_v35_mean_teacher_telemetry(
  env_cfg, *, runtime_filter_during_training: bool
) -> dict[str, Any]:
  """Replace only the training telemetry function; action execution is fixed."""
  action = env_cfg.actions.get("joint_pos")
  if action is None:
    raise RuntimeError("v35 mean teacher requires the joint_pos action")
  telemetry = env_cfg.rewards.get("online_safety_telemetry")
  if telemetry is None:
    raise RuntimeError("v35 mean teacher requires online safety telemetry")
  telemetry.func = v35_mean_teacher_telemetry
  telemetry.params = {
    "action_name": "joint_pos",
    "termination_name": "fell_over",
  }
  return {
    "enabled": True,
    "target_source": "same_state_cbf_projection_of_round_reference_mean",
    "rollout_action_source": "stochastic_policy_sample",
    "rollout_action_execution": (
      "runtime_cbf_filtered"
      if runtime_filter_during_training
      else "nominal_unshielded"
    ),
    "loss": "weighted_smooth_l1_per_action_mean",
    "eligibility": "deterministic_mean_cbf_interventions_only",
  }


@dataclass
class PaperMeanTeacherV35PpoAlgorithmCfg(CbfTeacherV30PpoAlgorithmCfg):
  """Config selecting the deterministic-mean v35 PPO subclass."""

  class_name: str = (
    "src.tasks.stairs_cbf.paper_teacher_v35:PaperMeanTeacherV35PPO"
  )


class PaperMeanTeacherV35PPO(CbfTeacherV30PPO):
  """Use a same-state filtered deterministic mean as the A2 target."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.teacher_mode != "residual" or self.teacher_gate != "all_interventions":
      raise ValueError(
        "v35 deterministic-mean teacher requires residual/all_interventions"
      )
    t = self.storage.num_transitions_per_env
    n = self.storage.num_envs
    action_dim = self.storage.actions.shape[-1]
    self.v35_policy_means = torch.zeros(t, n, action_dim, device=self.device)
    self.v35_safe_policy_means = torch.zeros_like(self.v35_policy_means)
    self.v35_mean_intervened = torch.zeros(
      t, n, dtype=torch.bool, device=self.device
    )
    self.v35_mean_correction_norm = torch.zeros(t, n, device=self.device)
    self.v35_mean_nominal_margin = torch.zeros(t, n, device=self.device)
    self.v35_mean_telemetry_present = torch.zeros(
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
      mean = extras.get("v35_policy_mean_action")
      safe = extras.get("v35_policy_mean_safe_action")
      intervened = extras.get("v35_policy_mean_intervened")
      correction_norm = extras.get("v35_policy_mean_correction_norm")
      margin = extras.get("v35_policy_mean_nominal_margin")
      values = (mean, safe, intervened, correction_norm, margin)
      if any(value is None for value in values):
        raise RuntimeError("v35 deterministic-mean telemetry is incomplete")
      assert mean is not None and safe is not None
      assert intervened is not None and correction_norm is not None
      assert margin is not None
      self.v35_policy_means[step].copy_(mean)
      self.v35_safe_policy_means[step].copy_(safe)
      self.v35_mean_intervened[step].copy_(intervened.bool())
      self.v35_mean_correction_norm[step].copy_(correction_norm)
      self.v35_mean_nominal_margin[step].copy_(margin)
      self.v35_mean_telemetry_present[step] = True
    super().process_env_step(obs, rewards, dones, extras)

  def _compute_teacher_labels(
    self, correction_norm: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    del correction_norm
    eligible, weights = intervention_teacher_weights(
      self.v35_mean_intervened,
      self.v35_mean_correction_norm,
      correction_scale=self.teacher_correction_scale,
    )
    zeros = torch.zeros_like(eligible)
    ones = torch.ones_like(eligible)
    return eligible, weights, {
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
    if not bool(self.v35_mean_telemetry_present.all()):
      missing = int((~self.v35_mean_telemetry_present).sum())
      raise RuntimeError(f"v35 mean telemetry missing on {missing} transitions")
    mean_error = float(
      torch.amax(torch.abs(self.v35_policy_means - self.v30_reference_means))
    )
    if mean_error > 1.0e-6:
      raise RuntimeError(
        "v35 staged mean differs from the frozen round policy mean: "
        f"{mean_error}"
      )

    metrics = super().relabel_teacher_transitions()
    correction = (
      self.v35_safe_policy_means - self.v35_policy_means
    ).detach()
    target = (
      self.v30_reference_means + float(self.teacher_eta) * correction
    ).detach()
    self.v30_teacher_targets.copy_(target)
    self.v30_correction_vectors.copy_(correction)
    self.teacher_correction_norm.copy_(self.v35_mean_correction_norm)

    flat_eligible = self.teacher_eligible.flatten()
    flat_weights = self.teacher_weights.flatten()
    effective = flat_weights * flat_eligible.to(flat_weights.dtype)
    before_distance, before_per_action = weighted_action_errors(
      self.v30_reference_means.flatten(0, 1),
      target.flatten(0, 1),
      flat_eligible,
      flat_weights,
    )
    intervened_count = int(self.v35_mean_intervened.sum())
    weighted_count = float(effective.sum())
    mean_weighted_correction = (
      float(
        (effective * self.v35_mean_correction_norm.flatten()).sum()
        / effective.sum().clamp_min(1.0e-8)
      )
      if weighted_count > 0.0
      else 0.0
    )
    metrics.update(
      {
        "v35_teacher_method_id": METHOD_ID,
        "v35_policy_mean_storage_max_abs_error": mean_error,
        "v35_policy_mean_intervention_count": float(intervened_count),
        "v35_policy_mean_intervention_fraction": float(
          self.v35_mean_intervened.float().mean()
        ),
        "v35_policy_mean_correction_mean": float(
          self.v35_mean_correction_norm.mean()
        ),
        "v35_policy_mean_nominal_margin_min": float(
          self.v35_mean_nominal_margin.min()
        ),
        "teacher_eligible_fraction_among_interventions": (
          int(self.teacher_eligible.sum()) / max(1, intervened_count)
        ),
        "teacher_actor_coordinate_correction_mean": float(
          self.v35_mean_correction_norm.mean()
        ),
        "mean_policy_to_teacher_action_distance": float(
          self.v35_mean_correction_norm[self.v35_mean_intervened].mean()
        )
        if intervened_count
        else 0.0,
        "mean_weighted_policy_to_teacher_action_distance": (
          mean_weighted_correction
        ),
        "mean_residual_target_norm": float(before_distance),
        "mean_policy_to_target_distance_before_update": float(before_distance),
        "per_action_teacher_error_before_update": [
          float(value) for value in before_per_action
        ],
        "teacher_tensor_shapes": {
          "raw_sampled_action": list(self.policy_actions.shape),
          "round_reference_mean": list(self.v30_reference_means.shape),
          "safe_deterministic_mean": list(self.v35_safe_policy_means.shape),
          "correction_vector": list(self.v30_correction_vectors.shape),
          "intervention": list(self.v35_mean_intervened.shape),
        },
      }
    )
    self.last_update_metrics.update(metrics)
    return metrics

  def clear_cbf_rollout(self) -> None:
    super().clear_cbf_rollout()
    if not hasattr(self, "v35_policy_means"):
      return
    self.v35_policy_means.zero_()
    self.v35_safe_policy_means.zero_()
    self.v35_mean_intervened.zero_()
    self.v35_mean_correction_norm.zero_()
    self.v35_mean_nominal_margin.zero_()
    self.v35_mean_telemetry_present.zero_()

  def save(self) -> dict[str, Any]:
    output = super().save()
    output["proximal_method_id"] = METHOD_ID
    output["v35_teacher_target_source"] = (
      "same_state_cbf_projection_of_round_reference_mean"
    )
    return output
