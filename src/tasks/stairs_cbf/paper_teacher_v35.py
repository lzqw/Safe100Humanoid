"""Deterministic-policy CBF teacher for the paper-aligned v35 study.

The rollout can execute either the filtered or nominal stochastic PPO action.
In parallel, the action term projects the frozen round-reference policy mean at
the identical pre-step state.  The auxiliary target therefore teaches only a
correction that the deployable deterministic actor itself requires, rather
than conditioning its mean on exploration noise that happened to trigger the
runtime filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .paper_dual_v35 import (
  capped_norm_balance_auxiliary_gradients,
  split_filter_actor_objective_masks,
  task_priority_project_auxiliary_gradients,
)
from .teacher_v26 import HigherRiserCbfAction, v26_online_safety_telemetry
from .teacher_v30 import CbfTeacherV30PPO, CbfTeacherV30PpoAlgorithmCfg
from .teacher_v30_math import (
  disjoint_terminal_outcomes,
  intervention_teacher_weights,
  outcome_gated_interventions,
  terminal_episode_transition_mask,
  weighted_action_errors,
)

METHOD_ID = "deterministic-mean-counterfactual-cbf-teacher-v35"
TEACHER_GRADIENT_MAXIMUM_SCALE = 4.0


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
  env_cfg,
  *,
  runtime_filter_during_training: bool,
  failure_only: bool = False,
  success_only: bool = False,
  failure_focused_actor: bool = False,
  distill_only_actor: bool = False,
  success_local_kl_beta: float = 0.0,
  split_filter_actor_objectives: bool = False,
  task_priority_gradient_surgery: bool = False,
  teacher_gradient_target_ratio: float = 0.0,
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
    "outcome_gate": (
      "failed_episodes"
      if failure_only
      else "successful_episodes"
      if success_only
      else "none"
    ),
    "actor_ppo_scope": (
      "none_distillation_only"
      if distill_only_actor
      else "nominal_filter_off_transitions"
      if split_filter_actor_objectives
      else "failed_episodes"
      if failure_focused_actor
      else "all_transitions"
    ),
    "successful_episode_actor_objective": (
      (
        "mean_CBF_distillation_plus_global_and_success_local_round_reference_KL"
        if success_local_kl_beta > 0.0
        else "mean_CBF_distillation_plus_global_round_reference_KL"
      )
      if distill_only_actor
      else "round_reference_KL_only"
      if failure_focused_actor
      else (
        "PPO_plus_mean_CBF_plus_extra_local_round_reference_KL"
        if success_local_kl_beta > 0.0
        else "PPO_plus_round_reference_KL"
      )
    ),
    "success_local_kl_beta": float(success_local_kl_beta),
    "split_filter_actor_objectives": bool(split_filter_actor_objectives),
    "task_priority_gradient_surgery": bool(
      task_priority_gradient_surgery
    ),
    "teacher_gradient_target_ratio": float(teacher_gradient_target_ratio),
    "teacher_gradient_maximum_scale": TEACHER_GRADIENT_MAXIMUM_SCALE,
    "filtered_transition_actor_objective": (
      "deterministic_mean_CBF_teacher_plus_global_round_reference_KL"
      if split_filter_actor_objectives
      else "shared_PPO_teacher_objective"
    ),
  }


@dataclass
class PaperMeanTeacherV35PpoAlgorithmCfg(CbfTeacherV30PpoAlgorithmCfg):
  """Config selecting the deterministic-mean v35 PPO subclass."""

  class_name: str = (
    "src.tasks.stairs_cbf.paper_teacher_v35:PaperMeanTeacherV35PPO"
  )
  v35_task_priority_gradient_surgery: bool = False
  v35_teacher_gradient_target_ratio: float = 0.0


class PaperMeanTeacherV35PPO(CbfTeacherV30PPO):
  """Use a same-state filtered deterministic mean as the A2 target."""

  def __init__(
    self,
    *args,
    v35_failure_only_mean_teacher: bool = False,
    v35_success_only_mean_teacher: bool = False,
    v35_failure_focused_actor: bool = False,
    v35_distill_only_actor: bool = False,
    v35_success_local_kl_beta: float = 0.0,
    v35_split_filter_actor_objectives: bool = False,
    v35_task_priority_gradient_surgery: bool = False,
    v35_teacher_gradient_target_ratio: float = 0.0,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    if self.teacher_mode != "residual" or self.teacher_gate != "all_interventions":
      raise ValueError(
        "v35 deterministic-mean teacher requires residual/all_interventions"
      )
    self.v35_failure_only_mean_teacher = bool(
      v35_failure_only_mean_teacher
    )
    self.v35_success_only_mean_teacher = bool(
      v35_success_only_mean_teacher
    )
    self.v35_failure_focused_actor = bool(v35_failure_focused_actor)
    self.v35_distill_only_actor = bool(v35_distill_only_actor)
    self.v35_split_filter_actor_objectives = bool(
      v35_split_filter_actor_objectives
    )
    self.v35_task_priority_gradient_surgery = bool(
      v35_task_priority_gradient_surgery
    )
    self.v35_teacher_gradient_target_ratio = float(
      v35_teacher_gradient_target_ratio
    )
    self.v35_success_local_kl_beta = float(v35_success_local_kl_beta)
    if (
      not math.isfinite(self.v35_success_local_kl_beta)
      or not 0.0 <= self.v35_success_local_kl_beta <= 4.0
    ):
      raise ValueError("v35 success-local KL beta must lie in [0, 4]")
    if self.v35_failure_focused_actor and not self.v35_failure_only_mean_teacher:
      raise ValueError(
        "v35 failure-focused actor requires failure-only mean teacher"
      )
    if self.v35_failure_only_mean_teacher and self.v35_success_only_mean_teacher:
      raise ValueError("v35 mean teacher outcome gates are mutually exclusive")
    if self.v35_split_filter_actor_objectives and (
      self.v35_distill_only_actor
      or self.v35_failure_focused_actor
      or self.v35_failure_only_mean_teacher
      or self.v35_success_only_mean_teacher
    ):
      raise ValueError(
        "v68 split filter actor objectives require ungated mixed A2 training"
      )
    if (
      self.v35_task_priority_gradient_surgery
      and not self.v35_split_filter_actor_objectives
    ):
      raise ValueError(
        "v69 task-priority gradient surgery requires v68 split objectives"
      )
    if (
      not math.isfinite(self.v35_teacher_gradient_target_ratio)
      or not 0.0 <= self.v35_teacher_gradient_target_ratio <= 1.0
    ):
      raise ValueError("v70 teacher gradient target ratio must lie in [0, 1]")
    if (
      self.v35_teacher_gradient_target_ratio > 0.0
      and not self.v35_task_priority_gradient_surgery
    ):
      raise ValueError(
        "v70 teacher gradient norm balancing requires gradient surgery"
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
    self.v35_failed_episode_transition = torch.zeros(
      t, n, dtype=torch.bool, device=self.device
    )
    self.v35_success_terminals = torch.zeros_like(
      self.v35_failed_episode_transition
    )
    self.v35_success_episode_transition = torch.zeros_like(
      self.v35_failed_episode_transition
    )
    self.v35_filter_execution_environment_mask = torch.zeros(
      n, dtype=torch.bool, device=self.device
    )
    self.v35_split_filter_mask_present = False

  def set_v35_filter_execution_environment_mask(
    self, filter_mask: torch.Tensor
  ) -> None:
    """Set the fixed per-world objective routing for the next rollout."""
    if not self.v35_split_filter_actor_objectives:
      raise RuntimeError("v68 split actor objective routing is disabled")
    if (
      filter_mask.shape != self.v35_filter_execution_environment_mask.shape
      or filter_mask.dtype != torch.bool
    ):
      raise ValueError("v68 filter execution mask must be boolean with shape [N]")
    # The pure helper owns the non-empty/disjoint routing contract.
    split_filter_actor_objective_masks(filter_mask, 1)
    self.v35_filter_execution_environment_mask.copy_(
      filter_mask.to(self.device)
    )
    self.v35_split_filter_mask_present = True

  def _split_filter_actor_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
    if not self.v35_split_filter_mask_present:
      raise RuntimeError("v68 split filter actor mask was not staged")
    return split_filter_actor_objective_masks(
      self.v35_filter_execution_environment_mask,
      self.storage.num_transitions_per_env,
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
      reached_top = extras.get("v35_reached_top")
      if reached_top is None:
        raise RuntimeError("v35 exact reached-top telemetry is missing")
      self.v35_success_terminals[step].copy_(
        dones.bool() & reached_top.bool()
      )
    super().process_env_step(obs, rewards, dones, extras)

  def _compute_teacher_labels(
    self, correction_norm: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    del correction_norm
    failed_terminal, success_terminal, _ = disjoint_terminal_outcomes(
      self.storage.dones.squeeze(-1).bool(),
      self.fall_events.bool(),
      self.v35_success_terminals,
    )
    failed_transition = terminal_episode_transition_mask(
      self.teacher_episode_ids, failed_terminal
    )
    success_transition = terminal_episode_transition_mask(
      self.teacher_episode_ids, success_terminal
    )
    self.v35_failed_episode_transition.copy_(failed_transition)
    self.v35_success_episode_transition.copy_(success_transition)
    gate = (
      "failed"
      if self.v35_failure_only_mean_teacher
      else "successful"
      if self.v35_success_only_mean_teacher
      else "none"
    )
    gated_intervention = outcome_gated_interventions(
      self.v35_mean_intervened,
      failed_transition,
      success_transition,
      gate=gate,
    )
    if self.v35_split_filter_actor_objectives:
      _, teacher_environment = self._split_filter_actor_masks()
      gated_intervention &= teacher_environment
    eligible, weights = intervention_teacher_weights(
      gated_intervention,
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

  def _actor_ppo_transition_mask(self) -> torch.Tensor:
    if self.v35_distill_only_actor:
      return torch.zeros_like(self.teacher_eligible, dtype=torch.bool)
    if self.v35_failure_focused_actor:
      return self.v35_failed_episode_transition
    if self.v35_split_filter_actor_objectives:
      ppo_environment, _ = self._split_filter_actor_masks()
      return ppo_environment
    return super()._actor_ppo_transition_mask()

  def _actor_local_kl_transition_mask(self) -> torch.Tensor:
    if self.v35_success_local_kl_beta > 0.0:
      return self.v35_success_episode_transition
    return super()._actor_local_kl_transition_mask()

  def _actor_local_kl_beta(self) -> float:
    return self.v35_success_local_kl_beta

  def _backward_actor_objectives(
    self,
    *,
    actor_parameters: list[torch.nn.Parameter],
    actor_loss: torch.Tensor,
    ppo_loss: torch.Tensor,
    moving_kl_loss: torch.Tensor,
    local_kl_loss: torch.Tensor,
    teacher_loss: torch.Tensor,
    entropy: torch.Tensor,
    actor_local_kl_beta: float,
  ) -> dict[str, float]:
    if not self.v35_task_priority_gradient_surgery:
      return super()._backward_actor_objectives(
        actor_parameters=actor_parameters,
        actor_loss=actor_loss,
        ppo_loss=ppo_loss,
        moving_kl_loss=moving_kl_loss,
        local_kl_loss=local_kl_loss,
        teacher_loss=teacher_loss,
        entropy=entropy,
        actor_local_kl_beta=actor_local_kl_beta,
      )

    # PPO on nominal worlds plus both reference anchors is the protected
    # deployment objective.  The filtered-world CBF teacher remains intact
    # when aligned, but loses the component that would increase this loss to
    # first order when the two gradients conflict.
    del actor_loss
    deployment_loss = (
      ppo_loss
      + self.moving_kl_beta * moving_kl_loss
      + actor_local_kl_beta * local_kl_loss
      - self.entropy_coef * entropy
    )
    teacher_objective = self.teacher_distillation_weight * teacher_loss
    deployment_gradients = tuple(
      torch.autograd.grad(
        deployment_loss,
        actor_parameters,
        retain_graph=True,
      )
    )
    teacher_gradients = tuple(
      torch.autograd.grad(teacher_objective, actor_parameters)
    )
    projected_teacher, diagnostics = (
      task_priority_project_auxiliary_gradients(
        deployment_gradients,
        teacher_gradients,
      )
    )
    teacher_for_update = projected_teacher
    deployment_norm = diagnostics["primary_gradient_norm"]
    projected_teacher_norm = diagnostics[
      "projected_auxiliary_gradient_norm"
    ]
    balance_diagnostics = {
      "auxiliary_gradient_balance_scale": 1.0,
      "auxiliary_gradient_requested_scale": 1.0,
      "auxiliary_gradient_scale_capped": 0.0,
      "balanced_auxiliary_gradient_norm": projected_teacher_norm,
      "balanced_auxiliary_to_primary_norm_ratio": (
        projected_teacher_norm / max(deployment_norm, 1.0e-12)
      ),
      "auxiliary_gradient_target_norm_ratio": 0.0,
      "auxiliary_gradient_maximum_scale": (
        TEACHER_GRADIENT_MAXIMUM_SCALE
      ),
    }
    if self.v35_teacher_gradient_target_ratio > 0.0:
      teacher_for_update, balance_diagnostics = (
        capped_norm_balance_auxiliary_gradients(
          deployment_gradients,
          projected_teacher,
          target_ratio=self.v35_teacher_gradient_target_ratio,
          maximum_scale=TEACHER_GRADIENT_MAXIMUM_SCALE,
        )
      )
    for parameter, deployment, teacher in zip(
      actor_parameters, deployment_gradients, teacher_for_update
    ):
      parameter.grad = deployment + teacher
    return {
      "actor_deployment_gradient_norm": diagnostics[
        "primary_gradient_norm"
      ],
      "actor_teacher_gradient_norm": diagnostics[
        "auxiliary_gradient_norm"
      ],
      "actor_projected_teacher_gradient_norm": diagnostics[
        "projected_auxiliary_gradient_norm"
      ],
      "actor_deployment_teacher_gradient_dot": diagnostics[
        "primary_auxiliary_gradient_dot"
      ],
      "actor_deployment_teacher_gradient_cosine": diagnostics[
        "primary_auxiliary_gradient_cosine"
      ],
      "actor_projected_deployment_teacher_gradient_dot": diagnostics[
        "projected_primary_auxiliary_gradient_dot"
      ],
      "actor_teacher_gradient_projection_coefficient": diagnostics[
        "auxiliary_gradient_projection_coefficient"
      ],
      "actor_teacher_gradient_retained_fraction": diagnostics[
        "auxiliary_gradient_retained_fraction"
      ],
      "actor_teacher_gradient_conflict": diagnostics[
        "auxiliary_gradient_conflict"
      ],
      "actor_teacher_gradient_balance_scale": balance_diagnostics[
        "auxiliary_gradient_balance_scale"
      ],
      "actor_teacher_gradient_requested_scale": balance_diagnostics[
        "auxiliary_gradient_requested_scale"
      ],
      "actor_teacher_gradient_scale_capped": balance_diagnostics[
        "auxiliary_gradient_scale_capped"
      ],
      "actor_balanced_teacher_gradient_norm": balance_diagnostics[
        "balanced_auxiliary_gradient_norm"
      ],
      "actor_balanced_teacher_to_deployment_norm_ratio": (
        balance_diagnostics["balanced_auxiliary_to_primary_norm_ratio"]
      ),
      "actor_teacher_gradient_target_norm_ratio": balance_diagnostics[
        "auxiliary_gradient_target_norm_ratio"
      ],
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
    failed_terminal, success_terminal, joint_terminal = (
      disjoint_terminal_outcomes(
        self.storage.dones.squeeze(-1).bool(),
        self.fall_events.bool(),
        self.v35_success_terminals,
      )
    )
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
    if self.v35_split_filter_actor_objectives:
      ppo_environment, teacher_environment = self._split_filter_actor_masks()
    else:
      ppo_environment = torch.ones_like(self.v35_mean_intervened)
      teacher_environment = torch.ones_like(self.v35_mean_intervened)
    teacher_environment_intervention_count = int(
      (self.v35_mean_intervened & teacher_environment).sum()
    )
    eligible_count = int(self.teacher_eligible.sum())
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
        "v35_failure_only_mean_teacher": (
          self.v35_failure_only_mean_teacher
        ),
        "v35_success_only_mean_teacher": (
          self.v35_success_only_mean_teacher
        ),
        "v35_failure_focused_actor": self.v35_failure_focused_actor,
        "v35_distill_only_actor": self.v35_distill_only_actor,
        "v35_success_local_kl_beta": self.v35_success_local_kl_beta,
        "v35_split_filter_actor_objectives": (
          self.v35_split_filter_actor_objectives
        ),
        "v35_task_priority_gradient_surgery": (
          self.v35_task_priority_gradient_surgery
        ),
        "v35_teacher_gradient_target_ratio": (
          self.v35_teacher_gradient_target_ratio
        ),
        "v35_teacher_gradient_maximum_scale": (
          TEACHER_GRADIENT_MAXIMUM_SCALE
        ),
        "v35_ppo_environment_transition_fraction": float(
          ppo_environment.float().mean()
        ),
        "v35_teacher_environment_transition_fraction": float(
          teacher_environment.float().mean()
        ),
        "v35_teacher_environment_intervention_count": float(
          teacher_environment_intervention_count
        ),
        "v35_failed_episode_count": float(
          failed_terminal.sum()
        ),
        "v35_failed_episode_transition_fraction": float(
          self.v35_failed_episode_transition.float().mean()
        ),
        "v35_success_episode_count": float(success_terminal.sum()),
        "v35_joint_success_fall_terminal_count": float(joint_terminal.sum()),
        "v35_success_episode_transition_fraction": float(
          self.v35_success_episode_transition.float().mean()
        ),
        "teacher_eligible_fraction_among_interventions": (
          int(self.teacher_eligible.sum())
          / max(1, teacher_environment_intervention_count)
        ),
        "teacher_actor_coordinate_correction_mean": float(
          self.v35_mean_correction_norm.mean()
        ),
        "mean_policy_to_teacher_action_distance": float(
          self.v35_mean_correction_norm[self.teacher_eligible].mean()
        )
        if eligible_count
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
    self.v35_failed_episode_transition.zero_()
    self.v35_success_terminals.zero_()
    self.v35_success_episode_transition.zero_()

  def save(self) -> dict[str, Any]:
    output = super().save()
    output["proximal_method_id"] = METHOD_ID
    output["v35_teacher_target_source"] = (
      "same_state_cbf_projection_of_round_reference_mean"
    )
    output["v35_failure_only_mean_teacher"] = (
      self.v35_failure_only_mean_teacher
    )
    output["v35_success_only_mean_teacher"] = (
      self.v35_success_only_mean_teacher
    )
    output["v35_failure_focused_actor"] = self.v35_failure_focused_actor
    output["v35_distill_only_actor"] = self.v35_distill_only_actor
    output["v35_success_local_kl_beta"] = self.v35_success_local_kl_beta
    output["v35_split_filter_actor_objectives"] = (
      self.v35_split_filter_actor_objectives
    )
    output["v35_task_priority_gradient_surgery"] = (
      self.v35_task_priority_gradient_surgery
    )
    output["v35_teacher_gradient_target_ratio"] = (
      self.v35_teacher_gradient_target_ratio
    )
    output["v35_teacher_gradient_maximum_scale"] = (
      TEACHER_GRADIENT_MAXIMUM_SCALE
    )
    return output
