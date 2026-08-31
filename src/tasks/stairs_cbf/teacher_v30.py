"""v30 corrective teachers with soft-only moving KL and fixed two-epoch PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .online import (
    validate_behavior_distribution_params,
    validate_behavior_log_prob,
)
from .proximal import ProximalHardRollback, diagonal_gaussian_forward_kl
from .teacher_math import weighted_gaussian_teacher_loss_v29
from .teacher_v29 import (
    CbfTeacherV29PPO,
    CbfTeacherV29PpoAlgorithmCfg,
    CbfTeacherV29Runner,
)
from .teacher_v30_math import (
    intervention_teacher_weights,
    masked_population_mean,
    residual_teacher_target,
    weighted_action_errors,
    weighted_smooth_l1_teacher_loss,
)

METHOD_ID = "soft-kl-cbf-corrective-teacher-v30"
VALID_CONFIGURATIONS = {
    ("none", "none", 0.0, 0.0),
    ("full_action", "local_success_50", 1.0, 0.1),
    ("residual", "all_interventions", 0.25, 1.0),
    ("residual", "all_interventions", 0.50, 1.0),
    ("residual", "all_interventions", 1.00, 1.0),
    ("residual", "local_success_50", 0.50, 1.0),
}


@dataclass
class CbfTeacherV30PpoAlgorithmCfg(CbfTeacherV29PpoAlgorithmCfg):
    """Configuration surface restricted to the six frozen v30 arms."""

    class_name: str = "src.tasks.stairs_cbf.teacher_v30:CbfTeacherV30PPO"
    teacher_mode: str = "residual"
    teacher_gate: str = "all_interventions"
    teacher_eta: float = 0.5
    teacher_smooth_l1_beta: float = 0.05
    v30_smoke_all_arm_diagnostics: bool = False


class CbfTeacherV30PPO(CbfTeacherV29PPO):
    """Complete two actor epochs while treating moving KL only as a loss."""

    accumulate_actor_microbatch_gradients = False

    def __init__(
        self,
        *args,
        teacher_mode: str = "residual",
        teacher_gate: str = "all_interventions",
        teacher_eta: float = 0.5,
        teacher_smooth_l1_beta: float = 0.05,
        teacher_distillation_weight: float = 1.0,
        v30_smoke_all_arm_diagnostics: bool = False,
        **kwargs,
    ) -> None:
        requested_weight = float(teacher_distillation_weight)
        # v29 validates its historical coefficient inside its constructor.
        # Initialize that storage path with 0.1, then install the prospectively
        # validated v30 coefficient before any rollout or optimizer step.
        super().__init__(
            *args,
            teacher_distillation_weight=0.1,
            **kwargs,
        )
        self.teacher_mode = str(teacher_mode)
        self.teacher_gate = str(teacher_gate)
        self.teacher_eta = float(teacher_eta)
        self.teacher_smooth_l1_beta = float(teacher_smooth_l1_beta)
        self.teacher_distillation_weight = requested_weight
        self.v30_smoke_all_arm_diagnostics = bool(v30_smoke_all_arm_diagnostics)
        key = (
            self.teacher_mode,
            self.teacher_gate,
            self.teacher_eta,
            self.teacher_distillation_weight,
        )
        if key not in VALID_CONFIGURATIONS:
            raise ValueError(f"configuration is not one of v30's six arms: {key}")
        if not math.isclose(
            self.teacher_smooth_l1_beta, 0.05, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("v30 fixes Smooth-L1 beta at 0.05")
        t = self.storage.num_transitions_per_env
        n = self.storage.num_envs
        action_dim = self.storage.actions.shape[-1]
        self.v30_reference_means = torch.zeros(t, n, action_dim, device=self.device)
        self.v30_reference_mean_present = torch.zeros(
            t, n, dtype=torch.bool, device=self.device
        )
        self.v30_correction_vectors = torch.zeros_like(self.v30_reference_means)
        self.v30_teacher_targets = torch.zeros_like(self.v30_reference_means)

    def act(self, obs) -> torch.Tensor:
        if self.round_reference_actor is None:
            raise RuntimeError("v30 round reference must be frozen before rollout")
        step = self.storage.step
        actions = super().act(obs)
        if step < self.storage.num_transitions_per_env:
            with torch.inference_mode():
                reference_mean = self.round_reference_actor(
                    obs, stochastic_output=False
                )
            self.v30_reference_means[step].copy_(reference_mean)
            self.v30_reference_mean_present[step] = True
        return actions

    def _ungated_diagnostics(self, eligible: torch.Tensor) -> dict[str, torch.Tensor]:
        zeros = torch.zeros_like(eligible)
        ones = torch.ones_like(eligible)
        return {
            "intervened": self.actual_cbf_intervened.bool(),
            "crossed_within_horizon": zeros,
            "no_fall_within_horizon": ones,
            "no_recovery_takeover_within_horizon": ones,
            "no_emergency_termination_within_horizon": ones,
            "no_unsafe_termination_within_horizon": ones,
            "horizon_outcome_observed": ones,
            "terminal_observed_within_horizon": zeros,
            "magnitude_weight": torch.clamp(
                self.teacher_correction_norm / float(self.teacher_correction_scale),
                0.0,
                1.0,
            ),
        }

    def _compute_teacher_labels(
        self, correction_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.teacher_gate == "local_success_50":
            return super()._compute_teacher_labels(correction_norm)
        if self.teacher_gate == "all_interventions":
            eligible, weights = intervention_teacher_weights(
                self.actual_cbf_intervened,
                correction_norm,
                correction_scale=self.teacher_correction_scale,
            )
            return eligible, weights, self._ungated_diagnostics(eligible)
        if self.teacher_gate == "none":
            eligible = torch.zeros_like(self.actual_cbf_intervened)
            weights = torch.zeros_like(correction_norm)
            return eligible, weights, self._ungated_diagnostics(eligible)
        raise RuntimeError(f"unsupported v30 teacher gate {self.teacher_gate!r}")

    def _teacher_loss(
        self,
        policy_mean: torch.Tensor,
        policy_std: torch.Tensor,
        target: torch.Tensor,
        eligible: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.teacher_mode == "none":
            return policy_mean.sum() * 0.0
        if self.teacher_mode == "full_action":
            return weighted_gaussian_teacher_loss_v29(
                policy_mean,
                policy_std,
                target,
                eligible,
                weights,
                epsilon=1.0e-8,
            )
        if self.teacher_mode == "residual":
            return weighted_smooth_l1_teacher_loss(
                policy_mean,
                target,
                eligible,
                weights,
                beta=self.teacher_smooth_l1_beta,
                epsilon=1.0e-8,
            )
        raise RuntimeError(f"unsupported v30 teacher mode {self.teacher_mode!r}")

    def _actor_ppo_transition_mask(self) -> torch.Tensor:
        """Select transitions that contribute PPO and entropy actor gradients."""
        return torch.ones_like(self.teacher_eligible, dtype=torch.bool)

    def _actor_ppo_transition_weights(self) -> torch.Tensor | None:
        """Optionally replace the historical boolean PPO mask with soft weights.

        ``None`` deliberately preserves v30's exact population-mean path.  New
        descendants may return a non-negative ``[T, N]`` tensor to attenuate
        PPO gradients without removing the corresponding critic transitions.
        """
        return None

    def _actor_local_kl_transition_mask(self) -> torch.Tensor:
        """Select transitions receiving an additional round-reference KL."""
        return torch.zeros_like(self.teacher_eligible, dtype=torch.bool)

    def _actor_local_kl_beta(self) -> float:
        return 0.0

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        denominator = weights.sum()
        if not bool(denominator > 0.0):
            return values.sum() * 0.0
        return (weights * values).sum() / (denominator + 1.0e-8)

    def relabel_teacher_transitions(self) -> dict[str, Any]:
        if not bool(self.v30_reference_mean_present.all()):
            missing = int((~self.v30_reference_mean_present).sum())
            raise RuntimeError(f"v30 reference mean missing on {missing} transitions")
        metrics = super().relabel_teacher_transitions()
        target, correction = residual_teacher_target(
            self.v30_reference_means,
            self.teacher_policy_actions,
            self.policy_actions,
            eta=(self.teacher_eta if self.teacher_mode == "residual" else 1.0),
        )
        if self.teacher_mode == "full_action":
            target = self.teacher_policy_actions.detach()
        elif self.teacher_mode == "none":
            target = self.v30_reference_means.detach()
        self.v30_teacher_targets.copy_(target)
        self.v30_correction_vectors.copy_(correction)
        stored_behavior_mean = self.storage.distribution_params[0]
        reference_mean_error = float(
            torch.amax(torch.abs(stored_behavior_mean - self.v30_reference_means))
        )
        if reference_mean_error > 1.0e-6:
            raise RuntimeError(
                "v30 stored behavior mean differs from round reference: "
                f"{reference_mean_error}"
            )
        flat_weights = self.teacher_weights.flatten()
        flat_eligible = self.teacher_eligible.flatten()
        effective = flat_weights * flat_eligible.to(flat_weights.dtype)
        correction_norm = torch.linalg.vector_norm(correction.flatten(0, 1), dim=-1)
        residual_norm = torch.linalg.vector_norm(
            (target - self.v30_reference_means).flatten(0, 1), dim=-1
        )
        before_distance, before_per_action = weighted_action_errors(
            stored_behavior_mean.flatten(0, 1),
            target.flatten(0, 1),
            flat_eligible,
            flat_weights,
        )
        intervention_count = int(self.actual_cbf_intervened.sum())
        metrics.update(
            {
                "teacher_mode": self.teacher_mode,
                "teacher_gate_mode": self.teacher_gate,
                "teacher_eta": self.teacher_eta,
                "teacher_smooth_l1_beta": self.teacher_smooth_l1_beta,
                "teacher_weight_sum": float(effective.sum()),
                "teacher_fraction_among_interventions": float(
                    self.teacher_eligible.sum() / max(1, intervention_count)
                ),
                "mean_cbf_correction_norm": float(
                    correction_norm[self.actual_cbf_intervened.flatten()].mean()
                )
                if intervention_count
                else 0.0,
                "mean_residual_target_norm": float(
                    self._weighted_mean(residual_norm, effective)
                ),
                "mean_policy_to_target_distance_before_update": float(before_distance),
                "per_action_teacher_error_before_update": [
                    float(value) for value in before_per_action
                ],
                "reference_behavior_mean_max_abs_error": reference_mean_error,
                "teacher_target_shape": list(target.shape),
                "teacher_target_stop_gradient": not target.requires_grad,
                "teacher_tensor_shapes": {
                    "raw_sampled_action": list(self.policy_actions.shape),
                    "safe_sampled_action": list(self.teacher_policy_actions.shape),
                    "round_reference_mean": list(self.v30_reference_means.shape),
                    "intervention": list(self.actual_cbf_intervened.shape),
                    "correction_vector": list(self.v30_correction_vectors.shape),
                    "riser": list(self.stair_indices.shape),
                    "fall": list(self.fall_events.shape),
                    "done": list(self.storage.dones.shape),
                    "episode": list(self.teacher_episode_ids.shape),
                },
            }
        )
        if self.v30_smoke_all_arm_diagnostics:
            means = stored_behavior_mean.flatten(0, 1)
            stds = self.storage.distribution_params[1].flatten(0, 1)
            safe = self.teacher_policy_actions.flatten(0, 1)
            reference = self.v30_reference_means.flatten(0, 1)
            raw = self.policy_actions.flatten(0, 1)
            all_eligible, all_weights_2d = intervention_teacher_weights(
                self.actual_cbf_intervened,
                self.teacher_correction_norm,
                correction_scale=self.teacher_correction_scale,
            )
            all_eligible = all_eligible.flatten()
            all_weights = all_weights_2d.flatten()
            success_eligible, success_weights, _ = super()._compute_teacher_labels(
                self.teacher_correction_norm
            )
            success_eligible = success_eligible.flatten()
            success_weights = success_weights.flatten()
            losses: dict[str, float] = {"A0": 0.0}
            losses["A1"] = float(
                weighted_gaussian_teacher_loss_v29(
                    means,
                    stds,
                    safe,
                    success_eligible,
                    success_weights,
                    epsilon=1.0e-8,
                )
            )
            for arm, eta, eligible, weights in (
                ("A2", 0.25, all_eligible, all_weights),
                ("A3", 0.50, all_eligible, all_weights),
                ("A4", 1.00, all_eligible, all_weights),
                ("A5", 0.50, success_eligible, success_weights),
            ):
                residual, _ = residual_teacher_target(reference, safe, raw, eta=eta)
                losses[arm] = float(
                    weighted_smooth_l1_teacher_loss(
                        means,
                        residual,
                        eligible,
                        weights,
                        beta=0.05,
                        epsilon=1.0e-8,
                    )
                )
            metrics["smoke_all_arm_teacher_losses"] = losses
            instantiated = []
            for mode, gate, eta, weight in sorted(VALID_CONFIGURATIONS):
                configuration = CbfTeacherV30PpoAlgorithmCfg(
                    teacher_mode=mode,
                    teacher_gate=gate,
                    teacher_eta=eta,
                    teacher_distillation_weight=weight,
                )
                instantiated.append(
                    (
                        configuration.teacher_mode,
                        configuration.teacher_gate,
                        configuration.teacher_eta,
                        configuration.teacher_distillation_weight,
                    )
                )
            metrics["smoke_all_arm_configurations_instantiated"] = instantiated
        self.last_update_metrics.update(metrics)
        return metrics

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
        """Backpropagate the historical summed objective.

        Subclasses may override this narrow hook to compose objective gradients
        while leaving rollout alignment, optimizer stepping, clipping, and all
        critic semantics owned by the frozen v30 update.
        """
        del (
            actor_parameters,
            ppo_loss,
            moving_kl_loss,
            local_kl_loss,
            teacher_loss,
            entropy,
            actor_local_kl_beta,
        )
        actor_loss.backward()
        return {}

    def update(self) -> dict[str, Any]:
        """Run exactly two actor epochs; KL is never a stop or rollback gate."""
        if self.round_reference_actor is None:
            raise RuntimeError("v30 round-start reference was not frozen")
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("v30 PPO requires one complete rollout")
        if (
            self.rnd
            or self.symmetry
            or self.actor.is_recurrent
            or self.critic.is_recurrent
        ):
            raise RuntimeError("v30 supports feed-forward PPO without RND/symmetry")

        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1).clone()
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1).clone()
        reference_params = tuple(
            value.flatten(0, 1).clone().detach()
            for value in self.storage.distribution_params
        )
        advantages = self.storage.advantages.flatten().detach()
        teacher_targets = self.v30_teacher_targets.flatten(0, 1).detach()
        teacher_eligible = self.teacher_eligible.flatten().detach()
        teacher_weights = self.teacher_weights.flatten().detach()
        actor_ppo_mask = self._actor_ppo_transition_mask().flatten().detach()
        actor_ppo_weights_2d = self._actor_ppo_transition_weights()
        actor_ppo_weights = (
            None
            if actor_ppo_weights_2d is None
            else actor_ppo_weights_2d.flatten().detach()
        )
        actor_local_kl_mask = (
            self._actor_local_kl_transition_mask().flatten().detach()
        )
        actor_local_kl_beta = float(self._actor_local_kl_beta())
        if (
            actor_ppo_mask.shape != advantages.shape
            or actor_ppo_mask.dtype != torch.bool
        ):
            raise RuntimeError("v30 actor PPO mask must be boolean with shape [T*N]")
        if actor_ppo_weights is not None:
            if actor_ppo_weights.shape != advantages.shape:
                raise RuntimeError(
                    "v30 actor PPO weights must have shape [T*N]"
                )
            if not bool(torch.isfinite(actor_ppo_weights).all()) or bool(
                (actor_ppo_weights < 0.0).any()
            ):
                raise RuntimeError(
                    "v30 actor PPO weights must be finite and non-negative"
                )
            actor_ppo_mask = actor_ppo_weights > 0.0
        if (
            actor_local_kl_mask.shape != advantages.shape
            or actor_local_kl_mask.dtype != torch.bool
        ):
            raise RuntimeError(
                "v30 actor local-KL mask must be boolean with shape [T*N]"
            )
        if not math.isfinite(actor_local_kl_beta) or actor_local_kl_beta < 0.0:
            raise RuntimeError("v30 actor local-KL beta must be finite and non-negative")
        if not bool(torch.isfinite(advantages).all()):
            raise ProximalHardRollback("non-finite v30 advantages")

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
            current_before = tuple(self.actor.output_distribution_params)
            current_log_prob_before = self.actor.get_output_log_prob(actions)
            current_param_error = validate_behavior_distribution_params(
                reference_params, current_before
            )
            current_log_prob_error = validate_behavior_log_prob(
                old_log_prob, current_log_prob_before
            )
        before_distance, before_per_action = weighted_action_errors(
            reference_params[0],
            teacher_targets,
            teacher_eligible,
            teacher_weights,
        )

        totals = {
            "actor": 0.0,
            "ppo": 0.0,
            "moving_kl": 0.0,
            "local_kl": 0.0,
            "teacher": 0.0,
            "entropy": 0.0,
        }
        actor_updates = 0
        actor_backward_passes = 0
        actor_gradient_norm_max = 0.0
        clipped_gradient_updates = 0
        teacher_minibatches_with_signal = 0
        teacher_minibatches_without_signal = 0
        actor_ppo_minibatches_with_signal = 0
        actor_ppo_minibatches_without_signal = 0
        actor_objective_diagnostic_updates = 0
        actor_objective_diagnostic_totals: dict[str, float] = {}
        minibatch_diagnostics: list[dict[str, float | int]] = []
        epoch_moving_kl: list[float] = []
        batch_size = actions.shape[0]
        if batch_size % self.num_mini_batches:
            raise RuntimeError("v30 rollout must divide evenly into minibatches")
        mini_batch_size = batch_size // self.num_mini_batches
        accumulate_actor_gradients = bool(
            self.accumulate_actor_microbatch_gradients
        )
        actor_loss_scale = (
            1.0 / self.num_mini_batches if accumulate_actor_gradients else 1.0
        )

        for epoch in range(self.num_learning_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            if accumulate_actor_gradients:
                self.actor_optimizer.zero_grad(set_to_none=True)
            for mini_batch in range(self.num_mini_batches):
                start = mini_batch * mini_batch_size
                indices = permutation[start : start + mini_batch_size]
                batch_observations = observations[indices]
                batch_actions = actions[indices]
                self.actor(batch_observations, stochastic_output=True)
                new_log_prob = self.actor.get_output_log_prob(batch_actions)
                current_params = tuple(self.actor.output_distribution_params)
                log_ratio = new_log_prob - old_log_prob[indices]
                ratio = torch.exp(log_ratio)
                advantage = advantages[indices]
                surrogate = -advantage * ratio
                surrogate_clipped = -advantage * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                batch_actor_ppo_mask = actor_ppo_mask[indices]
                surrogate_maximum = torch.maximum(surrogate, surrogate_clipped)
                if actor_ppo_weights is None:
                    ppo_loss = masked_population_mean(
                        surrogate_maximum, batch_actor_ppo_mask
                    )
                else:
                    ppo_loss = self._weighted_mean(
                        surrogate_maximum, actor_ppo_weights[indices]
                    )
                has_actor_ppo_signal = bool(batch_actor_ppo_mask.any())
                actor_ppo_minibatches_with_signal += int(has_actor_ppo_signal)
                actor_ppo_minibatches_without_signal += int(
                    not has_actor_ppo_signal
                )
                forward_kl = diagonal_gaussian_forward_kl(
                    current_params,
                    tuple(value[indices] for value in reference_params),
                )
                moving_kl_loss = forward_kl.mean()
                local_kl_loss = self._weighted_mean(
                    forward_kl,
                    actor_local_kl_mask[indices].to(forward_kl.dtype),
                )
                batch_eligible = teacher_eligible[indices]
                batch_weights = teacher_weights[indices]
                teacher_loss = self._teacher_loss(
                    current_params[0],
                    current_params[1],
                    teacher_targets[indices],
                    batch_eligible,
                    batch_weights,
                )
                has_signal = bool((batch_weights * batch_eligible).sum() > 0.0)
                teacher_minibatches_with_signal += int(has_signal)
                teacher_minibatches_without_signal += int(not has_signal)
                entropy = masked_population_mean(
                    self.actor.output_entropy.flatten(), batch_actor_ppo_mask
                )
                actor_loss = (
                    ppo_loss
                    + self.moving_kl_beta * moving_kl_loss
                    + actor_local_kl_beta * local_kl_loss
                    + self.teacher_distillation_weight * teacher_loss
                    - self.entropy_coef * entropy
                )
                if not bool(torch.isfinite(actor_loss)):
                    raise ProximalHardRollback("non-finite v30 actor loss")
                actor_parameters = list(self.actor.mlp.parameters())
                if not accumulate_actor_gradients:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                objective_gradient_diagnostics = (
                    self._backward_actor_objectives(
                        actor_parameters=actor_parameters,
                        actor_loss=actor_loss * actor_loss_scale,
                        ppo_loss=ppo_loss * actor_loss_scale,
                        moving_kl_loss=moving_kl_loss * actor_loss_scale,
                        local_kl_loss=local_kl_loss * actor_loss_scale,
                        teacher_loss=teacher_loss * actor_loss_scale,
                        entropy=entropy * actor_loss_scale,
                        actor_local_kl_beta=actor_local_kl_beta,
                    )
                )
                actor_backward_passes += 1
                if not self._finite_gradients(actor_parameters):
                    raise ProximalHardRollback("non-finite v30 actor gradient")
                if objective_gradient_diagnostics:
                    actor_objective_diagnostic_updates += 1
                    for name, value in objective_gradient_diagnostics.items():
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            raise ProximalHardRollback(
                                "non-finite actor objective-gradient diagnostic"
                            )
                        actor_objective_diagnostic_totals[name] = (
                            actor_objective_diagnostic_totals.get(name, 0.0)
                            + numeric
                        )
                totals["actor"] += actor_loss_scale * float(actor_loss.detach())
                totals["ppo"] += actor_loss_scale * float(ppo_loss.detach())
                totals["moving_kl"] += (
                    actor_loss_scale * float(moving_kl_loss.detach())
                )
                totals["local_kl"] += (
                    actor_loss_scale * float(local_kl_loss.detach())
                )
                totals["teacher"] += (
                    actor_loss_scale * float(teacher_loss.detach())
                )
                totals["entropy"] += actor_loss_scale * float(entropy.detach())
                should_step_actor = (
                    not accumulate_actor_gradients
                    or mini_batch + 1 == self.num_mini_batches
                )
                if should_step_actor:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        actor_parameters, self.max_grad_norm
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise ProximalHardRollback(
                            "non-finite v30 actor gradient norm"
                        )
                    clipped_gradient_updates += int(
                        float(gradient_norm) > self.max_grad_norm
                    )
                    self.actor_optimizer.step()
                    self._raise_if_corrupt(
                        f"v30 actor epoch {epoch + 1} optimizer step"
                    )
                    with torch.inference_mode():
                        self.actor(batch_observations, stochastic_output=True)
                        post_params = tuple(self.actor.output_distribution_params)
                        post_log_prob = self.actor.get_output_log_prob(batch_actions)
                        post_log_ratio = post_log_prob - old_log_prob[indices]
                        post_ratio = torch.exp(post_log_ratio)
                        diagnostic = {
                            "epoch": epoch + 1,
                            "minibatch": mini_batch + 1,
                            "accumulated_microbatches": (
                                self.num_mini_batches
                                if accumulate_actor_gradients
                                else 1
                            ),
                            "moving_forward_kl": float(
                                diagonal_gaussian_forward_kl(
                                    post_params,
                                    tuple(
                                        value[indices]
                                        for value in reference_params
                                    ),
                                ).mean()
                            ),
                            "behavior_approximate_kl": float(
                                (-post_log_ratio).mean()
                            ),
                            "clip_fraction": float(
                                (torch.abs(post_ratio - 1.0) > self.clip_param)
                                .float()
                                .mean()
                            ),
                            "action_mean_shift": float(
                                torch.linalg.vector_norm(
                                    post_params[0] - reference_params[0][indices],
                                    dim=-1,
                                ).mean()
                            ),
                            "actor_gradient_norm_pre_clip": float(gradient_norm),
                            **objective_gradient_diagnostics,
                        }
                    if not all(
                        math.isfinite(float(value))
                        for key, value in diagnostic.items()
                        if key not in (
                            "epoch",
                            "minibatch",
                            "accumulated_microbatches",
                        )
                    ):
                        raise ProximalHardRollback(
                            "non-finite v30 minibatch diagnostic"
                        )
                    minibatch_diagnostics.append(diagnostic)
                    actor_updates += 1
                    actor_gradient_norm_max = max(
                        actor_gradient_norm_max, float(gradient_norm)
                    )
            epoch_metrics = self._whole_batch_policy_metrics(
                observations, actions, old_log_prob, reference_params
            )
            epoch_moving_kl.append(epoch_metrics["moving_forward_kl"])

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
                    raise ProximalHardRollback("non-finite v30 critic loss")
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_parameters = list(self.critic.parameters())
                if not self._finite_gradients(critic_parameters):
                    raise ProximalHardRollback("non-finite v30 critic gradient")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, self.max_grad_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise ProximalHardRollback("non-finite v30 critic gradient norm")
                self.critic_optimizer.step()
                self._raise_if_corrupt(f"v30 critic epoch {epoch + 1}")
                critic_updates += 1
                critic_loss_total += float(value_loss.detach())
                critic_gradient_norm_max = max(
                    critic_gradient_norm_max, float(gradient_norm)
                )

        expected_actor_backward_passes = (
            self.num_learning_epochs * self.num_mini_batches
        )
        expected_actor_updates = self.num_learning_epochs * (
            1 if accumulate_actor_gradients else self.num_mini_batches
        )
        expected_critic_updates = self.critic_learning_epochs * self.num_mini_batches
        if (
            actor_backward_passes != expected_actor_backward_passes
            or actor_updates != expected_actor_updates
            or critic_updates != expected_critic_updates
        ):
            raise ProximalHardRollback("v30 did not complete every optimizer step")
        self.clamp_online_std()
        final_policy = self._whole_batch_policy_metrics(
            observations, actions, old_log_prob, reference_params
        )
        self._raise_if_corrupt("complete v30 update")
        with torch.inference_mode():
            self.actor(observations, stochastic_output=True)
            final_mean = self.actor.output_distribution_params[0]
            after_distance, after_per_action = weighted_action_errors(
                final_mean,
                teacher_targets,
                teacher_eligible,
                teacher_weights,
            )

        rollout_metrics = dict(self.last_update_metrics)
        result: dict[str, Any] = {
            "actor_loss": totals["actor"] / actor_updates,
            "surrogate": totals["ppo"] / actor_updates,
            "moving_forward_kl_loss": totals["moving_kl"] / actor_updates,
            "actor_local_preservation_kl_loss": (
                totals["local_kl"] / actor_updates
            ),
            "actor_local_preservation_kl_beta": actor_local_kl_beta,
            "actor_local_preservation_transition_count": int(
                actor_local_kl_mask.sum()
            ),
            "actor_local_preservation_transition_fraction": float(
                actor_local_kl_mask.float().mean()
            ),
            "teacher_loss": totals["teacher"] / actor_updates,
            "teacher_huber_loss_mean": (
                totals["teacher"] / actor_updates
                if self.teacher_mode == "residual"
                else 0.0
            ),
            "teacher_distillation_weight": self.teacher_distillation_weight,
            "teacher_mode": self.teacher_mode,
            "teacher_gate_mode": self.teacher_gate,
            "teacher_eta": self.teacher_eta,
            "teacher_smooth_l1_beta": self.teacher_smooth_l1_beta,
            "teacher_minibatches_with_signal": teacher_minibatches_with_signal,
            "teacher_minibatches_without_signal": teacher_minibatches_without_signal,
            "actor_ppo_transition_count": int(actor_ppo_mask.sum()),
            "actor_ppo_transition_fraction": float(actor_ppo_mask.float().mean()),
            "actor_ppo_soft_weights_enabled": actor_ppo_weights is not None,
            "actor_ppo_transition_weight_mean": (
                float(actor_ppo_weights.mean())
                if actor_ppo_weights is not None
                else float(actor_ppo_mask.float().mean())
            ),
            "actor_ppo_transition_weight_sum": (
                float(actor_ppo_weights.sum())
                if actor_ppo_weights is not None
                else float(actor_ppo_mask.sum())
            ),
            "actor_ppo_minibatches_with_signal": (
                actor_ppo_minibatches_with_signal
            ),
            "actor_ppo_minibatches_without_signal": (
                actor_ppo_minibatches_without_signal
            ),
            "actor_entropy_uses_ppo_transition_mask": True,
            "moving_kl_uses_all_transitions": True,
            "critic_uses_all_transitions": True,
            "value": critic_loss_total / critic_updates,
            "entropy": totals["entropy"] / actor_updates,
            "moving_kl_beta": self.moving_kl_beta,
            "target_kl_early_stopping_enabled": False,
            "hard_kl_rollback_enabled": False,
            "actor_epochs_completed": self.num_learning_epochs,
            "critic_epochs_completed": self.critic_learning_epochs,
            "actor_minibatches_completed": actor_backward_passes,
            "actor_optimizer_updates_completed": actor_updates,
            "actor_gradient_accumulation_enabled": accumulate_actor_gradients,
            "actor_gradient_accumulation_microbatches": (
                self.num_mini_batches if accumulate_actor_gradients else 1
            ),
            "critic_minibatches_completed": critic_updates,
            "epoch_moving_forward_kl": epoch_moving_kl,
            "minibatch_diagnostics": minibatch_diagnostics,
            "actor_gradient_norm_pre_clip_max": actor_gradient_norm_max,
            "actor_gradient_clipped_fraction": (
                clipped_gradient_updates / actor_updates
            ),
            "actor_objective_gradient_surgery_enabled": (
                actor_objective_diagnostic_updates > 0
            ),
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
            "mean_policy_to_target_distance_before_update": float(before_distance),
            "mean_policy_to_target_distance_after_update": float(after_distance),
            "per_action_teacher_error_before_update": [
                float(value) for value in before_per_action
            ],
            "per_action_teacher_error_after_update": [
                float(value) for value in after_per_action
            ],
            **final_policy,
            **rollout_metrics,
        }
        if actor_objective_diagnostic_updates:
            for name, total in actor_objective_diagnostic_totals.items():
                suffix = "fraction" if name.endswith("_conflict") else "mean"
                result[f"{name}_{suffix}"] = (
                    total / actor_objective_diagnostic_updates
                )
        result["mean_kl"] = result["moving_forward_kl"]
        self.storage.clear()
        self.clear_cbf_rollout()
        self.last_update_metrics = {}
        return result

    def clear_cbf_rollout(self) -> None:
        super().clear_cbf_rollout()
        if not hasattr(self, "v30_reference_means"):
            return
        self.v30_reference_means.zero_()
        self.v30_reference_mean_present.zero_()
        self.v30_correction_vectors.zero_()
        self.v30_teacher_targets.zero_()

    def save(self) -> dict[str, Any]:
        output = super().save()
        output.update(
            {
                "proximal_method_id": METHOD_ID,
                "v30_teacher_mode": self.teacher_mode,
                "v30_teacher_gate": self.teacher_gate,
                "v30_teacher_eta": self.teacher_eta,
                "v30_teacher_weight": self.teacher_distillation_weight,
                "v30_soft_kl_only": True,
            }
        )
        return output


class CbfTeacherV30Runner(CbfTeacherV29Runner):
    """Runner for exact v30 base warm-start and recovery checkpoints."""

    alg: CbfTeacherV30PPO

    def load_recovery_checkpoint(
        self, path: str, map_location: str | None = None
    ) -> dict[str, Any]:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        if loaded.get("proximal_method_id") != METHOD_ID:
            raise ValueError("recovery checkpoint is not a v30 checkpoint")
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
        self.alg._raise_if_corrupt("v30 recovery checkpoint load")
        return {
            "source_iteration": int(loaded.get("iter", -1)),
            "recovered_actor_optimizer": True,
            "recovered_critic_optimizer": True,
            "recovered_round_reference": reference is not None,
        }
