"""Intervention-aware CBF distillation PPO for filter-free deployment (v141)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .online import OnlineSafePPO
from .teacher_v30 import (
    CbfTeacherV30PPO,
    CbfTeacherV30PpoAlgorithmCfg,
    CbfTeacherV30Runner,
)
from .teacher_v30_math import weighted_action_errors


METHOD_ID = "intervention-aware-cbf-distillation-ppo-v141"
CORRECTION_WEIGHT_MODES = (
    "intervention_only",
    "positive_advantage",
    "episode_success_positive_advantage",
)


def normalize_context_group_advantages(
    advantages: torch.Tensor,
    target_environment_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standardize target and F1-retention advantages independently."""
    if advantages.ndim != 2:
        raise ValueError("v141 advantages must have shape [T, N]")
    if (
        target_environment_mask.shape != advantages.shape[1:]
        or target_environment_mask.dtype != torch.bool
    ):
        raise ValueError("v141 target mask must be boolean with shape [N]")
    if not bool(target_environment_mask.any()) or bool(
        target_environment_mask.all()
    ):
        raise ValueError("v141 target and F1 groups must both be non-empty")
    if not bool(torch.isfinite(advantages).all()):
        raise RuntimeError("v141 advantages contain non-finite values")

    normalized = torch.empty_like(advantages)
    metrics: dict[str, float] = {}
    for name, environment_mask in (
        ("target", target_environment_mask),
        ("retention_f1", ~target_environment_mask),
    ):
        values = advantages[:, environment_mask]
        mean = values.mean()
        std = values.std(unbiased=False)
        group = (values - mean) / (std + 1.0e-8)
        normalized[:, environment_mask] = group
        metrics.update(
            {
                f"{name}_advantage_count": float(values.numel()),
                f"{name}_advantage_mean_before": float(mean),
                f"{name}_advantage_std_before": float(std),
                f"{name}_advantage_mean_after": float(group.mean()),
                f"{name}_advantage_std_after": float(
                    group.std(unbiased=False)
                ),
            }
        )
    metrics.update(
        {
            "context_group_advantages_standardized_separately": 1.0,
            "balanced_advantage_mean": float(normalized.mean()),
            "balanced_advantage_std": float(normalized.std(unbiased=False)),
        }
    )
    return normalized, metrics


def intervention_aware_ppo_weights(
    intervention_mask: torch.Tensor, eta: float
) -> torch.Tensor:
    """Return ``1`` off intervention and ``eta`` on intervention."""
    if intervention_mask.dtype != torch.bool:
        raise ValueError("v141 intervention mask must be boolean")
    if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("v141 PPO intervention eta must lie in [0, 1]")
    return torch.where(
        intervention_mask,
        torch.full_like(intervention_mask, eta, dtype=torch.float32),
        torch.ones_like(intervention_mask, dtype=torch.float32),
    )


def successful_episode_transition_mask(
    episode_ids: torch.Tensor, success_terminal: torch.Tensor
) -> torch.Tensor:
    """Mark transitions belonging to an episode that ends successfully."""
    if episode_ids.shape != success_terminal.shape:
        raise ValueError("v141 episode IDs and terminal-success mask must match")
    if episode_ids.ndim != 2 or success_terminal.dtype != torch.bool:
        raise ValueError("v141 episode tensors must be [T, N] long/bool")
    output = torch.zeros_like(success_terminal)
    for environment in range(episode_ids.shape[1]):
        successful_ids = episode_ids[:, environment][
            success_terminal[:, environment]
        ]
        if successful_ids.numel():
            output[:, environment] = torch.isin(
                episode_ids[:, environment], successful_ids
            )
    return output


def correction_distillation_weights(
    intervention_mask: torch.Tensor,
    correction_norm: torch.Tensor,
    normalized_advantages: torch.Tensor,
    *,
    correction_scale: float,
    mode: str,
    successful_episode_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose the prescribed intervention/magnitude/outcome weight."""
    if not (
        intervention_mask.shape
        == correction_norm.shape
        == normalized_advantages.shape
    ):
        raise ValueError("v141 correction-weight tensors must have equal shape")
    if intervention_mask.dtype != torch.bool:
        raise ValueError("v141 correction intervention mask must be boolean")
    if mode not in CORRECTION_WEIGHT_MODES:
        raise ValueError(f"unsupported v141 correction weight mode {mode!r}")
    if not math.isfinite(correction_scale) or correction_scale <= 0.0:
        raise ValueError("v141 correction scale must be finite and positive")
    if not bool(
        torch.isfinite(correction_norm).all()
        and torch.isfinite(normalized_advantages).all()
    ):
        raise RuntimeError("v141 correction weights received non-finite values")

    weights = intervention_mask.to(correction_norm.dtype) * torch.clamp(
        correction_norm / correction_scale, 0.0, 1.0
    )
    if mode != "intervention_only":
        weights = weights * torch.relu(normalized_advantages)
    if mode == "episode_success_positive_advantage":
        if (
            successful_episode_mask is None
            or successful_episode_mask.shape != intervention_mask.shape
            or successful_episode_mask.dtype != torch.bool
        ):
            raise ValueError("v141 successful-episode mask is missing or invalid")
        weights = weights * successful_episode_mask.to(weights.dtype)
    return weights


@dataclass
class InterventionAwareCbfDistillationPpoAlgorithmCfg(
    CbfTeacherV30PpoAlgorithmCfg
):
    class_name: str = (
        "src.tasks.stairs_cbf.filter_free_v141:"
        "InterventionAwareCbfDistillationPPO"
    )
    teacher_mode: str = "residual"
    teacher_gate: str = "all_interventions"
    teacher_eta: float = 1.0
    teacher_distillation_weight: float = 1.0
    intervention_ppo_eta: float = 0.25
    correction_weight_mode: str = "positive_advantage"
    v141_dual_reward_scale: float = 0.25
    v141_target_fraction: float = 0.80
    v141_intervention_epsilon: float = 1.0e-5


class InterventionAwareCbfDistillationPPO(CbfTeacherV30PPO):
    """PPO on nominal samples, with softly downweighted shielded transitions."""

    def __init__(
        self,
        *args,
        teacher_mode: str = "residual",
        teacher_gate: str = "all_interventions",
        teacher_eta: float = 1.0,
        teacher_distillation_weight: float = 1.0,
        intervention_ppo_eta: float = 0.25,
        correction_weight_mode: str = "positive_advantage",
        v141_dual_reward_scale: float = 0.25,
        v141_target_fraction: float = 0.80,
        v141_intervention_epsilon: float = 1.0e-5,
        **kwargs,
    ) -> None:
        if (
            teacher_mode != "residual"
            or teacher_gate != "all_interventions"
            or not math.isclose(teacher_eta, 1.0, abs_tol=1.0e-12)
        ):
            raise ValueError(
                "v141 requires a full safe-action Smooth-L1 correction target"
            )
        requested_distillation_weight = float(teacher_distillation_weight)
        requested_actor_epochs = int(kwargs.get("num_learning_epochs", 2))
        if requested_actor_epochs not in (2, 3, 4):
            raise ValueError("v141 PPO epochs must be 2, 3, or 4")
        # The inherited v23 constructor historically admits at most two actor
        # epochs.  v30's update itself is epoch-generic, so initialize through
        # the compatible boundary and then install v141's requested count.
        if requested_actor_epochs > 2:
            kwargs["num_learning_epochs"] = 2
        # Initialize through one frozen valid v30 arm, then install v141's
        # independently tuned correction coefficient.
        super().__init__(
            *args,
            teacher_mode="residual",
            teacher_gate="all_interventions",
            teacher_eta=1.0,
            teacher_distillation_weight=1.0,
            **kwargs,
        )
        self.teacher_distillation_weight = requested_distillation_weight
        self.num_learning_epochs = requested_actor_epochs
        self.intervention_ppo_eta = float(intervention_ppo_eta)
        self.correction_weight_mode = str(correction_weight_mode)
        self.v141_dual_reward_scale = float(v141_dual_reward_scale)
        self.v141_target_fraction = float(v141_target_fraction)
        self.v141_intervention_epsilon = float(v141_intervention_epsilon)
        if (
            not math.isfinite(self.teacher_distillation_weight)
            or self.teacher_distillation_weight < 0.0
        ):
            raise ValueError("v141 correction coefficient must be non-negative")
        if not 0.0 <= self.intervention_ppo_eta <= 1.0:
            raise ValueError("v141 PPO intervention eta must lie in [0, 1]")
        if self.correction_weight_mode not in CORRECTION_WEIGHT_MODES:
            raise ValueError("v141 correction weighting mode is invalid")
        if self.v141_dual_reward_scale not in (0.0, 0.25, 1.0):
            raise ValueError("v141 dual reward scale must be 0, 0.25, or 1")
        if not 0.5 < self.v141_target_fraction < 1.0:
            raise ValueError("v141 target fraction must lie strictly in (0.5, 1)")
        if self.v141_intervention_epsilon <= 0.0:
            raise ValueError("v141 intervention epsilon must be positive")

        t = self.storage.num_transitions_per_env
        n = self.storage.num_envs
        self.v141_intervention_mask = torch.zeros(
            t, n, dtype=torch.bool, device=self.device
        )
        self.v141_success_terminal = torch.zeros_like(self.v141_intervention_mask)
        self.v141_success_telemetry_present = torch.zeros_like(
            self.v141_intervention_mask
        )
        self.v141_successful_episode = torch.zeros_like(
            self.v141_intervention_mask
        )
        self.v141_group_ids = torch.full(
            (t, n), -1, dtype=torch.long, device=self.device
        )
        self.v141_task_rewards = torch.zeros(t, n, device=self.device)
        self.v141_cbf_rewards = torch.zeros_like(self.v141_task_rewards)
        self.v141_reward_telemetry_present = torch.zeros_like(
            self.v141_intervention_mask
        )
        self.v141_target_environment_mask: torch.Tensor | None = None

    def set_context_group_mask(self, target_environment_mask: torch.Tensor) -> None:
        n = self.storage.num_envs
        if (
            target_environment_mask.shape != (n,)
            or target_environment_mask.dtype != torch.bool
        ):
            raise ValueError("v141 target environment mask must be boolean [N]")
        if not bool(target_environment_mask.any()) or bool(
            target_environment_mask.all()
        ):
            raise ValueError("v141 target and F1 groups must both be non-empty")
        actual_fraction = float(target_environment_mask.float().mean())
        tolerance = 1.0 / n + 1.0e-8
        if abs(actual_fraction - self.v141_target_fraction) > tolerance:
            raise ValueError(
                "v141 target environment fraction differs from configuration"
            )
        self.v141_target_environment_mask = (
            target_environment_mask.to(self.device).detach().clone()
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
            task_reward = extras.get("v141_task_reward")
            cbf_reward = extras.get("v141_cbf_reward")
            success = extras.get("v141_success_terminal")
            group_ids = extras.get("v141_context_index")
            if task_reward is not None and cbf_reward is not None:
                self.v141_task_rewards[step].copy_(task_reward)
                self.v141_cbf_rewards[step].copy_(cbf_reward)
                self.v141_reward_telemetry_present[step] = True
            if success is not None:
                self.v141_success_terminal[step].copy_(success.bool())
                self.v141_success_telemetry_present[step] = True
            if group_ids is not None:
                self.v141_group_ids[step].copy_(group_ids.long())
        super().process_env_step(obs, rewards, dones, extras)

    def _actor_ppo_transition_weights(self) -> torch.Tensor:
        return intervention_aware_ppo_weights(
            self.v141_intervention_mask, self.intervention_ppo_eta
        ).to(self.device)

    def relabel_teacher_transitions(self) -> dict[str, Any]:
        """Build the detached safe-action target without a teacher network."""
        metrics: dict[str, Any] = dict(
            OnlineSafePPO.relabel_pre_intervention_costs(self)
        )
        if not bool(self.filter_enabled.all()):
            missing = int((~self.filter_enabled).sum())
            raise RuntimeError(
                f"v141 training must execute the CBF on every transition ({missing} off)"
            )
        tensors = (
            self.safe_raw_actions,
            self.nominal_raw_actions,
            self.policy_actions,
        )
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise RuntimeError("v141 action telemetry contains non-finite values")
        nominal_policy_error = float(
            torch.amax(torch.abs(self.nominal_raw_actions - self.policy_actions))
        )
        if nominal_policy_error > 1.0e-6:
            raise RuntimeError(
                "v141 nominal raw action differs from the sampled policy action"
            )

        correction = self.safe_raw_actions - self.policy_actions
        correction_norm = torch.linalg.vector_norm(correction, dim=-1)
        intervention = correction_norm > self.v141_intervention_epsilon
        self.v141_intervention_mask.copy_(intervention)
        self.teacher_policy_actions.copy_(self.safe_raw_actions.detach())
        self.v30_teacher_targets.copy_(self.safe_raw_actions.detach())
        self.v30_correction_vectors.copy_(correction.detach())
        self.teacher_correction_norm.copy_(correction_norm)
        self.teacher_eligible.copy_(intervention)
        initial_weights = intervention.to(correction_norm.dtype) * torch.clamp(
            correction_norm / self.teacher_correction_scale, 0.0, 1.0
        )
        self.teacher_weights.copy_(initial_weights)
        intervention_count = int(intervention.sum())
        actual_disagreement = int((intervention != self.actual_cbf_intervened).sum())
        self.teacher_label_diagnostics = {
            "intervened": intervention,
            "magnitude_weight": initial_weights,
        }
        metrics.update(
            {
                "v141_method_id": METHOD_ID,
                "v141_intervention_count": intervention_count,
                "v141_would_intervene_fraction": float(intervention.float().mean()),
                "v141_mean_correction_norm": (
                    float(correction_norm[intervention].mean())
                    if intervention_count
                    else 0.0
                ),
                "v141_actual_intervention_mask_disagreement_count": (
                    actual_disagreement
                ),
                "v141_nominal_policy_max_abs_error": nominal_policy_error,
                "v141_safe_action_target_detached": True,
                "v141_teacher_network_present": False,
                "teacher_transition_count": float(intervention_count),
                "teacher_transition_fraction": float(intervention.float().mean()),
            }
        )
        self.last_update_metrics.update(metrics)
        return metrics

    def prepare_v141_advantage_weights(self) -> dict[str, float]:
        """Group-normalize PPO advantages and finish correction weights."""
        if self.v141_target_environment_mask is None:
            raise RuntimeError("v141 target/F1 environment mask was not configured")
        if not bool(self.v141_reward_telemetry_present.all()):
            raise RuntimeError("v141 task/CBF reward telemetry is incomplete")
        if not bool(self.v141_success_telemetry_present.all()):
            raise RuntimeError("v141 terminal-success telemetry is incomplete")
        expected_group_ids = (
            self.v141_target_environment_mask.long()
            .unsqueeze(0)
            .expand_as(self.v141_group_ids)
        )
        if not torch.equal(self.v141_group_ids, expected_group_ids):
            mismatch = int((self.v141_group_ids != expected_group_ids).sum())
            raise RuntimeError(
                f"v141 target/retention group routing differs on {mismatch} transitions"
            )
        advantages = self.storage.advantages.squeeze(-1)
        normalized, metrics = normalize_context_group_advantages(
            advantages, self.v141_target_environment_mask
        )
        self.storage.advantages.copy_(normalized.unsqueeze(-1))
        successful = successful_episode_transition_mask(
            self.teacher_episode_ids, self.v141_success_terminal
        )
        self.v141_successful_episode.copy_(successful)
        weights = correction_distillation_weights(
            self.v141_intervention_mask,
            self.teacher_correction_norm,
            normalized,
            correction_scale=self.teacher_correction_scale,
            mode=self.correction_weight_mode,
            successful_episode_mask=successful,
        )
        self.teacher_weights.copy_(weights)
        effective = weights * self.teacher_eligible.to(weights.dtype)
        before_distance, _ = weighted_action_errors(
            self.storage.distribution_params[0].flatten(0, 1),
            self.v30_teacher_targets.flatten(0, 1),
            self.teacher_eligible.flatten(),
            weights.flatten(),
        )
        metrics.update(
            {
                "v141_correction_weight_mode": self.correction_weight_mode,
                "v141_correction_weight_sum": float(effective.sum()),
                "v141_correction_weight_mean": float(effective.mean()),
                "v141_positive_advantage_fraction": float(
                    (normalized > 0.0).float().mean()
                ),
                "v141_successful_episode_transition_fraction": float(
                    successful.float().mean()
                ),
                "v141_success_terminal_count": float(
                    self.v141_success_terminal.sum()
                ),
                "v141_intervention_ppo_eta": self.intervention_ppo_eta,
                "v141_correction_loss_weight": self.teacher_distillation_weight,
                "v141_dual_reward_scale": self.v141_dual_reward_scale,
                "v141_target_environment_fraction": float(
                    self.v141_target_environment_mask.float().mean()
                ),
                "v141_mean_policy_to_safe_target_before_update": float(
                    before_distance
                ),
                "v141_mean_task_reward": float(self.v141_task_rewards.mean()),
                "v141_mean_scaled_cbf_reward": float(
                    self.v141_cbf_rewards.mean()
                ),
                "v141_reward_telemetry_fraction": float(
                    self.v141_reward_telemetry_present.float().mean()
                ),
                "v141_group_id_telemetry_fraction": float(
                    (self.v141_group_ids >= 0).float().mean()
                ),
                "v141_success_telemetry_fraction": float(
                    self.v141_success_telemetry_present.float().mean()
                ),
            }
        )
        self.last_update_metrics.update(metrics)
        return metrics

    def clear_cbf_rollout(self) -> None:
        super().clear_cbf_rollout()
        if not hasattr(self, "v141_intervention_mask"):
            return
        self.v141_intervention_mask.zero_()
        self.v141_success_terminal.zero_()
        self.v141_success_telemetry_present.zero_()
        self.v141_successful_episode.zero_()
        self.v141_group_ids.fill_(-1)
        self.v141_task_rewards.zero_()
        self.v141_cbf_rewards.zero_()
        self.v141_reward_telemetry_present.zero_()

    def save(self) -> dict[str, Any]:
        output = super().save()
        output.update(
            {
                "proximal_method_id": METHOD_ID,
                "v141_intervention_ppo_eta": self.intervention_ppo_eta,
                "v141_correction_weight_mode": self.correction_weight_mode,
                "v141_correction_loss_weight": self.teacher_distillation_weight,
                "v141_dual_reward_scale": self.v141_dual_reward_scale,
                "v141_target_fraction": self.v141_target_fraction,
                "v141_intervention_epsilon": self.v141_intervention_epsilon,
                "v141_runtime_filter_training": True,
                "v141_runtime_filter_deployment": False,
                "v141_teacher_network_present": False,
            }
        )
        return output


class InterventionAwareCbfDistillationRunner(CbfTeacherV30Runner):
    alg: InterventionAwareCbfDistillationPPO

    def load_recovery_checkpoint(
        self, path: str, map_location: str | None = None
    ) -> dict[str, Any]:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        if loaded.get("proximal_method_id") != METHOD_ID:
            raise ValueError("recovery checkpoint is not a v141 checkpoint")
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
        self.alg._raise_if_corrupt("v141 recovery checkpoint load")
        return {
            "source_iteration": int(loaded.get("iter", -1)),
            "actor_observation_dim": int(self.alg.actor.obs_dim),
            "critic_observation_dim": int(self.alg.critic.obs_dim),
            "source_optimizer_restored": True,
        }
