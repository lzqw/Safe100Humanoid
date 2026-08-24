"""v29 vectorized local-success CBF teacher and fixed PPO data path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .teacher import (
    CbfTeacherPPO,
    CbfTeacherPpoAlgorithmCfg,
    CbfTeacherRefinementRunner,
)
from .teacher_math import (
    vectorized_successful_teacher_labels_v29,
    weighted_gaussian_teacher_loss_v29,
)


METHOD_ID = "vectorized-local-success-cbf-action-teacher-v29"


@dataclass
class CbfTeacherV29PpoAlgorithmCfg(CbfTeacherPpoAlgorithmCfg):
    """The fixed v29 objective with no auxiliary actor, critic, or replay path."""

    class_name: str = "src.tasks.stairs_cbf.teacher_v29:CbfTeacherV29PPO"


class CbfTeacherV29PPO(CbfTeacherPPO):
    """Add explicit episode identities and v29's weight-normalized teacher NLL."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        t = self.storage.num_transitions_per_env
        n = self.storage.num_envs
        self.teacher_episode_ids = torch.zeros(
            t, n, dtype=torch.long, device=self.device
        )
        self.teacher_recovery_takeovers = torch.zeros(
            t, n, dtype=torch.bool, device=self.device
        )
        self.teacher_emergency_terminations = torch.zeros_like(
            self.teacher_recovery_takeovers
        )
        self._current_episode_ids = torch.zeros(
            n, dtype=torch.long, device=self.device
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
            self.teacher_episode_ids[step].copy_(self._current_episode_ids)
            recovery = extras.get("online_recovery_takeover")
            emergency = extras.get("online_emergency_termination")
            if recovery is not None:
                self.teacher_recovery_takeovers[step].copy_(recovery.bool())
            if emergency is not None:
                self.teacher_emergency_terminations[step].copy_(emergency.bool())
        super().process_env_step(obs, rewards, dones, extras)
        self._current_episode_ids.add_(dones.bool().long())

    def _compute_teacher_labels(
        self, correction_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return vectorized_successful_teacher_labels_v29(
            self.actual_cbf_intervened,
            correction_norm,
            self.teacher_pre_step_stair_indices,
            self.stair_indices,
            self.teacher_episode_ids,
            self.fall_events,
            self.teacher_recovery_takeovers,
            self.teacher_emergency_terminations,
            self.storage.dones.squeeze(-1),
            horizon=self.teacher_success_horizon,
            correction_scale=self.teacher_correction_scale,
        )

    def _compute_teacher_loss(
        self,
        policy_mean: torch.Tensor,
        policy_std: torch.Tensor,
        teacher_action: torch.Tensor,
        eligible: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_gaussian_teacher_loss_v29(
            policy_mean,
            policy_std,
            teacher_action,
            eligible,
            weights,
            epsilon=1.0e-8,
        )

    def relabel_teacher_transitions(self) -> dict[str, float]:
        metrics = super().relabel_teacher_transitions()
        dones = self.storage.dones.squeeze(-1).bool()
        if self.teacher_episode_ids.shape[0] > 1:
            expected = self.teacher_episode_ids[:-1] + dones[:-1].long()
            identity_error = torch.abs(self.teacher_episode_ids[1:] - expected)
            identity_max_error = int(identity_error.max())
        else:
            identity_max_error = 0
        if identity_max_error:
            raise RuntimeError(
                "v29 episode identity routing crosses an episode boundary: "
                f"{identity_max_error}"
            )
        intervention_count = int(self.actual_cbf_intervened.sum())
        eligible_count = int(self.teacher_eligible.sum())
        weighted_count = float(self.teacher_weights.sum())
        intervened_distances = self.teacher_correction_norm[
            self.actual_cbf_intervened
        ]
        mean_intervened_distance = (
            float(intervened_distances.mean()) if intervention_count else 0.0
        )
        mean_weighted_distance = (
            float(
                (
                    self.teacher_weights * self.teacher_correction_norm
                ).sum()
                / self.teacher_weights.sum().clamp_min(1.0e-8)
            )
            if weighted_count > 0.0
            else 0.0
        )
        metrics.update(
            {
                "teacher_eligible_count": float(eligible_count),
                "teacher_eligible_fraction_among_interventions": (
                    eligible_count / max(1, intervention_count)
                ),
                "teacher_weighted_count": weighted_count,
                "mean_policy_to_teacher_action_distance": (
                    mean_intervened_distance
                ),
                "mean_weighted_policy_to_teacher_action_distance": (
                    mean_weighted_distance
                ),
                "episode_identity_transition_max_abs_error": float(
                    identity_max_error
                ),
                "stored_episode_identity_count": float(
                    torch.unique(self.teacher_episode_ids).numel()
                ),
                "recovery_takeover_termination_count": float(
                    self.teacher_recovery_takeovers.sum()
                ),
                "emergency_termination_count": float(
                    self.teacher_emergency_terminations.sum()
                ),
                "teacher_label_vectorized_horizon_loops": float(
                    min(
                        self.teacher_success_horizon,
                        self.storage.num_transitions_per_env,
                    )
                ),
                "teacher_loss_weight_normalization_epsilon": 1.0e-8,
            }
        )
        self.last_update_metrics.update(metrics)
        return metrics

    def clear_cbf_rollout(self) -> None:
        super().clear_cbf_rollout()
        if not hasattr(self, "teacher_episode_ids"):
            return
        self.teacher_episode_ids.zero_()
        self.teacher_recovery_takeovers.zero_()
        self.teacher_emergency_terminations.zero_()
        self._current_episode_ids.zero_()

    def save(self) -> dict[str, Any]:
        output = super().save()
        output["proximal_method_id"] = METHOD_ID
        output["teacher_label_implementation"] = "v29_vectorized_future_offsets"
        output["teacher_loss_normalization"] = "sum_weights_plus_1e-8"
        return output


class CbfTeacherV29Runner(CbfTeacherRefinementRunner):
    """v29 runner with exact base warm start and v29-only recovery loading."""

    alg: CbfTeacherV29PPO

    def load_recovery_checkpoint(
        self, path: str, map_location: str | None = None
    ) -> dict[str, Any]:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        if loaded.get("proximal_method_id") != METHOD_ID:
            raise ValueError("recovery checkpoint is not a v29 teacher checkpoint")
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
        self.alg._raise_if_corrupt("v29 recovery checkpoint load")
        return {
            "source_iteration": int(loaded.get("iter", -1)),
            "recovered_actor_optimizer": True,
            "recovered_critic_optimizer": True,
            "recovered_round_reference": reference is not None,
        }
