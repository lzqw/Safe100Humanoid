"""CBF-teacher action path and PPO for the independent v25 experiment.

Everything in this module is additive.  In particular, the byte-frozen v23
and v24 action, telemetry, and proximal-PPO sources remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch

from .actions import (
    StairCbfJointPositionAction,
    StairCbfJointPositionActionCfg,
)
from .mdp import online_safety_telemetry
from .proximal import (
    METHOD_ID as V23_METHOD_ID,
)
from .proximal import (
    CbfProximalPPO,
    CbfProximalPpoAlgorithmCfg,
    CbfProximalRefinementRunner,
    ProximalHardRollback,
    diagonal_gaussian_forward_kl,
)
from .teacher_math import (
    actor_coordinate_teacher_action,
    successful_teacher_labels,
    swing_leg_action_scale,
    toe_riser_kick_event,
    weighted_gaussian_teacher_loss,
)

METHOD_ID = "success-gated-cbf-action-teacher-v25"
SWING_JOINT_SUFFIXES = (
    "hip_pitch_joint",
    "knee_joint",
    "ankle_pitch_joint",
)


@dataclass(kw_only=True)
class SwingUnderResponseCbfActionCfg(StairCbfJointPositionActionCfg):
    """Fixed deployment mismatch on three joints of the active swing leg."""

    swing_underresponse_gain: float = 0.90

    def build(self, env) -> SwingUnderResponseCbfAction:
        return SwingUnderResponseCbfAction(self, env)


class SwingUnderResponseCbfAction(StairCbfJointPositionAction):
    """Apply phase-selective under-response and expose an invertible teacher."""

    cfg: SwingUnderResponseCbfActionCfg

    def __init__(self, cfg: SwingUnderResponseCbfActionCfg, env) -> None:
        if cfg.deployment_action_delay_steps != 0:
            raise ValueError("v25 forbids action delay in the learnable fixed shift")
        if cfg.deployment_contact_delay_steps != 0:
            raise ValueError("v25 keeps contact sensing nominal")
        if not math.isclose(cfg.deployment_action_gain, 1.0, abs_tol=1.0e-12):
            raise ValueError("v25 uses only the per-swing-leg under-response gain")
        if cfg.deployment_action_scale is not None:
            raise ValueError("v25 constructs its phase-selective scale internally")
        if cfg.deployment_action_bias is not None and any(
            not math.isclose(value, 0.0, abs_tol=1.0e-12)
            for value in cfg.deployment_action_bias
        ):
            raise ValueError("v25 keeps action bias nominal")
        super().__init__(cfg, env)
        self._left_swing_joint_indices = self._resolve_swing_joint_indices("left")
        self._right_swing_joint_indices = self._resolve_swing_joint_indices("right")
        self.applied_plant_scale = torch.ones_like(self.nominal_raw_action)
        self.teacher_policy_action = torch.zeros_like(self.nominal_raw_action)
        self.teacher_reprojected_action = torch.zeros_like(self.nominal_raw_action)
        self.teacher_reprojection_error = torch.zeros(self.num_envs, device=self.device)
        self.pre_step_stair_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.pre_filter_selected_foot = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.swing_selection_matches = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.toe_riser_kick = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.toe_riser_overlap = torch.zeros_like(self.toe_riser_kick)

    def _resolve_swing_joint_indices(self, side: str) -> tuple[int, int, int]:
        expected = tuple(f"{side}_{suffix}" for suffix in SWING_JOINT_SUFFIXES)
        indices = []
        for name in expected:
            matches = [
                index
                for index, target in enumerate(self._target_names)
                if target == name
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"v25 expected exactly one action target {name!r}, got {matches}"
                )
            indices.append(matches[0])
        return tuple(indices)  # type: ignore[return-value]

    def _current_swing_foot(self) -> torch.Tensor:
        found = self._contact_sensor.data.found
        if found is None:
            raise RuntimeError("v25 swing-leg shift requires foot-contact state")
        contact = found > 0
        if contact.ndim > 2:
            contact = contact.any(dim=tuple(range(2, contact.ndim)))
        if contact.ndim != 2 or contact.shape[1] != 2:
            raise RuntimeError("v25 foot-contact state must have shape [N, 2]")
        in_air = ~contact
        air_time = self._contact_sensor.data.current_air_time
        scores = (
            in_air.float()
            if air_time is None
            else torch.where(in_air, air_time, torch.full_like(air_time, -1.0))
        )
        index = scores.argmax(dim=1)
        return torch.where(in_air.any(dim=1), index, torch.full_like(index, -1))

    def _current_stair_index(self) -> torch.Tensor:
        terrain = self._env.scene.terrain
        if terrain is None:
            raise RuntimeError("v25 requires exact stair metadata")
        edge_x = self._edge_x[terrain.terrain_levels, terrain.terrain_types]
        root_x = self._entity.data.root_link_pos_w[:, 0:1]
        return torch.sum(root_x >= edge_x, dim=1)

    def process_actions(self, actions: torch.Tensor) -> None:
        selected_foot = self._current_swing_foot()
        dynamic_scale = swing_leg_action_scale(
            selected_foot,
            action_dim=self.action_dim,
            left_joint_indices=self._left_swing_joint_indices,
            right_joint_indices=self._right_swing_joint_indices,
            gain=self.cfg.swing_underresponse_gain,
            dtype=actions.dtype,
        )
        self.pre_step_stair_index[:] = self._current_stair_index()
        self.pre_filter_selected_foot[:] = selected_foot
        # The frozen parent implements the plant transform and CBF projection.
        # A [N, A] tensor broadcasts exactly like its historical [A] scale.
        self._deployment_action_scale = dynamic_scale
        super().process_actions(actions)
        self.applied_plant_scale[:] = dynamic_scale
        self.swing_selection_matches[:] = self.selected_foot == selected_foot

        teacher, reprojected = actor_coordinate_teacher_action(
            self.safe_raw_action,
            dynamic_scale,
            plant_gain=self.cfg.deployment_action_gain,
            plant_bias=self._deployment_action_bias,
        )
        self.teacher_policy_action[:] = teacher
        self.teacher_reprojected_action[:] = reprojected
        self.teacher_reprojection_error[:] = torch.amax(
            torch.abs(reprojected - self.safe_raw_action), dim=-1
        )
        event, overlap = toe_riser_kick_event(
            self.h, self.geometric_active, self.toe_riser_overlap
        )
        self.toe_riser_kick[:] = event
        self.toe_riser_overlap[:] = overlap

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        # ``JointPositionAction`` may invoke reset while the subclass is still
        # being constructed, hence the explicit existence guard.
        if not hasattr(self, "teacher_policy_action"):
            return
        self.applied_plant_scale[env_ids] = 1.0
        self.teacher_policy_action[env_ids] = 0.0
        self.teacher_reprojected_action[env_ids] = 0.0
        self.teacher_reprojection_error[env_ids] = 0.0
        self.pre_step_stair_index[env_ids] = 0
        self.pre_filter_selected_foot[env_ids] = -1
        self.swing_selection_matches[env_ids] = True
        self.toe_riser_kick[env_ids] = False
        self.toe_riser_overlap[env_ids] = False


def v25_online_safety_telemetry(
    env,
    action_name: str = "joint_pos",
    termination_name: str = "fell_over",
) -> torch.Tensor:
    """Add v25 transition tensors before MJLab auto-resets finished worlds."""
    zeros = online_safety_telemetry(
        env, action_name=action_name, termination_name=termination_name
    )
    term = env.action_manager.get_term(action_name)
    if not isinstance(term, SwingUnderResponseCbfAction):
        raise TypeError("v25 telemetry requires SwingUnderResponseCbfAction")
    env.extras["v25_teacher_policy_action"] = (
        term.teacher_policy_action.detach().clone()
    )
    env.extras["v25_applied_plant_scale"] = term.applied_plant_scale.detach().clone()
    env.extras["v25_pre_step_stair_index"] = term.pre_step_stair_index.detach().clone()
    env.extras["v25_teacher_reprojection_error"] = (
        term.teacher_reprojection_error.detach().clone()
    )
    env.extras["v25_swing_selection_matches"] = (
        term.swing_selection_matches.detach().clone()
    )
    env.extras["v25_toe_riser_kick"] = term.toe_riser_kick.detach().clone()
    env.extras["v25_toe_riser_overlap"] = term.toe_riser_overlap.detach().clone()
    return zeros


def configure_v25_swing_underresponse(
    env_cfg,
    *,
    gain: float,
    runtime_filter: bool,
) -> dict[str, Any]:
    """Install the one-axis fixed v25 shift without changing observations."""
    original = env_cfg.actions["joint_pos"]
    if not isinstance(original, StairCbfJointPositionActionCfg):
        raise TypeError("v25 requires the historical stair CBF action config")
    kwargs = {
        field.name: getattr(original, field.name)
        for field in fields(original)
        if field.init
    }
    kwargs.update(
        enabled=bool(runtime_filter),
        deployment_action_gain=1.0,
        deployment_action_scale=None,
        deployment_action_bias=None,
        deployment_action_delay_steps=0,
        deployment_contact_delay_steps=0,
        deployment_contact_phase_offset=0.0,
        swing_underresponse_gain=float(gain),
    )
    env_cfg.actions["joint_pos"] = SwingUnderResponseCbfActionCfg(**kwargs)
    env_cfg.rewards.pop("specialist_failure_signal", None)
    telemetry = env_cfg.rewards["online_safety_telemetry"]
    telemetry.func = v25_online_safety_telemetry
    telemetry.params = {
        "action_name": "joint_pos",
        "termination_name": "fell_over",
    }
    return {
        "shift": "fixed_phase_selective_swing_leg_underresponse",
        "swing_underresponse_gain": float(gain),
        "affected_joints_per_swing_leg": list(SWING_JOINT_SUFFIXES),
        "stance_leg_scale": 1.0,
        "all_other_action_scales": 1.0,
        "runtime_filter": bool(runtime_filter),
        "terrain_geometry_changed": False,
        "friction_changed": False,
        "command_changed": False,
        "controller_changed": False,
        "actor_observation_fields_added": 0,
        "cbf_geometry_exact": True,
    }


@dataclass
class CbfTeacherPpoAlgorithmCfg(CbfProximalPpoAlgorithmCfg):
    """v23 proximal PPO plus the single fixed v25 teacher objective."""

    class_name: str = "src.tasks.stairs_cbf.teacher:CbfTeacherPPO"
    teacher_distillation_weight: float = 0.1
    teacher_success_horizon: int = 50
    teacher_correction_scale: float = 0.05


class CbfTeacherPPO(CbfProximalPPO):
    """Integrate success-gated CBF Gaussian NLL into each PPO actor step."""

    def __init__(
        self,
        *args,
        teacher_distillation_weight: float = 0.1,
        teacher_success_horizon: int = 50,
        teacher_correction_scale: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_distillation_weight = float(teacher_distillation_weight)
        self.teacher_success_horizon = int(teacher_success_horizon)
        self.teacher_correction_scale = float(teacher_correction_scale)
        if not math.isclose(
            self.teacher_distillation_weight, 0.1, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("v25 fixes the teacher coefficient at lambda_D=0.1")
        if self.teacher_success_horizon != 50:
            raise ValueError("v25 fixes H=50 control steps (1.0 s)")
        if not math.isclose(
            self.teacher_correction_scale, 0.05, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("v25 fixes the teacher correction scale at 0.05")
        t = self.storage.num_transitions_per_env
        n = self.storage.num_envs
        action_dim = self.storage.actions.shape[-1]
        self.teacher_policy_actions = torch.zeros(t, n, action_dim, device=self.device)
        self.teacher_telemetry_present = torch.zeros(
            t, n, dtype=torch.bool, device=self.device
        )
        self.teacher_eligible = torch.zeros_like(self.teacher_telemetry_present)
        self.teacher_weights = torch.zeros(t, n, device=self.device)
        self.teacher_correction_norm = torch.zeros_like(self.teacher_weights)
        self.teacher_pre_step_stair_indices = torch.zeros(
            t, n, dtype=torch.long, device=self.device
        )
        self.teacher_reprojection_errors = torch.zeros_like(self.teacher_weights)
        self.teacher_swing_selection_matches = torch.ones_like(
            self.teacher_telemetry_present
        )
        self.toe_riser_kick_events = torch.zeros_like(self.teacher_telemetry_present)
        self.toe_riser_overlaps = torch.zeros_like(self.teacher_telemetry_present)
        self.teacher_label_diagnostics: dict[str, torch.Tensor] = {}

    def process_env_step(
        self,
        obs,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        step = self.storage.step
        if step < self.storage.num_transitions_per_env:
            teacher = extras.get("v25_teacher_policy_action")
            pre_index = extras.get("v25_pre_step_stair_index")
            reprojection = extras.get("v25_teacher_reprojection_error")
            selection_match = extras.get("v25_swing_selection_matches")
            kick = extras.get("v25_toe_riser_kick")
            overlap = extras.get("v25_toe_riser_overlap")
            if teacher is not None:
                self.teacher_policy_actions[step].copy_(teacher)
                self.teacher_telemetry_present[step] = True
            if pre_index is not None:
                self.teacher_pre_step_stair_indices[step].copy_(pre_index)
            if reprojection is not None:
                self.teacher_reprojection_errors[step].copy_(reprojection)
            if selection_match is not None:
                self.teacher_swing_selection_matches[step].copy_(selection_match)
            if kick is not None:
                self.toe_riser_kick_events[step].copy_(kick)
            if overlap is not None:
                self.toe_riser_overlaps[step].copy_(overlap)
        super().process_env_step(obs, rewards, dones, extras)

    def relabel_teacher_transitions(self) -> dict[str, float]:
        """Run the frozen action audit, then apply the delayed success gate."""
        metrics = super().relabel_pre_intervention_costs()
        if not bool(self.teacher_telemetry_present.all()):
            missing = int((~self.teacher_telemetry_present).sum())
            raise RuntimeError(
                f"v25 teacher telemetry missing on {missing} transitions"
            )
        if not bool(self.teacher_swing_selection_matches.all()):
            mismatch = int((~self.teacher_swing_selection_matches).sum())
            raise RuntimeError(
                f"v25 swing-foot selections diverged on {mismatch} transitions"
            )
        maximum_reprojection_error = float(self.teacher_reprojection_errors.max())
        if maximum_reprojection_error > 1.0e-6:
            raise RuntimeError(
                "v25 actor-coordinate teacher does not reproject to the safe plant action: "
                f"{maximum_reprojection_error}"
            )
        correction_norm = torch.linalg.vector_norm(
            self.teacher_policy_actions - self.policy_actions, dim=-1
        )
        self.teacher_correction_norm.copy_(correction_norm)
        eligible, weights, diagnostics = successful_teacher_labels(
            self.actual_cbf_intervened,
            correction_norm,
            self.teacher_pre_step_stair_indices,
            self.stair_indices,
            self.fall_events,
            self.storage.dones.squeeze(-1),
            horizon=self.teacher_success_horizon,
            correction_scale=self.teacher_correction_scale,
        )
        self.teacher_eligible.copy_(eligible)
        self.teacher_weights.copy_(weights)
        self.teacher_label_diagnostics = diagnostics
        valid_count = int(eligible.sum())
        metrics.update(
            teacher_transition_count=float(valid_count),
            teacher_transition_fraction=float(eligible.float().mean()),
            teacher_weight_mean_over_valid=(
                float(weights[eligible].mean()) if valid_count else 0.0
            ),
            teacher_crossed_gate_count=float(
                (
                    self.actual_cbf_intervened & diagnostics["crossed_within_horizon"]
                ).sum()
            ),
            teacher_no_fall_gate_count=float(
                (
                    self.actual_cbf_intervened & diagnostics["no_fall_within_horizon"]
                ).sum()
            ),
            teacher_actor_coordinate_correction_mean=float(correction_norm.mean()),
            teacher_reprojection_max_abs_error=maximum_reprojection_error,
            swing_selection_mismatch_count=0.0,
            toe_riser_kick_event_count=float(self.toe_riser_kick_events.sum()),
            toe_riser_overlap_fraction=float(self.toe_riser_overlaps.float().mean()),
        )
        self.last_update_metrics.update(metrics)
        return metrics

    def update(self) -> dict[str, Any]:
        """Run raw-action PPO, moving KL, and teacher NLL in one Adam step."""
        if self.round_reference_actor is None:
            raise RuntimeError("round-start moving reference was not frozen")
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("v25 PPO requires one complete on-policy rollout")
        if (
            self.rnd
            or self.symmetry
            or self.actor.is_recurrent
            or self.critic.is_recurrent
        ):
            raise RuntimeError("v25 supports feed-forward PPO without RND/symmetry")

        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1).clone()
        old_log_prob = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1).clone()
        reference_params = tuple(
            value.flatten(0, 1).clone().detach()
            for value in self.storage.distribution_params
        )
        advantages = self.storage.advantages.flatten().detach()
        teacher_actions = self.teacher_policy_actions.flatten(0, 1).detach()
        teacher_eligible = self.teacher_eligible.flatten().detach()
        teacher_weights = self.teacher_weights.flatten().detach()
        if not bool(torch.isfinite(advantages).all()):
            raise ProximalHardRollback("non-finite whole-batch advantages")

        with torch.inference_mode():
            self.round_reference_actor(observations, stochastic_output=True)
            frozen_params = tuple(
                value.detach()
                for value in self.round_reference_actor.output_distribution_params
            )
            frozen_log_prob = self.round_reference_actor.get_output_log_prob(actions)
            from .online import (
                validate_behavior_distribution_params,
                validate_behavior_log_prob,
            )

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
        teacher_loss_total = 0.0
        entropy_total = 0.0
        actor_updates = 0
        actor_epochs_completed = 0
        actor_gradient_norm_max = 0.0
        teacher_minibatches_with_signal = 0
        teacher_minibatches_without_signal = 0
        teacher_samples_seen = 0
        epoch_moving_kl: list[float] = []
        target_kl_early_stopped = False
        batch_size = actions.shape[0]
        if batch_size % self.num_mini_batches:
            raise RuntimeError("v25 rollout must divide exactly into four minibatches")
        mini_batch_size = batch_size // self.num_mini_batches

        for epoch in range(self.num_learning_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for mini_batch in range(self.num_mini_batches):
                start = mini_batch * mini_batch_size
                indices = permutation[start : start + mini_batch_size]
                batch_observations = observations[indices]
                batch_actions = actions[indices]
                self.actor(batch_observations, stochastic_output=True)
                new_log_prob = self.actor.get_output_log_prob(batch_actions)
                current_params = tuple(self.actor.output_distribution_params)
                ratio = torch.exp(new_log_prob - old_log_prob[indices])
                advantage = advantages[indices]
                surrogate = -advantage * ratio
                surrogate_clipped = -advantage * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                ppo_loss = torch.maximum(surrogate, surrogate_clipped).mean()
                moving_kl_loss = diagonal_gaussian_forward_kl(
                    current_params,
                    tuple(value[indices].detach() for value in reference_params),
                ).mean()
                batch_teacher_eligible = teacher_eligible[indices]
                teacher_loss = weighted_gaussian_teacher_loss(
                    current_params[0],
                    current_params[1],
                    teacher_actions[indices],
                    batch_teacher_eligible,
                    teacher_weights[indices],
                )
                valid_in_batch = int(batch_teacher_eligible.sum())
                teacher_samples_seen += valid_in_batch
                if valid_in_batch:
                    teacher_minibatches_with_signal += 1
                else:
                    teacher_minibatches_without_signal += 1
                entropy = self.actor.output_entropy.mean()
                actor_loss = (
                    ppo_loss
                    + self.moving_kl_beta * moving_kl_loss
                    + self.teacher_distillation_weight * teacher_loss
                    - self.entropy_coef * entropy
                )
                if not bool(torch.isfinite(actor_loss)):
                    raise ProximalHardRollback("non-finite v25 actor loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_parameters = list(self.actor.mlp.parameters())
                if not self._finite_gradients(actor_parameters):
                    raise ProximalHardRollback("non-finite v25 actor gradient")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    actor_parameters, self.max_grad_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise ProximalHardRollback("non-finite v25 actor gradient norm")
                self.actor_optimizer.step()
                self._raise_if_corrupt(f"v25 actor epoch {epoch + 1}")
                actor_updates += 1
                actor_loss_total += float(actor_loss.detach())
                ppo_loss_total += float(ppo_loss.detach())
                moving_kl_loss_total += float(moving_kl_loss.detach())
                teacher_loss_total += float(teacher_loss.detach())
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
                    raise ProximalHardRollback("non-finite v25 value loss")
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_parameters = list(self.critic.parameters())
                if not self._finite_gradients(critic_parameters):
                    raise ProximalHardRollback("non-finite v25 critic gradient")
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, self.max_grad_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise ProximalHardRollback("non-finite v25 critic gradient norm")
                self.critic_optimizer.step()
                self._raise_if_corrupt(f"v25 critic epoch {epoch + 1}")
                critic_updates += 1
                critic_loss_total += float(value_loss.detach())
                critic_gradient_norm_max = max(
                    critic_gradient_norm_max, float(gradient_norm)
                )

        if actor_updates < 1 or critic_updates < 1:
            raise ProximalHardRollback("v25 update completed no optimizer steps")
        self.clamp_online_std()
        final_policy = self._whole_batch_policy_metrics(
            observations, actions, old_log_prob, reference_params
        )
        if final_policy["moving_forward_kl"] > self.hard_kl_ceiling:
            raise ProximalHardRollback(
                "moving forward KL exceeded hard ceiling after value fit", final_policy
            )
        self._raise_if_corrupt("complete v25 update")

        rollout_metrics = dict(self.last_update_metrics)
        result: dict[str, Any] = {
            "actor_loss": actor_loss_total / actor_updates,
            "surrogate": ppo_loss_total / actor_updates,
            "moving_forward_kl_loss": moving_kl_loss_total / actor_updates,
            "teacher_loss": teacher_loss_total / actor_updates,
            "teacher_distillation_weight": self.teacher_distillation_weight,
            "teacher_success_horizon": self.teacher_success_horizon,
            "teacher_correction_scale": self.teacher_correction_scale,
            "teacher_minibatches_with_signal": teacher_minibatches_with_signal,
            "teacher_minibatches_without_signal": teacher_minibatches_without_signal,
            "teacher_samples_seen_across_epochs": teacher_samples_seen,
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
        result["mean_kl"] = result["moving_forward_kl"]
        self.storage.clear()
        self.clear_cbf_rollout()
        self.last_update_metrics = {}
        return result

    def clear_cbf_rollout(self) -> None:
        super().clear_cbf_rollout()
        if not hasattr(self, "teacher_policy_actions"):
            return
        self.teacher_policy_actions.zero_()
        self.teacher_telemetry_present.zero_()
        self.teacher_eligible.zero_()
        self.teacher_weights.zero_()
        self.teacher_correction_norm.zero_()
        self.teacher_pre_step_stair_indices.zero_()
        self.teacher_reprojection_errors.zero_()
        self.teacher_swing_selection_matches.fill_(True)
        self.toe_riser_kick_events.zero_()
        self.toe_riser_overlaps.zero_()
        self.teacher_label_diagnostics = {}

    def save(self) -> dict[str, Any]:
        output = super().save()
        if output.get("proximal_method_id") != V23_METHOD_ID:
            raise RuntimeError("unexpected parent checkpoint identity")
        output["proximal_method_id"] = METHOD_ID
        output["teacher_distillation_weight"] = self.teacher_distillation_weight
        output["teacher_success_horizon"] = self.teacher_success_horizon
        output["teacher_correction_scale"] = self.teacher_correction_scale
        return output


class CbfTeacherRefinementRunner(CbfProximalRefinementRunner):
    """v25 runner using the exact v23 warm start and transactional snapshots."""

    alg: CbfTeacherPPO

    def load_recovery_checkpoint(
        self, path: str, map_location: str | None = None
    ) -> dict[str, Any]:
        loaded = torch.load(path, map_location=map_location, weights_only=False)
        if loaded.get("proximal_method_id") != METHOD_ID:
            raise ValueError("recovery checkpoint is not a v25 teacher checkpoint")
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
        self.alg._raise_if_corrupt("v25 recovery checkpoint load")
        return {
            "source_iteration": int(loaded.get("iter", -1)),
            "recovered_actor_optimizer": True,
            "recovered_critic_optimizer": True,
            "recovered_round_reference": reference is not None,
        }
