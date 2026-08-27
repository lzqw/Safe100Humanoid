"""Pure tensor math for v30 residual corrective teaching."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_population_mean(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Zero excluded losses and average over the original transition population."""
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("v35 masked actor terms must share one-dimensional shape")
    if mask.dtype != torch.bool:
        raise TypeError("v35 actor transition mask must be boolean")
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("v35 masked actor objective contains non-finite values")
    return (values * mask.to(values.dtype)).mean()


def terminal_episode_transition_mask(
    episode_ids: torch.Tensor, terminal_events: torch.Tensor
) -> torch.Tensor:
    """Mark every stored transition from episodes ending in ``terminal_events``."""
    if episode_ids.ndim != 2 or terminal_events.shape != episode_ids.shape:
        raise ValueError("v35 episode ids and terminal events must share [T, N]")
    if terminal_events.dtype != torch.bool:
        raise TypeError("v35 terminal episode events must be boolean")
    _, num_envs = episode_ids.shape
    environment_ids = torch.arange(
        num_envs, device=episode_ids.device, dtype=episode_ids.dtype
    ).expand_as(episode_ids)
    composite_ids = episode_ids * num_envs + environment_ids
    terminal_ids = composite_ids[terminal_events]
    if terminal_ids.numel() == 0:
        return torch.zeros_like(terminal_events)
    return torch.isin(composite_ids, terminal_ids)


def residual_teacher_target(
    reference_mean: torch.Tensor,
    safe_actor_action: torch.Tensor,
    raw_policy_action: torch.Tensor,
    *,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``mu_k + eta * (a_safe - a_policy)`` with stop-gradient."""
    if not (reference_mean.shape == safe_actor_action.shape == raw_policy_action.shape):
        raise ValueError("v30 residual target tensors must have equal shape")
    if reference_mean.ndim < 2:
        raise ValueError("v30 residual target needs an action dimension")
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("v30 residual eta must lie in [0, 1]")
    if not bool(
        torch.isfinite(reference_mean).all()
        and torch.isfinite(safe_actor_action).all()
        and torch.isfinite(raw_policy_action).all()
    ):
        raise RuntimeError("v30 residual target contains non-finite values")
    correction = safe_actor_action - raw_policy_action
    target = reference_mean + float(eta) * correction
    return target.detach(), correction.detach()


def intervention_teacher_weights(
    intervened: torch.Tensor,
    correction_norm: torch.Tensor,
    *,
    correction_scale: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all-intervention eligibility and clipped magnitude weights."""
    if intervened.shape != correction_norm.shape or intervened.ndim != 2:
        raise ValueError("v30 intervention tensors must share [T, N] shape")
    if not 0.0 < float(correction_scale):
        raise ValueError("v30 correction scale must be positive")
    if not bool(torch.isfinite(correction_norm).all()):
        raise RuntimeError("v30 correction norm contains non-finite values")
    if bool((correction_norm < 0.0).any()):
        raise ValueError("v30 correction norm must be non-negative")
    eligible = intervened.bool()
    magnitude = torch.clamp(correction_norm / float(correction_scale), 0.0, 1.0)
    return eligible, eligible.to(correction_norm.dtype) * magnitude


def weighted_smooth_l1_teacher_loss(
    policy_mean: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    weights: torch.Tensor,
    *,
    beta: float = 0.05,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Weighted per-action-mean Smooth-L1 with an exact empty-batch zero."""
    if policy_mean.shape != target.shape or policy_mean.ndim != 2:
        raise ValueError("v30 Smooth-L1 tensors must share [B, A] shape")
    if eligible.shape != policy_mean.shape[:-1] or weights.shape != eligible.shape:
        raise ValueError("v30 Smooth-L1 masks and weights must have shape [B]")
    if not 0.0 < float(beta) or not 0.0 < float(epsilon):
        raise ValueError("v30 Smooth-L1 beta and epsilon must be positive")
    if not bool(
        torch.isfinite(policy_mean).all()
        and torch.isfinite(target).all()
        and torch.isfinite(weights).all()
    ):
        raise RuntimeError("v30 Smooth-L1 objective contains non-finite values")
    if bool((weights < 0.0).any()):
        raise ValueError("v30 teacher weights must be non-negative")
    effective = weights * eligible.bool().to(weights.dtype)
    weight_sum = effective.sum()
    if not bool(weight_sum > 0.0):
        return policy_mean.sum() * 0.0
    per_action = F.smooth_l1_loss(
        policy_mean,
        target.detach(),
        reduction="none",
        beta=float(beta),
    )
    per_transition = per_action.mean(dim=-1)
    return (effective * per_transition).sum() / (weight_sum + float(epsilon))


def weighted_action_errors(
    policy_mean: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted L2 distance and per-action absolute errors."""
    if policy_mean.shape != target.shape or policy_mean.ndim != 2:
        raise ValueError("v30 error tensors must share [B, A] shape")
    effective = weights * eligible.bool().to(weights.dtype)
    weight_sum = effective.sum()
    if not bool(weight_sum > 0.0):
        zero = policy_mean.sum() * 0.0
        return zero, torch.zeros(
            policy_mean.shape[-1], device=policy_mean.device, dtype=policy_mean.dtype
        )
    delta = policy_mean - target.detach()
    l2 = (effective * torch.linalg.vector_norm(delta, dim=-1)).sum() / (
        weight_sum + float(epsilon)
    )
    per_action = (effective.unsqueeze(-1) * delta.abs()).sum(dim=0) / (
        weight_sum + float(epsilon)
    )
    return l2, per_action
