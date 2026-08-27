"""Pure tensor math for v30 residual corrective teaching."""

from __future__ import annotations

import math

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


def success_population_smooth_l1_loss(
    policy_mean: torch.Tensor,
    safe_action_target: torch.Tensor,
    successful_episode_transition: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Clone successful safe actions without renormalizing away success rate."""
    if policy_mean.ndim != 2 or safe_action_target.shape != policy_mean.shape:
        raise ValueError("v88 policy mean and safe target must share [B, A]")
    if successful_episode_transition.shape != policy_mean.shape[:1]:
        raise ValueError("v88 success mask must have shape [B]")
    if successful_episode_transition.dtype != torch.bool:
        raise TypeError("v88 success mask must be boolean")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("v88 Smooth-L1 beta must be finite and positive")
    per_transition = F.smooth_l1_loss(
        policy_mean,
        safe_action_target.detach(),
        beta=float(beta),
        reduction="none",
    ).mean(dim=-1)
    return masked_population_mean(
        per_transition,
        successful_episode_transition,
    )


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


def episode_balanced_outcome_advantage(
    episode_ids: torch.Tensor,
    successful_terminal: torch.Tensor,
    failed_terminal: torch.Tensor,
    filter_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Spread centered, episode-equal terminal outcomes over full episodes.

    Each filter-execution group receives +0.5 total mass across successful
    episodes and -0.5 across failed episodes.  Dividing each episode's mass by
    its stored length prevents long episodes from dominating.  The resulting
    transition credit is standardized separately inside each group so it can
    be combined at unit scale with group-normalized GAE.
    """
    if episode_ids.ndim != 2 or not (
        successful_terminal.shape == failed_terminal.shape == episode_ids.shape
    ):
        raise ValueError("v106 outcome tensors must share [T, N] shape")
    if episode_ids.dtype.is_floating_point:
        raise TypeError("v106 episode IDs must be integer tensors")
    if successful_terminal.dtype != torch.bool or failed_terminal.dtype != torch.bool:
        raise TypeError("v106 terminal outcomes must be boolean")
    if filter_mask.shape != episode_ids.shape[1:] or filter_mask.dtype != torch.bool:
        raise ValueError("v106 filter mask must be boolean with shape [N]")
    if not bool(filter_mask.any()) or bool(filter_mask.all()):
        raise ValueError("v106 requires non-empty filter-on and filter-off groups")
    if bool((successful_terminal & failed_terminal).any()):
        raise ValueError("v106 successful and failed terminals overlap")

    _, num_envs = episode_ids.shape
    environment_ids = torch.arange(
        num_envs, device=episode_ids.device, dtype=episode_ids.dtype
    ).expand_as(episode_ids)
    composite_ids = episode_ids * num_envs + environment_ids
    successful_transition = terminal_episode_transition_mask(
        episode_ids, successful_terminal
    )
    failed_transition = terminal_episode_transition_mask(
        episode_ids, failed_terminal
    )
    if bool((successful_transition & failed_transition).any()):
        raise RuntimeError("v106 episode outcome transition masks overlap")

    output = torch.zeros_like(episode_ids, dtype=torch.float32)
    metrics: dict[str, float] = {}
    for name, environment_mask in (
        ("filter_on", filter_mask),
        ("filter_off", ~filter_mask),
    ):
        group = environment_mask.unsqueeze(0).expand_as(episode_ids)
        success_ids = torch.unique(
            composite_ids[successful_terminal & group], sorted=True
        )
        failure_ids = torch.unique(
            composite_ids[failed_terminal & group], sorted=True
        )
        success_count = int(success_ids.numel())
        failure_count = int(failure_ids.numel())
        eligible = group & (successful_transition | failed_transition)
        eligible_count = int(eligible.sum())
        raw_sum = raw_mean = raw_std = normalized_mean = normalized_std = 0.0
        if success_count and failure_count:
            ids, inverse, lengths = torch.unique(
                composite_ids[eligible],
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            successful_episode = torch.isin(ids, success_ids)
            failed_episode = torch.isin(ids, failure_ids)
            if not bool((successful_episode | failed_episode).all()) or bool(
                (successful_episode & failed_episode).any()
            ):
                raise RuntimeError("v106 completed episode classification is invalid")
            episode_mass = torch.where(
                successful_episode,
                torch.full_like(ids, 0.5 / success_count, dtype=torch.float32),
                torch.full_like(ids, -0.5 / failure_count, dtype=torch.float32),
            )
            raw = episode_mass[inverse] / lengths[inverse].to(torch.float32)
            raw_sum = float(raw.sum())
            raw_mean = float(raw.mean())
            raw_std = float(raw.std(unbiased=False))
            normalized = raw / (raw.std(unbiased=False) + 1.0e-8)
            output[eligible] = normalized
            normalized_mean = float(normalized.mean())
            normalized_std = float(normalized.std(unbiased=False))
        metrics.update(
            {
                f"outcome_{name}_success_episode_count": float(success_count),
                f"outcome_{name}_failure_episode_count": float(failure_count),
                f"outcome_{name}_transition_count": float(eligible_count),
                f"outcome_{name}_raw_credit_sum": raw_sum,
                f"outcome_{name}_raw_credit_mean": raw_mean,
                f"outcome_{name}_raw_credit_std": raw_std,
                f"outcome_{name}_normalized_credit_mean": normalized_mean,
                f"outcome_{name}_normalized_credit_std": normalized_std,
            }
        )
    labeled = successful_transition | failed_transition
    metrics["outcome_labeled_transition_count"] = float(labeled.sum())
    metrics["outcome_labeled_transition_fraction"] = float(labeled.float().mean())
    metrics["outcome_unfinished_transition_count"] = float((~labeled).sum())
    return output, metrics


def disjoint_terminal_outcomes(
    done: torch.Tensor,
    fell: torch.Tensor,
    reached_top: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify simultaneous top/fall terminals once, with success priority.

    A foot can cross the top tolerance on the same simulator step that the
    body triggers ``fell_over``.  Deployment success is defined by reaching
    the top, so such a terminal must not label the same episode as both a
    successful and failed safety-teacher trajectory.
    """
    if not (done.shape == fell.shape == reached_top.shape):
        raise ValueError("v35 terminal outcome tensors must share one shape")
    if any(value.dtype != torch.bool for value in (done, fell, reached_top)):
        raise TypeError("v35 terminal outcome tensors must be boolean")
    successful = done & reached_top
    joint = successful & fell
    failed = done & fell & ~successful
    return failed, successful, joint


def outcome_gated_interventions(
    intervened: torch.Tensor,
    failed_episode_transition: torch.Tensor,
    successful_episode_transition: torch.Tensor,
    *,
    gate: str,
) -> torch.Tensor:
    """Restrict intervention labels to one complete-episode outcome."""
    if not (
        intervened.shape
        == failed_episode_transition.shape
        == successful_episode_transition.shape
    ) or intervened.ndim != 2:
        raise ValueError("v35 outcome-gate tensors must share [T, N] shape")
    if failed_episode_transition.dtype != torch.bool or (
        successful_episode_transition.dtype != torch.bool
    ):
        raise TypeError("v35 outcome transition masks must be boolean")
    if bool((failed_episode_transition & successful_episode_transition).any()):
        raise ValueError("v35 failed and successful episode masks overlap")
    eligible = intervened.bool()
    if gate == "none":
        return eligible
    if gate == "failed":
        return eligible & failed_episode_transition
    if gate == "successful":
        return eligible & successful_episode_transition
    raise ValueError(f"unknown v35 outcome gate {gate!r}")


def rotating_environment_filter_mask(
    num_envs: int,
    fraction: float,
    round_index: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a deterministic balanced per-round filter-execution mask."""
    if num_envs < 1 or round_index < 1:
        raise ValueError("v35 filter mask requires positive env and round counts")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("v35 runtime filter fraction must lie in [0, 1]")
    enabled_count = int(round(float(fraction) * num_envs))
    if fraction > 0.0:
        enabled_count = max(1, enabled_count)
    enabled_count = min(num_envs, enabled_count)
    if enabled_count == 0:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    offset = ((round_index - 1) * enabled_count) % num_envs
    environment_ids = torch.arange(num_envs, device=device)
    return ((environment_ids - offset) % num_envs) < enabled_count


def linear_filter_fraction_schedule(
    rounds: int,
    start_fraction: float,
    end_fraction: float,
) -> tuple[float, ...]:
    """Linearly anneal the executed-filter fraction across rollout rounds."""
    values = (float(start_fraction), float(end_fraction))
    if rounds < 2:
        raise ValueError("v56 filter annealing requires at least two rounds")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("v56 filter fractions must be finite and lie in [0, 1]")
    if start_fraction <= end_fraction:
        raise ValueError("v56 filter annealing must strictly decrease")
    denominator = rounds - 1
    return tuple(
        float(start_fraction)
        + (float(end_fraction) - float(start_fraction)) * index / denominator
        for index in range(rounds)
    )


def target_terrain_floor_schedule(
    rounds: int,
    num_rows: int,
    freeze_target_after_round: int | None,
) -> tuple[int, ...]:
    """Keep adaptive terrain early, then prevent retreat from the target row."""
    if rounds < 1 or num_rows < 2:
        raise ValueError("v60 terrain-floor schedule requires positive rounds and rows")
    if freeze_target_after_round is None:
        return (0,) * rounds
    if not 1 <= freeze_target_after_round <= rounds:
        raise ValueError("v60 target-freeze round must lie within training rounds")
    target_level = num_rows - 1
    return tuple(
        0 if round_index < freeze_target_after_round else target_level
        for round_index in range(1, rounds + 1)
    )


def filter_rescued_episode_mask(
    filter_on_success: torch.Tensor, filter_off_success: torch.Tensor
) -> torch.Tensor:
    """Select initial episodes that succeed only when the CBF executes."""
    if (
        filter_on_success.shape != filter_off_success.shape
        or filter_on_success.ndim != 1
    ):
        raise ValueError("v36 paired outcomes must share one-dimensional shape")
    if filter_on_success.dtype != torch.bool or filter_off_success.dtype != torch.bool:
        raise TypeError("v36 paired outcomes must be boolean")
    return filter_on_success & ~filter_off_success


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
