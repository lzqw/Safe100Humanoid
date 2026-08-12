"""Pure tensor operations for success-gated CBF action teaching.

This module deliberately has no MJLab or RSL-RL dependency.  The v25 action
term, PPO implementation, evaluator, and regression tests all share these
definitions so that the plant inverse and delayed outcome labels cannot drift
between execution and audit code.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def swing_leg_action_scale(
    selected_foot: torch.Tensor,
    *,
    action_dim: int,
    left_joint_indices: Sequence[int],
    right_joint_indices: Sequence[int],
    gain: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a per-environment plant scale for the currently swinging leg.

    ``selected_foot`` uses ``0`` for left, ``1`` for right, and ``-1`` when
    neither foot is in swing.  Only hip-pitch, knee, and ankle-pitch columns
    supplied by the caller are attenuated; every other action column remains
    exactly one.
    """
    if selected_foot.ndim != 1:
        raise ValueError("selected_foot must have shape [N]")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if not 0.5 <= float(gain) < 1.0:
        raise ValueError("swing under-response gain must lie in [0.5, 1.0)")
    left = tuple(int(index) for index in left_joint_indices)
    right = tuple(int(index) for index in right_joint_indices)
    if len(left) != 3 or len(right) != 3:
        raise ValueError("each swing leg must resolve exactly three pitch joints")
    if len(set(left + right)) != 6:
        raise ValueError("left/right swing-joint indices must be distinct")
    if any(index < 0 or index >= action_dim for index in left + right):
        raise ValueError("swing-joint index lies outside the action dimension")
    if bool(((selected_foot < -1) | (selected_foot > 1)).any()):
        raise ValueError("selected_foot contains an unsupported index")

    scale = torch.ones(
        selected_foot.shape[0],
        action_dim,
        device=selected_foot.device,
        dtype=dtype,
    )
    for foot, indices in ((0, left), (1, right)):
        rows = (selected_foot == foot).nonzero(as_tuple=False).flatten()
        if rows.numel():
            columns = torch.tensor(indices, device=selected_foot.device)
            scale[rows.unsqueeze(1), columns.unsqueeze(0)] = float(gain)
    return scale


def actor_coordinate_teacher_action(
    safe_plant_action: torch.Tensor,
    plant_scale: torch.Tensor,
    *,
    plant_gain: float = 1.0,
    plant_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the hidden plant map and return an actor-coordinate CBF target.

    The environment applies ``plant_gain * plant_scale * action + plant_bias``
    before the CBF.  Directly imitating the post-plant safe action would apply
    the gain a second time on the next step.  This inverse is therefore part of
    the v25 scientific dataflow, not merely a logging conversion.

    Returns the detached-coordinate target and its exact forward reprojection.
    """
    if safe_plant_action.shape != plant_scale.shape:
        raise ValueError("safe action and plant scale must have identical shape")
    if safe_plant_action.ndim < 2:
        raise ValueError("teacher actions must include batch and action dimensions")
    if not 0.0 < float(plant_gain) <= 2.0:
        raise ValueError("plant gain must lie in (0, 2]")
    if not bool(torch.isfinite(safe_plant_action).all()):
        raise RuntimeError("safe plant action contains non-finite values")
    if not bool(torch.isfinite(plant_scale).all() and (plant_scale > 0.0).all()):
        raise RuntimeError("plant scale must be finite and strictly positive")
    if plant_bias is None:
        bias = torch.zeros_like(safe_plant_action)
    else:
        bias = plant_bias
        if bias.shape == safe_plant_action.shape[-1:]:
            bias = bias.expand_as(safe_plant_action)
        if bias.shape != safe_plant_action.shape:
            raise ValueError("plant bias must broadcast over the action dimension")
        if not bool(torch.isfinite(bias).all()):
            raise RuntimeError("plant bias contains non-finite values")
    denominator = float(plant_gain) * plant_scale
    teacher = (safe_plant_action - bias) / denominator
    reprojected = float(plant_gain) * plant_scale * teacher + bias
    return teacher, reprojected


def successful_teacher_labels(
    intervened: torch.Tensor,
    correction_norm: torch.Tensor,
    pre_step_stair_index: torch.Tensor,
    post_step_stair_index: torch.Tensor,
    fell: torch.Tensor,
    dones: torch.Tensor,
    *,
    horizon: int,
    correction_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Label interventions followed by a next-riser crossing without a fall.

    Look-ahead includes the transition containing the intervention.  Episode
    boundaries terminate the scan, so an auto-reset trajectory can never supply
    another episode's crossing or survival evidence.
    """
    tensors = (
        intervened,
        correction_norm,
        pre_step_stair_index,
        post_step_stair_index,
        fell,
        dones,
    )
    shape = intervened.shape
    if intervened.ndim != 2 or any(tensor.shape != shape for tensor in tensors):
        raise ValueError("teacher label inputs must share [T, N] shape")
    if horizon < 1:
        raise ValueError("teacher success horizon must be positive")
    if not 0.0 < float(correction_scale):
        raise ValueError("teacher correction scale must be positive")
    if not bool(torch.isfinite(correction_norm).all()):
        raise RuntimeError("teacher correction norm contains non-finite values")
    if bool((correction_norm < 0.0).any()):
        raise ValueError("teacher correction norm must be non-negative")

    intervened = intervened.bool()
    fell = fell.bool()
    dones = dones.bool()
    crossed = torch.zeros_like(intervened)
    no_fall = torch.ones_like(intervened)
    horizon_observed = torch.zeros_like(intervened)
    time_steps = shape[0]
    for start in range(time_steps):
        alive = torch.ones(shape[1], dtype=torch.bool, device=intervened.device)
        crossed_from_start = torch.zeros_like(alive)
        no_fall_from_start = torch.ones_like(alive)
        stop = min(time_steps, start + horizon)
        for step in range(start, stop):
            no_fall_from_start &= ~(alive & fell[step])
            crossed_from_start |= alive & (
                post_step_stair_index[step] > pre_step_stair_index[start]
            )
            alive &= ~dones[step]
        crossed[start] = crossed_from_start
        no_fall[start] = no_fall_from_start
        # A terminal transition supplies a complete episode outcome.  An
        # otherwise ongoing trajectory must contain all H requested steps;
        # rollout truncation alone is not evidence of future survival.
        horizon_observed[start] = (~alive) | (stop - start == horizon)

    eligible = intervened & crossed & no_fall & horizon_observed
    magnitude_weight = torch.clamp(correction_norm / float(correction_scale), 0.0, 1.0)
    weights = eligible.float() * magnitude_weight
    diagnostics = {
        "intervened": intervened,
        "crossed_within_horizon": crossed,
        "no_fall_within_horizon": no_fall,
        "horizon_outcome_observed": horizon_observed,
        "magnitude_weight": magnitude_weight,
    }
    return eligible, weights, diagnostics


def weighted_gaussian_teacher_loss(
    policy_mean: torch.Tensor,
    policy_std: torch.Tensor,
    teacher_action: torch.Tensor,
    eligible: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return success-gated Gaussian NLL (up to a frozen constant).

    The denominator is the number of valid teacher transitions, as declared by
    the protocol.  Magnitude weights scale numerator contributions but do not
    change that denominator.  An empty minibatch returns an exact differentiable
    zero.
    """
    if not (policy_mean.shape == policy_std.shape == teacher_action.shape):
        raise ValueError("teacher Gaussian tensors must have equal [B, A] shape")
    if policy_mean.ndim != 2:
        raise ValueError("teacher Gaussian tensors must have shape [B, A]")
    if eligible.shape != policy_mean.shape[:-1] or weights.shape != eligible.shape:
        raise ValueError("teacher masks and weights must have shape [B]")
    if not bool(
        torch.isfinite(policy_mean).all()
        and torch.isfinite(policy_std).all()
        and torch.isfinite(teacher_action).all()
        and torch.isfinite(weights).all()
    ):
        raise RuntimeError("teacher objective contains non-finite values")
    if bool((policy_std <= 0.0).any() or (weights < 0.0).any()):
        raise ValueError(
            "teacher standard deviations/weights must be positive/non-negative"
        )
    eligible = eligible.bool()
    valid_count = eligible.sum()
    if not bool(valid_count):
        return policy_mean.sum() * 0.0
    standardized = (policy_mean - teacher_action.detach()) / policy_std.detach()
    negative_log_likelihood = 0.5 * standardized.square().sum(dim=-1)
    return (weights * negative_log_likelihood).sum() / valid_count


def toe_riser_kick_event(
    h: torch.Tensor,
    geometric_active: torch.Tensor,
    current_identity: torch.Tensor,
    previous_identity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Debounce entries into the exact unsafe half-space per toe/riser pair.

    A single overlap bit is insufficient when the selected swing foot or active
    riser changes without an intervening safe sample.  Encoding both identities
    prevents that transition from suppressing a real event on the new pair.
    ``-1`` is reserved for no active overlap.
    """
    tensors = (h, geometric_active, current_identity, previous_identity)
    if h.ndim != 1 or any(tensor.shape != h.shape for tensor in tensors):
        raise ValueError("toe-riser event tensors must share [N] shape")
    if (
        current_identity.dtype.is_floating_point
        or previous_identity.dtype.is_floating_point
    ):
        raise ValueError("toe-riser identities must be integer tensors")
    if bool((current_identity < 0).any()):
        raise ValueError("active toe-riser identities must be non-negative")
    overlap = geometric_active.bool() & torch.isfinite(h) & (h <= 0.0)
    event = overlap & (current_identity != previous_identity)
    next_identity = torch.where(
        overlap,
        current_identity,
        torch.full_like(current_identity, -1),
    )
    return event, overlap, next_identity
