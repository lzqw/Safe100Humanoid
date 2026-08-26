"""Pure-Torch task-metric velocity-CBF projection helpers for v34."""

from __future__ import annotations

import torch


def apply_velocity_task_metric_inverse(
    vector: torch.Tensor,
    diagonal_metric: torch.Tensor,
    jac_x: torch.Tensor,
    forward_weight: float,
    smoothness_weight: float,
    *,
    eps: float = 1.0e-9,
) -> torch.Tensor:
    """Apply ``(W + lambda_s I + lambda_x Jx'Jx)^-1`` on the GPU."""
    if vector.shape != jac_x.shape or vector.shape != diagonal_metric.shape:
        raise ValueError("v34 vector, metric and Jx must share shape")
    if forward_weight < 0.0 or smoothness_weight < 0.0:
        raise ValueError("v34 task weights must be non-negative")
    inverse_diagonal = (
        (diagonal_metric + float(smoothness_weight)).clamp_min(eps).reciprocal()
    )
    direct = inverse_diagonal * vector
    if forward_weight == 0.0:
        return direct
    inverse_jacobian = inverse_diagonal * jac_x
    numerator = torch.sum(jac_x * direct, dim=-1)
    denominator = 1.0 + float(forward_weight) * torch.sum(
        jac_x * inverse_jacobian, dim=-1
    )
    return direct - (
        float(forward_weight)
        * inverse_jacobian
        * (numerator / denominator.clamp_min(eps)).unsqueeze(-1)
    )


def project_task_metric_velocity_cbf(
    nominal_velocity: torch.Tensor,
    previous_safe_velocity: torch.Tensor,
    normal: torch.Tensor,
    rhs: torch.Tensor,
    diagonal_metric: torch.Tensor,
    jac_x: torch.Tensor,
    active: torch.Tensor,
    history_continuous: torch.Tensor,
    *,
    forward_weight: float,
    smoothness_weight: float,
    eps: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve v34's one-constraint weighted velocity QP in closed form.

    An already-safe nominal row is returned bit-for-bit unchanged, regardless
    of the smoothness term.
    """
    if not (
        nominal_velocity.shape
        == previous_safe_velocity.shape
        == normal.shape
        == diagonal_metric.shape
        == jac_x.shape
    ):
        raise ValueError("v34 velocity projection vector shapes differ")
    if rhs.shape != nominal_velocity.shape[:-1]:
        raise ValueError("v34 RHS shape differs")
    if active.shape != rhs.shape or history_continuous.shape != rhs.shape:
        raise ValueError("v34 active/history shape differs")

    nominal_margin = torch.sum(normal * nominal_velocity, dim=-1) - rhs
    violated = active & (nominal_margin < 0.0)
    reference_delta = torch.where(
        history_continuous.unsqueeze(-1),
        previous_safe_velocity - nominal_velocity,
        torch.zeros_like(nominal_velocity),
    )
    center = apply_velocity_task_metric_inverse(
        float(smoothness_weight) * reference_delta,
        diagonal_metric,
        jac_x,
        forward_weight,
        smoothness_weight,
        eps=eps,
    )
    inverse_normal = apply_velocity_task_metric_inverse(
        normal,
        diagonal_metric,
        jac_x,
        forward_weight,
        smoothness_weight,
        eps=eps,
    )
    denominator = torch.sum(normal * inverse_normal, dim=-1).clamp_min(eps)
    required = -nominal_margin
    center_normal = torch.sum(normal * center, dim=-1)
    projected = (
        center
        + (torch.relu(required - center_normal) / denominator).unsqueeze(-1)
        * inverse_normal
    )
    correction = torch.where(
        violated.unsqueeze(-1), projected, torch.zeros_like(projected)
    )
    safe_velocity = nominal_velocity + correction
    # Repair only float32 half-space cancellation, in the same weighted
    # direction. Safe nominal rows remain exactly untouched.
    residual = torch.relu(rhs - torch.sum(normal * safe_velocity, dim=-1))
    repair = (residual / denominator).unsqueeze(-1) * inverse_normal
    correction = correction + torch.where(
        violated.unsqueeze(-1), repair, torch.zeros_like(repair)
    )
    safe_velocity = nominal_velocity + correction
    projected_margin = torch.sum(normal * safe_velocity, dim=-1) - rhs
    return safe_velocity, correction, nominal_margin, projected_margin
