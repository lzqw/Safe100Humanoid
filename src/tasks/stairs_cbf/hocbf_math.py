"""Pure-Torch acceleration HOCBF and task-metric projection helpers."""

from __future__ import annotations

import torch


def apply_task_metric_inverse(
    vector: torch.Tensor,
    diagonal_metric: torch.Tensor,
    jac_x: torch.Tensor,
    forward_weight: float,
    smoothness_weight: float,
    *,
    eps: float = 1.0e-9,
) -> torch.Tensor:
    """Apply ``(D + lambda_s I + lambda_x Jx'Jx)^-1`` in closed form.

    Every row is independent and remains on the input device.  ``D`` may be a
    shared action-dimensional vector or a tensor broadcastable to ``vector``.
    """
    if vector.shape != jac_x.shape:
        raise ValueError(
            f"v33 vector/Jx shape mismatch: {vector.shape} != {jac_x.shape}"
        )
    if diagonal_metric.shape[-1:] != vector.shape[-1:]:
        raise ValueError("v33 diagonal metric must match the action dimension")
    if forward_weight < 0.0 or smoothness_weight < 0.0:
        raise ValueError("v33 task-metric weights must be non-negative")
    diagonal = diagonal_metric + float(smoothness_weight)
    # The fixed v33 metric is validated once during action construction.  A
    # clamp keeps this hot-path helper fully GPU-vectorized without a tensor
    # truth-value synchronization on every control step.
    inverse_diagonal = diagonal.clamp_min(eps).reciprocal()
    direct = inverse_diagonal * vector
    if forward_weight == 0.0:
        return direct
    inverse_jacobian = inverse_diagonal * jac_x
    numerator_inner = torch.sum(jac_x * direct, dim=-1)
    denominator = 1.0 + float(forward_weight) * torch.sum(
        jac_x * inverse_jacobian, dim=-1
    )
    denominator = denominator.clamp_min(eps)
    rank_one = (
        float(forward_weight)
        * inverse_jacobian
        * (numerator_inner / denominator).unsqueeze(-1)
    )
    return direct - rank_one


def estimate_hocbf_derivatives(
    barrier: torch.Tensor,
    joint_velocity: torch.Tensor,
    normal: torch.Tensor,
    previous_barrier: torch.Tensor,
    previous_barrier_derivative: torch.Tensor,
    previous_joint_velocity: torch.Tensor,
    previous_drift: torch.Tensor,
    identity_continuous: torch.Tensor,
    *,
    control_dt: float,
    drift_ema_previous: float = 0.8,
    drift_clip: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate ``h_dot``, measured acceleration, instantaneous drift and EMA.

    A discontinuous identity uses the kinematic fallback ``n @ q_dot`` and a
    zero drift exactly, as required after reset or foot/riser changes.
    """
    if barrier.shape != joint_velocity.shape[:-1]:
        raise ValueError("v33 barrier must have shape joint_velocity.shape[:-1]")
    if normal.shape != joint_velocity.shape:
        raise ValueError("v33 normal and joint velocity must share shape")
    if previous_joint_velocity.shape != joint_velocity.shape:
        raise ValueError("v33 previous joint velocity shape differs")
    for value in (
        previous_barrier,
        previous_barrier_derivative,
        previous_drift,
        identity_continuous,
    ):
        if value.shape != barrier.shape:
            raise ValueError("v33 scalar history shape differs")
    if control_dt <= 0.0:
        raise ValueError("v33 control dt must be positive")
    if not 0.0 <= drift_ema_previous < 1.0:
        raise ValueError("v33 drift EMA coefficient must lie in [0, 1)")
    if drift_clip <= 0.0:
        raise ValueError("v33 drift clip must be positive")

    finite_difference = (barrier - previous_barrier) / float(control_dt)
    kinematic = torch.sum(normal * joint_velocity, dim=-1)
    barrier_derivative = torch.where(identity_continuous, finite_difference, kinematic)
    measured_acceleration = (joint_velocity - previous_joint_velocity) / float(
        control_dt
    )
    instantaneous_drift = (
        (barrier_derivative - previous_barrier_derivative) / float(control_dt)
        - torch.sum(normal * measured_acceleration, dim=-1)
    ).clamp(-float(drift_clip), float(drift_clip))
    drift = (
        float(drift_ema_previous) * previous_drift
        + (1.0 - float(drift_ema_previous)) * instantaneous_drift
    ).clamp(-float(drift_clip), float(drift_clip))
    instantaneous_drift = torch.where(
        identity_continuous,
        instantaneous_drift,
        torch.zeros_like(instantaneous_drift),
    )
    drift = torch.where(identity_continuous, drift, torch.zeros_like(drift))
    return barrier_derivative, measured_acceleration, instantaneous_drift, drift


def hocbf_acceleration_rhs(
    barrier: torch.Tensor,
    barrier_derivative: torch.Tensor,
    drift: torch.Tensor,
    *,
    omega: float,
    zeta: float = 1.0,
) -> torch.Tensor:
    """Return the joint-acceleration right-hand side for the relative-degree-2 CBF."""
    if barrier.shape != barrier_derivative.shape or barrier.shape != drift.shape:
        raise ValueError("v33 HOCBF scalar tensors must share shape")
    if omega <= 0.0 or zeta <= 0.0:
        raise ValueError("v33 omega and zeta must be positive")
    return (
        -drift
        - 2.0 * float(zeta) * float(omega) * barrier_derivative
        - float(omega) ** 2 * barrier
    )


def project_task_consistent_hocbf(
    nominal_acceleration: torch.Tensor,
    normal: torch.Tensor,
    rhs: torch.Tensor,
    previous_correction: torch.Tensor,
    diagonal_metric: torch.Tensor,
    jac_x: torch.Tensor,
    active: torch.Tensor,
    *,
    forward_weight: float,
    smoothness_weight: float,
    eps: float = 1.0e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the single-constraint task-consistent HOCBF QP on the GPU.

    Returns ``safe_acceleration, correction, nominal_margin, projected_margin``.
    Rows whose nominal margin is safe are returned exactly unchanged; the
    smoothness center is never allowed to perturb an already-safe nominal row.
    """
    if nominal_acceleration.shape != normal.shape:
        raise ValueError("v33 nominal acceleration and normal must share shape")
    if previous_correction.shape != nominal_acceleration.shape:
        raise ValueError("v33 previous correction shape differs")
    if jac_x.shape != nominal_acceleration.shape:
        raise ValueError("v33 Jx shape differs")
    if rhs.shape != nominal_acceleration.shape[:-1] or active.shape != rhs.shape:
        raise ValueError("v33 RHS/active shape differs")

    nominal_margin = torch.sum(normal * nominal_acceleration, dim=-1) - rhs
    violated = active & (nominal_margin < 0.0)
    smooth_center = apply_task_metric_inverse(
        float(smoothness_weight) * previous_correction,
        diagonal_metric,
        jac_x,
        forward_weight,
        smoothness_weight,
        eps=eps,
    )
    inverse_normal = apply_task_metric_inverse(
        normal,
        diagonal_metric,
        jac_x,
        forward_weight,
        smoothness_weight,
        eps=eps,
    )
    required_normal_correction = -nominal_margin
    center_normal_correction = torch.sum(normal * smooth_center, dim=-1)
    residual = torch.relu(required_normal_correction - center_normal_correction)
    denominator = torch.sum(normal * inverse_normal, dim=-1).clamp_min(eps)
    projected_correction = (
        smooth_center + (residual / denominator).unsqueeze(-1) * inverse_normal
    )
    correction = torch.where(
        violated.unsqueeze(-1),
        projected_correction,
        torch.zeros_like(projected_correction),
    )
    safe_acceleration = nominal_acceleration + correction
    # One vectorized residual repair absorbs float32 cancellation at very large
    # acceleration magnitudes.  It is still the same weighted half-space
    # direction and leaves every nominally safe row bit-for-bit unchanged.
    numerical_residual = torch.relu(rhs - torch.sum(normal * safe_acceleration, dim=-1))
    repair = (numerical_residual / denominator).unsqueeze(-1) * inverse_normal
    correction = correction + torch.where(
        violated.unsqueeze(-1), repair, torch.zeros_like(repair)
    )
    safe_acceleration = nominal_acceleration + correction
    projected_margin = torch.sum(normal * safe_acceleration, dim=-1) - rhs
    return safe_acceleration, correction, nominal_margin, projected_margin
