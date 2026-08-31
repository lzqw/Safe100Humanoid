"""Offline reference for the v139 hardware observation/action/CBF bridge.

This module has no Unitree transport and never sends commands.  It isolates the
simulation-side semantics that a separately reviewed real-time controller must
reproduce before any hardware traversal is allowed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .cbf_math import project_halfspace, sloped_toe_clearance_constraint
from .edge_detection import select_active_riser

ACTOR_OBSERVATION_DIM = 405
ACTION_DIM = 12
HISTORY_LENGTH = 5
POLICY_PERIOD_S = 0.02
GAIT_PERIOD_S = 0.6
STAND_COMMAND_THRESHOLD = 0.1

ACTOR_TERM_WIDTHS = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("command", 3),
  ("phase", 2),
  ("joint_pos_relative", 29),
  ("joint_vel_relative", 29),
  ("previous_raw_action", 12),
)

G1_DEFAULT_JOINT_POSITION = (
  -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
  -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
  0.0, 0.0, 0.0,
  0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
  0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
)
LOWER_BODY_JOINT_INDICES = tuple(range(ACTION_DIM))
LOWER_BODY_ACTION_OFFSET = G1_DEFAULT_JOINT_POSITION[:ACTION_DIM]
LOWER_BODY_ACTION_SCALE = (
  0.5475464629911068,
  0.35066146637882434,
  0.5475464629911068,
  0.35066146637882434,
  0.43857731392336724,
  0.43857731392336724,
  0.5475464629911068,
  0.35066146637882434,
  0.5475464629911068,
  0.35066146637882434,
  0.43857731392336724,
  0.43857731392336724,
)


def _float_vector(name: str, value: torch.Tensor | Sequence[float], width: int):
  vector = torch.as_tensor(value, dtype=torch.float32).detach().clone().reshape(-1)
  if vector.numel() != width:
    raise ValueError(f"{name} must contain {width} values, got {vector.numel()}")
  if not bool(torch.isfinite(vector).all()):
    raise ValueError(f"{name} contains NaN or infinity")
  return vector


def gait_phase(
  episode_step: int,
  command: torch.Tensor | Sequence[float],
  *,
  step_dt: float = POLICY_PERIOD_S,
  period: float = GAIT_PERIOD_S,
) -> torch.Tensor:
  """Match the deployable two-channel simulation gait clock."""
  if episode_step < 0 or step_dt <= 0.0 or period <= 0.0:
    raise ValueError("gait step must be non-negative and timing must be positive")
  command_vector = _float_vector("command", command, 3)
  if float(torch.linalg.vector_norm(command_vector)) < STAND_COMMAND_THRESHOLD:
    return torch.zeros(2, dtype=torch.float32)
  phase = (float(episode_step) * float(step_dt) / float(period)) % 1.0
  angle = torch.tensor(phase * 2.0 * torch.pi, dtype=torch.float32)
  return torch.stack((torch.sin(angle), torch.cos(angle)))


class ActorObservationHistory:
  """Build the exact term-major, oldest-to-newest v139 actor tensor."""

  def __init__(self) -> None:
    self._history: dict[str, list[torch.Tensor]] = {}

  @property
  def initialized(self) -> bool:
    return bool(self._history)

  def reset(self) -> None:
    self._history.clear()

  def push(
    self,
    *,
    base_ang_vel: torch.Tensor | Sequence[float],
    projected_gravity: torch.Tensor | Sequence[float],
    command: torch.Tensor | Sequence[float],
    episode_step: int,
    joint_position: torch.Tensor | Sequence[float],
    joint_velocity: torch.Tensor | Sequence[float],
    previous_raw_action: torch.Tensor | Sequence[float],
  ) -> torch.Tensor:
    """Append one 20 ms sample and return a fixed-shape ``[1, 405]`` tensor."""
    command_vector = _float_vector("command", command, 3)
    joint_position_vector = _float_vector("joint_position", joint_position, 29)
    default_position = torch.tensor(G1_DEFAULT_JOINT_POSITION)
    frame = {
      "base_ang_vel": _float_vector("base_ang_vel", base_ang_vel, 3),
      "projected_gravity": _float_vector(
        "projected_gravity", projected_gravity, 3
      ),
      "command": command_vector,
      "phase": gait_phase(episode_step, command_vector),
      "joint_pos_relative": joint_position_vector - default_position,
      "joint_vel_relative": _float_vector("joint_velocity", joint_velocity, 29),
      "previous_raw_action": _float_vector(
        "previous_raw_action", previous_raw_action, ACTION_DIM
      ),
    }
    if not self.initialized:
      self._history = {
        name: [value.clone() for _ in range(HISTORY_LENGTH)]
        for name, value in frame.items()
      }
    else:
      for name, value in frame.items():
        self._history[name] = [*self._history[name][1:], value]
    return self.tensor()

  def tensor(self) -> torch.Tensor:
    if not self.initialized:
      raise RuntimeError("observation history has no samples")
    terms = []
    for name, width in ACTOR_TERM_WIDTHS:
      history = torch.stack(self._history[name], dim=0)
      if history.shape != (HISTORY_LENGTH, width):
        raise RuntimeError(f"internal history shape differs for {name}")
      terms.append(history.reshape(-1))
    observation = torch.cat(terms).reshape(1, -1)
    if observation.shape != (1, ACTOR_OBSERVATION_DIM):
      raise RuntimeError("actor observation contract is not 405-D")
    return observation


def nominal_lower_body_target(
  raw_action: torch.Tensor | Sequence[float],
) -> torch.Tensor:
  """Convert the actor output to the simulation-equivalent 12 joint targets."""
  raw = _float_vector("raw_action", raw_action, ACTION_DIM)
  offset = torch.tensor(LOWER_BODY_ACTION_OFFSET)
  scale = torch.tensor(LOWER_BODY_ACTION_SCALE)
  return offset + scale * raw


def embed_lower_body_target(
  lower_body_target: torch.Tensor | Sequence[float],
  approved_full_body_default: torch.Tensor | Sequence[float],
) -> torch.Tensor:
  """Place the 12 targets into an operator-approved 29-joint posture."""
  lower = _float_vector("lower_body_target", lower_body_target, ACTION_DIM)
  full = _float_vector(
    "approved_full_body_default", approved_full_body_default, 29
  )
  full[list(LOWER_BODY_JOINT_INDICES)] = lower
  return full


@dataclass(frozen=True)
class StairCbfConfig:
  step_dt: float = POLICY_PERIOD_S
  alpha: float = 10.0
  activation_distance: float = 0.30
  toe_margin: float = 0.08
  top_clearance: float = 0.025
  clearance_barrier_slope: float = 0.8
  recovery_distance: float = 0.15
  intervention_epsilon: float = 1.0e-5

  def __post_init__(self) -> None:
    positive = (
      self.step_dt,
      self.alpha,
      self.activation_distance,
      self.toe_margin,
      self.top_clearance,
      self.clearance_barrier_slope,
      self.recovery_distance,
      self.intervention_epsilon,
    )
    if any(value <= 0.0 for value in positive):
      raise ValueError("real-robot reference CBF parameters must be positive")
    if self.clearance_barrier_slope > 2.0:
      raise ValueError("clearance barrier slope must not exceed 2")


@dataclass(frozen=True)
class CbfStepResult:
  nominal_raw_action: torch.Tensor
  safe_raw_action: torch.Tensor
  nominal_position_target: torch.Tensor
  safe_position_target: torch.Tensor
  selected_foot: int
  selected_riser: int
  active: bool
  intervened: bool
  barrier: float
  nominal_margin: float
  projected_margin: float
  velocity_correction_norm: float
  target_correction_norm: float


def project_stair_cbf_action(
  *,
  raw_action: torch.Tensor | Sequence[float],
  lower_body_joint_position: torch.Tensor | Sequence[float],
  foot_position_xz: torch.Tensor | Sequence[Sequence[float]],
  foot_jacobian_xz: torch.Tensor | Sequence[Sequence[Sequence[float]]],
  foot_contact: torch.Tensor | Sequence[bool],
  edge_x: torch.Tensor | Sequence[float],
  edge_top_z: torch.Tensor | Sequence[float],
  foot_air_time: torch.Tensor | Sequence[float] | None = None,
  cfg: StairCbfConfig | None = None,
) -> CbfStepResult:
  """Apply the v139 single-half-space toe/riser projection to one state.

  Inputs must already be synchronized in one fixed staircase frame.  This
  function raises on missing/invalid state; a real controller must convert
  that failure into its independently approved fail-closed behavior.
  """
  cfg = StairCbfConfig() if cfg is None else cfg
  raw = _float_vector("raw_action", raw_action, ACTION_DIM)
  q = _float_vector(
    "lower_body_joint_position", lower_body_joint_position, ACTION_DIM
  )
  foot_xz = torch.as_tensor(foot_position_xz, dtype=torch.float32).detach().clone()
  jac_xz = torch.as_tensor(foot_jacobian_xz, dtype=torch.float32).detach().clone()
  contact = torch.as_tensor(foot_contact).detach().clone()
  edges_x = torch.as_tensor(edge_x, dtype=torch.float32).detach().clone().reshape(-1)
  edges_z = (
    torch.as_tensor(edge_top_z, dtype=torch.float32).detach().clone().reshape(-1)
  )
  if foot_xz.shape != (2, 2):
    raise ValueError("foot_position_xz must have shape [2, 2]")
  if jac_xz.shape != (2, 2, ACTION_DIM):
    raise ValueError("foot_jacobian_xz must have shape [2, 2, 12]")
  if contact.shape != (2,):
    raise ValueError("foot_contact must have shape [2]")
  if contact.dtype != torch.bool:
    raise TypeError("foot_contact must contain booleans")
  if edges_x.shape != edges_z.shape or edges_x.numel() < 1:
    raise ValueError("edge_x and edge_top_z must share a non-empty shape")
  numeric = (foot_xz, jac_xz, edges_x, edges_z)
  if not all(bool(torch.isfinite(value).all()) for value in numeric):
    raise ValueError("CBF geometry or kinematics contains NaN or infinity")
  if edges_x.numel() > 1 and not bool((torch.diff(edges_x) > 0.0).all()):
    raise ValueError("riser x coordinates must be strictly increasing")

  nominal_target = nominal_lower_body_target(raw)
  qdot_nominal = (nominal_target - q) / float(cfg.step_dt)
  in_air = ~contact
  if not bool(in_air.any()):
    selected_foot = -1
  elif foot_air_time is None:
    selected_foot = int(in_air.float().argmax())
  else:
    air_time = _float_vector("foot_air_time", foot_air_time, 2)
    if bool((air_time < 0.0).any()):
      raise ValueError("foot_air_time must be non-negative")
    scores = torch.where(in_air, air_time, torch.full_like(air_time, -1.0))
    selected_foot = int(scores.argmax())

  if selected_foot < 0:
    return CbfStepResult(
      nominal_raw_action=raw,
      safe_raw_action=raw.clone(),
      nominal_position_target=nominal_target,
      safe_position_target=nominal_target.clone(),
      selected_foot=-1,
      selected_riser=-1,
      active=False,
      intervened=False,
      barrier=float("inf"),
      nominal_margin=0.0,
      projected_margin=0.0,
      velocity_correction_norm=0.0,
      target_correction_norm=0.0,
    )

  selected_position = foot_xz[selected_foot]
  riser_index, horizontal_h, selected_top_z, edge_active = select_active_riser(
    selected_position[0].reshape(1),
    selected_position[1].reshape(1),
    edges_x.reshape(1, -1),
    edges_z.reshape(1, -1),
    toe_margin=cfg.toe_margin,
    top_clearance=cfg.top_clearance,
    activation_distance=cfg.activation_distance,
    recovery_distance=cfg.recovery_distance,
  )
  active = bool(edge_active.item())
  if not active:
    return CbfStepResult(
      nominal_raw_action=raw,
      safe_raw_action=raw.clone(),
      nominal_position_target=nominal_target,
      safe_position_target=nominal_target.clone(),
      selected_foot=selected_foot,
      selected_riser=-1,
      active=False,
      intervened=False,
      barrier=float("inf"),
      nominal_margin=0.0,
      projected_margin=0.0,
      velocity_correction_norm=0.0,
      target_correction_norm=0.0,
    )

  barrier, normal = sloped_toe_clearance_constraint(
    horizontal_h.reshape(1),
    selected_position[1].reshape(1),
    selected_top_z.reshape(1),
    jac_xz[selected_foot, 0].reshape(1, ACTION_DIM),
    jac_xz[selected_foot, 1].reshape(1, ACTION_DIM),
    top_clearance=cfg.top_clearance,
    slope=cfg.clearance_barrier_slope,
  )
  rhs = -float(cfg.alpha) * barrier
  projected_qdot, nominal_margin, projected_margin = project_halfspace(
    qdot_nominal.reshape(1, ACTION_DIM),
    normal,
    rhs,
    active=torch.ones(1, dtype=torch.bool),
  )
  safe_target = q + float(cfg.step_dt) * projected_qdot.reshape(-1)
  offset = torch.tensor(LOWER_BODY_ACTION_OFFSET)
  scale = torch.tensor(LOWER_BODY_ACTION_SCALE)
  safe_raw = (safe_target - offset) / scale
  velocity_correction = float(
    torch.linalg.vector_norm(projected_qdot.reshape(-1) - qdot_nominal)
  )
  target_correction = float(torch.linalg.vector_norm(safe_target - nominal_target))
  intervened = (
    float(nominal_margin.item()) < -cfg.intervention_epsilon
    or target_correction > cfg.intervention_epsilon
  )
  return CbfStepResult(
    nominal_raw_action=raw,
    safe_raw_action=safe_raw,
    nominal_position_target=nominal_target,
    safe_position_target=safe_target,
    selected_foot=selected_foot,
    selected_riser=int(riser_index.item()),
    active=True,
    intervened=intervened,
    barrier=float(barrier.item()),
    nominal_margin=float(nominal_margin.item()),
    projected_margin=float(projected_margin.item()),
    velocity_correction_norm=velocity_correction,
    target_correction_norm=target_correction,
  )
