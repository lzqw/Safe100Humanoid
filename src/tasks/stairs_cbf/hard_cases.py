"""On-policy hard-case state capture and replay for online stair refinement.

The bank stores simulator states from shortly before a *real* CBF projection.
Restoring only q/qd is insufficient: command latency, observation history,
contact air-time, previous actions, and stationary reward baselines are part of
the online MDP state.  This module captures those quantities explicitly while
keeping the actor interface unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


MIXED_FAILURE_TYPE = "mixed"
LATERAL_HEADING_DRIFT_FAILURE_TYPE = "lateral_heading_drift"
NON_LATERAL_HIGH_CBF_FAILURE_TYPE = "non_lateral_high_cbf_demand"
NON_LATERAL_BALANCE_FAILURE_TYPE = "non_lateral_balance_or_phase"
TARGET_FAILURE_TYPES = (
  LATERAL_HEADING_DRIFT_FAILURE_TYPE,
  NON_LATERAL_HIGH_CBF_FAILURE_TYPE,
  NON_LATERAL_BALANCE_FAILURE_TYPE,
)
SPECIALIST_MODES = ("lateral", "cbf", "balance")
SPECIALIST_FAILURE_TYPES = {
  "lateral": LATERAL_HEADING_DRIFT_FAILURE_TYPE,
  "cbf": NON_LATERAL_HIGH_CBF_FAILURE_TYPE,
  "balance": NON_LATERAL_BALANCE_FAILURE_TYPE,
}
SPECIALIST_FAILURE_BANK_KIND = "specialist_failure_precursor"
SPECIALIST_SUCCESS_POOL_KIND = "specialist_success_pool"
SPECIALIST_SUCCESS_BANK_KIND = "specialist_success_counterexample"
SPECIALIST_BANK_KINDS = (
  SPECIALIST_FAILURE_BANK_KIND,
  SPECIALIST_SUCCESS_POOL_KIND,
  SPECIALIST_SUCCESS_BANK_KIND,
)
LATERAL_CENTERLINE_WIDTH_FRACTION = 2.0 / 3.0
LATERAL_HEADING_THRESHOLD_RAD = math.pi / 2.0
HIGH_CBF_CORRECTION_THRESHOLD = 0.5


def classify_target_failure_mode(
  *,
  side_edge_breach: bool,
  max_abs_centerline_error: float,
  max_abs_heading_error: float,
  correction_max: float,
  stair_half_width: float = 1.2,
) -> str:
  """Assign one deterministic, mutually exclusive target-fall class.

  The lateral class uses geometry-derived thresholds: a root/foot edge breach,
  root drift beyond two thirds of the stair half-width, or heading error of at
  least 90 degrees. The remaining failures are split by high runtime-CBF
  demand. This classifier is used only for Branch-B diagnosis and hard-bank
  selection; it never changes the reward or promotion gates.
  """
  values = torch.tensor(
    [
      max_abs_centerline_error,
      max_abs_heading_error,
      correction_max,
      stair_half_width,
    ],
    dtype=torch.float64,
  )
  if not bool(torch.isfinite(values).all()):
    raise ValueError("failure classification inputs must be finite")
  if stair_half_width <= 0.0:
    raise ValueError("stair half-width must be positive")
  if min(max_abs_centerline_error, max_abs_heading_error, correction_max) < 0.0:
    raise ValueError("failure classification magnitudes must be non-negative")
  if (
    side_edge_breach
    or max_abs_centerline_error
    >= LATERAL_CENTERLINE_WIDTH_FRACTION * stair_half_width
    or max_abs_heading_error >= LATERAL_HEADING_THRESHOLD_RAD
  ):
    return LATERAL_HEADING_DRIFT_FAILURE_TYPE
  if correction_max >= HIGH_CBF_CORRECTION_THRESHOLD:
    return NON_LATERAL_HIGH_CBF_FAILURE_TYPE
  return NON_LATERAL_BALANCE_FAILURE_TYPE


_ACTION_MANAGER_ATTRS = ("_action", "_prev_action", "_prev_prev_action")
_ACTION_TERM_ATTRS = (
  "_raw_actions",
  "_processed_actions",
  "h",
  "psi_nominal",
  "psi_filtered",
  "filter_active",
  "geometric_active",
  "intervened",
  "would_intervene",
  "intervention_norm",
  "target_intervention_norm",
  "counterfactual_intervention_norm",
  "counterfactual_target_intervention_norm",
  "intervention_count",
  "selected_edge_top_z",
  "selected_foot",
  "nominal_target",
  "safe_target",
  "nominal_raw_action",
  "safe_raw_action",
  "executed_raw_action",
  "barrier_derivative",
  "predicted_h",
  "h_history",
  "correction_history",
  "_deployment_action_queue",
)
_COMMAND_ATTRS = (
  "time_left",
  "command_counter",
  "raw_command",
  "delivered_command",
  "command_derivative",
  "delay_steps",
  "_delay_queue",
  "_next_pulse_steps",
  "_pulse_steps_left",
  "_pulse_value",
  "_release_countdown",
  "_released",
  "centerline_error",
  "heading_error",
  "operator_correction",
  "correction_active",
)


def _copy_batched_attr(
  output: dict[str, torch.Tensor], prefix: str, owner: Any, names: tuple[str, ...]
) -> None:
  for name in names:
    value = getattr(owner, name, None)
    if isinstance(value, torch.Tensor) and value.ndim >= 1:
      output[f"{prefix}/{name}"] = value.detach().clone()


def capture_hard_case_state(
  env,
  *,
  asset_name: str = "robot",
  action_name: str = "joint_pos",
  command_name: str = "twist",
) -> dict[str, torch.Tensor]:
  """Capture a batched, Markov-complete state for the online stair task."""
  robot = env.scene[asset_name]
  origins = env.scene.env_origins
  root_pose = robot.data.root_link_pose_w.clone()
  root_pose[:, :3] -= origins
  state: dict[str, torch.Tensor] = {
    "robot/root_pose_relative": root_pose,
    "robot/root_velocity_world": robot.data.root_link_vel_w.detach().clone(),
    "robot/joint_pos": robot.data.joint_pos.detach().clone(),
    "robot/joint_vel": robot.data.joint_vel.detach().clone(),
    "robot/joint_pos_target": robot.data.joint_pos_target.detach().clone(),
    "robot/joint_vel_target": robot.data.joint_vel_target.detach().clone(),
    "robot/joint_effort_target": robot.data.joint_effort_target.detach().clone(),
    "env/episode_length": env.episode_length_buf.detach().clone(),
  }
  terrain = env.scene.terrain
  if terrain is not None:
    state["terrain/type"] = terrain.terrain_types.detach().clone()
    state["terrain/level"] = terrain.terrain_levels.detach().clone()

  _copy_batched_attr(
    state, "action_manager", env.action_manager, _ACTION_MANAGER_ATTRS
  )
  action_term = env.action_manager.get_term(action_name)
  _copy_batched_attr(state, "action_term", action_term, _ACTION_TERM_ATTRS)
  # Store the selected edge height relative to the source environment origin.
  edge_key = "action_term/selected_edge_top_z"
  if edge_key in state:
    state[edge_key] -= origins[:, 2]

  command_term = env.command_manager.get_term(command_name)
  _copy_batched_attr(state, "command", command_term, _COMMAND_ATTRS)

  for sensor_name, sensor in env.scene.sensors.items():
    air_state = getattr(sensor, "_air_time_state", None)
    if air_state is not None:
      for name in (
        "current_air_time",
        "last_air_time",
        "current_contact_time",
        "last_contact_time",
      ):
        value = getattr(air_state, name, None)
        if isinstance(value, torch.Tensor):
          state[f"sensor/{sensor_name}/air/{name}"] = value.detach().clone()
    history_state = getattr(sensor, "_history_state", None)
    if isinstance(history_state, dict):
      for name, value in history_state.items():
        if isinstance(value, torch.Tensor):
          state[f"sensor/{sensor_name}/history/{name}"] = value.detach().clone()

  observation_manager = env.observation_manager
  for group, buffers in observation_manager._group_obs_term_history_buffer.items():
    for term, buffer in buffers.items():
      if not buffer.is_initialized:
        continue
      state[f"observation/{group}/{term}/history"] = buffer.buffer.detach().clone()
      state[f"observation/{group}/{term}/pushes"] = (
        buffer._num_pushes.detach().clone()
      )

  for cfg in env.reward_manager._class_term_cfgs:
    term = cfg.func
    name = term.__class__.__name__
    previous_x = getattr(term, "_previous_x", None)
    if isinstance(previous_x, torch.Tensor):
      state[f"reward/{name}/previous_x_relative"] = (
        previous_x.detach().clone() - origins[:, 0]
      )
    for attr in ("_initialized", "_previous_index", "_paid"):
      value = getattr(term, attr, None)
      if isinstance(value, torch.Tensor):
        state[f"reward/{name}/{attr}"] = value.detach().clone()
  return state


def hard_case_state_shape_mismatches(
  current: dict[str, torch.Tensor],
  replay: dict[str, torch.Tensor],
) -> list[str]:
  """Describe legacy replay layouts that cannot be restored atomically."""
  mismatches: list[str] = []
  for key, expected in current.items():
    value = replay.get(key)
    if value is None:
      mismatches.append(f"{key}: missing")
    elif value.shape[1:] != expected.shape[1:]:
      mismatches.append(
        f"{key}: replay {tuple(value.shape[1:])} != current {tuple(expected.shape[1:])}"
      )
  return mismatches


def select_hard_case_state(
  state: dict[str, torch.Tensor], env_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
  """Select environment rows while retaining a leading batch dimension."""
  return {key: value.index_select(0, env_ids).detach().clone() for key, value in state.items()}


def _restore_attr(
  state: dict[str, torch.Tensor],
  prefix: str,
  owner: Any,
  names: tuple[str, ...],
  env_ids: torch.Tensor,
) -> None:
  for name in names:
    key = f"{prefix}/{name}"
    target = getattr(owner, name, None)
    if key in state and isinstance(target, torch.Tensor):
      target[env_ids] = state[key].to(device=target.device, dtype=target.dtype)


def restore_hard_case_state(
  env,
  state: dict[str, torch.Tensor],
  env_ids: torch.Tensor,
  *,
  asset_name: str = "robot",
  action_name: str = "joint_pos",
  command_name: str = "twist",
) -> None:
  """Restore selected rows into already-reset environments.

  The caller must run ``sim.forward()`` and ``sim.sense()`` after this call.
  """
  if env_ids.ndim != 1:
    raise ValueError("env_ids must be one-dimensional")
  count = len(env_ids)
  if count == 0:
    return
  for key, value in state.items():
    if value.shape[0] != count:
      raise ValueError(
        f"state row count for {key} is {value.shape[0]}, expected {count}"
      )

  robot = env.scene[asset_name]
  root_pose = state["robot/root_pose_relative"].to(env.device).clone()
  root_pose[:, :3] += env.scene.env_origins[env_ids]
  root_velocity = state["robot/root_velocity_world"].to(env.device)
  robot.write_root_state_to_sim(
    torch.cat([root_pose, root_velocity], dim=1), env_ids=env_ids
  )
  robot.write_joint_state_to_sim(
    state["robot/joint_pos"].to(env.device),
    state["robot/joint_vel"].to(env.device),
    env_ids=env_ids,
  )
  for key, target in (
    ("robot/joint_pos_target", robot.data.joint_pos_target),
    ("robot/joint_vel_target", robot.data.joint_vel_target),
    ("robot/joint_effort_target", robot.data.joint_effort_target),
  ):
    target[env_ids] = state[key].to(device=target.device, dtype=target.dtype)
  env.episode_length_buf[env_ids] = state["env/episode_length"].to(
    device=env.device, dtype=env.episode_length_buf.dtype
  )

  _restore_attr(
    state, "action_manager", env.action_manager, _ACTION_MANAGER_ATTRS, env_ids
  )
  action_term = env.action_manager.get_term(action_name)
  _restore_attr(state, "action_term", action_term, _ACTION_TERM_ATTRS, env_ids)
  edge_key = "action_term/selected_edge_top_z"
  if edge_key in state:
    action_term.selected_edge_top_z[env_ids] = state[edge_key].to(env.device) + (
      env.scene.env_origins[env_ids, 2]
    )

  command_term = env.command_manager.get_term(command_name)
  _restore_attr(state, "command", command_term, _COMMAND_ATTRS, env_ids)

  for sensor_name, sensor in env.scene.sensors.items():
    air_state = getattr(sensor, "_air_time_state", None)
    if air_state is not None:
      for name in (
        "current_air_time",
        "last_air_time",
        "current_contact_time",
        "last_contact_time",
      ):
        key = f"sensor/{sensor_name}/air/{name}"
        target = getattr(air_state, name, None)
        if key in state and isinstance(target, torch.Tensor):
          target[env_ids] = state[key].to(target.device, target.dtype)
      # Simulator time keeps advancing while bank entries are stored.  Air-time
      # deltas must restart from the current clock, not the captured clock.
      if getattr(sensor, "_data", None) is not None:
        air_state.last_time[env_ids] = sensor._data.time[env_ids]
    history_state = getattr(sensor, "_history_state", None)
    if isinstance(history_state, dict):
      for name, target in history_state.items():
        key = f"sensor/{sensor_name}/history/{name}"
        if key in state:
          target[env_ids] = state[key].to(target.device, target.dtype)
    sensor._invalidate_cache()

  observation_manager = env.observation_manager
  for group, buffers in observation_manager._group_obs_term_history_buffer.items():
    for term, buffer in buffers.items():
      history_key = f"observation/{group}/{term}/history"
      pushes_key = f"observation/{group}/{term}/pushes"
      if history_key not in state:
        continue
      history = state[history_key].to(buffer.device)
      start = (buffer._pointer + 1) % buffer.max_length
      chronological_indices = (
        torch.arange(buffer.max_length, device=buffer.device) + start
      ) % buffer.max_length
      if buffer._buffer is None:
        raise RuntimeError("destination observation history is not initialized")
      buffer._buffer[chronological_indices[:, None], env_ids[None, :]] = (
        history.transpose(0, 1)
      )
      buffer._num_pushes[env_ids] = state[pushes_key].to(
        buffer.device, buffer._num_pushes.dtype
      )
  observation_manager._obs_buffer = None

  for cfg in env.reward_manager._class_term_cfgs:
    term = cfg.func
    name = term.__class__.__name__
    previous_key = f"reward/{name}/previous_x_relative"
    if previous_key in state:
      term._previous_x[env_ids] = state[previous_key].to(env.device) + (
        env.scene.env_origins[env_ids, 0]
      )
    for attr in ("_initialized", "_previous_index", "_paid"):
      key = f"reward/{name}/{attr}"
      target = getattr(term, attr, None)
      if key in state and isinstance(target, torch.Tensor):
        target[env_ids] = state[key].to(target.device, target.dtype)


@dataclass
class HardCaseEntry:
  state: dict[str, torch.Tensor]
  priority: float
  riser_index: int
  terrain_type: int
  steps_before_fall: int | None = None
  lateral_drift_fraction: float = 0.0
  heading_drift_fraction: float = 0.0
  large_correction_fraction: float = 0.0
  no_subsequent_riser_crossing: bool = False
  failure_type: str = MIXED_FAILURE_TYPE
  outcome: str = "unspecified"
  specialist_mode: str | None = None
  gait_phase: float | None = None
  support_foot: int | None = None
  delivered_command: tuple[float, ...] = ()
  root_velocity: tuple[float, ...] = ()
  cbf_active: bool | None = None
  actor_observation: torch.Tensor | None = None
  balance_bucket: str | None = None
  selection_signal: float = 0.0
  matched_failure_index: int | None = None
  match_distance: float | None = None
  success_pool_index: int | None = None


@dataclass(frozen=True)
class LateFailureCandidate:
  history_index: int
  steps_before_fall: int
  riser_index: int
  lateral_drift_fraction: float
  heading_drift_fraction: float
  large_correction_fraction: float
  priority: float
  no_subsequent_riser_crossing: bool = True
  failure_type: str = MIXED_FAILURE_TYPE


@dataclass(frozen=True)
class SpecialistBankCandidate:
  """One mode-conditioned terminal-history state selected for a bank."""

  history_index: int
  steps_before_terminal: int
  riser_index: int
  gait_phase: float
  support_foot: int
  delivered_command: tuple[float, ...]
  root_velocity: tuple[float, ...]
  cbf_active: bool
  priority: float
  balance_bucket: str
  selection_signal: float
  outcome: str
  failure_type: str


def specialist_history_window(mode: str) -> tuple[int, int]:
  """Return the frozen precursor offset window for one specialist."""
  windows = {
    "lateral": (50, 150),
    "cbf": (10, 50),
    "balance": (20, 100),
  }
  if mode not in windows:
    raise ValueError(f"unsupported specialist mode: {mode!r}")
  return windows[mode]


def _validate_specialist_histories(
  *,
  mode: str,
  riser_history: torch.Tensor,
  component_histories: dict[str, torch.Tensor],
  gait_phase_history: torch.Tensor,
  support_foot_history: torch.Tensor,
  delivered_command_history: torch.Tensor,
  root_velocity_history: torch.Tensor,
  cbf_active_history: torch.Tensor,
) -> int:
  specialist_history_window(mode)
  one_dimensional = (
    riser_history,
    gait_phase_history,
    support_foot_history,
    cbf_active_history,
  )
  if any(value.ndim != 1 for value in one_dimensional):
    raise ValueError("scalar specialist histories must be one-dimensional")
  if delivered_command_history.ndim != 2 or root_velocity_history.ndim != 2:
    raise ValueError("vector specialist histories must be two-dimensional")
  required_components = {
    "lateral": {"centerline", "heading", "edge"},
    "cbf": {"intervention", "nominal_margin"},
    "balance": {
      "roll",
      "pitch",
      "angular_velocity",
      "slip",
      "contact_mismatch",
    },
  }[mode]
  missing = required_components - set(component_histories)
  if missing:
    raise ValueError(f"specialist component histories are missing {sorted(missing)}")
  length = len(riser_history)
  values = (
    *one_dimensional[1:],
    delivered_command_history,
    root_velocity_history,
    *(component_histories[name] for name in required_components),
  )
  if any(len(value) != length for value in values):
    raise ValueError("specialist histories must have equal length")
  if any(value.ndim != 1 for value in component_histories.values()):
    raise ValueError("specialist component histories must be one-dimensional")
  return length


def _specialist_signal(
  mode: str, component_histories: dict[str, torch.Tensor]
) -> torch.Tensor:
  weights = {
    "lateral": {"centerline": 0.45, "heading": 0.35, "edge": 0.20},
    "cbf": {"intervention": 0.60, "nominal_margin": 0.40},
    "balance": {
      "roll": 0.20,
      "pitch": 0.20,
      "angular_velocity": 0.20,
      "slip": 0.20,
      "contact_mismatch": 0.20,
    },
  }[mode]
  return sum(
    weight * component_histories[name] for name, weight in weights.items()
  )


def _specialist_bucket(
  mode: str, riser: int, gait_phase: float, support_foot: int
) -> str:
  phase_bin = min(3, max(0, int(math.floor((gait_phase % 1.0) * 4.0))))
  if mode == "lateral":
    # One bucket per late riser prevents the bank collapsing onto riser 11.
    return f"riser:{riser}"
  if mode == "balance":
    return f"support:{support_foot}/phase:{phase_bin}"
  return f"riser:{riser}/phase:{phase_bin}"


def _candidate_from_history(
  *,
  mode: str,
  index: int,
  length: int,
  riser_history: torch.Tensor,
  signal: torch.Tensor,
  gait_phase_history: torch.Tensor,
  support_foot_history: torch.Tensor,
  delivered_command_history: torch.Tensor,
  root_velocity_history: torch.Tensor,
  cbf_active_history: torch.Tensor,
  outcome: str,
  failure_type: str,
  priority: float,
) -> SpecialistBankCandidate:
  riser = int(riser_history[index])
  phase = float(gait_phase_history[index]) % 1.0
  support = int(support_foot_history[index])
  return SpecialistBankCandidate(
    history_index=index,
    steps_before_terminal=length - 1 - index,
    riser_index=riser,
    gait_phase=phase,
    support_foot=support,
    delivered_command=tuple(
      float(value) for value in delivered_command_history[index]
    ),
    root_velocity=tuple(float(value) for value in root_velocity_history[index]),
    cbf_active=bool(cbf_active_history[index]),
    priority=float(priority),
    balance_bucket=_specialist_bucket(mode, riser, phase, support),
    selection_signal=float(signal[index]),
    outcome=outcome,
    failure_type=failure_type,
  )


def select_specialist_failure_candidates(
  mode: str,
  riser_history: torch.Tensor,
  component_histories: dict[str, torch.Tensor],
  gait_phase_history: torch.Tensor,
  support_foot_history: torch.Tensor,
  delivered_command_history: torch.Tensor,
  root_velocity_history: torch.Tensor,
  cbf_active_history: torch.Tensor,
  *,
  minimum_riser: int,
  maximum_candidates: int = 4,
  failure_type: str | None = None,
) -> tuple[SpecialistBankCandidate, ...]:
  """Select diverse mode-specific precursors from one completed fall.

  Lateral candidates are grouped by riser, CBF candidates by riser/phase, and
  balance candidates by support-foot/phase.  Grouping happens before ranking,
  so a high-priority terminal state cannot fill the whole bank by itself.
  """
  length = _validate_specialist_histories(
    mode=mode,
    riser_history=riser_history,
    component_histories=component_histories,
    gait_phase_history=gait_phase_history,
    support_foot_history=support_foot_history,
    delivered_command_history=delivered_command_history,
    root_velocity_history=root_velocity_history,
    cbf_active_history=cbf_active_history,
  )
  if maximum_candidates < 1:
    raise ValueError("maximum specialist candidates must be positive")
  expected_failure = SPECIALIST_FAILURE_TYPES[mode]
  if failure_type is None:
    failure_type = expected_failure
  if failure_type != expected_failure:
    raise ValueError("specialist candidate failure type does not match its mode")
  minimum_offset, maximum_offset = specialist_history_window(mode)
  maximum_offset = min(maximum_offset, length - 1)
  if maximum_offset < minimum_offset:
    return ()
  signal = _specialist_signal(mode, component_histories)
  best_by_bucket: dict[str, SpecialistBankCandidate] = {}
  for offset in range(minimum_offset, maximum_offset + 1):
    index = length - 1 - offset
    riser = int(riser_history[index])
    if riser < minimum_riser:
      continue
    suffix = signal[index:]
    future_peak = float(suffix.max())
    future_mean = float(suffix.mean())
    growth = max(0.0, future_peak - float(signal[index]))
    # The mode must be visible after the precursor; this filters ordinary late
    # states that happened to precede an unrelated terminal fall.
    minimum_peak = {"lateral": 0.20, "cbf": 0.15, "balance": 0.12}[mode]
    if future_peak < minimum_peak:
      continue
    recency = (maximum_offset - offset) / max(
      1, maximum_offset - minimum_offset
    )
    priority = (
      float(riser)
      + 3.0 * future_peak
      + 2.0 * future_mean
      + 2.0 * growth
      + 0.10 * recency
    )
    candidate = _candidate_from_history(
      mode=mode,
      index=index,
      length=length,
      riser_history=riser_history,
      signal=signal,
      gait_phase_history=gait_phase_history,
      support_foot_history=support_foot_history,
      delivered_command_history=delivered_command_history,
      root_velocity_history=root_velocity_history,
      cbf_active_history=cbf_active_history,
      outcome="failure",
      failure_type=failure_type,
      priority=priority,
    )
    previous = best_by_bucket.get(candidate.balance_bucket)
    if previous is None or candidate.priority > previous.priority:
      best_by_bucket[candidate.balance_bucket] = candidate
  ordered = sorted(
    best_by_bucket.values(),
    key=lambda candidate: (-candidate.priority, candidate.balance_bucket),
  )
  # Round-robin ordering across risers is deterministic and makes the first
  # few entries diverse even before the bounded bank reaches capacity.
  if mode == "lateral":
    ordered = sorted(
      ordered,
      key=lambda candidate: (
        -candidate.riser_index,
        -candidate.priority,
        candidate.balance_bucket,
      ),
    )
  return tuple(ordered[:maximum_candidates])


def select_specialist_success_candidates(
  mode: str,
  riser_history: torch.Tensor,
  component_histories: dict[str, torch.Tensor],
  gait_phase_history: torch.Tensor,
  support_foot_history: torch.Tensor,
  delivered_command_history: torch.Tensor,
  root_velocity_history: torch.Tensor,
  cbf_active_history: torch.Tensor,
  *,
  minimum_riser: int,
  maximum_candidates: int = 8,
) -> tuple[SpecialistBankCandidate, ...]:
  """Select bucket-diverse states across a genuinely successful crossing."""
  length = _validate_specialist_histories(
    mode=mode,
    riser_history=riser_history,
    component_histories=component_histories,
    gait_phase_history=gait_phase_history,
    support_foot_history=support_foot_history,
    delivered_command_history=delivered_command_history,
    root_velocity_history=root_velocity_history,
    cbf_active_history=cbf_active_history,
  )
  if maximum_candidates < 1:
    raise ValueError("maximum specialist candidates must be positive")
  # Failure windows are mode-specific, but successful counterexamples must
  # cover every corresponding late riser.  Exclude only the final few states
  # so replay does not start inside the terminal-success transition.
  final_exclusion_steps = min(10, length)
  stop_index = length - final_exclusion_steps
  if stop_index <= 0:
    return ()
  signal = _specialist_signal(mode, component_histories)
  best_by_bucket: dict[str, SpecialistBankCandidate] = {}
  for index in range(stop_index):
    riser = int(riser_history[index])
    if riser < minimum_riser:
      continue
    phase = float(gait_phase_history[index]) % 1.0
    support = int(support_foot_history[index])
    bucket = _specialist_bucket(mode, riser, phase, support)
    # Prefer informative successful states, but keep priority bounded away
    # from zero so low-cost counterexamples remain sampleable.
    priority = 1.0 + float(signal[index]) + 0.05 * (
      index / max(1, stop_index - 1)
    )
    candidate = _candidate_from_history(
      mode=mode,
      index=index,
      length=length,
      riser_history=riser_history,
      signal=signal,
      gait_phase_history=gait_phase_history,
      support_foot_history=support_foot_history,
      delivered_command_history=delivered_command_history,
      root_velocity_history=root_velocity_history,
      cbf_active_history=cbf_active_history,
      outcome="success",
      failure_type=MIXED_FAILURE_TYPE,
      priority=priority,
    )
    previous = best_by_bucket.get(bucket)
    if previous is None or candidate.priority > previous.priority:
      best_by_bucket[bucket] = candidate
  if not best_by_bucket:
    return ()
  if mode == "lateral":
    ordered = sorted(
      best_by_bucket.values(),
      key=lambda candidate: (
        candidate.riser_index,
        -candidate.priority,
        candidate.balance_bucket,
      ),
    )
  elif mode == "balance":
    ordered = sorted(
      best_by_bucket.values(),
      key=lambda candidate: (
        min(3, max(0, int(math.floor(candidate.gait_phase * 4.0)))),
        candidate.support_foot,
        -candidate.priority,
      ),
    )
  else:
    by_riser: dict[int, list[SpecialistBankCandidate]] = {}
    for candidate in best_by_bucket.values():
      by_riser.setdefault(candidate.riser_index, []).append(candidate)
    for candidates in by_riser.values():
      candidates.sort(
        key=lambda candidate: (-candidate.priority, candidate.balance_bucket)
      )
    ordered = []
    for rank in range(max(len(candidates) for candidates in by_riser.values())):
      for riser in sorted(by_riser):
        if rank < len(by_riser[riser]):
          ordered.append(by_riser[riser][rank])
  return tuple(ordered[:maximum_candidates])


def select_late_failure_candidate(
  riser_history: torch.Tensor,
  centerline_error_history: torch.Tensor,
  heading_error_history: torch.Tensor,
  correction_history: torch.Tensor,
  *,
  minimum_steps_before_fall: int = 50,
  maximum_steps_before_fall: int = 150,
  minimum_riser: int = 5,
  lateral_threshold: float = 0.18,
  heading_threshold: float = 0.25,
  correction_threshold: float = 0.01,
  failure_type: str = MIXED_FAILURE_TYPE,
) -> LateFailureCandidate | None:
  """Choose one late-riser state from the window preceding a target fall."""
  if (
    failure_type != MIXED_FAILURE_TYPE
    and failure_type not in TARGET_FAILURE_TYPES
  ):
    raise ValueError(f"unsupported target failure type: {failure_type}")
  tensors = (
    riser_history,
    centerline_error_history,
    heading_error_history,
    correction_history,
  )
  if any(tensor.ndim != 1 for tensor in tensors):
    raise ValueError("late-failure histories must be one-dimensional")
  if len({len(tensor) for tensor in tensors}) != 1:
    raise ValueError("late-failure histories must have equal length")
  if minimum_steps_before_fall < 1:
    raise ValueError("minimum pre-fall offset must be positive")
  if maximum_steps_before_fall < minimum_steps_before_fall:
    raise ValueError("maximum pre-fall offset is below minimum")
  candidates: list[LateFailureCandidate] = []
  maximum_offset = min(maximum_steps_before_fall, len(riser_history) - 1)
  for offset in range(minimum_steps_before_fall, maximum_offset + 1):
    index = len(riser_history) - 1 - offset
    riser = int(riser_history[index])
    if riser < minimum_riser:
      continue
    suffix = slice(index, len(riser_history))
    # A state followed by another normal riser crossing was not the boundary
    # that decided this fall. Excluding it also prevents ordinary CBF
    # interventions on otherwise successful stair transitions entering the
    # failure-focused protocols.
    if bool(torch.any(riser_history[index + 1 :] > riser)):
      continue
    lateral_fraction = float(
      (centerline_error_history[suffix].abs() >= lateral_threshold).float().mean()
    )
    heading_fraction = float(
      (heading_error_history[suffix].abs() >= heading_threshold).float().mean()
    )
    correction_fraction = float(
      (correction_history[suffix] >= correction_threshold).float().mean()
    )
    # Later risers dominate. Persistent drift/correction identifies which
    # late boundary is most causally informative; a small recency term breaks
    # otherwise equal candidates without moving outside the 50--150 window.
    priority = (
      float(riser)
      + 2.0 * lateral_fraction
      + 2.0 * heading_fraction
      + 2.0 * correction_fraction
      + 0.1 * (maximum_steps_before_fall - offset)
      / max(1, maximum_steps_before_fall - minimum_steps_before_fall)
    )
    candidates.append(
      LateFailureCandidate(
        history_index=index,
        steps_before_fall=offset,
        riser_index=riser,
        lateral_drift_fraction=lateral_fraction,
        heading_drift_fraction=heading_fraction,
        large_correction_fraction=correction_fraction,
        priority=priority,
        no_subsequent_riser_crossing=True,
        failure_type=failure_type,
      )
    )
  return max(candidates, key=lambda candidate: candidate.priority, default=None)


class HardCaseStateBank:
  """Bounded priority bank of pre-intervention simulator states."""

  def __init__(
    self,
    capacity: int = 256,
    *,
    bank_kind: str = "general_intervention",
    source_domain: str | None = None,
    context_sha256: str | None = None,
    dominant_failure_type: str | None = None,
    specialist_mode: str | None = None,
  ) -> None:
    if capacity < 1:
      raise ValueError("hard-case capacity must be positive")
    self.capacity = capacity
    self.bank_kind = bank_kind
    self.source_domain = source_domain
    self.context_sha256 = context_sha256
    if (
      dominant_failure_type is not None
      and dominant_failure_type not in TARGET_FAILURE_TYPES
    ):
      raise ValueError(
        f"unsupported dominant target failure type: {dominant_failure_type}"
      )
    self.dominant_failure_type = dominant_failure_type
    if specialist_mode is not None and specialist_mode not in SPECIALIST_MODES:
      raise ValueError(f"unsupported specialist mode: {specialist_mode!r}")
    if bank_kind in SPECIALIST_BANK_KINDS and specialist_mode is None:
      raise ValueError("a specialist bank requires specialist_mode")
    if bank_kind not in SPECIALIST_BANK_KINDS and specialist_mode is not None:
      raise ValueError("specialist_mode is valid only for a specialist bank")
    self.specialist_mode = specialist_mode
    self.entries: list[HardCaseEntry] = []
    self.total_added = 0

  def __len__(self) -> int:
    return len(self.entries)

  def clear(self) -> int:
    removed = len(self.entries)
    self.entries.clear()
    self.total_added = 0
    return removed

  def add_batched(
    self,
    state: dict[str, torch.Tensor],
    env_ids: torch.Tensor,
    priorities: torch.Tensor,
    riser_indices: torch.Tensor,
  ) -> int:
    if self.dominant_failure_type is not None:
      raise ValueError(
        "a dominant-failure bank accepts only labeled late-failure entries"
      )
    if not (
      env_ids.ndim == priorities.ndim == riser_indices.ndim == 1
      and len(env_ids) == len(priorities) == len(riser_indices)
    ):
      raise ValueError("hard-case ids, priorities, and riser indices must align")
    added = 0
    for row, env_id in enumerate(env_ids.tolist()):
      selected = torch.tensor([env_id], device=env_ids.device)
      individual = {
        key: value.index_select(0, selected).detach().cpu()
        for key, value in state.items()
      }
      terrain_type = int(individual.get("terrain/type", torch.tensor([-1]))[0])
      entry = HardCaseEntry(
        state=individual,
        priority=float(priorities[row]),
        riser_index=int(riser_indices[row]),
        terrain_type=terrain_type,
      )
      if len(self.entries) < self.capacity:
        self.entries.append(entry)
        added += 1
      else:
        minimum = min(range(len(self.entries)), key=lambda i: self.entries[i].priority)
        if entry.priority > self.entries[minimum].priority:
          self.entries[minimum] = entry
          added += 1
      self.total_added += 1
    return added

  def add_late_failure(
    self,
    state: dict[str, torch.Tensor],
    env_id: int,
    candidate: LateFailureCandidate,
  ) -> int:
    """Insert one state selected only after its target episode ended in a fall."""
    if self.bank_kind != "target_late_failure":
      raise ValueError("late failures require a target_late_failure bank")
    if (
      self.dominant_failure_type is not None
      and candidate.failure_type != self.dominant_failure_type
    ):
      raise ValueError(
        "late-failure candidate does not match the frozen dominant failure "
        f"type: {candidate.failure_type} != {self.dominant_failure_type}"
      )
    selected = torch.tensor([env_id], device=next(iter(state.values())).device)
    individual = {
      key: value.index_select(0, selected).detach().cpu()
      for key, value in state.items()
    }
    terrain_type = int(individual.get("terrain/type", torch.tensor([-1]))[0])
    entry = HardCaseEntry(
      state=individual,
      priority=candidate.priority,
      riser_index=candidate.riser_index,
      terrain_type=terrain_type,
      steps_before_fall=candidate.steps_before_fall,
      lateral_drift_fraction=candidate.lateral_drift_fraction,
      heading_drift_fraction=candidate.heading_drift_fraction,
      large_correction_fraction=candidate.large_correction_fraction,
      no_subsequent_riser_crossing=candidate.no_subsequent_riser_crossing,
      failure_type=candidate.failure_type,
    )
    added = 0
    if len(self.entries) < self.capacity:
      self.entries.append(entry)
      added = 1
    else:
      minimum = min(range(len(self.entries)), key=lambda i: self.entries[i].priority)
      if entry.priority > self.entries[minimum].priority:
        self.entries[minimum] = entry
        added = 1
    self.total_added += 1
    return added

  def _insert_entry(self, entry: HardCaseEntry) -> int:
    """Insert by priority while preserving specialist bucket coverage."""
    if len(self.entries) < self.capacity:
      self.entries.append(entry)
      self.total_added += 1
      return 1
    replacement_candidates = list(range(len(self.entries)))
    if entry.balance_bucket is not None:
      same_bucket = [
        index
        for index, existing in enumerate(self.entries)
        if existing.balance_bucket == entry.balance_bucket
      ]
      if same_bucket:
        replacement_candidates = same_bucket
      else:
        counts: dict[str | None, int] = {}
        for existing in self.entries:
          counts[existing.balance_bucket] = (
            counts.get(existing.balance_bucket, 0) + 1
          )
        largest = max(counts.values())
        overrepresented = {
          bucket for bucket, count in counts.items() if count == largest
        }
        replacement_candidates = [
          index
          for index, existing in enumerate(self.entries)
          if existing.balance_bucket in overrepresented
        ]
    minimum = min(
      replacement_candidates, key=lambda index: self.entries[index].priority
    )
    added = 0
    if (
      entry.balance_bucket != self.entries[minimum].balance_bucket
      or entry.priority > self.entries[minimum].priority
    ):
      self.entries[minimum] = entry
      added = 1
    self.total_added += 1
    return added

  def add_specialist_candidate(
    self,
    state: dict[str, torch.Tensor],
    env_id: int,
    candidate: SpecialistBankCandidate,
    actor_observation: torch.Tensor,
  ) -> int:
    """Insert one failure precursor or successful source-pool state."""
    if self.bank_kind not in (
      SPECIALIST_FAILURE_BANK_KIND,
      SPECIALIST_SUCCESS_POOL_KIND,
    ):
      raise ValueError("candidate insertion requires a specialist source bank")
    if candidate.outcome == "failure":
      expected_kind = SPECIALIST_FAILURE_BANK_KIND
      expected_failure = SPECIALIST_FAILURE_TYPES[self.specialist_mode]
      if candidate.failure_type != expected_failure:
        raise ValueError("failure candidate does not match specialist mode")
    elif candidate.outcome == "success":
      expected_kind = SPECIALIST_SUCCESS_POOL_KIND
    else:
      raise ValueError("specialist candidate outcome must be failure or success")
    if self.bank_kind != expected_kind:
      raise ValueError("specialist candidate outcome does not match bank kind")
    if actor_observation.ndim != 1:
      raise ValueError("one specialist actor observation must be one-dimensional")
    selected = torch.tensor([env_id], device=next(iter(state.values())).device)
    individual = {
      key: value.index_select(0, selected).detach().cpu()
      for key, value in state.items()
    }
    terrain_type = int(individual.get("terrain/type", torch.tensor([-1]))[0])
    entry = HardCaseEntry(
      state=individual,
      priority=candidate.priority,
      riser_index=candidate.riser_index,
      terrain_type=terrain_type,
      steps_before_fall=(
        candidate.steps_before_terminal
        if candidate.outcome == "failure"
        else None
      ),
      no_subsequent_riser_crossing=candidate.outcome == "failure",
      failure_type=candidate.failure_type,
      outcome=candidate.outcome,
      specialist_mode=self.specialist_mode,
      gait_phase=candidate.gait_phase,
      support_foot=candidate.support_foot,
      delivered_command=candidate.delivered_command,
      root_velocity=candidate.root_velocity,
      cbf_active=candidate.cbf_active,
      actor_observation=actor_observation.detach().cpu().clone(),
      balance_bucket=candidate.balance_bucket,
      selection_signal=candidate.selection_signal,
    )
    return self._insert_entry(entry)

  def add_matched_success(
    self,
    source: HardCaseEntry,
    *,
    failure_index: int,
    success_pool_index: int,
    distance: float,
  ) -> int:
    """Copy one successful source state into the replay counterexample bank."""
    if self.bank_kind != SPECIALIST_SUCCESS_BANK_KIND:
      raise ValueError("matched success requires a counterexample bank")
    if source.outcome != "success" or source.specialist_mode != self.specialist_mode:
      raise ValueError("matched source is not a success from the same specialist")
    if not math.isfinite(distance) or distance < 0.0:
      raise ValueError("success match distance must be finite and non-negative")
    entry = HardCaseEntry(
      state={key: value.detach().clone() for key, value in source.state.items()},
      priority=max(1.0e-6, 1.0 / (1.0 + distance)),
      riser_index=source.riser_index,
      terrain_type=source.terrain_type,
      failure_type=MIXED_FAILURE_TYPE,
      outcome="success",
      specialist_mode=source.specialist_mode,
      gait_phase=source.gait_phase,
      support_foot=source.support_foot,
      delivered_command=source.delivered_command,
      root_velocity=source.root_velocity,
      cbf_active=source.cbf_active,
      actor_observation=(
        None
        if source.actor_observation is None
        else source.actor_observation.detach().clone()
      ),
      balance_bucket=source.balance_bucket,
      selection_signal=source.selection_signal,
      matched_failure_index=failure_index,
      match_distance=distance,
      success_pool_index=success_pool_index,
    )
    return self._insert_entry(entry)

  def sample(
    self,
    count: int,
    *,
    device: str | torch.device,
    generator: torch.Generator | None = None,
  ) -> dict[str, torch.Tensor]:
    if count < 0:
      raise ValueError("sample count must be non-negative")
    if count == 0:
      return {}
    if not self.entries:
      raise RuntimeError("cannot sample an empty hard-case bank")
    priorities = torch.tensor(
      [max(entry.priority, 1.0e-6) for entry in self.entries], dtype=torch.float64
    )
    indices = torch.multinomial(
      priorities,
      count,
      replacement=count > len(self.entries),
      generator=generator,
    )
    selected = [self.entries[index] for index in indices.tolist()]
    keys = selected[0].state.keys()
    return {
      key: torch.cat([entry.state[key] for entry in selected], dim=0).to(device)
      for key in keys
    }

  def state_dict(self) -> dict[str, Any]:
    return {
      "capacity": self.capacity,
      "total_added": self.total_added,
      "bank_kind": self.bank_kind,
      "source_domain": self.source_domain,
      "context_sha256": self.context_sha256,
      "dominant_failure_type": self.dominant_failure_type,
      "specialist_mode": self.specialist_mode,
      "entries": [
        {
          "state": entry.state,
          "priority": entry.priority,
          "riser_index": entry.riser_index,
          "terrain_type": entry.terrain_type,
          "steps_before_fall": entry.steps_before_fall,
          "lateral_drift_fraction": entry.lateral_drift_fraction,
          "heading_drift_fraction": entry.heading_drift_fraction,
          "large_correction_fraction": entry.large_correction_fraction,
          "no_subsequent_riser_crossing": entry.no_subsequent_riser_crossing,
          "failure_type": entry.failure_type,
          "outcome": entry.outcome,
          "specialist_mode": entry.specialist_mode,
          "gait_phase": entry.gait_phase,
          "support_foot": entry.support_foot,
          "delivered_command": list(entry.delivered_command),
          "root_velocity": list(entry.root_velocity),
          "cbf_active": entry.cbf_active,
          "actor_observation": entry.actor_observation,
          "balance_bucket": entry.balance_bucket,
          "selection_signal": entry.selection_signal,
          "matched_failure_index": entry.matched_failure_index,
          "match_distance": entry.match_distance,
          "success_pool_index": entry.success_pool_index,
        }
        for entry in self.entries
      ],
    }

  def load_state_dict(self, payload: dict[str, Any]) -> None:
    self.capacity = int(payload["capacity"])
    self.total_added = int(payload.get("total_added", 0))
    self.bank_kind = str(payload.get("bank_kind", "general_intervention"))
    self.source_domain = payload.get("source_domain")
    self.context_sha256 = payload.get("context_sha256")
    self.dominant_failure_type = payload.get("dominant_failure_type")
    self.specialist_mode = payload.get("specialist_mode")
    if (
      self.dominant_failure_type is not None
      and self.dominant_failure_type not in TARGET_FAILURE_TYPES
    ):
      raise ValueError("serialized bank has an unsupported dominant failure type")
    if self.specialist_mode is not None and self.specialist_mode not in SPECIALIST_MODES:
      raise ValueError("serialized bank has an unsupported specialist mode")
    if self.bank_kind in SPECIALIST_BANK_KINDS and self.specialist_mode is None:
      raise ValueError("serialized specialist bank has no specialist mode")
    self.entries = [
      HardCaseEntry(
        state=item["state"],
        priority=float(item["priority"]),
        riser_index=int(item["riser_index"]),
        terrain_type=int(item.get("terrain_type", -1)),
        steps_before_fall=(
          None
          if item.get("steps_before_fall") is None
          else int(item["steps_before_fall"])
        ),
        lateral_drift_fraction=float(item.get("lateral_drift_fraction", 0.0)),
        heading_drift_fraction=float(item.get("heading_drift_fraction", 0.0)),
        large_correction_fraction=float(
          item.get("large_correction_fraction", 0.0)
        ),
        no_subsequent_riser_crossing=bool(
          item.get("no_subsequent_riser_crossing", False)
        ),
        failure_type=str(item.get("failure_type", MIXED_FAILURE_TYPE)),
        outcome=str(item.get("outcome", "unspecified")),
        specialist_mode=item.get("specialist_mode"),
        gait_phase=(
          None if item.get("gait_phase") is None else float(item["gait_phase"])
        ),
        support_foot=(
          None if item.get("support_foot") is None else int(item["support_foot"])
        ),
        delivered_command=tuple(
          float(value) for value in item.get("delivered_command", ())
        ),
        root_velocity=tuple(
          float(value) for value in item.get("root_velocity", ())
        ),
        cbf_active=(
          None if item.get("cbf_active") is None else bool(item["cbf_active"])
        ),
        actor_observation=(
          None
          if item.get("actor_observation") is None
          else item["actor_observation"].detach().cpu().clone()
        ),
        balance_bucket=item.get("balance_bucket"),
        selection_signal=float(item.get("selection_signal", 0.0)),
        matched_failure_index=(
          None
          if item.get("matched_failure_index") is None
          else int(item["matched_failure_index"])
        ),
        match_distance=(
          None
          if item.get("match_distance") is None
          else float(item["match_distance"])
        ),
        success_pool_index=(
          None
          if item.get("success_pool_index") is None
          else int(item["success_pool_index"])
        ),
      )
      for item in payload.get("entries", [])
    ]
    if len(self.entries) > self.capacity:
      raise ValueError("serialized hard-case bank exceeds capacity")
    if self.dominant_failure_type is not None and any(
      entry.failure_type != self.dominant_failure_type
      for entry in self.entries
    ):
      raise ValueError(
        "serialized bank contains entries outside its dominant failure type"
      )
    if self.specialist_mode is not None and any(
      entry.specialist_mode != self.specialist_mode for entry in self.entries
    ):
      raise ValueError("serialized specialist bank contains a foreign-mode entry")

  def audit_metadata(self) -> dict[str, Any]:
    late_entries = [
      entry for entry in self.entries if entry.steps_before_fall is not None
    ]
    failure_type_counts = {
      failure_type: sum(
        entry.failure_type == failure_type for entry in late_entries
      )
      for failure_type in sorted({entry.failure_type for entry in late_entries})
    }
    return {
      "bank_kind": self.bank_kind,
      "source_domain": self.source_domain,
      "context_sha256": self.context_sha256,
      "dominant_failure_type": self.dominant_failure_type,
      "specialist_mode": self.specialist_mode,
      "failure_type_counts": failure_type_counts,
      "dominant_failure_type_purity_passed": (
        self.dominant_failure_type is None
        or all(
          entry.failure_type == self.dominant_failure_type
          for entry in late_entries
        )
      ),
      "size": len(self.entries),
      "capacity": self.capacity,
      "total_added": self.total_added,
      "late_failure_entry_count": len(late_entries),
      "steps_before_fall_min": (
        min(entry.steps_before_fall for entry in late_entries)
        if late_entries
        else None
      ),
      "steps_before_fall_max": (
        max(entry.steps_before_fall for entry in late_entries)
        if late_entries
        else None
      ),
      "riser_index_min": (
        min(entry.riser_index for entry in late_entries) if late_entries else None
      ),
      "riser_index_max": (
        max(entry.riser_index for entry in late_entries) if late_entries else None
      ),
      "mean_lateral_drift_fraction": (
        sum(entry.lateral_drift_fraction for entry in late_entries)
        / len(late_entries)
        if late_entries
        else 0.0
      ),
      "mean_heading_drift_fraction": (
        sum(entry.heading_drift_fraction for entry in late_entries)
        / len(late_entries)
        if late_entries
        else 0.0
      ),
      "mean_large_correction_fraction": (
        sum(entry.large_correction_fraction for entry in late_entries)
        / len(late_entries)
        if late_entries
        else 0.0
      ),
      "successful_crossing_exclusion_passed": all(
        entry.no_subsequent_riser_crossing for entry in late_entries
      ),
      "outcome_counts": {
        outcome: sum(entry.outcome == outcome for entry in self.entries)
        for outcome in sorted({entry.outcome for entry in self.entries})
      },
      "balance_bucket_counts": {
        bucket: sum(entry.balance_bucket == bucket for entry in self.entries)
        for bucket in sorted(
          {
            entry.balance_bucket
            for entry in self.entries
            if entry.balance_bucket is not None
          }
        )
      },
      "riser_index_counts": {
        str(riser): sum(entry.riser_index == riser for entry in self.entries)
        for riser in sorted({entry.riser_index for entry in self.entries})
      },
      "support_foot_counts": {
        str(support): sum(entry.support_foot == support for entry in self.entries)
        for support in sorted(
          {
            entry.support_foot
            for entry in self.entries
            if entry.support_foot is not None
          }
        )
      },
      "matched_entry_count": sum(
        entry.matched_failure_index is not None for entry in self.entries
      ),
      "unique_success_pool_source_count": len(
        {
          entry.success_pool_index
          for entry in self.entries
          if entry.success_pool_index is not None
        }
      ),
      "mean_match_distance": (
        sum(
          entry.match_distance
          for entry in self.entries
          if entry.match_distance is not None
        )
        / max(1, sum(entry.match_distance is not None for entry in self.entries))
      ),
    }


def _specialist_entry_distance(
  failure: HardCaseEntry, success: HardCaseEntry
) -> float:
  if failure.specialist_mode != success.specialist_mode:
    raise ValueError("cannot match entries from different specialist modes")
  if failure.actor_observation is None or success.actor_observation is None:
    raise ValueError("specialist matching requires actor observations")
  if failure.actor_observation.shape != success.actor_observation.shape:
    raise ValueError("specialist actor-observation shapes do not match")
  observation_distance = float(
    torch.sqrt(
      torch.mean(
        (
          failure.actor_observation.float()
          - success.actor_observation.float()
        ).square()
      )
    )
  )
  phase_delta = abs(float(failure.gait_phase) - float(success.gait_phase))
  phase_delta = min(phase_delta, 1.0 - phase_delta)
  command_a = torch.tensor(failure.delivered_command, dtype=torch.float64)
  command_b = torch.tensor(success.delivered_command, dtype=torch.float64)
  velocity_a = torch.tensor(failure.root_velocity, dtype=torch.float64)
  velocity_b = torch.tensor(success.root_velocity, dtype=torch.float64)
  if command_a.shape != command_b.shape or velocity_a.shape != velocity_b.shape:
    raise ValueError("specialist match-feature shapes do not agree")
  command_distance = float(torch.linalg.vector_norm(command_a - command_b))
  velocity_distance = float(torch.linalg.vector_norm(velocity_a - velocity_b))
  return (
    observation_distance
    + 1.50 * abs(failure.riser_index - success.riser_index)
    + 1.00 * phase_delta
    + 0.75 * (failure.support_foot != success.support_foot)
    + 0.50 * command_distance
    + 0.35 * velocity_distance
    + 0.50 * (failure.cbf_active != success.cbf_active)
  )


def match_specialist_success_counterexamples(
  failure_bank: HardCaseStateBank,
  success_pool: HardCaseStateBank,
  success_bank: HardCaseStateBank,
) -> dict[str, Any]:
  """Build a matched success bank using frozen bucketed actor-state features."""
  mode = failure_bank.specialist_mode
  if (
    failure_bank.bank_kind != SPECIALIST_FAILURE_BANK_KIND
    or success_pool.bank_kind != SPECIALIST_SUCCESS_POOL_KIND
    or success_bank.bank_kind != SPECIALIST_SUCCESS_BANK_KIND
    or mode is None
    or success_pool.specialist_mode != mode
    or success_bank.specialist_mode != mode
  ):
    raise ValueError("specialist match banks have incompatible roles or modes")
  if not failure_bank.entries:
    raise RuntimeError("cannot match an empty failure-precursor bank")
  if not success_pool.entries:
    raise RuntimeError("cannot match without successful source states")
  success_bank.clear()
  used_sources: set[int] = set()
  matches: list[dict[str, Any]] = []
  ordered_failures = sorted(
    enumerate(failure_bank.entries),
    key=lambda item: (-item[1].priority, item[0]),
  )
  for failure_index, failure in ordered_failures[: success_bank.capacity]:
    distances = [
      (_specialist_entry_distance(failure, success), success_index)
      for success_index, success in enumerate(success_pool.entries)
    ]
    unused = [item for item in distances if item[1] not in used_sources]
    distance, success_index = min(unused or distances)
    source = success_pool.entries[success_index]
    added = success_bank.add_matched_success(
      source,
      failure_index=failure_index,
      success_pool_index=success_index,
      distance=distance,
    )
    if added != 1:
      raise RuntimeError("matched success bank rejected a required counterexample")
    used_sources.add(success_index)
    matches.append(
      {
        "failure_index": failure_index,
        "success_pool_index": success_index,
        "distance": distance,
        "failure_riser": failure.riser_index,
        "success_riser": source.riser_index,
        "failure_bucket": failure.balance_bucket,
        "success_bucket": source.balance_bucket,
      }
    )
  return {
    "specialist_mode": mode,
    "failure_entry_count": len(failure_bank),
    "success_pool_entry_count": len(success_pool),
    "matched_entry_count": len(success_bank),
    "unique_success_source_count": len(used_sources),
    "one_match_per_replayed_failure": len(success_bank)
    == min(len(failure_bank), success_bank.capacity),
    "matches": matches,
  }


def specialist_destination_ids(
  num_envs: int,
  *,
  failure_fraction: float,
  success_fraction: float,
  device: str | torch.device,
  generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Select disjoint failure and matched-success replay slots."""
  fractions = torch.tensor(
    (failure_fraction, success_fraction), dtype=torch.float64
  )
  if not bool(torch.isfinite(fractions).all()):
    raise ValueError("specialist replay fractions must be finite")
  if min(failure_fraction, success_fraction) < 0.0:
    raise ValueError("specialist replay fractions must be non-negative")
  if failure_fraction + success_fraction > 1.0 + 1.0e-12:
    raise ValueError("specialist replay fractions exceed one")
  if num_envs < 0:
    raise ValueError("num_envs must be non-negative")
  failure_count = min(num_envs, int(round(num_envs * failure_fraction)))
  success_count = min(
    num_envs - failure_count, int(round(num_envs * success_fraction))
  )
  permutation = torch.randperm(num_envs, generator=generator)
  return (
    permutation[:failure_count].to(device=device),
    permutation[failure_count : failure_count + success_count].to(device=device),
  )


def reset_rollout_with_specialist_banks(
  vec_env,
  failure_bank: HardCaseStateBank,
  success_bank: HardCaseStateBank,
  *,
  failure_fraction: float = 0.15,
  success_fraction: float = 0.15,
  generator: torch.Generator | None = None,
):
  """Create one on-policy 70/15/15 target rollout start mixture."""
  if (
    failure_bank.bank_kind != SPECIALIST_FAILURE_BANK_KIND
    or success_bank.bank_kind != SPECIALIST_SUCCESS_BANK_KIND
    or failure_bank.specialist_mode != success_bank.specialist_mode
  ):
    raise ValueError("specialist replay banks have incompatible roles or modes")
  env = vec_env.unwrapped
  all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  env._reset_idx(all_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  env.observation_manager.compute(update_history=True)
  requested_failure, requested_success = specialist_destination_ids(
    env.num_envs,
    failure_fraction=failure_fraction,
    success_fraction=success_fraction,
    device=env.device,
    generator=generator,
  )
  failure_count = min(len(requested_failure), len(failure_bank))
  success_count = min(len(requested_success), len(success_bank))
  failure_ids = requested_failure[:failure_count]
  success_ids = requested_success[:success_count]
  shape_mismatches: dict[str, list[str]] = {"failure": [], "success": []}
  incompatible_counts = {"failure": 0, "success": 0}
  current = capture_hard_case_state(env)
  sampled_failure = (
    failure_bank.sample(failure_count, device=env.device, generator=generator)
    if failure_count
    else {}
  )
  sampled_success = (
    success_bank.sample(success_count, device=env.device, generator=generator)
    if success_count
    else {}
  )
  for label, bank, sampled, ids in (
    ("failure", failure_bank, sampled_failure, failure_ids),
    ("success", success_bank, sampled_success, success_ids),
  ):
    if not sampled:
      continue
    shape_mismatches[label] = hard_case_state_shape_mismatches(current, sampled)
    if shape_mismatches[label]:
      incompatible_counts[label] = bank.clear()
      if label == "failure":
        failure_count = 0
        failure_ids = requested_failure[:0]
      else:
        success_count = 0
        success_ids = requested_success[:0]
      continue
    restore_hard_case_state(env, sampled, ids)
  if failure_count or success_count:
    env.scene.write_data_to_sim()
    env.sim.forward()
    for sensor in env.scene.sensors.values():
      sensor._invalidate_cache()
    env.sim.sense()
  env.observation_manager._obs_buffer = None
  obs = env.observation_manager.compute(update_history=False)
  env.obs_buf = obs
  realized_normal = env.num_envs - failure_count - success_count
  return vec_env.get_observations(), {
    "specialist_mode": failure_bank.specialist_mode,
    "requested_mixture": {
      "normal": 1.0 - failure_fraction - success_fraction,
      "failure": failure_fraction,
      "success": success_fraction,
    },
    "failure_bank_size": len(failure_bank),
    "success_bank_size": len(success_bank),
    "failure_start_requested_count": len(requested_failure),
    "success_start_requested_count": len(requested_success),
    "failure_start_count": failure_count,
    "success_start_count": success_count,
    "normal_start_count": realized_normal,
    "failure_start_fraction": failure_count / max(1, env.num_envs),
    "success_start_fraction": success_count / max(1, env.num_envs),
    "normal_start_fraction": realized_normal / max(1, env.num_envs),
    "failure_start_ids": failure_ids.detach().cpu().tolist(),
    "success_start_ids": success_ids.detach().cpu().tolist(),
    "incompatible_bank_dropped_counts": incompatible_counts,
    "bank_shape_mismatches": shape_mismatches,
  }


def hard_case_destination_ids(
  num_envs: int,
  fraction: float,
  *,
  device: str | torch.device,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """Select a reproducible subset of environments for hard-case starts."""
  hard_ids, _ = curriculum_destination_ids(
    num_envs,
    hard_case_fraction=fraction,
    neighbor_command_fraction=0.0,
    device=device,
    generator=generator,
  )
  return hard_ids


def curriculum_destination_ids(
  num_envs: int,
  *,
  hard_case_fraction: float,
  neighbor_command_fraction: float,
  device: str | torch.device,
  generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Select disjoint hard-case and neighboring-command start subsets.

  The remaining environments keep their ordinary bottom reset.  Sampling is
  performed on CPU so the persisted curriculum generator is independent of
  CUDA graph capture and exactly reproducible across resumed online rounds.
  """
  fractions = torch.tensor(
    [hard_case_fraction, neighbor_command_fraction], dtype=torch.float64
  )
  if not bool(torch.isfinite(fractions).all()):
    raise ValueError("curriculum fractions must be finite")
  if hard_case_fraction < 0.0 or neighbor_command_fraction < 0.0:
    raise ValueError("curriculum fractions must be non-negative")
  if hard_case_fraction + neighbor_command_fraction > 1.0 + 1.0e-12:
    raise ValueError("hard-case and neighboring-command fractions exceed one")
  if num_envs < 0:
    raise ValueError("num_envs must be non-negative")

  hard_count = min(num_envs, int(round(num_envs * hard_case_fraction)))
  neighbor_count = min(
    num_envs - hard_count,
    int(round(num_envs * neighbor_command_fraction)),
  )
  permutation = torch.randperm(num_envs, generator=generator)
  hard_ids = permutation[:hard_count].to(device=device)
  neighbor_ids = permutation[
    hard_count : hard_count + neighbor_count
  ].to(device=device)
  return hard_ids, neighbor_ids


def perturb_joystick_command_state(
  env,
  env_ids: torch.Tensor,
  *,
  generator: torch.Generator | None = None,
  command_name: str = "twist",
  forward_scale_range: tuple[float, float] = (0.90, 1.10),
  delay_step_offset_range: tuple[int, int] = (-2, 2),
) -> dict[str, float]:
  """Apply a bounded neighboring command condition after a bottom reset.

  Geometry remains the fixed deployment target.  Only the operator trace is
  moved locally by scaling the held forward stick and shifting its delivery
  delay.  The actor still observes only the delivered command, and all future
  actions are freshly sampled by the behavior policy, preserving on-policy
  rollouts under the changed initial-state distribution.
  """
  if env_ids.ndim != 1:
    raise ValueError("neighbor command env_ids must be one-dimensional")
  low_scale, high_scale = forward_scale_range
  low_delay, high_delay = delay_step_offset_range
  if not 0.0 < low_scale <= high_scale:
    raise ValueError("neighbor forward scale range must be positive and ordered")
  if low_delay > high_delay:
    raise ValueError("neighbor delay offset range must be ordered")
  if len(env_ids) == 0:
    return {
      "neighbor_forward_scale_mean": 1.0,
      "neighbor_delay_step_offset_mean": 0.0,
    }

  command = env.command_manager.get_term(command_name)
  required = ("raw_command", "delivered_command", "command_derivative", "delay_steps")
  missing = [
    name
    for name in required
    if not isinstance(getattr(command, name, None), torch.Tensor)
  ]
  if missing:
    raise TypeError(
      f"neighbor command curriculum requires joystick state tensors: {missing}"
    )
  scales_cpu = low_scale + (high_scale - low_scale) * torch.rand(
    len(env_ids), generator=generator
  )
  offsets_cpu = torch.randint(
    low_delay,
    high_delay + 1,
    (len(env_ids),),
    generator=generator,
  )
  scales = scales_cpu.to(command.raw_command.device)
  offsets = offsets_cpu.to(command.delay_steps.device)
  command.raw_command[env_ids, 0] *= scales
  maximum_delay = int(getattr(command, "_max_delay_steps", 0))
  command.delay_steps[env_ids] = (
    command.delay_steps[env_ids] + offsets
  ).clamp(0, maximum_delay)
  command.delivered_command[env_ids] = 0.0
  command.command_derivative[env_ids] = 0.0
  delay_queue = getattr(command, "_delay_queue", None)
  if isinstance(delay_queue, torch.Tensor):
    delay_queue[env_ids] = 0.0
  return {
    "neighbor_forward_scale_mean": float(scales_cpu.mean()),
    "neighbor_delay_step_offset_mean": float(offsets_cpu.float().mean()),
  }


def reset_rollout_with_hard_cases(
  vec_env,
  bank: HardCaseStateBank,
  *,
  hard_case_fraction: float,
  neighbor_command_fraction: float = 0.0,
  neighbor_forward_scale_range: tuple[float, float] = (0.90, 1.10),
  neighbor_delay_step_offset_range: tuple[int, int] = (-2, 2),
  generator: torch.Generator | None = None,
):
  """Create bottom/hard-case/neighbor-command on-policy rollout starts."""
  env = vec_env.unwrapped
  all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  env._reset_idx(all_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  # Initialize valid bottom-state histories for every environment first.
  env.observation_manager.compute(update_history=True)

  requested_ids, neighbor_ids = curriculum_destination_ids(
    env.num_envs,
    hard_case_fraction=hard_case_fraction,
    neighbor_command_fraction=neighbor_command_fraction,
    device=env.device,
    generator=generator,
  )
  hard_count = min(len(requested_ids), len(bank))
  hard_ids = requested_ids[:hard_count]
  incompatible_count = 0
  shape_mismatches: list[str] = []
  if hard_count > 0:
    sampled = bank.sample(hard_count, device=env.device, generator=generator)
    current = capture_hard_case_state(env)
    shape_mismatches = hard_case_state_shape_mismatches(current, sampled)
    if shape_mismatches:
      incompatible_count = bank.clear()
      hard_count = 0
      hard_ids = requested_ids[:0]
    else:
      restore_hard_case_state(env, sampled, hard_ids)
      env.scene.write_data_to_sim()
      env.sim.forward()
      for sensor in env.scene.sensors.values():
        sensor._invalidate_cache()
      env.sim.sense()
  neighbor_metrics = perturb_joystick_command_state(
    env,
    neighbor_ids,
    generator=generator,
    forward_scale_range=neighbor_forward_scale_range,
    delay_step_offset_range=neighbor_delay_step_offset_range,
  )
  env.observation_manager._obs_buffer = None
  obs = env.observation_manager.compute(update_history=False)
  env.obs_buf = obs
  return vec_env.get_observations(), {
    "hard_case_bank_size": len(bank),
    "hard_case_start_requested_count": len(requested_ids),
    "hard_case_start_count": hard_count,
    "hard_case_start_fraction": hard_count / max(1, env.num_envs),
    "incompatible_hard_case_dropped_count": incompatible_count,
    "hard_case_shape_mismatches": shape_mismatches,
    "neighbor_command_start_count": len(neighbor_ids),
    "neighbor_command_start_fraction": len(neighbor_ids) / max(1, env.num_envs),
    "bottom_start_count": env.num_envs - hard_count - len(neighbor_ids),
    "hard_case_start_ids": hard_ids.detach().cpu().tolist(),
    "neighbor_command_start_ids": neighbor_ids.detach().cpu().tolist(),
    **neighbor_metrics,
  }
