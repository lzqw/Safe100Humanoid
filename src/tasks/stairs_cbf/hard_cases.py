"""On-policy hard-case state capture and replay for online stair refinement.

The bank stores simulator states from shortly before a *real* CBF projection.
Restoring only q/qd is insufficient: command latency, observation history,
contact air-time, previous actions, and stationary reward baselines are part of
the online MDP state.  This module captures those quantities explicitly while
keeping the actor interface unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


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
) -> LateFailureCandidate | None:
  """Choose one late-riser state from the window preceding a target fall."""
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
    # interventions on otherwise successful stair transitions entering v15.
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
  ) -> None:
    if capacity < 1:
      raise ValueError("hard-case capacity must be positive")
    self.capacity = capacity
    self.bank_kind = bank_kind
    self.source_domain = source_domain
    self.context_sha256 = context_sha256
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
      )
      for item in payload.get("entries", [])
    ]
    if len(self.entries) > self.capacity:
      raise ValueError("serialized hard-case bank exceeds capacity")

  def audit_metadata(self) -> dict[str, Any]:
    late_entries = [
      entry for entry in self.entries if entry.steps_before_fall is not None
    ]
    return {
      "bank_kind": self.bank_kind,
      "source_domain": self.source_domain,
      "context_sha256": self.context_sha256,
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
