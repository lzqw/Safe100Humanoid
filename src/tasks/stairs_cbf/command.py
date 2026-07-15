"""Hiking-style flat-patch target sampling and position velocity command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class StairTargetCommand(CommandTerm):
  cfg: "StairTargetCommandCfg"

  def __init__(self, cfg: "StairTargetCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    terrain = env.scene.terrain
    if terrain is None or cfg.patch_name not in terrain.flat_patches:
      raise RuntimeError(f"terrain has no flat-patch set {cfg.patch_name!r}")
    self._patches = terrain.flat_patches[cfg.patch_name]
    self._command = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.metrics["target_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["forward_progress"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _env_patches(self, env_ids: torch.Tensor) -> torch.Tensor:
    terrain = self._env.scene.terrain
    assert terrain is not None
    return self._patches[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    patches = self._env_patches(env_ids)
    root_x = self.robot.data.root_link_pos_w[env_ids, 0]
    ahead = patches[..., 0] > root_x.unsqueeze(-1) + self.cfg.reached_radius
    patch_indices = torch.arange(
      patches.shape[1], device=self.device, dtype=torch.long
    ).expand_as(ahead)
    candidate = torch.where(
      ahead,
      patch_indices,
      torch.full_like(patch_indices, patches.shape[1]),
    )
    index = candidate.min(dim=1).values.clamp_max(patches.shape[1] - 1)
    index = (index + self.cfg.lookahead - 1).clamp_max(patches.shape[1] - 1)
    self.target_index[env_ids] = index
    self.target_w[env_ids] = patches[
      torch.arange(len(env_ids), device=self.device), index
    ]

  def _update_metrics(self) -> None:
    delta = self.target_w - self.robot.data.root_link_pos_w
    self.metrics["target_distance"] += torch.linalg.vector_norm(delta[:, :2], dim=1)
    self.metrics["forward_progress"] += self.robot.data.root_link_pos_w[:, 0]

  def _update_command(self) -> None:
    delta_w = self.target_w - self.robot.data.root_link_pos_w
    reached = torch.linalg.vector_norm(delta_w[:, :2], dim=1) < self.cfg.reached_radius
    terrain = self._env.scene.terrain
    assert terrain is not None
    last = self._patches.shape[2] - 1
    advance = reached & (self.target_index < last)
    if torch.any(advance):
      env_ids = advance.nonzero(as_tuple=False).flatten()
      self.target_index[env_ids] += 1
      patches = self._env_patches(env_ids)
      self.target_w[env_ids] = patches[
        torch.arange(len(env_ids), device=self.device), self.target_index[env_ids]
      ]
      delta_w = self.target_w - self.robot.data.root_link_pos_w

    delta_b = quat_apply_inverse(self.robot.data.root_link_quat_w, delta_w)
    heading_error = torch.atan2(delta_b[:, 1], delta_b[:, 0])
    self._command[:, 0] = torch.clamp(
      self.cfg.position_gain * delta_b[:, 0], min=0.0, max=self.cfg.max_forward_velocity
    )
    self._command[:, 1] = torch.clamp(
      self.cfg.position_gain * delta_b[:, 1],
      min=-self.cfg.max_lateral_velocity,
      max=self.cfg.max_lateral_velocity,
    )
    self._command[:, 2] = torch.clamp(
      self.cfg.heading_gain * heading_error,
      min=-self.cfg.max_yaw_velocity,
      max=self.cfg.max_yaw_velocity,
    )
    finished = reached & (self.target_index == last)
    self._command[finished] = 0.0


@dataclass(kw_only=True)
class StairTargetCommandCfg(CommandTermCfg):
  entity_name: str
  patch_name: str = "stair_targets"
  position_gain: float = 1.5
  heading_gain: float = 1.5
  max_forward_velocity: float = 0.8
  max_lateral_velocity: float = 0.20
  max_yaw_velocity: float = 0.8
  reached_radius: float = 0.22
  lookahead: int = 2

  def build(self, env: "ManagerBasedRlEnv") -> StairTargetCommand:
    return StairTargetCommand(self, env)
