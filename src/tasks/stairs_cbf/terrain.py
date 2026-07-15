"""A forward staircase with explicit tread targets and riser geometry."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class ForwardStairsTerrainCfg(SubTerrainCfg):
  """Straight +x staircase used for the paper's swing-foot CBF."""

  step_height_range: tuple[float, float] = (0.08, 0.18)
  step_width: float = 0.35
  num_steps: int = 6
  stair_width: float = 2.4
  first_riser_x: float = 1.35
  spawn_x: float = 0.75
  top_platform_length: float = 1.2
  base_thickness: float = 0.10

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )
    center_y = 0.5 * self.size[1]
    gray = (0.32, 0.36, 0.42, 1.0)
    blue = (0.20, 0.42, 0.70, 1.0)
    geometries: list[TerrainGeometry] = []

    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.base_thickness / 2),
      pos=(self.size[0] / 2, center_y, -self.base_thickness / 2),
    )
    geometries.append(TerrainGeometry(geom=floor, color=gray))

    targets = []
    for k in range(self.num_steps):
      height = (k + 1) * step_height
      tread_start = self.first_riser_x + k * self.step_width
      tread = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.step_width / 2, self.stair_width / 2, height / 2),
        pos=(tread_start + self.step_width / 2, center_y, height / 2),
      )
      geometries.append(TerrainGeometry(geom=tread, color=blue))
      targets.append((tread_start + self.step_width / 2, center_y, height))

    top_height = self.num_steps * step_height
    top_start = self.first_riser_x + self.num_steps * self.step_width
    top_length = min(self.top_platform_length, self.size[0] - top_start)
    platform = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(top_length / 2, self.stair_width / 2, top_height / 2),
      pos=(top_start + top_length / 2, center_y, top_height / 2),
    )
    geometries.append(TerrainGeometry(geom=platform, color=blue))
    targets.append((top_start + 0.5 * top_length, center_y, top_height))

    origin = np.array([self.spawn_x, center_y, 0.0], dtype=np.float64)
    return TerrainOutput(
      origin=origin,
      geometries=geometries,
      flat_patches={"stair_targets": np.asarray(targets, dtype=np.float64)},
    )
