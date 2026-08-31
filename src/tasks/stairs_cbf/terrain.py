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
  step_height_profile: tuple[float, ...] | None = None
  step_width_profile: tuple[float, ...] | None = None

  def _geometry_profile(self, difficulty: float) -> tuple[np.ndarray, np.ndarray]:
    """Return explicit per-riser rise and tread arrays.

    Profiles are used for a fixed deployment target.  When they are absent the
    original curriculum behavior is preserved.  Keeping this decision inside
    the terrain generator makes the geometry metadata authoritative for the
    command generator, reward, CBF, and evaluator.
    """
    if self.step_height_profile is None:
      step_height = self.step_height_range[0] + difficulty * (
        self.step_height_range[1] - self.step_height_range[0]
      )
      rises = np.full(self.num_steps, step_height, dtype=np.float64)
    else:
      if len(self.step_height_profile) != self.num_steps:
        raise ValueError(
          "step_height_profile length must equal num_steps: "
          f"{len(self.step_height_profile)} != {self.num_steps}"
        )
      rises = np.asarray(self.step_height_profile, dtype=np.float64)

    if self.step_width_profile is None:
      treads = np.full(self.num_steps, self.step_width, dtype=np.float64)
    else:
      if len(self.step_width_profile) != self.num_steps:
        raise ValueError(
          "step_width_profile length must equal num_steps: "
          f"{len(self.step_width_profile)} != {self.num_steps}"
        )
      treads = np.asarray(self.step_width_profile, dtype=np.float64)

    if np.any(rises <= 0.0) or np.any(treads <= 0.0):
      raise ValueError("all stair rises and tread depths must be positive")
    return rises, treads

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    rises, tread_depths = self._geometry_profile(difficulty)
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
    risers = []
    cumulative_height = 0.0
    tread_start = self.first_riser_x
    for k in range(self.num_steps):
      cumulative_height += float(rises[k])
      tread_depth = float(tread_depths[k])
      height = cumulative_height
      tread = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(tread_depth / 2, self.stair_width / 2, height / 2),
        pos=(tread_start + tread_depth / 2, center_y, height / 2),
      )
      geometries.append(TerrainGeometry(geom=tread, color=blue))
      targets.append((tread_start + tread_depth / 2, center_y, height))
      # Store the exact vertical edge plane and its top height.  The y value is
      # retained for a uniform xyz metadata shape, but is not reconstructed.
      risers.append((tread_start, center_y, height))
      tread_start += tread_depth

    top_height = cumulative_height
    top_start = tread_start
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
      flat_patches={
        "stair_targets": np.asarray(targets, dtype=np.float64),
        "stair_risers": np.asarray(risers, dtype=np.float64),
      },
    )
