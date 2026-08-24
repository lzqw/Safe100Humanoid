"""Fixed uniform and nonuniform deployment contexts for v30."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .teacher_v26 import configure_v26_higher_riser


def configure_v30_context(
    env_cfg,
    *,
    context: str,
    runtime_filter: bool,
    context_spec: dict[str, Any],
    clearance_barrier_slope: float,
    recovery_distance_m: float,
    filter_alpha: float,
) -> dict[str, Any]:
    """Install exactly one v30 context without changing non-riser factors."""
    profile = context_spec.get("riser_profile_m")
    uniform = context_spec.get("riser_height_m")
    if (profile is None) == (uniform is None):
        raise ValueError("v30 context must define exactly one riser geometry")
    nominal_height = float(uniform if profile is None else 0.180)
    metadata = configure_v26_higher_riser(
        env_cfg,
        riser_height_m=nominal_height,
        runtime_filter=runtime_filter,
        clearance_barrier_slope=clearance_barrier_slope,
        recovery_distance_m=recovery_distance_m,
        filter_alpha=filter_alpha,
    )
    if profile is not None:
        profile = tuple(float(value) for value in profile)
        if len(profile) != 11:
            raise ValueError("v30 nonuniform profile must contain exactly 11 risers")
        terrain = env_cfg.scene.terrain
        generator = None if terrain is None else terrain.terrain_generator
        if generator is None or set(generator.sub_terrains) != {"forward_stairs"}:
            raise RuntimeError("v30 requires the single fixed forward-stair terrain")
        stairs = generator.sub_terrains["forward_stairs"]
        generator.sub_terrains["forward_stairs"] = replace(
            stairs,
            num_steps=len(profile),
            step_height_range=(0.180, 0.180),
            step_height_profile=profile,
        )
        generator.size = (
            stairs.first_riser_x
            + len(profile) * stairs.step_width
            + stairs.top_platform_length,
            generator.size[1],
        )
        action_cfg = env_cfg.actions["joint_pos"]
        env_cfg.actions["joint_pos"] = replace(
            action_cfg,
            num_steps=len(profile),
            step_height=0.180,
        )
        metadata.update(
            {
                "shift": "fixed_nonuniform_higher_riser_profile",
                "riser_height_m": None,
                "riser_profile_m": list(profile),
                "num_risers": len(profile),
            }
        )
    else:
        metadata.update(
            {
                "riser_profile_m": None,
                "num_risers": int(env_cfg.actions["joint_pos"].num_steps),
            }
        )
    metadata["context"] = context
    metadata["context_name"] = context_spec["name"]
    return metadata
