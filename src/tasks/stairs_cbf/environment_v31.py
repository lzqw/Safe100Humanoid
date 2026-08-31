"""Fixed uniform and nonuniform deployment contexts for v31."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .teacher_v26 import configure_v26_higher_riser


def resize_stair_patch_allocation(stairs, *, num_risers: int):
    """Return a stair config whose patch buffers match its generated geometry."""
    if num_risers < 1:
        raise ValueError("v31 requires at least one riser")
    allocation = dict(stairs.flat_patch_sampling or {})
    if set(allocation) != {"stair_targets", "stair_risers"}:
        raise RuntimeError("v31 requires target and riser flat-patch samplers")
    allocation["stair_targets"] = replace(
        allocation["stair_targets"], num_patches=num_risers + 1
    )
    allocation["stair_risers"] = replace(
        allocation["stair_risers"], num_patches=num_risers
    )
    return replace(
        stairs,
        num_steps=num_risers,
        flat_patch_sampling=allocation,
    )


def configure_v31_context(
    env_cfg,
    *,
    context: str,
    runtime_filter: bool,
    context_spec: dict[str, Any],
    clearance_barrier_slope: float,
    recovery_distance_m: float,
    filter_alpha: float,
) -> dict[str, Any]:
    """Install exactly one v31 context without changing non-riser factors."""
    profile = context_spec.get("riser_profile_m")
    uniform = context_spec.get("riser_height_m")
    if (profile is None) == (uniform is None):
        raise ValueError("v31 context must define exactly one riser geometry")
    nominal_height = float(uniform if profile is None else 0.180)
    metadata = configure_v26_higher_riser(
        env_cfg,
        riser_height_m=nominal_height,
        runtime_filter=runtime_filter,
        clearance_barrier_slope=clearance_barrier_slope,
        recovery_distance_m=recovery_distance_m,
        filter_alpha=filter_alpha,
    )
    terrain = env_cfg.scene.terrain
    generator = None if terrain is None else terrain.terrain_generator
    if generator is None or set(generator.sub_terrains) != {"forward_stairs"}:
        raise RuntimeError("v31 requires the single fixed forward-stair terrain")
    stairs = generator.sub_terrains["forward_stairs"]
    if profile is not None:
        profile = tuple(float(value) for value in profile)
        if len(profile) != 11:
            raise ValueError("v31 nonuniform profile must contain exactly 11 risers")
        stairs = replace(
            stairs,
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
        num_risers = int(env_cfg.actions["joint_pos"].num_steps)
        metadata.update(
            {
                "riser_profile_m": None,
                "num_risers": num_risers,
            }
        )
    num_risers = int(metadata["num_risers"])
    stairs = resize_stair_patch_allocation(stairs, num_risers=num_risers)
    generator.sub_terrains["forward_stairs"] = stairs
    target_slots = int(stairs.flat_patch_sampling["stair_targets"].num_patches)
    riser_slots = int(stairs.flat_patch_sampling["stair_risers"].num_patches)
    if target_slots != num_risers + 1 or riser_slots != num_risers:
        raise RuntimeError("v31 dynamic stair patch allocation is inconsistent")
    if int(context_spec.get("num_risers", num_risers)) != num_risers:
        raise RuntimeError("v31 protocol riser count differs from environment")
    if int(context_spec.get("stair_target_patch_slots", target_slots)) != target_slots:
        raise RuntimeError("v31 protocol target-patch count differs from environment")
    metadata["stair_target_patch_slots"] = target_slots
    metadata["stair_riser_patch_slots"] = riser_slots
    metadata["top_platform_patch_included"] = True
    metadata["context"] = context
    metadata["context_name"] = context_spec["name"]
    return metadata
