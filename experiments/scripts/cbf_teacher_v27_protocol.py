"""Prospectively fixed confirmatory protocol for the v27 recovery-window change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "safe100-success-gated-cbf-teacher-v27"
EXPERIMENT_NAME = "v27 Short-Recovery Sloped-Clearance CBF Teacher"
TASK_ID = "Unitree-G1-Stairs-Online-DQHMED"
ENVIRONMENT_VARIANT = "fixed_deployment_play"
BASE_CHECKPOINT_SHA256 = (
    "cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8"
)

RISER_HEIGHT_M = 0.180
CLEARANCE_BARRIER_SLOPE = 0.80
RECOVERY_DISTANCE_M = 0.02
CALIBRATION_EPISODES = 256
EVAL_BATCH_SIZE = 128
CALIBRATION_REPEATS = CALIBRATION_EPISODES // EVAL_BATCH_SIZE
CALIBRATION_SEED_BASE = 151_270_000
ALIGNMENT_COVERAGE_MINIMUM = 0.80
SHIELD_RESCUE_MINIMUM = 0.60
CALIBRATION_OFF_SUCCESS_BOUNDS = (0.40, 0.65)
CALIBRATION_ON_SUCCESS_BOUNDS = (0.80, 0.95)
DEVELOPMENT_SEED = 150_270_018

# Training/final seeds are fixed now, but used only after a qualifying calibration.
ADAPTATION_SEED = 152_270_001
FINAL_SEED_BASE = 153_270_000
FINAL_EPISODES = 512

SOURCE_FILES = (
    "src/tasks/stairs_cbf/actions.py",
    "src/tasks/stairs_cbf/cbf_math.py",
    "src/tasks/stairs_cbf/edge_detection.py",
    "src/tasks/stairs_cbf/teacher.py",
    "src/tasks/stairs_cbf/teacher_v26.py",
    "experiments/scripts/proximal_v23_io.py",
    "experiments/scripts/evaluate_cbf_teacher_v26.py",
    "experiments/scripts/cbf_teacher_v27_protocol.py",
    "experiments/scripts/freeze_cbf_teacher_v27_precalibration.py",
    "experiments/scripts/calibrate_cbf_teacher_v27.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def calibration_seed(repeat: int) -> int:
    if not 0 <= repeat < CALIBRATION_REPEATS:
        raise ValueError("v27 calibration repeat outside frozen schedule")
    return CALIBRATION_SEED_BASE + repeat


def fixed_environment_parameters() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "registered_variant": ENVIRONMENT_VARIANT,
        "geometry": "uniform_higher_riser",
        "riser_height_m": RISER_HEIGHT_M,
        "clearance_barrier": "sloped_xz",
        "clearance_barrier_slope": CLEARANCE_BARRIER_SLOPE,
        "recovery_distance_m": RECOVERY_DISTANCE_M,
        "friction": "nominal",
        "command": "nominal",
        "controller": "nominal",
        "plant_action_transform": "identity",
        "actor_observation_interface": "original_405D",
        "actor_observation_corruption": "disabled",
        "curriculum": "disabled",
    }


def calibration_gate(
    *,
    off_success_count: int,
    on_success_count: int,
    off_toe_riser_failure_count: int,
    off_failure_count: int,
    rescued_count: int,
    paired_count: int = CALIBRATION_EPISODES,
) -> dict[str, Any]:
    counts = (
        off_success_count,
        on_success_count,
        off_toe_riser_failure_count,
        off_failure_count,
        rescued_count,
        paired_count,
    )
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("v27 calibration counts must be non-negative integers")
    if paired_count < 1 or off_failure_count != paired_count - off_success_count:
        raise ValueError("v27 calibration failure count is inconsistent")
    if on_success_count > paired_count:
        raise ValueError("v27 on successes exceed paired count")
    if off_toe_riser_failure_count > off_failure_count or rescued_count > off_failure_count:
        raise ValueError("v27 aligned/rescued counts exceed failures")
    off_rate = off_success_count / paired_count
    on_rate = on_success_count / paired_count
    alignment = off_toe_riser_failure_count / max(1, off_failure_count)
    rescue = rescued_count / max(1, off_failure_count)
    conditions = {
        "alignment_coverage_at_least_80pct": (
            off_failure_count > 0 and alignment >= ALIGNMENT_COVERAGE_MINIMUM
        ),
        "shield_rescue_rate_at_least_60pct": (
            off_failure_count > 0 and rescue >= SHIELD_RESCUE_MINIMUM
        ),
        "off_success_in_40_to_65pct": (
            CALIBRATION_OFF_SUCCESS_BOUNDS[0]
            <= off_rate
            <= CALIBRATION_OFF_SUCCESS_BOUNDS[1]
        ),
        "on_success_in_80_to_95pct": (
            CALIBRATION_ON_SUCCESS_BOUNDS[0]
            <= on_rate
            <= CALIBRATION_ON_SUCCESS_BOUNDS[1]
        ),
    }
    return {
        "qualifies": all(conditions.values()),
        "conditions": conditions,
        "paired_count": paired_count,
        "off_success_count": off_success_count,
        "off_success_rate": off_rate,
        "on_success_count": on_success_count,
        "on_success_rate": on_rate,
        "off_failure_count": off_failure_count,
        "off_toe_riser_failure_count": off_toe_riser_failure_count,
        "alignment_coverage": alignment,
        "rescued_count": rescued_count,
        "shield_rescue_rate": rescue,
    }
