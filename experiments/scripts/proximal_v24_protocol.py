"""Prospective constants and pure audits for v24 Contact Completion."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from proximal_v23_protocol import (
    BASE_CHECKPOINT_SHA256 as V23_BASE_CHECKPOINT_SHA256,
)
from proximal_v23_protocol import (
    formal_algorithm_parameters as v23_formal_algorithm_parameters,
)
from proximal_v23_protocol import (
    repair_regression_counts,
)

BASE_CHECKPOINT_SHA256 = V23_BASE_CHECKPOINT_SHA256

PROTOCOL_ID = "safe100-cbf-proximal-contact-completion-v24"
POLICY_METHOD = "CBF-Shielded KL-Regularized Online PPO v23"
EXPERIMENT_NAME = "v24 Contact Completion"
CONTEXT_ID = "C_effect"
CONTEXT_MODE = "contact_stability"
CONTEXT_FAMILY = "pure_low_foot_friction"
TARGET_FAILURE_TYPE = "contact_stability"

V23_PROTOCOL_SHA256 = "745e888e47d9d33fe87fffa4bbaba618a7e91f37b55ebbf8cb08f578fc1d8f38"
V23_FINAL_TEST_SHA256 = (
    "7cbbbfc596e5ad39177c946998055fa460c646730a00385701b595f77cff0148"
)
V23_RESULT_GIT_TREE = "68f489ac49af8020912f7e3e71d317e7f8fc1f0f"
V23_FROZEN_RESULT = {
    "target_base_success": 0.693359375,
    "target_final_success": 0.689453125,
    "target_success_delta": -0.00390625,
    "target_fall_delta": 0.00390625,
    "target_repairs": 93,
    "target_regressions": 95,
}

# These are immutable parameter-grid identifiers from the prospectively
# declared v22 C_effect family.  v22 stopped after lateral, so no C_effect
# candidate was ever evaluated.  v24 uses entirely fresh execution randomness.
CALIBRATION_CANDIDATE_PARAMETER_SEEDS = tuple(range(51_108, 51_124))
CALIBRATION_FRICTIONS = (
    0.440000,
    0.427333,
    0.414667,
    0.402000,
    0.389333,
    0.376667,
    0.364000,
    0.351333,
    0.338667,
    0.326000,
    0.313333,
    0.300667,
    0.288000,
    0.275333,
    0.262667,
    0.250000,
)
CALIBRATION_EPISODES = 512
EVAL_BATCH_SIZE = 128
CALIBRATION_REPEATS = CALIBRATION_EPISODES // EVAL_BATCH_SIZE
CALIBRATION_EVALUATION_SEED_BASE = 132_400_000
CALIBRATION_SUCCESS_BOUNDS = (0.65, 0.75)
CALIBRATION_MINIMUM_FALLS = 100
CALIBRATION_MINIMUM_PURITY = 0.85

ADAPTATION_SEED = 133_240_001
FINAL_TARGET_SEED = 135_000_000
FINAL_D0_SEED = 136_000_000
FINAL_TARGET_EPISODES = 512
FINAL_D0_EPISODES = 256
REPORT_BOOTSTRAP_SEEDS = {
    "target": 137_000_000,
    "D0": 137_000_100,
}
REPORT_BOOTSTRAP_METRIC_COUNT = 9
REPORT_BOOTSTRAP_SAMPLES = 2_000

MINIMUM_TARGET_SUCCESS_DELTA = 0.03
MAXIMUM_TARGET_FALL_DELTA = 0.01
MINIMUM_D0_SUCCESS_DELTA = -0.05


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def formal_algorithm_parameters() -> dict[str, Any]:
    """Return the exact v23 algorithm contract without any v24 override."""
    return v23_formal_algorithm_parameters()


def calibration_evaluation_seed(candidate_index: int, repeat: int) -> int:
    if not 0 <= candidate_index < len(CALIBRATION_FRICTIONS):
        raise ValueError("v24 calibration candidate index is out of range")
    if not 0 <= repeat < CALIBRATION_REPEATS:
        raise ValueError("v24 calibration repeat is out of range")
    return CALIBRATION_EVALUATION_SEED_BASE + 100 * candidate_index + repeat


def all_v24_fresh_execution_seeds() -> list[int]:
    values = [
        calibration_evaluation_seed(candidate_index, repeat)
        for candidate_index in range(len(CALIBRATION_FRICTIONS))
        for repeat in range(CALIBRATION_REPEATS)
    ]
    values.append(ADAPTATION_SEED)
    values.extend(
        FINAL_TARGET_SEED + repeat
        for repeat in range(FINAL_TARGET_EPISODES // EVAL_BATCH_SIZE)
    )
    values.extend(
        FINAL_D0_SEED + repeat for repeat in range(FINAL_D0_EPISODES // EVAL_BATCH_SIZE)
    )
    for seed in REPORT_BOOTSTRAP_SEEDS.values():
        values.extend(seed + offset for offset in range(REPORT_BOOTSTRAP_METRIC_COUNT))
    return values


def _declares_seed_values(key: str) -> bool:
    normalized = key.lower()
    return normalized in ("seed", "seeds") or normalized.endswith(
        ("_seed", "_seeds", "_seed_base", "_seed_bases", "_seed_start")
    )


def _direct_seed_values(value: Any) -> Iterable[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        yield value
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, int) and not isinstance(child, bool):
                yield child
    elif isinstance(value, dict):
        for child in value.values():
            if isinstance(child, int) and not isinstance(child, bool):
                yield child


def _iter_declared_seeds(value: Any) -> Iterable[int]:
    if isinstance(value, dict):
        for key, child in value.items():
            if _declares_seed_values(key):
                yield from _direct_seed_values(child)
            yield from _iter_declared_seeds(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_declared_seeds(child)


def fresh_randomness_report(repo: Path) -> dict[str, Any]:
    """Prove all v24 execution seeds are disjoint from published evidence."""
    historical: set[int] = set()
    scanned: list[dict[str, str]] = []
    results_root = repo / "results/online"
    if results_root.is_dir():
        for path in sorted(results_root.rglob("*.json")):
            if "proximal_v24" in path.parts or "proximal_completion" in path.parts:
                continue
            try:
                rendered = path.read_text()
                payload = json.loads(rendered)
            except (json.JSONDecodeError, OSError):
                continue
            scanned.append(
                {
                    "file": str(path.relative_to(repo)),
                    "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                }
            )
            historical.update(_iter_declared_seeds(payload))
    proposed_values = all_v24_fresh_execution_seeds()
    counts = Counter(proposed_values)
    internal = sorted(seed for seed, count in counts.items() if count > 1)
    historical_collisions = sorted(set(proposed_values) & historical)
    collisions = sorted(set(internal) | set(historical_collisions))
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "historical_scope": (
            "all parseable JSON below results/online except proximal_v24 and "
            "proximal_completion"
        ),
        "historical_json_file_count": len(scanned),
        "historical_json_files_sha256": canonical_sha256({"files": scanned}),
        "historical_unique_declared_seed_count": len(historical),
        "proposed_seed_occurrence_count": len(proposed_values),
        "proposed_unique_seed_count": len(set(proposed_values)),
        "proposed_internal_collisions": internal,
        "historical_collisions": historical_collisions,
        "collisions": collisions,
        "passed": not collisions,
    }


def calibration_gate(
    *,
    success_count: int,
    fall_count: int,
    contact_fall_count: int,
    non_success_count: int,
    num_episodes: int = CALIBRATION_EPISODES,
) -> dict[str, Any]:
    counts = (
        success_count,
        fall_count,
        contact_fall_count,
        non_success_count,
        num_episodes,
    )
    if any(isinstance(value, bool) or int(value) != value for value in counts):
        raise ValueError("v24 calibration counts must be integers")
    if not (
        0 <= success_count <= num_episodes
        and 0 <= fall_count <= non_success_count
        and 0 <= contact_fall_count <= fall_count
        and success_count + non_success_count == num_episodes
    ):
        raise ValueError("v24 calibration counts are inconsistent")
    success_rate = success_count / num_episodes
    purity_over_falls = contact_fall_count / max(1, fall_count)
    purity_over_all_non_success = contact_fall_count / max(1, non_success_count)
    conditions = {
        "success_rate_in_closed_65_75_percent_interval": (
            CALIBRATION_SUCCESS_BOUNDS[0]
            <= success_rate
            <= CALIBRATION_SUCCESS_BOUNDS[1]
        ),
        "at_least_100_falls": fall_count >= CALIBRATION_MINIMUM_FALLS,
        # This denominator is stricter than falls-only purity when timeouts exist.
        "contact_purity_over_all_non_success_at_least_85_percent": (
            purity_over_all_non_success >= CALIBRATION_MINIMUM_PURITY
        ),
        "contact_purity_over_falls_at_least_85_percent": (
            purity_over_falls >= CALIBRATION_MINIMUM_PURITY
        ),
    }
    return {
        "qualifies": all(conditions.values()),
        "conditions": conditions,
        "success_rate": success_rate,
        "fall_count": fall_count,
        "contact_fall_count": contact_fall_count,
        "non_success_count": non_success_count,
        "contact_purity_over_falls": purity_over_falls,
        "contact_purity_over_all_non_success": purity_over_all_non_success,
    }


def pure_contact_context_audit(
    payload: Mapping[str, Any], *, candidate_index: int | None = None
) -> dict[str, Any]:
    """Prove the v24 target differs from the nominal deployment only in friction."""
    target = payload.get("target")
    scenario = payload.get("scenario")
    if not isinstance(target, Mapping) or not isinstance(scenario, Mapping):
        raise TypeError("v24 contact context is missing target/scenario parameters")
    expected_target = {
        "num_steps": 9,
        "rise_profile": [0.13] * 9,
        "tread_profile": [0.35] * 9,
        "command_forward_scale": 1.0,
        "command_delay_s": 0.10,
        "command_low_pass_s": 0.08,
        "action_gain": 1.0,
        "action_bias": [0.0] * 12,
        "action_delay_steps": 0,
        "encoder_bias": 0.0,
        "episode_length_s": 35.0,
    }
    expected_scenario = {
        "disturbance_pulses_with_centering": False,
        "lateral_command_bias": 0.0,
        "yaw_command_bias": 0.0,
        # The pulse ranges are inert because pulses are disabled.  They remain at
        # the frozen family defaults rather than becoming another changed axis.
        "lateral_pulse_min": 0.02,
        "lateral_pulse_max": 0.08,
        "yaw_pulse_min": 0.05,
        "yaw_pulse_max": 0.20,
        "pulse_interval_min_s": 3.0,
        "pulse_interval_max_s": 7.0,
        "pulse_duration_min_s": 0.2,
        "pulse_duration_max_s": 0.6,
        "contact_observation_delay_steps": 0,
        "gait_phase_offset": 0.0,
        "left_response_scale": 1.0,
        "right_response_scale": 1.0,
        "centerline_lateral_gain": 0.80,
        "centerline_heading_gain": 1.40,
        "centerline_max_lateral_velocity": 0.16,
        "centerline_max_yaw_velocity": 0.45,
        "toe_margin": 0.080,
        "recovery_reward_scale": 0.40,
        "edge_penalty_scale": 0.0,
        "centerline_heading_reference_bias": 0.0,
        "stair_half_width": 1.20,
    }
    mismatches: dict[str, Any] = {}
    for name, expected in expected_target.items():
        if target.get(name) != expected:
            mismatches[f"target.{name}"] = {
                "actual": target.get(name),
                "expected": expected,
            }
    for name, expected in expected_scenario.items():
        if scenario.get(name) != expected:
            mismatches[f"scenario.{name}"] = {
                "actual": scenario.get(name),
                "expected": expected,
            }
    friction = float(scenario.get("foot_friction", float("nan")))
    if candidate_index is not None:
        expected_friction = CALIBRATION_FRICTIONS[candidate_index]
        if not math.isclose(friction, expected_friction, abs_tol=5.0e-7):
            mismatches["scenario.foot_friction"] = {
                "actual": friction,
                "expected": expected_friction,
            }
        expected_seed = CALIBRATION_CANDIDATE_PARAMETER_SEEDS[candidate_index]
        if payload.get("calibration_candidate_seed") != expected_seed:
            mismatches["calibration_candidate_seed"] = {
                "actual": payload.get("calibration_candidate_seed"),
                "expected": expected_seed,
            }
    if payload.get("context_id") != CONTEXT_ID:
        mismatches["context_id"] = payload.get("context_id")
    if payload.get("specialist_mode") != CONTEXT_MODE:
        mismatches["specialist_mode"] = payload.get("specialist_mode")
    if payload.get("context_family") != CONTEXT_FAMILY:
        mismatches["context_family"] = payload.get("context_family")
    if mismatches:
        raise ValueError(f"v24 context is not pure low foot friction: {mismatches}")
    return {
        "passed": True,
        "only_changed_physical_axis": "foot_friction",
        "foot_friction": friction,
        "terrain_nominal": True,
        "command_nominal": True,
        "actuator_nominal": True,
        "sensor_nominal": True,
        "phase_nominal": True,
        "navigation_nominal": True,
        "disturbance_pulses_disabled": True,
    }


def validate_v24_calibrated_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the first qualifying base-only v24 contact context."""
    output = dict(payload)
    seed = output.get("calibration_candidate_seed")
    if seed not in CALIBRATION_CANDIDATE_PARAMETER_SEEDS:
        raise ValueError("v24 context candidate identifier is outside the frozen grid")
    selected_index = CALIBRATION_CANDIDATE_PARAMETER_SEEDS.index(int(seed))
    pure_contact_context_audit(output, candidate_index=selected_index)
    calibration = output.get("calibration")
    if not isinstance(calibration, Mapping):
        raise TypeError("v24 context is missing calibration evidence")
    if calibration.get("kind") != "base_policy_pure_contact_first_qualifying_v24":
        raise ValueError("unexpected v24 calibration kind")
    if calibration.get("adapted_policy_evaluations_used") is not False:
        raise ValueError("v24 calibration used an adapted policy")
    if calibration.get("candidate_parameter_seeds") != list(
        CALIBRATION_CANDIDATE_PARAMETER_SEEDS
    ):
        raise ValueError("v24 calibration candidate order differs from the freeze")
    attempts = calibration.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("v24 calibration attempts are missing")
    if len(attempts) > len(CALIBRATION_CANDIDATE_PARAMETER_SEEDS):
        raise ValueError("v24 calibration attempts exceed the candidate grid")
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise TypeError("v24 calibration attempt is not a mapping")
        if (
            attempt.get("candidate_parameter_seed")
            != (CALIBRATION_CANDIDATE_PARAMETER_SEEDS[index])
        ):
            raise ValueError("v24 calibration skipped or reordered a candidate")
        if attempt.get("base_policy_only") is not True:
            raise ValueError("v24 calibration attempt was not base-policy only")
        if attempt.get("num_episodes") != CALIBRATION_EPISODES:
            raise ValueError("v24 calibration attempt does not contain 512 episodes")
        gate = calibration_gate(
            success_count=int(attempt.get("success_count", -1)),
            fall_count=int(attempt.get("fall_count", -1)),
            contact_fall_count=int(attempt.get("contact_fall_count", -1)),
            non_success_count=int(attempt.get("non_success_count", -1)),
        )
        if bool(attempt.get("qualifies")) is not gate["qualifies"]:
            raise ValueError("v24 calibration qualification is inconsistent")
        if index < len(attempts) - 1 and gate["qualifies"]:
            raise ValueError("v24 calibration skipped an earlier qualifier")
        if index == len(attempts) - 1 and not gate["qualifies"]:
            raise ValueError("selected v24 contact context fails its gates")
    selected = attempts[-1]
    if selected["candidate_parameter_seed"] != seed:
        raise ValueError("selected v24 candidate differs from the context")
    if selected.get("parameters_sha256") != output.get("parameters_sha256"):
        raise ValueError("selected v24 parameter hash differs from the context")
    output["calibration"] = dict(calibration)
    return output


def development_gate(
    *,
    target_success_delta: float,
    target_fall_delta: float,
    d0_success_delta: float,
) -> dict[str, Any]:
    values = (target_success_delta, target_fall_delta, d0_success_delta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("v24 development metrics must be finite")
    conditions = {
        "target_success_delta_at_least_three_pp": (
            target_success_delta >= MINIMUM_TARGET_SUCCESS_DELTA
        ),
        "target_fall_delta_at_most_one_pp": (
            target_fall_delta <= MAXIMUM_TARGET_FALL_DELTA
        ),
        "d0_success_delta_at_least_minus_five_pp": (
            d0_success_delta >= MINIMUM_D0_SUCCESS_DELTA
        ),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "target_success_delta": target_success_delta,
        "target_fall_delta": target_fall_delta,
        "d0_success_delta": d0_success_delta,
        "confidence_intervals_are_gates": False,
    }


def paired_repair_regression_counts(
    baseline_success: Sequence[bool], final_success: Sequence[bool]
) -> dict[str, Any]:
    return repair_regression_counts(baseline_success, final_success)
