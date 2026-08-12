"""Prospective constants and pure audits for CBF-teacher v25."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_ID = "safe100-success-gated-cbf-teacher-v25"
EXPERIMENT_NAME = "v25 Swing-Foot Under-Clearance CBF Teacher"
POLICY_METHOD = "Success-Gated CBF Action Teacher + Moving-KL PPO v25"
CONTEXT_ID = "swing_underresponse_v25"
CONTEXT_FAMILY = "fixed_phase_selective_swing_leg_underresponse"
BASE_CHECKPOINT_SHA256 = (
    "cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8"
)

V23_PROTOCOL_SHA256 = "745e888e47d9d33fe87fffa4bbaba618a7e91f37b55ebbf8cb08f578fc1d8f38"
V23_FINAL_SHA256 = "7cbbbfc596e5ad39177c946998055fa460c646730a00385701b595f77cff0148"
V23_RESULT_GIT_TREE = "68f489ac49af8020912f7e3e71d317e7f8fc1f0f"
V24_PROTOCOL_SHA256 = "d52de9034523350dc5ebda1143d26ab02737d31cafccc764d3c9a092c4e6f39b"
V24_FINAL_SHA256 = "262c570ed5c15368f454e7fa2064ec789a29379cbefdf27481e9b0ad0a43a62a"
V24_RESULT_GIT_TREE = "6718715354ed4cb729d85f3cec4f36dcc16cc824"

ROUNDS = 8
NUM_ENVS = 64
ROLLOUT_STEPS = 1024
ACTOR_LEARNING_RATE = 5.0e-6
CRITIC_LEARNING_RATE = 1.0e-4
PPO_CLIP = 0.05
MAX_ACTOR_EPOCHS = 2
CRITIC_EPOCHS = 2
MINI_BATCHES = 4
MOVING_KL_BETA = 0.5
TARGET_KL = 0.003
HARD_KL_CEILING = 0.01
MAX_GRAD_NORM = 0.5
STD_SCALE_FROM_BASE = 0.35
MINIMUM_STD = 0.05
MAXIMUM_STD = 0.25
ENTROPY_COEFFICIENT = 0.0
GAMMA = 0.99
GAE_LAMBDA = 0.95
TEACHER_DISTILLATION_WEIGHT = 0.1
TEACHER_SUCCESS_HORIZON = 50
TEACHER_SUCCESS_HORIZON_SECONDS = 1.0
TEACHER_CORRECTION_SCALE = 0.05

CALIBRATION_GAINS = tuple(round(0.98 - 0.02 * index, 2) for index in range(25))
CALIBRATION_EPISODES = 512
EVAL_BATCH_SIZE = 128
CALIBRATION_REPEATS = CALIBRATION_EPISODES // EVAL_BATCH_SIZE
CALIBRATION_SEED_BASE = 142_500_000
CALIBRATION_SEED_STRIDE = 100
ALIGNMENT_COVERAGE_MINIMUM = 0.80
SHIELD_RESCUE_MINIMUM = 0.60
CALIBRATION_OFF_SUCCESS_BOUNDS = (0.40, 0.65)
CALIBRATION_ON_SUCCESS_BOUNDS = (0.80, 0.95)

ADAPTATION_SEED = 143_250_001
FINAL_SEED_BASE = 145_000_000
FINAL_EPISODES = 512
FINAL_REPEATS = FINAL_EPISODES // EVAL_BATCH_SIZE

MINIMUM_OFF_SUCCESS_DELTA = 0.05
MINIMUM_ON_SUCCESS_DELTA = 0.0
MINIMUM_INTERVENTION_REDUCTION = 0.20

SOURCE_FILES = (
    "src/tasks/stairs_cbf/actions.py",
    "src/tasks/stairs_cbf/config.py",
    "src/tasks/stairs_cbf/mdp.py",
    "src/tasks/stairs_cbf/online.py",
    "src/tasks/stairs_cbf/proximal.py",
    "src/tasks/stairs_cbf/teacher_math.py",
    "src/tasks/stairs_cbf/teacher.py",
    "experiments/scripts/proximal_v23_io.py",
    "experiments/scripts/cbf_teacher_v25_protocol.py",
    "experiments/scripts/evaluate_cbf_teacher_v25.py",
    "experiments/scripts/calibrate_cbf_teacher_v25.py",
    "experiments/scripts/freeze_cbf_teacher_v25_precalibration.py",
    "experiments/scripts/freeze_cbf_teacher_v25_protocol.py",
    "experiments/scripts/refine_cbf_teacher_v25.py",
    "experiments/scripts/audit_cbf_teacher_v25.py",
    "experiments/scripts/verify_cbf_teacher_v25.py",
    "experiments/scripts/plot_cbf_teacher_v25.py",
    "experiments/scripts/run_cbf_teacher_v25.sh",
    "experiments/tests/test_cbf_teacher_v25.py",
    "docs/CBF_TEACHER_V25_SWING_UNDERRESPONSE.md",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def formal_algorithm_parameters() -> dict[str, Any]:
    return {
        "rounds": ROUNDS,
        "num_envs": NUM_ENVS,
        "rollout_steps": ROLLOUT_STEPS,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "ppo_clip": PPO_CLIP,
        "maximum_actor_epochs": MAX_ACTOR_EPOCHS,
        "critic_epochs": CRITIC_EPOCHS,
        "mini_batches": MINI_BATCHES,
        "moving_kl_beta": MOVING_KL_BETA,
        "target_kl": TARGET_KL,
        "hard_kl_ceiling": HARD_KL_CEILING,
        "maximum_gradient_norm": MAX_GRAD_NORM,
        "freeze_log_std": True,
        "std_scale_from_base": STD_SCALE_FROM_BASE,
        "minimum_std": MINIMUM_STD,
        "maximum_std": MAXIMUM_STD,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "whole_batch_advantage_normalization": True,
        "teacher_distillation_weight": TEACHER_DISTILLATION_WEIGHT,
        "teacher_success_horizon_steps": TEACHER_SUCCESS_HORIZON,
        "teacher_success_horizon_seconds": TEACHER_SUCCESS_HORIZON_SECONDS,
        "teacher_correction_scale": TEACHER_CORRECTION_SCALE,
        "teacher_loss": "weighted standardized Gaussian NLL up to a constant",
        "teacher_normalization": "valid teacher transition count per minibatch",
        "empty_teacher_minibatch": "exact differentiable zero",
    }


def calibration_evaluation_seed(candidate_index: int, repeat: int) -> int:
    if not 0 <= candidate_index < len(CALIBRATION_GAINS):
        raise ValueError("candidate index outside the frozen grid")
    if not 0 <= repeat < CALIBRATION_REPEATS:
        raise ValueError("calibration repeat outside the frozen schedule")
    return CALIBRATION_SEED_BASE + candidate_index * CALIBRATION_SEED_STRIDE + repeat


def final_evaluation_seed(repeat: int) -> int:
    if not 0 <= repeat < FINAL_REPEATS:
        raise ValueError("final repeat outside the frozen schedule")
    return FINAL_SEED_BASE + repeat


def all_v25_fresh_execution_seeds() -> list[int]:
    """Return condition identities; off/on arms intentionally share each seed."""
    values = [
        calibration_evaluation_seed(candidate, repeat)
        for candidate in range(len(CALIBRATION_GAINS))
        for repeat in range(CALIBRATION_REPEATS)
    ]
    values.append(ADAPTATION_SEED)
    values.extend(final_evaluation_seed(repeat) for repeat in range(FINAL_REPEATS))
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
    historical: set[int] = set()
    scanned: list[dict[str, str]] = []
    results_root = repo / "results/online"
    if results_root.is_dir():
        for path in sorted(results_root.rglob("*.json")):
            if "proximal_v25" in path.parts:
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
    proposed = all_v25_fresh_execution_seeds()
    counts = Counter(proposed)
    internal = sorted(seed for seed, count in counts.items() if count > 1)
    historical_collisions = sorted(set(proposed) & historical)
    collisions = sorted(set(internal) | set(historical_collisions))
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "historical_scope": "all parseable JSON below results/online except proximal_v25",
        "historical_json_file_count": len(scanned),
        "historical_json_files_sha256": canonical_sha256({"files": scanned}),
        "proposed_seed_occurrence_count": len(proposed),
        "proposed_unique_seed_count": len(set(proposed)),
        "paired_arm_seed_reuse": "intentional within each off/on condition identity",
        "proposed_internal_collisions": internal,
        "historical_collisions": historical_collisions,
        "collisions": collisions,
        "passed": not collisions,
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
        raise ValueError("calibration counts must be non-negative integers")
    if paired_count < 1:
        raise ValueError("paired calibration count must be positive")
    if off_failure_count != paired_count - off_success_count:
        raise ValueError("off failure count must equal paired count minus successes")
    if on_success_count > paired_count:
        raise ValueError("on success count exceeds paired count")
    if off_toe_riser_failure_count > off_failure_count:
        raise ValueError("aligned failures exceed all off failures")
    if rescued_count > off_failure_count:
        raise ValueError("rescues exceed off failures")
    off_success_rate = off_success_count / paired_count
    on_success_rate = on_success_count / paired_count
    alignment_coverage = off_toe_riser_failure_count / max(1, off_failure_count)
    shield_rescue_rate = rescued_count / max(1, off_failure_count)
    conditions = {
        "alignment_coverage_at_least_80pct": (
            off_failure_count > 0 and alignment_coverage >= ALIGNMENT_COVERAGE_MINIMUM
        ),
        "shield_rescue_rate_at_least_60pct": (
            off_failure_count > 0 and shield_rescue_rate >= SHIELD_RESCUE_MINIMUM
        ),
        "off_success_in_40_to_65pct": (
            CALIBRATION_OFF_SUCCESS_BOUNDS[0]
            <= off_success_rate
            <= CALIBRATION_OFF_SUCCESS_BOUNDS[1]
        ),
        "on_success_in_80_to_95pct": (
            CALIBRATION_ON_SUCCESS_BOUNDS[0]
            <= on_success_rate
            <= CALIBRATION_ON_SUCCESS_BOUNDS[1]
        ),
    }
    return {
        "qualifies": all(conditions.values()),
        "conditions": conditions,
        "paired_count": paired_count,
        "off_success_count": off_success_count,
        "off_success_rate": off_success_rate,
        "on_success_count": on_success_count,
        "on_success_rate": on_success_rate,
        "off_failure_count": off_failure_count,
        "off_toe_riser_failure_count": off_toe_riser_failure_count,
        "alignment_coverage": alignment_coverage,
        "rescued_count": rescued_count,
        "shield_rescue_rate": shield_rescue_rate,
    }


def development_gate(
    *,
    off_success_delta: float,
    on_success_delta: float,
    base_off_kick_rate: float,
    final_off_kick_rate: float,
    base_on_intervention_per_riser: float,
    final_on_intervention_per_riser: float,
) -> dict[str, Any]:
    values = (
        off_success_delta,
        on_success_delta,
        base_off_kick_rate,
        final_off_kick_rate,
        base_on_intervention_per_riser,
        final_on_intervention_per_riser,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("v25 development metrics must be finite")
    if min(values[2:]) < 0.0:
        raise ValueError("v25 rates must be non-negative")
    intervention_reduction = (
        (base_on_intervention_per_riser - final_on_intervention_per_riser)
        / base_on_intervention_per_riser
        if base_on_intervention_per_riser > 0.0
        else float("-inf")
    )
    conditions = {
        "off_success_delta_at_least_five_pp": (
            off_success_delta >= MINIMUM_OFF_SUCCESS_DELTA
        ),
        "on_success_delta_nonnegative": on_success_delta >= MINIMUM_ON_SUCCESS_DELTA,
        "off_policy_kick_rate_strictly_decreases": (
            final_off_kick_rate < base_off_kick_rate
        ),
        "shield_interventions_per_riser_decrease_at_least_20pct": (
            intervention_reduction >= MINIMUM_INTERVENTION_REDUCTION
            or math.isclose(
                intervention_reduction,
                MINIMUM_INTERVENTION_REDUCTION,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "off_success_delta": off_success_delta,
        "on_success_delta": on_success_delta,
        "base_off_kick_rate": base_off_kick_rate,
        "final_off_kick_rate": final_off_kick_rate,
        "base_on_intervention_per_riser": base_on_intervention_per_riser,
        "final_on_intervention_per_riser": final_on_intervention_per_riser,
        "intervention_per_riser_relative_reduction": intervention_reduction,
        "point_estimates_only": True,
        "used_for_training_selection_or_rollback": False,
    }


def paired_repair_regression_counts(
    baseline: Sequence[bool], final: Sequence[bool]
) -> dict[str, Any]:
    if not baseline or len(baseline) != len(final):
        raise ValueError("paired vectors must be non-empty and equally sized")
    repairs = sum(
        (not bool(old)) and bool(new) for old, new in zip(baseline, final, strict=True)
    )
    regressions = sum(
        bool(old) and (not bool(new)) for old, new in zip(baseline, final, strict=True)
    )
    return {
        "paired_conditions": len(baseline),
        "repair_count": repairs,
        "regression_count": regressions,
        "net_success_change": repairs - regressions,
    }


def validate_v25_calibrated_context(context: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(context)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported v25 context schema")
    if payload.get("context_id") != CONTEXT_ID:
        raise ValueError("wrong v25 context id")
    shift = payload.get("shift", {})
    required_shift = {
        "family": CONTEXT_FAMILY,
        "phase_selective": True,
        "affected_joint_suffixes": [
            "hip_pitch_joint",
            "knee_joint",
            "ankle_pitch_joint",
        ],
        "stance_leg_gain": 1.0,
        "other_joint_gain": 1.0,
        "terrain_geometry": "nominal_fixed_DQHMED",
        "friction": "nominal",
        "command": "nominal",
        "controller": "nominal",
        "observation_interface": "original_405D",
        "cbf_geometry": "exact_generated_riser_metadata",
    }
    for key, expected in required_shift.items():
        if shift.get(key) != expected:
            raise ValueError(f"v25 context changes forbidden field {key!r}")
    index = shift.get("selected_candidate_index")
    gain = shift.get("swing_underresponse_gain")
    if not isinstance(index, int) or not 0 <= index < len(CALIBRATION_GAINS):
        raise ValueError("invalid selected v25 candidate index")
    if not math.isclose(
        float(gain), CALIBRATION_GAINS[index], rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("selected v25 gain does not match the frozen grid")
    calibration = payload.get("calibration", {})
    if calibration.get("base_policy_only") is not True:
        raise ValueError("v25 calibration must be base-policy only")
    if calibration.get("adapted_policy_evaluations_used") is not False:
        raise ValueError("adapted outcomes cannot select the v25 context")
    attempts = calibration.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != index + 1:
        raise ValueError(
            "v25 context must contain every attempt through first qualifier"
        )
    if [attempt.get("candidate_index") for attempt in attempts] != list(
        range(index + 1)
    ):
        raise ValueError("v25 calibration attempts are not ordered")
    if any(bool(attempt.get("qualifies")) for attempt in attempts[:-1]):
        raise ValueError("v25 selected candidate is not the first qualifier")
    if not bool(attempts[-1].get("qualifies")):
        raise ValueError("selected v25 candidate does not qualify")
    if attempts[-1].get("swing_underresponse_gain") != gain:
        raise ValueError("selected v25 attempt/gain mismatch")
    parameters = {
        "context_id": CONTEXT_ID,
        "family": CONTEXT_FAMILY,
        "selected_candidate_index": index,
        "swing_underresponse_gain": gain,
        "phase_selective": True,
        "affected_joint_suffixes": required_shift["affected_joint_suffixes"],
        "stance_leg_gain": 1.0,
        "other_joint_gain": 1.0,
    }
    if payload.get("parameters_sha256") != canonical_sha256(parameters):
        raise ValueError("v25 context parameter hash mismatch")
    return payload
