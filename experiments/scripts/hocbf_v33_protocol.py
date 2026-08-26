"""Prospectively fixed protocol for the independent v33 HOCBF experiment."""

from __future__ import annotations

from itertools import product
from typing import Any

from cbf_teacher_v31_protocol import (
    BASE_CHECKPOINT_SHA256,
    CLEARANCE_BARRIER_SLOPE,
    CONTEXTS,
    FORMAL_CONTEXTS,
    RECOVERY_DISTANCE_M,
    TASK_ID,
    arm_parameters,
    common_training_parameters,
    environment_parameters,
)

PROTOCOL_ID = "safe100-task-consistent-acceleration-hocbf-v33"
EXPERIMENT_NAME = "v33 Task-Consistent Acceleration HOCBF-QP"
CURRENT_CBF_MODE = "current_velocity_cbf"
HOCBF_MODE = "task_consistent_acceleration_hocbf"

ZETA = 1.0
DRIFT_EMA_PREVIOUS = 0.8
DRIFT_CLIP_M_PER_S2 = 20.0
TOP_CLEARANCE_M = 0.025
OMEGAS = (4.0, 8.0, 12.0)
FORWARD_WEIGHTS = (0.0, 8.0, 24.0)
SMOOTHNESS_WEIGHTS = (0.0, 0.1)
JOINT_METRIC = {
    "knee_pitch": 1.0,
    "ankle_pitch": 1.0,
    "hip_pitch": 2.0,
    "hip_roll_yaw_ankle_roll_other": 4.0,
}

SMOKE_ENVS = 8
SMOKE_STEPS = 256
SMOKE_SEED = 201_300_101
SCREEN_CONTEXTS = ("F1", "F2")
SCREEN_POLICIES = ("A1", "A2")
SCREEN_EPISODES = 128
CONFIRM_POLICIES = SCREEN_POLICIES
CONFIRM_EPISODES = 256
TOP_K = 4
FROZEN_POLICY_EPISODES = 512
FINAL_TARGET_EPISODES = 512
FINAL_D0_EPISODES = 256
PREFERRED_EVAL_BATCH_SIZE = 256
BOOTSTRAP_SAMPLES = 2_000
PRIMARY_TIE_TOLERANCE = 0.005
INTERFERENCE_TIE_TOLERANCE = 0.005

TRAINING_SEEDS = {"F1": 205_310_001, "F2": 205_320_001, "F3": 205_330_001}
V31_CHECKPOINT_SHA256 = {
    "F1": {
        "A1": "39b1b8fdcd976e5c7990f380d79e72bee3ffd09c432368dd31c9c753b42e1b2a",
        "A2": "65ebc05a007fda98076b4c671028cf3ce2026db0fc0da5ac6c056c207363967f",
    },
    "F2": {
        "A1": "d5fe4f54531a56d419fb9aa07f33f0251d6371ea6238c5cdd56dbbadef5c5af6",
        "A2": "00ed5967ebebe8f60832c53030f586e8a4fb39b9ad8ee3df70acdf00ba20abdb",
    },
    "F3": {
        "A1": "7efdf3864bb78fb83f334501452d5aacc5ac67094a0811d8c5668c5e8c58c39b",
        "A2": "8fc5134f0bf083c09e259f61acb72ea85427fc3053335d665e0992bf6ed06e64",
    },
}

SOURCE_FILES = (
    "src/tasks/stairs_cbf/hocbf_math.py",
    "src/tasks/stairs_cbf/hocbf_action.py",
    "experiments/scripts/hocbf_v33_protocol.py",
    "experiments/scripts/evaluate_hocbf_v33.py",
    "experiments/scripts/freeze_hocbf_v33.py",
    "experiments/scripts/run_hocbf_v33.py",
    "experiments/scripts/develop_hocbf_v33.py",
    "experiments/scripts/refine_hocbf_v33.py",
    "experiments/scripts/audit_hocbf_v33.py",
    "experiments/scripts/package_hocbf_v33.py",
    "experiments/scripts/run_hocbf_v33.sh",
    "experiments/tests/test_hocbf_v33.py",
)


def candidate_id(omega: float, forward: float, smoothness: float) -> str:
    return f"w{int(omega):02d}_x{int(forward):02d}_s{round(10 * smoothness):02d}"


def candidate_grid() -> list[dict[str, float | str]]:
    return [
        {
            "candidate": candidate_id(omega, forward, smoothness),
            "omega": omega,
            "lambda_x": forward,
            "lambda_s": smoothness,
        }
        for omega, forward, smoothness in product(
            OMEGAS, FORWARD_WEIGHTS, SMOOTHNESS_WEIGHTS
        )
    ]


def screen_seed(policy: str, context: str) -> int:
    return (
        202_300_000
        + 10_000 * (SCREEN_POLICIES.index(policy) + 1)
        + 100 * (SCREEN_CONTEXTS.index(context) + 1)
    )


def confirmation_seed(policy: str, context: str) -> int:
    return (
        203_300_000
        + 10_000 * (CONFIRM_POLICIES.index(policy) + 1)
        + 100 * (FORMAL_CONTEXTS.index(context) + 1)
    )


def frozen_audit_seed(policy: str, context: str) -> int:
    return (
        204_300_000
        + 10_000 * (SCREEN_POLICIES.index(policy) + 1)
        + 100 * (FORMAL_CONTEXTS.index(context) + 1)
    )


def final_target_seed(source_context: str, evaluation_context: str) -> int:
    return (
        206_300_000
        + 10_000 * (FORMAL_CONTEXTS.index(source_context) + 1)
        + 100 * (FORMAL_CONTEXTS.index(evaluation_context) + 1)
    )


def final_d0_seed(source_context: str) -> int:
    return 207_300_000 + 10_000 * (FORMAL_CONTEXTS.index(source_context) + 1)


def bootstrap_seed(context: str, comparison_index: int) -> int:
    return (
        208_300_000 + 10_000 * (FORMAL_CONTEXTS.index(context) + 1) + comparison_index
    )


def frozen_constants() -> dict[str, Any]:
    training = common_training_parameters()
    return {
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "geometry": {
            "clearance_barrier_slope": CLEARANCE_BARRIER_SLOPE,
            "top_clearance_m": TOP_CLEARANCE_M,
            "recovery_distance_m": RECOVERY_DISTANCE_M,
            "existing_riser_metadata": True,
            "existing_selected_swing_foot": True,
        },
        "hocbf": {
            "zeta": ZETA,
            "drift_ema_previous": DRIFT_EMA_PREVIOUS,
            "drift_clip_m_per_s2": DRIFT_CLIP_M_PER_S2,
            "joint_metric": JOINT_METRIC,
            "grid": candidate_grid(),
        },
        "development": {
            "screen_contexts": list(SCREEN_CONTEXTS),
            "policies": list(SCREEN_POLICIES),
            "screen_episodes": SCREEN_EPISODES,
            "confirmation_contexts": list(FORMAL_CONTEXTS),
            "confirmation_episodes": CONFIRM_EPISODES,
            "top_k": TOP_K,
            "primary_tie_tolerance": PRIMARY_TIE_TOLERANCE,
        },
        "formal": {
            "frozen_policy_episodes": FROZEN_POLICY_EPISODES,
            "target_episodes": FINAL_TARGET_EPISODES,
            "D0_episodes": FINAL_D0_EPISODES,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "training": training,
            "training_seeds": TRAINING_SEEDS,
        },
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "v31_checkpoint_sha256": V31_CHECKPOINT_SHA256,
        "contexts": {name: environment_parameters(name) for name in CONTEXTS},
        "A2": arm_parameters("A2"),
        "selection_or_training_gate": False,
    }
