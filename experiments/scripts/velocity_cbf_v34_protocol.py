"""Fixed development protocol for v34 outcome-optimized velocity CBF."""

from __future__ import annotations

import math
import random
from typing import Any

from cbf_teacher_v31_protocol import (
    BASE_CHECKPOINT_SHA256,
    CONTEXTS,
    FORMAL_CONTEXTS,
    TASK_ID,
    arm_parameters,
    common_training_parameters,
    environment_parameters,
)

PROTOCOL_ID = "safe100-outcome-optimized-task-metric-velocity-cbf-v34"
EXPERIMENT_NAME = "v34 Outcome-Optimized Task-Metric Velocity CBF"
CURRENT_CBF_MODE = "current_velocity_cbf"
OPTIMIZED_CBF_MODE = "task_metric_velocity_cbf"

SEARCH_RANDOM_SEED = 34_000_001
NUM_CANDIDATES = 60
STAGE1_EPISODES = 64
STAGE2_EPISODES = 256
STAGE2_TOP_K = 8
TRAIN_TOP_K = 2
TRAINED_DEVELOPMENT_EPISODES = 256
SMOKE_ENVS = 8
SMOKE_STEPS = 256
SMOKE_SEED = 234_100_001
PREFERRED_EVAL_BATCH = 256
MAX_DEVELOPMENT_GPU_HOURS = 12.0

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

PARAMETER_RANGES = {
    "barrier_slope": [0.4, 1.2],
    "alpha": [4.0, 16.0],
    "swing_knee_weight": [0.25, 2.0],
    "swing_ankle_pitch_weight": [0.25, 2.0],
    "swing_hip_pitch_weight": [0.5, 4.0],
    "stance_leg_weight": [2.0, 12.0],
    "hip_roll_yaw_weight": [2.0, 12.0],
    "other_joint_weight": [1.0, 8.0],
    "lambda_x": [0.0, 16.0],
    "lambda_s": [0.0, 1.0],
    "top_clearance": [0.015, 0.045],
    "toe_margin": [0.04, 0.10],
}

BASELINE_PARAMETERS: dict[str, float] = {
    "barrier_slope": 0.8,
    "alpha": 10.0,
    "swing_knee_weight": 1.0,
    "swing_ankle_pitch_weight": 1.0,
    "swing_hip_pitch_weight": 1.0,
    "stance_leg_weight": 1.0,
    "hip_roll_yaw_weight": 1.0,
    "other_joint_weight": 1.0,
    "lambda_x": 0.0,
    "lambda_s": 0.0,
    "top_clearance": 0.025,
    "toe_margin": 0.08,
}

SEARCH_CENTER_PARAMETERS: dict[str, float] = {
    **BASELINE_PARAMETERS,
    "stance_leg_weight": 2.0,
    "hip_roll_yaw_weight": 2.0,
}

SOURCE_FILES = (
    "src/tasks/stairs_cbf/velocity_cbf_math.py",
    "src/tasks/stairs_cbf/velocity_cbf_action.py",
    "experiments/scripts/velocity_cbf_v34_protocol.py",
    "experiments/scripts/freeze_velocity_cbf_v34.py",
    "experiments/scripts/evaluate_velocity_cbf_v34.py",
    "experiments/scripts/smoke_velocity_cbf_v34.py",
    "experiments/scripts/optimize_velocity_cbf_v34.py",
    "experiments/scripts/refine_velocity_cbf_v34.py",
    "experiments/scripts/select_velocity_cbf_v34.py",
    "experiments/scripts/freeze_velocity_cbf_v34_final.py",
    "experiments/scripts/audit_velocity_cbf_v34.py",
    "experiments/scripts/package_velocity_cbf_v34.py",
    "experiments/scripts/run_velocity_cbf_v34.sh",
    "experiments/tests/test_velocity_cbf_v34.py",
)


def _rounded(parameters: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in parameters.items()}


def _anchor(**changes: float) -> dict[str, float]:
    parameters = dict(SEARCH_CENTER_PARAMETERS)
    parameters.update(changes)
    return _rounded(parameters)


def _log_uniform(rng: random.Random, lower: float, upper: float) -> float:
    return math.exp(rng.uniform(math.log(lower), math.log(upper)))


def candidate_grid() -> list[dict[str, Any]]:
    """Return one current control plus 59 prospectively fixed candidates."""
    anchors = [_anchor(alpha=value) for value in (6.0, 8.0, 12.0, 14.0, 16.0)]
    anchors += [_anchor(barrier_slope=value) for value in (0.5, 0.65, 1.0, 1.15)]
    anchors += [_anchor(top_clearance=value) for value in (0.015, 0.02, 0.03, 0.04)]
    anchors += [_anchor(toe_margin=value) for value in (0.04, 0.06, 0.10)]
    metric = {
        "swing_knee_weight": 0.5,
        "swing_ankle_pitch_weight": 0.5,
        "swing_hip_pitch_weight": 1.0,
        "stance_leg_weight": 4.0,
        "hip_roll_yaw_weight": 4.0,
        "other_joint_weight": 2.0,
    }
    anchors += [
        _anchor(**metric),
        _anchor(**metric, lambda_x=2.0),
        _anchor(**metric, lambda_x=8.0),
        _anchor(**metric, lambda_s=0.05),
        _anchor(**metric, lambda_s=0.2),
        _anchor(**metric, alpha=8.0, barrier_slope=0.65),
        _anchor(**metric, alpha=12.0, top_clearance=0.03),
        _anchor(
            swing_knee_weight=0.25,
            swing_ankle_pitch_weight=0.25,
            swing_hip_pitch_weight=0.5,
            stance_leg_weight=8.0,
            hip_roll_yaw_weight=8.0,
            other_joint_weight=4.0,
        ),
    ]
    rng = random.Random(SEARCH_RANDOM_SEED)
    random_candidates: list[dict[str, float]] = []
    while len(anchors) + len(random_candidates) < NUM_CANDIDATES - 1:
        parameters = {
            "barrier_slope": rng.uniform(0.4, 1.2),
            "alpha": rng.uniform(4.0, 16.0),
            "swing_knee_weight": _log_uniform(rng, 0.25, 2.0),
            "swing_ankle_pitch_weight": _log_uniform(rng, 0.25, 2.0),
            "swing_hip_pitch_weight": _log_uniform(rng, 0.5, 4.0),
            "stance_leg_weight": _log_uniform(rng, 2.0, 12.0),
            "hip_roll_yaw_weight": _log_uniform(rng, 2.0, 12.0),
            "other_joint_weight": _log_uniform(rng, 1.0, 8.0),
            "lambda_x": 0.0 if rng.random() < 0.30 else rng.uniform(0.0, 16.0),
            "lambda_s": (0.0 if rng.random() < 0.35 else _log_uniform(rng, 0.01, 1.0)),
            "top_clearance": rng.uniform(0.015, 0.045),
            "toe_margin": rng.uniform(0.04, 0.10),
        }
        random_candidates.append(_rounded(parameters))
    candidates: list[dict[str, Any]] = [
        {
            "candidate": "c000_current",
            "candidate_index": 0,
            "mode": CURRENT_CBF_MODE,
            **BASELINE_PARAMETERS,
        }
    ]
    for index, parameters in enumerate(anchors + random_candidates, 1):
        candidates.append(
            {
                "candidate": f"c{index:03d}",
                "candidate_index": index,
                "mode": OPTIMIZED_CBF_MODE,
                **parameters,
            }
        )
    if len(candidates) != NUM_CANDIDATES:
        raise RuntimeError("v34 candidate generator count differs")
    return candidates


def stage1_seed(context: str) -> int:
    return 234_110_000 + 100 * (FORMAL_CONTEXTS.index(context) + 1)


def stage2_seed(context: str) -> int:
    return 234_120_000 + 100 * (FORMAL_CONTEXTS.index(context) + 1)


def training_seed(candidate_index: int, context: str) -> int:
    return (
        234_200_000
        + 10_000 * candidate_index
        + 100 * (FORMAL_CONTEXTS.index(context) + 1)
    )


def trained_development_seed(context: str) -> int:
    return 234_300_000 + 100 * (FORMAL_CONTEXTS.index(context) + 1)


def frozen_search_specification() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "main_objective": "mean F1/F2/F3 deterministic CBF-on success",
        "development_target_delta_pp": 3.0,
        "candidate_generator": {
            "method": "fixed-seed random search with local control anchors",
            "random_seed": SEARCH_RANDOM_SEED,
            "candidate_count_including_current_CBF0": NUM_CANDIDATES,
            "ranges": PARAMETER_RANGES,
            "candidates": candidate_grid(),
        },
        "stage1": {
            "episodes_per_candidate_context": STAGE1_EPISODES,
            "contexts": list(FORMAL_CONTEXTS),
            "ranking": "mean success only",
        },
        "stage2": {
            "top_k": STAGE2_TOP_K,
            "episodes_per_candidate_context": STAGE2_EPISODES,
            "contexts": list(FORMAL_CONTEXTS),
            "ranking": "mean success only",
        },
        "stage3": {
            "top_k": TRAIN_TOP_K,
            "training_runs": TRAIN_TOP_K * len(FORMAL_CONTEXTS),
            "trained_development_episodes_per_context": TRAINED_DEVELOPMENT_EPISODES,
            "ranking": "trained mean success only",
            "training": common_training_parameters(),
            "A2": arm_parameters("A2"),
        },
        "contexts": {name: environment_parameters(name) for name in CONTEXTS},
        "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "v31_checkpoint_sha256": V31_CHECKPOINT_SHA256,
        "final_seeds_created": False,
        "final_identities_accessible_during_development": False,
        "maximum_development_gpu_hours": MAX_DEVELOPMENT_GPU_HOURS,
        "selection_gates": False,
    }
