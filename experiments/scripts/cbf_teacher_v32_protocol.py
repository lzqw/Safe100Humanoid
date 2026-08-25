"""Prospectively fixed constants for v32 long-horizon continual refinement."""

from __future__ import annotations

from typing import Any

from cbf_teacher_v31_protocol import (
    ACTOR_EPOCHS,
    BASE_CHECKPOINT_SHA256,
    BEHAVIOR_LOG_PROB_ATOL,
    CRITIC_EPOCHS,
    ENTROPY_COEFFICIENT,
    FORMAL_CONTEXTS,
    GAE_LAMBDA,
    GAMMA,
    MAX_GRAD_NORM,
    MAXIMUM_STD,
    MINI_BATCHES,
    MINIMUM_STD,
    MOVING_KL_BETA,
    NUM_ENVS,
    PPO_CLIP,
    ROLLOUT_STEPS,
    SMOOTH_L1_BETA,
    STD_SCALE_FROM_BASE,
    TEACHER_CORRECTION_SCALE,
    TEACHER_EPSILON,
    TEACHER_HORIZON,
)
from cbf_teacher_v31_protocol import (
    arm_parameters as v31_arm_parameters,
)
from cbf_teacher_v31_protocol import (
    environment_parameters as v31_environment_parameters,
)

PROTOCOL_ID = "safe100-cbf-protected-continual-refinement-v32"
EXPERIMENT_NAME = "v32 Long-Horizon CBF-Protected Continual Refinement"
POLICY_METHOD = "v31 A2 Raw-Action PPO + Soft Moving-KL + CBF Correction"

SCHEDULE_LONG_CONSTANT = "LongConstant"
SCHEDULE_LONG_DECAY = "LongDecay"
CONTINUATION_SCHEDULES = (SCHEDULE_LONG_CONSTANT, SCHEDULE_LONG_DECAY)
MIXED_SCHEDULE = SCHEDULE_LONG_DECAY

CONTINUATION_START_ROUND = 8
CONTINUATION_FINAL_ROUND = 24
CONTINUATION_ADDED_ROUNDS = 16
MIXED_START_ROUND = 0
MIXED_FINAL_ROUND = 24
MONITOR_ROUNDS = (8, 16, 24)

ACTOR_LR_FULL = 5.0e-6
CRITIC_LR_FULL = 1.0e-4
ACTOR_LR_HALF = 2.5e-6
CRITIC_LR_HALF = 5.0e-5
ACTOR_LR_LOW = 1.0e-6
CRITIC_LR_LOW = 2.5e-5

V31_A2_ROUND8_SHA256 = {
    "F1": "65ebc05a007fda98076b4c671028cf3ce2026db0fc0da5ac6c056c207363967f",
    "F2": "00ed5967ebebe8f60832c53030f586e8a4fb39b9ad8ee3df70acdf00ba20abdb",
    "F3": "8fc5134f0bf083c09e259f61acb72ea85427fc3053335d665e0992bf6ed06e64",
}

CONTINUATION_SEEDS = {
    "F1": {SCHEDULE_LONG_CONSTANT: 191_410_001, SCHEDULE_LONG_DECAY: 191_410_002},
    "F2": {SCHEDULE_LONG_CONSTANT: 191_420_001, SCHEDULE_LONG_DECAY: 191_420_002},
    "F3": {SCHEDULE_LONG_CONSTANT: 191_430_001, SCHEDULE_LONG_DECAY: 191_430_002},
}
MIXED_SEED = 191_440_001
PREFLIGHT_CASES = (
    {
        "kind": "continuation",
        "context": "F1",
        "schedule": SCHEDULE_LONG_DECAY,
        "seed": 191_400_101,
    },
    {
        "kind": "mixed",
        "context": "mixed",
        "schedule": SCHEDULE_LONG_DECAY,
        "seed": 191_400_102,
    },
)

FORMAL_TARGET_SEED_BASES = {
    "F1": 192_410_000,
    "F2": 192_420_000,
    "F3": 192_430_000,
}
FORMAL_D0_SEED_BASES = {
    "F1": 193_410_000,
    "F2": 193_420_000,
    "F3": 193_430_000,
}
MIXED_D0_SEED_BASE = 193_440_000
MONITOR_SEED_BASES = {
    "F1": 194_410_000,
    "F2": 194_420_000,
    "F3": 194_430_000,
}
BOOTSTRAP_SEED_BASE = 195_410_000

FORMAL_TARGET_EPISODES = 512
FORMAL_D0_EPISODES = 256
MONITOR_EPISODES = 128
PREFERRED_EVAL_BATCH_SIZE = 256
FORMAL_BOOTSTRAP_SAMPLES = 2_000
PREFLIGHT_CONTINUATION_ENVS = 8
PREFLIGHT_STEPS = 128

MIXED_CONTEXT_CAPACITY = 22
MIXED_EXPOSED_ENVS = 64
MIXED_CONTEXTS = FORMAL_CONTEXTS

SOURCE_FILES = (
    "src/tasks/stairs_cbf/actions.py",
    "src/tasks/stairs_cbf/cbf_math.py",
    "src/tasks/stairs_cbf/config.py",
    "src/tasks/stairs_cbf/edge_detection.py",
    "src/tasks/stairs_cbf/mdp.py",
    "src/tasks/stairs_cbf/online.py",
    "src/tasks/stairs_cbf/proximal.py",
    "src/tasks/stairs_cbf/teacher.py",
    "src/tasks/stairs_cbf/teacher_math.py",
    "src/tasks/stairs_cbf/teacher_v26.py",
    "src/tasks/stairs_cbf/teacher_v29.py",
    "src/tasks/stairs_cbf/teacher_v30.py",
    "src/tasks/stairs_cbf/teacher_v30_math.py",
    "src/tasks/stairs_cbf/environment_v31.py",
    "src/tasks/stairs_cbf/terrain.py",
    "experiments/scripts/proximal_v23_io.py",
    "experiments/scripts/cbf_teacher_v31_protocol.py",
    "experiments/scripts/refine_cbf_teacher_v31.py",
    "experiments/scripts/evaluate_cbf_teacher_v31.py",
    "experiments/scripts/cbf_teacher_v31_eval_io.py",
    "experiments/scripts/cbf_teacher_v32_protocol.py",
    "experiments/scripts/mixed_vec_env_v32.py",
    "experiments/scripts/freeze_cbf_teacher_v32.py",
    "experiments/scripts/refine_cbf_teacher_v32.py",
    "experiments/scripts/preflight_cbf_teacher_v32.py",
    "experiments/scripts/evaluate_cbf_teacher_v32.py",
    "experiments/scripts/cbf_teacher_v32_eval_io.py",
    "experiments/scripts/monitor_cbf_teacher_v32.py",
    "experiments/scripts/audit_cbf_teacher_v32_formal.py",
    "experiments/scripts/package_cbf_teacher_v32.py",
    "experiments/scripts/run_cbf_teacher_v32.sh",
    "experiments/tests/test_cbf_teacher_v32.py",
)


def a2_parameters() -> dict[str, Any]:
    """Return the unchanged v31 A2 teacher configuration."""
    return v31_arm_parameters("A2")


def environment_parameters(context: str) -> dict[str, Any]:
    if context not in (*FORMAL_CONTEXTS, "D0"):
        raise ValueError(f"unknown v32 environment context {context!r}")
    return v31_environment_parameters(context)


def continuation_seed(context: str, schedule: str) -> int:
    try:
        return CONTINUATION_SEEDS[context][schedule]
    except KeyError as error:
        raise ValueError(
            f"unknown v32 continuation run {context}/{schedule}"
        ) from error


def learning_rates(
    kind: str, schedule: str, absolute_round: int
) -> tuple[float, float]:
    """Return the frozen per-round actor and critic learning rates."""
    if kind == "continuation":
        if schedule not in CONTINUATION_SCHEDULES:
            raise ValueError(f"unknown continuation schedule {schedule!r}")
        if not 9 <= absolute_round <= CONTINUATION_FINAL_ROUND:
            raise ValueError("continuation absolute round must be in [9, 24]")
        if schedule == SCHEDULE_LONG_CONSTANT or absolute_round <= 16:
            return ACTOR_LR_FULL, CRITIC_LR_FULL
        if absolute_round <= 20:
            return ACTOR_LR_HALF, CRITIC_LR_HALF
        return ACTOR_LR_LOW, CRITIC_LR_LOW
    if kind == "mixed":
        if schedule != MIXED_SCHEDULE:
            raise ValueError("v32 mixed policy uses LongDecay only")
        if not 1 <= absolute_round <= MIXED_FINAL_ROUND:
            raise ValueError("mixed absolute round must be in [1, 24]")
        if absolute_round <= 12:
            return ACTOR_LR_FULL, CRITIC_LR_FULL
        if absolute_round <= 18:
            return ACTOR_LR_HALF, CRITIC_LR_HALF
        return ACTOR_LR_LOW, CRITIC_LR_LOW
    raise ValueError(f"unknown v32 run kind {kind!r}")


def mixed_context_env_counts(absolute_round: int) -> dict[str, int]:
    """Rotate the 22nd environment evenly across F1, F2, and F3."""
    if not 1 <= absolute_round <= MIXED_FINAL_ROUND:
        raise ValueError("mixed round must be in [1, 24]")
    extra_context = MIXED_CONTEXTS[(absolute_round - 1) % len(MIXED_CONTEXTS)]
    return {
        context: 22 if context == extra_context else 21 for context in MIXED_CONTEXTS
    }


def common_algorithm_parameters() -> dict[str, Any]:
    return {
        "num_envs": NUM_ENVS,
        "rollout_steps_per_round": ROLLOUT_STEPS,
        "actor_epochs_exact": ACTOR_EPOCHS,
        "critic_epochs": CRITIC_EPOCHS,
        "minibatches": MINI_BATCHES,
        "ppo_clip": PPO_CLIP,
        "moving_kl_beta": MOVING_KL_BETA,
        "maximum_gradient_norm": MAX_GRAD_NORM,
        "freeze_log_std": True,
        "std_scale_from_base": STD_SCALE_FROM_BASE,
        "minimum_std": MINIMUM_STD,
        "maximum_std": MAXIMUM_STD,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "teacher": a2_parameters(),
        "teacher_horizon": TEACHER_HORIZON,
        "teacher_correction_scale": TEACHER_CORRECTION_SCALE,
        "teacher_epsilon": TEACHER_EPSILON,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "behavior_log_prob_atol": BEHAVIOR_LOG_PROB_ATOL,
        "target_kl_early_stopping": False,
        "hard_kl_rollback": False,
        "performance_gate": False,
        "candidate_line_search": False,
        "best_checkpoint_selection": False,
        "final_policy": "unconditional round 24",
    }


def run_matrix() -> list[dict[str, Any]]:
    runs = [
        {
            "kind": "continuation",
            "context": context,
            "schedule": schedule,
            "start_round": CONTINUATION_START_ROUND,
            "final_round": CONTINUATION_FINAL_ROUND,
            "additional_rounds": CONTINUATION_ADDED_ROUNDS,
            "seed": continuation_seed(context, schedule),
            "source_checkpoint_sha256": V31_A2_ROUND8_SHA256[context],
        }
        for context in FORMAL_CONTEXTS
        for schedule in CONTINUATION_SCHEDULES
    ]
    runs.append(
        {
            "kind": "mixed",
            "context": "mixed",
            "schedule": MIXED_SCHEDULE,
            "start_round": MIXED_START_ROUND,
            "final_round": MIXED_FINAL_ROUND,
            "additional_rounds": MIXED_FINAL_ROUND,
            "seed": MIXED_SEED,
            "source_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        }
    )
    return runs
