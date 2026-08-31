"""Prospectively fixed constants for the v30 overnight paper experiment."""

from __future__ import annotations

from typing import Any

PROTOCOL_ID = "safe100-cbf-teacher-overnight-v30"
EXPERIMENT_NAME = "v30 Overnight Paper-Grade CBF-Teacher Experiment"
POLICY_METHOD = "Raw-Action PPO + Soft Moving-KL + Corrective Teacher"
TASK_ID = "Unitree-G1-Stairs-Online-DQHMED"
BASE_CHECKPOINT_SHA256 = (
    "cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8"
)

CLEARANCE_BARRIER_SLOPE = 0.8
RECOVERY_DISTANCE_M = 0.15
FILTER_ALPHA = 10.0
D0_RISER_HEIGHT_M = 0.13
F3_PROFILE_M = (
    0.176,
    0.180,
    0.184,
    0.180,
    0.178,
    0.182,
    0.180,
    0.184,
    0.178,
    0.182,
    0.180,
)

ROUNDS = 8
NUM_ENVS = 64
ROLLOUT_STEPS = 1024
ACTOR_LEARNING_RATE = 5.0e-6
CRITIC_LEARNING_RATE = 1.0e-4
PPO_CLIP = 0.05
ACTOR_EPOCHS = 2
CRITIC_EPOCHS = 2
MINI_BATCHES = 4
MOVING_KL_BETA = 0.5
MAX_GRAD_NORM = 0.5
STD_SCALE_FROM_BASE = 0.35
MINIMUM_STD = 0.05
MAXIMUM_STD = 0.25
ENTROPY_COEFFICIENT = 0.0
GAMMA = 0.99
GAE_LAMBDA = 0.95
TEACHER_HORIZON = 50
TEACHER_CORRECTION_SCALE = 0.05
TEACHER_EPSILON = 1.0e-8
SMOOTH_L1_BETA = 0.05

ARMS: dict[str, dict[str, Any]] = {
    "A0": {
        "name": "no_teacher_control",
        "teacher_mode": "none",
        "teacher_gate": "none",
        "teacher_eta": 0.0,
        "teacher_loss": "none",
        "teacher_weight": 0.0,
    },
    "A1": {
        "name": "v29_full_action_nll_no_kl_rollback",
        "teacher_mode": "full_action",
        "teacher_gate": "local_success_50",
        "teacher_eta": 1.0,
        "teacher_loss": "frozen_std_gaussian_nll",
        "teacher_weight": 0.1,
    },
    "A2": {
        "name": "residual_eta_025_all_interventions",
        "teacher_mode": "residual",
        "teacher_gate": "all_interventions",
        "teacher_eta": 0.25,
        "teacher_loss": "weighted_smooth_l1_per_action_mean",
        "teacher_weight": 1.0,
    },
    "A3": {
        "name": "residual_eta_050_all_interventions",
        "teacher_mode": "residual",
        "teacher_gate": "all_interventions",
        "teacher_eta": 0.50,
        "teacher_loss": "weighted_smooth_l1_per_action_mean",
        "teacher_weight": 1.0,
    },
    "A4": {
        "name": "residual_eta_100_all_interventions",
        "teacher_mode": "residual",
        "teacher_gate": "all_interventions",
        "teacher_eta": 1.00,
        "teacher_loss": "weighted_smooth_l1_per_action_mean",
        "teacher_weight": 1.0,
    },
    "A5": {
        "name": "residual_eta_050_local_success_50",
        "teacher_mode": "residual",
        "teacher_gate": "local_success_50",
        "teacher_eta": 0.50,
        "teacher_loss": "weighted_smooth_l1_per_action_mean",
        "teacher_weight": 1.0,
    },
}
TEACHER_ARMS = ("A1", "A2", "A3", "A4", "A5")

CONTEXTS: dict[str, dict[str, Any]] = {
    "DEV": {
        "name": "development_uniform_18cm",
        "riser_height_m": 0.180,
        "riser_profile_m": None,
    },
    "F1": {
        "name": "main_uniform_higher_riser",
        "riser_height_m": 0.180,
        "riser_profile_m": None,
    },
    "F2": {
        "name": "hard_uniform_higher_riser",
        "riser_height_m": 0.184,
        "riser_profile_m": None,
    },
    "F3": {
        "name": "nonuniform_higher_riser_profile",
        "riser_height_m": None,
        "riser_profile_m": F3_PROFILE_M,
    },
    "D0": {
        "name": "nominal_uniform_13cm",
        "riser_height_m": D0_RISER_HEIGHT_M,
        "riser_profile_m": None,
    },
}
FORMAL_CONTEXTS = ("F1", "F2", "F3")

SMOKE_SEED = 158_290_001
SMOKE_ENVS = 8
SMOKE_STEPS = 128
DEVELOPMENT_ADAPTATION_SEED = 171_300_001
DEVELOPMENT_TARGET_SEED_BASE = 172_300_000
DEVELOPMENT_D0_SEED_BASE = 173_300_000
FORMAL_ADAPTATION_SEEDS = {
    "F1": 174_301_001,
    "F2": 174_302_001,
    "F3": 174_303_001,
}
FORMAL_TARGET_SEED_BASES = {
    "F1": 175_301_000,
    "F2": 175_302_000,
    "F3": 175_303_000,
}
FORMAL_D0_SEED_BASES = {
    "F1": 176_301_000,
    "F2": 176_302_000,
    "F3": 176_303_000,
}
MONITOR_SEED = 177_301_000
BOOTSTRAP_SEED_BASE = 178_300_000

DEVELOPMENT_TARGET_EPISODES = 256
DEVELOPMENT_D0_EPISODES = 128
FORMAL_TARGET_EPISODES = 512
FORMAL_D0_EPISODES = 256
MONITOR_EPISODES = 128
PREFERRED_EVAL_BATCH_SIZE = 256
FALLBACK_EVAL_BATCH_SIZE = 128
FORMAL_BOOTSTRAP_SAMPLES = 2_000

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
    "src/tasks/stairs_cbf/environment_v30.py",
    "src/tasks/stairs_cbf/terrain.py",
    "experiments/scripts/proximal_v23_io.py",
    "experiments/scripts/cbf_teacher_v30_protocol.py",
    "experiments/scripts/freeze_cbf_teacher_v30.py",
    "experiments/scripts/refine_cbf_teacher_v30.py",
    "experiments/scripts/evaluate_cbf_teacher_v30.py",
    "experiments/scripts/cbf_teacher_v30_eval_io.py",
    "experiments/scripts/audit_cbf_teacher_v30_development.py",
    "experiments/scripts/audit_cbf_teacher_v30_formal.py",
    "experiments/scripts/monitor_cbf_teacher_v30.py",
    "experiments/scripts/package_cbf_teacher_v30.py",
    "experiments/scripts/run_cbf_teacher_v30.sh",
    "experiments/tests/test_cbf_teacher_v30.py",
)


def common_training_parameters() -> dict[str, Any]:
    return {
        "online_rounds": ROUNDS,
        "num_envs": NUM_ENVS,
        "rollout_steps_per_round": ROLLOUT_STEPS,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "ppo_clip": PPO_CLIP,
        "actor_epochs_exact": ACTOR_EPOCHS,
        "critic_epochs": CRITIC_EPOCHS,
        "minibatches": MINI_BATCHES,
        "moving_kl_beta": MOVING_KL_BETA,
        "maximum_gradient_norm": MAX_GRAD_NORM,
        "freeze_log_std": True,
        "std_scale_from_base": STD_SCALE_FROM_BASE,
        "minimum_std": MINIMUM_STD,
        "maximum_std": MAXIMUM_STD,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "target_kl_early_stopping": False,
        "hard_kl_rollback": False,
        "performance_selection_or_rollback": False,
        "save_round_checkpoints": list(range(ROUNDS + 1)),
    }


def environment_parameters(context: str) -> dict[str, Any]:
    specification = CONTEXTS[context]
    return {
        "task_id": TASK_ID,
        "context": context,
        "name": specification["name"],
        "riser_height_m": specification["riser_height_m"],
        "riser_profile_m": (
            None
            if specification["riser_profile_m"] is None
            else list(specification["riser_profile_m"])
        ),
        "clearance_barrier_slope": CLEARANCE_BARRIER_SLOPE,
        "post_edge_recovery_window_m": RECOVERY_DISTANCE_M,
        "exponential_cbf_alpha": FILTER_ALPHA,
        "runtime_cbf_during_training": "always_on",
        "plant_action_transform": "identity",
        "observation_interface": "original_405D_actor_838D_critic",
        "all_non_riser_factors": "nominal",
    }


def arm_parameters(arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown v30 arm {arm!r}")
    return dict(ARMS[arm])


def adaptation_seed(phase: str, context: str) -> int:
    if phase == "development" and context == "DEV":
        return DEVELOPMENT_ADAPTATION_SEED
    if phase == "formal" and context in FORMAL_ADAPTATION_SEEDS:
        return FORMAL_ADAPTATION_SEEDS[context]
    raise ValueError(f"no fixed v30 adaptation seed for {phase}/{context}")
