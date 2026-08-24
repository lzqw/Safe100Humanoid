"""Fixed constants for the single v29 CBF-teacher online experiment."""

from __future__ import annotations

from typing import Any


PROTOCOL_ID = "safe100-cbf-teacher-proximal-online-v29"
EXPERIMENT_NAME = "v29 CBF-Teacher Proximal Online Fine-Tuning"
POLICY_METHOD = "Raw-Action PPO + Moving-KL + Local-Success CBF Teacher"
TASK_ID = "Unitree-G1-Stairs-Online-DQHMED"
ENVIRONMENT_VARIANT = "fixed_deployment_play"
BASE_CHECKPOINT_SHA256 = (
    "cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8"
)

RISER_HEIGHT_M = 0.180
D0_RISER_HEIGHT_M = 0.130
CLEARANCE_BARRIER_SLOPE = 0.80
RECOVERY_DISTANCE_M = 0.15
FILTER_ALPHA = 10.0

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
TEACHER_WEIGHT = 0.1
TEACHER_HORIZON = 50
TEACHER_CORRECTION_SCALE = 0.05
TEACHER_EPSILON = 1.0e-8

SMOKE_SEED = 158_290_001
SMOKE_ENVS = 8
SMOKE_STEPS = 128
ADAPTATION_SEED = 159_290_001
FINAL_SEED_BASE = 160_290_000
D0_SEED_BASE = 161_290_000
FINAL_EPISODES = 512
D0_EPISODES = 256
PREFERRED_EVAL_BATCH_SIZE = 256

CONDITIONS = (
    ("pi0_off", "base", "off"),
    ("pi0_on", "base", "on"),
    ("pi8_on", "final", "on"),
    ("pi8_off", "final", "off"),
)
D0_CONDITIONS = (
    ("pi0_on", "base", "on"),
    ("pi8_on", "final", "on"),
)

SOURCE_FILES = (
    "src/tasks/stairs_cbf/actions.py",
    "src/tasks/stairs_cbf/cbf_math.py",
    "src/tasks/stairs_cbf/config.py",
    "src/tasks/stairs_cbf/edge_detection.py",
    "src/tasks/stairs_cbf/mdp.py",
    "src/tasks/stairs_cbf/online.py",
    "src/tasks/stairs_cbf/proximal.py",
    "src/tasks/stairs_cbf/teacher_math.py",
    "src/tasks/stairs_cbf/teacher.py",
    "src/tasks/stairs_cbf/teacher_v26.py",
    "src/tasks/stairs_cbf/teacher_v29.py",
    "src/tasks/stairs_cbf/terrain.py",
    "experiments/scripts/proximal_v23_io.py",
    "experiments/scripts/evaluate_cbf_teacher_v26.py",
    "experiments/scripts/cbf_teacher_v29_protocol.py",
    "experiments/scripts/freeze_cbf_teacher_v29.py",
    "experiments/scripts/refine_cbf_teacher_v29.py",
    "experiments/scripts/evaluate_cbf_teacher_v29.py",
    "experiments/scripts/audit_cbf_teacher_v29.py",
    "experiments/scripts/package_cbf_teacher_v29.py",
    "experiments/scripts/run_cbf_teacher_v29.sh",
    "experiments/tests/test_cbf_teacher_v29.py",
)


def fixed_environment_parameters() -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "registered_variant": ENVIRONMENT_VARIANT,
        "uniform_riser_height_m": RISER_HEIGHT_M,
        "terrain_profile_other_than_riser_height": "nominal",
        "clearance_barrier": "sloped_xz",
        "clearance_barrier_slope": CLEARANCE_BARRIER_SLOPE,
        "post_edge_recovery_window_m": RECOVERY_DISTANCE_M,
        "exponential_cbf_alpha": FILTER_ALPHA,
        "runtime_cbf_during_training": "always_on",
        "foot_friction": "nominal",
        "action_gain": "nominal_identity",
        "action_delay": "zero",
        "encoder_bias": "absent",
        "command_delay": "zero",
        "command_low_pass": "nominal",
        "lateral_yaw_disturbance": "absent",
        "centerline_controller": "nominal",
        "observation_interface": "original_405D_actor_838D_critic",
        "gait_parameters": "nominal",
        "additional_observation_noise": "absent",
    }


def formal_algorithm_parameters() -> dict[str, Any]:
    return {
        "online_rounds": ROUNDS,
        "num_envs": NUM_ENVS,
        "rollout_steps_per_round": ROLLOUT_STEPS,
        "actor_learning_rate": ACTOR_LEARNING_RATE,
        "critic_learning_rate": CRITIC_LEARNING_RATE,
        "ppo_clip": PPO_CLIP,
        "maximum_actor_epochs": MAX_ACTOR_EPOCHS,
        "critic_epochs": CRITIC_EPOCHS,
        "minibatches": MINI_BATCHES,
        "moving_kl_beta": MOVING_KL_BETA,
        "teacher_weight": TEACHER_WEIGHT,
        "teacher_horizon_steps": TEACHER_HORIZON,
        "teacher_correction_scale": TEACHER_CORRECTION_SCALE,
        "teacher_loss_epsilon": TEACHER_EPSILON,
        "teacher_loss_normalization": "sum_weights_plus_epsilon",
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
        "whole_rollout_advantage_normalization": True,
        "teacher_target_stop_gradient": True,
        "teacher_future_label": "vectorized_up_to_50_full_tensor_offsets",
        "ppo_ratio_action": "raw_policy_action",
        "executed_action": "cbf_safe_action",
        "round_reference": "deepcopy_round_start_actor",
    }


def final_seed(repeat: int) -> int:
    repeats = FINAL_EPISODES // PREFERRED_EVAL_BATCH_SIZE
    if not 0 <= repeat < repeats:
        raise ValueError("v29 final repeat outside preferred schedule")
    return FINAL_SEED_BASE + repeat


def d0_seed(repeat: int) -> int:
    repeats = D0_EPISODES // PREFERRED_EVAL_BATCH_SIZE
    if not 0 <= repeat < repeats:
        raise ValueError("v29 D0 repeat outside preferred schedule")
    return D0_SEED_BASE + repeat
