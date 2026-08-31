"""Prospectively fixed development and formal constants for v141."""

from __future__ import annotations

from typing import Any


PROTOCOL_ID = "safe100-autonomous-filter-free-refinement-v141"
TASK_ID = "Unitree-G1-Stairs-Online-DQHMED"
METHOD_ID = "intervention-aware-cbf-distillation-ppo-v141"
BASE_CHECKPOINT_SHA256 = (
    "323f1e00b58d379b8746c0191a44272f2e1df134139050417c56e733cc484728"
)
SPECIALISTS = ("F2", "F3")
RETENTION_CONTEXT = "F1"

NUM_ENVS = 128
ROLLOUT_STEPS = 1024
GENERATION_1_ROUNDS = 2
DEV_EVALUATION_EPISODES = 128
FORMAL_EVALUATION_EPISODES = 512
TRAINING_ACTION_STD = 0.05
BASE_ACTOR_LEARNING_RATE = 2.5e-6
BASE_CRITIC_LEARNING_RATE = 1.0e-4
DEFAULT_MOVING_KL_BETA = 0.5
DEFAULT_TARGET_FRACTION = 0.80
CORRECTION_LOSS_WEIGHTS = (0.05, 0.1, 0.2, 0.4)
CORRECTION_LOSS_WEIGHT_SCHEDULES = ("constant", "front_high_decay")

DEV_TRAIN_SEED = 201_411_000
DEV_EVALUATION_SEED = 201_411_900
FORMAL_TRAINING_SEEDS = (201_412_000, 201_412_001, 201_412_002)
FORMAL_EVALUATION_SEED = 201_412_900

DEVELOPMENT_THRESHOLDS = {
    "target_off_improvement_pp": 2.0,
    "shield_gap_pp": 2.0,
    "shield_gap_reduction_fraction": 0.50,
    "would_intervene_reduction_fraction": 0.25,
    "f1_off_retention_loss_pp": 1.5,
}


def correction_loss_weight_for_round(
    base_weight: float, schedule: str, round_index: int
) -> float:
    """Return the prospectively configured v141 coefficient for one round.

    ``front_high_decay`` holds the requested coefficient for the first two
    rounds, then moves down one allowed coefficient level every two rounds.
    This keeps every realized value inside the objective's fixed search domain.
    """
    weight = float(base_weight)
    if weight not in CORRECTION_LOSS_WEIGHTS:
        raise ValueError("v141 correction coefficient is outside the search domain")
    if schedule not in CORRECTION_LOSS_WEIGHT_SCHEDULES:
        raise ValueError("v141 correction schedule is invalid")
    if round_index < 1:
        raise ValueError("v141 round index must be positive")
    if schedule == "constant":
        return weight
    level = CORRECTION_LOSS_WEIGHTS.index(weight)
    decay_steps = (round_index - 1) // 2
    return CORRECTION_LOSS_WEIGHTS[max(0, level - decay_steps)]


# The first generation is a compact orthogonal screen: every required eta,
# correction gate, and dual-reward scale is exercised while heavy GPU jobs stay
# sequential.  Lambda is varied enough to expose under/over-distillation.
GENERATION_1_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate": "g1_eta0_posadv_r0_l02",
        "intervention_ppo_eta": 0.0,
        "correction_weight_mode": "positive_advantage",
        "dual_reward_scale": 0.0,
        "correction_loss_weight": 0.2,
    },
    {
        "candidate": "g1_eta025_posadv_r025_l02",
        "intervention_ppo_eta": 0.25,
        "correction_weight_mode": "positive_advantage",
        "dual_reward_scale": 0.25,
        "correction_loss_weight": 0.2,
    },
    {
        "candidate": "g1_eta05_posadv_r1_l02",
        "intervention_ppo_eta": 0.5,
        "correction_weight_mode": "positive_advantage",
        "dual_reward_scale": 1.0,
        "correction_loss_weight": 0.2,
    },
    {
        "candidate": "g1_eta1_posadv_r025_l02",
        "intervention_ppo_eta": 1.0,
        "correction_weight_mode": "positive_advantage",
        "dual_reward_scale": 0.25,
        "correction_loss_weight": 0.2,
    },
    {
        "candidate": "g1_eta025_int_r0_l01",
        "intervention_ppo_eta": 0.25,
        "correction_weight_mode": "intervention_only",
        "dual_reward_scale": 0.0,
        "correction_loss_weight": 0.1,
    },
    {
        "candidate": "g1_eta025_success_r025_l04",
        "intervention_ppo_eta": 0.25,
        "correction_weight_mode": "episode_success_positive_advantage",
        "dual_reward_scale": 0.25,
        "correction_loss_weight": 0.4,
    },
    {
        "candidate": "g1_eta0_success_r1_l04",
        "intervention_ppo_eta": 0.0,
        "correction_weight_mode": "episode_success_positive_advantage",
        "dual_reward_scale": 1.0,
        "correction_loss_weight": 0.4,
    },
    {
        "candidate": "g1_eta05_int_r025_l005",
        "intervention_ppo_eta": 0.5,
        "correction_weight_mode": "intervention_only",
        "dual_reward_scale": 0.25,
        "correction_loss_weight": 0.05,
    },
)


def candidate_score(*, target_off: float, f1_off: float, shield_gap: float) -> float:
    return 0.8 * target_off + 0.2 * f1_off - 0.25 * shield_gap
