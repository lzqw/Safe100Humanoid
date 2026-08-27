"""Paper-aligned CBF-RL reward variants for the v35 outcome study."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

import torch

PAPER_ARXIV_ID = "2510.14959v6"
PAPER_DEMO_COMMIT = "68955c8ba9e929d974b6677635370ee93eecc63a"
PAPER_DOMAIN_RANDOMIZATION_MODES = ("off", "paper_static", "paper_full")


def normalize_filter_group_advantages(
  advantages: torch.Tensor,
  filter_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
  """Normalize filtered and nominal rollout advantages independently.

  A mixed rollout intentionally contains two transition kernels: half of the
  worlds execute the CBF projection and half execute the nominal policy.  A
  single population normalization lets either reward/return scale dominate
  the PPO step.  Keeping each fixed environment group at zero mean and unit
  variance gives the two training distributions equal actor-gradient scale
  without changing returns used by the critic.
  """
  if advantages.ndim != 2:
    raise ValueError("mixed-filter advantages must have shape [T, N]")
  if filter_mask.shape != advantages.shape[1:] or filter_mask.dtype != torch.bool:
    raise ValueError("mixed-filter mask must be boolean with shape [N]")
  if not bool(filter_mask.any()) or bool(filter_mask.all()):
    raise ValueError("mixed-filter advantage groups must both be non-empty")
  if not bool(torch.isfinite(advantages).all()):
    raise RuntimeError("mixed-filter advantages contain non-finite values")

  output = torch.empty_like(advantages)
  metrics: dict[str, float] = {}
  for name, environment_mask in (
    ("filter_on", filter_mask),
    ("filter_off", ~filter_mask),
  ):
    values = advantages[:, environment_mask]
    mean = values.mean()
    std = values.std(unbiased=False)
    normalized = (values - mean) / (std + 1.0e-8)
    output[:, environment_mask] = normalized
    metrics.update(
      {
        f"{name}_advantage_count": float(values.numel()),
        f"{name}_advantage_mean_before": float(mean),
        f"{name}_advantage_std_before": float(std),
        f"{name}_advantage_mean_after": float(normalized.mean()),
        f"{name}_advantage_std_after": float(
          normalized.std(unbiased=False)
        ),
      }
    )
  metrics["filter_group_balanced_advantages"] = 1.0
  metrics["balanced_advantage_mean"] = float(output.mean())
  metrics["balanced_advantage_std"] = float(output.std(unbiased=False))
  return output, metrics

# ``raw_demo`` reproduces the two weights and action-coordinate distance used
# by the authors' public navigation demo. The intermediate variants are needed
# because humanoid reward rates and action geometry differ from a 2-D point
# robot; they change only the strength of the same two CBF-RL terms.
PAPER_DUAL_CANDIDATES: dict[str, dict[str, float | str]] = {
  "current": {
    "correction_space": "target",
    "sigma": 0.5,
    "margin_weight": 1.0,
    "intervention_weight": 1.0,
  },
  "raw_moderate": {
    "correction_space": "raw_action",
    "sigma": 0.5,
    "margin_weight": 1.0,
    "intervention_weight": 10.0,
  },
  "raw_strong": {
    "correction_space": "raw_action",
    "sigma": 0.5,
    "margin_weight": 2.0,
    "intervention_weight": 50.0,
  },
  "raw_demo": {
    "correction_space": "raw_action",
    "sigma": 0.5,
    "margin_weight": 10.0,
    "intervention_weight": 100.0,
  },
  "paper_stair_exact": {
    # Humanoid Eq. (27) uses the displacement of the reduced-order swing-foot
    # state.  Five centimetres keeps the exponential informative at the scale
    # of one 50 Hz filtered foot update.
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 1.0,
    "intervention_weight": 1.0,
  },
  "paper_stair_demo_scale": {
    # Keep the humanoid reduced-order distance from Eq. (27), but use the
    # 10x margin / 100x action-proximity scaling in the authors' public code.
    # This prevents the sparse CBF signal from disappearing underneath the
    # nominal locomotion return after whole-rollout advantage normalization.
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 10.0,
    "intervention_weight": 100.0,
  },
  "paper_stair_sloped_demo_scale": {
    # Keep the paper's reduced-order swing-foot correction and public-demo
    # reward scale, but pair it with the task-compatible sloped clearance CBF.
    # The horizontal Eq. (27) geometry only reacts at the riser plane; on the
    # 18.4 cm target stairs it did not lift the toe early enough under fresh
    # dynamics randomization.  This variant preserves the CBF-RL training
    # principle while making the barrier anticipate the required vertical
    # clearance.
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 10.0,
    "intervention_weight": 100.0,
  },
}


def configure_paper_dual_reward(
  env_cfg,
  candidate: str,
  *,
  runtime_filter_during_training: bool = True,
) -> dict[str, Any]:
  """Install one v35 reward variant without changing the safety filter."""
  if candidate not in PAPER_DUAL_CANDIDATES:
    raise ValueError(f"unknown v35 reward candidate {candidate!r}")
  parameters = dict(PAPER_DUAL_CANDIDATES[candidate])
  reward = env_cfg.rewards.get("cbf_dual")
  if reward is None:
    raise RuntimeError("v35 requires the CBF dual reward term")
  reward.weight = 1.0
  reward.params = {"action_name": "joint_pos", **parameters}
  return {
    "candidate": candidate,
    "paper_arxiv_id": PAPER_ARXIV_ID,
    "paper_demo_commit": PAPER_DEMO_COMMIT,
    "reward_parameters": parameters,
    "runtime_filter_during_training": bool(runtime_filter_during_training),
    "historical_default_preserved": candidate == "current",
  }


def configure_paper_training_domain_randomization(
  env_cfg,
  mode: str,
  *,
  strength: float = 1.0,
) -> dict[str, Any]:
  """Restore repository-native G1 randomization for CBF-RL training.

  The online deployment task intentionally removes physical randomization and
  its play variant also disables actor observation corruption. That is the
  correct evaluation protocol, but it differs from the paper's training
  setup. Reusing the native G1 distributions keeps their ranges owned by one
  authoritative configuration rather than copying them by hand.
  """
  if mode not in PAPER_DOMAIN_RANDOMIZATION_MODES:
    raise ValueError(
      "paper domain-randomization mode must be one of "
      f"{PAPER_DOMAIN_RANDOMIZATION_MODES}, got {mode!r}"
    )
  if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
    raise ValueError("paper domain-randomization strength must lie in (0, 1]")
  if mode == "off":
    return {
      "mode": mode,
      "enabled": False,
      "actor_observation_corruption": False,
      "event_terms": [],
      "strength": 0.0,
    }

  # Import lazily to keep the reward-only module lightweight and avoid adding
  # a configuration dependency to pure CBF math users.
  from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

  native_training_cfg = unitree_g1_rough_env_cfg(play=False)
  event_names = ["encoder_bias", "foot_friction", "base_com"]
  if mode == "paper_full":
    event_names.append("push_robot")
  missing = [name for name in event_names if name not in native_training_cfg.events]
  if missing:
    raise RuntimeError(f"native G1 training config lacks DR events: {missing}")
  for name in event_names:
    env_cfg.events[name] = deepcopy(native_training_cfg.events[name])
  env_cfg.observations["actor"].enable_corruption = True

  # Continuation policies already encode a narrow nominal gait. Scale every
  # native perturbation continuously around its identity/no-noise value so DR
  # can be introduced as a curriculum instead of an abrupt distribution jump.
  for term in env_cfg.observations["actor"].terms.values():
    noise = term.noise
    if noise is not None:
      term.noise = replace(
        noise,
        n_min=float(noise.n_min) * strength,
        n_max=float(noise.n_max) * strength,
      )
  encoder_event = env_cfg.events["encoder_bias"]
  encoder_event.params["bias_range"] = tuple(
    float(value) * strength for value in encoder_event.params["bias_range"]
  )
  friction_event = env_cfg.events["foot_friction"]
  friction_event.params["ranges"] = tuple(
    1.0 + (float(value) - 1.0) * strength
    for value in friction_event.params["ranges"]
  )
  base_com_event = env_cfg.events["base_com"]
  base_com_event.params["ranges"] = {
    axis: tuple(float(value) * strength for value in bounds)
    for axis, bounds in base_com_event.params["ranges"].items()
  }
  if mode == "paper_full":
    push_event = env_cfg.events["push_robot"]
    push_event.params["velocity_range"] = {
      axis: tuple(float(value) * strength for value in bounds)
      for axis, bounds in push_event.params["velocity_range"].items()
    }

  friction = env_cfg.events["foot_friction"].params
  base_com = env_cfg.events["base_com"].params
  encoder = env_cfg.events["encoder_bias"].params
  metadata: dict[str, Any] = {
    "mode": mode,
    "enabled": True,
    "source": "unitree_g1_rough_env_cfg(play=False)",
    "actor_observation_corruption": True,
    "strength": strength,
    "event_terms": event_names,
    "encoder_bias_range": list(encoder["bias_range"]),
    "foot_friction_range": list(friction["ranges"]),
    "foot_friction_operation": friction["operation"],
    "foot_friction_shared_random": bool(friction["shared_random"]),
    "base_com_operation": base_com["operation"],
    "base_com_ranges_m": {
      str(axis): list(bounds) for axis, bounds in base_com["ranges"].items()
    },
    "external_pushes": mode == "paper_full",
  }
  if mode == "paper_full":
    push = env_cfg.events["push_robot"]
    metadata["push_interval_range_s"] = list(push.interval_range_s)
    metadata["push_velocity_range"] = {
      axis: list(bounds)
      for axis, bounds in push.params["velocity_range"].items()
    }
  return metadata
