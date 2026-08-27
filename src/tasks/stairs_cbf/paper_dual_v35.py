"""Paper-aligned CBF-RL reward variants for the v35 outcome study."""

from __future__ import annotations

from typing import Any

PAPER_ARXIV_ID = "2510.14959v6"
PAPER_DEMO_COMMIT = "68955c8ba9e929d974b6677635370ee93eecc63a"

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
