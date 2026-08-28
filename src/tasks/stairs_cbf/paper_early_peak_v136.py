"""Conservative early selection for an exact paper-PPO performance tie."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Sequence

METHOD_ID = "paper-cbf-dual-earliest-exact-peak-v136"
V132_ROUND_METRICS_SHA256 = (
  "7c0aa01acfaed0cc32f22a245384b20220234ada511c8a83b9d7300e7414ec9c"
)
V132_EARLIEST_PEAK_CHECKPOINT_SHA256 = (
  "f0c18b0965668fb8eab5e3fab6e8f2edc6555c35f9cb65df4b419c4c3df34b91"
)


def earliest_exact_peak_decision(
  candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
  """Select the earliest rollout attaining the exact maximum success rate."""
  if not candidates:
    raise ValueError("v136 requires at least one aligned rollout candidate")
  normalized: list[dict[str, Any]] = []
  observed_rounds: set[int] = set()
  for candidate in candidates:
    rollout_round = int(candidate["rollout_round"])
    success_count = int(candidate["success_count"])
    episode_count = int(candidate["episode_count"])
    if (
      rollout_round < 1
      or rollout_round in observed_rounds
      or episode_count < 1
      or not 0 <= success_count <= episode_count
    ):
      raise ValueError("v136 candidate counts or rollout rounds are invalid")
    observed_rounds.add(rollout_round)
    normalized.append(
      {
        **candidate,
        "rollout_round": rollout_round,
        "success_count": success_count,
        "episode_count": episode_count,
        "exact_success_rate": Fraction(success_count, episode_count),
      }
    )

  exact_peak = max(candidate["exact_success_rate"] for candidate in normalized)
  tied = [
    candidate
    for candidate in normalized
    if candidate["exact_success_rate"] == exact_peak
  ]
  selected = min(tied, key=lambda candidate: candidate["rollout_round"])
  return {
    **{key: value for key, value in selected.items() if key != "exact_success_rate"},
    "method_id": METHOD_ID,
    "success_rate": float(exact_peak),
    "exact_rate_numerator": exact_peak.numerator,
    "exact_rate_denominator": exact_peak.denominator,
    "exact_peak_tie_count": len(tied),
    "exact_peak_tied_rollout_rounds": [
      candidate["rollout_round"] for candidate in tied
    ],
    "selection_tie_break": "earliest_rollout_minimum_update_count",
    "additional_evaluation_count": 0,
  }
