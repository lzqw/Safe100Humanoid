"""Prospective constants and pure audits for CBF-Proximal PPO v23."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_ID = "safe100-cbf-proximal-online-refinement-v23"
POLICY_METHOD = "CBF-Shielded KL-Regularized Online PPO v23"
CONTEXT_ID = "L_effect"
CONTEXT_REPO_PATH = (
  "results/online/specialist_v22/calibration/L_effect/context.json"
)
CONTEXT_FILE_SHA256 = (
  "650a97519168382bf4f7fc45580fa179cb3c51a1f18195f4850c5667d6f0d6a7"
)
CONTEXT_PARAMETERS_SHA256 = (
  "f3d4470ec01c7f55982d93b4be53dcafb13c0d2d82f4c03f34835f84c99cf4ae"
)
BASE_CHECKPOINT_SHA256 = (
  "cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8"
)

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

ADAPTATION_SEED = 123_230_001
FINAL_TARGET_SEED = 125_000_000
FINAL_D0_SEED = 126_000_000
REPORT_BOOTSTRAP_SEEDS = {
  "target": 127_000_000,
  "D0": 127_000_010,
}
EVAL_BATCH_SIZE = 128
FINAL_TARGET_EPISODES = 512
FINAL_D0_EPISODES = 256
REPORT_BOOTSTRAP_SAMPLES = 2_000

MINIMUM_TARGET_SUCCESS_DELTA = 0.03
MAXIMUM_TARGET_FALL_DELTA = 0.01
MINIMUM_D0_SUCCESS_DELTA = -0.05


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
  }


def all_v23_fresh_seed_values() -> list[int]:
  values = [ADAPTATION_SEED]
  values.extend(
    FINAL_TARGET_SEED + repeat
    for repeat in range(FINAL_TARGET_EPISODES // EVAL_BATCH_SIZE)
  )
  values.extend(
    FINAL_D0_SEED + repeat
    for repeat in range(FINAL_D0_EPISODES // EVAL_BATCH_SIZE)
  )
  values.extend(REPORT_BOOTSTRAP_SEEDS.values())
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
  """Prove every fresh v23 seed is disjoint from published prior evidence."""
  historical: set[int] = set()
  scanned: list[dict[str, str]] = []
  results_root = repo / "results/online"
  if results_root.is_dir():
    for path in sorted(results_root.rglob("*.json")):
      if "proximal_v23" in path.parts:
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
  proposed_values = all_v23_fresh_seed_values()
  counts = Counter(proposed_values)
  internal = sorted(seed for seed, count in counts.items() if count > 1)
  historical_collisions = sorted(set(proposed_values) & historical)
  collisions = sorted(set(internal) | set(historical_collisions))
  return {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "historical_scope": "all parseable JSON below results/online except proximal_v23",
    "historical_json_file_count": len(scanned),
    "historical_json_files_sha256": canonical_sha256({"files": scanned}),
    "historical_unique_declared_seed_count": len(historical),
    "proposed_seed_occurrence_count": len(proposed_values),
    "proposed_unique_seed_count": len(set(proposed_values)),
    "proposed_internal_collisions": internal,
    "historical_collisions": historical_collisions,
    "collisions": collisions,
    "passed": not collisions,
  }


def development_gate(
  *,
  target_success_delta: float,
  target_fall_delta: float,
  d0_success_delta: float,
) -> dict[str, Any]:
  values = (target_success_delta, target_fall_delta, d0_success_delta)
  if not all(math.isfinite(value) for value in values):
    raise ValueError("v23 development metrics must be finite")
  conditions = {
    "target_success_delta_at_least_three_pp": (
      target_success_delta >= MINIMUM_TARGET_SUCCESS_DELTA
    ),
    "target_fall_delta_at_most_one_pp": (
      target_fall_delta <= MAXIMUM_TARGET_FALL_DELTA
    ),
    "d0_success_delta_at_least_minus_five_pp": (
      d0_success_delta >= MINIMUM_D0_SUCCESS_DELTA
    ),
  }
  return {
    "passed": all(conditions.values()),
    "conditions": conditions,
    "target_success_delta": target_success_delta,
    "target_fall_delta": target_fall_delta,
    "d0_success_delta": d0_success_delta,
    "confidence_intervals_are_gates": False,
  }


def repair_regression_counts(
  baseline_success: Sequence[bool], final_success: Sequence[bool]
) -> dict[str, Any]:
  if not baseline_success or len(baseline_success) != len(final_success):
    raise ValueError("paired repair/regression vectors must be non-empty and equal")
  repairs = sum(
    (not bool(old)) and bool(new)
    for old, new in zip(baseline_success, final_success, strict=True)
  )
  regressions = sum(
    bool(old) and (not bool(new))
    for old, new in zip(baseline_success, final_success, strict=True)
  )
  base_failures = sum(not bool(value) for value in baseline_success)
  base_successes = sum(bool(value) for value in baseline_success)
  return {
    "paired_conditions": len(baseline_success),
    "base_failure_count": base_failures,
    "base_success_count": base_successes,
    "repair_count": repairs,
    "regression_count": regressions,
    "repair_rate_given_base_failure": repairs / max(1, base_failures),
    "regression_rate_given_base_success": regressions / max(1, base_successes),
    "net_success_change": repairs - regressions,
  }
