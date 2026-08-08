"""Prospective constants and pure gates for deployment-unit specialist v21."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.tasks.stairs_cbf.deployment_context import (
  V21_CONTEXT_SPECS,
  V21_DEVELOPMENT_CONTEXTS,
  V21_FORMAL_CONTEXTS,
)

PROTOCOL_ID = "safe100-deployment-unit-local-matched-success-v21"
POLICY_METHOD = "Local Matched-Success Preservation v21"
CONTROL_METHOD = "v20-style matched-success PPO advantage control"
SPECIALIST_MODES = ("lateral", "contact_stability")
FORMAL_CONTEXTS_BY_MODE = {
  "lateral": tuple(
    context_id for context_id in V21_FORMAL_CONTEXTS if context_id.startswith("L")
  ),
  "contact_stability": tuple(
    context_id for context_id in V21_FORMAL_CONTEXTS if context_id.startswith("C")
  ),
}
BETA_GRID = (0.0, 1.0, 4.0, 16.0)
FORMAL_ROUNDS = 8
FORMAL_TARGET_EPISODES = 1024
FORMAL_D0_EPISODES = 256
FORMAL_MONITOR_EPISODES = 256
DEVELOPMENT_SELECTION_EPISODES = 512
FORMAL_EVAL_BATCH_SIZE = 128
FORMAL_BOOTSTRAP_SAMPLES = 10_000
CANDIDATE_FRACTIONS = (0.5, 1.0, 1.5)
CANDIDATE_SCREEN_EPISODES = 64
CONFIRMATION_BLOCKS = 3
CONFIRMATION_EPISODES_PER_BLOCK = 64
CANDIDATE_D0_EPISODES = 128
CANDIDATE_EVALUATION_EPISODES_PER_ROUND = (
  len(CANDIDATE_FRACTIONS) * CANDIDATE_SCREEN_EPISODES
  + CONFIRMATION_BLOCKS * CONFIRMATION_EPISODES_PER_BLOCK
)
CALIBRATION_EPISODES = 512
CALIBRATION_BATCH_SIZE = 128
TELEMETRY_ENVIRONMENT_ID_PER_BATCH = 0
CONTEXTS = (*V21_DEVELOPMENT_CONTEXTS, *V21_FORMAL_CONTEXTS)
CONTEXT_ADAPTATION_SEEDS = {
  "L_dev": 21_001,
  "C_dev": 21_002,
  "L1": 21_101,
  "L2": 21_102,
  "L3": 21_103,
  "L4": 21_104,
  "L5": 21_105,
  "C1": 21_201,
  "C2": 21_202,
  "C3": 21_203,
  "C4": 21_204,
  "C5": 21_205,
}
CONTEXT_CALIBRATION_CANDIDATE_SEEDS = {
  context_id: tuple(
    range(
      int(V21_CONTEXT_SPECS[context_id]["candidate_seed_prefix"]) * 100 + 8,
      int(V21_CONTEXT_SPECS[context_id]["candidate_seed_prefix"]) * 100 + 20,
    )
  )
  for context_id in CONTEXTS
}
CONTEXT_CALIBRATION_EVALUATION_SEEDS = {
  context_id: 7_100_000 + 10_000 * index
  for index, context_id in enumerate(CONTEXTS)
}
CONTEXT_FORMAL_AUDIT_SEEDS = {
  context_id: 8_100_000 + 10_000 * index
  for index, context_id in enumerate(V21_FORMAL_CONTEXTS)
}
CONTEXT_DEVELOPMENT_SELECTION_SEEDS = {
  context_id: 7_900_000 + 10_000 * index
  for index, context_id in enumerate(V21_DEVELOPMENT_CONTEXTS)
}
CONTEXT_MONITOR_SEEDS = {
  context_id: 9_100_000 + 10_000 * index
  for index, context_id in enumerate(CONTEXTS)
}
FORMAL_BOOTSTRAP_SEED = 10_100_000


@dataclass(frozen=True)
class FixedBudgetStatus:
  actual_rounds: int
  retained_update_count: int
  protocol_valid: bool
  stop_reason: str
  retained_update_count_is_gate: bool = False


def fixed_budget_status(
  *, actual_rounds: int, retained_update_count: int, formal: bool = True
) -> FixedBudgetStatus:
  if actual_rounds < 0 or retained_update_count < 0:
    raise ValueError("v21 round and retained-update counts cannot be negative")
  if retained_update_count > actual_rounds:
    raise ValueError("v21 retained updates cannot exceed completed rounds")
  if formal and actual_rounds != FORMAL_ROUNDS:
    raise ValueError(
      f"formal v21 requires exactly {FORMAL_ROUNDS} rounds, got {actual_rounds}"
    )
  return FixedBudgetStatus(
    actual_rounds=actual_rounds,
    retained_update_count=retained_update_count,
    protocol_valid=(actual_rounds == FORMAL_ROUNDS if formal else True),
    stop_reason=(
      "fixed_round_budget_completed"
      if actual_rounds == FORMAL_ROUNDS
      else "development_or_smoke_budget_completed"
    ),
  )


def confirmation_block_gate(
  success_deltas: Sequence[float],
  fall_deltas: Sequence[float],
  *,
  mean_kl: float,
  finite: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
  """Apply the prospectively frozen three-block target candidate gate."""
  if len(success_deltas) != CONFIRMATION_BLOCKS or len(fall_deltas) != (
    CONFIRMATION_BLOCKS
  ):
    raise ValueError("v21 confirmation requires exactly three blocks")
  values = [*success_deltas, *fall_deltas, mean_kl]
  if not all(math.isfinite(float(value)) for value in values):
    raise ValueError("v21 confirmation metrics must be finite")
  mean_success_delta = sum(success_deltas) / CONFIRMATION_BLOCKS
  mean_fall_delta = sum(fall_deltas) / CONFIRMATION_BLOCKS
  positive_blocks = sum(delta > 0.0 for delta in success_deltas)
  reasons: list[str] = []
  if not finite:
    reasons.append("candidate parameters are non-finite")
  if not mean_success_delta > 0.0:
    reasons.append("mean confirmation success delta is not strictly positive")
  if positive_blocks < 2:
    reasons.append("fewer than two confirmation blocks have positive success delta")
  if mean_fall_delta > 0.03:
    reasons.append("mean confirmation fall increase exceeds 3 percentage points")
  if not mean_kl < 0.01:
    reasons.append("candidate KL is not below 0.01")
  metrics = {
    "success_deltas": list(success_deltas),
    "fall_deltas": list(fall_deltas),
    "mean_success_delta": mean_success_delta,
    "mean_fall_delta": mean_fall_delta,
    "positive_success_blocks": positive_blocks,
    "required_positive_success_blocks": 2,
    "mean_kl": mean_kl,
  }
  return not reasons, reasons, metrics


def repair_regression_rates(
  baseline_success: Sequence[bool], candidate_success: Sequence[bool]
) -> dict[str, float | int]:
  if len(baseline_success) != len(candidate_success) or not baseline_success:
    raise ValueError("paired repair/regression outcomes must be non-empty and equal")
  base_failures = sum(not value for value in baseline_success)
  base_successes = sum(bool(value) for value in baseline_success)
  repairs = sum(
    (not old) and bool(new)
    for old, new in zip(baseline_success, candidate_success, strict=True)
  )
  regressions = sum(
    bool(old) and (not new)
    for old, new in zip(baseline_success, candidate_success, strict=True)
  )
  repair_rate = repairs / max(1, base_failures)
  regression_rate = regressions / max(1, base_successes)
  return {
    "base_failure_count": base_failures,
    "base_success_count": base_successes,
    "repair_count": repairs,
    "regression_count": regressions,
    "repair_rate": repair_rate,
    "regression_rate": regression_rate,
    "selection_score": repair_rate - regression_rate,
  }


def deployment_mode_gate(
  target_success_deltas: Sequence[float],
  target_fall_deltas: Sequence[float],
  d0_success_deltas: Sequence[float],
) -> dict[str, Any]:
  """Apply the formal gate with deployment context as the statistical unit."""
  expected = 5
  if not (
    len(target_success_deltas)
    == len(target_fall_deltas)
    == len(d0_success_deltas)
    == expected
  ):
    raise ValueError("v21 formal mode gate requires exactly five contexts")
  values = [*target_success_deltas, *target_fall_deltas, *d0_success_deltas]
  if not all(math.isfinite(float(value)) for value in values):
    raise ValueError("v21 formal mode gate metrics must be finite")
  mean_success_delta = sum(target_success_deltas) / expected
  positive_contexts = sum(delta > 0.0 for delta in target_success_deltas)
  mean_fall_delta = sum(target_fall_deltas) / expected
  mean_d0_success_delta = sum(d0_success_deltas) / expected
  conditions = {
    "mean_target_success_delta_strictly_positive": mean_success_delta > 0.0,
    "at_least_four_of_five_contexts_positive": positive_contexts >= 4,
    "mean_target_fall_delta_at_most_three_pp": mean_fall_delta <= 0.03,
    "mean_d0_success_delta_at_least_minus_five_pp": (
      mean_d0_success_delta >= -0.05
    ),
  }
  return {
    "passed": all(conditions.values()),
    "conditions": conditions,
    "mean_target_success_delta": mean_success_delta,
    "positive_context_count": positive_contexts,
    "context_count": expected,
    "mean_target_fall_delta": mean_fall_delta,
    "mean_d0_success_delta": mean_d0_success_delta,
  }


def select_development_beta(
  per_beta_context_metrics: Mapping[float, Mapping[str, Mapping[str, float]]]
) -> dict[str, Any]:
  """Select one beta from both excluded contexts using only RR minus RG."""
  if set(per_beta_context_metrics) != set(BETA_GRID):
    raise ValueError("development results must cover the frozen beta grid")
  rows = []
  for beta in BETA_GRID:
    contexts = per_beta_context_metrics[beta]
    if set(contexts) != set(V21_DEVELOPMENT_CONTEXTS):
      raise ValueError("each beta must cover both excluded development contexts")
    scores = []
    regression_rates = []
    for context_id in V21_DEVELOPMENT_CONTEXTS:
      metrics = contexts[context_id]
      rr = float(metrics["repair_rate"])
      rg = float(metrics["regression_rate"])
      if not all(math.isfinite(value) for value in (rr, rg)):
        raise ValueError("development RR/RG must be finite")
      scores.append(rr - rg)
      regression_rates.append(rg)
    rows.append(
      {
        "beta": beta,
        "context_scores": dict(zip(V21_DEVELOPMENT_CONTEXTS, scores, strict=True)),
        "mean_selection_score": sum(scores) / len(scores),
        "worst_context_selection_score": min(scores),
        "mean_regression_rate": sum(regression_rates) / len(regression_rates),
      }
    )
  selected = max(
    rows,
    key=lambda row: (
      row["mean_selection_score"],
      row["worst_context_selection_score"],
      -row["mean_regression_rate"],
      -row["beta"],
    ),
  )
  return {
    "selection_metric": "mean across L_dev/C_dev of repair_rate - regression_rate",
    "tie_breaks": [
      "higher worst-context score",
      "lower mean regression rate",
      "lower beta",
    ],
    "rows": rows,
    "selected_beta": selected["beta"],
  }


def _declares_seed_values(key: str) -> bool:
  normalized = key.lower()
  return normalized in ("seed", "seeds") or normalized.endswith(
    ("_seed", "_seeds", "_seed_base", "_seed_bases", "_seed_start", "_seed_order")
  )


def _direct_seed_values(value: Any) -> Iterable[tuple[str, int]]:
  if isinstance(value, int) and not isinstance(value, bool):
    yield "", value
  elif isinstance(value, list):
    for index, child in enumerate(value):
      if isinstance(child, int) and not isinstance(child, bool):
        yield f"[{index}]", child
  elif isinstance(value, dict):
    for key, child in value.items():
      if isinstance(child, int) and not isinstance(child, bool):
        yield f".{key}", child


def _iter_declared_seeds(value: Any, path: str = "") -> Iterable[tuple[str, int]]:
  if isinstance(value, dict):
    for key, child in value.items():
      child_path = f"{path}.{key}" if path else key
      if _declares_seed_values(key):
        for suffix, seed in _direct_seed_values(child):
          yield f"{child_path}{suffix}", seed
      yield from _iter_declared_seeds(child, child_path)
  elif isinstance(value, list):
    for index, child in enumerate(value):
      yield from _iter_declared_seeds(child, f"{path}[{index}]")


def fresh_randomness_report(repo: Path) -> dict[str, Any]:
  historical: list[dict[str, Any]] = []
  for version in range(17, 21):
    root = repo / "results/online" / f"specialist_v{version}"
    if not root.is_dir():
      continue
    for path in sorted(root.rglob("*.json")):
      try:
        payload = json.loads(path.read_text())
      except (json.JSONDecodeError, OSError):
        continue
      for json_path, seed in _iter_declared_seeds(payload):
        historical.append(
          {
            "seed": seed,
            "source": str(path.relative_to(repo)),
            "json_path": json_path,
          }
        )
  proposed = {
    *CONTEXT_ADAPTATION_SEEDS.values(),
    *CONTEXT_CALIBRATION_EVALUATION_SEEDS.values(),
    *CONTEXT_FORMAL_AUDIT_SEEDS.values(),
    *CONTEXT_DEVELOPMENT_SELECTION_SEEDS.values(),
    *CONTEXT_MONITOR_SEEDS.values(),
    FORMAL_BOOTSTRAP_SEED,
  }
  for seeds in CONTEXT_CALIBRATION_CANDIDATE_SEEDS.values():
    proposed.update(seeds)
  historical = [
    {"seed": seed, "source": source, "json_path": json_path}
    for seed, source, json_path in sorted(
      {
        (record["seed"], record["source"], record["json_path"])
        for record in historical
      }
    )
  ]
  historical_values = {record["seed"] for record in historical}
  collisions = sorted(proposed & historical_values)
  return {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "historical_versions_scanned": [17, 18, 19, 20],
    "historical_declared_seed_records": historical,
    "proposed_seed_count": len(proposed),
    "collisions": collisions,
    "passed": not collisions,
  }


def canonical_sha256(value: Mapping[str, Any]) -> str:
  return hashlib.sha256(
    json.dumps(
      value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
  ).hexdigest()
