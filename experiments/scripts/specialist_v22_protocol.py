"""Prospective constants and pure gates for v22 effect-first development."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from specialist_v21_protocol import configure_v21_policy_evaluation_algorithm
from src.tasks.stairs_cbf.deployment_context import (
  V22_CONTEXT_SCHEMA_VERSION,
  V22_CONTEXT_SPECS,
)

PROTOCOL_ID = "safe100-effect-first-development-v22"
POLICY_METHOD = "Effect-First Best-So-Far PPO v22"
CONTEXTS = ("L_effect", "C_effect")
MODES = {
  "L_effect": "lateral",
  "C_effect": "contact_stability",
}

ROUNDS = 8
NUM_ENVS = 64
ROLLOUT_STEPS = 1024
DUAL_ROLLOUT_BATCHES = 2
NORMAL_FAILURE_SUCCESS_SLOTS = (40, 12, 12)
FAILURE_START_FRACTION = NORMAL_FAILURE_SUCCESS_SLOTS[1] / NUM_ENVS
SUCCESS_START_FRACTION = NORMAL_FAILURE_SUCCESS_SLOTS[2] / NUM_ENVS
FAILURE_DISCOVERY_MAX_ROLLOUTS = 12
CANDIDATE_FRACTIONS = (0.5, 1.0, 1.5)
CANDIDATE_SCREEN_EPISODES = 64
CANDIDATE_CONFIRM_EPISODES = 128
CANDIDATE_D0_EPISODES = 128
CALIBRATION_EPISODES = 512
EVAL_BATCH_SIZE = 128
VALIDATION_EPISODES = 256
FINAL_TARGET_EPISODES = 512
FINAL_D0_EPISODES = 256
REPORT_BOOTSTRAP_SAMPLES = 2_000

CALIBRATION_SUCCESS_BOUNDS = (0.65, 0.75)
CALIBRATION_MINIMUM_FALLS = 100
CALIBRATION_MINIMUM_PURITY = 0.85
CONFIRMATION_MAXIMUM_FALL_DELTA = 0.03
D0_MINIMUM_SUCCESS_DELTA = -0.05
VALIDATION_MAXIMUM_FALL_DELTA = 0.02
DEVELOPMENT_MINIMUM_SUCCESS_DELTA = 0.03
DEVELOPMENT_MAXIMUM_FALL_DELTA = 0.01

CONTEXT_CALIBRATION_CANDIDATE_SEEDS = {
  context_id: tuple(
    range(
      int(spec["candidate_seed_prefix"]) * 100
      + int(spec["candidate_index_bounds"][0]),
      int(spec["candidate_seed_prefix"]) * 100
      + int(spec["candidate_index_bounds"][1])
      + 1,
    )
  )
  for context_id, spec in V22_CONTEXT_SPECS.items()
}
CONTEXT_CALIBRATION_EVALUATION_SEEDS = {
  "L_effect": 92_000_000,
  "C_effect": 92_100_000,
}
CONTEXT_ADAPTATION_SEEDS = {
  "L_effect": 22_210_001,
  "C_effect": 22_220_001,
}
CONTEXT_VALIDATION_SEEDS = {
  "L_effect": 94_000_000,
  "C_effect": 94_100_000,
}
CONTEXT_FINAL_TARGET_SEEDS = {
  "L_effect": 95_000_000,
  "C_effect": 95_100_000,
}
CONTEXT_FINAL_D0_SEEDS = {
  "L_effect": 96_000_000,
  "C_effect": 96_100_000,
}
CONTEXT_REPORT_BOOTSTRAP_SEEDS = {
  "L_effect": {"target": 97_000_000, "D0": 97_000_010},
  "C_effect": {"target": 97_100_000, "D0": 97_100_010},
}


def calibration_evaluation_seed(context_id: str, candidate_index: int) -> int:
  return CONTEXT_CALIBRATION_EVALUATION_SEEDS[context_id] + 100 * candidate_index


def failure_discovery_seed(context_id: str, discovery_index: int) -> int:
  return CONTEXT_ADAPTATION_SEEDS[context_id] + 500_000 + discovery_index


def dual_rollout_seed(
  context_id: str, round_index: int, batch_index: int
) -> int:
  return (
    CONTEXT_ADAPTATION_SEEDS[context_id]
    + 1_000_000
    + 10 * round_index
    + batch_index
  )


def candidate_screen_seed(context_id: str, round_index: int) -> int:
  return CONTEXT_ADAPTATION_SEEDS[context_id] + 20_000 * round_index


def candidate_confirmation_seed(context_id: str, round_index: int) -> int:
  return candidate_screen_seed(context_id, round_index) + 10_000


def candidate_d0_seed(context_id: str) -> int:
  return CONTEXT_ADAPTATION_SEEDS[context_id] + 300_000


def configure_v22_policy_evaluation_algorithm(cfg: Any) -> None:
  """Use the frozen v20/v21 learning architecture with beta exactly zero."""
  configure_v21_policy_evaluation_algorithm(
    cfg, matched_success_preservation_beta=0.0
  )


def candidate_confirmation_gate(
  *,
  success_delta: float,
  fall_delta: float,
  finite: bool,
) -> tuple[bool, list[str]]:
  values = (success_delta, fall_delta)
  if not all(math.isfinite(float(value)) for value in values):
    raise ValueError("v22 confirmation metrics must be finite")
  reasons: list[str] = []
  if not finite:
    reasons.append("candidate parameters are non-finite")
  if not success_delta > 0.0:
    reasons.append("confirmation success delta is not strictly positive")
  if fall_delta > CONFIRMATION_MAXIMUM_FALL_DELTA:
    reasons.append("confirmation fall increase exceeds 3 percentage points")
  return not reasons, reasons


def development_success_gate(
  *,
  target_success_delta: float,
  target_fall_delta: float,
  d0_success_delta: float,
) -> dict[str, Any]:
  values = (target_success_delta, target_fall_delta, d0_success_delta)
  if not all(math.isfinite(float(value)) for value in values):
    raise ValueError("v22 final metrics must be finite")
  conditions = {
    "target_success_delta_at_least_three_pp": (
      target_success_delta >= DEVELOPMENT_MINIMUM_SUCCESS_DELTA
    ),
    "target_fall_delta_at_most_one_pp": (
      target_fall_delta <= DEVELOPMENT_MAXIMUM_FALL_DELTA
    ),
    "d0_success_delta_at_least_minus_five_pp": (
      d0_success_delta >= D0_MINIMUM_SUCCESS_DELTA
    ),
  }
  return {
    "passed": all(conditions.values()),
    "conditions": conditions,
    "target_success_delta": target_success_delta,
    "target_fall_delta": target_fall_delta,
    "d0_success_delta": d0_success_delta,
    "strict_zero_fall_increase_passed": target_fall_delta <= 0.0,
  }


def select_best_so_far(
  rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
  """Select highest validation SR among D0-safe, validation-fall-safe rows."""
  if not rows or int(rows[0].get("round", -1)) != 0:
    raise ValueError("v22 best-so-far rows must start with pi0 at round zero")
  baseline_fall = float(rows[0]["fall_rate"])
  eligible: list[Mapping[str, Any]] = []
  for row in rows:
    success_rate = float(row["success_rate"])
    fall_rate = float(row["fall_rate"])
    if not all(math.isfinite(value) for value in (success_rate, fall_rate)):
      raise ValueError("v22 validation metrics must be finite")
    if bool(row.get("d0_safe")) and (
      fall_rate <= baseline_fall + VALIDATION_MAXIMUM_FALL_DELTA
    ):
      eligible.append(row)
  if not eligible:
    raise ValueError("v22 validation selection lost the always-eligible pi0")
  selected = max(
    eligible,
    key=lambda row: (
      float(row["success_rate"]),
      -float(row["fall_rate"]),
      -int(row["round"]),
    ),
  )
  return dict(selected)


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


def all_v22_random_seeds() -> set[int]:
  """Expand every explicit and derived seed used by the frozen v22 runner."""
  seeds: set[int] = set()
  for context_id in CONTEXTS:
    candidates = CONTEXT_CALIBRATION_CANDIDATE_SEEDS[context_id]
    seeds.update(candidates)
    for index in range(len(candidates)):
      evaluation_seed = calibration_evaluation_seed(context_id, index)
      seeds.update(evaluation_seed + repeat for repeat in range(4))
    adaptation = CONTEXT_ADAPTATION_SEEDS[context_id]
    seeds.add(adaptation)
    seeds.update(
      failure_discovery_seed(context_id, index)
      for index in range(FAILURE_DISCOVERY_MAX_ROLLOUTS)
    )
    seeds.update(
      dual_rollout_seed(context_id, round_index, batch_index)
      for round_index in range(1, ROUNDS + 1)
      for batch_index in range(DUAL_ROLLOUT_BATCHES)
    )
    seeds.update(
      candidate_screen_seed(context_id, round_index)
      for round_index in range(1, ROUNDS + 1)
    )
    seeds.update(
      candidate_confirmation_seed(context_id, round_index)
      for round_index in range(1, ROUNDS + 1)
    )
    seeds.add(candidate_d0_seed(context_id))
    seeds.update(CONTEXT_VALIDATION_SEEDS[context_id] + repeat for repeat in range(2))
    seeds.update(CONTEXT_FINAL_TARGET_SEEDS[context_id] + repeat for repeat in range(4))
    seeds.update(CONTEXT_FINAL_D0_SEEDS[context_id] + repeat for repeat in range(2))
    seeds.update(CONTEXT_REPORT_BOOTSTRAP_SEEDS[context_id].values())
  return seeds


def fresh_randomness_report(repo: Path) -> dict[str, Any]:
  historical: set[int] = set()
  scanned_files: list[dict[str, str]] = []
  for version in range(17, 22):
    root = repo / "results/online" / f"specialist_v{version}"
    if not root.is_dir():
      continue
    for path in sorted(root.rglob("*.json")):
      try:
        rendered = path.read_text()
        payload = json.loads(rendered)
      except (json.JSONDecodeError, OSError):
        continue
      scanned_files.append(
        {
          "file": str(path.relative_to(repo)),
          "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
      )
      historical.update(_iter_declared_seeds(payload))
  proposed = all_v22_random_seeds()
  collisions = sorted(proposed & historical)
  return {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "historical_versions_scanned": [17, 18, 19, 20, 21],
    "historical_json_file_count": len(scanned_files),
    "historical_json_files_sha256": canonical_sha256({"files": scanned_files}),
    "historical_unique_declared_seed_count": len(historical),
    "proposed_expanded_seed_count": len(proposed),
    "collisions": collisions,
    "passed": not collisions,
  }


def canonical_sha256(value: Mapping[str, Any]) -> str:
  return hashlib.sha256(
    json.dumps(
      value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
  ).hexdigest()
